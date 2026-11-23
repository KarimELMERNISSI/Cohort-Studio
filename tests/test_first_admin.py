"""The first administrator of a fresh installation: a one-time setup token written to the server log (M4)."""

from __future__ import annotations

import codecs
import logging
import re
from pathlib import Path

import pytest

import utils.data_paths as dp
from manage.iam import setup_token
from manage.iam.manager import IAMConfigurationError, IAMManager
from manage.iam.schemas import IAMConfig, IAMProviderType, UserCredentials, UserRole


@pytest.fixture
def iam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> IAMManager:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("CMP_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("CMP_ADMIN_PASSWORD_FILE", raising=False)
    return IAMManager(
        config=IAMConfig(provider_type=IAMProviderType.LOCAL, duckdb_path=str(tmp_path / "users.duckdb"))
    )


def _token_from(caplog: pytest.LogCaptureFixture) -> str:
    match = re.search(r"one-time setup token: (\S+)", caplog.text)
    assert match, caplog.text
    return match.group(1)


def test_a_fresh_installation_creates_its_administrator_with_the_logged_token(
    iam: IAMManager, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    with caplog.at_level(logging.WARNING):
        ok, message = iam.ensure_bootstrap_admin()
    assert not ok and message == "Setup token issued"
    token = _token_from(caplog)
    assert iam.setup_pending()
    stored = (tmp_path / setup_token.TOKEN_FILE).read_text(encoding="utf-8")
    assert token not in stored  # only its hash is kept

    assert iam.create_first_admin("wrong-token", "chief", "a-long-password-1") == (
        False,
        "The setup token is not valid. Copy it from the server log (the most recent one).",
    )
    ok, _ = iam.create_first_admin(token, "chief", "a-long-password-1")
    assert ok and not iam.setup_pending()
    user = iam.get_user("chief")
    assert user is not None and str(user.role) == "admin" and user.is_active
    assert iam.create_first_admin(token, "second", "a-long-password-2")[0] is False  # used once

    with caplog.at_level(logging.WARNING):
        assert iam.ensure_bootstrap_admin() == (False, "An active administrator already exists")


def test_a_configured_password_needs_no_token(iam: IAMManager, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CMP_ADMIN_USERNAME", "boss")
    monkeypatch.setenv("CMP_ADMIN_PASSWORD", "a-long-password-3")
    ok, _ = iam.ensure_bootstrap_admin()
    assert ok and not iam.setup_pending()
    assert iam.create_first_admin("anything", "x", "a-long-password-4")[0] is False


SHORT = "short-pw-1"  # 10 characters: the minimum is 12


def _stopped(iam: IAMManager, caplog: pytest.LogCaptureFixture) -> str:
    """The words of a start stopped by the first administrator's password: no administrator, and no setup token."""
    with caplog.at_level(logging.WARNING), pytest.raises(IAMConfigurationError) as stopped:
        iam.ensure_bootstrap_admin()
    assert not iam.setup_pending() and "one-time setup token:" not in caplog.text
    assert not any(str(user.role) == "admin" for user in iam.list_users())
    return str(stopped.value)  # said by whoever stops the start: the page and the server's log, the terminal


def _signs_in(iam: IAMManager, username: str, password: str) -> bool:
    return iam.authenticate(UserCredentials(username=username, password=password)).success


@pytest.mark.parametrize("given", ["variable", "file"])
def test_a_configured_password_shorter_than_the_minimum_stops_the_start(
    given: str,
    iam: IAMManager,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """CMP_ADMIN_PASSWORD (or its file) is held to CMP_MIN_PASSWORD_LENGTH as every other password is. No setup
    token takes its place: whoever sets a password keeps every credential out of the log. The words name the
    setting and the minimum, never the password."""
    secret = tmp_path / "admin_password.txt"
    monkeypatch.setenv("CMP_ADMIN_USERNAME", "boss")
    if given == "variable":
        setting = "CMP_ADMIN_PASSWORD"
        monkeypatch.setenv(setting, SHORT)
    else:
        setting = "CMP_ADMIN_PASSWORD_FILE"
        secret.write_text(SHORT + "\n", encoding="utf-8")
        monkeypatch.setenv(setting, str(secret))
    said = _stopped(iam, caplog)
    assert said == (
        f"the first administrator cannot be created from {setting}. Password must be at least 12 characters. Give "
        f"a password that can be used, or remove {setting} to create the first administrator with a one-time setup "
        "token; then restart the application."
    )
    assert SHORT not in caplog.text and str(tmp_path) not in caplog.text
    assert _stopped(iam, caplog) == said  # every page load until it is corrected
    # corrected: the administrator is created
    if given == "variable":
        monkeypatch.setenv(setting, "a-long-password-9")
    else:
        secret.write_text("a-long-password-9\n", encoding="utf-8")
    assert iam.ensure_bootstrap_admin()[0] is True and _signs_in(iam, "boss", "a-long-password-9")


@pytest.mark.parametrize(
    ("content", "why"),
    [
        (None, "Its file could not be read (FileNotFoundError"),
        (b"  \r\n", "Its file is empty."),
        (
            ("P" + chr(0xE4) + "sswort-erfunden-1").encode("cp1252"),
            "Its file is not UTF-8 text (UnicodeDecodeError).",
        ),
        (
            codecs.BOM_UTF16_LE + "a-long-password-9".encode("utf-16-le"),
            "Its file is not UTF-8 text (UnicodeDecodeError).",
        ),
        ("a-long-password-9".encode("utf-16-le"), "Password holds a character that cannot be stored."),
    ],
    ids=["missing", "blank", "windows-1252", "utf-16", "utf-16 without its mark"],
)
def test_a_password_file_that_cannot_be_used_stops_the_start(
    content: bytes | None,
    why: str,
    iam: IAMManager,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """A file that is missing, blank or in another encoding went unsaid (a setup token was issued), or raised an
    error whose own text quotes a byte of the password and its place, into the log and onto the page. The words
    give the error's kind, and neither the file's path nor anything of what it holds."""
    secret = tmp_path / "admin_password.txt"
    if content is not None:
        secret.write_bytes(content)
    monkeypatch.setenv("CMP_ADMIN_PASSWORD_FILE", str(secret))
    said = _stopped(iam, caplog)
    assert said.startswith("the first administrator cannot be created from CMP_ADMIN_PASSWORD_FILE. " + why)
    for held in ("0xe4", "0xff", "position", "erfunden", str(tmp_path), secret.name):
        assert held not in said and held not in caplog.text, held


def test_a_byte_order_mark_is_no_character_of_the_password(
    iam: IAMManager, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Notepad's "UTF-8 with BOM": the mark was counted in the length and stored with the password, so that 11
    typed characters passed for 12 and the administrator could not sign in with what was typed."""
    secret = tmp_path / "admin_password.txt"
    monkeypatch.setenv("CMP_ADMIN_USERNAME", "boss")
    monkeypatch.setenv("CMP_ADMIN_PASSWORD_FILE", str(secret))
    secret.write_bytes(codecs.BOM_UTF8 + b"eleven-char\r\n")
    assert "Password must be at least 12 characters." in _stopped(iam, caplog)
    secret.write_bytes(codecs.BOM_UTF8 + b"twelve-chars\r\n")
    assert iam.ensure_bootstrap_admin()[0] is True and not iam.setup_pending()
    assert _signs_in(iam, "boss", "twelve-chars")


def test_the_variable_comes_before_the_file_and_the_words_say_so(
    iam: IAMManager, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    secret = tmp_path / "admin_password.txt"
    secret.write_text("a-long-password-9\n", encoding="utf-8")
    monkeypatch.setenv("CMP_ADMIN_PASSWORD_FILE", str(secret))
    monkeypatch.setenv("CMP_ADMIN_PASSWORD", SHORT)
    assert _stopped(iam, caplog).startswith(
        "the first administrator cannot be created from CMP_ADMIN_PASSWORD (CMP_ADMIN_PASSWORD_FILE is set too and "
        "is not read: the variable comes first). Password must be at least 12 characters."
    )


def test_the_minimum_is_the_installations_own(
    iam: IAMManager, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("CMP_ADMIN_USERNAME", "boss")
    monkeypatch.setenv("CMP_ADMIN_PASSWORD", "twelve-chars")  # 12 characters
    monkeypatch.setattr("manage.iam.authz.MIN_PASSWORD_LENGTH", 15)
    assert "Password must be at least 15 characters." in _stopped(iam, caplog)
    monkeypatch.setattr("manage.iam.authz.MIN_PASSWORD_LENGTH", 12)
    assert iam.ensure_bootstrap_admin()[0] is True and not iam.setup_pending()


def test_a_password_that_would_create_nothing_does_not_stop_the_start(
    iam: IAMManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The account of CMP_ADMIN_USERNAME exists and is not an active administrator: its configured password creates
    nothing, whatever its length, and an active administrator's is never read."""
    assert iam.create_user("boss", "a-long-password-9")[0]  # pending
    monkeypatch.setenv("CMP_ADMIN_USERNAME", "boss")
    monkeypatch.setenv("CMP_ADMIN_PASSWORD", SHORT)
    assert iam.ensure_bootstrap_admin() == (False, "User 'boss' exists but is not an active administrator")
    assert iam.create_user("chief", "a-long-password-1", role=UserRole.ADMIN, auto_activate=True)[0]
    assert iam.ensure_bootstrap_admin() == (False, "An active administrator already exists")


def test_the_accounts_refuse_a_password_the_rule_refuses_whoever_calls(
    iam: IAMManager, caplog: pytest.LogCaptureFixture
) -> None:
    """The rule is held by IAMManager itself, not by each of its callers: no account is created with, or given, a
    password of one letter, of spaces, or longer than the hash takes."""
    assert iam.create_user("dr_lee", "a-long-password-9", auto_activate=True)[0]
    short = "Password must be at least 12 characters."
    too_long = "Password is too long: at most 72 bytes (a letter with an accent counts for two or more)."
    for refused, why in (
        ("", short),
        ("x", short),
        (" " * 12, "Password cannot be made of spaces only."),
        ("x" * 73, too_long),
        (chr(0xE9) * 37, too_long),
        ("twelve-chars" + chr(0), "Password holds a character that cannot be stored."),
    ):
        assert iam.update_user_password("dr_lee", refused) == (False, why)
        assert iam.create_user("another", refused) == (False, why)
    assert iam.get_user("another") is None and _signs_in(iam, "dr_lee", "a-long-password-9")
    assert iam.update_user_password("dr_lee", "x" * 72)[0] is True and _signs_in(iam, "dr_lee", "x" * 72)
    # the first administrator's page: a refused password keeps the token for the next try
    with caplog.at_level(logging.WARNING):
        iam.ensure_bootstrap_admin()
    token = _token_from(caplog)
    assert iam.create_first_admin(token, "chief", "x") == (False, short)
    assert iam.create_first_admin(token, "chief", "a-long-password-1")[0] is True


def test_the_logged_token_is_kept_until_a_new_one_is_asked_for(
    iam: IAMManager, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        iam.ensure_bootstrap_admin()  # server start-up
    first = _token_from(caplog)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert iam.ensure_bootstrap_admin() == (False, "Setup token pending")  # a page load: no new token
    assert "one-time setup token:" not in caplog.text
    second = iam.issue_setup_token()  # python manage_users.py setup-token
    assert second is not None and second != first
    assert iam.create_first_admin(first, "chief", "a-long-password-5")[0] is False
    assert iam.create_first_admin(second, "chief", "a-long-password-5")[0] is True
    assert iam.issue_setup_token() is None  # an administrator exists


def test_the_command_line_bootstrap_and_setup_token(
    iam: IAMManager, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import manage_users

    monkeypatch.setattr(manage_users, "get_iam_manager", lambda: iam)
    monkeypatch.setattr("manage.iam.get_iam_manager", lambda: iam)
    manage_users.main(["bootstrap"])
    assert "Administrator bootstrap: Setup token issued" in capsys.readouterr().out
    manage_users.main(["setup-token"])
    token = capsys.readouterr().out.split("One-time setup token: ", 1)[1].split()[0]
    assert iam.create_first_admin(token, "chief", "a-long-password-7")[0] is True
    with pytest.raises(SystemExit):
        manage_users.main(["setup-token"])  # not needed any more


def test_create_admin_reuses_an_existing_folder_only_when_told(
    iam: IAMManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The folder of a name may hold another account's files (the earlier folder of an account that has not signed
    in since): the operator reuses it only with --reuse-folder (a users database replaced, its folders kept)."""
    import manage_users

    monkeypatch.setattr(manage_users, "get_iam_manager", lambda: iam)
    monkeypatch.setenv("CMP_ADMIN_USERNAME", "chief")  # the account CMP_ADMIN_PASSWORD is the password of
    monkeypatch.setenv("CMP_ADMIN_PASSWORD", "a-long-password-8")
    (tmp_path / "users" / "chief" / "datasets").mkdir(parents=True)
    with pytest.raises(SystemExit) as refused:
        manage_users.main(["create-admin", "chief"])
    assert "--reuse-folder" in str(refused.value.code)  # the message is the exit status
    assert "chief" not in iam.get_all_users()
    with pytest.raises(SystemExit) as created:
        manage_users.main(["create-admin", "chief", "--reuse-folder"])
    assert created.value.code == 0 and "chief" in iam.get_all_users()


def test_the_command_line_asks_for_a_password_that_is_not_the_first_administrators(
    iam: IAMManager, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """CMP_ADMIN_PASSWORD is the password of the account of CMP_ADMIN_USERNAME. Read for every command, it gave the
    account of a reset-password, and any other administrator created, the first administrator's password without
    a question."""
    import manage_users

    monkeypatch.setattr(manage_users, "get_iam_manager", lambda: iam)
    monkeypatch.setenv("CMP_ADMIN_USERNAME", "chief")
    monkeypatch.setenv("CMP_ADMIN_PASSWORD", "a-long-password-0")
    assert iam.create_user("dr_lee", "a-long-password-9", auto_activate=True)[0]
    asked: list[str] = []

    def typed(prompt: str) -> str:
        asked.append(prompt)
        return "a-long-password-2"

    monkeypatch.setattr(manage_users, "_at_a_terminal", lambda: True)
    monkeypatch.setattr(manage_users.getpass, "getpass", typed)
    with pytest.raises(SystemExit) as done:
        manage_users.main(["reset-password", "dr_lee"])
    assert done.value.code == 0 and asked == ["New password: ", "Confirm password: "]
    assert _signs_in(iam, "dr_lee", "a-long-password-2") and not _signs_in(iam, "dr_lee", "a-long-password-0")
    with pytest.raises(SystemExit) as done:
        manage_users.main(["create-admin", "second_admin"])  # not the account of CMP_ADMIN_USERNAME
    assert done.value.code == 0 and len(asked) == 4
    assert _signs_in(iam, "second_admin", "a-long-password-2")
    assert not _signs_in(iam, "second_admin", "a-long-password-0")
    with pytest.raises(SystemExit) as done:
        manage_users.main(["create-admin", "chief"])  # the one account the setting is read for
    assert done.value.code == 0 and len(asked) == 4 and _signs_in(iam, "chief", "a-long-password-0")
    # where nobody can type: the command says so at once and changes nothing
    monkeypatch.setattr(manage_users, "_at_a_terminal", lambda: False)
    with pytest.raises(SystemExit) as refused:
        manage_users.main(["reset-password", "dr_lee"])
    assert "typed on a terminal, and there is none here" in str(refused.value.code)
    assert len(asked) == 4 and _signs_in(iam, "dr_lee", "a-long-password-2")
    capsys.readouterr()
