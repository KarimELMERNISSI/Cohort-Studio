"""The first administrator of a fresh installation: a one-time setup token written to the server log (M4)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

import utils.data_paths as dp
from manage.iam import setup_token
from manage.iam.manager import IAMManager
from manage.iam.schemas import IAMConfig, IAMProviderType


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
    monkeypatch.setenv("CMP_ADMIN_PASSWORD", "a-long-password-8")
    (tmp_path / "users" / "chief" / "datasets").mkdir(parents=True)
    with pytest.raises(SystemExit) as refused:
        manage_users.main(["create-admin", "chief"])
    assert "--reuse-folder" in str(refused.value.code)  # the message is the exit status
    assert "chief" not in iam.get_all_users()
    with pytest.raises(SystemExit) as created:
        manage_users.main(["create-admin", "chief", "--reuse-folder"])
    assert created.value.code == 0 and "chief" in iam.get_all_users()
