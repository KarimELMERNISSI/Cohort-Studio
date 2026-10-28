"""
Workspace folders at sign-in and sign-up (core.workspace.safe_dirname, AppTest).

Earlier versions gave a name the folder of its characters with each run of others replaced by "_" and its ends
trimmed, so two names could share one. A name that is no folder name now has a code of its own: at its first sign-in,
the folder an earlier version gave it is moved there when it is clearly its own. A new account never takes a folder
that exists: an earlier version's folder of another account may have its name until that account signs in.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PASSWORD = "a-long-password-6"


def _script() -> None:
    import streamlit as st

    from manage.iam.manager import IAMManager
    from manage.iam.schemas import IAMConfig, IAMProviderType
    from ui import auth

    iam = IAMManager(
        config=IAMConfig(provider_type=IAMProviderType.LOCAL, duckdb_path=st.session_state["db"])
    )
    auth._login_card(iam)


@pytest.fixture
def card(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    import utils.data_paths as dp
    from manage.iam.manager import IAMManager
    from manage.iam.schemas import IAMConfig, IAMProviderType, UserRole

    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    db = str(tmp_path / "users.duckdb")
    iam = IAMManager(config=IAMConfig(provider_type=IAMProviderType.LOCAL, duckdb_path=db))
    assert iam.create_user("chief", PASSWORD, role=UserRole.ADMIN, auto_activate=True)[0]
    # a name earlier versions accepted, whose folder lost its final dot
    assert iam.create_user("drlee.", PASSWORD, auto_activate=True)[0]
    at = AppTest.from_function(_script, default_timeout=120)
    at.session_state["db"] = db
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    return at


def _fill(at: AppTest, fields: dict[str, str], button: str) -> None:
    inputs = {t.label: t for t in at.text_input}
    for label, value in fields.items():
        inputs[label].input(value)
    next(b for b in at.button if b.label == button).click().run()


def test_the_first_sign_in_moves_the_folder_an_earlier_version_gave_the_account(
    card: AppTest, tmp_path: Path
) -> None:
    earlier = tmp_path / "users" / "drlee" / "datasets" / "cohort_v1.parquet"
    earlier.parent.mkdir(parents=True)
    earlier.write_bytes(b"rows")
    _fill(card, {"Username": "drlee.", "Password": PASSWORD}, "Sign in")
    assert not card.exception, [str(e.value)[:300] for e in card.exception]
    assert card.session_state["username"] == "drlee."
    moved = tmp_path / "users" / "_drlee_x2e_" / "datasets" / "cohort_v1.parquet"
    assert moved.read_bytes() == b"rows" and not (tmp_path / "users" / "drlee").exists()


def test_a_new_account_never_takes_a_folder_that_exists(card: AppTest, tmp_path: Path) -> None:
    (tmp_path / "users" / "newbie" / "datasets").mkdir(parents=True)
    fields = {"New username": "newbie", "New password": PASSWORD, "Confirm password": PASSWORD}
    _fill(card, fields, "Create account")
    assert any("already exists" in e.value for e in card.error), [e.value for e in card.error]
    _fill(card, {**fields, "New username": "newcomer"}, "Create account")
    assert not any("already exists" in e.value for e in card.error)


def test_another_identity_provider_moves_nothing_and_names_the_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Better Auth may list its accounts in pages, or under their e-mail: its accounts' earlier folders are left for
    an administrator (dev_scripts/move_workspaces.py), and the log names them."""
    from types import SimpleNamespace

    import utils.data_paths as dp
    from manage.iam.schemas import IAMProviderType
    from ui import auth

    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    earlier = tmp_path / "users" / "j.smith_chl.lu" / "datasets"
    earlier.mkdir(parents=True)

    def never_listed() -> list:
        raise AssertionError("not asked: nothing is moved")

    provider = SimpleNamespace(provider_type=IAMProviderType.BETTER_AUTH, list_users=never_listed)
    with caplog.at_level("WARNING", logger="ui.auth"):
        auth._move_earlier_workspace(provider, "j.smith@chl.lu")
    assert earlier.exists() and not (tmp_path / "users" / "_j.smith_at_chl.lu").exists()
    assert any(
        "j.smith_chl.lu" in r.getMessage() and "administrator" in r.getMessage()
        for r in caplog.records
    )


def test_a_failing_list_of_accounts_never_stops_a_sign_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import utils.data_paths as dp
    from manage.iam.schemas import IAMProviderType
    from ui import auth

    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    (tmp_path / "users" / "drlee" / "datasets").mkdir(parents=True)

    def failing() -> list:
        raise ConnectionError("the provider did not answer")

    provider = SimpleNamespace(provider_type=IAMProviderType.LOCAL, list_users=failing)
    auth._move_earlier_workspace(provider, "drlee.")  # logs, raises nothing
    assert (tmp_path / "users" / "drlee").exists()
