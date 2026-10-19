"""
The Users page manages the accounts of the configured identity provider (M5): it used to manage
the local DuckDB accounts whatever CMP_IAM_PROVIDER said, so under Better Auth its changes reached
nobody who signs in.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def _script() -> None:
    import streamlit as st

    from app_pages import users_management

    st.session_state["user_role"] = "admin"
    users_management.app()


@pytest.fixture
def iam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The provider the app signs users in with: local accounts in a temporary database."""
    import manage.iam.manager as manager
    import utils.data_paths as dp
    from manage.iam.schemas import IAMConfig, IAMProviderType, UserRole

    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(manager, "_GLOBAL_IAM_MANAGER", None)
    signs_in = manager.get_iam_manager(
        IAMConfig(provider_type=IAMProviderType.LOCAL, duckdb_path=str(tmp_path / "signin.duckdb"))
    )
    signs_in.create_user("chief", "a-long-password-6", role=UserRole.ADMIN, auto_activate=True)
    signs_in.create_user("dr_lee", "a-long-password-7")
    yield signs_in
    monkeypatch.setattr(manager, "_GLOBAL_IAM_MANAGER", None)


def test_the_page_lists_and_changes_the_accounts_that_sign_in(iam) -> None:
    at = AppTest.from_function(_script, default_timeout=120)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    shown = " ".join(m.value for m in at.markdown)
    assert "dr_lee" in shown and "chief" in shown
    assert any("Accounts of the identity provider: local accounts" in c.value for c in at.caption)
    next(b for b in at.button if b.label == "Activate").click().run()
    assert not at.exception
    # in the database that signs in
    assert next(u for u in iam.list_users() if u.username == "dr_lee").is_active


def test_without_sign_in_there_is_nothing_to_manage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import manage.iam.manager as manager
    from manage.iam.schemas import IAMConfig, IAMProviderType

    monkeypatch.setattr(manager, "_GLOBAL_IAM_MANAGER", None)
    manager.get_iam_manager(IAMConfig(provider_type=IAMProviderType.DISABLED))
    at = AppTest.from_function(_script, default_timeout=120)
    at.run()
    assert not at.exception
    assert any("no accounts to manage" in i.value for i in at.info)
    monkeypatch.setattr(manager, "_GLOBAL_IAM_MANAGER", None)
