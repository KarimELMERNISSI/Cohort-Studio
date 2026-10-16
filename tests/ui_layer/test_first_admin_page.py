"""The first-run page: the login screen creates the first administrator with the setup token (M4, AppTest)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def _script() -> None:
    import streamlit as st

    from manage.iam.manager import IAMManager
    from manage.iam.schemas import IAMConfig, IAMProviderType
    from ui import auth

    ss = st.session_state
    iam = IAMManager(config=IAMConfig(provider_type=IAMProviderType.LOCAL, duckdb_path=ss["db"]))
    auth._login_card(iam)
    ss["pending"] = iam.setup_pending()


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import utils.data_paths as dp

    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("CMP_ADMIN_PASSWORD", raising=False)
    return tmp_path


def _form(at: AppTest, token: str, username: str, password: str) -> None:
    fields = {t.label: t for t in at.text_input}
    fields["Setup token"].input(token)
    fields["Administrator username"].input(username)
    fields["Password"].input(password)
    fields["Confirm password"].input(password)
    next(b for b in at.button if b.label == "Create the administrator").click().run()


def test_the_login_screen_creates_the_first_administrator(root: Path) -> None:
    from manage.iam.manager import IAMManager
    from manage.iam.schemas import IAMConfig, IAMProviderType

    db = str(root / "users.duckdb")
    iam = IAMManager(config=IAMConfig(provider_type=IAMProviderType.LOCAL, duckdb_path=db))
    from manage.iam import setup_token

    token = setup_token.issue(root)
    assert iam.setup_pending()

    at = AppTest.from_function(_script, default_timeout=120)
    at.session_state["db"] = db
    at.run()
    assert not at.exception
    assert any("First run: create the administrator" in m.value for m in at.markdown)
    _form(at, "not-the-token", "chief", "a-long-password-6")
    assert any("setup token is not valid" in e.value for e in at.error)
    _form(at, token, "chief", "a-long-password-6")
    assert not at.exception
    assert at.session_state["pending"] is False
    assert any("Administrator 'chief' created" in s.value for s in at.success)
    assert any(t.label == "Username" for t in at.text_input)  # the usual login form
