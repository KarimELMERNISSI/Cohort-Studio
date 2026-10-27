"""
The services of a session are built in one place (ui.services, M5b step 5): a page builds none.

The AI engine is built on first use, by Settings or a feature that needs it: Statistics & correlations and Enrich
built one on every visit, loading the AI packages for nothing. An engine of a class reloaded after a code change is
built again with the key typed for it, and stopped first; an engine a test puts in the session is kept.
"""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from ui import nav

ROOT = Path(__file__).resolve().parents[2]


def test_opening_a_page_builds_no_ai_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    monkeypatch.setitem(os.environ, "CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(str(ROOT / "main.py"), default_timeout=240)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "services_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    for page in ("explore", "enrich", "validate", "visualize"):
        at.switch_page(nav.page_path(page)).run()
        assert not at.exception, [str(e.value)[:300] for e in at.exception]
        assert "rag_manager" not in at.session_state, page
    assert "db_manager" in at.session_state  # the dataset store is the session's (AppState)


def _reloaded() -> None:
    import streamlit as st

    from manage import rag_manager
    from ui.services import ai_engine

    def stop(self, reason: str) -> None:
        st.session_state["stopped"] = reason

    # the class as it was before a code change: another object, with the same module and name
    before = type("RAGManager", (), {"__module__": rag_manager.__name__, "stop": stop})
    old = before()
    old.api_key = "TYPED-KEY"
    st.session_state["rag_manager"] = old
    engine = ai_engine()
    st.session_state["rebuilt"] = (type(engine) is rag_manager.RAGManager, engine.api_key)


def test_an_engine_of_a_reloaded_class_is_stopped_and_built_again_with_its_key() -> None:
    pytest.importorskip("langchain_core", reason="the AI engine is in the optional 'ai' extra")
    at = AppTest.from_function(_reloaded, default_timeout=60).run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert at.session_state["rebuilt"] == (True, "TYPED-KEY")
    assert "code change" in at.session_state["stopped"]


def _injected_then_dropped() -> None:
    import streamlit as st

    from ui.services import ai_engine, drop_ai_engine

    class Fake:
        def stop(self, reason: str) -> None:
            st.session_state["stopped"] = reason

    fake = Fake()
    st.session_state["rag_manager"] = fake
    st.session_state["kept"] = ai_engine() is fake
    drop_ai_engine("the user signed out")
    st.session_state["dropped"] = "rag_manager" not in st.session_state


def test_an_engine_put_in_the_session_is_kept_and_dropping_one_stops_it() -> None:
    at = AppTest.from_function(_injected_then_dropped, default_timeout=60).run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert at.session_state["kept"] and at.session_state["dropped"]
    assert at.session_state["stopped"] == "the user signed out"
