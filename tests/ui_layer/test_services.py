"""
The services of a session are built in one place (ui.services, M5b step 5): a page builds none.

The AI engine is built on first use, by Settings or a feature that needs it: Statistics & correlations and Enrich
built one on every visit, loading the AI packages for nothing. An engine of a class reloaded after a code change is
built again with the key typed for it, and stopped first; an engine a test puts in the session is kept.
"""

import importlib.util
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


def _sessions_services() -> None:
    import streamlit as st

    from ui.state import get_state

    st.session_state["username"] = st.session_state["as_user"]
    state = get_state()
    datasets, trace = state.db_manager, state.transformation_manager
    st.session_state["folders"] = [datasets.dataset_dir, datasets.stats_dir, trace.trace_dir]
    if st.session_state.get("with_ai"):
        st.session_state["folders"].append(state.rag_manager.documents_dir)


def test_a_signed_in_sessions_services_keep_to_the_users_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The services take the user's workspace from the session (AppState, ai_engine), and never ask for it themselves;
    a name that keeps nothing as a folder name has a workspace of its own, not the shared folders."""
    from core.workspace import UserScope

    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    with_ai = importlib.util.find_spec("langchain_core") is not None
    workspaces = set()
    for user in ("alice", "Мария", "Мариам"):  # the last two keep nothing as a folder name
        at = AppTest.from_function(_sessions_services, default_timeout=60)
        at.session_state["as_user"] = user
        at.session_state["with_ai"] = with_ai
        at.run()
        assert not at.exception, [str(e.value)[:300] for e in at.exception]
        folders = at.session_state["folders"]
        assert len(folders) == (4 if with_ai else 3)
        workspace = UserScope.for_user(user).root.resolve()
        for folder in folders:
            assert Path(folder).resolve().is_relative_to(workspace), (user, folder)
        workspaces.add(workspace)
    assert len(workspaces) == 3  # one each
    assert {w.parent for w in workspaces} == {(tmp_path / "users").resolve()}


def _reloaded() -> None:
    import streamlit as st

    from manage import rag_manager
    from ui.services import ai_engine

    def stop(self, reason: str) -> None:
        st.session_state["stopped"] = reason

    st.session_state["username"] = "alice"
    # the class as it was before a code change: another object, with the same module and name
    before = type("RAGManager", (), {"__module__": rag_manager.__name__, "stop": stop})
    old = before()
    old.api_key = "TYPED-KEY"
    st.session_state["rag_manager"] = old
    engine = ai_engine()
    built = (type(engine) is rag_manager.RAGManager, engine.api_key, engine.scope.username)
    st.session_state["rebuilt"] = built


def test_an_engine_of_a_reloaded_class_is_stopped_and_built_again_with_its_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("langchain_core", reason="the AI engine is in the optional 'ai' extra")
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    at = AppTest.from_function(_reloaded, default_timeout=60).run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert at.session_state["rebuilt"] == (True, "TYPED-KEY", "alice")
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
