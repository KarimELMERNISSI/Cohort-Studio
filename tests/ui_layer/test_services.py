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
    assert "db_manager" in at.session_state  # the dataset store is the session's (ui.services)


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


def test_an_engine_put_in_the_session_is_kept_and_dropping_one_stops_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))  # the session's stores are built too
    at = AppTest.from_function(_injected_then_dropped, default_timeout=60).run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert at.session_state["kept"] and at.session_state["dropped"]
    assert at.session_state["stopped"] == "the user signed out"


def _a_session() -> None:
    """A run that first puts in the session what a test plans (another user's store or engine), then asks for the
    session's services as the signed-in user of the plan."""
    from pathlib import Path

    import streamlit as st

    from core.workspace import UserScope
    from manage.transformation_manager import TransformationManager
    from ui.services import services

    plan = st.session_state.get("plan")
    if plan is None:  # the session was ended: nothing of it is left
        return

    class Engine:
        def __init__(self, owner: str) -> None:
            self.scope = UserScope.for_user(owner)

        def stop(self, reason: str) -> None:
            Path(plan["log"]).write_text(reason, encoding="utf-8")

    if plan.get("engine_of") and "rag_manager" not in st.session_state:
        st.session_state["rag_manager"] = Engine(plan["engine_of"])
    if plan.get("trace_of") and "transformation_manager" not in st.session_state:
        planted = TransformationManager(scope=UserScope.for_user(plan["trace_of"]))
        st.session_state["transformation_manager"] = planted
        st.session_state["planted"] = id(planted)
    st.session_state["data"] = f"the dataset of {plan['user']}"
    st.session_state["username"] = plan["user"]
    built = services()
    st.session_state["seen"] = {
        "user": built.scope.username if built.scope is not None else None,
        "trace": id(built.trace),
        "folder": built.trace.trace_dir,
        "keys": st.session_state["db_manager"] is built.datasets
        and st.session_state["transformation_manager"] is built.trace,
    }


SESSION_KEYS = (
    "_services",
    "db_manager",
    "transformation_manager",
    "rag_manager",
    "username",
    "data",
)


def _session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path / "data"))
    return AppTest.from_function(_a_session, default_timeout=60)


def test_another_user_signed_in_to_the_session_ends_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The user changes only by signing in again: a session that finds another user ends, as a sign-out ends it, and
    nothing of the last user stays (their dataset, stores, engine)."""
    log = tmp_path / "stopped.txt"
    at = _session(tmp_path, monkeypatch)
    at.session_state["plan"] = {"user": "alice", "engine_of": "alice", "log": str(log)}
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    alice = at.session_state["seen"]
    assert alice["user"] == "alice" and alice["keys"]
    at.run()  # the same user: the same services
    assert at.session_state["seen"]["trace"] == alice["trace"]
    at.session_state["plan"] = {"user": "bob", "log": str(log)}
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert log.read_text(encoding="utf-8") == "another user was signed in to this session"
    for key in (*SESSION_KEYS, "plan"):
        assert key not in at.session_state, key


@pytest.mark.parametrize("planted", ["trace_of", "engine_of"])
def test_another_users_store_or_engine_in_the_session_ends_it(
    planted: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "stopped.txt"
    at = _session(tmp_path, monkeypatch)
    at.session_state["plan"] = {"user": "alice", planted: "bob", "log": str(log)}
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    for key in (*SESSION_KEYS, "plan"):
        assert key not in at.session_state, key
    if planted == "engine_of":
        assert log.read_text(encoding="utf-8") == "another user was signed in to this session"


def test_a_session_older_than_its_services_keeps_its_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session that outlives a code change keeps the trace it holds, when it is the same user's."""
    at = _session(tmp_path, monkeypatch)
    at.session_state["plan"] = {"user": "alice", "trace_of": "alice"}
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    seen = at.session_state["seen"]
    assert seen["user"] == "alice" and seen["keys"] and seen["trace"] == at.session_state["planted"]


def test_services_built_before_the_sign_in_are_built_again_for_the_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from core.workspace import UserScope

    at = _session(tmp_path, monkeypatch)
    at.session_state["plan"] = {"user": None}
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    before = at.session_state["seen"]
    assert before["user"] is None
    at.session_state["plan"] = {"user": "alice"}
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    alice = at.session_state["seen"]
    assert alice["user"] == "alice" and alice["keys"] and alice["trace"] != before["trace"]
    assert Path(alice["folder"]).resolve() == UserScope.for_user("alice").traces_dir().resolve()


def _signing_out() -> None:
    from pathlib import Path

    import streamlit as st

    from ui.auth import logout
    from ui.services import services

    log = st.session_state.get("log")
    if log is None:  # after the sign-out: nothing of the session is left
        return
    st.session_state["username"] = "alice"
    services()

    class Engine:
        def stop(self, reason: str) -> None:
            Path(log).write_text(reason, encoding="utf-8")

    st.session_state["rag_manager"] = Engine()
    logout()


def test_signing_out_stops_the_engine_and_drops_the_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path / "data"))
    log = tmp_path / "stopped.txt"
    at = AppTest.from_function(_signing_out, default_timeout=60)
    at.session_state["log"] = str(log)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert log.read_text(encoding="utf-8") == "the user signed out"
    for key in ("_services", "db_manager", "transformation_manager", "rag_manager", "username"):
        assert key not in at.session_state, key


def _one_user() -> None:
    import streamlit as st

    from ui.services import services, shared

    st.session_state["username"] = st.session_state["as_user"]
    built = services()
    st.session_state["seen"] = (id(shared()), id(built.shared), built.datasets.dataset_dir)


def test_two_users_share_only_the_services_of_the_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    seen = {}
    for user in ("alice", "bob"):
        at = AppTest.from_function(_one_user, default_timeout=60)
        at.session_state["as_user"] = user
        at.run()
        assert not at.exception, [str(e.value)[:300] for e in at.exception]
        seen[user] = at.session_state["seen"]
    assert seen["alice"][0] == seen["bob"][0] == seen["alice"][1] == seen["bob"][1]
    assert seen["alice"][2] != seen["bob"][2]
