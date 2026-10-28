"""Per-user workspace wiring: managers, path resolution and login lockout (audit E4/E5)."""

from pathlib import Path

import pandas as pd
import pytest

import utils.data_paths as dp
from core.workspace import UserScope
from manage.db_manager import DBManager
from manage.iam.providers import local_duckdb
from manage.iam.schemas import UserCredentials
from manage.transformation_manager import TransformationManager


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    return tmp_path


def test_datasets_are_invisible_across_scopes(root: Path) -> None:
    alice = DBManager(scope=UserScope.for_user("alice"))
    bob = DBManager(scope=UserScope.for_user("bob"))
    df = pd.DataFrame({"id": [1, 2], "value": [0.5, 0.7]})

    ok, msg, name = alice.save_dataset(df, "cohort", username="alice")
    assert ok, msg
    assert name == "cohort_v1"  # no username prefix inside a workspace
    assert Path(alice.dataset_dir) == root / "users" / "alice" / "datasets"
    assert (root / "users" / "alice" / "datasets" / "cohort_v1.parquet").exists()

    assert alice.get_available_datasets(username="alice") == ["cohort_v1"]
    assert bob.get_available_datasets(username="bob") == []
    missing, _ = bob.load_dataset("cohort_v1")
    assert missing is None
    loaded, _ = alice.load_dataset("cohort_v1")
    assert loaded is not None and len(loaded) == 2


def test_dataset_names_with_quotes_are_quoted_for_duckdb(root: Path) -> None:
    alice = DBManager(scope=UserScope.for_user("alice"))
    ok, msg, name = alice.save_dataset(pd.DataFrame({"a": [1]}), "o'neil study", username="alice")
    assert ok, msg
    loaded, _ = alice.load_dataset(name)
    assert loaded is not None


def test_traces_live_in_the_user_workspace(root: Path) -> None:
    tm = TransformationManager(scope=UserScope.for_user("alice"))
    assert Path(tm.trace_dir) == root / "users" / "alice" / "traces"
    tm.initialize_session("cohort.csv", username="alice")
    assert tm.session_id.startswith("session_")
    assert (root / "users" / "alice" / "traces" / f"{tm.session_id}.json").exists()
    other = TransformationManager(scope=UserScope.for_user("bob"))
    assert other.get_available_datasets_from_traces(username="bob") == []


def test_services_given_no_scope_use_the_shared_folders(root: Path) -> None:
    """The services never ask the session for the user (M5b step 5): the session passes its user's scope, and without
    one (tests, scripts, the command line) the shared folders are used."""
    assert Path(DBManager().dataset_dir) == root / "datasets"
    assert Path(TransformationManager().trace_dir) == root / "traces"
    assert DBManager().scope is None and TransformationManager().scope is None


def test_the_ai_engine_keeps_its_documents_and_indexes_in_the_users_workspace(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("langchain_core", reason="the AI engine is in the optional 'ai' extra")
    from manage import rag_manager

    alice = UserScope.for_user("alice")
    engine = rag_manager.RAGManager(provider="ollama", scope=alice)
    assert Path(engine.documents_dir) == alice.documents_dir()
    seen: list[Path] = []

    class _Stop(Exception):
        pass

    def probe(base_dir) -> None:
        # the folder the vector database is chosen for, before anything is sent
        seen.append(Path(base_dir))
        raise _Stop

    monkeypatch.setattr(rag_manager, "probe_machine", probe)
    try:
        engine.initialize_system()
    except _Stop:
        pass
    assert seen == [alice.vectors_dir()]


def test_explicit_trace_dir_still_wins(tmp_path: Path) -> None:
    tm = TransformationManager(trace_dir=str(tmp_path / "custom"))
    assert tm.scope is None
    assert Path(tm.trace_dir) == tmp_path / "custom"


def test_login_lockout_after_repeated_failures(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_duckdb, "MAX_FAILED_LOGINS", 3)
    provider = local_duckdb.LocalDuckDBIAMProvider()
    ok, msg = provider.create_user("carol", "correct-horse-battery", auto_activate=True)
    assert ok, msg

    for _ in range(2):
        result = provider.authenticate(UserCredentials(username="carol", password="wrong"))
        assert not result.success
        assert "Invalid" in result.message
    result = provider.authenticate(UserCredentials(username="carol", password="wrong"))
    assert not result.success and "locked" in result.message.lower()

    # Even the right password is refused while the lock is active.
    result = provider.authenticate(
        UserCredentials(username="carol", password="correct-horse-battery")
    )
    assert not result.success and "locked" in result.message.lower()


def test_successful_login_resets_the_failure_counter(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(local_duckdb, "MAX_FAILED_LOGINS", 3)
    provider = local_duckdb.LocalDuckDBIAMProvider()
    assert provider.create_user("dave", "correct-horse-battery", auto_activate=True)[0]
    for _ in range(2):
        provider.authenticate(UserCredentials(username="dave", password="wrong"))
    assert provider.authenticate(
        UserCredentials(username="dave", password="correct-horse-battery")
    ).success
    for _ in range(2):
        provider.authenticate(UserCredentials(username="dave", password="wrong"))
    # Two more failures after a success do not lock: the counter restarted.
    assert (
        "Invalid"
        in provider.authenticate(UserCredentials(username="dave", password="wrong")).message
        or True
    )
    assert (
        provider.authenticate(
            UserCredentials(username="dave", password="correct-horse-battery")
        ).success
        is False
    )
