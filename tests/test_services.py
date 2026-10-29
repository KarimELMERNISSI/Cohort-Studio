"""
The composition root (manage.services, M5b step 5): the services of the server process and of a session, built in
one place. It is headless: M8's command line builds the same services, without Streamlit or the AI packages.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import utils.data_paths as dp
from core.workspace import UserScope
from manage.paths import PathResolver
from manage.services import build_services, build_shared, same_user
from utils.settings import get_settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path / "data"))
    return tmp_path / "data"


def test_the_shared_services_are_the_process_instances_read_at_each_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import graph.manager
    import manage.compute
    import manage.iam.manager

    singletons = (
        (manage.iam.manager, "_GLOBAL_IAM_MANAGER"),
        (graph.manager, "_GLOBAL_MANAGER"),
        (manage.compute, "_budget"),
    )
    for module, name in singletons:
        monkeypatch.setattr(module, name, None)
    shared = build_shared(get_settings())
    assert all(getattr(module, name) is None for module, name in singletons)  # built nothing
    for _ in range(2):  # as a test installs its own, twice: never a copy kept from the first access
        installed = [object(), object(), object()]
        for (module, name), instance in zip(singletons, installed, strict=True):
            monkeypatch.setattr(module, name, instance)
        assert (
            shared.iam is installed[0]
            and shared.graphs is installed[1]
            and shared.mi_workers is installed[2]
        )


def test_the_shared_settings_are_the_current_ones_unless_a_command_line_gave_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = build_shared()
    monkeypatch.setenv("CMP_MI_MAX_WORKERS", "3")
    assert shared.settings.statistics.mi_max_workers == 3
    monkeypatch.setenv("CMP_MI_MAX_WORKERS", "2")
    assert shared.settings.statistics.mi_max_workers == 2  # read at each access
    given = get_settings()
    monkeypatch.setenv("CMP_MI_MAX_WORKERS", "1")
    assert build_shared(given).settings is given


def test_a_sessions_stores_are_built_on_first_use_with_its_scope(root: Path) -> None:
    shared = build_shared(get_settings())
    build_services(shared, None, PathResolver.for_files([]))
    assert not root.exists()  # a command line that saves nothing creates no folder
    alice = UserScope.for_user("alice")
    mine = build_services(shared, alice, PathResolver(scope=alice))
    assert mine.datasets.scope == alice and mine.trace.scope == alice
    datasets, trace = mine.datasets, mine.trace
    assert mine.datasets is datasets and mine.trace is trace  # built once
    assert Path(mine.trace.trace_dir) == alice.traces_dir()
    assert mine.paths.scope == alice and mine.shared is shared
    assert mine.protocols.root == alice.protocols_dir() and mine.protocols is mine.protocols
    with pytest.raises(ValueError, match="no user"):  # no shared folder of study definitions
        _ = build_services(shared, None, PathResolver.for_files([])).protocols


def test_a_session_keeps_the_stores_it_already_holds(root: Path) -> None:
    shared = build_shared(get_settings())
    alice = UserScope.for_user("alice")
    first = build_services(shared, alice, PathResolver(scope=alice))
    again = build_services(
        shared, alice, PathResolver(scope=alice), datasets=first.datasets, trace=first.trace
    )
    assert again.datasets is first.datasets and again.trace is first.trace


def test_two_scopes_are_the_same_users_when_they_name_the_same_folder() -> None:
    alice = UserScope.for_user("alice")
    assert same_user(alice, UserScope.for_user("alice", "admin"))  # a change of role
    assert not same_user(alice, UserScope.for_user("bob"))
    assert not same_user(alice, None) and same_user(None, None)
    # the class of a reloaded core.workspace
    reloaded = type("UserScope", (), {"username": "alice"})()
    assert same_user(alice, reloaded)


def test_a_command_line_builds_the_services_without_streamlit_or_the_ai_packages(
    tmp_path: Path,
) -> None:
    """M8's command line builds the same services, with the settings it loaded and only the files it names, and
    uses its trace: neither Streamlit nor an AI package is loaded (rule R5)."""
    probe = (
        "import sys; "
        "from manage.paths import PathResolver; "
        "from manage.services import build_services, build_shared; "
        "from utils.settings import load_settings; "
        "services = build_services(build_shared(load_settings()), None, PathResolver.for_files([])); "
        "services.trace.initialize_session('cohort.csv'); services.datasets; "
        "from manage.protocol_store import WorkspaceProtocolStore; "
        "WorkspaceProtocolStore(None, root=__import__('pathlib').Path('protocols')).studies(); "
        "print(sorted(m for m in ('streamlit', 'langchain_core', 'lancedb', 'qdrant_client', 'google.genai', "
        "'anthropic', 'openai', 'mistralai', 'ollama') if m in sys.modules))"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT), "DATA_ROOT": str(tmp_path / "data")}
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=True,
        timeout=300,
    )
    assert out.stdout.strip() == "[]"
