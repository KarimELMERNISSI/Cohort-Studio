"""The snapshots of changes that never reach the trace are deleted, and only those (manage/snapshots.py, M5b)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import utils.data_paths as dp
from core.workspace import UserScope
from manage.snapshots import delete_unreferenced, referenced


@pytest.fixture
def scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> UserScope:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    return UserScope.for_user("alice")


def _snapshot(folder: Path, name: str) -> Path:
    path = folder / name
    path.write_bytes(b"PAR1")
    return path


def _trace(scope: UserScope, name: str, steps: list[dict]) -> None:
    trace = {"session_id": name, "steps": steps}
    (scope.traces_dir() / name).write_text(json.dumps(trace), encoding="utf-8")


def test_a_snapshot_no_trace_names_is_deleted(scope: UserScope) -> None:
    kept = _snapshot(scope.snapshots_dir(), "20261020_120000_enrichment_aaaa1111.parquet")
    gone = _snapshot(scope.snapshots_dir(), "20261020_120100_enrichment_bbbb2222.parquet")
    _trace(scope, "session_1.json", [{"output_dataset_path": f"traces/snapshots/{kept.name}"}])
    # recorded relative to the workspace (AppState.recorded_path), or absolute
    deleted = delete_unreferenced([f"traces/snapshots/{kept.name}", str(gone)], scope)
    assert deleted == [gone.name] and kept.exists() and not gone.exists()


def test_a_name_in_any_trace_or_in_the_session_keeps_the_file(scope: UserScope) -> None:
    folder = scope.snapshots_dir()
    named, unsaved, free = (
        _snapshot(folder, n) for n in ("a_Imputation.parquet", "b.parquet", "c.parquet")
    )
    # recorded on another machine, in a step's parameters: the replay looks a file up by its name
    windows = "C:\\Users\\x\\traces\\snapshots\\A_IMPUTATION.parquet"
    _trace(scope, "cohort_v2_trace.json", [{"params": {"file": windows}}])
    # the session's own trace, not saved yet
    session = [{"output_dataset_path": "traces/snapshots/b.parquet"}]
    deleted = delete_unreferenced([str(named), str(unsaved), str(free)], scope, session)
    assert deleted == ["c.parquet"] and named.exists() and unsaved.exists()


def test_nothing_outside_the_snapshots_folder_is_deleted(scope: UserScope, tmp_path: Path) -> None:
    dataset = _snapshot(scope.datasets_dir(), "cohort_v1.parquet")
    nested = scope.snapshots_dir() / "sub"
    nested.mkdir()
    inner = _snapshot(nested, "x.parquet")
    text = _snapshot(scope.snapshots_dir(), "notes.csv")
    outside = _snapshot(tmp_path, "elsewhere.parquet")
    paths = [
        "datasets/cohort_v1.parquet",
        "traces/snapshots/../../datasets/cohort_v1.parquet",
        str(dataset),
        "traces/snapshots/sub/x.parquet",
        "traces/snapshots/notes.csv",
        "../../elsewhere.parquet",
        str(outside),
        "traces/snapshots/missing.parquet",
        None,
        "",
    ]
    assert delete_unreferenced(paths, scope) == []
    assert all(p.exists() for p in (dataset, inner, text, outside))


def test_an_unreadable_trace_keeps_every_snapshot(scope: UserScope) -> None:
    snapshot = _snapshot(scope.snapshots_dir(), "d.parquet")
    (scope.traces_dir() / "broken.json").write_text("{not json", encoding="utf-8")
    assert referenced(scope.traces_dir()) is None
    assert delete_unreferenced([str(snapshot)], scope) == [] and snapshot.exists()


def test_outside_a_session_the_data_folder_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    folder = Path(dp.get_traces_dir()) / "snapshots"
    folder.mkdir()
    snapshot = _snapshot(folder, "e.parquet")
    assert delete_unreferenced(["traces/snapshots/e.parquet"], None) == ["e.parquet"]
    assert not snapshot.exists()


def test_a_trace_nested_too_deeply_keeps_every_snapshot(scope: UserScope) -> None:
    snapshot = _snapshot(scope.snapshots_dir(), "f.parquet")
    deep = "[" * 100_000 + "]" * 100_000  # json raises RecursionError, not ValueError
    (scope.traces_dir() / "deep.json").write_text(deep, encoding="utf-8")
    assert referenced(scope.traces_dir()) is None
    assert delete_unreferenced([str(snapshot)], scope) == [] and snapshot.exists()


def test_a_snapshot_whose_writing_fails_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pandas as pd

    from app_pages.data_enrichment import save_snapshot

    def fail(self, path, *args, **kwargs):
        Path(path).write_bytes(b"PAR1 partial")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    assert save_snapshot(pd.DataFrame({"a": [1]}), "x", base_dir=str(tmp_path)) is None
    assert list(tmp_path.iterdir()) == []
