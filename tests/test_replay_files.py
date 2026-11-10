"""
The files a replay reads (M8, the study package): ``manage.replay.engine.files_needed`` lists every file a trace makes
the replay read, with its role and the names the load recorded, ``locate`` finds one as the replay does, the Reproduce
page asks for the missing ones from that list, and a study package's resolver (``PathResolver.for_package``) reads its
members only, each by the very string the trace records, never another file of the recipient's. ``trace_problems``
says why a file is not a trace before anything reads it, and ``on_step`` sees how each step was passed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

import utils.data_paths as dp
from manage.paths import PathResolver
from manage.replay.engine import (
    NOT_REPLAYABLE,
    NeededFile,
    StepSeen,
    apply_steps,
    files_needed,
    load_source,
    locate,
    run_trace,
    trace_problems,
)

SNAPSHOT_STEP = sorted(NOT_REPLAYABLE)[0]


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path / "data"))
    return tmp_path


def _trace() -> dict:
    return {
        "source_dataset": "traces/artifacts/s1/source.csv",
        "steps": [
            {
                "function": "initial_load",
                "params": {"source_path": "uploads/cohort.csv", "file_hash": "a" * 64},
            },
            {
                "function": "enrichment",
                "params": {"enrichment_file_path": "uploads/labs.csv", "file_hash": "b" * 64},
            },
            {"function": "enrichment", "params": {"enrichment_file_path": "v_2026"}},
            {
                "function": SNAPSHOT_STEP,
                "output_dataset_path": "traces/artifacts/s1/step_3.parquet",
            },
            {
                "function": "rename_columns",
                "output_dataset_path": "traces/artifacts/s1/step_4.parquet",
            },
            {
                "function": "rename_columns",
                "replayable": False,
                "output_dataset_path": "traces/s1/step_5.parquet",
            },
            {"function": "enrichment", "params": {"enrichment_file_path": "uploads/labs.csv"}},
            # a documenting step is never redone, and a recorded table that is no parquet is never read
            {"function": "save_version", "replayable": False, "output_dataset_path": "x.parquet"},
            {"function": SNAPSHOT_STEP, "output_dataset_path": "traces/s1/step_8.csv"},
        ],
    }


def test_every_file_a_replay_reads_is_listed_with_its_role_and_step() -> None:
    assert files_needed(_trace()) == [
        NeededFile(
            "source",
            None,
            ("traces/artifacts/s1/source.csv", "uploads/cohort.csv"),
            checksum="a" * 64,
        ),
        NeededFile("enrichment", 1, ("uploads/labs.csv",), checksum="b" * 64),
        NeededFile(
            "enrichment", 2, ("v_2026",)
        ),  # read through its path only, as the replay reads a merge
        NeededFile("snapshot", 3, ("traces/artifacts/s1/step_3.parquet",)),
        # step 4 is recomputed: its recorded table is never read
        NeededFile("snapshot", 5, ("traces/s1/step_5.parquet",)),
        NeededFile("enrichment", 6, ("uploads/labs.csv",)),
    ]
    assert files_needed({"steps": [{"function": "rename_columns"}]}) == []


def test_the_saved_version_a_source_stands_for_is_the_one_its_load_recorded() -> None:
    begun_from_a_version = {  # as AppState records a saved version loaded again
        "source_dataset": "cohort",
        "parent": {"dataset": "cohort_v1", "trace_session": "s0"},
        "steps": [
            {
                "function": "initial_load",
                "params": {
                    "source_path": "datasets/cohort_v1.parquet",
                    "dataset_name": "cohort",
                    "version": 1,
                },
            }
        ],
    }
    (source,) = files_needed(begun_from_a_version)
    assert source == NeededFile(
        "saved_version", None, ("cohort", "datasets/cohort_v1.parquet"), ("cohort_v1",)
    )
    without_lineage = json.loads(json.dumps(begun_from_a_version))
    del without_lineage["parent"]
    assert files_needed(without_lineage)[0].versions == ("cohort_v1",)
    legacy = files_needed({"source_dataset": "cohort_v1", "steps": []})  # no load recorded
    assert legacy == [NeededFile("saved_version", None, ("cohort_v1",), ("cohort_v1",))]
    replayed = {
        "source_dataset": "s1_step3",
        "parent": {"trace_session": "s1", "upto_step": 3},
        "steps": [],
    }
    assert files_needed(replayed) == [NeededFile("replayed_state", None, ("s1_step3",))]


@pytest.mark.parametrize(
    ("trace", "problem"),
    [
        ([], "it is not a JSON object"),
        ({"steps": {"function": "x"}}, "it has no list of steps"),
        ({"steps": ["initial_load"]}, "step 1 is not an object"),
        ({"steps": [{"function": "initial_load"}, {"params": {}}]}, "step 2 names no step kind"),
        (
            {"steps": [{"function": "enrichment", "params": ["a.csv"]}]},
            "step 1's settings are not an object",
        ),
        ({"source_dataset": ["a.csv"], "steps": []}, "its source dataset is not a file name"),
        ({"parent": "v1", "steps": []}, "its parent is not an object"),
        ({"steps": [{"function": "x", "population": [1]}]}, "step 1's population is not an object"),
        ({"steps": [{"function": "x", "data": "fp"}]}, "step 1's data record is not an object"),
        (
            {"steps": [{"function": "enrichment", "params": {"enrichment_file_path": 3}}]},
            "step 1's merged file is not a file name",
        ),
        (
            {"steps": [{"function": "initial_load", "params": {"parse_options": "csv"}}]},
            "step 1's reading options cannot be read",
        ),
    ],
)
def test_a_file_that_is_no_trace_is_refused_with_the_reason(trace: object, problem: str) -> None:
    assert problem in trace_problems(trace)
    assert trace_problems(_trace()) == []
    if isinstance(trace, dict):
        error = run_trace(trace).error or ""
        assert error.startswith("Not a trace this version can replay: ")
        files_needed(trace)  # never raises on a JSON object


def test_a_package_resolver_reads_its_members_only(root: Path) -> None:
    """A received trace must never reach the recipient's own files: no search by name, no recorded path read."""
    members = root / "package"
    members.mkdir()
    (members / "data_1.csv").write_text("who,n\na,1\n", encoding="utf-8")
    pd.DataFrame({"x": [1]}).to_parquet(members / "version_1.parquet")
    own = root / "uploads" / "labs.csv"  # the recipient's file, of the very name the trace records
    own.parent.mkdir()
    own.write_text("who,n\nmine,9\n", encoding="utf-8")
    paths = PathResolver.for_package(
        {
            "uploads/cohort.csv": str(members / "data_1.csv"),
            "gone.csv": str(members / "absent.csv"),
        },
        {"v_2026": str(members / "version_1.parquet")},
    )
    assert paths.resolve("uploads/cohort.csv") == str((members / "data_1.csv").resolve())
    for recorded in (str(own), "uploads/labs.csv", "labs.csv", "data_1.csv", "gone.csv", "", None):
        assert paths.resolve(recorded) is None, recorded
    assert paths.dataset("v_2026") == str((members / "version_1.parquet").resolve())
    assert paths.dataset("version_1") is None and paths.dataset("../package/version_1") is None
    # a file the user would choose on the Reproduce page is no member either
    assert paths.with_named({"uploads/labs.csv": str(own)}).resolve("uploads/labs.csv") is None


def test_a_recorded_study_replays_from_a_packages_members_alone(tmp_path: Path) -> None:
    """The files of a recorded merge, as files_needed lists them, extracted under other names, while other tables lie
    at every recorded path: the replay reads the members and reproduces every recorded fingerprint."""
    from tests.test_enrichment_replay import _record

    labs = tmp_path / "labs.csv"
    pd.DataFrame({"pid": [1, 2, 3], "hb": [12.5, 13.1, 11.8]}).to_csv(labs, index=False)
    main = pd.DataFrame({"pid": [1, 2, 3], "age": [61, 70, 55]})
    tm, session = _record(tmp_path, main, pd.read_csv(labs), labs, ["pid"])
    trace = json.loads(json.dumps(tm.get_trace(), default=str))
    package = tmp_path / "package"
    package.mkdir()
    members: dict[str, str] = {}
    for n, needed in enumerate(files_needed(trace)):
        found = next(p for p in needed.recorded if Path(p).is_file())
        member = package / f"member_{n}{Path(found).suffix}"
        shutil.copy(found, member)
        members.update(dict.fromkeys(needed.recorded, str(member)))
    assert [needed.role for needed in files_needed(trace)] == ["source", "enrichment"]
    for original in (tmp_path / "cohort.csv", labs):  # decoys at the recorded paths
        original.write_text("pid,age,hb\n9,9,9\n", encoding="utf-8")
    result = run_trace(trace, paths=PathResolver.for_package(members))
    assert result.error is None and result.reproducible, result.warnings
    assert result.verified_steps >= 2
    missing = run_trace(trace, paths=PathResolver.for_package({})).error or ""
    assert missing.startswith("Source file not found")


class _Recording:
    """A resolver that notes every string the replay looks up, and finds what ``inner`` finds."""

    def __init__(self, inner: PathResolver) -> None:
        self.inner, self.asked = inner, []

    def resolve(self, recorded: object) -> str | None:
        self.asked.append(recorded)
        return self.inner.resolve(recorded)

    def dataset(self, name: object) -> str | None:
        self.asked.append(name)
        return self.inner.dataset(name)


def test_the_replay_looks_up_only_what_files_needed_lists(tmp_path: Path) -> None:
    """Each string a replay asks its resolver for is one files_needed lists, recorded path or saved version."""
    package = tmp_path / "package"
    package.mkdir()
    pd.DataFrame({"pid": [1, 2], "age": [60, 70]}).to_csv(package / "source.csv", index=False)
    pd.DataFrame({"pid": [1, 2], "hb": [12.0, 13.0]}).to_csv(package / "labs.csv", index=False)
    table = pd.DataFrame({"pid": [1], "age": [60]})
    table.index = pd.Index([0], name="__row_id")
    table.to_parquet(package / "kept.parquet")
    trace = {
        "source_dataset": "a/source.csv",
        "steps": [
            {"function": "initial_load", "params": {"source_path": "b/source.csv"}},
            {
                "function": "enrichment",
                "params": {
                    "enrichment_file_path": "c/labs.csv",
                    "identifier": ["pid"],
                    "strategy": "left",
                    "conflict_resolution": "keep",
                },
            },
            {"function": SNAPSHOT_STEP, "output_dataset_path": "d/kept.parquet"},
        ],
    }
    recording = _Recording(
        PathResolver.for_package(
            {
                "b/source.csv": str(package / "source.csv"),
                "c/labs.csv": str(package / "labs.csv"),
                "d/kept.parquet": str(package / "kept.parquet"),
            }
        )
    )
    result = run_trace(trace, paths=recording)  # type: ignore[arg-type]
    assert result.error is None, result.error
    listed = {
        name for needed in files_needed(trace) for name in (*needed.recorded, *needed.versions)
    }
    assert recording.asked and set(recording.asked) <= listed, recording.asked
    assert result.from_snapshot == [2] and not result.reproducible  # taken from its recorded table


def test_on_step_sees_how_each_step_was_passed() -> None:
    seen: list[StepSeen] = []
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    steps = [
        {"function": "rename_columns", "params": {"mapping": {"a": "x"}}},
        {"function": "initial_load"},  # documents the state only
        {"function": SNAPSHOT_STEP},  # cannot be recomputed, and no recorded table: skipped
        {
            "function": "variable_computation",
            "params": {"formula": "nowhere +", "variable_name": "z"},
        },
    ]
    result = apply_steps(frame, steps, on_step=seen.append)
    assert result.error is not None  # the replay stops at the formula it cannot read: never seen
    assert [(s.index, s.mode) for s in seen] == [
        (0, "recomputed"),
        (1, "documented"),
        (2, "skipped"),
    ]
    assert list(seen[0].before.columns) == ["a", "b"] and list(seen[0].after.columns) == ["x", "b"]
    assert all(s.fingerprint and s.expected is None for s in seen)


def test_a_saved_version_is_found_by_the_name_its_load_recorded(root: Path) -> None:
    """A version's base name is not a version: with the version gone and a table of the base name in the folder,
    nothing is read; a legacy trace naming the version is read whatever path its load recorded."""
    folder = Path(dp.get_datasets_dir())
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"other": [9]}).to_parquet(folder / "cohort.parquet")
    begun = {
        "source_dataset": "cohort",
        "parent": {"dataset": "cohort_v1"},
        "steps": [
            {"function": "initial_load", "params": {"source_path": "datasets/cohort_v1.parquet"}}
        ],
    }
    df, error = load_source(begun, PathResolver.unconfined())
    assert df is None and error == "Source file not found: datasets/cohort_v1.parquet"
    pd.DataFrame({"x": [1, 2]}).to_parquet(folder / "cohort_v1.parquet")
    df, error = load_source(begun, PathResolver.unconfined())
    assert error is None and df is not None and list(df.columns) == ["x"]
    legacy = {
        "source_dataset": "cohort_v1",
        "steps": [{"function": "initial_load", "params": {"source_path": "uploads/gone.csv"}}],
    }
    df, error = load_source(legacy, PathResolver.unconfined())
    assert error is None and df is not None and len(df) == 2
    (needed,) = [f for f in files_needed(begun) if f.step is None]
    assert locate(needed, PathResolver.unconfined()) is not None


def test_the_reproduce_page_asks_for_each_missing_file_once(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A snapshot is never asked for, a file two merges read is asked for once, naming both steps, and a merge
    recorded by a saved version's name is asked for (the replay reads a merge through its path only)."""
    import streamlit as st

    from app_pages import reproduce_analysis as page

    monkeypatch.setattr(page, "_resolver", PathResolver.unconfined)
    monkeypatch.setattr(st, "session_state", {})
    missing = page._missing_files(_trace())
    assert list(missing) == ["traces/artifacts/s1/source.csv", "uploads/labs.csv", "v_2026"]
    assert missing["traces/artifacts/s1/source.csv"]["type"] == "source"
    assert missing["uploads/labs.csv"]["description"] == "Enrichment File (Steps 2, 7)"
    assert missing["v_2026"]["description"] == "Enrichment File (Step 3)"
