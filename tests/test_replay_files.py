"""
The files a replay reads (M8, the study package): ``manage.replay.engine.files_needed`` lists every file a trace makes
the replay read, with its role and the names the load recorded, ``locate`` finds one as the replay does, the Reproduce
page asks for the missing ones from that list, and a study package's resolver (``PathResolver.for_package``) reads its
members only, each by the step that reads it and the very string the trace records, never another file of the
recipient's; a member that holds a table as its step read it is read as that table, and no member by DuckDB.
``trace_problems`` says why a file is not a trace before anything reads it, ``on_step`` sees how each step was passed,
and ``ReplayOptions`` gives a legacy MissForest step the seed a package records and draws an imputation on request.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

import utils.data_paths as dp
from core.package import Cut, DataFile
from manage.paths import PathResolver, read_version
from manage.replay import engine
from manage.replay.engine import (
    NOT_REPLAYABLE,
    NeededFile,
    ReplayOptions,
    StepSeen,
    apply_steps,
    files_needed,
    load_source,
    locate,
    package_paths,
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
        # each step numbered from 1, as a study package's index numbers it
        NeededFile("enrichment", 2, ("uploads/labs.csv",), checksum="b" * 64),
        # read through its path only, as the replay reads a merge
        NeededFile("enrichment", 3, ("v_2026",)),
        NeededFile("snapshot", 4, ("traces/artifacts/s1/step_3.parquet",)),
        # the fifth step is recomputed: its recorded table is never read
        NeededFile("snapshot", 6, ("traces/s1/step_5.parquet",)),
        NeededFile("enrichment", 7, ("uploads/labs.csv",)),
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
            (None, "uploads/cohort.csv"): str(members / "data_1.csv"),
            (None, "gone.csv"): str(members / "absent.csv"),
        },
        {"v_2026": str(members / "version_1.parquet")},
    )
    assert paths.resolve("uploads/cohort.csv") == str((members / "data_1.csv").resolve())
    for recorded in (str(own), "uploads/labs.csv", "labs.csv", "data_1.csv", "gone.csv", "", None):
        assert paths.resolve(recorded) is None, recorded
    assert paths.resolve("uploads/cohort.csv", 2) is None  # another step's file is another member
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
        members.update({(needed.step, recorded): str(member) for recorded in needed.recorded})
    assert [needed.role for needed in files_needed(trace)] == ["source", "enrichment"]
    for original in (tmp_path / "cohort.csv", labs):  # decoys at the recorded paths
        original.write_text("pid,age,hb\n9,9,9\n", encoding="utf-8")
    result = run_trace(trace, paths=PathResolver.for_package(members))
    assert result.error is None and result.reproducible, result.warnings
    assert result.verified_steps >= 2
    missing = run_trace(trace, paths=PathResolver.for_package({})).error or ""
    assert missing.startswith("Source file not found")


class _Recording:
    """A resolver that notes every (step, string) the replay looks up, and finds what ``inner`` finds."""

    def __init__(self, inner: PathResolver) -> None:
        self.inner, self.asked = inner, []

    package = True

    def resolve(self, recorded: object, step: int | None = None) -> str | None:
        self.asked.append((step, recorded))
        return self.inner.resolve(recorded, step)

    def reads_as_table(self, recorded: object, step: int | None = None) -> bool:
        return self.inner.reads_as_table(recorded, step)

    def dataset(self, name: object) -> str | None:
        self.asked.append((None, name))
        return self.inner.dataset(name)


def test_the_replay_looks_up_only_what_files_needed_lists(tmp_path: Path) -> None:
    """Each (step, string) a replay asks its resolver for is one files_needed lists, recorded path or saved
    version, at the step that reads it."""
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
                (None, "b/source.csv"): str(package / "source.csv"),
                (2, "c/labs.csv"): str(package / "labs.csv"),
                (3, "d/kept.parquet"): str(package / "kept.parquet"),
            }
        )
    )
    result = run_trace(trace, paths=recording)  # type: ignore[arg-type]
    assert result.error is None, result.error
    listed = {
        (needed.step, name)
        for needed in files_needed(trace)
        for name in (*needed.recorded, *needed.versions)
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


# ----------------------------------------------------------------------------- C4a: per step, tables, options


def _merge(path: str, column: str) -> dict:
    return {
        "function": "enrichment",
        "params": {
            "enrichment_file_path": path,
            "identifier": ["pid"],
            "strategy": "left",
            "conflict_resolution": "keep",
            "retained_enrichment_columns": ["pid", column],
            "file_hash": "f" * 64,  # the original workbook's, which no cut has
            # the workbook's reading options: a cut is the table read, never read with them again
            "parse_options": {"sheet_name": "labs", "separator": ";"},
        },
    }


def _two_sheets(package: Path) -> dict:
    """A study that merged two sheets of one workbook, its trace recording one string for both; the package's members
    are the two tables as their merges read them, one per step."""
    pd.DataFrame({"pid": [1, 2], "age": [60, 70]}).to_csv(package / "source.csv", index=False)
    pd.DataFrame({"pid": [1, 2], "hb": [12.0, 13.0]}).to_parquet(package / "merge_01.parquet")
    pd.DataFrame({"pid": [1, 2], "crp": [3.0, 8.0]}).to_parquet(package / "merge_02.parquet")
    return {
        "source_dataset": "a/source.csv",
        "steps": [
            {"function": "initial_load", "params": {"source_path": "a/source.csv"}},
            _merge("uploads/visits.xlsx", "hb"),
            _merge("uploads/visits.xlsx", "crp"),
        ],
    }


def test_two_merges_of_one_workbook_read_their_own_members(tmp_path: Path) -> None:
    """Two steps that record one string (two sheets of one workbook) are two members of a package: the resolver
    answers per step, and each cut is read as the table its merge read, without the workbook's reading options and
    without the warning that the original file's checksum differs."""
    package = tmp_path / "package"
    package.mkdir()
    trace = _two_sheets(package)
    needed = files_needed(trace)
    assert [(n.role, n.step, n.recorded) for n in needed[1:]] == [
        ("enrichment", 2, ("uploads/visits.xlsx",)),
        ("enrichment", 3, ("uploads/visits.xlsx",)),
    ]
    cuts = {
        (2, "uploads/visits.xlsx"): "merge_01.parquet",
        (3, "uploads/visits.xlsx"): "merge_02.parquet",
    }
    paths = PathResolver.for_package(
        {(None, "a/source.csv"): str(package / "source.csv")}
        | {key: str(package / member) for key, member in cuts.items()},
        tables=cuts,
    )
    assert paths.reads_as_table("uploads/visits.xlsx", 2) and not paths.reads_as_table(
        "a/source.csv"
    )
    assert locate(needed[2], paths) == str((package / "merge_02.parquet").resolve())
    result = run_trace(trace, paths=paths)
    assert result.error is None, result.error
    assert result.df is not None and list(result.df.columns) == ["pid", "age", "hb", "crp"]
    assert result.df["crp"].tolist() == [3.0, 8.0]
    assert not any("checksum differs" in warning for warning in result.warnings), result.warnings
    # without the per-step key, a member is no member of that step
    one = PathResolver.for_package({(None, "a/source.csv"): str(package / "source.csv")})
    assert run_trace(trace, paths=one).missing_file == (1, "uploads/visits.xlsx")


def test_a_package_resolver_is_built_from_its_index(tmp_path: Path) -> None:
    """package_paths matches each file of the trace with the index's file of that role and step: the members, the
    tables as read, nothing else; for_package refuses a table that is no member."""
    package = tmp_path / "package"
    package.mkdir()
    trace = _two_sheets(package)
    files = (
        DataFile("source", None, "source.csv", False, 2, 2),
        DataFile("enrichment", 2, "merge_01.parquet", True, 2, 2, Cut("rows_taken", 5, "f" * 64)),
        DataFile("enrichment", 3, "merge_02.parquet", True, 2, 2, Cut("every_row", 2, "f" * 64)),
    )
    paths = package_paths(trace, files, package)
    assert paths.reads_as_table("uploads/visits.xlsx", 3) and not paths.reads_as_table(
        "a/source.csv"
    )
    result = run_trace(trace, paths=paths)
    assert result.error is None and result.matches, result.warnings
    assert result.df is not None and list(result.df.columns) == ["pid", "age", "hb", "crp"]
    # a merge the package does not hold is missing, never looked for elsewhere
    held = package_paths(trace, files[:2], package)
    assert run_trace(trace, paths=held).missing_file == (2, "uploads/visits.xlsx")
    with pytest.raises(ValueError):
        PathResolver.for_package({}, tables=[(2, "uploads/visits.xlsx")])


def test_a_source_carried_as_its_table_is_read_as_that_table(tmp_path: Path) -> None:
    """A workbook's table, carried as source.parquet: read by pandas as it was written, never as a workbook, and
    replayed from there."""
    package = tmp_path / "package"
    package.mkdir()
    pd.DataFrame({"pid": [1, 2, 3], "age": [60, 70, 80]}).to_parquet(package / "source.parquet")
    trace = {
        "source_dataset": "uploads/cohort.xlsx",
        "steps": [
            {"function": "initial_load", "params": {"source_path": "uploads/cohort.xlsx"}},
            {"function": "rename_columns", "params": {"mapping": {"age": "age_years"}}},
        ],
    }
    paths = package_paths(trace, [DataFile("source", None, "source.parquet", True, 3, 2)], package)
    df, error = load_source(trace, paths)
    assert error is None and df is not None and df["age"].tolist() == [60, 70, 80]
    assert df.index.name == "__row_id"  # the row ids a fresh load gives
    result = run_trace(trace, paths=paths)
    assert result.error is None and result.df is not None
    assert list(result.df.columns) == ["pid", "age_years"]


def test_no_member_of_a_package_is_read_by_duckdb_or_by_a_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A member pandas cannot read is not read: a received file has one reader. A member the loader does not know
    is not looked up as a saved version of its name, and an error names the file, never its folder (which can name
    the account it lies under)."""
    import duckdb

    called: list[object] = []
    real_connect = duckdb.connect

    def noted(*args: object, **kwargs: object) -> object:
        called.append(
            args
        )  # noted, not raised: read_version's own fallback would swallow an exception
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", noted)
    damaged = b"PAR1 not a parquet table PAR1"
    (tmp_path / "version.parquet").write_bytes(damaged)
    trace = {"source_dataset": "cohort_v1", "steps": []}
    paths = PathResolver.for_package({}, {"cohort_v1": str(tmp_path / "version.parquet")})
    df, error = load_source(trace, paths)
    assert df is None and error == "Failed to load source dataset (cohort_v1)"
    assert read_version(str(tmp_path / "version.parquet"), duckdb_fallback=False) is None
    # a whole package replayed with a damaged source table, cut and recorded table: each refused, none by DuckDB
    package = tmp_path / "damaged"
    package.mkdir()
    for name in ("source.parquet", "merge_01.parquet", "snapshot_01.parquet"):
        (package / name).write_bytes(damaged)
    pd.DataFrame({"pid": [1, 2], "age": [60, 70]}).to_parquet(package / "fine_source.parquet")
    workbook = {
        "source_dataset": "uploads/cohort.xlsx",
        "steps": [
            {"function": "initial_load", "params": {"source_path": "uploads/cohort.xlsx"}},
            _merge("uploads/visits.xlsx", "hb"),
            {"function": SNAPSHOT_STEP, "output_dataset_path": "t/step_3.parquet"},
        ],
    }
    start = DataFile("source", None, "source.parquet", True, 2, 2)
    cut = DataFile("enrichment", 2, "merge_01.parquet", True, 2, 2, Cut("rows_taken", 2, "f" * 64))
    kept = DataFile("snapshot", 3, "snapshot_01.parquet", False, 2, 2)
    assert run_trace(workbook, paths=package_paths(workbook, [start, cut, kept], package)).error
    (package / "source.parquet").write_bytes((package / "fine_source.parquet").read_bytes())
    assert run_trace(workbook, paths=package_paths(workbook, [start, cut, kept], package)).error
    without_merge = {**workbook, "steps": [workbook["steps"][0], {**workbook["steps"][2]}]}
    snapshot_only = DataFile("snapshot", 2, "snapshot_01.parquet", False, 2, 2)
    replayed = run_trace(
        without_merge, paths=package_paths(without_merge, [start, snapshot_only], package)
    )
    assert replayed.error is not None
    assert called == []
    # a merge whose member has an extension the loader does not know, under a folder that names its account
    folder = tmp_path / "Jean Dupont canary-9X4"
    folder.mkdir()
    pd.DataFrame({"pid": [1, 2], "age": [60, 70]}).to_csv(folder / "source.csv", index=False)
    (folder / "labs.dat").write_text("pid,hb\n1,12\n", encoding="utf-8")
    pd.DataFrame({"pid": [1, 2], "hb": [9.0, 9.0]}).to_parquet(
        folder / "labs.parquet"
    )  # a decoy name
    merged = {
        "source_dataset": "a/source.csv",
        "steps": [
            {"function": "initial_load", "params": {"source_path": "a/source.csv"}},
            _merge("uploads/labs", "hb"),
        ],
    }
    members = {
        (None, "a/source.csv"): str(folder / "source.csv"),
        (2, "uploads/labs"): str(folder / "labs.dat"),
    }
    result = run_trace(
        merged, paths=PathResolver.for_package(members, {"labs": str(folder / "labs.parquet")})
    )
    assert result.error is not None and "labs.dat" in result.error
    assert "canary-9X4" not in result.error and "Dupont" not in result.error


def test_a_legacy_missforest_step_draws_with_the_given_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A step recorded before MissForest's seed was kept draws with the seed the caller gives (a package records its
    exporter's); a step that recorded its own keeps it, a recorded 0 too; without the caller's, this process's, as
    before. (The seed reaches MissForest only, and matters there for a column fitted on more than 200,000 rows.)"""
    seen: list[object] = []
    real = engine.impute_rows

    def recording(*args: object, **kwargs: object) -> object:
        seen.append(kwargs.get("random_state"))
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "impute_rows", recording)
    frame = pd.DataFrame({"x": [1.0, None, 3.0, 4.0], "y": [2.0, 3.0, None, 5.0]})
    legacy = {"function": "imputation", "params": {"numerical_imputation_method": "Mean"}}

    def kept(seed: int) -> dict:
        params = {"numerical_imputation_method": "Mean", "random_state": seed}
        return {"function": "imputation", "params": params}

    for step in (legacy, kept(7), kept(0)):
        apply_steps(frame, [step], options=ReplayOptions(process_seed=1234))
    apply_steps(frame, [legacy])
    assert seen == [1234, 7, 0, None]


def _recorded(df: pd.DataFrame, spec: object) -> dict:
    """An mi_impute step as the session records one: its counts, its cells' hash, its own hash."""
    from biostats.mi import impute

    run = impute(df, spec)  # type: ignore[arg-type]
    return {
        "function": "mi_impute",
        "hash": "e" * 64,
        "params": {
            "spec": run.spec.model_dump(),
            "methods": dict(run.methods),
            "m": run.mi.m,
            "n_rows": len(run.mi.base),
            "n_imputed": run.mi.n_imputed(),
            "cells_fp": run.mi.fingerprint(),
        },
    }


def test_an_imputation_is_drawn_on_request_and_checked_when_it_is_not() -> None:
    """The pre-flight holds the session's imputation: a step it does not name (by its hash) is checked without its
    draw against the counts it recorded, passed as not drawn and never counted as reproduced; a step named is drawn
    in this process."""
    from biostats.mi import ImputationSpec
    from tests.test_mi_replay import _data

    df = _data(n=60)
    spec = ImputationSpec(columns=["x", "g", "y"], m=2, maxit=1, seed=5)
    step = _recorded(df, spec)
    seen: list[StepSeen] = []
    held = apply_steps(df, [step], options=ReplayOptions(draw=frozenset()), on_step=seen.append)
    assert held.error is None and held.not_drawn == [0] and not held.imputations
    assert held.matches and not held.reproducible and held.verified_steps == 0
    assert [s.mode for s in seen] == ["not_drawn"] and held.mi_rows[0].equals(df.index)
    progress: list[float] = []
    named = ReplayOptions(draw=frozenset({step["hash"]}), progress=progress.append)
    drawn = apply_steps(df, [step], options=named)
    assert drawn.error is None and drawn.reproducible and drawn.imputations[0].m == 2
    assert progress and progress[-1] == 1.0
    # drawn on other inputs than the replay gives: listed as a divergence, with the counts
    other = {**step, "params": {**step["params"], "n_rows": 278, "n_imputed": {"x": 999}}}
    differs = apply_steps(df, [other], options=ReplayOptions(draw=frozenset()))
    assert differs.mi_divergence == [0] and not differs.matches
    (said,) = [w for w in differs.warnings if "other inputs" in w]
    assert "drawn on 278 rows, 60 now" in said and "x: 999 cells imputed then" in said
    # an unknown way of deriving variables is refused even on a step not drawn
    unknown = {**step, "params": {**step["params"], "derived": {"approach": "later"}}}
    refused = apply_steps(df, [unknown], options=ReplayOptions(draw=frozenset()))
    assert refused.error is not None and "derivation approach" in refused.error


def test_matches_fails_a_step_skipped_or_redone_in_part(tmp_path: Path) -> None:
    """The gate of a study package's checks: a step of a kind this version does not know, a step redone in part
    only, and a recorded table whose member is missing each leave it False, as a divergence does."""
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    unknown = {"function": "a_later_kind"}
    in_part = {
        "function": "rename_columns",
        "replayable": False,
        "params": {"mapping": {"a": "x"}},
        "output_dataset_path": "t/step_1.parquet",
    }
    gone = {"function": SNAPSHOT_STEP, "output_dataset_path": "t/step_1.parquet"}
    for step in (unknown, in_part, gone):
        result = apply_steps(frame, [step], paths=PathResolver.for_package({}))
        assert result.error is None and result.not_replayable == [0], step
        assert not result.matches, step
    assert apply_steps(frame, [{"function": "rename_columns", "params": {"mapping": {}}}]).matches


def test_a_step_not_drawn_on_as_many_other_rows_is_named() -> None:
    """Counts alone cannot see a population of the same size made of other rows: its fingerprint does."""
    from biostats.mi import ImputationSpec
    from core.fingerprint import ids_fingerprint
    from tests.test_mi_replay import _data

    df = _data(n=60)
    spec = ImputationSpec(columns=["x", "g", "y"], m=2, maxit=1, seed=5)
    step = _recorded(df, spec)
    elsewhere = {**step, "population": {"rows_fp": ids_fingerprint(range(100, 160))}}
    result = apply_steps(df, [elsewhere], options=ReplayOptions(draw=frozenset()))
    assert result.mi_divergence == [0] and result.mi_inputs_differ == [0] and not result.matches
    assert any("drawn on 60 other rows" in w for w in result.warnings)
    same = {**step, "population": {"rows_fp": ids_fingerprint(df.index)}}
    assert apply_steps(df, [same], options=ReplayOptions(draw=frozenset())).matches
    with pytest.raises(TypeError):
        ReplayOptions(draw=frozenset({1}))  # type: ignore[arg-type]


def test_a_package_resolver_reads_a_saved_version_and_a_recorded_table(tmp_path: Path) -> None:
    """A trace that starts from a saved version, whose table a later step took from its recording: the version by
    the names its load recorded, with its row ids, and the recorded table at its step."""
    package = tmp_path / "package"
    package.mkdir()
    version = pd.DataFrame({"pid": [1, 2, 3], "age": [60, 70, 80]})
    version.index = pd.Index([4, 7, 9], name="__row_id")
    version.to_parquet(package / "version.parquet")
    version.iloc[:2].to_parquet(package / "snapshot_01.parquet")
    trace = {
        "source_dataset": "cohort",
        "steps": [
            {
                "function": "initial_load",
                "params": {
                    "source_path": "datasets/cohort_v1.parquet",
                    "dataset_name": "cohort",
                    "version": 1,
                },
            },
            {"function": SNAPSHOT_STEP, "output_dataset_path": "traces/s1/step_2.parquet"},
        ],
    }
    files = (
        DataFile("saved_version", None, "version.parquet", False, 3, 2),
        DataFile("snapshot", 2, "snapshot_01.parquet", False, 2, 2),
    )
    paths = package_paths(trace, files, package)
    df, error = load_source(trace, paths)
    assert error is None and df is not None and list(df.index) == [4, 7, 9]
    result = run_trace(trace, paths=paths)
    assert result.error is None and result.from_snapshot == [1] and result.matches
    assert result.df is not None and list(result.df.index) == [4, 7]
