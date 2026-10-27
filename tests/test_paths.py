"""
The files a replay may read (manage.paths, M5b step 5).

A trace is user-supplied JSON. In a session, a path it records is looked for in the user's workspace only and a saved
version it names in the datasets folder only; a command line reads only the files named on it; tests and scripts ask
for the shared folders explicitly; a replay given no resolver, or a resolver given no place, reads no file; a path on
a network share is never opened.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from core.workspace import UserScope, resolve_within
from manage.paths import PathResolver, normalize_path, read_version
from manage.reproduction_manager import apply_steps, run_trace


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path / "data"))
    return tmp_path


def _write(path: Path, text: str = "who,n\nanyone,1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _same(found: str | None, expected: Path) -> bool:
    return found is not None and os.path.realpath(found) == os.path.realpath(expected)


def test_a_users_file_is_found_in_their_workspace_only(root: Path) -> None:
    alice, bob = UserScope.for_user("alice"), UserScope.for_user("bob")
    mine = _write(alice.uploads_dir() / "cohort.csv")
    theirs = _write(bob.uploads_dir() / "private.csv")
    paths = PathResolver(scope=alice)
    # recorded relative to the workspace (AppState)
    assert _same(paths.resolve("uploads/cohort.csv"), mine)
    assert _same(paths.resolve(str(mine)), mine)
    # recorded on another machine: found by its name in the user's folders
    assert _same(paths.resolve(r"C:\Users\someone\Documents\cohort.csv"), mine)
    # another user's file, by its full path or through the parent folder: never read
    assert paths.resolve(str(theirs)) is None
    assert paths.resolve("../bob/uploads/private.csv") is None
    assert paths.resolve("uploads") is None  # a folder is no file
    # a value of another type in the trace's JSON reads nothing
    for recorded in ("", None, ["uploads/cohort.csv"], 3):
        assert paths.resolve(recorded) is None, recorded


def test_a_path_outside_the_workspace_finds_the_users_file_of_that_name(root: Path) -> None:
    alice, bob = UserScope.for_user("alice"), UserScope.for_user("bob")
    theirs = _write(bob.uploads_dir() / "cohort.csv")
    mine = _write(alice.datasets_dir() / "cohort.csv")
    assert _same(PathResolver(scope=alice).resolve(str(theirs)), mine)


def test_a_file_the_user_chose_comes_first_and_stays_in_the_workspace(root: Path) -> None:
    alice, bob = UserScope.for_user("alice"), UserScope.for_user("bob")
    chosen = _write(alice.uploads_dir() / "renamed.csv")
    by_name = _write(alice.uploads_dir() / "cohort.csv")
    theirs = _write(bob.uploads_dir() / "cohort.csv")
    paths = PathResolver(scope=alice)
    recorded = "D:/study/cohort.csv"
    assert _same(paths.with_named({recorded: str(chosen)}).resolve(recorded), chosen)
    # a choice outside the workspace, or one that no longer exists: the recorded name is looked for instead
    assert _same(paths.with_named({recorded: str(theirs)}).resolve(recorded), by_name)
    gone = paths.with_named({recorded: str(alice.uploads_dir() / "gone.csv")})
    assert _same(gone.resolve(recorded), by_name)
    assert paths.with_named({recorded: str(chosen)}).named == {recorded: str(chosen)}
    assert paths.named == {}  # with_named gives another resolver


def test_a_resolver_given_no_place_reads_no_file(root: Path) -> None:
    shared = _write(Path(dp.get_uploads_dir()) / "cohort.csv")
    pd.DataFrame({"x": [1]}).to_parquet(Path(dp.get_datasets_dir()) / "v1.parquet")
    nothing = PathResolver()  # a session whose user has no workspace gets this one
    assert nothing.resolve(str(shared)) is None
    assert nothing.resolve("uploads/cohort.csv") is None
    assert nothing.with_named({"x.csv": str(shared)}).resolve("x.csv") is None
    assert nothing.dataset("v1") is None


def test_tests_and_scripts_ask_for_the_shared_folders_of_the_data_root(root: Path) -> None:
    elsewhere = _write(root / "elsewhere" / "kept.csv")
    shared = _write(Path(dp.get_uploads_dir()) / "test_data.csv")
    pd.DataFrame({"x": [1]}).to_parquet(Path(dp.get_datasets_dir()) / "v1.parquet")
    paths = PathResolver.unconfined()
    # an existing path, as recorded; a path recorded on Windows, by its name
    assert paths.resolve(str(elsewhere)) == str(elsewhere)
    assert _same(paths.resolve(r"C:\Users\User\Documents\test_data.csv"), shared)
    assert _same(paths.resolve("uploads/test_data.csv"), shared)  # relative to the data folder
    assert paths.resolve("non_existent_file.xyz") is None
    assert _same(paths.dataset("v1"), Path(dp.get_datasets_dir()) / "v1.parquet")


def test_a_saved_version_is_read_from_the_datasets_folder_only(root: Path) -> None:
    alice, bob = UserScope.for_user("alice"), UserScope.for_user("bob")
    frame = pd.DataFrame({"x": [1, 2]})
    frame.to_parquet(alice.datasets_dir() / "v1.parquet")
    frame.to_parquet(bob.datasets_dir() / "secret.parquet")
    paths = PathResolver(scope=alice)
    found = paths.dataset("v1")
    assert _same(found, alice.datasets_dir() / "v1.parquet")
    assert found is not None
    pd.testing.assert_frame_equal(read_version(found), frame)
    # "../datasets/v1" stays in the folder: only a name is taken, whatever the path would reach
    escapes = (
        "../../bob/datasets/secret",
        r"..\..\bob\datasets\secret",
        "sub/v1",
        "../datasets/v1",
    )
    for name in ("secret", *escapes, "..", ".", "", None, ["v1"]):
        assert paths.dataset(name) is None, name
    assert read_version(str(alice.datasets_dir() / "missing.parquet")) is None


def test_a_command_line_reads_only_the_files_named_on_it(root: Path) -> None:
    folder = root / "cli"
    named = _write(folder / "cohort.csv")
    unnamed = _write(folder / "other.csv")
    _write(Path(dp.get_uploads_dir()) / "other.csv")
    pd.DataFrame({"x": [1]}).to_parquet(folder / "v1.parquet")
    paths = PathResolver.for_files([named, folder / "v1.parquet"])
    assert _same(paths.resolve(r"C:\anywhere\cohort.csv"), named)
    assert paths.resolve(str(unnamed)) is None  # it exists, but the command line does not name it
    assert paths.resolve("uploads/other.csv") is None
    assert _same(paths.dataset("v1"), folder / "v1.parquet")
    assert PathResolver.for_files([]).resolve(str(named)) is None
    # a chosen file is read only when the command line names it
    assert paths.with_named({"x/other.csv": str(unnamed)}).resolve("x/other.csv") is None
    assert _same(paths.with_named({"x/renamed.csv": str(named)}).resolve("x/renamed.csv"), named)


def test_a_replay_given_no_resolver_reads_no_file(root: Path) -> None:
    source = _write(Path(dp.get_uploads_dir()) / "cohort.csv")
    pd.DataFrame({"x": [1]}).to_parquet(Path(dp.get_datasets_dir()) / "v1.parquet")
    for recorded in (str(source), "v1"):
        trace = {"source_dataset": recorded, "steps": []}
        assert run_trace(trace).error == f"Source file not found: {recorded}"
        assert run_trace(trace, paths=PathResolver()).error == f"Source file not found: {recorded}"
        result = run_trace(trace, paths=PathResolver.unconfined())
        assert result.error is None and result.df is not None and len(result.df) == 1, recorded


def test_the_steps_of_a_replay_read_the_users_files_only(root: Path) -> None:
    alice, bob = UserScope.for_user("alice"), UserScope.for_user("bob")
    theirs = _write(bob.uploads_dir() / "extra.csv", "id,score\n1,5\n")
    pd.DataFrame({"id": [1], "who": ["bob"]}).to_parquet(bob.snapshots_dir() / "theirs.parquet")
    pd.DataFrame({"id": [1], "who": ["alice"]}).to_parquet(alice.snapshots_dir() / "mine.parquet")
    frame = pd.DataFrame({"id": [1]})
    paths = PathResolver(scope=alice)
    merge = {
        "function": "enrichment",
        "params": {"enrichment_file_path": str(theirs), "identifier": "id"},
        "description": "Merge",
    }
    assert apply_steps(frame, [merge], paths=paths).missing_file == (0, str(theirs))

    def snapshot_of(path: str) -> dict:  # a change that cannot be recomputed: its snapshot is read
        return {
            "function": "manual_edit",
            "replayable": False,
            "output_dataset_path": path,
            "params": {},
        }

    result = apply_steps(
        frame, [snapshot_of(str(bob.snapshots_dir() / "theirs.parquet"))], paths=paths
    )
    assert result.df is not None and "who" not in result.df.columns
    assert any("has no snapshot" in warning for warning in result.warnings), result.warnings
    result = apply_steps(frame, [snapshot_of("traces/snapshots/mine.parquet")], paths=paths)
    assert result.df is not None and result.df["who"].tolist() == ["alice"]


@pytest.mark.skipif(os.name != "nt", reason="a junction is a folder link of Windows")
def test_a_folder_link_out_of_the_workspace_is_not_followed(root: Path) -> None:
    import _winapi  # ty: ignore[unresolved-import]

    alice, bob = UserScope.for_user("alice"), UserScope.for_user("bob")
    _write(bob.uploads_dir() / "private.csv")
    link = alice.uploads_dir() / "link"
    _winapi.CreateJunction(str(bob.uploads_dir()), str(link))
    try:
        assert PathResolver(scope=alice).resolve("uploads/link/private.csv") is None
    finally:
        os.rmdir(link)  # the link only


@pytest.mark.skipif(
    os.name != "nt", reason="a network path is a path of Windows (on Linux, // names the root)"
)
def test_a_network_path_is_refused_before_it_is_opened(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alice = UserScope.for_user("alice")
    mine = _write(alice.uploads_dir() / "x.csv")
    opened: list[str] = []
    for name in ("realpath", "isfile", "exists"):

        def spy(path, *args, _real=getattr(os.path, name), **kwargs):
            opened.append(str(path))
            return _real(path, *args, **kwargs)

        monkeypatch.setattr(os.path, name, spy)
    for recorded in (
        r"\\attacker\share\x.csv",
        "//attacker/share/x.csv",
        r"\\?\UNC\attacker\share\x.csv",
    ):
        # its name, in alice's folders
        assert _same(PathResolver(scope=alice).resolve(recorded), mine)
        assert PathResolver.unconfined().resolve(recorded) is None
        assert PathResolver.unconfined().with_named({"y.csv": recorded}).resolve("y.csv") is None
        with pytest.raises(ValueError):
            resolve_within(alice.root, recorded)
    assert not [path for path in opened if "attacker" in path.lower()]


def test_a_path_recorded_on_windows_keeps_its_name_elsewhere() -> None:
    recorded = r"C:\data\users\alice\uploads\cohort.csv"
    assert os.path.basename(normalize_path(recorded)) == "cohort.csv"
    assert normalize_path("") == ""


def _replay_in_a_session() -> None:
    import streamlit as st

    from app_pages.reproduce_analysis import _resolver
    from manage.reproduction_manager import run_trace

    st.session_state["username"] = st.session_state.get("as_user", "alice")
    for key, source in st.session_state["sources"].items():
        result = run_trace({"source_dataset": source, "steps": []}, paths=_resolver())
        rows = None if result.df is None else result.df["who"].tolist()
        st.session_state[key] = (result.error, rows)


def test_a_replay_in_a_session_finds_the_users_file_and_never_anothers(root: Path) -> None:
    alice, bob = UserScope.for_user("alice"), UserScope.for_user("bob")
    mine = _write(alice.uploads_dir() / "cohort.csv", "who,n\nalice,1\n")
    theirs = _write(bob.uploads_dir() / "cohort.csv", "who,n\nbob,1\n")
    only_theirs = _write(bob.uploads_dir() / "private.csv", "who,n\nbob,1\n")
    at = AppTest.from_function(_replay_in_a_session, default_timeout=60)
    at.session_state["sources"] = {
        "own": str(mine),
        "same_name": str(theirs),
        "theirs": str(only_theirs),
        "through_parent": "../bob/uploads/private.csv",
    }
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert at.session_state["own"] == (None, ["alice"])
    # bob's path: alice's file of that name
    assert at.session_state["same_name"] == (None, ["alice"])
    for key in ("theirs", "through_parent"):
        error, rows = at.session_state[key]
        assert rows is None and error.startswith("Source file not found"), key


def test_a_session_whose_user_has_no_workspace_reads_no_file(root: Path) -> None:
    shared = _write(Path(dp.get_uploads_dir()) / "cohort.csv", "who,n\nshared,1\n")
    at = AppTest.from_function(_replay_in_a_session, default_timeout=60)
    at.session_state["as_user"] = "Мария"  # no letter or digit a folder name keeps
    at.session_state["sources"] = {"shared": str(shared), "relative": "uploads/cohort.csv"}
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    for key in ("shared", "relative"):
        error, rows = at.session_state[key]
        assert rows is None and error.startswith("Source file not found"), key
