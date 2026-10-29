"""
The study definitions in the workspace (manage.protocol_store, M6): the working copy and the revisions, written in one
step and read back only as written; another session's copy kept, for a while and until discarded; a revision never
rewritten nor its number given twice; a file that cannot be read gives none and is logged without its content; a file
that cannot be written names no path; one user's studies out of another's reach.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pytest

import utils.data_paths as dp
from core.protocol import ProtocolMeta, StudyProtocol
from core.workspace import UserScope
from manage.protocol_store import (
    HISTORY,
    HISTORY_DAYS,
    MAX_BYTES,
    ProtocolFileError,
    ProtocolUnchanged,
    WorkspaceProtocolStore,
)

PLANTED = "P-0042 Jane Doe"


def _protocol(
    expression: str = "df['age'] > 110",
    study_id: str = "cohort",
    description: str = "Ages out of range.",
    **meta,
) -> StudyProtocol:
    rule = {"expression": expression, "description": description}
    body = {"mask_families": {"clinical_anomalies": {"operator": "any", "Implausible age": rule}}}
    return StudyProtocol(body, ProtocolMeta(study_id, **meta))


@pytest.fixture
def store(tmp_path: Path) -> WorkspaceProtocolStore:
    return WorkspaceProtocolStore(None, root=tmp_path / "protocols")


def test_the_working_copy_is_saved_and_read_back(store: WorkspaceProtocolStore) -> None:
    protocol = _protocol(title="Cohort")
    saved = store.save_working(protocol, expected=None)
    assert saved.kept is None
    stored = store.working("cohort")
    assert stored is not None and stored.etag == saved.etag
    assert stored.protocol.hash == protocol.hash and stored.protocol.meta.title == "Cohort"
    assert stored.protocol.meta.revision is None
    assert stored.protocol.meta.saved_at and stored.protocol.meta.app_version
    assert store.working("other") is None


def test_another_sessions_copy_is_kept_in_the_history(store: WorkspaceProtocolStore) -> None:
    first = store.save_working(_protocol("df['age'] > 110"), expected=None)  # this session
    theirs = store.working("cohort")
    assert theirs is not None
    store.save_working(_protocol("df['age'] > 120"), expected=theirs.etag)  # another session
    mine = store.save_working(_protocol("df['age'] > 130"), expected=first.etag)
    assert mine.kept is not None and store.history("cohort") == [mine.kept]
    kept = json.loads((store.root / "cohort" / "history" / mine.kept).read_text(encoding="utf-8"))
    read, _ = StudyProtocol.from_file(kept)
    assert read is not None and read.hash == _protocol("df['age'] > 120").hash
    now = store.working("cohort")
    assert now is not None and now.protocol.hash == _protocol("df['age'] > 130").hash


def test_a_copy_that_cannot_be_read_is_replaced_not_kept(store: WorkspaceProtocolStore) -> None:
    path = store.root / "cohort" / "working.json"
    path.parent.mkdir(parents=True)
    path.write_text(f"not json: {PLANTED}", encoding="utf-8")
    saved = store.save_working(_protocol(), expected=None)
    assert saved.kept is None and store.history("cohort") == []


def test_the_history_keeps_the_last_copies_for_a_while(store: WorkspaceProtocolStore) -> None:
    for n in range(HISTORY + 3):
        store.save_working(_protocol(f"df['age'] > {n}"), expected="not the current one")
    kept = store.history("cohort")
    assert len(kept) == HISTORY
    folder = store.root / "cohort" / "history"
    assert "df['age'] > 11" in (folder / kept[-1]).read_text(
        encoding="utf-8"
    )  # the latest replaced
    # older than HISTORY_DAYS: gone at the next save, whoever saves
    old = time.time() - (HISTORY_DAYS + 1) * 86400
    os.utime(folder / kept[0], (old, old))
    store.save_working(_protocol(), expected=store.working("cohort").etag)  # type: ignore[union-attr]
    assert kept[0] not in store.history("cohort") and len(store.history("cohort")) == HISTORY - 1


def test_a_corrected_text_can_be_discarded_from_the_history(
    store: WorkspaceProtocolStore,
) -> None:
    """A name typed into a rule, corrected in the working copy, stays in the copy another session replaced until the
    history is discarded."""
    first = store.save_working(_protocol(description=f"Ages of {PLANTED}"), expected=None)
    store.save_working(_protocol(description="Ages out of range."), expected="another session's")
    store.save_working(_protocol(description="Ages out of range."), expected=first.etag)
    folder = store.root / "cohort" / "history"
    assert any(PLANTED in p.read_text(encoding="utf-8") for p in folder.iterdir())
    kept = store.history("cohort")
    assert len(kept) == 2 and store.discard_history("cohort") == 2
    assert store.history("cohort") == []
    assert not any(PLANTED in p.read_text(encoding="utf-8") for p in folder.iterdir())


def test_a_write_that_never_ended_leaves_no_copy_behind(store: WorkspaceProtocolStore) -> None:
    folder = store.root / "cohort"
    folder.mkdir(parents=True)
    stale, fresh = folder / ".working.abc.tmp", folder / ".working.def.tmp"
    for path in (stale, fresh):
        path.write_text(PLANTED, encoding="utf-8")
    old = time.time() - 2 * 3600
    os.utime(stale, (old, old))
    store.save_working(_protocol(), expected=None)
    assert (
        not stale.exists() and fresh.exists()
    )  # a write still running in another session keeps its file


def test_revisions_are_numbered_and_chained_by_checksum(store: WorkspaceProtocolStore) -> None:
    saved = [store.save_revision(_protocol(f"df['age'] > {n}")) for n in (110, 120, 130)]
    assert [r.meta.revision for r in saved] == [1, 2, 3]
    assert [r.meta.parent for r in saved] == [None, saved[0].checksum, saved[1].checksum]
    assert [r.meta.parent_revision for r in saved] == [None, 1, 2]
    listed = store.revisions("cohort")
    assert [(r.number, r.checksum, r.parent) for r in listed] == [
        (r.meta.revision, r.checksum, r.meta.parent) for r in saved
    ]
    second = store.revision("cohort", 2)
    assert second is not None and second.hash == saved[1].hash
    assert store.revision("cohort", 9) is None


def test_an_unchanged_body_is_no_revision(store: WorkspaceProtocolStore) -> None:
    store.save_revision(_protocol())
    with pytest.raises(ProtocolUnchanged, match="unchanged since revision 1"):
        store.save_revision(_protocol())
    with pytest.raises(ProtocolUnchanged):  # a label is not a decision
        store.save_revision(_protocol(description="Reworded."))


def test_a_revision_is_never_rewritten_nor_its_number_given_twice(
    store: WorkspaceProtocolStore, caplog: pytest.LogCaptureFixture
) -> None:
    first = store.save_revision(_protocol("df['age'] > 110"))
    files = store.root / "cohort" / "revisions"
    (path,) = files.glob("r0001-*.json")
    written = path.read_bytes()
    second = store.save_revision(_protocol("df['age'] > 120"))
    assert path.read_bytes() == written
    # a revision that can no longer be read keeps its number; the next one follows the latest readable
    (broken,) = files.glob("r0002-*.json")
    broken.write_text("{", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="manage.protocol_store"):
        third = store.save_revision(_protocol("df['age'] > 130"))
    assert third.meta.revision == 3 and third.meta.parent == first.checksum != second.checksum
    assert [r.number for r in store.revisions("cohort")] == [1, 3]
    assert "not a JSON file" in caplog.text
    # a revision deleted (by hand, or later by the application) leaves its number taken
    for path in files.glob("r0003-*.json"):
        path.unlink()
    assert store.save_revision(_protocol("df['age'] > 140")).meta.revision == 4
    # a number another process took first is passed over
    (files / "r0005.number").touch()
    assert store.save_revision(_protocol("df['age'] > 150")).meta.revision == 6


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (f"not json: {PLANTED}", "not a JSON file"),
        ("edited", "changed outside CohortStudio"),
        ("another study", "names another study"),
        ("a revision", "holds a revision"),
        ("oversized", "larger than 5 MB"),
        # whichever of the JSON parser and the reading reaches its depth limit first
        ("nested too deep", ("not a JSON file", "not a study definition this version can read")),
    ],
)
def test_a_file_that_cannot_be_read_gives_none_and_its_content_is_not_logged(
    store: WorkspaceProtocolStore,
    caplog: pytest.LogCaptureFixture,
    content: str,
    reason: str | tuple[str, ...],
) -> None:
    path = store.root / "cohort" / "working.json"
    path.parent.mkdir(parents=True)
    data = _protocol(f"df['name'] == '{PLANTED}'").to_file()
    if content == "edited":
        data["body"]["mask_families"]["clinical_anomalies"]["Implausible age"]["expression"] = (
            PLANTED
        )
    elif content == "another study":
        data["meta"]["study_id"] = "other"
    elif content == "a revision":
        data["meta"]["revision"] = 1
    text = json.dumps(data) if not content.startswith("not json") else content
    if content == "oversized":
        text = text + " " * MAX_BYTES
    if content == "nested too deep":  # under the parser's own limit, over Python's recursion limit
        text = text.replace('"df', "[" * 990 + '"df', 1).replace(
            PLANTED + "'\"", PLANTED + "'\"" + "]" * 990, 1
        )
    path.write_text(text, encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="manage.protocol_store"):
        assert store.working("cohort") is None
        assert [s.study_id for s in store.studies()] == ["cohort"]  # still listed, without a title
    reasons = (reason,) if isinstance(reason, str) else reason
    assert "cohort" in caplog.text and any(r in caplog.text for r in reasons)
    assert PLANTED not in caplog.text


def test_a_file_that_cannot_be_written_is_told_without_its_path(
    store: WorkspaceProtocolStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def refused(path, text, *, prefix):
        raise PermissionError(
            13, "Access is denied", r"C:\Users\someone\protocols\cohort\working.json"
        )

    monkeypatch.setattr("manage.protocol_store.write_atomic", refused)
    with caplog.at_level(logging.WARNING, logger="manage.protocol_store"):
        with pytest.raises(ProtocolFileError) as raised:
            store.save_working(_protocol(), expected=None)
    assert "PermissionError" in str(raised.value) and "Users" not in str(raised.value)
    assert raised.value.__cause__ is None and raised.value.__suppress_context__
    assert (
        "working.json not saved (PermissionError)" in caplog.text and "someone" not in caplog.text
    )


def test_a_definition_larger_than_the_store_reads_is_not_written(
    store: WorkspaceProtocolStore,
) -> None:
    with pytest.raises(ProtocolFileError, match="larger than 5 MB"):
        store.save_working(_protocol(description="x" * MAX_BYTES), expected=None)
    with pytest.raises(ProtocolFileError, match="larger than 5 MB"):
        store.save_revision(_protocol(description="x" * MAX_BYTES))
    assert not (store.root / "cohort" / "working.json").exists()
    assert store.revisions("cohort") == [] and not list((store.root / "cohort").rglob("*.number"))


@pytest.mark.parametrize("study_id", ["..", "a/b", "a\\b", "CON", "nul", "", "Cohort", "x" * 65])
def test_a_study_id_that_is_not_a_slug_is_refused(
    store: WorkspaceProtocolStore, study_id: str
) -> None:
    with pytest.raises(ValueError, match="not a study id"):
        store.working(study_id)
    with pytest.raises(ValueError, match="not a study id"):
        store.save_working(_protocol(study_id=study_id), expected=None)
    assert not (store.root.exists() and any(store.root.iterdir()))


def test_a_store_needs_a_user_or_a_folder_of_its_own() -> None:
    with pytest.raises(ValueError, match="no user is given"):
        WorkspaceProtocolStore(None)


def test_one_users_studies_are_out_of_anothers_reach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path / "data"))
    alice_scope = UserScope.for_user("alice")
    alice = WorkspaceProtocolStore(alice_scope)
    bob = WorkspaceProtocolStore(UserScope.for_user("bob"))
    alice.save_working(_protocol(title="Alice's cohort"), expected=None)
    alice.save_revision(_protocol())
    assert (alice_scope.protocols_dir() / "cohort" / "working.json").is_file()
    (study,) = alice.studies()
    assert (study.study_id, study.title, study.revision) == ("cohort", "Alice's cohort", 1)
    assert study.updated_at
    assert bob.studies() == [] and bob.working("cohort") is None and bob.revisions("cohort") == []
