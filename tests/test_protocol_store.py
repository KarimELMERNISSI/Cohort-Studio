"""
The study definitions in the workspace (manage.protocol_store, M6): the working copy and the revisions, written in one
step and read back only as written; another session's copy kept, a revision never rewritten nor its number given
twice; a file that cannot be read gives none and is logged without its content; one user's studies out of another's
reach.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

import utils.data_paths as dp
from core.protocol import ProtocolMeta, StudyProtocol
from core.workspace import UserScope
from manage.protocol_store import HISTORY, MAX_BYTES, ProtocolUnchanged, WorkspaceProtocolStore

PLANTED = "P-0042 Jane Doe"


def _protocol(
    expression: str = "df['age'] > 110", study_id: str = "cohort", **meta
) -> StudyProtocol:
    rule = {"expression": expression, "description": "Ages out of range."}
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
    assert mine.kept is not None
    kept = json.loads((store.root / "cohort" / "history" / mine.kept).read_text(encoding="utf-8"))
    assert StudyProtocol.from_file(kept)[0].hash == _protocol("df['age'] > 120").hash  # type: ignore[union-attr]
    now = store.working("cohort")
    assert now is not None and now.protocol.hash == _protocol("df['age'] > 130").hash


def test_the_history_keeps_the_last_copies(store: WorkspaceProtocolStore) -> None:
    for n in range(HISTORY + 3):
        store.save_working(_protocol(f"df['age'] > {n}"), expected="not the current one")
    kept = sorted((store.root / "cohort" / "history").iterdir())
    assert len(kept) == HISTORY
    assert "df['age'] > 11" in kept[-1].read_text(encoding="utf-8")  # the latest replaced


def test_revisions_are_numbered_and_chained_by_fingerprint(store: WorkspaceProtocolStore) -> None:
    saved = [store.save_revision(_protocol(f"df['age'] > {n}")) for n in (110, 120, 130)]
    assert [r.meta.revision for r in saved] == [1, 2, 3]
    assert [r.meta.parent for r in saved] == [None, saved[0].hash, saved[1].hash]
    assert [r.meta.parent_revision for r in saved] == [None, 1, 2]
    listed = store.revisions("cohort")
    assert [(r.number, r.hash, r.parent) for r in listed] == [
        (r.meta.revision, r.hash, r.meta.parent) for r in saved
    ]
    second = store.revision("cohort", 2)
    assert second is not None and second.hash == saved[1].hash
    assert store.revision("cohort", 9) is None


def test_an_unchanged_body_is_no_revision(store: WorkspaceProtocolStore) -> None:
    store.save_revision(_protocol())
    with pytest.raises(ProtocolUnchanged, match="unchanged since revision 1"):
        store.save_revision(_protocol())
    reworded = _protocol()
    body = json.loads(json.dumps(reworded.to_file()["body"]))
    body["mask_families"]["clinical_anomalies"]["Implausible age"]["description"] = "Reworded."
    with pytest.raises(ProtocolUnchanged):  # a label is not a decision
        store.save_revision(StudyProtocol(body, reworded.meta))


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
    assert third.meta.revision == 3 and third.meta.parent == first.hash != second.hash
    assert [r.number for r in store.revisions("cohort")] == [1, 3]
    assert "not a JSON file" in caplog.text


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (f"not json: {PLANTED}", "not a JSON file"),
        ("edited", "changed outside CohortStudio"),
        ("another study", "names another study"),
        ("a revision", "holds a revision"),
        ("oversized", "larger than 5 MB"),
    ],
)
def test_a_file_that_cannot_be_read_gives_none_and_its_content_is_not_logged(
    store: WorkspaceProtocolStore, caplog: pytest.LogCaptureFixture, content: str, reason: str
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
    path.write_text(text, encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="manage.protocol_store"):
        assert store.working("cohort") is None
    assert "cohort" in caplog.text and reason in caplog.text
    assert PLANTED not in caplog.text


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
