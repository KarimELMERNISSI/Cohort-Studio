"""Per-user workspace: directory isolation and path containment (audit E4)."""

from pathlib import Path

import pytest

import utils.data_paths as dp
from core.workspace import (
    UserScope,
    all_user_scopes,
    remove_pdf,
    resolve_within,
    safe_dirname,
    safe_filename,
    validate_username,
)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    return tmp_path


def test_username_validation() -> None:
    assert validate_username(" dr.lee_2 ") == "dr.lee_2"
    for bad in ("ab", "../x", "name with space", "", None, "x" * 40):
        with pytest.raises(ValueError):
            validate_username(bad)


def test_safe_dirname_handles_legacy_names_but_never_escapes() -> None:
    assert safe_dirname("admin") == "admin"
    assert safe_dirname("Dr Lee/ops") == "Dr_Lee_ops"
    assert safe_dirname("../../etc") == "etc"
    for bad in ("", "..", "./", None):
        with pytest.raises(ValueError):
            safe_dirname(bad)


def test_safe_filename() -> None:
    assert safe_filename("../../x.parquet") == "x.parquet"
    assert safe_filename("my file (v2).csv") == "my file (v2).csv"
    assert safe_filename("...") == "file"
    assert safe_filename(None, default="dataset") == "dataset"


def test_scope_directories_are_per_user(root: Path) -> None:
    alice = UserScope.for_user("alice")
    bob = UserScope.for_user("bob", role="admin")
    assert alice.datasets_dir() == root / "users" / "alice" / "datasets"
    assert bob.datasets_dir() == root / "users" / "bob" / "datasets"
    assert alice.artifacts_dir("session_1").parent == alice.traces_dir() / "artifacts"
    assert alice.snapshots_dir().exists() and alice.vectors_dir().exists()
    assert bob.is_admin and not alice.is_admin
    assert {s.username for s in all_user_scopes()} == {"alice", "bob"}


def test_resolve_within_blocks_escapes(root: Path) -> None:
    alice = UserScope.for_user("alice")
    inside = alice.uploads_dir() / "data.csv"
    inside.write_text("x")
    assert alice.resolve_within("data.csv", "uploads") == inside.resolve()
    assert alice.resolve_within(str(inside)) == inside.resolve()
    with pytest.raises(ValueError):
        alice.resolve_within("../bob/datasets/secret.parquet", "uploads")
    with pytest.raises(ValueError):
        alice.resolve_within(str(root / "users" / "bob" / "datasets" / "secret.parquet"))
    with pytest.raises(ValueError):
        resolve_within(
            alice.root, "C:/Windows/system32" if str(root).startswith("C:") else "/etc/passwd"
        )
    with pytest.raises(ValueError):
        alice.dir("nope")


def test_a_sibling_directory_sharing_the_prefix_is_outside(tmp_path: Path) -> None:
    # a prefix check must stop at a separator: alice2 is not inside alice
    (tmp_path / "alice").mkdir()
    (tmp_path / "alice2").mkdir()
    with pytest.raises(ValueError):
        resolve_within(tmp_path / "alice", tmp_path / "alice2" / "secret.csv")
    with pytest.raises(ValueError):
        resolve_within(tmp_path / "alice", "../alice2/secret.csv")
    assert resolve_within(tmp_path / "alice", ".") == (tmp_path / "alice").resolve()
    assert (
        resolve_within(tmp_path / "alice", "a/../b.csv") == (tmp_path / "alice" / "b.csv").resolve()
    )


def test_only_a_pdf_of_the_library_can_be_removed(root: Path) -> None:
    library = UserScope.for_user("alice").documents_dir()
    (library / "paper.pdf").write_bytes(b"%PDF-1.4")
    (library / "notes.txt").write_text("x")
    (root / "outside.pdf").write_bytes(b"%PDF-1.4")
    assert remove_pdf(library, "paper.pdf") and not (library / "paper.pdf").exists()
    assert not remove_pdf(library, "paper.pdf")  # already gone
    assert not remove_pdf(library, "notes.txt") and (library / "notes.txt").exists()  # not a PDF
    with pytest.raises(ValueError):
        remove_pdf(library, "../../../outside.pdf")
    assert (root / "outside.pdf").exists()
