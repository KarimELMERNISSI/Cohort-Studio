"""Per-user workspace: directory isolation and path containment (audit E4)."""

import unicodedata
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
    for bad in ("", "  ", None):  # no user
        with pytest.raises(ValueError):
            safe_dirname(bad)


def test_a_name_that_keeps_nothing_as_a_folder_name_gets_a_folder_of_its_own() -> None:
    """Such a user had no workspace, and used the shared folders as every such user did: each now gets a folder named
    after the name's SHA-256, starting with "_", which no other name's folder starts with."""
    # 67 characters
    long_mail = "firstname.middlename.lastname.department@hospital-group.example.org"
    names = ("Мария", "Мариам", "..", "./", "_", long_mail, long_mail.upper())
    folders = [safe_dirname(name) for name in names]
    assert len(set(folders)) == len(names)  # one folder per name
    for folder in folders:
        assert len(folder) == 33 and folder[0] == "_" and int(folder[1:], 16) >= 0
    assert safe_dirname(" Мария ") == folders[0]  # the same name, the same folder
    # an ordinary name's folder never starts with "_": it can never be one of these
    for ordinary in ("_x", "__init__", "-admin", ".hidden", "a" * 64):
        assert not safe_dirname(ordinary).startswith("_"), ordinary
    assert UserScope.for_user("Мария").username == folders[0]


def test_a_hashed_name_is_never_normalised_into_another_names_folder() -> None:
    """Among the names that keep nothing as a folder name, two spellings an identity provider may tell apart keep two
    folders: at worst one person's files are split, never two people's merged. (An ordinary name loses the characters
    a folder name cannot keep, so "José" and "Josè" share "Jos": local sign-up refuses the second.)"""
    composed = "Ёлка"  # Ё as one character, or as Е and a combining diaeresis
    assert safe_dirname(unicodedata.normalize("NFD", composed)) != safe_dirname(composed)
    assert safe_dirname("ａｌｉｃｅ") != safe_dirname("alice")  # full-width letters
    assert safe_dirname("МАРИЯ") != safe_dirname("Мария")


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
