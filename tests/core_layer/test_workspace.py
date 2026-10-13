"""Per-user workspace: directory isolation and path containment (audit E4)."""

from pathlib import Path

import pytest

import utils.data_paths as dp
from core.workspace import (
    UserScope,
    all_user_scopes,
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
