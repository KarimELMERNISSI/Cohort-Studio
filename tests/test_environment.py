"""
The software recorded with a dataset and printed in the dossier (utils.environment, M8): the full commit of a checkout
or of the image's build, a fingerprint of CohortStudio's own code that names the code that ran (a checkout with local
changes runs code no commit names), and the lock file's checksum.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from utils.environment import (
    ROOT,
    code_fingerprint,
    environment,
    environment_text,
    git_commit,
    lock_checksum,
)

SHA = "0e88a9276843deae38324f8fa20b3d26d4d4b505"


def _git(root: Path, head: str, refs: dict[str, str] | None = None, packed: str = "") -> Path:
    git = root / ".git"
    git.mkdir(parents=True)
    (git / "HEAD").write_text(head + "\n", encoding="utf-8")
    for ref, value in (refs or {}).items():
        (git / ref).parent.mkdir(parents=True, exist_ok=True)
        (git / ref).write_text(value + "\n", encoding="utf-8")
    if packed:
        (git / "packed-refs").write_text(packed, encoding="utf-8")
    return root


def test_the_full_commit_is_read_from_a_checkout_or_from_the_images_build(tmp_path: Path) -> None:
    assert git_commit(_git(tmp_path / "a", "ref: refs/heads/main", {"refs/heads/main": SHA})) == SHA
    assert git_commit(_git(tmp_path / "b", SHA.upper())) == SHA  # a detached head
    packed = f"# pack-refs with: peeled\n{SHA} refs/heads/main\n"
    assert git_commit(_git(tmp_path / "c", "ref: refs/heads/main", packed=packed)) == SHA
    image = tmp_path / "image"  # no .git: the file its build writes
    image.mkdir()
    (image / "BUILD_COMMIT").write_text(f"{SHA}\n", encoding="utf-8")
    assert git_commit(image) == SHA
    for written in ("", "abc1234", "not a commit", SHA + "0"):
        (image / "BUILD_COMMIT").write_text(written, encoding="utf-8")
        assert git_commit(image) is None, written
    assert git_commit(tmp_path / "nowhere") is None


def test_a_worktree_s_commit_is_read_through_its_git_file(tmp_path: Path) -> None:
    """A worktree's .git is a file naming its own folder, which names the common one holding the refs."""
    common = _git(tmp_path / "main", "ref: refs/heads/main", {"refs/heads/main": "1" * 40}) / ".git"
    for name, head in (("detached", SHA), ("branch", "ref: refs/heads/main")):
        own = common / "worktrees" / name
        own.mkdir(parents=True)
        (own / "HEAD").write_text(head, encoding="utf-8")  # the reader strips the line's end
        (own / "commondir").write_text("../..", encoding="utf-8")
        checkout = tmp_path / name
        checkout.mkdir()
        (checkout / ".git").write_text(f"gitdir: {own}", encoding="utf-8")
    assert git_commit(tmp_path / "detached") == SHA
    assert git_commit(tmp_path / "branch") == "1" * 40


def _sources(root: Path, files: dict[str, bytes]) -> Path:
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    return root


def test_the_code_fingerprint_names_the_code_whatever_its_line_endings(tmp_path: Path) -> None:
    unix = _sources(
        tmp_path / "unix",
        {"core/a.py": b"x = 1\ny = 2\n", "ui/b.py": b"z = 3\n", "main.py": b"run()\n"},
    )
    windows = _sources(
        tmp_path / "windows",
        {"main.py": b"run()\r\n", "ui/b.py": b"z = 3\r\n", "core/a.py": b"x = 1\r\ny = 2\r\n"},
    )
    assert code_fingerprint(unix) == code_fingerprint(windows)
    assert re.fullmatch(r"[0-9a-f]{64}", code_fingerprint(unix) or "")
    changed = _sources(
        tmp_path / "changed",
        {"core/a.py": b"x = 1\ny = 20\n", "ui/b.py": b"z = 3\n", "main.py": b"run()\n"},
    )
    moved = _sources(
        tmp_path / "moved",
        {"core/b.py": b"x = 1\ny = 2\n", "ui/b.py": b"z = 3\n", "main.py": b"run()\n"},
    )
    outside = (
        _sources(  # neither a package nor an entry point, nor a cache: not CohortStudio's code
            tmp_path / "outside",
            {
                "core/a.py": b"x = 1\ny = 2\n",
                "ui/b.py": b"z = 3\n",
                "main.py": b"run()\n",
                "tests/t.py": b"assert True\n",
                "core/__pycache__/a.py": b"cached\n",
            },
        )
    )
    assert code_fingerprint(changed) != code_fingerprint(unix) != code_fingerprint(moved)
    assert code_fingerprint(outside) == code_fingerprint(unix)
    assert code_fingerprint(tmp_path / "empty") is None


def test_the_lock_checksum_ignores_line_endings(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "uv.lock").write_bytes(b"version = 1\n")
    (tmp_path / "b" / "uv.lock").write_bytes(b"version = 1\r\n")
    unix = lock_checksum(tmp_path / "a")
    assert unix is not None and unix == lock_checksum(tmp_path / "b")
    assert lock_checksum(tmp_path) is None


def test_this_installation_records_its_code_and_says_it_in_a_line() -> None:
    found = environment()
    assert re.fullmatch(r"[0-9a-f]{64}", found["code"])
    if (ROOT / ".git").exists():
        assert re.fullmatch(r"[0-9a-f]{40}", found["commit"])
        assert f"commit {found['commit'][:7]}, code {found['code'][:8]}" in environment_text()
    if (ROOT / "uv.lock").exists():
        assert re.fullmatch(r"[0-9a-f]{64}", found["lock"])


@pytest.mark.parametrize("root", [ROOT])
def test_the_fingerprint_is_computed_once_per_process(root: Path) -> None:
    assert code_fingerprint(root) is code_fingerprint(root)
