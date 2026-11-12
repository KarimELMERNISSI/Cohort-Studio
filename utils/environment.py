"""
The software a dataset was loaded and analysed with: CohortStudio's version, its full commit and a fingerprint of its
code, the lock file's checksum, Python and the key libraries. It is recorded in the trace when a dataset is loaded (the
``initial_load`` step) and printed in the study dossier, so that a reader knows which versions produced the results.

A commit names the code only when the code is the commit's: a source checkout with local changes, or an image built
without one, runs code no commit names. The *code fingerprint* names the code that ran, whatever its origin: a SHA-256
over CohortStudio's own Python sources, read with LF line endings (a Windows checkout may hold CRLF), which a
verification recomputes (M8).
"""

from __future__ import annotations

import hashlib
import platform
import re
import tomllib
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRARIES = ("numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "lifelines", "streamlit")
# CohortStudio's own code: its packages (pyproject's wheel) and its two entry points
PACKAGES = (
    "app_pages",
    "biostats",
    "core",
    "enrich",
    "explore",
    "graph",
    "ingest",
    "manage",
    "monitor",
    "prompts",
    "ui",
    "utils",
)
ENTRY_POINTS = ("main.py", "manage_users.py")
BUILD_COMMIT = "BUILD_COMMIT"  # written into a Docker image by its build (the image holds no .git)
_COMMIT = re.compile(
    r"[0-9a-f]{40}(?:[0-9a-f]{24})?"
)  # SHA-1, or SHA-256 for a repository that uses it


def app_version() -> str:
    """CohortStudio's version: the installed package, else pyproject.toml (a source checkout)."""
    try:
        return version("cohortstudio")
    except PackageNotFoundError:
        pass
    try:
        return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
            "version"
        ]
    except OSError, KeyError, tomllib.TOMLDecodeError:
        return "unknown"


def _commit(text: str) -> str | None:
    text = text.strip().lower()
    return text if _COMMIT.fullmatch(text) else None


def _git_dirs(root: Path) -> tuple[Path, Path]:
    """The checkout's git folder and the folder its refs are kept in: the same, except in a worktree, whose ``.git``
    is a file naming its own folder ("gitdir: ..."), which names the common one ("commondir")."""
    git = root / ".git"
    if git.is_file():
        pointer = git.read_text(encoding="utf-8").strip()
        if pointer.startswith("gitdir: "):
            git = (root / pointer.removeprefix("gitdir: ").strip()).resolve()
    common = git
    if (git / "commondir").is_file():
        common = (git / (git / "commondir").read_text(encoding="utf-8").strip()).resolve()
    return git, common


def git_commit(root: Path = ROOT) -> str | None:
    """The full commit of a source checkout (read from .git, without running git; a worktree's too), else the one an
    image was built from (its ``BUILD_COMMIT`` file), else None."""
    try:
        git, common = _git_dirs(root)
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return _commit(head)
        ref = head.removeprefix("ref: ")
        for folder in dict.fromkeys((git, common)):
            loose = folder / ref
            if loose.is_file():
                return _commit(loose.read_text(encoding="utf-8"))
        for line in (common / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line.endswith(f" {ref}"):
                return _commit(line.split(" ", 1)[0])
    except OSError:
        pass  # no git folder to read (an image, an archive): the commit recorded at build time is next
    try:
        return _commit((root / BUILD_COMMIT).read_text(encoding="utf-8"))
    except OSError:
        return None


def _normalised(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


@cache
def code_fingerprint(root: Path = ROOT) -> str | None:
    """
    SHA-256 over CohortStudio's own Python sources (its packages and entry points): each file named by its path from
    ``root`` with forward slashes, its length and its bytes with LF line endings, in the order of the paths. None when
    none is found. Computed once per process (about half a second).
    """
    files = sorted(
        {
            path
            for package in PACKAGES
            for path in (root / package).rglob("*.py")
            if "__pycache__" not in path.parts
        }
        | {root / name for name in ENTRY_POINTS if (root / name).is_file()}
    )
    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        data = _normalised(path)
        digest.update(f"{path.relative_to(root).as_posix()}\0{len(data)}\0".encode())
        digest.update(data)
    return digest.hexdigest()


def lock_checksum(root: Path = ROOT) -> str | None:
    """SHA-256 of ``uv.lock``, the exact versions the installation was resolved to, with LF line endings; None
    without it (an installed package)."""
    try:
        return hashlib.sha256(_normalised(root / "uv.lock")).hexdigest()
    except OSError:
        return None


def environment() -> dict[str, str]:
    """CohortStudio, its commit and code fingerprint and lock checksum (when known), Python and the versions of the key
    libraries that are installed."""
    found = {"cohortstudio": app_version()}
    for name, value in (
        ("commit", git_commit()),
        ("code", code_fingerprint()),
        ("lock", lock_checksum()),
    ):
        if value:
            found[name] = value
    found["python"] = platform.python_version()
    for name in LIBRARIES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


def environment_text() -> str:
    """The environment in one line, for reports: "CohortStudio 2.4.1 (commit abc1234, code 9f3c2a1b), Python 3.14.3,
    numpy 2.4, ..."."""
    env = environment()
    names = [f"commit {env['commit'][:7]}"] if "commit" in env else []
    names += [f"code {env['code'][:8]}"] if "code" in env else []
    head = f"CohortStudio {env['cohortstudio']}" + (f" ({', '.join(names)})" if names else "")
    rest = [f"Python {env['python']}"] + [
        f"{name} {env[name]}" for name in LIBRARIES if name in env
    ]
    return ", ".join([head, *rest])


__all__ = [
    "BUILD_COMMIT",
    "LIBRARIES",
    "app_version",
    "code_fingerprint",
    "environment",
    "environment_text",
    "git_commit",
    "lock_checksum",
]
