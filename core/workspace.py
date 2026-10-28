"""
Per-user storage scope (audit finding E4).

Every user gets a directory tree under ``<DATA_ROOT>/users/<username>/`` and all persistence goes
through a :class:`UserScope`, instead of filename prefixes checked inconsistently across pages.
Paths supplied by users (uploads, manual file paths) are resolved with :meth:`UserScope.resolve_within`
so they can never leave the user's workspace.

This module has no Streamlit dependency; ``ui.session_scope`` derives the current scope from the
session state.
"""

from __future__ import annotations

import hashlib
import ntpath
import os
import re
from dataclasses import dataclass
from pathlib import Path

# a new name starts and ends with a letter or a digit: it is its own workspace folder (core.workspace.safe_dirname
# drops a final ".", "_" or "-", so "drlee." and "drlee" would share one; data-protection review, M5b)
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,30}[A-Za-z0-9]$")
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
KINDS: tuple[str, ...] = (
    "datasets",
    "stats",
    "traces",
    "uploads",
    "taxonomy",
    "knowledge_graphs",
    "documents",
    "vectors",
    "rag_eval_logs",
)


def data_root() -> Path:
    """The application data root (``DATA_ROOT``), resolved through ``utils.data_paths`` so tests can patch it."""
    from utils.data_paths import get_data_root

    return Path(get_data_root()).resolve()


def validate_username(username: str | None) -> str:
    """Strict validation for new accounts; raises ``ValueError`` otherwise."""
    value = (username or "").strip()
    if not USERNAME_PATTERN.match(value):
        raise ValueError(
            "Username must be 3-32 characters: letters, digits, '.', '_' or '-' (no spaces)."
        )
    return value


def safe_dirname(username: str | None) -> str:
    """
    Directory name for a (possibly legacy) username: never empty, never a path component escape. A name that keeps
    nothing as a folder name (no ASCII letter or digit, such as "Мария") or that is longer than 64 characters (a long
    e-mail from an identity provider) gets a folder named after its SHA-256, starting with "_", which no other name's
    folder starts with: such a user had no workspace and used the shared folders, as every such user did.
    """
    name = (username or "").strip()
    if not name:
        raise ValueError("A workspace directory needs a username.")
    value = _UNSAFE_CHARS.sub("_", name).strip("._-")
    if not value or len(value) > 64:
        return "_" + hashlib.sha256(name.encode("utf-8", "surrogatepass")).hexdigest()[:32]
    return value


def safe_filename(name: str | None, default: str = "file") -> str:
    """File-system safe base name (no separators, no leading dots); falls back to ``default``."""
    raw = Path(str(name or "")).name  # drop any directory part
    value = re.sub(r"[^A-Za-z0-9 _.()-]+", "_", raw).strip().strip(".")
    return value or default


def file_name_of(source: object) -> str:
    """
    The file's name in a path written on any system (both separators are read: an index can move between systems),
    "" for none or a folder. What leaves the server names a file by it, never by the folders it is in, which name the
    user's account and the server's layout.
    """
    return ntpath.basename(str(source or ""))


def network_share(path: str | Path) -> str:
    """
    The network share or device a Windows path names (``\\\\server\\share``, ``//server/share``, ``\\\\?\\UNC\\...``,
    ``\\\\.\\pipe\\...``), normalised; "" for a local path, and always on Linux, where such a path is local. Opening it
    connects to that server, which then receives the account's credentials.
    """
    drive = os.path.splitdrive(str(path))[0]
    return os.path.normcase(drive) if drive[:2] in ("\\\\", "//") else ""


def resolve_within(base: Path, candidate: str | Path) -> Path:
    """Resolve ``candidate`` (absolute or relative to ``base``) and require it to stay inside ``base``.

    Written as normalise-then-check-the-prefix (``os.path.realpath``, then ``startswith``), the form that code
    scanning (CodeQL) recognises as confining a path chosen by the user. A path on another network share is refused
    before it is opened (:func:`network_share`): ``realpath`` would connect to it.
    """
    root = os.path.realpath(base)
    joined = os.path.join(root, candidate)  # an absolute candidate replaces root
    share = network_share(joined)
    if share and share != network_share(root):
        raise ValueError(f"'{candidate}' is outside the allowed directory {root}.")
    path = os.path.realpath(joined)
    if path == root:
        return Path(root)
    inside = root if root.endswith(os.sep) else root + os.sep  # a drive root already ends with one
    if not path.startswith(inside):
        raise ValueError(f"'{candidate}' is outside the allowed directory {root}.")
    return Path(path)


def remove_pdf(folder: str | Path, name: str) -> bool:
    """Delete the PDF ``name`` of ``folder`` (a document of the library), never a file outside it; False when
    there is no such PDF."""
    base = Path(folder).resolve()
    path = resolve_within(base, name)
    if path.parent != base or path.suffix.lower() != ".pdf" or not path.is_file():
        return False
    path.unlink()
    return True


@dataclass(frozen=True)
class UserScope:
    """All storage locations of one user."""

    username: str
    role: str = "researcher"

    @classmethod
    def for_user(cls, username: str | None, role: str | None = None) -> UserScope:
        return cls(username=safe_dirname(username), role=str(role or "researcher"))

    @property
    def is_admin(self) -> bool:
        return self.role.lower() == "admin"

    @property
    def root(self) -> Path:
        path = data_root() / "users" / self.username
        path.mkdir(parents=True, exist_ok=True)
        return path

    def dir(self, kind: str) -> Path:
        if kind not in KINDS:
            raise ValueError(f"Unknown storage kind {kind!r}")
        path = self.root / kind
        path.mkdir(parents=True, exist_ok=True)
        return path

    def datasets_dir(self) -> Path:
        return self.dir("datasets")

    def stats_dir(self) -> Path:
        return self.dir("stats")

    def traces_dir(self) -> Path:
        return self.dir("traces")

    def artifacts_dir(self, session_id: str | None = None) -> Path:
        path = self.traces_dir() / "artifacts"
        if session_id:
            path = path / safe_filename(session_id, "session")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def snapshots_dir(self) -> Path:
        path = self.traces_dir() / "snapshots"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def uploads_dir(self) -> Path:
        return self.dir("uploads")

    def taxonomy_dir(self) -> Path:
        return self.dir("taxonomy")

    def knowledge_graphs_dir(self) -> Path:
        return self.dir("knowledge_graphs")

    def documents_dir(self) -> Path:
        return self.dir("documents")

    def vectors_dir(self) -> Path:
        """Vector indexes of the document library, one folder per engine (``vectors/lancedb``, ``vectors/qdrant``)."""
        return self.dir("vectors")

    def legacy_chroma_dir(self) -> Path:
        """ChromaDB index of earlier versions (no longer used; not created)."""
        return self.root / "chroma_db"

    def rag_eval_logs_dir(self) -> Path:
        return self.dir("rag_eval_logs")

    def resolve_within(self, candidate: str | Path, kind: str | None = None) -> Path:
        """A path the user may read: inside the whole workspace, or inside one storage kind."""
        base = self.dir(kind) if kind else self.root
        return resolve_within(base, candidate)

    def search_dirs(self) -> list[Path]:
        """Directories searched when a trace references a file by name."""
        return [
            self.uploads_dir(),
            self.artifacts_dir(),
            self.snapshots_dir(),
            self.datasets_dir(),
            self.root,
        ]


def all_user_scopes() -> list[UserScope]:
    """Scopes for every user directory present on disk (administration and migration)."""
    users_dir = data_root() / "users"
    if not users_dir.exists():
        return []
    return [UserScope(username=p.name) for p in sorted(users_dir.iterdir()) if p.is_dir()]


__all__ = [
    "KINDS",
    "USERNAME_PATTERN",
    "UserScope",
    "all_user_scopes",
    "data_root",
    "network_share",
    "resolve_within",
    "safe_dirname",
    "safe_filename",
    "validate_username",
]
