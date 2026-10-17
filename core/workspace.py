"""
Per-user storage scope (audit finding E4).

Every user gets a directory tree under ``<DATA_ROOT>/users/<username>/`` and all persistence goes
through a :class:`UserScope`, instead of filename prefixes checked inconsistently across pages.
Paths supplied by users (uploads, manual file paths) are resolved with :meth:`UserScope.resolve_within`
so they can never leave the user's workspace.

This module has no Streamlit dependency; ``utils.session_scope`` derives the current scope from the
session state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,31}$")
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
    """Directory name for a (possibly legacy) username: never empty, never a path component escape."""
    value = _UNSAFE_CHARS.sub("_", (username or "").strip()).strip("._-")
    if not value or value in (".", "..") or len(value) > 64:
        raise ValueError(f"Cannot derive a workspace directory from username {username!r}.")
    return value


def safe_filename(name: str | None, default: str = "file") -> str:
    """File-system safe base name (no separators, no leading dots); falls back to ``default``."""
    raw = Path(str(name or "")).name  # drop any directory part
    value = re.sub(r"[^A-Za-z0-9 _.()-]+", "_", raw).strip().strip(".")
    return value or default


def resolve_within(base: Path, candidate: str | Path) -> Path:
    """Resolve ``candidate`` (absolute or relative to ``base``) and require it to stay inside ``base``."""
    base = Path(base).resolve()
    path = Path(candidate)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"'{candidate}' is outside the allowed directory {base}.") from exc
    return path


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
    "resolve_within",
    "safe_dirname",
    "safe_filename",
    "validate_username",
]
