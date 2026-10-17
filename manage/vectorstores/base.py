"""
Contract shared by the vector-database engines (LanceDB, Qdrant).

The application never talks to an engine directly: it stores *chunks* (text, vector, source document,
page, metadata) and asks for the nearest chunks of a query vector, optionally restricted to some
source documents. Each engine translates that into its own storage and filter language. No
Streamlit here (service layer).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1


class VectorEngine(StrEnum):
    LANCEDB = "lancedb"
    QDRANT = "qdrant"


class VectorStoreError(RuntimeError):
    """An engine is unavailable, unreachable or holds an incompatible index."""


@dataclass(frozen=True)
class ChunkRow:
    """One indexed piece of a document."""

    id: str
    text: str
    vector: list[float]
    source: str = ""
    page: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)

    def metadata_json(self) -> str:
        return json.dumps(self.metadata, default=str, ensure_ascii=False)


@dataclass(frozen=True)
class Hit:
    """A chunk returned by a search; ``score`` is the cosine similarity (1 = identical direction)."""

    text: str
    metadata: dict[str, Any]
    score: float
    vector: list[float] | None = None


@dataclass(frozen=True)
class IndexManifest:
    """What an index holds and how it was built (stored next to it as ``manifest.json``)."""

    engine: str
    embedding: str  # "<provider>:<model>", the vectors are only comparable with the same model
    dim: int
    chunks: int
    documents: tuple[str, ...] = ()
    location: str = ""  # folder (embedded) or server URL
    built_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    version: int = MANIFEST_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["documents"] = list(self.documents)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexManifest:
        known = set(cls.__dataclass_fields__)
        clean = {k: v for k, v in data.items() if k in known}
        clean["documents"] = tuple(clean.get("documents") or ())
        return cls(**clean)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MANIFEST_NAME
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: Path) -> IndexManifest | None:
        path = directory / MANIFEST_NAME
        if not path.exists():
            return None
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except OSError, ValueError, TypeError:
            return None


def index_matches(manifest: IndexManifest | None, documents: Iterable[str]) -> bool:
    """Whether an index was built from exactly these documents: after a document is added to or removed from the
    library, the index must be rebuilt rather than reused."""
    return manifest is not None and sorted(set(manifest.documents)) == sorted(set(documents))


class BaseVectorIndex(ABC):
    """Storage of chunks in one engine. Implementations: :mod:`.lancedb_index`, :mod:`.qdrant_index`."""

    engine: ClassVar[VectorEngine]

    def __init__(self, manifest_dir: Path) -> None:
        self.manifest_dir = manifest_dir

    @classmethod
    @abstractmethod
    def availability(cls) -> tuple[bool, str]:
        """Whether the engine's client library can be used, and why not."""

    @property
    @abstractmethod
    def location(self) -> str:
        """Where the vectors live (folder or server), for display and the manifest."""

    @abstractmethod
    def exists(self) -> bool:
        """An index (possibly empty) is present."""

    @abstractmethod
    def reset(self, dim: int) -> None:
        """Drop any previous index and create an empty one for vectors of ``dim`` dimensions."""

    @abstractmethod
    def add(self, rows: list[ChunkRow]) -> None: ...

    @abstractmethod
    def query(self, vector: list[float], k: int, sources: list[str] | None = None) -> list[Hit]:
        """The ``k`` nearest chunks (cosine), with their vectors, optionally only from ``sources``."""

    @abstractmethod
    def count(self) -> int: ...

    def optimize(self) -> None:
        """Build an approximate-search index when the engine benefits from it (large collections)."""

    def close(self) -> None:
        """Release files or connections."""

    def manifest(self) -> IndexManifest | None:
        return IndexManifest.load(self.manifest_dir)

    def write_manifest(self, manifest: IndexManifest) -> None:
        manifest.save(self.manifest_dir)


__all__ = [
    "MANIFEST_NAME",
    "BaseVectorIndex",
    "ChunkRow",
    "Hit",
    "IndexManifest",
    "VectorEngine",
    "VectorStoreError",
]
