"""
Vector-database connector: the RAG code is agnostic to the engine that stores document chunks.

Engines: **LanceDB** (embedded, on disk; the default) and **Qdrant** (embedded in memory, or a server via
``CMP_QDRANT_URL``), a connector family on :mod:`core.registry`: :data:`VECTOR_STORES`. Adding an engine means one
module implementing :class:`~manage.vectorstores.base.BaseVectorIndex` and one registration below.

The package is light: the registry imports no engine, and the other names are loaded when first used (PEP 562), so
reading the registry (Settings, the start-up check) never loads the AI stack.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from core.registry import ConnectorRegistry

if TYPE_CHECKING:
    from .base import (
        BaseVectorIndex,
        ChunkRow,
        Hit,
        IndexManifest,
        VectorEngine,
        VectorStoreError,
        index_matches,
    )
    from .document_index import DocumentIndex, IndexRetriever
    from .selection import (
        AUTO,
        MachineProfile,
        Recommendation,
        VectorStoreSettings,
        estimate_chunks,
        estimate_index_mb,
        open_index,
        probe_machine,
        qdrant_embedded_warnings,
        recommend,
        select_engine,
    )

# The names are stored in the users' preferences, the index manifests and CMP_VECTOR_STORE: they never change.
VECTOR_STORES: ConnectorRegistry[BaseVectorIndex] = ConnectorRegistry(
    "vector database",
    setting="CMP_VECTOR_STORE",
    auto=("lancedb", "qdrant"),  # after the recommendation for this machine (selection.recommend)
)
VECTOR_STORES.register("lancedb", "manage.vectorstores.lancedb_index:LanceDBIndex", extra="ai")
VECTOR_STORES.register("qdrant", "manage.vectorstores.qdrant_index:QdrantIndex", extra="ai")

# where each name is defined, loaded on first use (the module names are written out, so the layer test sees them)
_LAZY = {
    **dict.fromkeys(
        (
            "BaseVectorIndex",
            "ChunkRow",
            "Hit",
            "IndexManifest",
            "VectorEngine",
            "VectorStoreError",
            "index_matches",
        ),
        "manage.vectorstores.base",
    ),
    **dict.fromkeys(("DocumentIndex", "IndexRetriever"), "manage.vectorstores.document_index"),
    **dict.fromkeys(
        (
            "AUTO",
            "MachineProfile",
            "Recommendation",
            "VectorStoreSettings",
            "estimate_chunks",
            "estimate_index_mb",
            "open_index",
            "probe_machine",
            "qdrant_embedded_warnings",
            "recommend",
            "select_engine",
        ),
        "manage.vectorstores.selection",
    ),
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # the next access does not come here
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = [
    "AUTO",
    "VECTOR_STORES",
    "BaseVectorIndex",
    "ChunkRow",
    "DocumentIndex",
    "Hit",
    "IndexManifest",
    "IndexRetriever",
    "MachineProfile",
    "Recommendation",
    "VectorEngine",
    "VectorStoreError",
    "VectorStoreSettings",
    "estimate_chunks",
    "estimate_index_mb",
    "index_matches",
    "open_index",
    "probe_machine",
    "qdrant_embedded_warnings",
    "recommend",
    "select_engine",
]
