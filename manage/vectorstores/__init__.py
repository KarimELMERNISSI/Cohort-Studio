"""
Vector-database connector: the RAG code is agnostic to the engine that stores document chunks.

Engines: **LanceDB** (embedded, on disk; the default) and **Qdrant** (embedded in memory, or a
server via ``CMP_QDRANT_URL``). Adding an engine means one module implementing
:class:`~manage.vectorstores.base.BaseVectorIndex` and one branch in
:func:`~manage.vectorstores.selection.open_index`.
"""

from .base import BaseVectorIndex, ChunkRow, Hit, IndexManifest, VectorEngine, VectorStoreError
from .document_index import DocumentIndex, IndexRetriever
from .selection import (
    AUTO,
    ENGINE_SUMMARIES,
    ENGINE_TITLES,
    MachineProfile,
    Recommendation,
    VectorStoreSettings,
    estimate_chunks,
    estimate_index_mb,
    open_index,
    probe_machine,
    qdrant_embedded_warnings,
    recommend,
    resolve_engine,
)

__all__ = [
    "AUTO",
    "ENGINE_SUMMARIES",
    "ENGINE_TITLES",
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
    "open_index",
    "probe_machine",
    "qdrant_embedded_warnings",
    "recommend",
    "resolve_engine",
]
