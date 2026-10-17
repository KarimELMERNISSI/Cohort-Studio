"""
Engine-agnostic document index used by the RAG code.

:class:`DocumentIndex` wraps a :class:`~manage.vectorstores.base.BaseVectorIndex` and the embedding
model. It exposes the LangChain vector-store calls the application uses (``similarity_search``,
``max_marginal_relevance_search``, ``as_retriever``), so the RAG code is identical whatever the
engine. Filtering by document is expressed as a list of sources; the legacy Chroma filter
(``{"source": x}`` or ``{"source": {"$in": [...]}}``) is still accepted.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from pydantic import ConfigDict

from .base import BaseVectorIndex, ChunkRow, Hit, IndexManifest, VectorStoreError

ProgressCallback = Callable[[int, int], None]  # (chunks done, chunks total)


def _sources_from_filter(filter_: Any) -> list[str] | None:
    """Translate the legacy Chroma filter on ``source`` into a list of sources."""
    if not isinstance(filter_, dict) or "source" not in filter_:
        return None
    value = filter_["source"]
    if isinstance(value, dict):
        value = value.get("$in") or value.get("$eq")
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return None


def _page(value: Any) -> int:
    try:
        return int(value)
    except TypeError, ValueError:
        return -1


def chunk_id(document: Document, position: int) -> str:
    """Stable identifier of a chunk: same document, page, position and text give the same id."""
    md = document.metadata or {}
    key = f"{md.get('source', '')}|{md.get('page', '')}|{position}|{document.page_content}"
    return hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()


def _to_document(hit: Hit) -> Document:
    return Document(page_content=hit.text, metadata={**hit.metadata, "score": round(hit.score, 4)})


class DocumentIndex:
    """Chunks of the user's documents in the selected engine, searched with the selected embedding model."""

    def __init__(self, backend: BaseVectorIndex, embeddings: Embeddings) -> None:
        self.backend = backend
        self.embeddings = embeddings

    # ------------------------------------------------------------------ information

    @property
    def engine(self) -> str:
        return self.backend.engine.value

    @property
    def location(self) -> str:
        return self.backend.location

    @property
    def manifest(self) -> IndexManifest | None:
        return self.backend.manifest()

    def count(self) -> int:
        return self.backend.count()

    # ------------------------------------------------------------------ building

    def _embed(self, texts: list[str], retries: int, first: int) -> list[list[float]]:
        last: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                vectors = self.embeddings.embed_documents(texts)
                if len(vectors) != len(texts):
                    raise VectorStoreError(
                        f"{len(vectors)} vectors returned for {len(texts)} chunks"
                    )
                return vectors
            except Exception as exc:  # rate limits and transient network errors
                last = exc
                time.sleep(min(1.0 * attempt, 5.0))
        raise VectorStoreError(
            f"Embedding failed for chunks {first + 1}-{first + len(texts)} after {retries} attempts: {last}"
        )

    def build(
        self,
        chunks: Sequence[Document],
        *,
        embedding_label: str,
        documents: Sequence[str] = (),
        batch_size: int = 16,
        pause: float = 0.0,
        retries: int = 3,
        progress: ProgressCallback | None = None,
    ) -> IndexManifest:
        """Embed every chunk and replace the index; returns (and stores) the manifest."""
        total = len(chunks)
        dim: int | None = None
        for start in range(0, total, batch_size):
            batch = list(chunks[start : start + batch_size])
            vectors = self._embed([c.page_content for c in batch], retries, start)
            if dim is None:
                dim = len(vectors[0])
                self.backend.reset(dim)
            self.backend.add(
                [
                    ChunkRow(
                        id=chunk_id(doc, start + i),
                        text=doc.page_content,
                        vector=list(map(float, vec)),
                        source=str((doc.metadata or {}).get("source", "")),
                        page=_page((doc.metadata or {}).get("page")),
                        metadata=dict(doc.metadata or {}),
                    )
                    for i, (doc, vec) in enumerate(zip(batch, vectors, strict=True))
                ]
            )
            if progress:
                progress(min(start + len(batch), total), total)
            if pause:
                time.sleep(pause)
        if dim is None:  # no text at all: an empty index of the right dimension
            dim = len(self.embeddings.embed_query("dimension probe"))
            self.backend.reset(dim)
        self.backend.optimize()
        manifest = IndexManifest(
            engine=self.engine,
            embedding=embedding_label,
            dim=dim,
            chunks=self.backend.count(),
            documents=tuple(sorted(set(documents))),
            location=self.location,
        )
        self.backend.write_manifest(manifest)
        return manifest

    def check_compatible(self, embedding_label: str) -> None:
        """Refuse to reuse an index built with another embedding model (its vectors are not comparable)."""
        manifest = self.manifest
        if manifest is None:
            if self.backend.exists() and self.count() > 0:
                return  # built before manifests existed: trust it
            raise VectorStoreError("No index to reuse yet: build it from your documents.")
        if manifest.embedding != embedding_label:
            raise VectorStoreError(
                f"The existing {manifest.engine} index was built with {manifest.embedding} "
                f"({manifest.dim} dimensions); the selected embedding model is {embedding_label}. "
                "Rebuild the index (turn off 'Reuse the existing index')."
            )

    # ------------------------------------------------------------------ searching

    def similarity_search(
        self, query: str, k: int = 4, sources: list[str] | None = None, **kwargs: Any
    ) -> list[Document]:
        sources = sources or _sources_from_filter(kwargs.get("filter"))
        vector = self.embeddings.embed_query(query)
        return [_to_document(h) for h in self.backend.query(vector, k, sources)]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        sources: list[str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Relevant and diverse chunks: the ``k`` best of the ``fetch_k`` nearest, penalising near-duplicates."""
        sources = sources or _sources_from_filter(kwargs.get("filter"))
        vector = self.embeddings.embed_query(query)
        hits = self.backend.query(vector, max(fetch_k, k), sources)
        candidates = [h for h in hits if h.vector is not None]
        if len(candidates) <= k or len(candidates) != len(hits):
            return [_to_document(h) for h in hits[:k]]
        picked = maximal_marginal_relevance(
            np.asarray(vector, dtype=float),
            [h.vector for h in candidates],
            lambda_mult=lambda_mult,
            k=k,
        )
        return [_to_document(candidates[i]) for i in picked]

    def as_retriever(
        self, search_type: str = "similarity", search_kwargs: dict | None = None, **_: Any
    ):
        kwargs = dict(search_kwargs or {})
        sources = kwargs.pop("sources", None) or _sources_from_filter(kwargs.pop("filter", None))
        return IndexRetriever(
            index=self,
            search_type=search_type,
            k=int(kwargs.get("k", 4)),
            fetch_k=int(kwargs.get("fetch_k", 20)),
            lambda_mult=float(kwargs.get("lambda_mult", 0.5)),
            sources=sources,
        )


class IndexRetriever(BaseRetriever):
    """LangChain retriever over a :class:`DocumentIndex` (plain similarity or MMR)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: Any
    search_type: str = "similarity"
    k: int = 4
    fetch_k: int = 20
    lambda_mult: float = 0.5
    sources: list[str] | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if self.search_type == "mmr":
            return self.index.max_marginal_relevance_search(
                query,
                k=self.k,
                fetch_k=self.fetch_k,
                lambda_mult=self.lambda_mult,
                sources=self.sources,
            )
        return self.index.similarity_search(query, k=self.k, sources=self.sources)


__all__ = ["DocumentIndex", "IndexRetriever", "chunk_id"]
