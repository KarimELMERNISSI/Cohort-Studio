"""Vector-database connector: LanceDB and Qdrant behave the same behind DocumentIndex; engine recommendation."""

import hashlib
import math
import re
from pathlib import Path

import pytest

pytest.importorskip("langchain_core", reason="the AI stack is in the optional 'ai' extra")
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from manage.vectorstores import (
    DocumentIndex,
    IndexManifest,
    MachineProfile,
    VectorEngine,
    VectorStoreError,
    estimate_chunks,
    index_matches,
    open_index,
    qdrant_embedded_warnings,
    recommend,
)
from manage.vectorstores.qdrant_index import release_clients

DIM = 64


class WordHashEmbeddings(Embeddings):
    """Deterministic bag-of-words vectors: texts sharing words point in similar directions."""

    def __init__(self, label: str = "fake:words") -> None:
        self.label = label
        self.calls = 0

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


CHUNKS = [
    Document(
        "Body mass index is weight divided by height squared",
        metadata={"source": "a/bmi.pdf", "page": 1},
    ),
    Document("BMI above 30 defines obesity in adults", metadata={"source": "a/bmi.pdf", "page": 2}),
    Document(
        "eGFR estimates kidney function from serum creatinine",
        metadata={"source": "b/kidney.pdf", "page": 4},
    ),
    Document(
        "Creatinine clearance and eGFR staging of chronic kidney disease",
        metadata={"source": "b/kidney.pdf", "page": 5},
    ),
    Document(
        "Blood pressure targets in hypertension trials",
        metadata={"source": "c/it's bp.pdf", "page": 1},
    ),
    Document(
        "Systolic blood pressure above 140 mmHg is hypertension",
        metadata={"source": "c/it's bp.pdf", "page": 2},
    ),
]


@pytest.fixture(params=[VectorEngine.LANCEDB, VectorEngine.QDRANT], ids=["lancedb", "qdrant"])
def index(request, tmp_path: Path):
    backend = open_index(request.param, base_dir=tmp_path / "vectors", username="alice")
    yield DocumentIndex(backend, WordHashEmbeddings())
    release_clients()


def test_build_count_and_manifest(index: DocumentIndex) -> None:
    progress: list[tuple[int, int]] = []
    manifest = index.build(
        CHUNKS,
        embedding_label="fake:words",
        documents=["bmi.pdf", "kidney.pdf", "it's bp.pdf"],
        batch_size=4,
        progress=lambda done, total: progress.append((done, total)),
    )
    assert index.count() == len(CHUNKS) == manifest.chunks
    assert manifest.dim == DIM and manifest.engine == index.engine
    assert progress == [(4, 6), (6, 6)]
    assert index.manifest == manifest  # stored next to the index


def test_search_finds_the_nearest_chunks(index: DocumentIndex) -> None:
    index.build(CHUNKS, embedding_label="fake:words")
    docs = index.similarity_search("kidney function eGFR creatinine", k=2)
    assert {d.metadata["source"] for d in docs} == {"kidney.pdf"}  # stored by name, never by path
    assert docs[0].metadata["page"] in (4, 5) and 0 < docs[0].metadata["score"] <= 1.0


def test_filter_by_document_including_quotes_and_legacy_filter(index: DocumentIndex) -> None:
    index.build(CHUNKS, embedding_label="fake:words")
    only_bp = index.similarity_search("kidney", k=5, sources=["c/it's bp.pdf"])
    assert only_bp and {d.metadata["source"] for d in only_bp} == {"it's bp.pdf"}
    legacy = index.similarity_search("obesity", k=5, filter={"source": {"$in": ["a/bmi.pdf"]}})
    assert {d.metadata["source"] for d in legacy} == {"bmi.pdf"}
    by_name = index.similarity_search("obesity", k=5, sources=["bmi.pdf"])
    assert {d.metadata["source"] for d in by_name} == {"bmi.pdf"}


def test_mmr_and_retriever(index: DocumentIndex) -> None:
    index.build(CHUNKS, embedding_label="fake:words")
    mmr = index.max_marginal_relevance_search("blood pressure hypertension kidney", k=3, fetch_k=6)
    assert len(mmr) == 3 and len({d.page_content for d in mmr}) == 3
    retriever = index.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "fetch_k": 6, "sources": ["a/bmi.pdf"]}
    )
    docs = retriever.invoke("body mass index obesity")
    assert len(docs) == 2 and all(d.metadata["source"] == "bmi.pdf" for d in docs)


def test_the_index_stores_no_folder_of_a_document(index: DocumentIndex) -> None:
    """A Qdrant server receives the index: a chunk names its document, never the folder it is in on the server."""
    folder = "/srv/cohortstudio/data/users/alice/documents"
    chunks = [
        Document(c.page_content, metadata={**c.metadata, "source": f"{folder}/{c.metadata['source'].split('/')[-1]}"})
        for c in CHUNKS
    ]
    index.build(chunks, embedding_label="fake:words")
    stored = index.similarity_search("kidney blood pressure obesity", k=len(CHUNKS))
    assert len(stored) == len(CHUNKS)
    assert not any("alice" in str(d.metadata) for d in stored)
    assert {d.metadata["source"] for d in stored} == {"bmi.pdf", "kidney.pdf", "it's bp.pdf"}
    # a filter still names a document by its path, as Document insight does
    kidney = index.similarity_search("eGFR", k=5, sources=[f"{folder}/kidney.pdf"])
    assert {d.metadata["source"] for d in kidney} == {"kidney.pdf"}


def test_an_index_built_by_an_earlier_version_is_still_filtered_by_path(index: DocumentIndex) -> None:
    """Its chunks hold the path (a rebuild drops it): a filter by path still finds them."""
    from manage.vectorstores.base import ChunkRow

    embeddings = WordHashEmbeddings()
    index.backend.reset(DIM)
    index.backend.add(
        [
            ChunkRow(
                id=f"old-{i}",
                text=c.page_content,
                vector=embeddings.embed_query(c.page_content),
                source=f"/srv/data/{c.metadata['source']}",
                page=int(c.metadata["page"]),
                metadata={**c.metadata, "source": f"/srv/data/{c.metadata['source']}"},
            )
            for i, c in enumerate(CHUNKS)
        ]
    )
    old = index.similarity_search("eGFR kidney", k=5, sources=["/srv/data/b/kidney.pdf"])
    assert old and {d.metadata["source"] for d in old} == {"/srv/data/b/kidney.pdf"}


def test_reopen_reuse_and_embedding_mismatch(index: DocumentIndex, tmp_path: Path) -> None:
    index.build(CHUNKS, embedding_label="fake:words")
    reopened = DocumentIndex(
        open_index(index.backend.engine, base_dir=tmp_path / "vectors", username="alice"),
        WordHashEmbeddings(),
    )
    assert reopened.count() == len(CHUNKS)
    reopened.check_compatible("fake:words")
    with pytest.raises(VectorStoreError, match="built with fake:words"):
        reopened.check_compatible("ollama:other-model")


def test_rebuild_replaces_the_index(index: DocumentIndex) -> None:
    index.build(CHUNKS, embedding_label="fake:words")
    index.build(CHUNKS[:2], embedding_label="fake:words")
    assert index.count() == 2


def test_embedding_failures_are_reported_not_skipped(index: DocumentIndex) -> None:
    class Failing(WordHashEmbeddings):
        def embed_documents(self, texts):
            raise RuntimeError("rate limited")

    broken = DocumentIndex(index.backend, Failing())
    with pytest.raises(VectorStoreError, match="Embedding failed for chunks 1-6 after 2 attempts"):
        broken.build(CHUNKS, embedding_label="fake:words", batch_size=10, retries=2)


def test_recommendation_follows_machine_and_library() -> None:
    laptop = MachineProfile(cpu_count=8, memory_gb=16.0, available_gb=8.0, disk_free_gb=200.0)
    rec = recommend(laptop, chunks=5_000)
    assert rec.engine is VectorEngine.LANCEDB and rec.mode == "embedded"
    assert any("Exact search" in r for r in rec.reasons)
    server = recommend(laptop, chunks=5_000, qdrant_server=True)
    assert server.engine is VectorEngine.QDRANT and server.mode == "server"
    tiny_disk = recommend(MachineProfile(4, 4.0, 1.0, 0.01), chunks=200_000)
    assert tiny_disk.warnings and "free on disk" in tiny_disk.warnings[0]
    assert any("approximate (IVF)" in r for r in tiny_disk.reasons)
    big = qdrant_embedded_warnings(MachineProfile(4, 4.0, 1.0, 50.0), chunks=300_000)
    assert any("Qdrant server" in w for w in big) and any(
        "quarter of the free memory" in w for w in big
    )
    assert estimate_chunks([2_500_000]) == 1_000


def test_an_index_is_reused_only_when_it_holds_the_same_documents() -> None:
    manifest = IndexManifest(engine="lancedb", embedding="ollama:x", dim=4, chunks=2, documents=("a.pdf", "b.pdf"))
    assert index_matches(manifest, ["b.pdf", "a.pdf"])
    assert not index_matches(manifest, ["a.pdf"])  # a document was removed from the library
    assert not index_matches(manifest, ["a.pdf", "b.pdf", "c.pdf"])  # one was added
    assert not index_matches(None, ["a.pdf"])  # no index yet
