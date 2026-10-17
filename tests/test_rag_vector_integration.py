"""RAGManager end to end on both vector databases: build, chat on one document, reuse, rebuild after a change of
the library, embedding guard."""

import os
from pathlib import Path

import pytest

pytest.importorskip("langchain_core", reason="the AI stack is in the optional 'ai' extra")
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import manage.rag_manager as rm
import utils.data_paths as dp
from manage.vectorstores.qdrant_index import release_clients
from tests.test_vectorstores import WordHashEmbeddings

TEXTS = {
    "bmi.pdf": [
        "Body mass index is weight divided by height squared",
        "BMI above 30 defines obesity",
    ],
    "kidney.pdf": [
        "eGFR estimates kidney function from creatinine",
        "Chronic kidney disease staging uses eGFR",
    ],
}


class FakeLoader:
    """Stands for PDFDocumentLoader: one Document per page, ``source`` is the file path."""

    def __init__(self, directory: str) -> None:
        self.directory = directory

    def load(self, recursive: bool = True) -> list[Document]:
        return [
            Document(text, metadata={"source": os.path.join(self.directory, name), "page": page})
            for name, pages in TEXTS.items()
            for page, text in enumerate(pages, start=1)
        ]

    def load_single_pdf(self, path: str) -> list[Document]:
        return [d for d in self.load() if d.metadata["source"] == path]


@pytest.fixture
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("CMP_VECTOR_STORE", raising=False)
    monkeypatch.delenv("CMP_QDRANT_URL", raising=False)
    docs = tmp_path / "docs"
    docs.mkdir()
    embeddings = WordHashEmbeddings()
    monkeypatch.setattr(rm, "PDFDocumentLoader", FakeLoader)
    monkeypatch.setattr(
        rm.ConnectorFactory,
        "create_connectors",
        classmethod(
            lambda cls, **kw: (
                FakeListChatModel(responses=["answer from the documents"] * 20),
                embeddings,
            )
        ),
    )
    monkeypatch.setattr(
        rm.RAGManager,
        "_analyze_global_context",
        lambda self, texts, dataset_columns=None: {"domain": "Nephrology", "roles": {}},
    )
    manager = rm.RAGManager(documents_dir=str(docs), provider="ollama")
    manager.embeddings_used = embeddings
    yield manager
    release_clients()


def _init(manager, engine: str, *, reuse: bool, embedding: str = "nomic-embed-text:latest"):
    return manager.initialize_system(
        provider="ollama",
        model_name="llama3.2:latest",
        embedding_model=embedding,
        use_existing_db=reuse,
        vector_engine=engine,
    )


@pytest.mark.parametrize("engine", ["lancedb", "qdrant"])
def test_rag_runs_on_both_vector_databases(manager, engine: str) -> None:
    ok, message = _init(manager, engine, reuse=False)
    assert ok, message
    assert f"{engine} index" in message and manager.vector_engine == engine
    assert manager.vector_store.count() == 4
    assert manager.vector_store.manifest.embedding == "ollama:nomic-embed-text:latest"

    hits = manager.vector_store.similarity_search("kidney eGFR", k=2)
    assert {os.path.basename(d.metadata["source"]) for d in hits} == {"kidney.pdf"}
    assert (
        manager.chat_with_specific_doc("bmi.pdf", "What defines obesity?")
        == "answer from the documents"
    )

    calls = manager.embeddings_used.calls
    ok, message = _init(
        manager, engine, reuse=True
    )  # reuses the index: no new embedding of documents
    assert ok, message
    assert manager.embeddings_used.calls == calls
    assert manager.vector_store.count() == 4

    ok, message = _init(manager, engine, reuse=True, embedding="mxbai-embed-large:latest")
    assert not ok and "built with ollama:nomic-embed-text:latest" in message


def test_a_refused_key_is_not_shown(manager, monkeypatch: pytest.MonkeyPatch) -> None:
    # the shape of a Google key, built here so that no secret scanner sees one
    key = "AIza" + "Sy0abcDEF_ghi-JKL" * 2 + "m"

    def refuse(texts):
        raise RuntimeError(f"403 PERMISSION_DENIED. Consumer 'api_key:{key}' has been suspended.")

    monkeypatch.setattr(manager.embeddings_used, "embed_documents", refuse)
    monkeypatch.setattr("manage.vectorstores.document_index.time.sleep", lambda seconds: None)
    ok, message = _init(manager, "lancedb", reuse=False)
    assert not ok and "has been suspended" in message and key not in message


@pytest.mark.parametrize("engine", ["lancedb", "qdrant"])
def test_an_index_is_rebuilt_when_the_library_changes(
    manager, engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    ok, message = _init(manager, engine, reuse=False)
    assert ok, message

    monkeypatch.setitem(TEXTS, "anaemia.pdf", ["Haemoglobin below 13 g/dL defines anaemia in men"])
    calls = manager.embeddings_used.calls
    ok, message = _init(manager, engine, reuse=True)  # a document added: rebuilt, not reused
    assert ok, message
    assert manager.embeddings_used.calls > calls
    assert manager.vector_store.count() == 5
    assert manager.vector_store.manifest.documents == ("anaemia.pdf", "bmi.pdf", "kidney.pdf")

    monkeypatch.delitem(TEXTS, "bmi.pdf")
    ok, message = _init(manager, engine, reuse=True)  # a document removed: rebuilt without it
    assert ok, message
    assert manager.vector_store.count() == 3
    assert manager.vector_store.manifest.documents == ("anaemia.pdf", "kidney.pdf")
