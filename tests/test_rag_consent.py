"""
Withdrawing the cloud consent stops the AI engine (M5). The consent was checked only when the engine started, so an
engine kept answering, and sending data to its provider, after the consent was withdrawn in Settings.
"""

import pytest

pytest.importorskip("langchain_core", reason="the AI stack is in the optional 'ai' extra")

import manage.rag_manager as rm


def _running_engine(monkeypatch: pytest.MonkeyPatch, tmp_path, provider: str) -> rm.RAGManager:
    """An engine started with fake connectors: a chat model that counts its answers, local embeddings."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    import utils.data_paths as dp
    from tests.test_rag_vector_integration import FakeLoader
    from tests.test_vectorstores import WordHashEmbeddings

    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("CMP_VECTOR_STORE", raising=False)
    monkeypatch.setattr(rm, "PDFDocumentLoader", FakeLoader)
    monkeypatch.setattr(
        rm.ConnectorFactory,
        "create_connectors",
        classmethod(
            lambda cls, **kwargs: (FakeListChatModel(responses=["{}"] * 20), WordHashEmbeddings())
        ),
    )
    monkeypatch.setattr(
        rm.RAGManager,
        "_analyze_global_context",
        lambda self, texts, dataset_columns=None: {"roles": {}},
    )
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    manager = rm.RAGManager(documents_dir=str(docs), provider=provider, api_key="sk-test")
    manager.cloud_consent = provider in rm.CLOUD_PROVIDERS
    ok, message = manager.initialize_system(
        provider=provider, model_name="model", embedding_model="embedding", vector_engine="lancedb"
    )
    assert ok, message
    return manager


def test_withdrawing_the_consent_stops_a_cloud_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    manager = _running_engine(monkeypatch, tmp_path, "gemini")
    llm = manager.llm
    assert manager.initialized and manager.uses_cloud()

    assert manager.withdraw_consent() is True
    assert not manager.initialized and manager.cloud_consent is False
    assert manager.llm is None and manager.vector_store is None and manager.qa_chain is None
    assert "consent was withdrawn" in (manager.stop_reason or "")
    answers_before = llm.i
    assert manager.suggest_column_renaming(["age", "sex"]) == (None, "RAG system not initialized.")
    assert llm.i == answers_before  # nothing more was sent to the provider

    manager.cloud_consent = True  # consent given again, and the engine started again
    ok, message = manager.initialize_system(
        provider="gemini", model_name="model", embedding_model="embedding", vector_engine="lancedb"
    )
    assert ok, message
    assert manager.initialized and manager.stop_reason is None


def test_a_local_engine_keeps_running(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    manager = _running_engine(monkeypatch, tmp_path, "ollama")
    assert not manager.uses_cloud()
    assert manager.withdraw_consent() is False
    assert manager.initialized and manager.llm is not None and manager.stop_reason is None


def test_cloud_embeddings_count_as_cloud(tmp_path) -> None:
    # the embedding provider embeds every question: with a cloud one, the questions leave the server
    manager = rm.RAGManager(documents_dir=str(tmp_path), provider="ollama")
    manager.embedding_provider = "openai"
    assert manager.uses_cloud()
