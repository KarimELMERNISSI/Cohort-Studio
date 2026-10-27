"""
Unit Tests for RAG Connectors and Document Loaders.

Validates provider configuration, factory instantiation, error resilience,
and native PDF loader functionality across Gemini, Ollama, OpenAI, and Mistral.
"""


import pytest

pytest.importorskip("langchain_core", reason="the AI stack is in the optional 'ai' extra")
from langchain_core.messages import HumanMessage

from manage.rag import (
    ConnectorFactory,
    PDFDocumentLoader,
    ProviderConfig,
    ProviderType,
    RAGConfig,
)
from manage.rag.providers.gemini import GeminiChatConnector, GeminiEmbeddingConnector
from manage.rag.providers.mistral import MistralChatConnector, MistralEmbeddingConnector
from manage.rag.providers.ollama import OllamaChatConnector, OllamaEmbeddingConnector
from manage.rag.providers.openai import OpenAIChatConnector, OpenAIEmbeddingConnector


class TestRAGSchemas:
    """Tests for Pydantic configuration schemas."""

    def test_provider_config_defaults(self):
        cfg = ProviderConfig()  # local, as the application: no longer Google Gemini, a cloud provider nobody chose
        assert cfg.provider == ProviderType.OLLAMA
        assert cfg.model_name == "llama3.2:latest"
        assert cfg.embedding_model == "nomic-embed-text:latest"
        assert cfg.temperature == 0.3
        assert cfg.max_retries == 3

    def test_a_provider_s_configuration_takes_its_own_models(self):
        gemini = ProviderConfig(provider=ProviderType.GEMINI)  # not the fields' defaults, Ollama's names
        assert (gemini.model_name, gemini.embedding_model) == (
            "gemini-3.8-flash",
            "models/gemini-embedding-2",
        )
        assert ProviderConfig(provider="custom").model_name == "gpt-4o-mini"  # OpenAI's protocol
        given = ProviderConfig(provider="gemini", model_name="gemini-2.5-pro")
        assert given.model_name == "gemini-2.5-pro"

    def test_base_url_sanitization(self):
        cfg = ProviderConfig(
            provider=ProviderType.OLLAMA,
            base_url="http://localhost:11434/"
        )
        assert cfg.base_url == "http://localhost:11434"

    def test_rag_config_validation(self):
        rag_cfg = RAGConfig()
        assert rag_cfg.chunk_size == 1000
        assert rag_cfg.chunk_overlap == 200
        assert rag_cfg.top_k == 8
        assert rag_cfg.adherence_score == 0.5


class TestConnectorFactory:
    """Tests for connector factory instantiation across providers."""

    def test_gemini_factory(self):
        cfg = ProviderConfig(provider=ProviderType.GEMINI, api_key="fake-key")
        llm = ConnectorFactory.get_llm_connector(cfg)
        emb = ConnectorFactory.get_embedding_connector(cfg)
        assert isinstance(llm, GeminiChatConnector)
        assert isinstance(emb, GeminiEmbeddingConnector)
        assert llm._llm_type == "gemini-genai-v2"

    def test_ollama_factory(self):
        cfg = ProviderConfig(
            provider=ProviderType.OLLAMA,
            model_name="llama3.2:latest",
            embedding_model="nomic-embed-text:latest"
        )
        llm = ConnectorFactory.get_llm_connector(cfg)
        emb = ConnectorFactory.get_embedding_connector(cfg)
        assert isinstance(llm, OllamaChatConnector)
        assert isinstance(emb, OllamaEmbeddingConnector)
        assert llm._llm_type == "ollama-local"
        assert emb.base_url == "http://localhost:11434"

    def test_openai_factory(self):
        cfg = ProviderConfig(provider=ProviderType.OPENAI, api_key="sk-fake")
        llm = ConnectorFactory.get_llm_connector(cfg)
        emb = ConnectorFactory.get_embedding_connector(cfg)
        assert isinstance(llm, OpenAIChatConnector)
        assert isinstance(emb, OpenAIEmbeddingConnector)
        assert llm._llm_type == "openai-chat"

    def test_mistral_factory(self):
        cfg = ProviderConfig(provider=ProviderType.MISTRAL, api_key="fake-key")
        llm = ConnectorFactory.get_llm_connector(cfg)
        emb = ConnectorFactory.get_embedding_connector(cfg)
        assert isinstance(llm, MistralChatConnector)
        assert isinstance(emb, MistralEmbeddingConnector)
        assert llm._llm_type == "mistral-chat"

    def test_create_connectors_convenience(self):
        llm, emb = ConnectorFactory.create_connectors(
            provider="ollama",
            model_name="mistral:latest",
            embedding_model="bge-m3:latest"
        )
        assert isinstance(llm, OllamaChatConnector)
        assert isinstance(emb, OllamaEmbeddingConnector)
        assert llm.config.model_name == "mistral:latest"


class TestOllamaConnectorResilience:
    """Tests failure resilience when Ollama server is unreachable."""

    def test_ollama_unreachable_graceful_fail(self):
        cfg = ProviderConfig(
            provider=ProviderType.OLLAMA,
            base_url="http://127.0.0.1:9999",  # Unused port
            timeout_seconds=1.0,
        )
        llm = OllamaChatConnector(config=cfg)
        result = llm._generate([HumanMessage(content="Hello")])
        assert len(result.generations) == 1
        output_text = result.generations[0].message.content
        assert "Error" in output_text or "Connection" in output_text

    def test_ollama_fallback_models_on_unreachable(self):
        cfg = ProviderConfig(
            provider=ProviderType.OLLAMA,
            base_url="http://127.0.0.1:9999",
        )
        llm = OllamaChatConnector(config=cfg)
        models = llm.get_available_models()
        assert len(models) > 0
        assert "llama3.2:latest" in models


class TestPDFDocumentLoader:
    """Tests for native pypdf loader."""

    def test_nonexistent_path_returns_empty(self):
        loader = PDFDocumentLoader("non_existent_folder_xyz_123")
        docs = loader.load()
        assert docs == []

    def test_non_pdf_file_returns_empty(self, tmp_path):
        txt_file = tmp_path / "sample.txt"
        txt_file.write_text("Hello world", encoding="utf-8")
        loader = PDFDocumentLoader(txt_file)
        docs = loader.load()
        assert docs == []

    def test_a_page_names_its_file_never_its_folder(self, tmp_path, monkeypatch):
        """The source travels with every passage, to the index (a Qdrant server) and the prompts: the folder names the
        user's account and the server's layout."""
        from types import SimpleNamespace

        import manage.rag.document_loader as loader_module

        pdf = tmp_path / "users" / "alice" / "documents" / "Smith_J.pdf"
        pdf.parent.mkdir(parents=True)
        pdf.write_bytes(b"%PDF-1.4")
        page = SimpleNamespace(extract_text=lambda: "eGFR falls with age.")
        monkeypatch.setattr(loader_module, "PdfReader", lambda path: SimpleNamespace(pages=[page]))
        docs = PDFDocumentLoader(pdf.parent).load_single_pdf(str(pdf))
        assert [d.metadata["source"] for d in docs] == ["Smith_J.pdf"]
        assert "alice" not in str(docs[0].metadata)
