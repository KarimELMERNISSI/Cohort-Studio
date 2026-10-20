"""Anthropic Claude connector: message translation, request, response, errors, factory and RAG manager wiring."""

from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core", reason="the AI stack is in the optional 'ai' extra")
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import manage.rag_manager as rm
from manage.rag import ConnectorFactory, ProviderConfig, ProviderType
from manage.rag.providers.claude import CLAUDE_MODELS, ClaudeChatConnector, to_anthropic_messages
from manage.rag.providers.ollama import OllamaEmbeddingConnector
from manage.rag.providers.openai import OpenAIEmbeddingConnector


class FakeMessages:
    def __init__(self, fail: bool = False) -> None:
        self.requests: list[dict] = []
        self.fail = fail

    def create(self, **request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("overloaded")
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="..."),
                SimpleNamespace(type="text", text="Hello"),
            ],
            model=request["model"],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=12, output_tokens=3),
        )


def _connector(fail: bool = False) -> tuple[ClaudeChatConnector, FakeMessages]:
    connector = ClaudeChatConnector(
        config=ProviderConfig(provider=ProviderType.ANTHROPIC, model_name="claude-fable-5-1")
    )
    messages = FakeMessages(fail=fail)
    connector.client = SimpleNamespace(messages=messages)
    return connector, messages


def test_messages_are_translated_for_the_messages_api() -> None:
    system, turns = to_anthropic_messages(
        [
            SystemMessage("You are a clinical research assistant."),
            SystemMessage("Answer from the documents."),
            HumanMessage("First question"),
            HumanMessage([{"type": "text", "text": "with a second part"}]),
            AIMessage("An answer"),
            HumanMessage("Follow-up"),
        ]
    )
    assert system == "You are a clinical research assistant.\n\nAnswer from the documents."
    assert [t["role"] for t in turns] == [
        "user",
        "assistant",
        "user",
    ]  # alternating, same-role turns merged
    assert turns[0]["content"] == "First question\n\nwith a second part"
    _, starts_with_assistant = to_anthropic_messages([AIMessage("Hi")])
    assert starts_with_assistant[0]["role"] == "user"  # the API needs a user turn first


def test_request_and_response() -> None:
    connector, fake = _connector()
    result = connector.invoke(
        [SystemMessage("Be brief."), HumanMessage("Define NNT.")], stop=["END"]
    )
    request = fake.requests[0]
    assert request["model"] == "claude-fable-5-1" and request["max_tokens"] == 4096
    assert request["system"] == "Be brief." and request["messages"] == [
        {"role": "user", "content": "Define NNT."}
    ]
    assert request["stop_sequences"] == ["END"]
    assert "temperature" not in request  # Claude models choose their own sampling
    assert result.content == "Hello"  # text blocks only
    assert result.response_metadata["output_tokens"] == 3


def test_works_inside_an_lcel_chain() -> None:
    connector, fake = _connector()
    chain = (
        ChatPromptTemplate.from_messages([("system", "Context: {context}"), ("human", "{input}")])
        | connector
        | StrOutputParser()
    )
    assert chain.invoke({"context": "eGFR", "input": "What is it?"}) == "Hello"
    assert fake.requests[0]["system"] == "Context: eGFR"


def test_errors_and_missing_key() -> None:
    connector, _ = _connector(fail=True)
    assert connector.invoke("hi").content == "Error: Claude API Error: overloaded"
    no_key = ClaudeChatConnector(config=ProviderConfig(provider=ProviderType.ANTHROPIC))
    assert no_key.client is None
    assert no_key.invoke("hi").content == "Error: Anthropic API key is missing."
    assert no_key.get_available_models() == list(CLAUDE_MODELS)
    assert no_key._llm_type == "anthropic-claude"


def test_factory_pairs_claude_with_another_embedding_provider() -> None:
    llm, emb = ConnectorFactory.create_connectors(provider="anthropic", api_key="sk-test")
    assert isinstance(llm, ClaudeChatConnector) and isinstance(emb, OllamaEmbeddingConnector)
    assert (
        llm.config.model_name == "claude-fable-5-1"
        and emb.config.embedding_model == "nomic-embed-text:latest"
    )
    llm, emb = ConnectorFactory.create_connectors(
        provider="anthropic",
        api_key="sk-test",
        embedding_provider="openai",
        embedding_api_key="sk-openai",
        embedding_model="text-embedding-3-small",
    )
    assert isinstance(emb, OpenAIEmbeddingConnector) and emb.config.api_key == "sk-openai"
    with pytest.raises(ValueError, match="no embedding API"):
        ConnectorFactory.get_embedding_connector(
            ProviderConfig(provider=ProviderType.ANTHROPIC, api_key="sk-test")
        )


def test_rag_manager_requires_consent_and_keys(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    manager = rm.RAGManager(documents_dir=str(tmp_path), provider="anthropic")
    ok, message = manager.initialize_system(provider="anthropic", model_name="claude-fable-5-1")
    assert not ok and "explicit confirmation" in message
    manager.cloud_consent = True
    ok, message = manager.initialize_system(provider="anthropic", model_name="claude-fable-5-1")
    assert not ok and message == "Anthropic API Key is required."
    manager.api_key = "sk-test"
    ok, message = manager.initialize_system(provider="anthropic", embedding_provider="anthropic")
    assert not ok and "no embedding API" in message
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    manager.consents.add("openai")  # the embedding provider needs the confirmation too
    ok, message = manager.initialize_system(provider="anthropic", embedding_provider="openai")
    assert not ok and message == "Openai API Key is required for the embeddings."


def test_rag_manager_passes_the_embedding_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Claude answers, Ollama embeds: the factory gets the embedding provider and the index records it."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    import utils.data_paths as dp
    from manage.vectorstores.qdrant_index import release_clients
    from tests.test_rag_vector_integration import FakeLoader
    from tests.test_vectorstores import WordHashEmbeddings

    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("CMP_VECTOR_STORE", raising=False)
    calls: list[dict] = []

    def fake_create(cls, **kwargs):
        calls.append(kwargs)
        return FakeListChatModel(responses=["ok"] * 5), WordHashEmbeddings()

    monkeypatch.setattr(rm, "PDFDocumentLoader", FakeLoader)
    monkeypatch.setattr(rm.ConnectorFactory, "create_connectors", classmethod(fake_create))
    monkeypatch.setattr(
        rm.RAGManager,
        "_analyze_global_context",
        lambda self, texts, dataset_columns=None: {"roles": {}},
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    manager = rm.RAGManager(documents_dir=str(docs), provider="anthropic", api_key="sk-test")
    manager.cloud_consent = True
    ok, message = manager.initialize_system(
        provider="anthropic",
        model_name="claude-fable-5-1",
        embedding_model="nomic-embed-text:latest",
        embedding_base_url="http://localhost:11434",
        vector_engine="lancedb",
    )
    release_clients()
    assert ok, message
    assert (
        calls[0]["embedding_provider"] == "ollama"
        and calls[0]["embedding_base_url"] == "http://localhost:11434"
    )
    assert calls[0]["api_key"] == "sk-test" and calls[0]["embedding_api_key"] is None
    assert manager.vector_store.manifest.embedding == "ollama:nomic-embed-text:latest"
    assert manager.embedding_provider == "ollama"
