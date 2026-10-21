"""The AI providers as a connector family (M5b): declared without langchain or an SDK, so that the registry and its
contract run without the optional 'ai' group; the declarations say what the lists they replace said."""

import pytest

from core.registry import Availability
from manage.rag.providers import AI_PROVIDERS
from tests import connector_contract as contract

NAMES = ("ollama", "gemini", "openai", "mistral", "anthropic")


def test_the_stored_names_never_change() -> None:
    # kept in the users' preferences (ai_provider, ai_claude_embedding_provider, the models by provider)
    contract.check_names(AI_PROVIDERS, NAMES)
    assert AI_PROVIDERS.valid() == ("auto", *NAMES)


def test_the_registry_keeps_the_connector_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    contract.check_availability(AI_PROVIDERS, monkeypatch)
    contract.check_classes(AI_PROVIDERS)
    contract.check_precedence(AI_PROVIDERS)
    contract.check_unknown_setting(AI_PROVIDERS)
    contract.check_unknown_names(AI_PROVIDERS)


@pytest.mark.parametrize(
    ("name", "module"),
    [
        ("ollama", "langchain_core"),
        ("gemini", "google.genai"),
        ("openai", "openai"),
        ("mistral", "openai"),
        ("anthropic", "anthropic"),
    ],
)
def test_a_missing_package_names_its_group(
    name: str, module: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract.check_missing_package(AI_PROVIDERS, name, module, monkeypatch)


def test_listing_the_providers_loads_neither_langchain_nor_an_sdk() -> None:
    contract.check_light(
        "manage.rag.providers:AI_PROVIDERS",
        never_loaded=("langchain_core", "google.genai", "anthropic", "openai", "streamlit"),
    )


def test_each_provider_declares_what_the_application_needs() -> None:
    for name in NAMES:
        provider = AI_PROVIDERS.get(name)
        assert provider.chat_models and provider.chat_connector.count(":") == 1, name
        assert provider.cloud == (provider.key_variable is not None), name
        assert (provider.embedding_connector is None) == (not provider.embeds()), name
    embedders = [name for name in NAMES if AI_PROVIDERS.get(name).embeds()]
    assert embedders == [
        "ollama",
        "gemini",
        "openai",
        "mistral",
    ]  # Claude answers; another provider embeds
    assert [name for name in NAMES if not AI_PROVIDERS.get(name).cloud] == ["ollama"]
    assert (
        AI_PROVIDERS.select(check=lambda name: Availability(True)).chosen == "ollama"
    )  # local first


def test_the_declarations_keep_what_users_and_settings_rely_on() -> None:
    """The lists the declarations replaced (golden test of 168ad63), pinned: the defaults are the first of each list,
    the settings that hold the keys keep their names, and the consent and the embedding fall-back depend on them."""
    pins = {
        name: (p.cloud, p.key_variable, p.chat_models[0], p.embedding_models[:1])
        for name in NAMES
        for p in [AI_PROVIDERS.get(name)]
    }
    assert pins == {
        "ollama": (False, None, "llama3.2:latest", ("nomic-embed-text:latest",)),
        "gemini": (True, "GOOGLE_API_KEY", "gemini-3.8-flash", ("models/gemini-embedding-2",)),
        "openai": (True, "OPENAI_API_KEY", "gpt-4o-mini", ("text-embedding-3-small",)),
        "mistral": (True, "MISTRAL_API_KEY", "mistral-small-latest", ("mistral-embed",)),
        "anthropic": (True, "ANTHROPIC_API_KEY", "claude-fable-5-1", ()),
    }


def test_the_readers_go_through_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    from manage.ai_payload import local_providers, provider_label, recipient
    from manage.rag.providers import cloud_providers, embedding_providers, server_key

    assert cloud_providers() == {"gemini", "openai", "mistral", "anthropic"}
    assert local_providers() == {"ollama"}
    assert embedding_providers() == ("ollama", "gemini", "openai", "mistral")
    assert provider_label("anthropic") == "Anthropic Claude" and provider_label("zz9") == "zz9"
    assert recipient("custom") == "custom"  # a provider not declared local needs the confirmation
    monkeypatch.setenv("MISTRAL_API_KEY", "planted-mistral-key")
    assert server_key("Mistral") == "planted-mistral-key" and server_key("ollama") is None


def test_the_factory_refuses_a_provider_it_does_not_offer() -> None:
    pytest.importorskip("langchain_core", reason="the connectors need the optional 'ai' group")
    from manage.rag import ConnectorFactory

    with pytest.raises(
        ValueError
    ):  # before, an unknown name became Gemini, a cloud provider nobody chose
        ConnectorFactory.create_connectors(provider="zz9-cloud")
    llm, embeddings = ConnectorFactory.create_connectors(provider="anthropic", api_key="k")
    assert type(llm).__name__ == "ClaudeChatConnector"
    assert (
        type(embeddings).__name__ == "OllamaEmbeddingConnector"
    )  # Claude's documents: local Ollama by default
