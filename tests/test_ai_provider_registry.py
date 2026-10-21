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


def test_the_declarations_say_what_the_lists_they_replace_said() -> None:
    """Golden, until the lists are deleted (the next commit): Settings', the RAG manager's, the consent's, the settings'
    and the factory's lists equal the declarations."""
    pytest.importorskip(
        "langchain_core", reason="the pages and the factory need the optional 'ai' group"
    )
    from app_pages.rag_sidebar import (
        CLOUD_CATALOG,
        CLOUD_PROVIDERS,
        EMBEDDING_PROVIDERS,
        OLLAMA_CHAT_FALLBACK,
        OLLAMA_EMBEDDINGS,
        PROVIDER_NAMES,
    )
    from manage.ai_payload import LOCAL_PROVIDERS, PROVIDER_LABELS
    from manage.rag.providers.claude import CLAUDE_MODELS
    from manage.rag.providers.gemini import GEMINI_CHAT_MODELS
    from manage.rag_manager import CHAT_ONLY_PROVIDERS
    from manage.rag_manager import CLOUD_PROVIDERS as MANAGER_CLOUD

    declared = {name: AI_PROVIDERS.get(name) for name in NAMES}
    assert PROVIDER_NAMES == {name: p.display_name for name, p in declared.items()}
    assert PROVIDER_LABELS == {name: p.label for name, p in declared.items()}
    assert CLOUD_PROVIDERS == MANAGER_CLOUD == {name for name, p in declared.items() if p.cloud}
    assert set(LOCAL_PROVIDERS) == {name for name, p in declared.items() if not p.cloud}
    assert CHAT_ONLY_PROVIDERS == {name for name, p in declared.items() if not p.embeds()}
    assert tuple(EMBEDDING_PROVIDERS) == tuple(name for name, p in declared.items() if p.embeds())
    assert tuple(OLLAMA_CHAT_FALLBACK) == declared["ollama"].chat_models
    assert tuple(OLLAMA_EMBEDDINGS) == declared["ollama"].embedding_models
    assert tuple(GEMINI_CHAT_MODELS) == declared["gemini"].chat_models
    assert tuple(CLAUDE_MODELS) == declared["anthropic"].chat_models
    for name, entry in CLOUD_CATALOG.items():
        provider = declared[name]
        assert (entry["env"], entry["label"]) == (provider.key_variable, provider.key_label), name
        assert tuple(entry["chat"]) == provider.chat_models, name
        assert tuple(entry["embed"]) == provider.embedding_models, name
