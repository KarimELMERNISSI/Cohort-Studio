"""
The AI providers, a connector family on :mod:`core.registry`: :data:`AI_PROVIDERS`. Each is declared in
:mod:`.declarations` (its names, models, key, whether it is a cloud provider), read without importing langchain or its
SDK; its connectors live in its own module, imported when an engine starts:

- ollama: local or network Ollama (open-source models, offline);
- gemini: Google GenAI (the google.genai SDK);
- openai: OpenAI;
- mistral: Mistral AI (OpenAI's protocol);
- anthropic: Anthropic Claude (chat only: another provider embeds the documents).

The package is light: the connectors' names are loaded when first used (PEP 562).
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any

from core.registry import ConnectorRegistry, Selection

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from utils.settings import Settings

    from .claude import ClaudeChatConnector
    from .declarations import BaseAIProvider
    from .gemini import GeminiChatConnector, GeminiEmbeddingConnector
    from .mistral import MistralChatConnector, MistralEmbeddingConnector
    from .ollama import OllamaChatConnector, OllamaEmbeddingConnector
    from .openai import OpenAIChatConnector, OpenAIEmbeddingConnector

# Settings' order; the providers that embed, in this order, embed for a chat-only one (Ollama first: local)
AI_PROVIDERS: ConnectorRegistry[BaseAIProvider] = ConnectorRegistry(
    "AI provider", setting="CMP_AI_PROVIDER", auto=("ollama",)
)
AI_PROVIDERS.register("ollama", "manage.rag.providers.declarations:Ollama", extra="ai")
AI_PROVIDERS.register("gemini", "manage.rag.providers.declarations:Gemini", extra="ai")
AI_PROVIDERS.register("openai", "manage.rag.providers.declarations:OpenAI", extra="ai")
AI_PROVIDERS.register("mistral", "manage.rag.providers.declarations:Mistral", extra="ai")
AI_PROVIDERS.register("anthropic", "manage.rag.providers.declarations:Claude", extra="ai")


def declared(name: str) -> type[BaseAIProvider]:
    """The declaration of the provider ``name`` (UnknownConnectorError, listing the valid names, when it is not one)."""
    return AI_PROVIDERS.get(name)


def cloud_providers() -> frozenset[str]:
    """The providers that send data out of the server: the confirmation of each session applies to them."""
    return frozenset(name for name in AI_PROVIDERS.names() if declared(name).cloud)


def file_readers() -> frozenset[str]:
    """The providers that Document insight sends the PDF file itself (their chat connector's read_document); the
    others are sent its text. The graph extraction and the consent notice both read this."""
    return frozenset(name for name in AI_PROVIDERS.names() if declared(name).sends_files)


def embedding_providers() -> tuple[str, ...]:
    """The providers that can embed documents, in the registry's order: the first (Ollama, local) embeds for one that
    cannot (Claude) unless another is chosen."""
    return tuple(name for name in AI_PROVIDERS.names() if declared(name).embeds())


def declared_for(name: str) -> type[BaseAIProvider]:
    """The declaration a provider's connectors follow: its own, or OpenAI's for a custom endpoint, which speaks its
    protocol (UnknownConnectorError for another name)."""
    from .declarations import CUSTOM_PROTOCOL

    return declared(CUSTOM_PROTOCOL if name == "custom" else name)


def embedder_for(provider: str, embedding_provider: str | None = None) -> type[BaseAIProvider]:
    """Who embeds the documents of an engine answering with ``provider``: the embedding provider given, else the
    provider itself when it can, else the first that can (local Ollama embeds for Claude)."""
    if embedding_provider:
        return declared_for(embedding_provider.lower())
    answering = declared_for(provider)
    return answering if answering.embeds() else declared(embedding_providers()[0])


def default_models(provider: str, embedding_provider: str | None = None) -> tuple[str, str]:
    """The models used when none is given: the first chat model ``provider`` declares, and the first embedding model of
    the provider that embeds ("" when it declares none). The factory and the engine read them here."""
    embedder = embedder_for(provider, embedding_provider)
    return declared_for(provider).chat_models[0], (embedder.embedding_models or ("",))[0]


def provider_setting(settings: Settings | None = None) -> str:
    """CMP_AI_PROVIDER as the selection reads it ("auto" when unset). A value that names no provider raises
    SettingsError with the variable and the valid names, never the value (a mistyped value can be a pasted key);
    start-up stops on it first (manage.connectors)."""
    from utils.settings import SettingsError, get_settings

    value = (settings or get_settings()).ai.provider
    problem = AI_PROVIDERS.setting_problem(value)
    if problem:
        raise SettingsError(f"{problem}.")
    return value or "auto"


def select_provider(
    *,
    session: str | None = None,
    preference: str | None = None,
    settings: Settings | None = None,
) -> Selection:
    """The AI provider to use: the session's choice, then the saved preference, then the server's setting
    (:func:`provider_setting`), then local Ollama; a name that cannot run here falls through, and the selection says
    why."""
    return AI_PROVIDERS.select(
        session=session, preference=preference, setting=provider_setting(settings)
    )


def default_provider(settings: Settings | None = None) -> str:
    """
    The server's AI provider: the one CMP_AI_PROVIDER names, when it can run here, else Ollama (at OLLAMA_BASE_URL),
    named even when it cannot run here: never a cloud provider nobody chose. Settings preselects it for a user who
    chose none; a cloud provider named so is only preselected: it receives none of the user's data until the session
    confirms it.
    """
    selection = select_provider(settings=settings)
    for warning in selection.warnings:
        logger.warning("%s", warning)
    # none can run here (no AI group): named all the same
    return selection.chosen or AI_PROVIDERS.names()[0]


def server_key(name: str) -> str | None:
    """The server's API key for the provider ``name`` (the setting it declares, read by utils.settings), or None."""
    from utils.settings import get_settings

    canonical = AI_PROVIDERS.canonical(name)
    if canonical is None or canonical == "auto":
        return None
    return get_settings().ai.key(declared(canonical).key_variable)


# where each connector is defined, loaded on first use (written out, so the layer test sees the edges)
_LAZY = {
    "ClaudeChatConnector": "manage.rag.providers.claude",
    "GeminiChatConnector": "manage.rag.providers.gemini",
    "GeminiEmbeddingConnector": "manage.rag.providers.gemini",
    "MistralChatConnector": "manage.rag.providers.mistral",
    "MistralEmbeddingConnector": "manage.rag.providers.mistral",
    "OllamaChatConnector": "manage.rag.providers.ollama",
    "OllamaEmbeddingConnector": "manage.rag.providers.ollama",
    "OpenAIChatConnector": "manage.rag.providers.openai",
    "OpenAIEmbeddingConnector": "manage.rag.providers.openai",
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
    "AI_PROVIDERS",
    "cloud_providers",
    "declared",
    "declared_for",
    "default_models",
    "default_provider",
    "embedder_for",
    "embedding_providers",
    "file_readers",
    "provider_setting",
    "select_provider",
    "server_key",
    "ClaudeChatConnector",
    "GeminiChatConnector",
    "GeminiEmbeddingConnector",
    "MistralChatConnector",
    "MistralEmbeddingConnector",
    "OllamaChatConnector",
    "OllamaEmbeddingConnector",
    "OpenAIChatConnector",
    "OpenAIEmbeddingConnector",
]
