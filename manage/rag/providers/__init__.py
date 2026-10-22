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
from typing import TYPE_CHECKING, Any

from core.registry import ConnectorRegistry

if TYPE_CHECKING:
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
    "embedding_providers",
    "file_readers",
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
