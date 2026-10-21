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
