"""
RAG subsystem: model-agnostic chat and embedding connectors for several providers (see
:mod:`manage.rag.providers`, a connector family: Ollama, Google Gemini, OpenAI, Mistral AI, Anthropic Claude).

The package is light: its names are loaded when first used (PEP 562), so that reading the providers' registry
(Settings, the start-up check) loads neither langchain nor a provider's SDK.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .document_loader import PDFDocumentLoader
    from .factory import ConnectorFactory
    from .schemas import ProviderConfig, ProviderType, RAGConfig

# where each name is defined, loaded on first use (written out, so the layer test sees the edges)
_LAZY = {
    "ConnectorFactory": "manage.rag.factory",
    "PDFDocumentLoader": "manage.rag.document_loader",
    "ProviderConfig": "manage.rag.schemas",
    "ProviderType": "manage.rag.schemas",
    "RAGConfig": "manage.rag.schemas",
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
    "ConnectorFactory",
    "PDFDocumentLoader",
    "ProviderConfig",
    "ProviderType",
    "RAGConfig",
]
