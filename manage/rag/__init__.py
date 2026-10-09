"""
RAG Subsystem Package.

Provides model-agnostic LLM and Embedding connectors supporting multiple providers:
- Google Gemini (google-genai SDK)
- Ollama (Local open-source, offline, sovereign)
- OpenAI (OpenAI official API)
- Mistral AI (European sovereign models)
"""

from .document_loader import PDFDocumentLoader
from .factory import ConnectorFactory
from .schemas import ProviderConfig, ProviderType, RAGConfig

__all__ = [
    "ConnectorFactory",
    "PDFDocumentLoader",
    "ProviderConfig",
    "ProviderType",
    "RAGConfig",
]
