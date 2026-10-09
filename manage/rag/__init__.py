"""
RAG Subsystem Package.

Provides model-agnostic LLM and Embedding connectors supporting multiple providers:
- Google Gemini (google-genai SDK)
- Ollama (Local open-source, offline, sovereign)
- OpenAI (OpenAI official API)
- Mistral AI (European sovereign models)
"""

from .schemas import ProviderType, ProviderConfig, RAGConfig
from .factory import ConnectorFactory
from .document_loader import PDFDocumentLoader

__all__ = [
    "ProviderType",
    "ProviderConfig",
    "RAGConfig",
    "ConnectorFactory",
    "PDFDocumentLoader",
]
