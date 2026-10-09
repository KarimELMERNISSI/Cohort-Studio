"""
Provider Implementations for RAG Connectors.

Modules:
- gemini: Google GenAI connector (modern google.genai SDK)
- ollama: Ollama connector (local open-source models, offline)
- openai: OpenAI connector (GPT-4o, embeddings)
- mistral: Mistral AI connector (European sovereign models)
"""

from .gemini import GeminiChatConnector, GeminiEmbeddingConnector
from .mistral import MistralChatConnector, MistralEmbeddingConnector
from .ollama import OllamaChatConnector, OllamaEmbeddingConnector
from .openai import OpenAIChatConnector, OpenAIEmbeddingConnector

__all__ = [
    "GeminiChatConnector",
    "GeminiEmbeddingConnector",
    "MistralChatConnector",
    "MistralEmbeddingConnector",
    "OllamaChatConnector",
    "OllamaEmbeddingConnector",
    "OpenAIChatConnector",
    "OpenAIEmbeddingConnector",
]
