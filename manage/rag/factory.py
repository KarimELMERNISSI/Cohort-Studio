"""
Connector Factory for RAG LLMs and Embeddings.

Instantiates provider-specific connectors based on ProviderConfig or UI parameters,
providing a single entry point for model-agnostic execution.
"""

import logging
from typing import Optional, Tuple, Dict, Any

from .schemas import ProviderConfig, ProviderType
from .base import BaseLLMConnector, BaseEmbeddingConnector
from .providers.gemini import GeminiChatConnector, GeminiEmbeddingConnector
from .providers.ollama import OllamaChatConnector, OllamaEmbeddingConnector
from .providers.openai import OpenAIChatConnector, OpenAIEmbeddingConnector
from .providers.mistral import MistralChatConnector, MistralEmbeddingConnector

logger = logging.getLogger(__name__)


class ConnectorFactory:
    """
    Factory for creating LLM and Embedding connectors.
    """

    @classmethod
    def get_llm_connector(cls, config: ProviderConfig) -> BaseLLMConnector:
        """
        Instantiate the appropriate LLM chat connector.
        
        Args:
            config: Provider configuration specification.
            
        Returns:
            Instance of BaseLLMConnector.
        """
        provider = config.provider
        logger.info(f"Instantiating LLM connector for provider: {provider.value} (model={config.model_name})")

        if provider == ProviderType.GEMINI:
            return GeminiChatConnector(config=config)
        elif provider == ProviderType.OLLAMA:
            return OllamaChatConnector(config=config)
        elif provider == ProviderType.OPENAI:
            return OpenAIChatConnector(config=config)
        elif provider == ProviderType.MISTRAL:
            return MistralChatConnector(config=config)
        else:
            # Fallback to OpenAI-compatible connector for custom provider endpoints
            return OpenAIChatConnector(config=config)

    @classmethod
    def get_embedding_connector(cls, config: ProviderConfig) -> BaseEmbeddingConnector:
        """
        Instantiate the appropriate Embedding connector.
        
        Args:
            config: Provider configuration specification.
            
        Returns:
            Instance of BaseEmbeddingConnector.
        """
        provider = config.provider
        logger.info(f"Instantiating Embedding connector for provider: {provider.value} (model={config.embedding_model})")

        if provider == ProviderType.GEMINI:
            return GeminiEmbeddingConnector(config=config)
        elif provider == ProviderType.OLLAMA:
            return OllamaEmbeddingConnector(config=config)
        elif provider == ProviderType.OPENAI:
            return OpenAIEmbeddingConnector(config=config)
        elif provider == ProviderType.MISTRAL:
            return MistralEmbeddingConnector(config=config)
        else:
            return OpenAIEmbeddingConnector(config=config)

    @classmethod
    def create_connectors(
        cls,
        provider: str = "gemini",
        model_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> Tuple[BaseLLMConnector, BaseEmbeddingConnector]:
        """
        Convenience factory method constructing both connectors from keyword arguments.
        """
        try:
            ptype = ProviderType(provider.lower())
        except ValueError:
            ptype = ProviderType.GEMINI

        # Default model pairings if not explicitly supplied
        default_models = {
            ProviderType.GEMINI: ("gemini-1.5-flash", "models/gemini-embedding-001"),
            ProviderType.OLLAMA: ("llama3.2:latest", "nomic-embed-text:latest"),
            ProviderType.OPENAI: ("gpt-4o-mini", "text-embedding-3-small"),
            ProviderType.MISTRAL: ("mistral-small-latest", "mistral-embed"),
            ProviderType.CUSTOM: ("gpt-4o-mini", "text-embedding-3-small"),
        }
        def_llm, def_embed = default_models.get(ptype, ("gemini-1.5-flash", "models/gemini-embedding-001"))

        config = ProviderConfig(
            provider=ptype,
            model_name=model_name or def_llm,
            embedding_model=embedding_model or def_embed,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            **kwargs,
        )

        llm = cls.get_llm_connector(config)
        embeddings = cls.get_embedding_connector(config)
        return llm, embeddings
