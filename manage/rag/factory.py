"""
Connector Factory for RAG LLMs and Embeddings.

Instantiates provider-specific connectors based on ProviderConfig or UI parameters,
providing a single entry point for model-agnostic execution.
"""

import logging
from typing import Any

from .base import BaseEmbeddingConnector, BaseLLMConnector
from .providers.claude import ClaudeChatConnector
from .providers.gemini import GeminiChatConnector, GeminiEmbeddingConnector
from .providers.mistral import MistralChatConnector, MistralEmbeddingConnector
from .providers.ollama import OllamaChatConnector, OllamaEmbeddingConnector
from .providers.openai import OpenAIChatConnector, OpenAIEmbeddingConnector
from .schemas import ProviderConfig, ProviderType

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
        elif provider == ProviderType.ANTHROPIC:
            return ClaudeChatConnector(config=config)
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
        elif provider == ProviderType.ANTHROPIC:
            raise ValueError(
                "Anthropic has no embedding API: choose an embedding provider (local Ollama by default) for Claude."
            )
        else:
            return OpenAIEmbeddingConnector(config=config)

    @classmethod
    def create_connectors(
        cls,
        provider: str = "gemini",
        model_name: str | None = None,
        embedding_model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.3,
        embedding_provider: str | None = None,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        **kwargs: Any,
    ) -> tuple[BaseLLMConnector, BaseEmbeddingConnector]:
        """
        Convenience factory method constructing both connectors from keyword arguments.

        ``embedding_provider`` lets the documents be embedded by another provider than the one that answers;
        it is required in practice for Claude (Anthropic has no embedding API) and defaults to local Ollama
        there. ``embedding_api_key`` / ``embedding_base_url`` configure that provider.
        """
        try:
            ptype = ProviderType(provider.lower())
        except ValueError:
            ptype = ProviderType.GEMINI

        # Default model pairings if not explicitly supplied
        default_models = {
            ProviderType.GEMINI: ("gemini-2.5-flash", "models/gemini-embedding-2"),
            ProviderType.OLLAMA: ("llama3.2:latest", "nomic-embed-text:latest"),
            ProviderType.OPENAI: ("gpt-4o-mini", "text-embedding-3-small"),
            ProviderType.MISTRAL: ("mistral-small-latest", "mistral-embed"),
            ProviderType.ANTHROPIC: ("claude-fable-5-1", "nomic-embed-text:latest"),
            ProviderType.CUSTOM: ("gpt-4o-mini", "text-embedding-3-small"),
        }
        def_llm, def_embed = default_models.get(ptype, ("gemini-2.5-flash", "models/gemini-embedding-2"))

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

        embed_type = ptype
        if embedding_provider:
            embed_type = ProviderType(embedding_provider.lower())
        elif ptype == ProviderType.ANTHROPIC:
            embed_type = ProviderType.OLLAMA
        if embed_type == ptype:
            embeddings = cls.get_embedding_connector(config)
        else:
            embed_config = ProviderConfig(
                provider=embed_type,
                model_name=default_models.get(embed_type, (def_llm, def_embed))[0],
                embedding_model=embedding_model or default_models.get(embed_type, (def_llm, def_embed))[1],
                api_key=embedding_api_key,
                base_url=embedding_base_url,
                temperature=temperature,
                **kwargs,
            )
            embeddings = cls.get_embedding_connector(embed_config)
        return llm, embeddings
