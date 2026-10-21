"""
Connector Factory for RAG LLMs and Embeddings.

Instantiates provider-specific connectors from a :class:`ProviderConfig` through the providers' registry
(:data:`manage.rag.providers.AI_PROVIDERS`): each provider declares its chat and embedding connectors, imported when an
engine starts, and its default models. A provider this version does not offer is refused with the valid names, never
replaced by another (an unknown name used to become Gemini, a cloud provider nobody chose).
"""

import importlib
import logging
from typing import Any

from .base import BaseEmbeddingConnector, BaseLLMConnector
from .providers import declared, embedding_providers
from .providers.declarations import BaseAIProvider
from .schemas import ProviderConfig, ProviderType

logger = logging.getLogger(__name__)

CUSTOM_PROTOCOL = "openai"  # a custom endpoint (ProviderType.CUSTOM) speaks OpenAI's protocol


def _declared(provider: ProviderType | str) -> type[BaseAIProvider]:
    name = provider.value if isinstance(provider, ProviderType) else str(provider)
    return declared(CUSTOM_PROTOCOL if name == ProviderType.CUSTOM.value else name)


def _connector(target: str) -> Any:
    module, _, name = target.partition(":")
    return getattr(importlib.import_module(module), name)


class ConnectorFactory:
    """
    Factory for creating LLM and Embedding connectors.
    """

    @classmethod
    def get_llm_connector(cls, config: ProviderConfig) -> BaseLLMConnector:
        """The chat connector of ``config.provider``."""
        provider = _declared(config.provider)
        logger.info("Instantiating LLM connector for provider: %s (model=%s)", provider.name, config.model_name)
        return _connector(provider.chat_connector)(config=config)

    @classmethod
    def get_embedding_connector(cls, config: ProviderConfig) -> BaseEmbeddingConnector:
        """The embedding connector of ``config.provider``; ValueError for a provider without an embedding API."""
        provider = _declared(config.provider)
        logger.info(
            "Instantiating Embedding connector for provider: %s (model=%s)", provider.name, config.embedding_model
        )
        if provider.embedding_connector is None:
            raise ValueError(
                f"{provider.label} has no embedding API: choose an embedding provider (local Ollama by default) for it."
            )
        return _connector(provider.embedding_connector)(config=config)

    @classmethod
    def create_connectors(
        cls,
        provider: str,
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

        ``embedding_provider`` lets the documents be embedded by another provider than the one that answers; a provider
        without an embedding API (Claude) is embedded by the first provider that has one (local Ollama) unless another
        is given. ``embedding_api_key`` / ``embedding_base_url`` configure that provider. The default models are the
        first each provider declares.
        """
        ptype = ProviderType(provider.lower())  # a provider this version does not offer: ValueError
        answering = _declared(ptype)
        if embedding_provider:
            embedder = _declared(embedding_provider.lower())
        elif answering.embeds():
            embedder = answering
        else:
            embedder = declared(embedding_providers()[0])

        config = ProviderConfig(
            provider=ptype,
            model_name=model_name or answering.chat_models[0],
            embedding_model=embedding_model or (embedder.embedding_models or ("",))[0],
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            **kwargs,
        )
        llm = cls.get_llm_connector(config)

        if embedder.name == answering.name:
            embeddings = cls.get_embedding_connector(config)
        else:
            embed_config = ProviderConfig(
                provider=ProviderType(embedder.name),
                model_name=embedder.chat_models[0],
                embedding_model=embedding_model or embedder.embedding_models[0],
                api_key=embedding_api_key,
                base_url=embedding_base_url,
                temperature=temperature,
                **kwargs,
            )
            embeddings = cls.get_embedding_connector(embed_config)
        return llm, embeddings
