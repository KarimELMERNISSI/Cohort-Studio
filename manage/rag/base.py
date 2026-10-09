"""
Base Connector Interfaces for LLM and Embedding Providers.

Provides abstract and extensible base classes conforming to LangChain Core contracts
(BaseChatModel, Embeddings) for seamless interoperability with LCEL chains, Vector Stores,
and evaluation components.
"""

from abc import abstractmethod
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import Field

from .schemas import ProviderConfig


class BaseLLMConnector(BaseChatModel):
    """
    Abstract base class for all LLM connectors.
    
    Subclasses must implement:
    - _generate: LangChain core generation interface.
    - _llm_type: String identifier of the LLM provider.
    - get_available_models: Query available models for the provider.
    """
    config: ProviderConfig = Field(default_factory=ProviderConfig)
    
    model_config = {
        "arbitrary_types_allowed": True
    }

    @property
    @abstractmethod
    def _llm_type(self) -> str:
        """Provider identifier string for LangChain introspection."""

    @abstractmethod
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Core execution method converting messages to model completion."""

    def get_available_models(self) -> list[str]:
        """
        Retrieve available text generation models for this provider.
        Default fallback returns the configured model.
        """
        return [self.config.model_name]


class BaseEmbeddingConnector(Embeddings):
    """
    Abstract base class for embedding connectors.
    
    Subclasses must implement:
    - embed_documents: Batch embedding generation.
    - embed_query: Single query string embedding.
    """

    def __init__(self, config: ProviderConfig):
        """Initialize with provider configuration."""
        self.config = config

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for a batch of text strings."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Compute an embedding for a single search query."""

    def get_available_models(self) -> list[str]:
        """
        Retrieve available embedding models for this provider.
        Default fallback returns the configured model.
        """
        return [self.config.embedding_model]
