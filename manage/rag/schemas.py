"""
RAG Provider and Model Configuration Schemas.

Defines strongly-typed Pydantic schemas for LLM and Embedding provider settings,
ensuring validation, serialization, and support for multi-provider configurations.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .providers import default_models
from .providers.declarations import Ollama


class ProviderType(str, Enum):
    """Supported LLM and Embedding providers."""
    GEMINI = "gemini"
    OLLAMA = "ollama"
    OPENAI = "openai"
    MISTRAL = "mistral"
    ANTHROPIC = "anthropic"  # Claude: chat only, embeddings from another provider
    CUSTOM = "custom"


class ProviderConfig(BaseModel):
    """
    Configuration specification for a specific LLM and Embedding provider.
    
    Attributes:
        provider: Provider identifier (gemini, ollama, openai, mistral, anthropic, custom).
        model_name: Name of the chat/completion model (e.g., 'gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-3.8-flash', 'llama3.2', 'gpt-4o-mini').
        embedding_model: Name of the embedding model (e.g., 'models/gemini-embedding-2', 'models/text-embedding-005', 'nomic-embed-text').
        api_key: Optional API key for commercial cloud endpoints.
        base_url: Optional custom endpoint URL (crucial for Ollama or local inference servers).
        temperature: Sampling temperature for generation (0.0 to 1.0).
        timeout_seconds: Request timeout in seconds.
        max_retries: Maximum number of retry attempts upon transient network failures.
        extra_headers: Optional HTTP headers for custom gateways.
    """
    # local by default, as the application: nothing leaves the server unless a cloud provider is chosen
    provider: ProviderType = Field(
        default=ProviderType.OLLAMA,
        description="The AI provider backend to use."
    )
    model_name: str = Field(
        default=Ollama.chat_models[0],
        description="Model identifier for text generation."
    )
    embedding_model: str = Field(
        default=Ollama.embedding_models[0],
        description="Model identifier for vector embeddings."
    )
    api_key: str | None = Field(
        default=None,
        description="API Key for authenticated endpoints (None for local Ollama)."
    )
    base_url: str | None = Field(
        default=None,
        description="Base API endpoint URL (e.g., 'http://localhost:11434' for Ollama)."
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Creativity temperature parameter."
    )
    timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        description="Network request timeout in seconds."
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts on network error."
    )
    extra_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Custom HTTP headers."
    )

    @model_validator(mode="before")
    @classmethod
    def _declared_models(cls, data: Any) -> Any:
        """A provider's configuration without models takes those it declares first (manage.rag.providers.default_models;
        a custom endpoint, OpenAI's): a Gemini configuration never holds the fields' defaults, Ollama's names."""
        if not isinstance(data, dict) or data.get("provider") in (None, ""):
            return data
        name = str(getattr(data["provider"], "value", data["provider"])).strip().lower()
        try:
            chat, embedding = default_models(name)
        except LookupError:  # not a provider: the field refuses it
            return data
        return {
            **data,
            "model_name": data.get("model_name") or chat,
            "embedding_model": data.get("embedding_model") or embedding or Ollama.embedding_models[0],
        }

    @field_validator("base_url")
    @classmethod
    def sanitize_base_url(cls, v: str | None) -> str | None:
        """Strip trailing slashes for consistent endpoint path appending."""
        if v:
            return v.rstrip("/")
        return v


class RAGConfig(BaseModel):
    """
    Global configuration for the RAG engine and retrieval pipelines.
    
    Attributes:
        provider_config: Underlying provider configuration.
        documents_dir: Directory path storing source PDF documents.
        chunk_size: Number of characters per text chunk during ingestion.
        chunk_overlap: Character overlap between consecutive chunks.
        top_k: Number of most relevant documents to retrieve per query.
        fetch_k: Candidate pool size for Maximum Marginal Relevance (MMR) retrieval.
        search_type: Retrieval strategy ('mmr' or 'similarity').
        adherence_score: Degree of adherence to document context (0.0=creative, 1.0=strict).
    """
    provider_config: ProviderConfig = Field(
        default_factory=ProviderConfig,
        description="Active provider configuration."
    )
    documents_dir: str = Field(
        default="DOCUMENTS",
        description="Path to local documents storage directory."
    )
    chunk_size: int = Field(
        default=1000,
        gt=100,
        description="Chunk character size for document splitting."
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Chunk character overlap for context preservation."
    )
    top_k: int = Field(
        default=8,
        gt=0,
        description="Number of chunks retrieved for QA context."
    )
    fetch_k: int = Field(
        default=20,
        gt=0,
        description="Candidate chunk pool size before MMR reranking."
    )
    search_type: str = Field(
        default="mmr",
        description="Vector search method ('mmr' or 'similarity')."
    )
    adherence_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Adherence weighting between retrieved context and general knowledge."
    )
