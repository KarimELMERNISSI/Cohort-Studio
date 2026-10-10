"""
RAG Provider and Model Configuration Schemas.

Defines strongly-typed Pydantic schemas for LLM and Embedding provider settings,
ensuring validation, serialization, and support for multi-provider configurations.
"""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ProviderType(str, Enum):
    """Supported LLM and Embedding providers."""
    GEMINI = "gemini"
    OLLAMA = "ollama"
    OPENAI = "openai"
    MISTRAL = "mistral"
    CUSTOM = "custom"


class ProviderConfig(BaseModel):
    """
    Configuration specification for a specific LLM and Embedding provider.
    
    Attributes:
        provider: Provider identifier (gemini, ollama, openai, mistral, custom).
        model_name: Name of the chat/completion model (e.g., 'gemini-2.5-flash', 'gemini-2.5-pro', 'llama3.2', 'gpt-4o-mini').
        embedding_model: Name of the embedding model (e.g., 'models/text-embedding-004', 'nomic-embed-text').
        api_key: Optional API key for commercial cloud endpoints.
        base_url: Optional custom endpoint URL (crucial for Ollama or local inference servers).
        temperature: Sampling temperature for generation (0.0 to 1.0).
        timeout_seconds: Request timeout in seconds.
        max_retries: Maximum number of retry attempts upon transient network failures.
        extra_headers: Optional HTTP headers for custom gateways.
    """
    provider: ProviderType = Field(
        default=ProviderType.GEMINI,
        description="The AI provider backend to use."
    )
    model_name: str = Field(
        default="gemini-2.5-flash",
        description="Model identifier for text generation."
    )
    embedding_model: str = Field(
        default="models/text-embedding-004",
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
