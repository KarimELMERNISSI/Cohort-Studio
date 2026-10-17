# Backend Management Logic

## authz.py

_Authorization and credential-policy helpers shared by the UI and the CLI.

Privileges are derived from the role stored by the IAM provider (``UserProfile.role``), never from
the username string (audit finding E5)._

**Imports**:
`__future__.annotations`, `os`, `re`, `schemas.UserRole`

### def `is_admin_role` (authz.py)

- **Arguments**: `role: ComplexType`

- **Returns**: `bool`

True when the role value denotes an administrator.

### def `validate_username` (authz.py)

- **Arguments**: `username: ComplexType`

- **Returns**: `ComplexType`

Return an error message when the username is unacceptable, else None.

### def `validate_password` (authz.py)

- **Arguments**: `password: ComplexType`

- **Returns**: `ComplexType`

Return an error message when the password is unacceptable, else None.

### def `validate_new_credentials` (authz.py)

- **Arguments**: `username: ComplexType, password: ComplexType, confirm: ComplexType`

- **Returns**: `ComplexType`

Validate a sign-up form; returns the first problem found or None.

---

## base.py

_Base Abstract Interface for Identity and Access Management (IAM) Providers.

Defines the contract that all authentication and user management backends
must implement, enabling framework-agnostic switching between local, remote,
or disabled authentication._

**Imports**:
`abc.ABC`, `abc.abstractmethod`, `schemas.AuthResult`, `schemas.UserCredentials`, `schemas.UserProfile`, `schemas.UserRole`

### class `BaseIAMProvider` (base.py)

Abstract Base Class for IAM Providers.

Subclasses implement credential verification, user provisioning, role
management, and session token resolution against specific backends.

**Methods:**

- **provider_name**(`self`) -> `str`
  > Return the unique name identifier of the IAM provider.

- **is_enabled**(`self`) -> `bool`
  > Indicates whether authentication is actively enforced.
  > Default is True; disabled/bypass providers override to False.

- **authenticate**(`self, credentials: UserCredentials`) -> `AuthResult`
  > Authenticate user credentials and evaluate account activation status.
  >
  > Args:
  >     credentials: UserCredentials containing username and password.
  >
  > Returns:
  >     AuthResult with success status, user profile, and informative message.

- **create_user**(`self, username: str, password: str, role: UserRole, auto_activate: bool`) -> `tuple[ComplexType]`
  > Provision a new user account.
  >
  > Args:
  >     username: Unique alphanumeric username.
  >     password: Raw password string to be securely hashed.
  >     role: Assigned clinical user role.
  >     auto_activate: Whether account is active immediately without admin review.
  >
  > Returns:
  >     Tuple of (success_flag, status_message).

- **verify_session**(`self, token: str`) -> `AuthResult`
  > Validate an active bearer token or session identifier.
  >
  > Args:
  >     token: Session token or bearer string.
  >
  > Returns:
  >     AuthResult with user details if token is valid.

- **get_user**(`self, username: str`) -> `ComplexType`
  > Retrieve structured user profile by username.
  >
  > Args:
  >     username: Target user account name.
  >
  > Returns:
  >     UserProfile if found, else None.

- **list_users**(`self`) -> `list[UserProfile]`
  > List all registered platform users with activation and role status.
  >
  > Returns:
  >     List of UserProfile instances.

- **activate_user**(`self, username: str`) -> `tuple[ComplexType]`
  > Activate a pending user account (administrator authorization).
  >
  > Args:
  >     username: Target user account name.
  >
  > Returns:
  >     Tuple of (success_flag, status_message).

- **deactivate_user**(`self, username: str`) -> `tuple[ComplexType]`
  > Suspend an active user account.
  >
  > Args:
  >     username: Target user account name.
  >
  > Returns:
  >     Tuple of (success_flag, status_message).

- **delete_user**(`self, username: str`) -> `tuple[ComplexType]`
  > Permanently remove a user account from the directory.
  >
  > Args:
  >     username: Target user account name.
  >
  > Returns:
  >     Tuple of (success_flag, status_message).

- **update_password**(`self, username: str, new_password: str`) -> `tuple[ComplexType]`
  > Update credentials for an existing account.
  >
  > Args:
  >     username: Target user account name.
  >     new_password: New raw password string.
  >
  > Returns:
  >     Tuple of (success_flag, status_message).

---

## base.py

_Base Connector Interfaces for LLM and Embedding Providers.

Provides abstract and extensible base classes conforming to LangChain Core contracts
(BaseChatModel, Embeddings) for seamless interoperability with LCEL chains, Vector Stores,
and evaluation components._

**Imports**:
`abc.abstractmethod`, `typing.Any`, `langchain_core.callbacks.manager.CallbackManagerForLLMRun`, `langchain_core.embeddings.Embeddings`, `langchain_core.language_models.chat_models.BaseChatModel`, `langchain_core.messages.BaseMessage`, `langchain_core.outputs.ChatResult`, `pydantic.Field`, `schemas.ProviderConfig`

### class `BaseLLMConnector` (base.py)

Abstract base class for all LLM connectors.

Subclasses must implement:
- _generate: LangChain core generation interface.

- _llm_type: String identifier of the LLM provider.

- get_available_models: Query available models for the provider.

**Methods:**

- **_llm_type**(`self`) -> `str`
  > Provider identifier string for LangChain introspection.

- **_generate**(`self, messages: list[BaseMessage], stop: ComplexType, run_manager: ComplexType`) -> `ChatResult`
  > Core execution method converting messages to model completion.

- **get_available_models**(`self`) -> `list[str]`
  > Retrieve available text generation models for this provider.
  > Default fallback returns the configured model.

### class `BaseEmbeddingConnector` (base.py)

Abstract base class for embedding connectors.

Subclasses must implement:
- embed_documents: Batch embedding generation.

- embed_query: Single query string embedding.

**Methods:**

- **__init__**(`self, config: ProviderConfig`) -> `None`
  > Initialize with provider configuration.

- **embed_documents**(`self, texts: list[str]`) -> `list[list[float]]`
  > Compute embeddings for a batch of text strings.

- **embed_query**(`self, text: str`) -> `list[float]`
  > Compute an embedding for a single search query.

- **get_available_models**(`self`) -> `list[str]`
  > Retrieve available embedding models for this provider.
  > Default fallback returns the configured model.

---

## base.py

_Contract shared by the vector-database engines (LanceDB, Qdrant).

The application never talks to an engine directly: it stores *chunks* (text, vector, source document,
page, metadata) and asks for the nearest chunks of a query vector, optionally restricted to some
source documents. Each engine translates that into its own storage and filter language. No
Streamlit here (service layer)._

**Imports**:
`__future__.annotations`, `json`, `abc.ABC`, `abc.abstractmethod`, `collections.abc.Iterable`, `dataclasses.asdict`, `dataclasses.dataclass`, `dataclasses.field`, `datetime.datetime`, `enum.StrEnum`, `pathlib.Path`, `typing.Any`, `typing.ClassVar`

### class `VectorEngine` (base.py)

No description available.

### class `VectorStoreError` (base.py)

An engine is unavailable, unreachable or holds an incompatible index.

### class `ChunkRow` (base.py)

One indexed piece of a document.

**Methods:**

- **metadata_json**(`self`) -> `str`
  > No description available.

### class `Hit` (base.py)

A chunk returned by a search; ``score`` is the cosine similarity (1 = identical direction).

### class `IndexManifest` (base.py)

What an index holds and how it was built (stored next to it as ``manifest.json``).

**Methods:**

- **to_dict**(`self`) -> `dict[ComplexType]`
  > No description available.

- **from_dict**(`cls, data: dict[ComplexType]`) -> `IndexManifest`
  > No description available.

- **save**(`self, directory: Path`) -> `Path`
  > No description available.

- **load**(`cls, directory: Path`) -> `ComplexType`
  > No description available.

### class `BaseVectorIndex` (base.py)

Storage of chunks in one engine. Implementations: :mod:`.lancedb_index`, :mod:`.qdrant_index`.

**Methods:**

- **__init__**(`self, manifest_dir: Path`) -> `None`
  > No description available.

- **availability**(`cls`) -> `tuple[ComplexType]`
  > Whether the engine's client library can be used, and why not.

- **location**(`self`) -> `str`
  > Where the vectors live (folder or server), for display and the manifest.

- **exists**(`self`) -> `bool`
  > An index (possibly empty) is present.

- **reset**(`self, dim: int`) -> `None`
  > Drop any previous index and create an empty one for vectors of ``dim`` dimensions.

- **add**(`self, rows: list[ChunkRow]`) -> `None`
  > Store ``rows`` (their vectors, text and metadata) in the index.

- **query**(`self, vector: list[float], k: int, sources: ComplexType`) -> `list[Hit]`
  > The ``k`` nearest chunks (cosine), with their vectors, optionally only from ``sources``.

- **count**(`self`) -> `int`
  > The number of chunks the index holds.

- **optimize**(`self`) -> `None`
  > Build an approximate-search index when the engine benefits from it (large collections).

- **close**(`self`) -> `None`
  > Release files or connections.

- **manifest**(`self`) -> `ComplexType`
  > No description available.

- **write_manifest**(`self, manifest: IndexManifest`) -> `None`
  > No description available.

### def `index_matches` (base.py)

- **Arguments**: `manifest: ComplexType, documents: Iterable[str]`

- **Returns**: `bool`

Whether an index was built from exactly these documents: after a document is added to or removed from the
library, the index must be rebuilt rather than reused.

---

## better_auth.py

_Better Auth Remote IAM Provider.

Enables seamless integration with modern authentication frameworks such as
Better Auth (and compatible RESTful identity microservices). Features connection
pooling, short-lived in-memory TTL session caching to eliminate UI latency, and
resilient error handling._

**Imports**:
`logging`, `time`, `requests`, `base.BaseIAMProvider`, `schemas.AuthResult`, `schemas.UserCredentials`, `schemas.UserProfile`, `schemas.UserRole`

### class `BetterAuthIAMProvider` (better_auth.py)

Remote IAM Provider integrating with Better Auth or compatible REST IAM services.

Includes an in-memory TTL cache to ensure that Streamlit rerun cycles do not
degrade performance with repeated synchronous network requests.

**Methods:**

- **__init__**(`self, base_url: ComplexType, api_key: ComplexType, cache_ttl_seconds: int, request_timeout: float`) -> `None`
  > Initialize the Better Auth provider.
  >
  > Args:
  >     base_url: Base endpoint URL of the Better Auth service (e.g., "http://localhost:3000/api/auth").
  >     api_key: Optional Bearer or service API key for administrative queries.
  >     cache_ttl_seconds: Duration to cache verified sessions in memory (default 300s).
  >     request_timeout: HTTP request timeout in seconds (default 3.0s).

- **provider_name**(`self`) -> `str`
  > Return provider identifier.

- **_get_cached_auth**(`self, cache_key: str`) -> `ComplexType`
  > Check in-memory TTL cache to avoid redundant network round-trips.

- **_set_cached_auth**(`self, cache_key: str, result: AuthResult`) -> `None`
  > Store verified authentication result in TTL cache.

- **health_check**(`self`) -> `tuple[ComplexType]`
  > Verify connectivity to the Better Auth service.

- **authenticate**(`self, credentials: UserCredentials`) -> `AuthResult`
  > Authenticate credentials via Better Auth sign-in endpoint.

- **create_user**(`self, username: str, password: str, role: UserRole, auto_activate: bool`) -> `tuple[ComplexType]`
  > Provision a new user account through Better Auth sign-up endpoint.

- **verify_session**(`self, token: str`) -> `AuthResult`
  > Verify existing session token using Better Auth get-session endpoint.

- **get_user**(`self, username: str`) -> `ComplexType`
  > Fetch user profile from Better Auth admin endpoint.

- **list_users**(`self`) -> `list[UserProfile]`
  > Fetch user list from Better Auth admin endpoint.

- **activate_user**(`self, username: str`) -> `tuple[ComplexType]`
  > Activate user via Better Auth admin API.

- **deactivate_user**(`self, username: str`) -> `tuple[ComplexType]`
  > Deactivate user via Better Auth admin API.

- **delete_user**(`self, username: str`) -> `tuple[ComplexType]`
  > Delete user account in Better Auth directory.

- **update_password**(`self, username: str, new_password: str`) -> `tuple[ComplexType]`
  > Reset user password via Better Auth admin API.

---

## claude.py

_Anthropic Claude Provider Connector.

Chat connector for Claude models through the official ``anthropic`` SDK (Messages API). Anthropic has no
embedding API: when Claude answers, documents are embedded by another provider chosen in Settings (local
Ollama by default, so the document library never leaves the server for indexing). See
``ConnectorFactory.create_connectors(embedding_provider=...)``.

Claude models choose their own sampling: the temperature setting of the app is not sent._

**Imports**:
`__future__.annotations`, `logging`, `typing.Any`, `langchain_core.callbacks.manager.CallbackManagerForLLMRun`, `langchain_core.messages.AIMessage`, `langchain_core.messages.BaseMessage`, `langchain_core.messages.SystemMessage`, `langchain_core.outputs.ChatGeneration`, `langchain_core.outputs.ChatResult`, `utils.logging_config.redact_secrets`, `base.BaseLLMConnector`, `schemas.ProviderConfig`

### class `ClaudeChatConnector` (claude.py)

Chat connector for Anthropic Claude models (cloud; requires an API key and the data-sharing consent).

**Methods:**

- **__init__**(`self, config: ProviderConfig`) -> `None`
  > No description available.

- **_llm_type**(`self`) -> `str`
  > No description available.

- **get_available_models**(`self`) -> `list[str]`
  > No description available.

- **_generate**(`self, messages: list[BaseMessage], stop: ComplexType, run_manager: ComplexType`) -> `ChatResult`
  > No description available.

- **_error**(`message: str`) -> `ChatResult`
  > No description available.

### def `_text` (claude.py)

- **Arguments**: `content: Any`

- **Returns**: `str`

LangChain message content (string or list of parts) as plain text.

### def `to_anthropic_messages` (claude.py)

- **Arguments**: `messages: list[BaseMessage]`

- **Returns**: `tuple[ComplexType]`

LangChain messages -> (system prompt, Messages API turns).

System messages are joined into the ``system`` parameter. The turns alternate user / assistant and start
with a user turn, as the API requires: consecutive turns of the same role are merged.

---

## db_manager.py

_No module description._

**Imports**:
`glob`, `logging`, `os`, `re`, `typing.Any`, `duckdb`, `pandas`, `core.workspace.UserScope`, `manage.iam.providers.local_duckdb.LocalDuckDBIAMProvider`, `manage.iam.schemas.UserCredentials`, `manage.iam.schemas.UserRole`, `utils.data_paths.get_datasets_dir`, `utils.data_paths.get_stats_dir`, `utils.session_scope.get_current_scope`

### class `DBManager` (db_manager.py)

Helper class to manage DuckDB connections and data persistence using Parquet with versioning

**Methods:**

- **__init__**(`self, db_path: str, scope: ComplexType`) -> `None`
  > No description available.

- **_get_user_connection**(`self`) -> `None`
  > Get a new DuckDB connection for users DB.

- **init_user_db**(`self`) -> `None`
  > Initialize the users table with activation support.

- **create_user**(`self, username: str, password: str, auto_activate: bool, role: UserRole`) -> `tuple[ComplexType]`
  > Create a new user (pending approval unless auto_activate is set).

- **verify_user**(`self, username: str, password: str`) -> `tuple[ComplexType]`
  > Verify user credentials and check if account is active.

- **get_all_users**(`self`) -> `list[str]`
  > Get list of all registered usernames.

- **get_all_users_with_status**(`self`) -> `list[dict[ComplexType]]`
  > Get list of all users with their activation status.

- **activate_user**(`self, username: str`) -> `tuple[ComplexType]`
  > Activate a user account (admin only).

- **deactivate_user**(`self, username: str`) -> `tuple[ComplexType]`
  > Deactivate a user account (admin only). Cannot deactivate admin.

- **update_user_password**(`self, username: str, new_password: str`) -> `tuple[ComplexType]`
  > Update a user's password.

- **delete_user**(`self, username: str`) -> `tuple[ComplexType]`
  > Delete a user account. Protects admin from deletion.

- **_get_connection**(`self`) -> `None`
  > Get a new DuckDB connection.

- **_get_next_version**(`self, base_name: str`) -> `int`
  > Determines the next version number for a given base name.

- **save_dataframe**(`self, df: pd.DataFrame, name: str, folder: ComplexType`) -> `tuple[ComplexType]`
  > Persists a pandas DataFrame to a Parquet file via DuckDB (Low level).

- **save_dataset**(`self, df: pd.DataFrame, base_name: str, username: ComplexType`) -> `tuple[ComplexType]`
  > Persists a pandas DataFrame to a versioned Parquet file (row ids and dtypes are kept).

- **load_dataframe**(`self, name: str, folder: ComplexType`) -> `tuple[ComplexType]`
  > Loads a Parquet file into a pandas DataFrame via DuckDB (Low level).

- **load_dataset**(`self, file_name: str`) -> `tuple[ComplexType]`
  > Loads a Parquet file into a pandas DataFrame via DuckDB.
  >
  >
  > **Parameters**:
  >
  > file_name : str
  >     The name of the file to load (without extension)
  >
  >
  > **Returns**:
  >
  > tuple
  >     (DataFrame, message) - The loaded DataFrame and a success/error message

- **lineage_path**(`self, file_name: str`) -> `str`
  > No description available.

- **save_lineage**(`self, file_name: str, lineage: dict[ComplexType]`) -> `None`
  > The lineage sidecar of a saved version (where it comes from: trace, fingerprint, population).

- **load_lineage**(`self, file_name: str`) -> `ComplexType`
  > No description available.

- **get_available_tables**(`self`) -> `list[str]`
  > List available parquet files in the current directory.

- **get_available_datasets**(`self, username: ComplexType`) -> `list[str]`
  > List available parquet files in the current directory, sorted by modification time.
  > Filters by username if provided (unless admin).

- **save_stats**(`self, df: pd.DataFrame, dataset_name: str, stats_type: str`) -> `tuple[ComplexType]`
  > Save statistical results to parquet cache.

- **load_stats**(`self, dataset_name, stats_type`) -> `None`
  > Load statistical results from parquet cache.

- **clear_stats**(`self, dataset_name`) -> `None`
  > Delete cached statistics for a given dataset to force recalculation.

- **clear_all_stats**(`self`) -> `None`
  > Delete ALL cached statistics files.

### def `_sql_literal` (db_manager.py)

- **Arguments**: `value: str`

- **Returns**: `str`

Escape a string for use inside a single-quoted DuckDB literal (COPY ... TO does not bind parameters).

### def `write_parquet` (db_manager.py)

- **Arguments**: `df: pd.DataFrame, path: str`

- **Returns**: `None`

Write a dataset version: pyarrow keeps the dtypes and the row ids (the ``__row_id`` index); DuckDB
is the fallback for the rare column pyarrow refuses (the ids then travel as a ``__row_id`` column).

---

## disabled.py

_Disabled IAM Provider (Bypass / No-Auth Mode).

Allows seamless operation in single-researcher offline workstations or automated
pipelines where user authentication is explicitly disabled. Automatically provides
a pre-authorized administrator session without gating the application._

**Imports**:
`base.BaseIAMProvider`, `schemas.AuthResult`, `schemas.UserCredentials`, `schemas.UserProfile`, `schemas.UserRole`

### class `DisabledIAMProvider` (disabled.py)

Bypass provider that allows unrestricted access as a local researcher.

**Methods:**

- **__init__**(`self, default_username: str`) -> `None`
  > No description available.

- **provider_name**(`self`) -> `str`
  > Return provider identifier.

- **is_enabled**(`self`) -> `bool`
  > Authentication is disabled.

- **authenticate**(`self, credentials: UserCredentials`) -> `AuthResult`
  > Always return active authorized session.

- **create_user**(`self, username: str, password: str, role: UserRole, auto_activate: bool`) -> `tuple[ComplexType]`
  > No-op user provisioning.

- **verify_session**(`self, token: str`) -> `AuthResult`
  > Always validate token.

- **get_user**(`self, username: str`) -> `ComplexType`
  > Return default user profile.

- **list_users**(`self`) -> `list[UserProfile]`
  > Return default user.

- **activate_user**(`self, username: str`) -> `tuple[ComplexType]`
  > No description available.

- **deactivate_user**(`self, username: str`) -> `tuple[ComplexType]`
  > No description available.

- **delete_user**(`self, username: str`) -> `tuple[ComplexType]`
  > No description available.

- **update_password**(`self, username: str, new_password: str`) -> `tuple[ComplexType]`
  > No description available.

---

## document_index.py

_Engine-agnostic document index used by the RAG code.

:class:`DocumentIndex` wraps a :class:`~manage.vectorstores.base.BaseVectorIndex` and the embedding
model. It exposes the LangChain vector-store calls the application uses (``similarity_search``,
``max_marginal_relevance_search``, ``as_retriever``), so the RAG code is identical whatever the
engine. Filtering by document is expressed as a list of sources; the legacy Chroma filter
(``{"source": x}`` or ``{"source": {"$in": [...]}}``) is still accepted._

**Imports**:
`__future__.annotations`, `hashlib`, `time`, `collections.abc.Callable`, `collections.abc.Sequence`, `typing.Any`, `numpy`, `langchain_core.callbacks.CallbackManagerForRetrieverRun`, `langchain_core.documents.Document`, `langchain_core.embeddings.Embeddings`, `langchain_core.retrievers.BaseRetriever`, `langchain_core.vectorstores.utils.maximal_marginal_relevance`, `pydantic.ConfigDict`, `base.BaseVectorIndex`, `base.ChunkRow`, `base.Hit`, `base.IndexManifest`, `base.VectorStoreError`

### class `DocumentIndex` (document_index.py)

Chunks of the user's documents in the selected engine, searched with the selected embedding model.

**Methods:**

- **__init__**(`self, backend: BaseVectorIndex, embeddings: Embeddings`) -> `None`
  > No description available.

- **engine**(`self`) -> `str`
  > No description available.

- **location**(`self`) -> `str`
  > No description available.

- **manifest**(`self`) -> `ComplexType`
  > No description available.

- **count**(`self`) -> `int`
  > No description available.

- **_embed**(`self, texts: list[str], retries: int, first: int`) -> `list[list[float]]`
  > No description available.

- **build**(`self, chunks: Sequence[Document]`) -> `IndexManifest`
  > Embed every chunk and replace the index; returns (and stores) the manifest.

- **check_compatible**(`self, embedding_label: str`) -> `None`
  > Refuse to reuse an index built with another embedding model (its vectors are not comparable).

- **similarity_search**(`self, query: str, k: int, sources: ComplexType`) -> `list[Document]`
  > No description available.

- **max_marginal_relevance_search**(`self, query: str, k: int, fetch_k: int, lambda_mult: float, sources: ComplexType`) -> `list[Document]`
  > Relevant and diverse chunks: the ``k`` best of the ``fetch_k`` nearest, penalising near-duplicates.

- **as_retriever**(`self, search_type: str, search_kwargs: ComplexType`) -> `None`
  > No description available.

### class `IndexRetriever` (document_index.py)

LangChain retriever over a :class:`DocumentIndex` (plain similarity or MMR).

**Methods:**

- **_get_relevant_documents**(`self, query: str`) -> `list[Document]`
  > No description available.

### def `_sources_from_filter` (document_index.py)

- **Arguments**: `filter_: Any`

- **Returns**: `ComplexType`

Translate the legacy Chroma filter on ``source`` into a list of sources.

### def `_page` (document_index.py)

- **Arguments**: `value: Any`

- **Returns**: `int`

No description available.

### def `chunk_id` (document_index.py)

- **Arguments**: `document: Document, position: int`

- **Returns**: `str`

Stable identifier of a chunk: same document, page, position and text give the same id.

### def `_to_document` (document_index.py)

- **Arguments**: `hit: Hit`

- **Returns**: `Document`

No description available.

---

## document_loader.py

_High-Performance PDF Document Loader.

Replaces deprecated `langchain_community.document_loaders` with native `pypdf` extraction,
improving performance, eliminating deprecation warnings, and resiliently handling errors._

**Imports**:
`logging`, `pathlib.Path`, `langchain_core.documents.Document`, `pypdf.PdfReader`

### class `PDFDocumentLoader` (document_loader.py)

Lightweight, native PDF document loader using pypdf.

Converts PDF pages into standardized `langchain_core.documents.Document`
objects with complete metadata (source, filename, page, total_pages).

**Methods:**

- **__init__**(`self, file_or_dir_path: ComplexType`) -> `None`
  > Initialize the loader with a file or directory path.
  >
  > Args:
  >     file_or_dir_path: Path to a single PDF or a directory containing PDFs.

- **load_single_pdf**(`self, file_path: ComplexType`) -> `list[Document]`
  > Extract text from an individual PDF file page by page.
  >
  > Args:
  >     file_path: Absolute or relative path to the PDF file.
  >
  > Returns:
  >     List of Document objects representing non-empty pages.

- **load**(`self, recursive: bool`) -> `list[Document]`
  > Load all PDF documents from the initialized path.
  >
  > Args:
  >     recursive: If True and path is a directory, searches subdirectories.
  >
  > Returns:
  >     Aggregated list of Document objects from all discovered PDFs.

---

## dossier.py

_The study dossier (Word): what a methods section and its appendix need, built from the session's records.

Sections:

1. Provenance: the dataset, its source file and hash, the parse options, the data fingerprint, the trace and
   its integrity, the software versions;
2. Analysis population and participant flow (STROBE item 13): the flow diagram, the exclusions with the
   criteria behind them, and the definition of every criterion;
3. Data handling: the recorded steps that changed the data (merges, derived variables, conversions,
   clipping, imputations);
4. Missing data: missing values per variable, values filled by single imputation, and the multiple imputation
   with its methods paragraph (reporting items of Sterne et al. 2009);
5. Results: each result added to the dossier with what it was computed on;
6. References.

The input is plain data (:class:`DossierInput`); the app gathers it (app_pages/study_dossier.py). The dossier
supports these reporting guidelines; it does not certify compliance, and it is not an electronically signed record._

**Imports**:
`__future__.annotations`, `dataclasses.dataclass`, `dataclasses.field`, `datetime.datetime`, `io.BytesIO`, `typing.Any`, `pandas`, `docx.Document`, `docx.shared.Cm`, `docx.shared.Pt`, `docx.shared.RGBColor`, `core.attrition.FlowExit`, `core.attrition.ParticipantFlow`, `manage.report_tables.ResultBlocks`

### class `DossierResult` (dossier.py)

No description available.

### class `DossierInput` (dossier.py)

No description available.

### def `_small` (dossier.py)

- **Arguments**: `paragraph: Any, text: str`

- **Returns**: `None`

No description available.

### def `_table` (dossier.py)

- **Arguments**: `doc: Any, frame: pd.DataFrame`

- **Returns**: `None`

No description available.

### def `_pairs` (dossier.py)

- **Arguments**: `doc: Any, rows: list[tuple[ComplexType]]`

- **Returns**: `None`

No description available.

### def `_exclusions` (dossier.py)

- **Arguments**: `flow: ParticipantFlow`

- **Returns**: `pd.DataFrame`

No description available.

### def `build_dossier` (dossier.py)

- **Arguments**: `inp: DossierInput`

- **Returns**: `BytesIO`

No description available.

### def `exits_text` (dossier.py)

- **Arguments**: `exit_: FlowExit`

- **Returns**: `str`

One exclusion in words, for captions.

---

## factory.py

_Connector Factory for RAG LLMs and Embeddings.

Instantiates provider-specific connectors based on ProviderConfig or UI parameters,
providing a single entry point for model-agnostic execution._

**Imports**:
`logging`, `typing.Any`, `base.BaseEmbeddingConnector`, `base.BaseLLMConnector`, `providers.claude.ClaudeChatConnector`, `providers.gemini.GeminiChatConnector`, `providers.gemini.GeminiEmbeddingConnector`, `providers.mistral.MistralChatConnector`, `providers.mistral.MistralEmbeddingConnector`, `providers.ollama.OllamaChatConnector`, `providers.ollama.OllamaEmbeddingConnector`, `providers.openai.OpenAIChatConnector`, `providers.openai.OpenAIEmbeddingConnector`, `schemas.ProviderConfig`, `schemas.ProviderType`

### class `ConnectorFactory` (factory.py)

Factory for creating LLM and Embedding connectors.

**Methods:**

- **get_llm_connector**(`cls, config: ProviderConfig`) -> `BaseLLMConnector`
  > Instantiate the appropriate LLM chat connector.
  >
  > Args:
  >     config: Provider configuration specification.
  >
  > Returns:
  >     Instance of BaseLLMConnector.

- **get_embedding_connector**(`cls, config: ProviderConfig`) -> `BaseEmbeddingConnector`
  > Instantiate the appropriate Embedding connector.
  >
  > Args:
  >     config: Provider configuration specification.
  >
  > Returns:
  >     Instance of BaseEmbeddingConnector.

- **create_connectors**(`cls, provider: str, model_name: ComplexType, embedding_model: ComplexType, api_key: ComplexType, base_url: ComplexType, temperature: float, embedding_provider: ComplexType, embedding_api_key: ComplexType, embedding_base_url: ComplexType`) -> `tuple[ComplexType]`
  > Convenience factory method constructing both connectors from keyword arguments.
  >
  > ``embedding_provider`` lets the documents be embedded by another provider than the one that answers;
  > it is required in practice for Claude (Anthropic has no embedding API) and defaults to local Ollama
  > there. ``embedding_api_key`` / ``embedding_base_url`` configure that provider.

---

## file_handling.py

_No module description._

**Imports**:
`logging`, `os`, `pandas`, `detect_delimiter.detect`, `utils.logging_config.single_line`

### def `create_folders` (file_handling.py)

- **Arguments**: `config`

- **Returns**: `None`

Create folders based on the configuration.

Parameters:
    config (dict): Configuration dictionary containing folder names.

Returns:
    None

### def `list_files_in_directory` (file_handling.py)

- **Arguments**: `directory, file_extensions`

- **Returns**: `None`

Lists files with specified extensions in a directory.

Parameters:
- directory (str): The path to the directory.

- file_extensions (list of str, optional): List of file extensions to filter. Default is None.

Returns:
- list of str: List of file paths that match the specified extensions.

If no file_extensions are provided, all files in the directory are listed.

### def `detect_encoding` (file_handling.py)

- **Arguments**: `file_path, num_lines`

- **Returns**: `None`

Detects the encoding of a file using charset-normalizer.

Parameters:
- file_path (str): The path to the file.

- num_lines (int): The number of lines to read from the file for encoding detection. Default is 100.

Returns:
- str: The detected encoding of the file.

Note:
The function attempts to detect the encoding of the file by analyzing a portion of its content.
It reads the specified number of lines and uses the charset-normalizer library to determine the encoding.

### def `load_csv_with_separator` (file_handling.py)

- **Arguments**: `file_path, encoding, num_lines`

- **Returns**: `None`

Load a CSV file with automatic delimiter detection.

- file_path (str): The path to the CSV file.

- encoding (str): The encoding of the file.

- num_lines (int): The number of lines to read for delimiter detection. Default is 10.

Returns:
- pd.DataFrame: The loaded DataFrame.

### def `load_dataframe` (file_handling.py)

- **Arguments**: `file_path, encoding`

- **Returns**: `None`

Loads a DataFrame from a file.

Parameters:
- file_path (str): The path to the file to load.

- encoding (str, optional): The encoding of the file. Default is 'utf-8'.

Returns:
- DataFrame or None: The loaded DataFrame if successful, otherwise None.

This function attempts to load a DataFrame from the specified file. It first detects the file extension
to determine the file type. For CSV files, it uses a custom function to handle loading with the specified
encoding. For Excel files, it uses pandas read_excel method. If the file format is unsupported or an error
occurs during loading, it logs an error message and returns None.

### def `auto_name` (file_handling.py)

- **Arguments**: `input_file_path, config, output_extension, descriptive`

- **Returns**: `None`

Generate an output file path based on the input file path.

Parameters:
    input_file_path (str): The path to the input file.
    output_extension (str): The extension of the output file. Default is 'csv'.
    descriptive (bool): Whether to use a descriptive folder. Default is False.

Returns:
    str: The output file path.

Raises:
    None

---

## flow_figure.py

_The participant flow as a static picture (PNG) for the Word dossier: the same boxes and geometry as the app's
diagram (core.attrition.flow_layout), drawn with matplotlib's object API (no pyplot state, safe in a server)._

**Imports**:
`__future__.annotations`, `io.BytesIO`, `textwrap.wrap`, `matplotlib.figure.Figure`, `matplotlib.patches.FancyBboxPatch`, `core.attrition.ANALYSIS`, `core.attrition.POPULATION`, `core.attrition.FlowExit`, `core.attrition.ParticipantFlow`, `core.attrition.flow_layout`

### def `flow_png` (flow_figure.py)

- **Arguments**: `flow: ParticipantFlow`

- **Returns**: `bytes`

No description available.

---

## gemini.py

_Google Gemini Provider Connectors.

Utilizes the modern `google-genai` SDK (v2.x) with retry logic,
temperature controls, and embedding integration._

**Imports**:
`logging`, `random`, `time`, `typing.Any`, `langchain_core.callbacks.manager.CallbackManagerForLLMRun`, `langchain_core.messages.AIMessage`, `langchain_core.messages.BaseMessage`, `langchain_core.messages.HumanMessage`, `langchain_core.messages.SystemMessage`, `langchain_core.outputs.ChatGeneration`, `langchain_core.outputs.ChatResult`, `utils.logging_config.redact_secrets`, `base.BaseEmbeddingConnector`, `base.BaseLLMConnector`, `schemas.ProviderConfig`

### class `GeminiChatConnector` (gemini.py)

Chat connector for Google Gemini models via google-genai SDK.

**Methods:**

- **__init__**(`self, config: ProviderConfig`) -> `None`
  > No description available.

- **_llm_type**(`self`) -> `str`
  > No description available.

- **model_name**(`self`) -> `str`
  > No description available.

- **get_available_models**(`self`) -> `list[str]`
  > Query Gemini API for available text models.

- **_generate**(`self, messages: list[BaseMessage], stop: ComplexType, run_manager: ComplexType`) -> `ChatResult`
  > No description available.

### class `GeminiEmbeddingConnector` (gemini.py)

Embedding connector for Google Gemini models.

**Methods:**

- **__init__**(`self, config: ProviderConfig`) -> `None`
  > No description available.

- **get_available_models**(`self`) -> `list[str]`
  > No description available.

- **_embed_single**(`self, text: str`) -> `list[float]`
  > Embeds single text with automatic fallback across supported Gemini models.

- **embed_documents**(`self, texts: list[str]`) -> `list[list[float]]`
  > No description available.

- **embed_query**(`self, text: str`) -> `list[float]`
  > No description available.

---

## lancedb_index.py

_LanceDB engine: embedded, serverless, columnar storage on disk (Lance format).

Vectors live in files under the user's workspace, so memory use stays low and nothing has to run
beside the application. Search is exact (brute force, cosine) up to :data:`ANN_THRESHOLD` chunks;
beyond that :meth:`LanceDBIndex.optimize` builds an IVF index for approximate search._

**Imports**:
`__future__.annotations`, `logging`, `pathlib.Path`, `typing.Any`, `base.BaseVectorIndex`, `base.ChunkRow`, `base.Hit`, `base.VectorEngine`, `base.VectorStoreError`

### class `LanceDBIndex` (lancedb_index.py)

No description available.

**Methods:**

- **__init__**(`self, directory: Path`) -> `None`
  > No description available.

- **availability**(`cls`) -> `tuple[ComplexType]`
  > No description available.

- **location**(`self`) -> `str`
  > No description available.

- **_connect**(`self`) -> `None`
  > No description available.

- **_tables**(`self`) -> `list[str]`
  > No description available.

- **exists**(`self`) -> `bool`
  > No description available.

- **_table**(`self`) -> `None`
  > No description available.

- **reset**(`self, dim: int`) -> `None`
  > No description available.

- **add**(`self, rows: list[ChunkRow]`) -> `None`
  > No description available.

- **query**(`self, vector: list[float], k: int, sources: ComplexType`) -> `list[Hit]`
  > No description available.

- **count**(`self`) -> `int`
  > No description available.

- **optimize**(`self`) -> `None`
  > No description available.

- **close**(`self`) -> `None`
  > No description available.

### def `_sql_literal` (lancedb_index.py)

- **Arguments**: `value: str`

- **Returns**: `str`

No description available.

---

## local_duckdb.py

_Local DuckDB IAM Provider.

Default, zero-dependency embedded authentication backend utilizing DuckDB
and bcrypt password hashing. Designed for single-node deployments, air-gapped
clinical environments, and full backward compatibility._

**Imports**:
`logging`, `os`, `datetime.datetime`, `datetime.timedelta`, `typing.Any`, `typing.cast`, `bcrypt`, `duckdb`, `utils.data_paths`, `utils.logging_config.single_line`, `base.BaseIAMProvider`, `schemas.AuthResult`, `schemas.UserCredentials`, `schemas.UserProfile`, `schemas.UserRole`

### class `LocalDuckDBIAMProvider` (local_duckdb.py)

Local embedded IAM backend using DuckDB and bcrypt.

**Methods:**

- **__init__**(`self, db_path: ComplexType`) -> `None`
  > Initialize the local DuckDB IAM provider.
  >
  > Args:
  >     db_path: Optional custom path to users.duckdb file. If None,
  >              resolves via standard data paths.

- **provider_name**(`self`) -> `str`
  > Return provider identifier.

- **_get_connection**(`self`) -> `ComplexType`
  > Acquire an on-demand DuckDB connection.
  > Closes explicitly per operation to prevent Windows file lock conflicts.

- **_init_database**(`self`) -> `None`
  > Initialize user directory table and apply schema migrations.

- **authenticate**(`self, credentials: UserCredentials`) -> `AuthResult`
  > Authenticate local credentials against bcrypt password hashes.

- **create_user**(`self, username: str, password: str, role: UserRole, auto_activate: bool`) -> `tuple[ComplexType]`
  > Provision a new local user with bcrypt password hashing.

- **verify_session**(`self, token: str`) -> `AuthResult`
  > Validate local session token.

- **_row_padded**(`raw: Any`) -> `tuple[ComplexType]`
  > username, is_active, role, created_at (missing trailing columns become None).

- **get_user**(`self, username: str`) -> `ComplexType`
  > Retrieve user profile from local database.

- **list_users**(`self`) -> `list[UserProfile]`
  > List all users in local database.

- **activate_user**(`self, username: str`) -> `tuple[ComplexType]`
  > Activate a user account.

- **_is_admin_account**(`self, username: str`) -> `bool`
  > True when the account exists and holds the administrator role.

- **deactivate_user**(`self, username: str`) -> `tuple[ComplexType]`
  > Deactivate a user account (administrator accounts are protected).

- **delete_user**(`self, username: str`) -> `tuple[ComplexType]`
  > Delete a user account (administrator accounts are protected).

- **update_password**(`self, username: str, new_password: str`) -> `tuple[ComplexType]`
  > Update password for an existing user.

---

## manager.py

_IAM Manager and Factory.

Provides a unified, framework-agnostic facade for identity and access management.
Dynamically resolves and instantiates the active IAM provider (Local DuckDB,
Better Auth, or Disabled) based on environment configuration or runtime parameters._

**Imports**:
`logging`, `os`, `typing.Any`, `utils.logging_config.single_line`, `.setup_token`, `base.BaseIAMProvider`, `providers.better_auth.BetterAuthIAMProvider`, `providers.disabled.DisabledIAMProvider`, `providers.local_duckdb.LocalDuckDBIAMProvider`, `schemas.AuthResult`, `schemas.IAMConfig`, `schemas.IAMProviderType`, `schemas.UserCredentials`, `schemas.UserProfile`, `schemas.UserRole`

### class `IAMManager` (manager.py)

Central facade managing IAM operations and provider lifecycle.

**Methods:**

- **__init__**(`self, config: ComplexType`) -> `None`
  > Initialize IAM Manager with configuration or environment variables.

- **_resolve_config_from_env**(`cls`) -> `IAMConfig`
  > Read IAM configuration from process environment variables.

- **_build_provider**(`cls, config: IAMConfig`) -> `BaseIAMProvider`
  > Construct the designated provider instance.

- **provider**(`self`) -> `BaseIAMProvider`
  > Get the active IAM provider instance.

- **is_enabled**(`self`) -> `bool`
  > Whether authentication gating is active.

- **provider_name**(`self`) -> `str`
  > Name of the active provider.

- **set_provider**(`self, provider: BaseIAMProvider`) -> `None`
  > Manually inject a custom or mock IAM provider.

- **reset_provider**(`self`) -> `None`
  > Reset provider to default environment configuration.

- **authenticate**(`self, credentials: UserCredentials`) -> `AuthResult`
  > Authenticate user credentials.

- **verify_user**(`self, username: str, password: str`) -> `tuple[ComplexType]`
  > Convenience verification method returning (success, message).
  > Maintains complete drop-in compatibility with legacy DBManager calls.

- **create_user**(`self, username: str, password: str, role: UserRole, auto_activate: bool`) -> `tuple[ComplexType]`
  > Provision a new user account.

- **verify_session**(`self, token: str`) -> `AuthResult`
  > Verify an existing session token.

- **get_user**(`self, username: str`) -> `ComplexType`
  > Fetch user profile.

- **list_users**(`self`) -> `list[UserProfile]`
  > List registered user profiles.

- **get_all_users**(`self`) -> `list[str]`
  > Return list of all registered usernames.

- **get_all_users_with_status**(`self`) -> `list[dict[ComplexType]]`
  > Return structured dictionary list of all users with status.
  > Maintains complete drop-in compatibility with legacy DBManager calls.

- **activate_user**(`self, username: str`) -> `tuple[ComplexType]`
  > Activate user account.

- **deactivate_user**(`self, username: str`) -> `tuple[ComplexType]`
  > Deactivate user account.

- **delete_user**(`self, username: str`) -> `tuple[ComplexType]`
  > Delete user account.

- **update_user_password**(`self, username: str, new_password: str`) -> `tuple[ComplexType]`
  > Update user credentials.

- **_setup_dir**(``) -> `str`
  > No description available.

- **_has_active_admin**(`self`) -> `bool`
  > No description available.

- **setup_pending**(`self`) -> `bool`
  > True when the first administrator must be created with the setup token of the server log.

- **issue_setup_token**(`self`) -> `ComplexType`
  > A new first-run setup token (the previous one stops working); None when no setup is needed.

- **create_first_admin**(`self, token: str, username: str, password: str`) -> `tuple[ComplexType]`
  > Create the first administrator with the one-time setup token (follow-up programme M4).

- **ensure_bootstrap_admin**(`self`) -> `tuple[ComplexType]`
  > Create the initial administrator when no active administrator exists.
  >
  > Credentials come from CMP_ADMIN_USERNAME (default 'admin') and CMP_ADMIN_PASSWORD or
  > CMP_ADMIN_PASSWORD_FILE. Without them, a one-time setup token is written to the server log and the
  > login page creates the first administrator with it (manage/iam/setup_token.py). Nothing happens when
  > the IAM provider is disabled or when an active administrator already exists.

### def `get_iam_manager` (manager.py)

- **Arguments**: `config: ComplexType`

- **Returns**: `IAMManager`

Retrieve or initialize the global IAMManager instance.

---

## mi_store.py

_Files of a multiple imputation saved with a dataset version (in the ``mi/`` folder of the datasets).

``<version>.cells.parquet``  the imputed cells: row id, column, imputation number, value (number or text)
``<version>.rows.parquet``   row id, hash of the model columns at fitting time, stratum (binding)
``<version>.json``           the binding (spec, methods, population), the engine information, diagnostics

The observed data are the saved version itself; together they rebuild the ``MIData`` and its status
against the current analysis population. Writes are atomic (temporary file, then replace)._

**Imports**:
`__future__.annotations`, `json`, `os`, `pathlib.Path`, `typing.Any`, `numpy`, `pandas`, `biostats.mi.binding.MIBinding`, `biostats.mi.binding.StoredMI`, `biostats.mi.data.MIData`

### def `mi_dir` (mi_store.py)

- **Arguments**: `dataset_dir: ComplexType`

- **Returns**: `Path`

No description available.

### def `_paths` (mi_store.py)

- **Arguments**: `dataset_dir: ComplexType, name: str`

- **Returns**: `dict[ComplexType]`

No description available.

### def `_atomic` (mi_store.py)

- **Arguments**: `path: Path, write`

- **Returns**: `None`

No description available.

### def `_jsonable` (mi_store.py)

- **Arguments**: `value: Any`

- **Returns**: `Any`

No description available.

### def `save_mi` (mi_store.py)

- **Arguments**: `stored: StoredMI, dataset_dir: ComplexType, name: str`

- **Returns**: `list[Path]`

Write the three files of ``stored`` for the saved version ``name``.

### def `has_mi` (mi_store.py)

- **Arguments**: `dataset_dir: ComplexType, name: str`

- **Returns**: `bool`

No description available.

### def `load_mi` (mi_store.py)

- **Arguments**: `dataset_dir: ComplexType, name: str, base: pd.DataFrame`

- **Returns**: `ComplexType`

Rebuild the imputation saved with version ``name`` on its observed data ``base`` (None if absent).

### def `delete_mi` (mi_store.py)

- **Arguments**: `dataset_dir: ComplexType, name: str`

- **Returns**: `None`

No description available.

---

## mistral.py

_Mistral AI Provider Connectors.

Integrates European sovereign models (Mistral Small, Mistral Large, Codestral)
and Mistral Embeddings via Mistral OpenAI-compatible API._

**Imports**:
`logging`, `typing.Any`, `langchain_core.callbacks.manager.CallbackManagerForLLMRun`, `langchain_core.messages.AIMessage`, `langchain_core.messages.BaseMessage`, `langchain_core.messages.HumanMessage`, `langchain_core.messages.SystemMessage`, `langchain_core.outputs.ChatGeneration`, `langchain_core.outputs.ChatResult`, `utils.logging_config.redact_secrets`, `base.BaseEmbeddingConnector`, `base.BaseLLMConnector`, `schemas.ProviderConfig`

### class `MistralChatConnector` (mistral.py)

Chat connector for Mistral AI sovereign models.

**Methods:**

- **__init__**(`self, config: ProviderConfig`) -> `None`
  > No description available.

- **_llm_type**(`self`) -> `str`
  > No description available.

- **get_available_models**(`self`) -> `list[str]`
  > No description available.

- **_generate**(`self, messages: list[BaseMessage], stop: ComplexType, run_manager: ComplexType`) -> `ChatResult`
  > No description available.

### class `MistralEmbeddingConnector` (mistral.py)

Embedding connector for Mistral AI embeddings (mistral-embed).

**Methods:**

- **__init__**(`self, config: ProviderConfig`) -> `None`
  > No description available.

- **get_available_models**(`self`) -> `list[str]`
  > No description available.

- **embed_documents**(`self, texts: list[str]`) -> `list[list[float]]`
  > No description available.

- **embed_query**(`self, text: str`) -> `list[float]`
  > No description available.

---

## ollama.py

_Ollama Local & Open-Source Provider Connectors.

Enables 100% offline, privacy-preserving, sovereign execution using locally hosted
open-source models (Llama 3.2, Mistral, DeepSeek, Qwen) and local embeddings
(Nomic Embed Text, BGE-M3, etc.) via Ollama REST API._

**Imports**:
`logging`, `typing.Any`, `httpx`, `langchain_core.callbacks.manager.CallbackManagerForLLMRun`, `langchain_core.messages.AIMessage`, `langchain_core.messages.BaseMessage`, `langchain_core.messages.HumanMessage`, `langchain_core.messages.SystemMessage`, `langchain_core.outputs.ChatGeneration`, `langchain_core.outputs.ChatResult`, `utils.logging_config.redact_secrets`, `base.BaseEmbeddingConnector`, `base.BaseLLMConnector`

### class `OllamaChatConnector` (ollama.py)

Chat connector for local Ollama instances via native REST API.
Zero cloud dependencies, completely offline and private.

**Methods:**

- **_llm_type**(`self`) -> `str`
  > No description available.

- **base_url**(`self`) -> `str`
  > No description available.

- **get_available_models**(`self`) -> `list[str]`
  > Query the local Ollama server for currently downloaded models.
  > Falls back to recommended open-source models if Ollama is unreachable.

- **_generate**(`self, messages: list[BaseMessage], stop: ComplexType, run_manager: ComplexType`) -> `ChatResult`
  > No description available.

### class `OllamaEmbeddingConnector` (ollama.py)

Embedding connector for local Ollama embedding models.
Supports high-performance embeddings like nomic-embed-text, bge-m3, all-minilm.

**Methods:**

- **base_url**(`self`) -> `str`
  > No description available.

- **get_available_models**(`self`) -> `list[str]`
  > No description available.

- **_call_embed**(`self, input_data: Any`) -> `Any`
  > Call either /api/embed or fallback to /api/embeddings.

- **embed_documents**(`self, texts: list[str]`) -> `list[list[float]]`
  > No description available.

- **embed_query**(`self, text: str`) -> `list[float]`
  > No description available.

---

## openai.py

_OpenAI Provider Connectors.

Integrates OpenAI chat completion models (GPT-4o, GPT-4o-mini) and embeddings
(text-embedding-3-small, text-embedding-3-large) using official OpenAI client._

**Imports**:
`logging`, `typing.Any`, `langchain_core.callbacks.manager.CallbackManagerForLLMRun`, `langchain_core.messages.AIMessage`, `langchain_core.messages.BaseMessage`, `langchain_core.messages.HumanMessage`, `langchain_core.messages.SystemMessage`, `langchain_core.outputs.ChatGeneration`, `langchain_core.outputs.ChatResult`, `utils.logging_config.redact_secrets`, `base.BaseEmbeddingConnector`, `base.BaseLLMConnector`, `schemas.ProviderConfig`

### class `OpenAIChatConnector` (openai.py)

Chat connector for OpenAI API models (GPT-4o, GPT-4o-mini, o3-mini).

**Methods:**

- **__init__**(`self, config: ProviderConfig`) -> `None`
  > No description available.

- **_llm_type**(`self`) -> `str`
  > No description available.

- **get_available_models**(`self`) -> `list[str]`
  > No description available.

- **_generate**(`self, messages: list[BaseMessage], stop: ComplexType, run_manager: ComplexType`) -> `ChatResult`
  > No description available.

### class `OpenAIEmbeddingConnector` (openai.py)

Embedding connector for OpenAI embeddings (text-embedding-3-small, text-embedding-3-large).

**Methods:**

- **__init__**(`self, config: ProviderConfig`) -> `None`
  > No description available.

- **get_available_models**(`self`) -> `list[str]`
  > No description available.

- **embed_documents**(`self, texts: list[str]`) -> `list[list[float]]`
  > No description available.

- **embed_query**(`self, text: str`) -> `list[float]`
  > No description available.

---

## qdrant_index.py

_Qdrant engine, in two modes.

- **Server** (``CMP_QDRANT_URL``, optional ``CMP_QDRANT_API_KEY``): a Qdrant service shared by every
  session, with HNSW approximate search, payload indexes and concurrent access. One collection
  per user (``cmp_<user>_chunks``). Best for several users or very large libraries.

- **Embedded** (no URL): Qdrant's local mode in the user's workspace. Every vector is held in
  memory and a folder can be opened by one client only, so the client is shared per folder in this
  process. Suited to small and medium libraries (tens of thousands of chunks).

Every rebuild writes a new collection (``<base>_<id>``) and then drops the previous one; the active
name is kept in ``collection.txt`` next to the manifest. Embedded Qdrant on Windows can reload a
deleted collection's files when the same name is recreated, so names are never reused._

**Imports**:
`__future__.annotations`, `json`, `re`, `threading`, `uuid`, `pathlib.Path`, `typing.Any`, `base.BaseVectorIndex`, `base.ChunkRow`, `base.Hit`, `base.VectorEngine`, `base.VectorStoreError`

### class `QdrantIndex` (qdrant_index.py)

No description available.

**Methods:**

- **__init__**(`self, manifest_dir: Path`) -> `None`
  > No description available.

- **availability**(`cls`) -> `tuple[ComplexType]`
  > No description available.

- **mode**(`self`) -> `str`
  > No description available.

- **location**(`self`) -> `str`
  > No description available.

- **_client**(`self`) -> `None`
  > No description available.

- **ping**(`self`) -> `tuple[ComplexType]`
  > Reachability of the server (or the local folder).

- **active_collection**(`self`) -> `str`
  > Collection holding the current index (the base name for indexes built before generations).

- **exists**(`self`) -> `bool`
  > No description available.

- **reset**(`self, dim: int`) -> `None`
  > No description available.

- **add**(`self, rows: list[ChunkRow]`) -> `None`
  > No description available.

- **query**(`self, vector: list[float], k: int, sources: ComplexType`) -> `list[Hit]`
  > No description available.

- **count**(`self`) -> `int`
  > No description available.

### def `_client` (qdrant_index.py)

- **Arguments**: ``

- **Returns**: `None`

One client per folder (local mode allows a single client) or per server and key.

### def `release_clients` (qdrant_index.py)

- **Arguments**: ``

- **Returns**: `None`

Close every shared client (tests, shutdown).

### def `user_collection` (qdrant_index.py)

- **Arguments**: `username: ComplexType`

- **Returns**: `str`

Collection of a user on a shared Qdrant server.

---

## rag_computed_vars.py

_Computed Variables Mixin for RAGManager.

Contains all computed variable suggestion and validation methods.
This is part of the RAGManager class composition pattern._

**Imports**:
`ast`, `hashlib`, `json`, `logging`, `re`, `utils.llm_utils.ProviderError`, `utils.llm_utils.parse_json_safe`, `utils.llm_utils.provider_error`, `utils.llm_utils.validate_and_parse`, `rag_schemas.AlternativeFormula`, `rag_schemas.ProxyVariable`

### class `ComputedVarsMixin` (rag_computed_vars.py)

Mixin class containing computed variable functionality for RAGManager.

Methods:
    - _expand_search_hint

    - suggest_computed_variables (and _suggest_variables, the pipeline it guards)

    - _get_theoretical_formulas

    - _map_formulas_to_data

    - suggest_computed_variables_with_validation

    - suggest_proxy_variable

    - suggest_alternative_formula

**Methods:**

- **_expand_search_hint**(`self, search_hint`) -> `None`
  > Pre-processing: Expand acronyms or ambiguous terms to standard medical concepts.

- **suggest_computed_variables**(`self, columns, search_hint, num_suggestions, suggestion_mode, allow_missing_variables, use_taxonomy, existing_categories, progress_callback`) -> `None`
  > Suggests computed variables based on dataset columns and medical knowledge.

- **_suggest_variables**(`self, columns, search_hint, num_suggestions, suggestion_mode, allow_missing_variables, use_taxonomy, existing_categories, progress_callback`) -> `None`
  > The suggestions as JSON; raises ProviderError when the model answers with an error.

- **_get_theoretical_formulas**(`self, search_hint, suggestion_mode, context_text, count, available_columns, use_taxonomy`) -> `None`
  > Step 1: Identify WHAT to calculate (Standard Medical Knowledge).

- **_map_formulas_to_data**(`self, concepts, columns, allow_missing, limit, use_taxonomy, existing_categories`) -> `None`
  > Step 2: Map theoretical inputs to actual dataset columns.

- **suggest_computed_variables_with_validation**(`self, columns, search_hint, num_suggestions, suggestion_mode, allow_missing_variables, use_taxonomy, existing_categories, progress_callback`) -> `None`
  > Wrapper that calls the pipeline and performs final AST validation with Auto-Correction.

- **suggest_proxy_variable**(`self, target_variable, available_columns`) -> `None`
  > Suggests a proxy variable when the target is missing.

- **suggest_alternative_formula**(`self, target_concept, missing_variable, available_columns`) -> `None`
  > Suggests an alternative formula avoiding a missing variable.

- **suggest_imputation_formulas**(`self, target_variable, available_columns, search_hint, context_columns, context_stats, num_suggestions`) -> `None`
  > Suggests formulas to impute missing values for a specific target variable
  > based on relationships with other available columns.
  >
  > Parameters:
  > - target_variable (str): Name of the variable to impute.
  > - available_columns (list): List of all available column names.
  > - search_hint (str): Optional user-provided hint (e.g. "Reverse BMI").
  > - context_columns (list): Optional list of specific columns selected by the user.
  > - context_stats (dict): Optional dictionary of statistics for the context columns.
  > - num_suggestions (int): Number of suggestions to generate.

### def `parse_formula_vars` (rag_computed_vars.py)

- **Arguments**: `formula_str: str`

- **Returns**: `tuple[ComplexType]`

Parses variable identifiers from a formula expression using Python AST.

Handles:
- Standard variable identifiers (e.g., `weight`, `height`)

- Power operator normalization (`^` to `**`)

- Escaped column names enclosed in double quotes (e.g., `""Systolic BP""`)

- Column names enclosed in backticks (e.g., `` `Diastolic BP` ``)

Args:
    formula_str: Mathematical expression string to parse.

Returns:
    tuple[set[str], bool]: Set of extracted variable names and syntax validity flag.

---

## rag_documents.py

_Documents Mixin for RAGManager.

Contains all document processing, knowledge graph extraction and chat methods.
This is part of the RAGManager class composition pattern._

**Imports**:
`os`, `time`

### class `DocumentsMixin` (rag_documents.py)

Mixin class containing document processing functionality for RAGManager.

Methods:
    - get_available_documents

    - extract_custom_graph_from_doc

    - extract_merged_graph_from_docs

    - match_columns_to_graph

    - chat_with_specific_doc

**Methods:**

- **get_available_documents**(`self`) -> `None`
  > Returns a list of PDF files in the documents directory.

- **extract_custom_graph_from_doc**(`self, file_name, progress_callback`) -> `None`
  > Extract a Knowledge Graph of key entities and relationships from a specific document.
  > Supports both Gemini File API and universal text-based LLM extraction (Ollama, OpenAI, Mistral).

- **extract_merged_graph_from_docs**(`self, doc_list, progress_callback`) -> `None`
  > Extracts and merges graphs from multiple documents.
  > Entities are unified across documents and relation endpoints are remapped to the merged
  > entities (graph.builders.document.merge_document_graphs). Returns (merged_json, error).

- **match_columns_to_graph**(`self, columns, graph_json`) -> `None`
  > Matches dataset columns to Knowledge Graph nodes using fuzzy string matching.
  > Returns dict: {column_name: {match_found: bool, node: data, confidence: float}}

- **chat_with_specific_doc**(`self, doc_names, query`) -> `None`
  > Chat with specific document(s) using the vector store.
  > doc_names: string or list of strings.

---

## rag_evaluator.py

_RAG Evaluator Module

Provides comprehensive evaluation metrics for the RAG system including:
- Retrieval quality (context relevance, semantic similarity)

- Generation quality (faithfulness, answer relevance, hallucination detection)

- Embedding quality assessment

- Logging and tracking of evaluation results_

**Imports**:
`hashlib`, `json`, `datetime.datetime`, `pathlib.Path`, `typing.Any`, `prompts.evaluation.detect_hallucination`, `prompts.evaluation.evaluate_answer_relevance`, `prompts.evaluation.evaluate_context_relevance`, `prompts.evaluation.evaluate_faithfulness`, `utils.data_paths.get_rag_eval_logs_dir`

### class `RAGEvaluator` (rag_evaluator.py)

Evaluates RAG system outputs for quality monitoring.

Uses LLM-as-judge approach for semantic evaluation metrics
and direct validation for structural metrics.

**Methods:**

- **__init__**(`self, llm, log_dir: str`) -> `None`
  > Initialize the RAG Evaluator.
  >
  > Args:
  >     llm: The LLM instance to use for evaluation (same as RAG system)
  >     log_dir: Directory to store evaluation logs (defaults to DATA_ROOT/rag_eval_logs)

- **_init_summary**(`self`) -> `None`
  > Initialize the metrics summary file.

- **_save_summary**(`self, summary: dict`) -> `None`
  > Save the metrics summary to file.

- **_load_summary**(`self`) -> `dict`
  > Load the metrics summary from file.

- **_generate_eval_id**(`self, query: str, timestamp: str`) -> `str`
  > Generate a unique evaluation ID.

- **evaluate_retrieval_quality**(`self, query: str, retrieved_chunks: list[str], k: int`) -> `dict[ComplexType]`
  > Evaluate the quality of retrieved context.
  >
  > Args:
  >     query: The original query
  >     retrieved_chunks: List of retrieved text chunks
  >     k: Number of top chunks to consider
  >
  > Returns:
  >     Dictionary with relevance scores and analysis

- **calculate_semantic_similarity**(`self, query_embedding: list[float], chunk_embeddings: list[list[float]]`) -> `dict[ComplexType]`
  > Calculate semantic similarity between query and chunks.
  >
  > Args:
  >     query_embedding: The query's embedding vector
  >     chunk_embeddings: List of chunk embedding vectors
  >
  > Returns:
  >     Dictionary with similarity metrics

- **evaluate_generation_quality**(`self, query: str, context: str, response: str, function_name: str`) -> `dict[ComplexType]`
  > Evaluate the quality of generated response.
  >
  > Args:
  >     query: The original query
  >     context: The retrieved context used
  >     response: The generated response
  >     function_name: Name of the RAG function being evaluated
  >
  > Returns:
  >     Dictionary with quality scores

- **validate_json_output**(`self, response: str`) -> `dict[ComplexType]`
  > Validate if the response contains valid JSON.
  >
  > Args:
  >     response: The generated response
  >
  > Returns:
  >     Dictionary with validation results

- **evaluate**(`self, query: str, context: str, response: str, function_name: str, retrieved_chunks: ComplexType`) -> `dict[ComplexType]`
  > Run full evaluation on a RAG query-response pair.
  >
  > Args:
  >     query: The original query
  >     context: The retrieved context
  >     response: The generated response
  >     function_name: Name of the RAG function
  >     retrieved_chunks: Optional list of individual chunks
  >
  > Returns:
  >     Complete evaluation results

- **log_result**(`self, result: dict[ComplexType]`) -> `None`
  > Log evaluation result to file and update summary.
  >
  > Args:
  >     result: Evaluation result dictionary

- **_parse_evaluation_response**(`self, response: str, metric_type: str`) -> `dict[ComplexType]`
  > Parse LLM evaluation response into structured format.

- **load_test_suite**(`self`) -> `list[dict[ComplexType]]`
  > Load the test suite from file.

- **save_test_suite**(`self, test_cases: list[dict[ComplexType]]`) -> `None`
  > Save test cases to file.

- **add_test_case**(`self, query: str, expected_concepts: list[str], ground_truth: ComplexType, expected_sources: ComplexType`) -> `None`
  > Add a new test case to the suite.

- **run_test_suite**(`self, rag_manager`) -> `dict[ComplexType]`
  > Run all test cases against the RAG system.
  >
  > Args:
  >     rag_manager: The RAGManager instance to test
  >
  > Returns:
  >     Test suite results with pass/fail for each case

- **get_summary**(`self`) -> `dict[ComplexType]`
  > Get the current metrics summary.

- **get_recent_evaluations**(`self, limit: int`) -> `list[dict[ComplexType]]`
  > Get recent evaluation results.

- **get_metrics_by_date**(`self, days: int`) -> `dict[ComplexType]`
  > Get metrics aggregated by date for the last N days.

---

## rag_manager.py

_RAG Manager - Core Module.

This is the main RAGManager class that composes functionality from mixin classes:
- TaxonomyMixin: Variable taxonomy generation and enrichment

- DocumentsMixin: Document processing and knowledge graph extraction

- ComputedVarsMixin: Computed variable suggestions

The mixin pattern allows splitting a large class into focused modules
while maintaining a single class interface for existing code._

**Imports**:
`json`, `os`, `re`, `pathlib.Path`, `pandas`, `rapidfuzz.process`, `prompts.anomaly_criteria_prompt`, `prompts.context_analysis`, `prompts.refine_taxonomy`, `prompts.wise_enrichment`, `utils.llm_utils.ProviderError`, `utils.llm_utils.StructuredOutputHelper`, `utils.llm_utils.clean_json_response`, `utils.llm_utils.parse_json_safe`, `utils.llm_utils.provider_error`, `utils.logging_config.redact_secrets`, `rag_computed_vars.ComputedVarsMixin`, `rag_documents.DocumentsMixin`, `rag_taxonomy.TaxonomyMixin`

### class `RAGManager` (rag_manager.py)

Main RAG Manager class that composes functionality from mixins.

Core Methods (defined here):
    - __init__

    - is_available

    - get_available_models

    - _clean_json_response

    - _analyze_global_context

    - initialize_system

    - update_adherence_score

    - _update_internal_state

Taxonomy Methods (from TaxonomyMixin):
    - suggest_column_renaming

    - generate_variable_taxonomy

    - enrich_variable_taxonomy

    - refine_variable_taxonomy

Document Methods (from DocumentsMixin):
    - get_available_documents

    - extract_custom_graph_from_doc

    - extract_merged_graph_from_docs

    - match_columns_to_graph

    - chat_with_specific_doc

Computed Variable Methods (from ComputedVarsMixin):
    - suggest_computed_variables

    - suggest_computed_variables_with_validation

    - suggest_proxy_variable

    - suggest_alternative_formula

**Methods:**

- **__init__**(`self, documents_dir, api_key, provider, base_url`) -> `None`
  > No description available.

- **is_available**(`self`) -> `None`
  > Check if RAG dependencies are available.

- **get_available_models**(`self, provider: str, base_url: str`) -> `None`
  > Get list of available models for the specified provider.

- **_clean_json_response**(`self, text`) -> `None`
  > Clean JSON output from LLM using robust parsing utilities.
  >
  > This method is a wrapper around llm_utils.clean_json_response
  > for backward compatibility with existing code.

- **_parse_json_safe**(`self, text, default`) -> `None`
  > Safely parse JSON with multiple fallback strategies.
  >
  > Uses llm_utils.parse_json_safe for robust parsing with:
  > - Automatic JSON extraction from markdown
  > - Trailing comma removal
  > - Bracket balancing
  > - AST literal_eval fallback

- **_get_structured_output_helper**(`self`) -> `None`
  > Get a StructuredOutputHelper instance for schema-validated LLM calls.

- **_analyze_global_context**(`self, texts, dataset_columns`) -> `None`
  > Analyzes a sample of the documents and dataset columns to determine the global context.

- **initialize_system**(`self, model_name, adherence_score, temperature, dataset_columns, use_existing_db, progress_callback, selected_files, embedding_model, provider, base_url, vector_engine, qdrant_url, qdrant_api_key, embedding_provider, embedding_api_key, embedding_base_url`) -> `None`
  > Initialize the RAG system with documents, embeddings, the vector database and the LLM.
  >
  > ``embedding_provider`` embeds the documents when it differs from the answering ``provider``; Claude
  > (``anthropic``) has no embedding API and uses local Ollama unless another provider is given.
  >
  > ``vector_engine`` is "lancedb" or "qdrant" (None: the server default ``CMP_VECTOR_STORE``, else
  > the engine recommended for this machine); ``qdrant_url`` selects a Qdrant server instead of
  > the embedded mode (default ``CMP_QDRANT_URL``).

- **update_adherence_score**(`self, adherence_score`) -> `None`
  > Updates the adherence score and rebuilds the chain without full re-initialization.

- **_update_internal_state**(`self, adherence_score`) -> `None`
  > Updates internal state using the Modern LangChain 0.3 Architecture.

- **enrich_variable_taxonomy**(`self, current_taxonomy, columns_info, distance, progress_callback`) -> `None`
  > Enriches an existing taxonomy using 'Wise Enrichment' strategy (Formula-centric).
  > Returns (new_candidates, new_formulas, error) where new_candidates is a dict of proposed variables.
  > Does NOT merge automatically.

- **refine_variable_taxonomy**(`self, current_taxonomy, feedback_dict, columns_info`) -> `None`
  > Refines specific variables in the taxonomy based on user feedback.

- **suggest_anomaly_criteria**(`self, description, columns, sample_data, mode`) -> `None`
  > Suggests anomaly or inclusion criteria based on natural language description.
  > Returns a list of criteria dictionaries.

### def `_env_api_key` (rag_manager.py)

- **Arguments**: `provider: ComplexType`

- **Returns**: `ComplexType`

Server-configured API key for a cloud provider; used only when the user typed none.

---

## rag_schemas.py

_RAG Response Schemas.

Pydantic models for structured LLM outputs, enabling type-safe parsing
and Gemini's native response_schema enforcement._

**Imports**:
`pydantic.BaseModel`, `pydantic.Field`

### class `TheoreticalConcept` (rag_schemas.py)

Schema for a theoretical medical concept/formula.

### class `TheoreticalConceptList` (rag_schemas.py)

Container for multiple theoretical concepts.

### class `ComputedVariableSuggestion` (rag_schemas.py)

Schema for a computed variable suggestion mapped to dataset columns.

### class `SuggestionResponse` (rag_schemas.py)

Container for computed variable suggestions.

### class `ProxyVariable` (rag_schemas.py)

Schema for proxy variable suggestion.

### class `AlternativeFormula` (rag_schemas.py)

Schema for alternative formula suggestion.

### class `ExpandedTerm` (rag_schemas.py)

Schema for expanded medical term.

### class `FormulaCorrection` (rag_schemas.py)

Schema for corrected formula.

### class `MarkdownFormulas` (rag_schemas.py)

Schema for LaTeX formula conversions.

---

## rag_taxonomy.py

_Taxonomy Mixin for RAGManager.

Contains all variable taxonomy generation and enrichment methods.
This is part of the RAGManager class composition pattern._

**Imports**:
`json`

### class `TaxonomyMixin` (rag_taxonomy.py)

Mixin class containing taxonomy-related functionality for RAGManager.

Methods:
    - suggest_column_renaming

    - generate_variable_taxonomy

    - _generate_simple_taxonomy

    - _generate_advanced_taxonomy

    - _merge_stats

    - _identify_standard_concepts

    - _enrich_with_formulas

    - _contextualize_variables

    - _resolve_formula_links

    - _unify_synonyms

    - _enrich_graph_metadata

    - enrich_variable_taxonomy

    - refine_variable_taxonomy

**Methods:**

- **suggest_column_renaming**(`self, columns, strategy, progress_callback`) -> `None`
  > Suggests standardized column names based on medical literature.

- **generate_variable_taxonomy**(`self, columns_info, existing_mapping, deep_analysis, progress_callback`) -> `None`
  > Generates a taxonomy mapping for the provided columns.
  > If deep_analysis is True, uses a multi-step pipeline for richer results.

- **_merge_stats**(`self, taxonomy, columns_info`) -> `None`
  > Merges statistical info from columns_info into the taxonomy.

- **_generate_simple_taxonomy**(`self, columns_info, existing_mapping, deep_analysis, progress_callback`) -> `None`
  > Generates a taxonomy mapping for the provided columns.
  > columns_info: List of dicts with 'name', 'type', 'stats', etc.
  > existing_mapping: Dict of manual renames (original -> new) to use as ground truth.

- **_generate_advanced_taxonomy**(`self, columns_info, existing_mapping, progress_callback`) -> `None`
  > Orchestrates the multi-step taxonomy generation pipeline.

- **_enrich_graph_metadata**(`self, taxonomy, columns_info`) -> `None`
  > Step 4: Identify node types and relationships for graph visualization.

- **_identify_standard_concepts**(`self, columns_info, existing_mapping`) -> `None`
  > Step 1: Map variables to standard medical concepts.

- **_enrich_with_formulas**(`self, taxonomy, columns_info`) -> `None`
  > Step 2: Identify formula relationships (Structured).

- **_contextualize_variables**(`self, taxonomy, columns_info`) -> `None`
  > Step 3: Add clinical usage and interpretation context.

- **_resolve_formula_links**(`self, taxonomy`) -> `None`
  > Step 4: Resolve 'External' formula inputs to internal keys via LLM.

- **_unify_synonyms**(`self, new_candidates, existing_keys`) -> `None`
  > Uses LLM to identify and merge synonyms within the new candidates
  > and against existing taxonomy keys.
  > Returns a mapping { 'alias_id': 'canonical_id' }.

---

## report_tables.py

_The results of the study dossier as report blocks: tables, lines of text, references and, for survival
analyses, a Kaplan-Meier figure. One function, :func:`result_blocks`, reads every type of result the analysis
pages store (biostats results, pooled results over imputed datasets, the dictionaries of a few pages) and
returns plain data that the Word builder (:mod:`manage.dossier`) lays out._

**Imports**:
`__future__.annotations`, `math`, `dataclasses.dataclass`, `dataclasses.field`, `io.BytesIO`, `typing.Any`, `pandas`

### class `ResultBlocks` (report_tables.py)

No description available.

### def `_num` (report_tables.py)

- **Arguments**: `value: Any, digits: int`

- **Returns**: `str`

No description available.

### def `_p` (report_tables.py)

- **Arguments**: `value: Any`

- **Returns**: `str`

No description available.

### def `_peq` (report_tables.py)

- **Arguments**: `value: Any`

- **Returns**: `str`

'p = 0.251' or 'p < 0.001', for sentences.

### def `_ci` (report_tables.py)

- **Arguments**: `ci: Any`

- **Returns**: `str`

'low to high' from a CI object, a pair or a mapping.

### def `_test_line` (report_tables.py)

- **Arguments**: `test: Any`

- **Returns**: `str`

One line for a TestResult: estimate with CI, statistic, df, p and n.

### def `_test_blocks` (report_tables.py)

- **Arguments**: `test: Any`

- **Returns**: `ResultBlocks`

No description available.

### def `_model_table` (report_tables.py)

- **Arguments**: `table: pd.DataFrame`

- **Returns**: `tuple[ComplexType]`

Term, ratio (or coefficient) with its 95% CI and p, from a model's coefficient table.

### def `km_png` (report_tables.py)

- **Arguments**: `curves: list[tuple[ComplexType]]`

- **Returns**: `bytes`

Kaplan-Meier curves (label, times, survival) as a step plot.

### def `_median` (report_tables.py)

- **Arguments**: `value: Any`

- **Returns**: `str`

No description available.

### def `_survival_blocks` (report_tables.py)

- **Arguments**: `res: Any`

- **Returns**: `ResultBlocks`

No description available.

### def `_pooled_frame` (report_tables.py)

- **Arguments**: `result: Any`

- **Returns**: `pd.DataFrame`

No description available.

### def `_pooled_blocks` (report_tables.py)

- **Arguments**: `res: Any`

- **Returns**: `ResultBlocks`

No description available.

### def `_univariate_blocks` (report_tables.py)

- **Arguments**: `value: dict`

- **Returns**: `ResultBlocks`

No description available.

### def `_pooled_comparison_blocks` (report_tables.py)

- **Arguments**: `value: dict`

- **Returns**: `ResultBlocks`

No description available.

### def `result_blocks` (report_tables.py)

- **Arguments**: `value: Any`

- **Returns**: `ResultBlocks`

The report blocks of one stored result (unknown types give a short text).

---

## reproduction_manager.py

_Reproduction engine: replays a transformation trace headlessly (no Streamlit).

One engine for every caller (audit E10): :func:`run_trace` executes the recorded steps against the
source dataset (or a pre-loaded frame) up to an optional step index, resolving file paths through
the user's workspace and the ``path_overrides`` chosen in the UI. :func:`execute_trace_pipeline`
keeps the historical ``(df, error)`` signature.

Traces recorded since schema 2 carry, for every step, the fingerprint of the table it produced. The
replay checks each replayed step against it and reports the first *divergence*; steps that cannot be
recomputed (an adopted multiple imputation, a table replaced outside the app) are listed as not
replayable, and their recorded snapshot is used when there is one. A replay is *reproducible* when every
step was replayed and matched._

**Imports**:
`__future__.annotations`, `logging`, `os`, `collections.abc.Callable`, `dataclasses.dataclass`, `dataclasses.field`, `typing.Any`, `pandas`, `enrich.external_data`, `core.fingerprint.ensure_row_ids`, `core.fingerprint.frame_fingerprint`, `core.fingerprint.ranges_to_ids`, `core.population.criteria_from_trace`, `core.population.evaluate_population`, `core.session.restrict_cells`, `core.session.sort_filled`, `core.state.ParseOptions`, `enrich.formulas.evaluate_formula_safely`, `enrich.imputation_service.fill_with_formula`, `enrich.imputation_service.impute_rows`, `enrich.transformation_logic.apply_variable_transformation`, `manage.db_manager.DBManager`, `utils.date_parser.smart_parse_dates`, `utils.path_utils.resolve_path`

### class `ReplayResult` (reproduction_manager.py)

No description available.

**Methods:**

- **reproducible**(`self`) -> `bool`
  > Every step replayed, and every recorded fingerprint matched.

### class `_NotReplayable` (reproduction_manager.py)

A step the engine cannot recompute; carries the frame to continue from.

### def `_resolve` (reproduction_manager.py)

- **Arguments**: `path: ComplexType, overrides: ComplexType`

- **Returns**: `ComplexType`

No description available.

### def `_read_any` (reproduction_manager.py)

- **Arguments**: `path: str, parse_options: ComplexType`

- **Returns**: `ComplexType`

Read csv/tsv/txt/xlsx/xls/parquet through the import engine; other names are DuckDB dataset names.

### def `load_source` (reproduction_manager.py)

- **Arguments**: `trace_data: dict[ComplexType], path_overrides: ComplexType`

- **Returns**: `tuple[ComplexType]`

The source dataset of a trace: the recorded artifact (newer traces) or the original path/name.

### def `_population_rows` (reproduction_manager.py)

- **Arguments**: `df: pd.DataFrame, block: dict[ComplexType], index: int, warnings: list[str]`

- **Returns**: `pd.Index`

Rows of ``df`` kept by the population recorded in a step (criteria re-evaluated as they were).

### def `_snapshot` (reproduction_manager.py)

- **Arguments**: `step: dict[ComplexType], overrides: ComplexType`

- **Returns**: `ComplexType`

No description available.

### def `_enrichment` (reproduction_manager.py)

- **Arguments**: `df, step, index, overrides, warnings`

- **Returns**: `None`

No description available.

### def `_imputation` (reproduction_manager.py)

- **Arguments**: `df, step, index, overrides, warnings`

- **Returns**: `None`

No description available.

### def `_targeted_imputation` (reproduction_manager.py)

- **Arguments**: `df, step, index, overrides, warnings`

- **Returns**: `None`

No description available.

### def `_variable_transformation` (reproduction_manager.py)

- **Arguments**: `df, step, index, overrides, warnings`

- **Returns**: `None`

No description available.

### def `_variable_computation` (reproduction_manager.py)

- **Arguments**: `df, step, index, overrides, warnings`

- **Returns**: `None`

No description available.

### def `_data_type_conversion` (reproduction_manager.py)

- **Arguments**: `df, step, index, overrides, warnings`

- **Returns**: `None`

No description available.

### def `_missing_value_handling` (reproduction_manager.py)

- **Arguments**: `df, step, index, overrides, warnings`

- **Returns**: `None`

No description available.

### def `_rename_columns` (reproduction_manager.py)

- **Arguments**: `df, step, index, overrides, warnings`

- **Returns**: `None`

No description available.

### def `_apply_filter` (reproduction_manager.py)

- **Arguments**: `df, step, index, overrides, warnings`

- **Returns**: `None`

No description available.

### def `_population_definition` (reproduction_manager.py)

- **Arguments**: `df, step, index, overrides, warnings`

- **Returns**: `None`

No description available.

### def `_mi_impute` (reproduction_manager.py)

- **Arguments**: `df, step, index, overrides, warnings, result`

- **Returns**: `None`

Re-run a multiple imputation on its population and check the imputed cells against the recorded hash.

### def `_freeze_population` (reproduction_manager.py)

- **Arguments**: `df, step, index, overrides, warnings`

- **Returns**: `None`

No description available.

### def `_outlier_clipping` (reproduction_manager.py)

- **Arguments**: `df, step, index, overrides, warnings`

- **Returns**: `None`

No description available.

### def `_apply_step` (reproduction_manager.py)

- **Arguments**: `df: pd.DataFrame, step: dict[ComplexType], index: int, overrides: ComplexType, warnings: list[str], result: ComplexType`

- **Returns**: `tuple[ComplexType]`

Apply one recorded step; returns (df, missing_path). Raises _NotReplayable for steps it cannot redo.

### def `apply_steps` (reproduction_manager.py)

- **Arguments**: `df: pd.DataFrame, steps: list[dict[ComplexType]]`

- **Returns**: `ReplayResult`

Replay ``steps`` on ``df`` (step numbers in messages start at ``start``), checking fingerprints.

### def `run_trace` (reproduction_manager.py)

- **Arguments**: `trace_data: dict[ComplexType]`

- **Returns**: `ReplayResult`

Replay ``trace_data`` (all steps, or up to and including ``upto``) and return the resulting frame.

### def `execute_trace_pipeline` (reproduction_manager.py)

- **Arguments**: `trace_data: dict[ComplexType], progress_callback: ComplexType`

- **Returns**: `tuple[ComplexType]`

Historical signature: ``(result_dataframe, error_message)``.

---

## schemas.py

_Identity and Access Management (IAM) Schemas and Data Models.

Defines strict Pydantic models for user profiles, credentials, authentication
results, and IAM provider configurations._

**Imports**:
`enum.Enum`, `typing.Any`, `pydantic.BaseModel`, `pydantic.Field`

### class `IAMProviderType` (schemas.py)

Supported IAM provider backends.

### class `UserRole` (schemas.py)

Clinical user authorization roles.

### class `UserProfile` (schemas.py)

Structured representation of a registered platform user.

### class `UserCredentials` (schemas.py)

User authentication credentials for sign-in.

### class `AuthResult` (schemas.py)

Result of an authentication or verification attempt.

### class `IAMConfig` (schemas.py)

Configuration specification for the IAM subsystem.

---

## schemas.py

_RAG Provider and Model Configuration Schemas.

Defines strongly-typed Pydantic schemas for LLM and Embedding provider settings,
ensuring validation, serialization, and support for multi-provider configurations._

**Imports**:
`enum.Enum`, `pydantic.BaseModel`, `pydantic.Field`, `pydantic.field_validator`

### class `ProviderType` (schemas.py)

Supported LLM and Embedding providers.

### class `ProviderConfig` (schemas.py)

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

**Methods:**

- **sanitize_base_url**(`cls, v: ComplexType`) -> `ComplexType`
  > Strip trailing slashes for consistent endpoint path appending.

### class `RAGConfig` (schemas.py)

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

---

## selection.py

_Choosing and opening a vector database for the user's machine and needs.

- :class:`VectorStoreSettings` reads the server defaults (``CMP_VECTOR_STORE``, ``CMP_QDRANT_URL``,
  ``CMP_QDRANT_API_KEY``); the Settings page lets each user override the engine for the session.

- :func:`probe_machine` measures CPU cores, memory and free disk without extra dependencies.

- :func:`recommend` turns the machine, the size of the document library and the presence of a
  Qdrant server into a recommendation with its reasons and warnings (pure, unit-tested).

- :func:`open_index` opens the selected engine in the user's workspace._

**Imports**:
`__future__.annotations`, `ctypes`, `os`, `platform`, `shutil`, `sys`, `dataclasses.dataclass`, `dataclasses.field`, `pathlib.Path`, `base.BaseVectorIndex`, `base.VectorEngine`, `base.VectorStoreError`, `lancedb_index.ANN_THRESHOLD`, `lancedb_index.LanceDBIndex`, `qdrant_index.QdrantIndex`, `qdrant_index.user_collection`

### class `VectorStoreSettings` (selection.py)

No description available.

**Methods:**

- **from_env**(`cls`) -> `VectorStoreSettings`
  > No description available.

### class `MachineProfile` (selection.py)

No description available.

### class `Recommendation` (selection.py)

No description available.

### def `_memory_bytes` (selection.py)

- **Arguments**: ``

- **Returns**: `tuple[ComplexType]`

Total and available physical memory, without psutil.

### def `probe_machine` (selection.py)

- **Arguments**: `storage: ComplexType`

- **Returns**: `MachineProfile`

No description available.

### def `estimate_chunks` (selection.py)

- **Arguments**: `pdf_sizes: list[int]`

- **Returns**: `int`

Rough number of chunks a set of PDFs will produce (from their file sizes).

### def `estimate_index_mb` (selection.py)

- **Arguments**: `chunks: int, dim: int`

- **Returns**: `float`

Approximate size of an index: float32 vectors plus text and metadata.

### def `recommend` (selection.py)

- **Arguments**: `machine: MachineProfile`

- **Returns**: `Recommendation`

The engine that suits this machine and library best, with the reasons.

### def `qdrant_embedded_warnings` (selection.py)

- **Arguments**: `machine: MachineProfile`

- **Returns**: `list[str]`

Why embedded Qdrant may not suit this library or machine.

### def `resolve_engine` (selection.py)

- **Arguments**: `choice: ComplexType, settings: VectorStoreSettings, recommendation: Recommendation`

- **Returns**: `VectorEngine`

The user's choice, else the server default, else the recommendation.

### def `open_index` (selection.py)

- **Arguments**: `engine: ComplexType`

- **Returns**: `BaseVectorIndex`

Open the engine's index in ``base_dir/<engine>`` (Qdrant on a server when a URL is given).

---

## setup_token.py

_First-run administrator (follow-up programme M4).

A fresh installation (a new Docker container) has no administrator. When no bootstrap password is configured
(``CMP_ADMIN_PASSWORD`` / ``CMP_ADMIN_PASSWORD_FILE``), the server issues a one-time setup token and writes it to
its log; the login page then offers to create the first administrator with it. Whoever can read the server log
operates the server, which is the trust this relies on (as Jupyter's token does).

Only a SHA-256 hash of the token is stored, in the data root. The token is valid until it is used or until an
administrator exists; a restart without an administrator issues a new one._

**Imports**:
`__future__.annotations`, `contextlib`, `hashlib`, `hmac`, `json`, `os`, `secrets`, `datetime.UTC`, `datetime.datetime`, `pathlib.Path`

### def `_path` (setup_token.py)

- **Arguments**: `directory: ComplexType`

- **Returns**: `Path`

No description available.

### def `_digest` (setup_token.py)

- **Arguments**: `token: str`

- **Returns**: `str`

No description available.

### def `issue` (setup_token.py)

- **Arguments**: `directory: ComplexType`

- **Returns**: `str`

A new token (the previous one, if any, stops working); only its hash is written.

### def `pending` (setup_token.py)

- **Arguments**: `directory: ComplexType`

- **Returns**: `bool`

No description available.

### def `check` (setup_token.py)

- **Arguments**: `directory: ComplexType, token: str`

- **Returns**: `bool`

True when ``token`` is the current setup token (constant-time comparison).

### def `clear` (setup_token.py)

- **Arguments**: `directory: ComplexType`

- **Returns**: `None`

No description available.

---

## trace_documenter.py

_No module description._

**Imports**:
`json`, `os`, `datetime.datetime`, `docx.Document`, `docx.enum.text.WD_ALIGN_PARAGRAPH`, `docx.shared.Pt`, `docx.shared.RGBColor`, `core.lineage.verify_trace`

### class `TraceDocumenter` (trace_documenter.py)

No description available.

**Methods:**

- **__init__**(`self, trace_source`) -> `None`
  > Initialize with either a dictionary trace or a path to a json trace file.

- **generate_report**(`self`) -> `None`
  > Generates a Word document report of the transformation trace.
  > Returns the BytesIO object of the document.

- **_add_title**(`self, doc`) -> `None`
  > No description available.

- **_add_session_info**(`self, doc`) -> `None`
  > No description available.

- **_add_step**(`self, doc, index, step`) -> `None`
  > No description available.

- **_add_mi_methods**(`self, doc, params`) -> `None`
  > The methods paragraph of a multiple imputation, written from its recorded specification.

- **_render_params**(`self, doc, params, indent`) -> `None`
  > No description available.

- **_add_row**(`self, table, label, value`) -> `None`
  > No description available.

- **_get_readable_function_name**(`self, func_name`) -> `None`
  > No description available.

- **_add_footer**(`self, doc`) -> `None`
  > No description available.

---

## transformation_manager.py

_No module description._

**Imports**:
`contextlib`, `json`, `logging`, `os`, `datetime.datetime`, `typing.Any`, `pandas`, `core.lineage.TRACE_SCHEMA`, `core.lineage.genesis_hash`, `core.lineage.seal_step`, `core.lineage.verify_trace`, `core.workspace.UserScope`, `utils.data_paths.get_traces_dir`, `utils.session_scope.get_current_scope`

### class `TransformationManager` (transformation_manager.py)

No description available.

**Methods:**

- **__init__**(`self, trace_dir: ComplexType, scope: ComplexType`) -> `None`
  > Initialize the TransformationManager.
  >
  >
  > **Parameters**:
  >
  > trace_dir : str, optional
  >      Directory where transformation traces will be stored.
  >      Defaults to DATA_ROOT/traces via data_paths utility.

- **initialize_session**(`self, dataset_name: str, username: ComplexType`) -> `None`
  > Initializes a new session trace (``parent`` links it to the trace of a saved version it starts from).

- **get_trace_path**(`self`) -> `ComplexType`
  > Generates the trace file path based on the session ID.

- **add_step**(`self, function_name, params, description, output_dataset_path, stats_cat_path, stats_num_path, corr_matrix_path`) -> `None`
  > Records a transformation step with optional references to intermediate data.
  >
  >
  > **Parameters**:
  >
  > function_name : str
  >     Name of the function/transformation applied.
  > params : dict
  >      Dictionary of parameters used in the transformation.
  > description : str
  >      Human-readable description of the step.
  > output_dataset_path : str, optional
  >      Path to the output dataset (snapshot).
  > stats_cat_path : str, optional
  >      Path to categorical statistics file.
  > stats_num_path : str, optional
  >      Path to numerical statistics file.
  > corr_matrix_path : str, optional
  >      Path to correlation matrix file.
  > data : dict, optional
  >      What the step did to the table (rows/columns in and out, fingerprints): core.lineage.Change.
  > population : dict, optional
  >      The analysis population the step used (criteria, counts, rows kept).
  > replayable : bool, optional
  >      False when the replay engine cannot redo the step.
  > recorded_at : str, optional
  >      When the change was made, if it was applied later (an Enrich draft); the step then also records
  >      ``applied_at``.

- **_make_serializable**(`self, obj`) -> `None`
  > Recursively converts objects to JSON-serializable formats.

- **get_trace**(`self`) -> `None`
  > Returns the current transformation trace.

- **verify**(`self`) -> `None`
  > Integrity of the hash chain (core.lineage.TraceVerification).

- **save_trace**(`self, filepath`) -> `None`
  > Saves the trace to a JSON file (UTF-8, atomic replace).
  >
  > A relative ``filepath`` is reduced to its file name and placed in ``trace_dir``, so a trace never
  > lands outside the user's workspace (for example in the process working directory).

- **load_trace**(`self, filepath`) -> `None`
  > Loads a trace from a JSON file.

- **get_available_datasets_from_traces**(`self, username`) -> `None`
  > Scans all trace files in the trace directory and returns a list of available datasets.
  > Each entry contains: session_id, timestamp, step_description, file_path.
  > Filters by username if provided (unless admin).

---
