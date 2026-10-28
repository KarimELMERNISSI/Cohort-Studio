# Backend Management Logic

## ai_payload.py

_What the AI features may send about a dataset to a cloud provider: a summary of each column, never a row.

The column renaming, the variable taxonomy, the unit hints of Enrich and the suggested criteria of Validate all
describe the columns with :func:`column_profile`, and the data-sharing notice of Settings lists exactly this:
- numeric columns: the 5th, 25th, 50th, 75th and 95th percentiles; never the minimum or the maximum, which can
  single out one participant;

- other columns: their most frequent values with their counts, only values shared by at least ``MIN_COUNT`` rows;

- date columns: the first and the last year.

Statistics that reach a prompt by another way (a variable taxonomy imported from a file, edited, or built before
this rule) pass through allow-lists: :func:`safe_stats` keeps the percentiles only, and :func:`taxonomy_for_prompt`
drops the keys of a taxonomy that hold values of the data.

The module also names the recipients of the data (:func:`recipient`, :func:`store_recipient`, :func:`provider_label`):
the cloud providers, and an Ollama or a Qdrant server outside this machine and the private network, which need the
user's confirmation._

**Imports**:
`__future__.annotations`, `ipaddress`, `logging`, `os`, `socket`, `collections.abc.Iterable`, `collections.abc.Mapping`, `typing.Any`, `urllib.parse.urlparse`, `pandas`, `core.workspace.file_name_of`, `manage.rag.providers.AI_PROVIDERS`, `manage.rag.providers.declared`

### def `_frequent_values` (ai_payload.py)

- **Arguments**: `series: pd.Series`

- **Returns**: `dict[ComplexType]`

No description available.

### def `column_profile` (ai_payload.py)

- **Arguments**: `df: pd.DataFrame, columns: ComplexType`

- **Returns**: `list[dict[ComplexType]]`

One summary per column (``name``, ``type`` and ``stats``, ``top_values`` or ``range``). ``date_columns`` are read
as dates even when they hold text (only their years are sent).

### def `_two_digits` (ai_payload.py)

- **Arguments**: `value: float`

- **Returns**: `str`

A value to two significant digits, written plainly (120, 0.95, 2,500): enough to tell a unit.

### def `typical_ranges` (ai_payload.py)

- **Arguments**: `df: pd.DataFrame, columns: ComplexType`

- **Returns**: `dict[ComplexType]`

The typical values of each number column, from the summary of :func:`column_profile` (its 5th, 50th and 95th
percentiles, never the minimum or the maximum, and none for a column of fewer than 21 values), to two
significant digits: "52 to 98 (median 75)", a clue to the column's unit. ``date_columns`` (a date kept as a
number) give none.

### def `column_kinds` (ai_payload.py)

- **Arguments**: `df: pd.DataFrame, columns: Iterable[str]`

- **Returns**: `dict[ComplexType]`

How each column is stored, for a prompt that writes formulas on them: ``number``, ``text``, ``date`` or
``yes/no``. ``date_columns`` are dates even when they hold text (formulas compute with them: a subtraction gives
days). Like the column names, it sends no value: a model told that a column holds text compares it with text, not
with a number.

### def `safe_stats` (ai_payload.py)

- **Arguments**: `stats: Any`

- **Returns**: `dict[ComplexType]`

What a prompt may carry of a variable's statistics: the percentiles of :func:`column_profile`, nothing else (an
allow-list). The minimum, maximum, mean, "0%" or "100%" of an imported or edited map are dropped.

### def `frequent_values` (ai_payload.py)

- **Arguments**: `values: Any`

- **Returns**: `dict[ComplexType]`

The values of :func:`column_profile` that may be sent: each shared by at least ``MIN_COUNT`` rows.

### def `taxonomy_for_prompt` (ai_payload.py)

- **Arguments**: `taxonomy: Mapping[ComplexType]`

- **Returns**: `dict[ComplexType]`

A variable taxonomy as a prompt may carry it (allow-lists): the keys that describe each variable
(``DESCRIPTIVE_KEYS``) and the percentiles of its statistics. An entry that is not a mapping is left out. The
user's map itself is unchanged.

### def `document_name` (ai_payload.py)

- **Arguments**: `metadata: ComplexType`

- **Returns**: `str`

The document a passage comes from, as a prompt names it: the file's name, never its path on the server, which
names the user's account and the server's folders. Both separators are read: an index can move between systems.

### def `host_of` (ai_payload.py)

- **Arguments**: `url: ComplexType`

- **Returns**: `str`

The host of a service address ("" when there is none), as an HTTP client reads it: a backslash counts as a
slash, so a name and an "@" before it cannot hide the host the client connects to.

### def `_numeric_address` (ai_payload.py)

- **Arguments**: `host: str`

- **Returns**: `ComplexType`

The address a numeric host stands for, as the system's resolver reads it without a lookup (134744072 and
0x08.0x08.0x08.0x08 are 8.8.8.8), or None for a name.

### def `on_this_machine` (ai_payload.py)

- **Arguments**: `url: ComplexType`

- **Returns**: `bool`

No description available.

### def `outside_private_network` (ai_payload.py)

- **Arguments**: `url: ComplexType`

- **Returns**: `bool`

Whether an address may be anywhere: neither this machine nor a private network (the private and link-local
address ranges, a single-label name such as a Docker service, a .local, .lan or .internal name). Data sent
there may leave the user's organisation.

### def `recipient` (ai_payload.py)

- **Arguments**: `provider: ComplexType, base_url: ComplexType`

- **Returns**: `ComplexType`

The recipient that needs the user's confirmation: a cloud provider (any provider not known as local), or an
Ollama outside the private network ("ollama@<host>"); None for an Ollama on this machine or the private network.

### def `store_recipient` (ai_payload.py)

- **Arguments**: `url: ComplexType`

- **Returns**: `ComplexType`

A Qdrant server outside this machine and the private network ("qdrant@<host>"), which needs the confirmation as
an Ollama there does: it keeps the text of every indexed PDF and receives the vector of every search. None for
embedded Qdrant (no address) and a server on this machine or the private network.

### def `provider_label` (ai_payload.py)

- **Arguments**: `provider: str`

- **Returns**: `str`

The name of a recipient: "Google Gemini", or "Ollama at <host>" and "Qdrant at <host>" outside the network.

### def `local_providers` (ai_payload.py)

- **Arguments**: ``

- **Returns**: `frozenset[str]`

The providers that run on this machine or the user's network, as they declare (manage.rag.providers); every
other one, an unknown one included, needs the confirmation.

### def `disable_remote_tracing` (ai_payload.py)

- **Arguments**: ``

- **Returns**: `list[str]`

Switches off the LangChain tracing service (LangSmith) at start-up: a tracing variable left in ``.env`` or the
environment would send every prompt there, outside the data-sharing confirmation. LangSmith's global switch is
set off too, which wins over any variable it reads later. Returns the variables that were on.

---

## authz.py

_Authorization and credential-policy helpers shared by the UI and the CLI.

Privileges are derived from the role stored by the IAM provider (``UserProfile.role``), never from
the username string (audit finding E5)._

**Imports**:
`__future__.annotations`, `re`, `utils.settings.get_settings`, `schemas.UserRole`

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
`abc.ABC`, `abc.abstractmethod`, `typing.ClassVar`, `core.registry.Availability`, `schemas.AuthResult`, `schemas.IAMConfig`, `schemas.UserCredentials`, `schemas.UserProfile`, `schemas.UserRole`

### class `BaseIAMProvider` (base.py)

Abstract Base Class for IAM Providers.

Subclasses implement credential verification, user provisioning, role
management, and session token resolution against specific backends.

**Methods:**

- **from_config**(`cls, config: IAMConfig`) -> `BaseIAMProvider`
  > The provider the configuration describes (IAMManager builds the configured one so). A provider not
  > registered in IAM_PROVIDERS is installed with IAMManager.set_provider instead.

- **availability**(`cls`) -> `Availability`
  > Whether it can run here: its libraries are the application's own (a Better Auth service that does not
  > answer is said at sign-in).

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
`abc.abstractmethod`, `collections.abc.Callable`, `typing.Any`, `langchain_core.callbacks.manager.CallbackManagerForLLMRun`, `langchain_core.embeddings.Embeddings`, `langchain_core.language_models.chat_models.BaseChatModel`, `langchain_core.messages.BaseMessage`, `langchain_core.outputs.ChatResult`, `pydantic.Field`, `schemas.ProviderConfig`

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

- **read_document**(`self, path: str, prompt: str, progress: ComplexType`) -> `str`
  > The model's answer to ``prompt`` about the document at ``path``, sent to the provider as a file. Only a
  > provider that declares ``sends_files`` (manage.rag.providers.declarations) implements it: the others are given
  > the document's text, which the consent notice says. ``progress`` receives a percentage and a message. Raises
  > on any failure: never returns an error as text, as a chat answer does.

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
`__future__.annotations`, `json`, `abc.ABC`, `abc.abstractmethod`, `collections.abc.Iterable`, `dataclasses.asdict`, `dataclasses.dataclass`, `dataclasses.field`, `datetime.datetime`, `enum.StrEnum`, `pathlib.Path`, `typing.Any`, `typing.ClassVar`, `core.registry.Availability`

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

Storage of chunks in one engine, a connector of :data:`manage.vectorstores.VECTOR_STORES`. Implementations:
:mod:`.lancedb_index`, :mod:`.qdrant_index`.

**Methods:**

- **__init__**(`self, manifest_dir: Path`) -> `None`
  > No description available.

- **availability**(`cls`) -> `Availability`
  > Whether the engine's client library is installed, found without importing it; never raises.

- **open**(`cls, directory: Path`) -> `BaseVectorIndex`
  > The engine's index in ``directory``, or on its server when it has one and ``url`` is given (the user's own
  > collection there).

- **location**(`self`) -> `str`
  > Where the vectors live (folder or server), for display and the manifest.

- **remote**(`self`) -> `bool`
  > Whether the index lives on a server: what it is sent leaves this process (a search's filter included).

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
`logging`, `time`, `requests`, `base.BaseIAMProvider`, `schemas.AuthResult`, `schemas.IAMConfig`, `schemas.UserCredentials`, `schemas.UserProfile`, `schemas.UserRole`

### class `BetterAuthIAMProvider` (better_auth.py)

Remote IAM Provider integrating with Better Auth or compatible REST IAM services.

Includes an in-memory TTL cache to ensure that Streamlit rerun cycles do not
degrade performance with repeated synchronous network requests.

**Methods:**

- **from_config**(`cls, config: IAMConfig`) -> `BetterAuthIAMProvider`
  > No description available.

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
`__future__.annotations`, `logging`, `typing.Any`, `langchain_core.callbacks.manager.CallbackManagerForLLMRun`, `langchain_core.messages.AIMessage`, `langchain_core.messages.BaseMessage`, `langchain_core.messages.SystemMessage`, `langchain_core.outputs.ChatGeneration`, `langchain_core.outputs.ChatResult`, `utils.logging_config.redact_secrets`, `base.BaseLLMConnector`, `schemas.ProviderConfig`, `declarations.Claude`

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

## compute.py

_The process-wide budget of worker processes for the multiple imputation (M5b).

Each session asked for cpu_count - 1 workers, so two people imputing at once started twice the processor's cores.
The budget is shared by every session of the server process: a lease grants what is free, never waits, and gives
it back when the run ends, however it ends. When fewer than two are free, the run takes place in the session's
own process (one worker), which uses none of the budget.

``CMP_MI_MAX_WORKERS`` (utils.settings) sets the total. The default is the processors of this process minus one; a
container's CPU quota is not visible in that count, so set the variable there._

**Imports**:
`__future__.annotations`, `os`, `threading`, `collections.abc.Iterator`, `contextlib.contextmanager`, `utils.settings.get_settings`

### class `WorkerBudget` (compute.py)

Worker processes shared by the sessions of this server process.

**Methods:**

- **__init__**(`self, total: int`) -> `None`
  > No description available.

- **free**(`self`) -> `int`
  > No description available.

- **lease**(`self, wanted: int`) -> `Iterator[int]`
  > Grants ``min(wanted, free)`` worker processes, or 1 (a run in this process, outside the budget) when one or
  > none would be granted; returns them when the run ends.

### def `default_total` (compute.py)

- **Arguments**: ``

- **Returns**: `int`

No description available.

### def `mi_workers` (compute.py)

- **Arguments**: ``

- **Returns**: `WorkerBudget`

The budget of this server process, from the ``CMP_MI_MAX_WORKERS`` setting, read once.

---

## connectors.py

_The connector settings at start-up (M5b, step 4): a setting that names no connector of this version stops start-up.

``check_connector_settings(settings)`` runs in ``main.py`` right after ``utils.settings.load_settings``, inside the same
``try``: it raises SettingsError, so the page and the server log say what to correct before any connector is
chosen. The message names each wrong variable and the valid names, never the value: a mistyped value can be a pasted
key or an address, and the page shows it before sign-in. Each family's registry is asked for its names only, so no
connector module is imported. ``manage_users`` does not import this module: the lazy names of ``manage.rag`` and
``manage.vectorstores`` lead to the AI packages, which a command-line entry point never reaches (rule R5,
tests/test_layering.py)._

**Imports**:
`__future__.annotations`, `graph.renderers.GRAPH_RENDERERS`, `manage.iam.providers.IAM_PROVIDERS`, `manage.rag.providers.AI_PROVIDERS`, `manage.vectorstores.VECTOR_STORES`, `utils.settings.Settings`, `utils.settings.SettingsError`

### def `connector_setting_problems` (connectors.py)

- **Arguments**: `settings: Settings`

- **Returns**: `list[str]`

For each connector setting that names nothing this version offers, the variable and the valid names.

### def `check_connector_settings` (connectors.py)

- **Arguments**: `settings: Settings`

- **Returns**: `Settings`

``settings`` when every connector setting names a connector of this version; SettingsError otherwise.

---

## db_manager.py

_No module description._

**Imports**:
`glob`, `logging`, `os`, `re`, `functools.cached_property`, `typing.Any`, `duckdb`, `pandas`, `core.workspace.UserScope`, `manage.iam.providers.local_duckdb.LocalDuckDBIAMProvider`, `manage.iam.schemas.UserCredentials`, `manage.iam.schemas.UserRole`, `utils.data_paths.get_datasets_dir`, `utils.data_paths.get_stats_dir`

### class `DBManager` (db_manager.py)

Helper class to manage DuckDB connections and data persistence using Parquet with versioning

**Methods:**

- **__init__**(`self, db_path: str, scope: ComplexType`) -> `None`
  > No description available.

- **iam_provider**(`self`) -> `LocalDuckDBIAMProvider`
  > The local accounts, opened on first use: a session's store never opens the users database for nothing
  > (the application signs users in through manage.iam).

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

## declarations.py

_The AI providers, declared (M5b connectors on one kit): what Settings, the factory, the consent notices and the settings
need to know about each provider, read without importing langchain or a provider's SDK. The connectors themselves
(manage/rag/providers/<provider>.py) are imported when an engine starts, from ``chat_connector`` and
``embedding_connector``.

The names are stored in the users' preferences: they never change. The first model of each list is the default._

**Imports**:
`__future__.annotations`, `typing.ClassVar`, `core.registry.Availability`

### class `BaseAIProvider` (declarations.py)

One AI provider, a connector of :data:`manage.rag.providers.AI_PROVIDERS`.

**Methods:**

- **availability**(`cls`) -> `Availability`
  > langchain_core and the provider's SDK, found without importing them; every missing one is named.

- **embeds**(`cls`) -> `bool`
  > No description available.

### class `Ollama` (declarations.py)

No description available.

### class `Gemini` (declarations.py)

No description available.

### class `OpenAI` (declarations.py)

No description available.

### class `Mistral` (declarations.py)

No description available.

### class `Claude` (declarations.py)

No description available.

---

## disabled.py

_Disabled IAM Provider (Bypass / No-Auth Mode).

Allows seamless operation in single-researcher offline workstations or automated
pipelines where user authentication is explicitly disabled. Automatically provides
a pre-authorized administrator session without gating the application._

**Imports**:
`base.BaseIAMProvider`, `schemas.AuthResult`, `schemas.IAMConfig`, `schemas.UserCredentials`, `schemas.UserProfile`, `schemas.UserRole`

### class `DisabledIAMProvider` (disabled.py)

Bypass provider that allows unrestricted access as a local researcher.

**Methods:**

- **from_config**(`cls, config: IAMConfig`) -> `DisabledIAMProvider`
  > No description available.

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
(``{"source": x}`` or ``{"source": {"$in": [...]}}``) is still accepted.

A chunk is stored under its document's name, never its path on the server: a Qdrant server receives the index, and
the path names the user's account and the server's folders. A filter sent to a server names documents only; a local
index built by an earlier version, which stored paths, is matched by path too (the paths stay on this machine)._

**Imports**:
`__future__.annotations`, `hashlib`, `time`, `collections.abc.Callable`, `collections.abc.Sequence`, `typing.Any`, `numpy`, `langchain_core.callbacks.CallbackManagerForRetrieverRun`, `langchain_core.documents.Document`, `langchain_core.embeddings.Embeddings`, `langchain_core.retrievers.BaseRetriever`, `langchain_core.vectorstores.utils.maximal_marginal_relevance`, `pydantic.ConfigDict`, `core.workspace.file_name_of`, `base.BaseVectorIndex`, `base.ChunkRow`, `base.Hit`, `base.IndexManifest`, `base.VectorStoreError`

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

### def `_matching` (document_index.py)

- **Arguments**: `sources: ComplexType`

- **Returns**: `ComplexType`

The sources a filter matches: by the document's name, and on this machine also as given (a local index built by
an earlier version stored paths). A server is sent names only: an index there named after the account, the only
kind that held paths, is never reused.

### def `_named` (document_index.py)

- **Arguments**: `document: Document`

- **Returns**: `Document`

The chunk as the index stores it: its source is its document's name, never a path.

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
`__future__.annotations`, `dataclasses.dataclass`, `dataclasses.field`, `datetime.datetime`, `io.BytesIO`, `typing.Any`, `pandas`, `docx.Document`, `docx.shared.Cm`, `docx.shared.Pt`, `docx.shared.RGBColor`, `core.attrition.MISSING_VALUES_NOTE`, `core.attrition.FlowExit`, `core.attrition.ParticipantFlow`, `manage.report_tables.ResultBlocks`

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

### def `_frozen_criteria` (dossier.py)

- **Arguments**: `flow: ParticipantFlow`

- **Returns**: `list[str]`

The criteria of a population kept as the dataset, with their roles, one line each under the step that kept it (so
that two are told apart, and a criterion is never read as one more rule of a threshold): the freeze unticks them,
so the table of the ticked criteria no longer gives their definitions. Its exit lists the step's description, then
a line per criterion.

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

Instantiates provider-specific connectors from a :class:`ProviderConfig` through the providers' registry
(:data:`manage.rag.providers.AI_PROVIDERS`): each provider declares its chat and embedding connectors, imported when an
engine starts, and its default models. A provider this version does not offer is refused with the valid names, never
replaced by another (an unknown name used to become Gemini, a cloud provider nobody chose)._

**Imports**:
`importlib`, `logging`, `typing.Any`, `base.BaseEmbeddingConnector`, `base.BaseLLMConnector`, `providers.declared_for`, `providers.default_models`, `providers.embedder_for`, `providers.declarations.BaseAIProvider`, `schemas.ProviderConfig`, `schemas.ProviderType`

### class `ConnectorFactory` (factory.py)

Factory for creating LLM and Embedding connectors.

**Methods:**

- **get_llm_connector**(`cls, config: ProviderConfig`) -> `BaseLLMConnector`
  > The chat connector of ``config.provider``.

- **get_embedding_connector**(`cls, config: ProviderConfig`) -> `BaseEmbeddingConnector`
  > The embedding connector of ``config.provider``; ValueError for a provider without an embedding API.

- **create_connectors**(`cls, provider: str, model_name: ComplexType, embedding_model: ComplexType, api_key: ComplexType, base_url: ComplexType, temperature: float, embedding_provider: ComplexType, embedding_api_key: ComplexType, embedding_base_url: ComplexType`) -> `tuple[ComplexType]`
  > Convenience factory method constructing both connectors from keyword arguments.
  >
  > ``embedding_provider`` lets the documents be embedded by another provider than the one that answers; a provider
  > without an embedding API (Claude) is embedded by the first provider that has one (local Ollama) unless another
  > is given. ``embedding_api_key`` / ``embedding_base_url`` configure that provider. The default models are the
  > first each provider declares.

### def `_declared` (factory.py)

- **Arguments**: `provider: ComplexType`

- **Returns**: `type[BaseAIProvider]`

No description available.

### def `_connector` (factory.py)

- **Arguments**: `target: str`

- **Returns**: `Any`

No description available.

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
`logging`, `mimetypes`, `random`, `secrets`, `time`, `collections.abc.Callable`, `typing.Any`, `langchain_core.callbacks.manager.CallbackManagerForLLMRun`, `langchain_core.messages.AIMessage`, `langchain_core.messages.BaseMessage`, `langchain_core.messages.HumanMessage`, `langchain_core.messages.SystemMessage`, `langchain_core.outputs.ChatGeneration`, `langchain_core.outputs.ChatResult`, `utils.logging_config.redact_secrets`, `base.BaseEmbeddingConnector`, `base.BaseLLMConnector`, `schemas.ProviderConfig`, `declarations.Gemini`

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
  > The chat models the key may use (GEMINI_CHAT_MODELS when the API cannot be asked).

- **_generate**(`self, messages: list[BaseMessage], stop: ComplexType, run_manager: ComplexType`) -> `ChatResult`
  > No description available.

- **read_document**(`self, path: str, prompt: str, progress: ComplexType`) -> `str`
  > The model's answer to ``prompt`` about the document at ``path``, uploaded to Google's file API from an open
  > file under a name chosen here (Google is sent neither the file's name nor its path), then Google is asked to
  > delete it however the answer ends, a failed upload included (Google also expires uploaded files on its own).
  > Raises on any failure; gives up when Google is still processing the file after five minutes.

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

### def `is_chat_model` (gemini.py)

- **Arguments**: `name: str, actions: ComplexType`

- **Returns**: `bool`

A Gemini or Gemma model that answers text (generateContent) and is not one of the specialised models.

---

## lancedb_index.py

_LanceDB engine: embedded, serverless, columnar storage on disk (Lance format).

Vectors live in files under the user's workspace, so memory use stays low and nothing has to run
beside the application. Search is exact (brute force, cosine) up to :data:`ANN_THRESHOLD` chunks;
beyond that :meth:`LanceDBIndex.optimize` builds an IVF index for approximate search._

**Imports**:
`__future__.annotations`, `logging`, `pathlib.Path`, `typing.Any`, `core.registry.Availability`, `base.BaseVectorIndex`, `base.ChunkRow`, `base.Hit`, `base.VectorEngine`, `base.VectorStoreError`

### class `LanceDBIndex` (lancedb_index.py)

No description available.

**Methods:**

- **__init__**(`self, directory: Path`) -> `None`
  > No description available.

- **availability**(`cls`) -> `Availability`
  > No description available.

- **open**(`cls, directory: Path`) -> `LanceDBIndex`
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
`logging`, `os`, `collections.abc.Iterable`, `datetime.datetime`, `datetime.timedelta`, `typing.Any`, `typing.cast`, `bcrypt`, `duckdb`, `utils.data_paths`, `core.workspace.safe_dirname`, `utils.logging_config.single_line`, `utils.settings.get_settings`, `base.BaseIAMProvider`, `schemas.AuthResult`, `schemas.IAMConfig`, `schemas.UserCredentials`, `schemas.UserProfile`, `schemas.UserRole`

### class `LocalDuckDBIAMProvider` (local_duckdb.py)

Local embedded IAM backend using DuckDB and bcrypt.

**Methods:**

- **from_config**(`cls, config: IAMConfig`) -> `LocalDuckDBIAMProvider`
  > No description available.

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

### def `_folder` (local_duckdb.py)

- **Arguments**: `username: str`

- **Returns**: `ComplexType`

No description available.

### def `_shares_a_folder` (local_duckdb.py)

- **Arguments**: `username: str, existing: Iterable[str]`

- **Returns**: `bool`

Whether ``username`` would have the workspace folder of an existing account (core.workspace.safe_dirname).

---

## manager.py

_IAM Manager and Factory.

Provides a unified, framework-agnostic facade for identity and access management.
Dynamically resolves and instantiates the active IAM provider (Local DuckDB,
Better Auth, or Disabled) based on environment configuration or runtime parameters._

**Imports**:
`logging`, `os`, `typing.Any`, `core.workspace.workspace_exists`, `utils.logging_config.single_line`, `utils.settings.Settings`, `utils.settings.get_settings`, `.setup_token`, `base.BaseIAMProvider`, `providers.IAM_PROVIDERS`, `schemas.AuthResult`, `schemas.IAMConfig`, `schemas.IAMProviderType`, `schemas.UserCredentials`, `schemas.UserProfile`, `schemas.UserRole`

### class `IAMConfigurationError` (manager.py)

The identity provider is misconfigured: the app must not start with another one in its place.

### class `IAMManager` (manager.py)

Central facade managing IAM operations and provider lifecycle.

**Methods:**

- **__init__**(`self, config: ComplexType`) -> `None`
  > Initialize IAM Manager with configuration or environment variables.

- **_resolve_config_from_env**(`cls, settings: ComplexType`) -> `IAMConfig`
  > The IAM configuration of the settings (utils.settings; the current ones by default). An unknown
  > CMP_IAM_PROVIDER raises IAMConfigurationError: a misspelt provider used to give local accounts without a word,
  > so a server meant to sign users in elsewhere ran with its own accounts instead.

- **_build_provider**(`cls, config: IAMConfig`) -> `BaseIAMProvider`
  > Construct the designated provider instance.

- **provider**(`self`) -> `BaseIAMProvider`
  > Get the active IAM provider instance.

- **is_enabled**(`self`) -> `bool`
  > Whether authentication gating is active.

- **provider_name**(`self`) -> `str`
  > Name of the active provider.

- **provider_type**(`self`) -> `IAMProviderType`
  > The configured provider (CMP_IAM_PROVIDER, or the configuration given).

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
  > Provision a new user account. A name whose workspace folder exists is refused, with the message of a name taken
  > (core.workspace.workspace_exists): the folder may hold another account's files, such as the folder an earlier
  > version gave an account that has not signed in since. The server's operator may reuse it (``reuse_folder``,
  > manage_users.py create-admin).

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

## paths.py

_The files a replay may read (M5b step 5): where the paths a trace records are looked for.

A trace is user-supplied JSON, so in a session a path it records is never read as written. :class:`PathResolver`
finds each file in the places the session may read:

- the files the user chose for recorded paths on the Reproduce page (``named``), first, in the same places;

- the user's workspace (``scope``): the recorded path when it lies inside it, else the file's name in its folders;

- for a command line (M8), only the files named on it, found by their names (:meth:`PathResolver.for_files`);

- for tests and scripts (:meth:`PathResolver.unconfined`): the recorded path when it exists, else the file's name in
  the shared folders of the data root.

A resolver given none of these reads no file, as a replay given no resolver: a session without a signed-in user
reads nothing (every signed-in user has a workspace, :func:`core.workspace.safe_dirname`). A saved version a trace
names is looked up in the datasets folder only (:meth:`PathResolver.dataset`): a name such as
``../../bob/datasets/x`` finds nothing. A path on a network share is never opened
(:func:`core.workspace.network_share`)._

**Imports**:
`__future__.annotations`, `logging`, `os`, `collections.abc.Iterable`, `collections.abc.Mapping`, `dataclasses.dataclass`, `dataclasses.field`, `dataclasses.replace`, `pathlib.Path`, `pandas`, `core.workspace.UserScope`, `core.workspace.network_share`, `core.workspace.resolve_within`, `utils.logging_config.single_line`

### class `PathResolver` (paths.py)

Where the files a trace names are looked for (see the module docstring). ``PathResolver()`` reads none.

**Methods:**

- **for_files**(`cls, files: Iterable[ComplexType]`) -> `PathResolver`
  > Only these files, found by their names (a command line's): nothing else is read.

- **unconfined**(`cls`) -> `PathResolver`
  > Tests and scripts, without a user: the recorded path when it exists, else its name in the shared folders of
  > the data root. Never for a session: a trace could name any file of the server.

- **with_named**(`self, named: Mapping[ComplexType]`) -> `PathResolver`
  > The same places, and the files the user chose for recorded paths (Reproduce).

- **resolve**(`self, recorded: object`) -> `ComplexType`
  > The file to read for a path a trace records, or None when the session may read none.

- **dataset**(`self, name: object`) -> `ComplexType`
  > The saved version ``name`` (``<datasets>/<name>.parquet``), never a file outside that folder: the name comes
  > from a trace.

- **_chosen**(`self, recorded: str`) -> `ComplexType`
  > The file the user chose for ``recorded``, in the places this resolver reads.

- **_named_file**(`self, name: str`) -> `ComplexType`
  > No description available.

- **_search_dirs**(`self`) -> `list[str]`
  > No description available.

### def `normalize_path` (paths.py)

- **Arguments**: `path_str: str`

- **Returns**: `str`

A path with forward slashes: one recorded on Windows keeps its backslashes, which are no separators on Linux.

### def `read_version` (paths.py)

- **Arguments**: `path: str`

- **Returns**: `ComplexType`

A saved version's table: read by pandas, or by DuckDB for a file pandas cannot read. None when it is missing.

---

## preferences.py

_The user's preferences: the choices made in Settings, kept for the next session (M5b, step 3).

A preference is not a setting. A setting is the server's (utils.settings, read from the environment); a preference is
the user's choice among what the server allows: the day-first reading of dates on import, the AI engine and its
models, the graph renderer, the vector database. No key, password, consent or address is ever stored: a cloud provider
saved here starts nothing until the user confirms again in the session (audit E6), and the address of a model
server (Ollama, Qdrant), which decides where the data go, is entered again in each session.

The file is ``<workspace>/preferences.json``, read whole and written whole, atomically, under a lock. A missing or
unreadable file gives no preference; a field that is not valid is left out, and the others are kept._

**Imports**:
`__future__.annotations`, `contextlib`, `json`, `logging`, `os`, `tempfile`, `threading`, `time`, `pathlib.Path`, `typing.Annotated`, `typing.Any`, `pydantic.BaseModel`, `pydantic.ConfigDict`, `pydantic.Field`, `pydantic.StringConstraints`, `pydantic.ValidationError`, `core.workspace.UserScope`

### class `Preferences` (preferences.py)

The choices kept for the next session. Every field is optional: unset, the setting or the default applies.

### def `_lock` (preferences.py)

- **Arguments**: `path: Path`

- **Returns**: `threading.Lock`

No description available.

### def `preferences_path` (preferences.py)

- **Arguments**: `scope: UserScope`

- **Returns**: `Path`

No description available.

### def `_valid_fields` (preferences.py)

- **Arguments**: `raw: dict[ComplexType]`

- **Returns**: `dict[ComplexType]`

The fields of ``raw`` that are valid on their own: one wrong field does not lose the others.

### def `load_preferences` (preferences.py)

- **Arguments**: `scope: UserScope`

- **Returns**: `Preferences`

The user's preferences; none when the file is missing or unreadable (logged without its content).

### def `_write_atomic` (preferences.py)

- **Arguments**: `path: Path, text: str`

- **Returns**: `None`

No description available.

### def `update_preferences` (preferences.py)

- **Arguments**: `scope: UserScope`

- **Returns**: `ComplexType`

Keep these choices for the next session. ``None`` forgets a choice; for the models, a mapping changes the
providers it names (``None`` for a provider forgets its model). Never raises for the file: a preference that
cannot be saved is logged, the session goes on, and None is returned. An unknown name is a programming error.

---

## qdrant_index.py

_Qdrant engine, in two modes.

- **Server** (``CMP_QDRANT_URL``, optional ``CMP_QDRANT_API_KEY``): a Qdrant service shared by every
  session, with HNSW approximate search, payload indexes and concurrent access. One collection
  per user, named at random (``cmp_<32 hexadecimal digits>``), never after the account: the server
  may belong to someone else (Qdrant Cloud). Best for several users or very large libraries.

- **Embedded** (no URL): Qdrant's local mode in the user's workspace. Every vector is held in
  memory and a folder can be opened by one client only, so the client is shared per folder in this
  process. Suited to small and medium libraries (tens of thousands of chunks).

Every rebuild writes a new collection and then drops the previous one; the active name is kept in
the user's workspace, in ``collection.txt`` next to the index's manifest. Each place has its own
folder for both: the embedded index the engine's folder, each server ``servers/<key>`` (the key is a
hash of its address as the client reads it), so that building in one place never loses, nor is
taken for, the index of another. Embedded Qdrant on Windows can reload a deleted collection's files
when the same name is recreated, so names are never reused.

Earlier versions kept one manifest and one pointer for every place, and named a server collection
after the account (``cmp_<user>_chunks``, then ``cmp_<user>_chunks_<id>``). Opening the engine moves
such an index's manifest and pointer to its server's folder (the manifest names the server); it is
never reused there, and the next rebuild on its server deletes it._

**Imports**:
`__future__.annotations`, `hashlib`, `json`, `re`, `threading`, `uuid`, `pathlib.Path`, `typing.Any`, `urllib.parse.urlparse`, `core.registry.Availability`, `base.MANIFEST_NAME`, `base.BaseVectorIndex`, `base.ChunkRow`, `base.Hit`, `base.IndexManifest`, `base.VectorEngine`, `base.VectorStoreError`

### class `QdrantIndex` (qdrant_index.py)

No description available.

**Methods:**

- **__init__**(`self, manifest_dir: Path`) -> `None`
  > No description available.

- **availability**(`cls`) -> `Availability`
  > No description available.

- **open**(`cls, directory: Path`) -> `QdrantIndex`
  > No description available.

- **mode**(`self`) -> `str`
  > No description available.

- **_generation_name**(`self`) -> `str`
  > The name of a new collection: at random on a server (never the account), ``chunks_<id>`` embedded.

- **remote**(`self`) -> `bool`
  > No description available.

- **_pointer**(`self`) -> `Path`
  > The file naming the collection in use, in this place's folder.

- **_pointed**(`self`) -> `str`
  > The collection the pointer names ("" when none; embedded, never a server's, as an earlier version wrote).

- **named_after_account**(`self`) -> `bool`
  > Whether the index in use on a server was built by an earlier version, under the account's name (read from
  > the workspace: no request is made). It is never reused: its rebuild deletes it.

- **location**(`self`) -> `str`
  > No description available.

- **_client**(`self`) -> `None`
  > No description available.

- **ping**(`self`) -> `tuple[ComplexType]`
  > Reachability of the server (or the local folder).

- **active_collection**(`self`) -> `str`
  > Collection holding the current index: the one the pointer names; else the base name of the earliest indexes
  > (before generations), on a server only when this workspace built one there (two installations can share a
  > server and a user name); else ``NO_INDEX``.

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

### def `_read` (qdrant_index.py)

- **Arguments**: `pointer: Path`

- **Returns**: `str`

The collection a pointer file names ("" when there is none).

### def `explicit_address` (qdrant_index.py)

- **Arguments**: `url: ComplexType, api_key: ComplexType`

- **Returns**: `ComplexType`

An address with the scheme the client gives one written without (https with an API key, http without), so that
the address badge, the server's folder and the recipient read the server the client connects to.

### def `_server_key` (qdrant_index.py)

- **Arguments**: `url: str`

- **Returns**: `str`

An address as the client reads it (http when no scheme is given, the host in lower case, port 6333 when none
is given, the path without a final slash): two spellings of one server share one folder.

### def `server_folder` (qdrant_index.py)

- **Arguments**: `directory: Path, url: str`

- **Returns**: `Path`

The folder that keeps the manifest and the collection's name of one server's index, in the engine's folder.

### def `_built_on_server` (qdrant_index.py)

- **Arguments**: `manifest: ComplexType`

- **Returns**: `ComplexType`

The server and the collection a manifest's index was built on, read from its location ("<url> (<name>)"); None
for an embedded index.

### def `adopt_earlier_index` (qdrant_index.py)

- **Arguments**: `directory: Path`

- **Returns**: `None`

An earlier version kept one manifest and one pointer for every place, in ``directory``, both written by the last
build. One built on a server moves to that server's folder (the manifest names the server), so that a build
elsewhere neither loses it nor is taken for it. Reads and writes the workspace only.

### def `unrecorded_collection` (qdrant_index.py)

- **Arguments**: `directory: Path`

- **Returns**: `ComplexType`

A server collection an earlier version's pointer names without the manifest that names its server (a build that
stopped half-way): which server holds it cannot be known, so it can only be reported.

### def `release_clients` (qdrant_index.py)

- **Arguments**: ``

- **Returns**: `None`

Close every shared client (tests, shutdown).

### def `user_collection` (qdrant_index.py)

- **Arguments**: `username: ComplexType`

- **Returns**: `str`

The name earlier versions gave a user's collection on a server, after the account: kept to find an index built
then (:attr:`QdrantIndex.named_after_account`), which the next rebuild replaces.

---

## rag_computed_vars.py

_Computed Variables Mixin for RAGManager.

Contains all computed variable suggestion and validation methods.
This is part of the RAGManager class composition pattern._

**Imports**:
`hashlib`, `json`, `logging`, `unicodedata`, `core.derived.KINDS`, `core.derived.DerivedVariable`, `enrich.formulas.MAX_FORMULA_LENGTH`, `enrich.formulas.formula_from_model`, `enrich.formulas.read_formula`, `manage.ai_payload.document_name`, `manage.ai_payload.taxonomy_for_prompt`, `utils.llm_utils.ProviderError`, `utils.llm_utils.parse_json_safe`, `utils.llm_utils.provider_error`, `utils.llm_utils.validate_and_parse`, `utils.logging_config.redact_secrets`, `rag_schemas.AlternativeFormula`, `rag_schemas.ProxyVariable`

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

- **suggest_imputation_formulas**(`self, target_variable, available_columns, search_hint, context_columns, context_stats, num_suggestions, column_kinds`) -> `None`
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
  > - column_kinds (dict): Optional, how each column is stored (manage.ai_payload.column_kinds; no value).
  >
  > The formulas come back in the application's notation (enrich.formulas.formula_from_model); the page tries
  > them on the data before it offers them.

- **suggest_derived_variables**(`self, columns: list[str], column_kinds: ComplexType, units: ComplexType, ranges: ComplexType, max_candidates: int`) -> `tuple[ComplexType]`
  > The derived variables the model recognises among ``columns``: definitions that compute one column from others
  > of the same row, with their inverses (core.derived, origin "ai"), in the application's notation. Sent: the
  > column names, how each is stored, the variable map's units and, when given, each number column's typical
  > values (manage.ai_payload.typical_ranges); never a row. A proposal that cannot be read, names a column the
  > data do not have or uses the column it computes is dropped here; the page checks the others against the data
  > before it offers them, and adds none by itself.

### def `_passages` (rag_computed_vars.py)

- **Arguments**: `docs`

- **Returns**: `str`

The passages a formula search reads, each with its document's name (never its path on the server).

### def `_in_app_notation` (rag_computed_vars.py)

- **Arguments**: `suggestion: dict, formula: str`

- **Returns**: `str`

The model's formula in the application's notation (enrich.formulas.formula_from_model; a leading "name =" goes
when it names the suggestion or one of ``targets``), kept on the suggestion with what the model wrote, so that its
diagram and "Apply to Editor" compute exactly what was checked.

### def `_unwrapped` (rag_computed_vars.py)

- **Arguments**: `answer: object`

- **Returns**: `str`

A formula a model answered, without the code fence or the one pair of quotes it may be wrapped in (never the
``""a name""`` quotes of a column at either end). An answer too long to be a formula gives nothing, before any
reading of it: a model's text has no bound.

### def `_model_text` (rag_computed_vars.py)

- **Arguments**: `value: object, limit: int`

- **Returns**: `str`

A model's words kept with a definition (its source, its units): one line, no control or bidirectional
character, cut. They are shown as text (ui.components.text.plain), never applied.

### def `_derived_formula` (rag_computed_vars.py)

- **Arguments**: `answer: object, target: str, columns: set[str]`

- **Returns**: `ComplexType`

A formula of the model for ``target``, in the application's notation, kept only when it reads, names columns
the data have, and does not use ``target`` itself.

### def `_derived_candidate` (rag_computed_vars.py)

- **Arguments**: `entry: object, columns: set[str]`

- **Returns**: `ComplexType`

One proposal of the model as a definition (only the keys asked for), or None when it is not one.

---

## rag_documents.py

_Documents Mixin for RAGManager.

Contains all document processing, knowledge graph extraction and chat methods.
This is part of the RAGManager class composition pattern._

**Imports**:
`os`, `manage.ai_payload.document_name`, `manage.rag.providers.file_readers`, `utils.logging_config.redact_secrets`

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
  > Extract a Knowledge Graph of key entities and relationships from a specific document: sent as a file to a
  > provider that declares ``sends_files`` (its connector's read_document), as its text to the others.

- **extract_merged_graph_from_docs**(`self, doc_list, progress_callback, skipped`) -> `None`
  > Extracts and merges graphs from multiple documents.
  > Entities are unified across documents and relation endpoints are remapped to the merged
  > entities (graph.builders.document.merge_document_graphs). Returns (merged_json, error): an error when no
  > document gave a graph. ``skipped``, a list, receives each document left out of the graph with the reason.

- **match_columns_to_graph**(`self, columns, graph_json`) -> `None`
  > Matches dataset columns to Knowledge Graph nodes using fuzzy string matching.
  > Returns dict: {column_name: {match_found: bool, node: data, confidence: float}}

- **chat_with_specific_doc**(`self, doc_names, query`) -> `None`
  > Chat with specific document(s) using the vector store.
  > doc_names: string or list of strings.

### def `_passages` (rag_documents.py)

- **Arguments**: `docs`

- **Returns**: `str`

The passages a question about chosen documents is answered from, each with its document's name (never its
path on the server).

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
`json`, `logging`, `os`, `re`, `pathlib.Path`, `pandas`, `rapidfuzz.process`, `core.workspace.UserScope`, `manage.ai_payload.column_profile`, `manage.ai_payload.document_name`, `manage.ai_payload.frequent_values`, `manage.ai_payload.local_providers`, `manage.ai_payload.provider_label`, `manage.ai_payload.recipient`, `manage.ai_payload.safe_stats`, `manage.ai_payload.store_recipient`, `manage.ai_payload.taxonomy_for_prompt`, `manage.rag.providers.AI_PROVIDERS`, `manage.rag.providers.cloud_providers`, `manage.rag.providers.declared`, `manage.rag.providers.default_models`, `manage.rag.providers.default_provider`, `manage.rag.providers.embedding_providers`, `manage.rag.providers.server_key`, `prompts.anomaly_criteria_prompt`, `prompts.context_analysis`, `prompts.refine_taxonomy`, `prompts.wise_enrichment`, `utils.llm_utils.ProviderError`, `utils.llm_utils.StructuredOutputHelper`, `utils.llm_utils.clean_json_response`, `utils.llm_utils.parse_json_safe`, `utils.llm_utils.provider_error`, `utils.logging_config.redact_secrets`, `rag_computed_vars.ComputedVarsMixin`, `rag_documents.DocumentsMixin`, `rag_taxonomy.TaxonomyMixin`

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

- **recipients_of**(`provider: ComplexType, base_url: ComplexType, embedding_provider: ComplexType, embedding_base_url: ComplexType, qdrant_url: ComplexType`) -> `frozenset[str]`
  > The recipients of a configuration that need the confirmation: its cloud chat and embedding providers, an
  > Ollama outside the private network ("ollama@<host>"), and the Qdrant server there that keeps the index
  > ("qdrant@<host>").

- **recipients**(`self`) -> `frozenset[str]`
  > The recipients the engine sends data to: for answers, for embeddings, and the index's Qdrant server.

- **uses_cloud**(`self`) -> `bool`
  > Whether data leaves this machine and the private network.

- **confirm**(`self`) -> `None`
  > The user confirms that data may be sent to these recipients together (one box of Settings).

- **is_confirmed**(`self, recipients: frozenset[str]`) -> `bool`
  > No description available.

- **cloud_consent**(`self`) -> `bool`
  > Whether the user confirmed the recipients the engine sends data to.

- **holds_connections**(`self`) -> `bool`
  > Whether a connection to a provider is still held (a running engine, or one stopped half-way).

- **_drop_connections**(`self`) -> `None`
  > No description available.

- **stop**(`self, reason: str`) -> `None`
  > Stops the engine: its connections to the providers are dropped, so nothing more can be sent to them. The
  > document index stays on disk, and the next initialisation reuses it.

- **withdraw_consent**(`self`) -> `bool`
  > The user withdraws the confirmation of ``recipients`` (all of them when none is named): every confirmation
  > that names one of them goes. An engine that sends data to one of them stops at once: the confirmation used
  > to be checked only when the engine started, so it kept answering. Returns whether it stopped.

- **is_available**(`self`) -> `None`
  > Check if RAG dependencies are available.

- **get_available_models**(`self, provider: ComplexType, base_url: str`) -> `None`
  > Get list of available models for the specified provider (the engine's own by default).

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
  >
  > Without a ``provider``, the engine's own (the server's ``CMP_AI_PROVIDER``, local Ollama by default); without
  > a model, the first its provider declares.

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

### def `_passages` (rag_manager.py)

- **Arguments**: `docs`

- **Returns**: `str`

The passages the chat answers from, each with its document's name (never its path on the server) and page.

### def `_env_api_key` (rag_manager.py)

- **Arguments**: `provider: ComplexType`

- **Returns**: `ComplexType`

Server-configured API key for a cloud provider (the setting it declares); used only when the user typed none.

### def `_chat_only` (rag_manager.py)

- **Arguments**: `provider: str`

- **Returns**: `bool`

A provider that answers but has no embedding API (Claude): another provider embeds the documents.

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
`json`, `manage.ai_payload.frequent_values`, `manage.ai_payload.safe_stats`

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
  > Merges statistical info from columns_info into the taxonomy, only what may be sent (manage.ai_payload):
  > later prompts carry the taxonomy.

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
source dataset (or a pre-loaded frame) up to an optional step index. It reads the files a trace names
through the resolver it is given (:class:`manage.paths.PathResolver`: the user's workspace and the files
chosen in the UI, or a command line's files); given none, it reads no file. :func:`execute_trace_pipeline`
keeps the historical ``(df, error)`` signature.

Traces recorded since schema 2 carry, for every step, the fingerprint of the table it produced. The
replay checks each replayed step against it and reports the first *divergence*; steps that cannot be
recomputed (an adopted multiple imputation, a table replaced outside the app) are listed as not
replayable, and their recorded snapshot is used when there is one. A replay is *reproducible* when every
step was replayed and matched._

**Imports**:
`__future__.annotations`, `logging`, `os`, `collections.abc.Callable`, `collections.abc.Mapping`, `dataclasses.dataclass`, `dataclasses.field`, `typing.Any`, `numpy`, `pandas`, `enrich.external_data`, `core.fingerprint.ensure_row_ids`, `core.fingerprint.frame_fingerprint`, `core.fingerprint.ranges_to_ids`, `core.population.criteria_from_trace`, `core.population.evaluate_population`, `core.registry.UnknownIdError`, `core.session.restrict_cells`, `core.session.sort_filled`, `core.state.ParseOptions`, `core.steps.DOCUMENT`, `core.steps.SNAPSHOT`, `core.steps.ids`, `enrich.derived.APPROACHES`, `enrich.derived.impute_and_derive`, `enrich.formulas.evaluate_formula_safely`, `enrich.formulas.read_formula`, `enrich.imputation_service.describe_formula_fill`, `enrich.imputation_service.describe_imputation`, `enrich.imputation_service.fill_with_formula`, `enrich.imputation_service.impute_rows`, `enrich.imputation_service.left_missing`, `enrich.transformation_logic.apply_variable_transformation`, `manage.paths.PathResolver`, `manage.paths.read_version`, `utils.date_parser.smart_parse_dates`

### class `StepRecord` (reproduction_manager.py)

What a recomputed step did on the replayed data, in the trace's terms (its settings and description): an
imputation's counts and rows depend on the data, so a redone Enrich draft step is recorded with these.

### class `ReplayResult` (reproduction_manager.py)

No description available.

**Methods:**

- **reproducible**(`self`) -> `bool`
  > Every step replayed, and every recorded fingerprint matched.

### class `_NotReplayable` (reproduction_manager.py)

A step the engine cannot recompute; carries the frame to continue from.

### def `_resolve` (reproduction_manager.py)

- **Arguments**: `path: ComplexType, paths: ComplexType`

- **Returns**: `ComplexType`

The file a recorded path stands for; None without a resolver (a replay given none reads no file).

### def `_version` (reproduction_manager.py)

- **Arguments**: `name: str, paths: ComplexType`

- **Returns**: `ComplexType`

A saved version a trace names, from the datasets folder only (PathResolver.dataset).

### def `_read_any` (reproduction_manager.py)

- **Arguments**: `path: str, parse_options: ComplexType, paths: ComplexType`

- **Returns**: `ComplexType`

Read csv/tsv/txt/xlsx/xls/parquet through the import engine; other names are saved versions.

### def `load_source` (reproduction_manager.py)

- **Arguments**: `trace_data: dict[ComplexType], paths: ComplexType`

- **Returns**: `tuple[ComplexType]`

The source dataset of a trace: the recorded artifact (newer traces) or the original path/name.

### def `_population_rows` (reproduction_manager.py)

- **Arguments**: `df: pd.DataFrame, block: dict[ComplexType], index: int, warnings: list[str], frames: ComplexType`

- **Returns**: `pd.Index`

Rows of ``df`` kept by the population recorded in a step (criteria re-evaluated as they were), on the table it was
evaluated on when ``frames`` holds it (by its fingerprint), else on ``df``. Every step of an Enrich draft is scoped
to the population of the data before the draft, which a fill of the same draft can change: a weight filled above
an anomaly rule's bound must not take its row out of the next fill.

### def `_snapshot` (reproduction_manager.py)

- **Arguments**: `step: dict[ComplexType], paths: ComplexType`

- **Returns**: `ComplexType`

No description available.

### def `_enrichment` (reproduction_manager.py)

- **Arguments**: `df, step, index, paths, warnings`

- **Returns**: `None`

No description available.

### def `_with_inputs` (reproduction_manager.py)

- **Arguments**: `step: dict[ComplexType]`

- **Returns**: `ComplexType`

A targeted imputation's params with the variables of its formula other than its target ("inputs"), which
earlier versions did not record: a value an exact formula computed from a single-imputed one, or from none, is
single-imputed (core.session.sort_filled).

### def `_scoped_rows` (reproduction_manager.py)

- **Arguments**: `df: pd.DataFrame, step: dict[ComplexType], index: int, warnings: list[str], result: ComplexType`

- **Returns**: `ComplexType`

The rows an imputation is fitted on and fills: the population recorded with the step, as when it was applied;
None (every row) when none was recorded, as by earlier versions, or when no criterion was ticked.

### def `_imputation` (reproduction_manager.py)

- **Arguments**: `df, step, index, paths, warnings, result`

- **Returns**: `None`

No description available.

### def `_targeted_imputation` (reproduction_manager.py)

- **Arguments**: `df, step, index, paths, warnings, result`

- **Returns**: `None`

No description available.

### def `_variable_transformation` (reproduction_manager.py)

- **Arguments**: `df, step, index, paths, warnings`

- **Returns**: `None`

No description available.

### def `_variable_computation` (reproduction_manager.py)

- **Arguments**: `df, step, index, paths, warnings`

- **Returns**: `None`

No description available.

### def `_data_type_conversion` (reproduction_manager.py)

- **Arguments**: `df, step, index, paths, warnings`

- **Returns**: `None`

No description available.

### def `_missing_value_handling` (reproduction_manager.py)

- **Arguments**: `df, step, index, paths, warnings`

- **Returns**: `None`

No description available.

### def `_rename_columns` (reproduction_manager.py)

- **Arguments**: `df, step, index, paths, warnings`

- **Returns**: `None`

No description available.

### def `_apply_filter` (reproduction_manager.py)

- **Arguments**: `df, step, index, paths, warnings`

- **Returns**: `None`

No description available.

### def `_population_definition` (reproduction_manager.py)

- **Arguments**: `df, step, index, paths, warnings`

- **Returns**: `None`

No description available.

### def `_mi_impute` (reproduction_manager.py)

- **Arguments**: `df, step, index, paths, warnings, result`

- **Returns**: `None`

Re-run a multiple imputation on its population and check the imputed cells against the recorded hash.

### def `_freeze_population` (reproduction_manager.py)

- **Arguments**: `df, step, index, paths, warnings`

- **Returns**: `None`

No description available.

### def `_outlier_clipping` (reproduction_manager.py)

- **Arguments**: `df, step, index, paths, warnings`

- **Returns**: `None`

No description available.

### def `_apply_step` (reproduction_manager.py)

- **Arguments**: `df: pd.DataFrame, step: dict[ComplexType], index: int, paths: ComplexType, warnings: list[str], result: ComplexType`

- **Returns**: `tuple[ComplexType]`

Apply one recorded step; returns (df, missing_path). Raises _NotReplayable for steps it cannot redo.

### def `apply_steps` (reproduction_manager.py)

- **Arguments**: `df: pd.DataFrame, steps: list[dict[ComplexType]]`

- **Returns**: `ReplayResult`

Replay ``steps`` on ``df`` (step numbers in messages start at ``start``), checking fingerprints; ``paths`` finds
the files the steps name (none read without it). ``frames``:
tables a step's population may name by fingerprint although the replay does not pass through them (a draft
redone one change at a time names the dataset it was built on).

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
`enum.Enum`, `typing.Any`, `pydantic.BaseModel`, `pydantic.Field`, `pydantic.field_validator`, `pydantic.model_validator`, `providers.default_models`, `providers.declarations.Ollama`

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

- **_declared_models**(`cls, data: Any`) -> `Any`
  > A provider's configuration without models takes those it declares first (manage.rag.providers.default_models;
  > a custom endpoint, OpenAI's): a Gemini configuration never holds the fields' defaults, Ollama's names.

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

- :func:`select_engine` chooses the engine: the session's choice, the saved preference, the server's setting, then
  the recommendation (through the registry, :data:`manage.vectorstores.VECTOR_STORES`).

- :func:`open_index` opens the selected engine in the user's workspace._

**Imports**:
`__future__.annotations`, `ctypes`, `os`, `platform`, `shutil`, `sys`, `dataclasses.dataclass`, `dataclasses.field`, `pathlib.Path`, `core.registry.AUTO`, `core.registry.Selection`, `manage.vectorstores.VECTOR_STORES`, `utils.settings.Settings`, `utils.settings.SettingsError`, `utils.settings.get_settings`, `base.BaseVectorIndex`, `base.VectorEngine`, `base.VectorStoreError`, `lancedb_index.ANN_THRESHOLD`

### class `VectorStoreSettings` (selection.py)

No description available.

**Methods:**

- **from_env**(`cls, settings: ComplexType`) -> `VectorStoreSettings`
  > The vector store's settings (utils.settings; the current ones by default).

- **server_for**(`self, asked: ComplexType`) -> `ComplexType`
  > The Qdrant server to use: the address asked for, None for embedded (an empty one), or the server's
  > (CMP_QDRANT_URL) when none is asked for.

- **key_for**(`self, url: ComplexType`) -> `ComplexType`
  > The server's API key (CMP_QDRANT_API_KEY), for its own address (CMP_QDRANT_URL) only: it is never sent to an
  > address a user typed.

### class `MachineProfile` (selection.py)

No description available.

### class `Recommendation` (selection.py)

No description available.

### def `_same_address` (selection.py)

- **Arguments**: `first: str, second: str`

- **Returns**: `bool`

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

### def `select_engine` (selection.py)

- **Arguments**: ``

- **Returns**: `Selection`

The engine to use: the session's choice, then the saved preference, then the server's setting, then the engine
recommended for this machine (with its reasons), then the family's order. A choice that cannot run here falls
through, and the selection says why.

### def `open_index` (selection.py)

- **Arguments**: `engine: str`

- **Returns**: `BaseVectorIndex`

Open the engine's index in ``base_dir/<engine>`` (on a server when the engine has one and a URL is given).

---

## services.py

_The services of the application, built in one place (M5b step 5, the composition root).

- :class:`Shared`, one per server process, holds nothing of a user's: the settings, the identity provider, the graph
  renderers and the multiple-imputation worker budget, each read from its process-wide instance at each access (a
  test that installs another finds it; a copy kept here would be stale). A command line passes the settings it
  loaded instead.

- :class:`Services`, one per session (a user in a browser tab, or one command-line run): the user's scope, the
  resolver of the files a trace names, and the dataset store and the trace, built on first use with that scope, so
  that a command line that saves nothing creates no folder.

The module is headless (rule R5): it reaches neither Streamlit nor the AI packages. The session builds the AI engine
(``ui.services.ai_engine``), which needs them, and ``main.py`` checks the connector settings._

**Imports**:
`__future__.annotations`, `dataclasses.dataclass`, `functools.cached_property`, `typing.TYPE_CHECKING`, `core.workspace.UserScope`, `manage.paths.PathResolver`, `utils.settings.Settings`

### class `Shared` (services.py)

One per server process; never a user's data.

**Methods:**

- **settings**(`self`) -> `Settings`
  > No description available.

- **iam**(`self`) -> `IAMManager`
  > No description available.

- **graphs**(`self`) -> `GraphRendererManager`
  > No description available.

- **mi_workers**(`self`) -> `WorkerBudget`
  > No description available.

### class `Services` (services.py)

One session's services. ``scope`` is the signed-in user's workspace; None for no user (tests, scripts, the command
line), whose stores use the shared folders. Never passed to a worker process: the budget holds a lock.

**Methods:**

- **datasets**(`self`) -> `DBManager`
  > The dataset store (saved versions and statistics), in the user's workspace.

- **trace**(`self`) -> `TransformationManager`
  > The session's trace, in the user's workspace.

### def `build_shared` (services.py)

- **Arguments**: `settings: ComplexType`

- **Returns**: `Shared`

The process's shared services. Builds nothing: each one is built on its first use, as before. ``settings``: those
a command line loaded (``utils.settings.load_settings``); without them, the process's current ones.

### def `build_services` (services.py)

- **Arguments**: `shared: Shared, scope: ComplexType, paths: PathResolver`

- **Returns**: `Services`

One session's services. ``datasets`` and ``trace`` are stores the session already holds for the same user, kept
rather than built again: a session that outlives a code change keeps its trace.

### def `same_user` (services.py)

- **Arguments**: `first: ComplexType, second: ComplexType`

- **Returns**: `bool`

Whether two scopes are the same user's workspace: compared by folder, which survives a reload of
``core.workspace`` (a reloaded class makes two equal scopes unequal) and a change of role.

### def `_folder` (services.py)

- **Arguments**: `scope: ComplexType`

- **Returns**: `ComplexType`

No description available.

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

## snapshots.py

_The snapshots of the Enrich draft (M5b).

Each change prepared on the Enrich page writes a copy of the table to the snapshots folder of the workspace
(``traces/snapshots``) before it joins the draft. Applying the draft writes the snapshot's path into the trace: the
replay uses it for a step it cannot recompute, and Reproduce loads it in its fast mode. A change that never reaches
the trace left its copy behind: a draft discarded, redone on newer data or dropped when another dataset is loaded,
and a change that changed nothing. :func:`delete_unreferenced` deletes those copies.

A file is deleted only when it is a ``.parquet`` file directly in the snapshots folder and no trace of the workspace
names it. The replay also finds a file by its name alone (:class:`manage.paths.PathResolver`), so a trace that names a
file of the same name in any folder keeps it. When a trace cannot be read, nothing is deleted._

**Imports**:
`__future__.annotations`, `contextlib`, `json`, `logging`, `os`, `uuid`, `collections.abc.Iterable`, `collections.abc.Iterator`, `collections.abc.Mapping`, `datetime.datetime`, `pathlib.Path`, `typing.Any`, `pandas`, `core.workspace.UserScope`, `core.workspace.resolve_within`, `utils.data_paths.get_data_root`, `utils.data_paths.get_traces_dir`, `utils.logging_config.single_line`

### def `_reason` (snapshots.py)

- **Arguments**: `exc: BaseException`

- **Returns**: `str`

Why an operation failed, without the full path (it names the server's account).

### def `folders` (snapshots.py)

- **Arguments**: `scope: ComplexType`

- **Returns**: `tuple[ComplexType]`

The folder that recorded paths are relative to, the traces folder and the snapshots folder of ``scope``; outside
a session, those of the data folder (as ``AppState.recorded_path`` and the Enrich page use them).

### def `_name` (snapshots.py)

- **Arguments**: `value: str`

- **Returns**: `str`

The file name of a recorded path, whatever its separators (a trace recorded on Windows keeps backslashes).

### def `_parquet_names` (snapshots.py)

- **Arguments**: `value: Any`

- **Returns**: `Iterator[str]`

The names of the .parquet files mentioned anywhere in a trace.

### def `referenced` (snapshots.py)

- **Arguments**: `traces: Path, steps: Iterable[Mapping[ComplexType]]`

- **Returns**: `ComplexType`

The names (case-folded) of the .parquet files named by the traces saved in ``traces`` and by ``steps`` (the
session's own trace); ``None`` when a trace cannot be read.

### def `snapshot_file` (snapshots.py)

- **Arguments**: `value: ComplexType, root: Path, snapshots: Path`

- **Returns**: `ComplexType`

The file a recorded path names when it is a .parquet file directly in ``snapshots``, otherwise ``None``.

### def `save` (snapshots.py)

- **Arguments**: `frame: pd.DataFrame, step_name: str, folder: Path`

- **Returns**: `ComplexType`

Write a copy of ``frame`` to ``folder`` (a snapshots folder, :func:`folders`) for a change of the Enrich draft;
returns its path, or ``None`` when it cannot be written (a copy written part-way is removed: it holds data).

### def `delete_unreferenced` (snapshots.py)

- **Arguments**: `paths: Iterable[ComplexType], scope: ComplexType, steps: Iterable[Mapping[ComplexType]]`

- **Returns**: `list[str]`

Delete the snapshots among ``paths`` (recorded paths of changes that will never reach the trace) that no trace
names, ``steps`` included; returns the names deleted. Best effort: a file that cannot be deleted is logged and
kept.

---

## trace_documenter.py

_No module description._

**Imports**:
`json`, `os`, `datetime.datetime`, `docx.Document`, `docx.enum.text.WD_ALIGN_PARAGRAPH`, `docx.shared.Pt`, `docx.shared.RGBColor`, `core.lineage.verify_trace`, `core.steps.step_title`

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
`contextlib`, `json`, `logging`, `os`, `datetime.datetime`, `typing.Any`, `pandas`, `core.lineage.TRACE_SCHEMA`, `core.lineage.genesis_hash`, `core.lineage.seal_step`, `core.lineage.verify_trace`, `core.steps.STEPS`, `core.workspace.UserScope`, `utils.data_paths.get_traces_dir`, `utils.settings.get_settings`

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
  >      Defaults to the traces folder of ``scope``, else DATA_ROOT/traces via data_paths utility.
  > scope : UserScope, optional
  >      The user's workspace, passed by the session's services (manage.services); None outside a
  >      session.

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
