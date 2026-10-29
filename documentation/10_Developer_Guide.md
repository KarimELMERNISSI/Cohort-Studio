# Developer Guide

This guide covers the architecture, design patterns, and key subsystems of CohortStudio for developers who want to understand, extend, or maintain the application.

---

## Architecture

CohortStudio is a set of Python packages in layers. Each layer imports only the layers below it: the statistics engine
runs without the web interface, and the pages hold no logic of their own. Two mechanisms carry every extension point,
one place builds the services, and a fixed set of tests keeps all of this true as the code grows.

### The layers

| Layer | Packages | Role |
| --- | --- | --- |
| L9 | `app_pages` (the page files in `app_pages/views/`) | the pages: inputs, a service call, a result card |
| L8 | `ui` | the shell, `AppState`, the components, `ui/services.py` |
| L7 | `manage` | application services, stores and connectors, in groups: iam, ai, repro, storage |
| L6 | `graph` | the graph model, its builders and its renderers |
| L5 | `explore`, `monitor` | the quality audit and correlations; outliers and comparisons |
| L4 | `enrich` | formulas, transformations, imputation, merging |
| L3 | `ingest` | reading files, with one loader |
| L2 | `core` | the domain kernel, and the extension kit (`core/registry.py`) |
| L1 | `biostats`, `prompts` | the statistics library, which imports nothing else of the project; prompt texts |
| L0 | `utils` | settings, logging, optional extras, paths, export helpers |

`main.py` and `manage_users.py` sit above every layer. The platform modules of `utils` (`utils.settings`,
`utils.optional`, `utils.data_paths`, ...) are L0; the older ones count with the package they serve (`UTILS_GROUPS` in
the test) until a milestone moves them. The groups of `manage` are not ordered among themselves yet.

**Rules:**
- **R1:** imports go strictly down, and two groups on the same layer do not import each other.
- **R2:** no import cycles. The one exception is the multiple-imputation engine and its worker module
  (`biostats.mi.engine`, `biostats.mi.parallel`), by design.
- **R3:** nothing below `ui` reaches Streamlit, through any chain of imports. A graph renderer imports it inside its
  `render` method only (and yFiles' component package inside `translate`).
- **R4:** the AI providers' SDKs are imported only in `manage/rag/providers`.
- **R5:** the headless entry points (`manage_users`, `manage.reproduction_manager`, `manage.services`, and
  `manage.replay` from M8) reach neither Streamlit nor the packages of the `ai` extra.

**How `tests/test_layering.py` holds them.** It reads the import graph from the source: an import inside a function
counts, one under `if TYPE_CHECKING:` does not. R1 and R2 use the imports as written; R3 and R5 follow what runs,
parent packages and modules named in strings (the lazy registries) included. Today's exceptions are listed at the top
of the test (`KNOWN_UPWARD`, `CYCLES_BY_DESIGN`, `KNOWN_STREAMLIT`, `KNOWN_SDK`, `KNOWN_ENVIRONMENT`) and only shrink:
a new violation fails, and so does a listed one that no longer occurs, until it is deleted. The same file checks that
every package of the wheel has a layer, that `core`, `biostats` and `ingest` load no UI, plotting or AI library, that
`biostats` imports only itself, that only `utils.settings` reads the environment, and that `core` and `biostats`
never read the settings.

### Two extension mechanisms

Every extension point takes one of two shapes, both in `core/registry.py` (standard library only):

| | Connector family (`ConnectorRegistry`) | Catalogue (`Catalogue`) |
| --- | --- | --- |
| What it holds | the implementations of one service, one used at a time | kinds side by side, each named by an id |
| Who chooses | the session, then the preference, the setting, `auto` | the data: a trace or a rule names its kind |
| Registered as | `"module:Class"`, imported on first use | one frozen dataclass per kind |
| An unknown name | in a setting, stops start-up with the valid names | `UnknownIdError`, listing the valid ids |
| Used by | identity providers, AI providers, vector databases, graph renderers | trace steps, criterion kinds |
| Tested by | `tests/connector_contract.py`, called by each family's tests | one consistency test per catalogue |

**A connector family.** Each connector is a class with `name`, `display_name` and a class method `availability()`
that answers without raising or connecting: an `Availability(ok, reason, extra)`, where `extra` is the optional group
to install (`Availability.of_module("lancedb", extra="ai")` finds a package without importing it). The registry
imports a connector only when it is used, so a missing library never breaks the others. `select(...)` returns a
`Selection` (`chosen`, `source`, `reasons`, `warnings`, `alternatives`): the session's choice, then the saved
preference, the server's setting and the family's own order; a name that is unknown or cannot run here falls through,
with a warning that shows it as code. An unknown `CMP_*` value stops start-up instead (`main.py` calls
`manage.connectors.check_connector_settings` after loading the settings), naming the variable and the valid names,
never the value, which may be a pasted key. An old spelling stays accepted as an alias.

| Family | Registry | Setting | Automatic order |
| --- | --- | --- | --- |
| Identity providers | `manage.iam.providers.IAM_PROVIDERS` | `CMP_IAM_PROVIDER` | none: the setting only (`local`) |
| AI providers | `manage.rag.providers.AI_PROVIDERS` | `CMP_AI_PROVIDER` | `ollama` |
| Vector databases | `manage.vectorstores.VECTOR_STORES` | `CMP_VECTOR_STORE` | a recommendation, `lancedb`, `qdrant` |
| Graph renderers | `graph.renderers.GRAPH_RENDERERS` | `CMP_GRAPH_RENDERER` | `plotly`, `graphviz`; never `yfiles` |

Settings shows each choice and its reasons with one control, `ui.components.connector_selector`.

**A catalogue.** `register(kind, legacy=False, aliases=())` adds a kind, and the catalogue lists its kinds in the
order they were registered. `get(id)` finds a kind by its id or an alias, and raises `UnknownIdError` (a
`LookupError`) for any other; `ids(legacy=False)` lists what is still offered. An id stored in a trace, a
configuration or a result is never renamed: a kind that nothing writes any more stays registered with `legacy=True`,
so that what was stored still reads. A test that needs a kind of its own registers it in `copy()`, patched over the
module's catalogue for that test only. The catalogues are `core.steps.STEPS`, the kinds of trace step, and
`core.criteria.CRITERION_KINDS`, the criterion kinds (M6's date order and M9's concepts will be criterion kinds).

### The composition root

- **`manage/services.py`**, headless (rule R5); the AI engine, which needs the AI packages, is built in `ui`:
  - `build_shared(settings=None)` returns `Shared`, one per server process: the settings, the identity provider
    (`iam`), the graph renderers (`graphs`) and the multiple-imputation worker budget (`mi_workers`), each read from
    its process-wide instance when used. A command line passes the settings it loaded.
  - `build_services(shared, scope, paths)` returns `Services`, one per session: the user's scope
    (`core.workspace.UserScope`), the resolver of the files a trace names (`manage.paths.PathResolver`), and the
    dataset store (`datasets`) and the trace (`trace`), built on first use with that scope.
- **`ui/services.py`** keeps them: `shared()` for the process (`st.cache_resource`); `services()` in the session
  state (`_services`; the stores also under `db_manager` and `transformation_manager`, which pages read); and
  `ai_engine()`, built on first use, under `rag_manager`. A session holding anything of another user is ended
  (`end_session`); a dropped engine is stopped first (`drop_ai_engine`). Not `st.cache_resource(scope="session")`:
  its entries go when the browser's connection drops, while the session state survives.

**Why pages build no service.** A service takes the user's scope as a parameter and never asks the session for the
user: its files land in that user's workspace, and the command line of M8 builds the same services with a resolver
limited to the files it names. One AI engine per session means that a withdrawn consent stops the only engine that
could send data. `tests/test_quality_gates.py` holds this (`test_pages_never_build_services`,
`test_the_session_builds_its_services_with_the_users_workspace`, `test_only_the_services_write_their_keys`), and
`tests/test_services.py` builds the services without Streamlit or the AI packages.

### Settings and preferences

- **`utils/settings.py`** holds one `Settings` object, in sections (`paths`, `logging`, `iam`, `ai`, `vector_store`,
  `graph`, `statistics`, `development`), with keys and passwords as `SecretStr`. `main.py` and `manage_users.py` call
  `load_settings()` first: it loads `.env` (a variable already set wins) and checks every value, and a wrong one
  stops start-up with the variable's name, never its value. Elsewhere, read them with `get_settings()`.
- **One source.** `utils.settings` is the only module that reads the environment (`tests/test_layering.py`), and
  `.env.example` lists exactly its variables (`tests/test_settings.py`). `.streamlit/secrets.toml` is not read; its
  presence is logged. `core` and `biostats` never read the settings: they receive values as parameters.
- **Preferences** are the non-secret choices made in Settings: the day-first default, the AI provider and its models,
  the vector database, the graph renderer. `manage/preferences.py` keeps them per user in
  `<workspace>/preferences.json` (`load_preferences`, `update_preferences`; written whole and atomically), and
  `ui/preferences.py` seeds the session after sign-in (`seed_session`). A preference never holds a key, a password, a
  consent or an address (`tests/test_preferences.py`). Details: *Settings, preferences and API keys*, below.

### Session state

**Declared keys.** A session key that one module writes and another reads is declared in `ui/keys.py` as a `Key`:
its `name`, its `owner` (the module that writes it), its `lifetime`, its `readers` and a `note`. The lifetime decides
what `AppState` clears:

| Lifetime | Kept | Cleared by `AppState` |
| --- | --- | --- |
| `session` | for the whole session | never |
| `page` | while the user stays on its page | with another dataset too, when `reset_on_dataset` is set |
| `dataset` | while the same dataset changes | when another dataset is loaded |
| `data_change` | until the data change | on any change of the data, and with another dataset |

`ui.state.DERIVED_KEYS`, which `AppState.set_dataset` and `clear_dataset` clear, and `KEPT_ON_DATA_CHANGE` are derived
from the declarations; after a change of the data, `AppState` clears the others (`DATA_DERIVED_KEYS`).
`tests/ui_layer/test_session_keys.py` checks each declaration against `AppState` and against the code of its owner
and readers, and a quality gate counts the keys written without a declaration: that count can only fall.

**The tables.** `AppState` (`ui/state.py`) is the only code that changes the data, through
`core.session.DataSession`: `commit_change()` for the dataset, `stage()` for the Enrich draft, `note()` for a step
that changes nothing. Each change is one trace step; with `CMP_STRICT_STATE=1` (the test suite), changing a view
below in place raises `UntracedMutationError`.
- `working_df` (and `data`) is the analysis population: the rows the ticked criteria keep, as a view reinstalled at
  the start of every run (`AppState.begin_run`). While a ticked criterion cannot be evaluated, it holds every row and
  a banner stops the analyses; `AppState.analysis_frame()` raises `PopulationError` instead.
- `AppState.draft_frame()` is what Enrich works on: the draft when there is one, otherwise every row of the dataset,
  as a copy. The `enriched_df` key shows the same table.

### Runtime

- **Only the open tab runs** on Validate (`app_pages/data_monitoring.py`), Statistics & correlations
  (`app_pages/home.py`) and Hypothesis tests & epidemiology (`app_pages/epidemiology.py`):
  `st.tabs(list(TABS), key=TAB_KEY, on_change="rerun")`, then `if tab.open:` around each tab's body. Validate's
  Compare tab always runs: its file uploaders cannot keep a file while it is closed. The tab keys are declared in
  `ui/keys.py`, and every control of these pages has a `key` and, but for the exceptions the test lists,
  `persist_state="page"`, so a closed tab keeps its values (`tests/ui_layer/test_lazy_tabs.py`).
- **Computations are kept by fingerprint.** `DataSession.memo(name, key, compute)` keeps a value until the data
  fingerprint or the caller's key changes. The population is kept by the data fingerprint and the fingerprints of the
  ticked criteria; `AppState.analyzer()` keeps the column profile, and `AppState.quality_audit()` the audit, keyed by
  the criteria, the derived variables and the method. The statistics of Statistics & correlations are cached per
  session (`st.cache_data(scope="session")`), keyed by the table's fingerprint, and their Reset clears that session's
  only; the correlation workbooks are cached eight at most, for an hour.
- **One worker budget per process for the multiple imputation.** `manage.compute.mi_workers()` returns the
  `WorkerBudget`: a run leases what is free (`lease(wanted)`), never waits, and gives it back however it ends. With
  fewer than two workers free, the run takes place in the session's own process. `CMP_MI_MAX_WORKERS` sets the total
  (the processors minus one by default; a container's CPU quota is not visible there, so set it).
- **A discarded Enrich draft deletes its snapshot files**, but never one that a saved trace uses
  (`manage.snapshots.delete_unreferenced`).

### Keeping it so

| Check | Where | What may only move one way |
| --- | --- | --- |
| Layers, R1–R5 | `tests/test_layering.py` | the exception lists shrink |
| Lint | `ruff check` (CI, pre-commit) | |
| Formatting | `ruff format --check` (CI, pre-commit) | `[tool.ruff.format] exclude` in `pyproject.toml` shrinks |
| Types | `dev_scripts/check_type_baseline.py` (pyright, CI) | the errors per file of `dev_scripts/pyright_baseline.json` fall |
| Coverage | `dev_scripts/check_coverage_floors.py` (CI) | a floor per package, raised as coverage grows |
| Source rules | `tests/test_quality_gates.py` | ceilings that only fall |

- **Formatting and types.** A file leaves the exclude list in a commit that only reformats it, listed in
  `.git-blame-ignore-revs`. A type error in a file outside the baseline fails; a count that falls is written down
  with `--update`, which never raises one.
- **Coverage.** CI fails below 70 % in all, and below each package's floor (`FLOORS`); a package more than 4 points
  above its floor (`MARGIN`) fails too, until the floor is raised.
- **Source rules.** No `eval` outside the expression evaluator; only `AppState` writes the dataset and the trace
  steps; pages build no service; rules are written through their kinds. Raw HTML, `print()` calls, undeclared
  session keys and unformatted files each have a ceiling that only falls.

### Recipes

#### Add a connector

1. Write the class in its family's package, on the family's base class (`BaseIAMProvider`, `BaseAIProvider`,
   `BaseVectorIndex`, `BaseGraphRenderer`), with `name`, `display_name` and `availability()`, which never raises,
   never connects, and names the optional group its package comes from.
2. Register it in the family's registry, for example `GRAPH_RENDERERS.register("name", "graph.renderers.name:Class",
   extra="group")`. The name is stored in preferences and settings: never rename it; an old spelling is an alias.
3. A package it needs goes into an optional group of `pyproject.toml` and into `utils.optional.EXTRAS`
   (`tests/test_ai_provider_sdks.py` keeps them equal). An AI provider imports its SDK in `manage/rag/providers`
   only (R4); a renderer imports Streamlit in `render` only (R3).
4. Nothing else changes: the setting accepts the name, and Settings offers it where it can run (not an identity
   provider, which the setting alone chooses).

Test: add the name to the family's `test_the_stored_names_never_change`; the family's tests run the checks of
`tests/connector_contract.py` on it (availability, light listing, a missing package, precedence, unknown names).

#### Add a kind of trace step

1. Register a `StepKind` in `core/steps.py` with `_register(StepKind(id, title, category, replay, changes_data=...))`:
   `category` is one of `CATEGORIES`, `replay` is `RECOMPUTE`, `SNAPSHOT` or `DOCUMENT`, and `flow` and `fills` are
   given when they apply. `title` is its heading in the trace report (`step_title`, read by
   `manage/trace_documenter.py`).
2. Record it through `AppState` (`commit_change`, `stage` or `note`), never with `add_step` (a quality gate).
3. A step that the replay recomputes gets its handler in `_HANDLED`, in `manage/reproduction_manager.py`.
4. Never rename an id: a kind that nothing records any more is registered with `legacy=True`.

Test: `tests/core_layer/test_steps.py` (every step written is registered, the report titles every kind, the replay
handles each kind by its mode); with `CMP_STRICT_STATE=1`, an unregistered step raises `UnknownIdError`.

#### Add a criterion kind

See *Adding a criterion kind*, below: its functions in a module of its own, one registration in `core/criteria.py`
after the five kinds of M5, a writer, and its examples in `tests/core_layer/test_criterion_kinds.py`.

#### Add an analysis

Follow `app_pages/analysis_survival.py`:
1. The computation goes into `biostats`: plain values in, a result dataclass out, no import of the application.
2. In the page module, every control has a key. On the button: compute, then `remember(key, result, config=...,
   n_analysed=..., columns=...)` (`ui/results.py`) and `mark_step("analyse", ...)`; an exception goes to
   `error_card(...)`.
3. On every run, `recall(key, config=config)` returns the result only while its data, population and settings still
   hold. Show it, then call `provenance(key)`, which says what it was computed on and offers to add it to the dossier.
4. Give the key a title in `RESULT_TITLES` (`ui/results.py`; without one, the dossier shows the key in words). A new
   type of result gets its tables in `manage/report_tables.result_blocks`, which gives an unknown type a short text.
5. Compare its numbers with R: the reference computation goes into a script of `tests/reference/r/`, the manual
   workflow `.github/workflows/r-fixtures.yml` runs it, its values are copied into `tests/reference/fixtures/`, and a
   parity test in `tests/biostats_layer/` (as `test_rates_r_parity.py`) compares, skipped until they are there.

Test: an AppTest of the page (as `tests/ui_layer/test_survival_page.py`), and the module added to the list of
`test_analysis_pages_keep_results_in_the_result_store` in `tests/test_quality_gates.py`.

#### Add a page panel

1. Write it in a module of its own, with one `render_...()` function called in its tab, as
   `app_pages/enrichment_derived.py` (`render_derived_variables`) is in Enrich: a long page function does not grow.
2. On a page that runs only its open tab, the panel goes inside `if tab.open:`. For a link to open its tab, add the
   tab's id and labels to the page's `PageDef.tabs` in `ui/nav.py`, build the row with `linked_tabs(...)`, and link
   with `page_link(..., tab=...)` or `switch_to(..., tab=...)`, which refuse a tab the page does not declare.
3. Every control has a `key` and `persist_state="page"`, except a confirmation and an input whose value comes from a
   session key that already keeps it. A tab's label never changes (a count in it would reset the open tab).
4. A key it shares with another tab or page is declared in `ui/keys.py`, with the lifetime it needs.
5. It changes the data through `AppState` and takes its services from `ui/services.py`.

Test: an AppTest in `tests/ui_layer/`, its module in `LAZY` (`tests/ui_layer/test_lazy_tabs.py`) when its page runs
only the open tab, and `tests/ui_layer/test_session_keys.py`.

---

## RAG Manager (Mixin Pattern)

The RAG system is structured using the **Mixin composition pattern** to keep each concern in its own module:

```text
RAGManager(TaxonomyMixin, DocumentsMixin, ComputedVarsMixin)
├── rag_taxonomy.py      # Taxonomy generation & refinement
├── rag_documents.py     # PDF processing, summarisation, knowledge graphs
├── rag_computed_vars.py # Variable suggestion & imputation formulas
└── rag_manager.py       # Orchestrator: initialisation, vector index, shared utilities
```

### Key Design Decisions

- **The vector database is a connector** in `manage/vectorstores/`: **LanceDB** (embedded, on disk; the
  default) or **Qdrant** (embedded in memory, or a server). The RAG code only sees `DocumentIndex`
  (`similarity_search`, `max_marginal_relevance_search`, `as_retriever`, `build`), so it is identical whatever the
  engine; filtering by document is a list of sources that each engine translates (SQL `IN` for LanceDB, a `MatchAny`
  payload filter for Qdrant). See "Vector databases" below.
- **AI providers are pluggable connectors** in `manage/rag/` (local Ollama by default; Google Gemini, OpenAI, Mistral and Anthropic Claude as cloud providers that require an explicit data-sharing consent).
- **Anthropic Claude** (`manage/rag/providers/claude.py`, official `anthropic` SDK) answers but has no embedding
  API. `ConnectorFactory.create_connectors(provider="anthropic", embedding_provider=...)` pairs it with another
  provider's embeddings (local Ollama by default, so the library is indexed on the server and only the retrieved
  passages reach Claude); the index manifest records that embedding provider. Claude models choose their own
  sampling, so the app's temperature is not sent. Model IDs: `claude-fable-5-1` (default, most capable),
  `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`.
- Each mixin accesses shared resources (`self.vectorstore`, `self.llm`) from the base `RAGManager`.

---

### Vector databases

| Engine            | Mode     | Where vectors live                                                                          | Suits                                                                                                       |
| ----------------- | -------- | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| LanceDB (default) | embedded | `<workspace>/vectors/lancedb/`, columnar Lance files on disk                                | one machine, any library size; low memory; exact search below 50 000 chunks, an IVF index above             |
| Qdrant            | embedded | `<workspace>/vectors/qdrant/storage/`, every vector in memory                               | small and medium libraries (up to ~20 000 chunks); one process per folder (the client is shared per folder) |
| Qdrant            | server   | a Qdrant service (`CMP_QDRANT_URL`, optional `CMP_QDRANT_API_KEY`), one collection per user | several users, very large libraries, concurrent sessions; HNSW search and payload indexes                   |

- **Choice** — Settings > AI engine > _Vector database_. The default comes from `CMP_VECTOR_STORE` (`auto`, `lancedb`
  or `qdrant`); `auto` follows `manage.vectorstores.recommend()`, which weighs the machine (CPU cores, memory, free
  disk, probed without extra dependencies), the size of the library (from the current index, else estimated from the
  PDF sizes) and whether a Qdrant server is configured, and explains its choice.
- **Manifest** — every index has a `manifest.json` (engine, embedding model, dimensions, chunks, documents, location,
  build time). Reusing an index built with another embedding model is refused, because the vectors would not be
  comparable. Embedding failures are reported after three attempts instead of leaving silent gaps in the index.
- **Qdrant rebuilds** write a new collection (`<base>_<id>`, recorded in `collection.txt`) and then drop the previous
  one; embedded Qdrant on Windows can reload a deleted collection's files when the same name is recreated.
- **Adding an engine** — one module implementing `BaseVectorIndex` (`availability`, `exists`, `reset`, `add`, `query`,
  `count`, optional `optimize`) and one branch in `open_index()`; `tests/test_vectorstores.py` runs the same contract
  tests on every engine.

## Authentication & User Management

### Account Lifecycle

```text
Sign Up → Pending → Admin Activates → Active → (Admin can Deactivate/Delete)
```

| State           | Description                                           |
| --------------- | ----------------------------------------------------- |
| **Pending**     | Account created but not yet activated. Cannot log in. |
| **Active**      | Admin-approved. Full access to the application.       |
| **Deactivated** | Admin-disabled. Cannot log in until reactivated.      |
| **Deleted**     | Permanently removed from the database.                |

### Implementation

- Passwords are hashed with **bcrypt** before storage.
- User records are stored in `users.duckdb` (shared data folder) by the local IAM provider (`manage/iam/providers/local_duckdb.py`), opened through `db_manager.py`.
- The `is_active` field controls login eligibility.
- The **Admin > Users** page goes through `get_iam_manager()`, so it manages the accounts of the configured provider
  (local or Better Auth), never the local database in its place.
- Administrators are created from `CMP_ADMIN_USERNAME` / `CMP_ADMIN_PASSWORD(_FILE)` on first start or with `python manage_users.py create-admin <name>`; the role (not the username) grants access to the **Admin > Users** page.
- **First-run setup token** — Without a bootstrap password, the first start (or `python manage_users.py bootstrap`, which the Docker image runs before the server) writes a one-time setup token to the server log, and the login page creates the first administrator with it. `python manage_users.py setup-token` issues a new token (the previous one stops working). Only a SHA-256 hash is stored (`setup_token.json` in the data root, `manage/iam/setup_token.py`); the token is cleared once an administrator exists.

### Data Isolation

Per-user isolation is enforced by `core/workspace.py`:

- **Workspace directories** — `UserScope.for_user(username)` maps every storage kind (`datasets`, `stats`, `traces` with `artifacts/` and `snapshots/`, `uploads`, `taxonomy`, `knowledge_graphs`, `documents`, `vectors`, `rag_eval_logs`) to `<DATA_ROOT>/users/<folder>/<kind>/`. The folder is the username when it is a folder name already (ASCII letters and digits, with `_`, `.` or `-` inside, at most 64 characters, not a Windows device name), and otherwise `_` and a readable code of it that no other name has (`core.workspace.safe_dirname`: `j.smith@chl.lu` gets `_j.smith_at_chl.lu`); a code longer than 64 characters gets `_` and 32 hexadecimal characters of the name's SHA-256. The folder an earlier version gave a local account is moved at its sign-in when it is clearly its own (`move_earlier_workspace`); `dev_scripts/move_workspaces.py` plans and moves them for another identity provider. `DBManager`, `TransformationManager` and `RAGManager` take the scope they are given (`scope=`) and never ask the session: `AppState` and `ui/services.ai_engine()` pass the signed-in user's (`ui/session_scope.get_current_scope()`). Without one (command line, tests) they use the shared directories, where a `username=` argument selects the `<user>_` files.
- **Path containment** — `UserScope.resolve_within()` rejects any manually entered or trace-recorded path that resolves outside the workspace; uploaded file names go through `safe_filename()`.
- **Migration** — `python dev_scripts/migrate_user_layout.py --dry-run` previews, and without the flag moves, the legacy flat layout (`data/datasets/<user>_*.parquet`, `data/traces/<user>_session_*.json`, `data/taxonomy/<user>/`, `data/knowledge_graphs/<user>_*.json`, `DOCUMENTS/`) into the per-user workspaces. Files without a username prefix are assigned to the administrator.
- **Login lockout** — `CMP_MAX_FAILED_LOGINS` consecutive failures lock an account for `CMP_LOCKOUT_MINUTES` (defaults 5 and 15).

---

## Pydantic Response Schemas

`manage/rag_schemas.py` defines Pydantic models for structured LLM responses:

| Schema                                             | Purpose                                                                   |
| -------------------------------------------------- | ------------------------------------------------------------------------- |
| `TheoreticalConcept`, `TheoreticalConceptList`     | A medical concept or score with its standard formula and required inputs. |
| `ComputedVariableSuggestion`, `SuggestionResponse` | AI-suggested computed variables mapped to dataset columns.                |
| `ProxyVariable`                                    | A proxy for a missing variable.                                           |
| `AlternativeFormula`                               | An alternative formula that avoids a missing variable.                    |
| `ExpandedTerm`                                     | The full standard name of an abbreviated term.                            |
| `FormulaCorrection`                                | A corrected formula and the changes made.                                 |
| `MarkdownFormulas`                                 | LaTeX versions of formulas.                                               |

Only the proxy-variable and alternative-formula suggestions are validated against their schema
(`validate_and_parse()` in `manage/rag_computed_vars.py`). The other responses (taxonomy, documents, suggestions)
are parsed as plain JSON with `parse_json_safe()` or `json.loads()` (for example in `manage/rag_taxonomy.py`),
without schema validation.

### Why Schemas Matter

- **Type safety** — Catch malformed LLM output before it reaches the UI.
- **Repair** — If parsing fails, `utils/llm_utils.py` repairs common JSON errors. Re-prompting the LLM with the error
  is implemented in `StructuredOutputHelper.invoke_with_schema()`, which no feature calls yet.
- **Consistency** — Downstream code can rely on well-typed objects instead of raw dictionaries.

---

## LLM Utilities (`utils/llm_utils.py`)

Robust JSON parsing pipeline for LLM responses:

1. **Markdown Extraction** — Strips ```json fences from the response.
2. **Bracket Balancing** — Fixes unmatched `[`, `{`, `]`, `}`.
3. **Trailing Comma Removal** — Cleans common JSON syntax errors.
4. **Pydantic Validation** — Validates the parsed object against the expected schema (`validate_and_parse()`, where a schema is used).
5. **LLM Self-Repair** — `StructuredOutputHelper` can send the malformed output back to the LLM with an error message and ask for a corrected version; no feature uses it yet.

---

## Data Analysis Utilities

### `ingest/analyzer.py` — DataAnalyzer

Classifies DataFrame columns into semantic types:

| Type                           | Description                                                         |
| ------------------------------ | ------------------------------------------------------------------- |
| `numeric_cols`                 | Continuous numerical columns (float/int, high cardinality).         |
| `categorical_cols`             | String/object columns.                                              |
| `binary_cols`                  | Columns with exactly 2 unique values.                               |
| `date_cols`                    | Datetime columns.                                                   |
| `low_cardinality_numeric_cols` | Numerical columns with few unique values (may be coded categories). |

Pages do not build one themselves: `AppState.analyzer()` (the analysis rows) or `AppState.analyzer("base")` (every
row) returns a new analyzer each call, built from a column profile that is computed once per dataset and population
(`DataSession.memo`). The Validate dashboard's quality audit is kept the same way (`AppState.quality_audit`).

### `utils/statistics_utils.py`

- Normality tests (Shapiro-Wilk, D'Agostino).
- Test selection guidelines based on sample size and distribution.

### `utils/visualization_utils.py`

- Plotly chart generators with statistical overlays.
- Palette generation (`generate_palette()`, cached with `@functools.cache`).

### `utils/clustering_utils.py`

- `prepare_data_for_clustering()` — Standardises data and handles missing values.
- `fit_kmeans()`, `fit_dbscan()`, `fit_gaussian_mixture()` — Algorithm wrappers returning labels and model objects.

---

## Trace & Reproduction System

### TransformationManager

Manages the recording of analysis steps:

- `initialize_session(dataset_name, username)` — Starts a new trace session.
- `add_step(function_name, params, description, output_dataset_path=None, ..., *, data=None, population=None, replayable=None)` — Records a transformation step. Pages do not call it directly: changes of the data go through `AppState` and `DataSession`, which record the step.
- `get_available_datasets_from_traces()` — Lists datasets referenced in past traces.

### TraceDocumenter

Generates `.docx` reports from trace JSON files using `python-docx`:

- Session metadata header.
- Step-by-step table with function, parameters, and description.
- Downloadable via Streamlit's file download button.

### Replay (`manage/reproduction_manager.py`)

`run_trace()` is the single replay engine: it replays a trace up to a step and checks each step against its
recorded fingerprint. It reads the files a trace names through the resolver it is given (`manage/paths.PathResolver`):
the user's workspace in the app, only the named files for a command line; given none, it reads no file.

### Kinds of trace step (`core/steps.py`)

Every kind of step is registered once, in the `STEPS` catalogue (built on `core/registry.py`). Each kind declares:

| Field          | Meaning                                                                                                                       |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `id`           | the name stored in traces: never renamed                                                                                      |
| `title`        | the heading of the step in the trace report                                                                                   |
| `category`     | the part of the workflow whose page records it (`load`, `validate`, `enrich`, `population`, `analyse`, `report`, `other`)     |
| `replay`       | `recompute` (redone from its parameters), `snapshot` (the recorded table is used) or `document` (nothing to redo)             |
| `changes_data` | whether it changes the values or the rows (renaming or retyping columns does not); a step with fingerprints is judged by them |
| `flow`         | the noun of its box in the participant flow ("After the merge")                                                               |
| `fills`        | `fitted` or `formula` for a step that fills missing values with one value each                                                |

The replay, the trace report, the workflow progress, the participant flow, the study dossier and the imputation
bookkeeping read the catalogue.

To add a kind of step:

1. Register a `StepKind` in `core/steps.py`.
2. If the replay recomputes it, add its handler to `_HANDLED` in `manage/reproduction_manager.py`.
3. Never rename an id; register a kind that nothing records any more with `legacy=True`, so old traces still read.

`TransformationManager.add_step` logs a step recorded under an unregistered name. With `CMP_STRICT_STATE=1` (the test
suite) it raises `UnknownIdError` instead. `tests/core_layer/test_steps.py` checks that every step the application
writes is registered, that the report titles every kind, and that the replay handles every recomputed kind.

---

## RAG Quality Monitor

The `rag_monitoring.py` module renders the **Admin > AI quality** page. Scores come from an LLM acting as judge
(`manage/rag_evaluator.py`), on a 0-10 scale:

| Tab                    | Description                                                                                                                                                                                                                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Quality Overview**   | Average context relevance, faithfulness and answer relevance, JSON success rate, a 7-day trend, scores per function, and a warning when the hallucination rate exceeds 10%.                                                                                                                            |
| **Evaluation Details** | The recent evaluations with their scores, the judge's reasoning and detected hallucinations.                                                                                                                                                                                                           |
| **Embedding Health**   | The retrieval test suite (queries and expected concepts) and a form to add test cases. It shows no embedding statistics; what the index holds (chunks, documents, embedding model and dimensions) is shown in Settings > AI engine > _Vector database_.                                                |
| **Run Evaluation**     | Evaluate one query against a chosen function, or run the test suite (a test passes with an overall score of at least 5 and at least half of the expected concepts in the answer); export the results as JSON. Without an initialised AI engine, only a demo evaluation with fixed scores is available. |

This is primarily a developer and admin tool for diagnosing RAG performance issues.

---

## Configuration

### Study configuration (JSON)

The study configuration lives in the session (`st.session_state.config`). It starts from
`utils.config_loader.create_empty_config()`. A file uploaded on **Data > Validate** (sidebar, _Load Existing Config_)
adds the families and rules the session lacks, each whole: a rule the session holds keeps its definition, and the page
names those whose definition in the file differs (`core.population.merge_families`).

The criteria are `config["mask_families"]`. Each family maps rule names to definitions, and its `operator` says how it
combines them. Operators are read without regard to case, and a Validate tab gives one only to a family it creates.

| Operator           | The family is                                          | A row is excluded when        |
| ------------------ | ------------------------------------------------------ | ----------------------------- |
| `"any"` (or none)  | an exclusion family (`clinical_anomalies`, `outliers`) | a ticked rule fires           |
| `"all"`            | an inclusion family (`study_inclusion`)                | it fails a ticked criterion   |
| `{"threshold": k}` | one exclusion rule, "at least k of these rules"        | at least k of its rules fire  |

Do not add keys at family level other than `operator`. `description` in a rule is shown as the help text of its
criterion in the sidebar, and in its box on Validate.

A family's id is stored (configuration files, criterion ids such as `clinical_anomalies::Implausible age`, traces,
the population fingerprint) and never changes; the pages, the participant flow and the dossier name it in words,
`core.population.family_words`: `clinical_anomalies` is shown as *implausible values*, the family of the
plausibility rules, and any other family as its id with spaces.

#### Criterion kinds

Each kind of rule is one `CriterionKind` in the catalogue `core.criteria.CRITERION_KINDS` (a `core.registry.Catalogue`).
The five kinds of M5 are read from their keys, in this order:

| Kind         | Label            | Definition                                                                     | Fires when                                   |
| ------------ | ---------------- | ------------------------------------------------------------------------------ | -------------------------------------------- |
| `threshold`  | At least k rules | `{"threshold": k, "members": {name: definition}}`                              | at least k of its rules fire                 |
| `row_ids`    | Listed rows      | `{"row_ids": [[first, last]], "method", "params", "columns", "description"}`   | the row is listed (outliers frozen as rows)  |
| `bounds`     | Frozen bounds    | `{"bounds": {"column": [lower, upper]}}`                                       | a value of any column is outside its range   |
| `numeric`    | Numeric range    | `{"numeric": true, "lower_bound": a, "upper_bound": b, "strategy": "include"}` | inside the range, or outside with _exclude_  |
| `expression` | Expression       | `{"numeric": false, "expression": "df['age'] > 110"}`                          | the expression is true                       |

A numeric range reads the column named like the rule, or the one its `"column"` names.

An expression uses `df['column']` (bare names are rejected), `& | ~` with parenthesised comparisons, and the
allow-listed `pd.*` / `np.*` functions and Series methods of `core.expressions.safe_eval` (`isin`, `between`,
`.str.*`, `.dt.*`, `pd.to_datetime(..., dayfirst=True)`, ...). A missing value meets no condition of the other kinds.
In an expression it meets `.isna()`, and may also meet a condition written with `!=` or `~`: the flow and the dossier
say so, and how such expressions should read a missing value is an open decision.

`core.criteria.rule_kind` is the only place that decides a rule's kind:

1. the five kinds above, from their keys and in their order, whatever `"kind"` says: a definition that reads today
   reads the same once a kind it names is registered, so that its recorded population still replays;
2. otherwise a `"kind"` that names a kind registered without a recogniser (from M6: `date_order`, then M9's
   `concept`): that kind;
3. nothing read: refused, with the kind it states when it states one. An older version refuses a newer file, and says
   which kind it does not know.

A rule holding both a numeric range and an expression is refused rather than read one way or the other. `"kind"`
inside a definition is the catalogue's key; a criterion's `"kind"` in a population block of a trace is its role,
exclusion or inclusion.

**The interface.** Pages and services never call a kind's functions. They call the free functions of `core.criteria`,
which read the kind:

- `criterion_mask(df, name, definition)`: the rows where it fires. It raises `CriterionError` and nothing else.
- `criterion_columns(name, definition)`: the columns it reads, those of the rules it combines included; `()` when they
  cannot be read.
- `describe_definition(name, definition)`: what makes it fire, in the words of the participant flow, the dossier and
  Validate's boxes ("18 <= age <= 85", "one of the 3 listed rows (flagged by |z| > 3 on sbp)"). A rule that the
  checks refuse reads "cannot be evaluated" and why. It never raises.
- `criterion_problems(name, definition)`: what makes it unusable on any table. `core.population.configuration_problems`
  refuses an uploaded file holding such a rule, an unknown operator, a threshold that is not a whole number, or a
  family entry that is not a rule. What evaluates (a bound written `"18"`, a threshold of 2.5) still loads.
- `holds_row_list` and `shared_definition`: the copy for sharing (`core.population.configuration_for_sharing`, the
  **Download for Sharing** button) keeps a rule only when it lists no row of the dataset, can be read and passes the
  checks, rebuilt from its kind's keys with values of the types declared. It names what it leaves out.
- The writers `numeric_rule`, `expression_rule`, `listed_rows_rule` and `threshold_rule`: every rule the application
  writes comes from one of them (a gate in `tests/test_quality_gates.py`).

A malformed rule never stops a page: the sidebar counts every criterion before each page. Listed rows are found on
the table's sorted row ids, whatever the length of the ranges they list.

**A stored definition is never rewritten.** A criterion's fingerprint (`core.population.Criterion.fingerprint`)
hashes its definition: it keys the population memos, and every population block of a trace holds it, which the trace
chain hashes. No reader adds, removes or reorders a key. So the five kinds keep their keys, and only the kinds
registered from M6 on carry `"kind"`. A kind's id is stored with every rule of its kind: what a kind selects never
changes under the same id. A changed selection is a new kind, and the old id stays registered, marked legacy.

#### Adding a criterion kind

M6's `date_order` and M9's `concept` are one registration each.

1. Write its functions in a module of its own (`core/temporal_checks.py`), which must not import `core.criteria` (rule
   R2): `mask(df, name, definition)`, `columns(name, definition)`, `describe(name, definition)` and
   `problems(name, definition)`. Raise `ValueError` for what cannot be evaluated: `criterion_mask` names the criterion.
2. Register it in `core/criteria.py`, after the five, without a recogniser:
   `CRITERION_KINDS.register(CriterionKind("date_order", "Date order", mask=..., columns=..., describe=...,
   problems=..., keys={"first": is_text, "then": is_text, "tolerance": is_scalar, "description": is_text}))`. A kind
   that combines rules names the key it keeps them under (`members_key`): the free functions add their checks and
   columns. Say how it treats a missing value (`missing`).
3. Add a writer that states the kind (`{"kind": "date_order", ...}`), used by every page that writes the rule.
4. Give it examples, the rows each selects (`ROWS`), a read-back of its words and malformed examples in
   `tests/core_layer/test_criterion_kinds.py`, which fails until they are there. Its definitions hold none of the keys
   of the five. A test that registers a kind of its own does so in `CRITERION_KINDS.copy()`, patched over the
   catalogue for that test.

#### The population

Ticked criteria define the analysis population (`core/population.py`). A ticked criterion that cannot be evaluated
(unknown column, syntax error) stops the analyses: the population is not `ok`, and the pages show the error until the
criterion is unticked or corrected. The **Run the Plausibility Check** and **Run Inclusion Criteria Check** buttons of the
Validate tabs and the quality dashboard evaluate the rules with the same functions (`core.population.family_masks`),
so they count what the population counts. A rule that cannot be evaluated is shown with its reason, and the family
result is left out until it is corrected.

`assets/tutorial/tutorial_config.json` is a complete, tested example (8 plausibility rules, 4 inclusion criteria).
`ui.state.tutorial_config()` merges it into the default configuration (`folder_names` stay machine-specific).

### Settings, preferences and API keys

**One settings object.** `utils/settings.py` holds every setting the application reads, one field per environment
variable (the names of `.env.example`, which a test keeps equal to the fields), in sections: `paths`, `logging`,
`iam`, `ai`, `vector_store`, `graph`, `statistics` and `development`. Keys and passwords are `SecretStr`. Read them
with `get_settings()`, which validates again only when a variable has changed; it is the only module that reads the
environment (`tests/test_layering.py` checks every module, with a short list of exceptions that can only shrink).
`core` and `biostats` never read settings: they receive values as parameters.

**When `.env` is read.** `main.py` and `manage_users.py` call `load_settings()` first: it loads the `.env` next to
them (once per process; a variable already set in the environment wins, which is how Docker passes them), then
validates every setting. A wrong one stops start-up with the variable's name and what is expected, never its value.
`main.py` then checks the connector settings (`manage.connectors.check_connector_settings`, below). The tests never
read `.env`: `tests/conftest.py` turns `load_dotenv` into a no-op, and `tests/test_startup_config.py`
checks the loading in a subprocess. `st.secrets` is not read; a `.streamlit/secrets.toml` is logged at start-up.

**API keys.** Cloud providers read the server's key from the settings: `manage.rag.providers.server_key(provider)`
reads `get_settings().ai.key(variable)` for the variable the provider declares (`GOOGLE_API_KEY`, `OPENAI_API_KEY`,
`MISTRAL_API_KEY`, `ANTHROPIC_API_KEY`). A key typed in Settings > AI engine is kept for the session only.

**Preferences.** The user's choices in Settings (the AI provider and models, the vector database, the graph engine,
the day-first default) are kept for the next session in `<workspace>/preferences.json` (`manage/preferences.py`,
written whole and atomically). `ui/preferences.py` seeds the session after sign-in, once per user, only for the keys
the session has not set, and the controls save them from their `on_change` callbacks. A preference never holds a
key, a password, a consent or an address (an Ollama or Qdrant address decides where the data go: it is entered again
in each session); a choice the server no longer offers is not used, and the page names it. The file is read only if
it is small (64 KiB) and its names short; the lock serialises the writes of one server process (two servers on one
`DATA_ROOT`: the last write wins).

**Which connector is used.** The AI provider, the vector database and the graph engine are chosen with one control,
`ui.components.connector_selector`, on the kit (`core/registry.py` `ConnectorRegistry.select`): the session's choice
(for the AI provider, first the engine that runs), then the saved preference, then the server's setting
(`CMP_AI_PROVIDER`, `CMP_VECTOR_STORE`, `CMP_GRAPH_RENDERER`), then `auto` (Ollama at `OLLAMA_BASE_URL`; the
recommendation for this machine; Plotly, then Graphviz). It offers only what can run here, and a name that cannot run
here falls through. On Settings it says why the connector used was chosen, which saved choice or setting was set
aside, and what cannot run here with the group to install (`explain_choice`). A cloud provider preselected so
receives none of the user's data until the session confirms it, and an engine started without a provider uses the
server's (`manage.rag.providers.default_provider`).

**An unknown connector name stops start-up.** `CMP_IAM_PROVIDER` accepts `local`, `better_auth` (or `betterauth`,
`better-auth`, `remote`) and `disabled` (or `none`, `off`, `bypass`); `CMP_AI_PROVIDER`, `CMP_VECTOR_STORE` and
`CMP_GRAPH_RENDERER` accept `auto` and their registered names; all in any case, an empty value meaning the default.
`main.py` checks the four right after loading the settings (`manage.connectors.check_connector_settings`, which asks
each registry for its names only): any other value stops start-up with the variable and the valid names, never the
value, instead of an identity provider or an engine other than the one meant. `manage_users.py` does not import that
module (its lazy names lead to the AI packages: rule R5); there the identity manager refuses an unknown provider
(`IAMConfigurationError`, printed as one line).
The identity providers are a connector family chosen by this setting only (`manage.iam.providers.IAM_PROVIDERS`, no
automatic choice, no user's choice): each declares its `name` and `display_name` and builds itself from the
configuration (`from_config`), and is loaded when used. A custom provider is installed with `IAMManager.set_provider`.

---

## Contributing

1. Follow the existing Mixin pattern when adding new RAG capabilities.
2. Validate all LLM responses with Pydantic schemas.
3. Change the data only through `AppState` (`ui/state.py`), which writes through `DataSession` (`core/session.py`):
   `commit_change()` for a change of the dataset, `stage()` for the Enrich draft (traced when applied), `note()` for a
   trace step that does not change the data. Each change becomes one trace step; with `CMP_STRICT_STATE=1` (the test
   suite), an in-place change of the session data raises `UntracedMutationError`.
4. A tab reads its own inputs: it never uses a value that another tab of the page computed in the same run. A new
   session key that one tab or page writes and another reads is declared in `ui/keys.py`, with its owner, its
   lifetime and its readers; a test checks the declarations against the code and against what `AppState` clears. On
   a page whose tabs run only when open, every control has a key and `persist_state="page"`, except a confirmation
   and an input whose `value=` comes from a session key that already keeps it (a kept value would come back stale).
5. Put new code in the layer it belongs to (*Architecture*, above), never in a page: a computation in `biostats`
   or `core`, a service in `manage`, a component in `ui`. `utils` holds platform code (settings, logging, paths,
   optional extras); its older modules are being moved out, so do not add to them.
6. Test with the scripts in `tests/` (e.g. `test_rag_full_integration.py`, `test_trace_documenter.py`).
