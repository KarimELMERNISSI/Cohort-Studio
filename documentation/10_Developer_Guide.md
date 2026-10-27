# Developer Guide

This guide covers the architecture, design patterns, and key subsystems of CohortStudio for developers who want to understand, extend, or maintain the application.

---

## Architecture Overview

```text
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Streamlit   │────│   App Pages      │────│  Backend         │
│  Frontend    │    │  (app_pages/)    │    │  Managers        │
│              │    │                  │    │  (manage/)       │
└──────────────┘    └──────────────────┘    └──────────────────┘
                           │                       │
                    ┌──────▼──────┐          ┌─────▼──────┐
                    │  Utilities  │          │  Storage   │
                    │  (utils/)   │          │  (data/)   │
                    └─────────────┘          └────────────┘
```

- **App Pages** — Streamlit UI modules, one per page tab.
- **Managers** — Backend logic for RAG, database, and reproduction.
- **Utilities** — Reusable helpers for analysis, statistics, visualisation, and LLM interaction.
- **Storage** — DuckDB databases, vector indexes (LanceDB or Qdrant), Parquet snapshots, and trace JSON files.

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

- **Workspace directories** — `UserScope.for_user(username)` maps every storage kind (`datasets`, `stats`, `traces` with `artifacts/` and `snapshots/`, `uploads`, `taxonomy`, `knowledge_graphs`, `documents`, `vectors`, `rag_eval_logs`) to `<DATA_ROOT>/users/<username>/<kind>/`. `DBManager`, `TransformationManager` and `RAGManager` resolve the signed-in user's scope from the session (`utils/session_scope.get_current_scope()`); outside a Streamlit session (CLI, tests) they fall back to the legacy shared directories and the username-prefix convention.
- **Path containment** — `UserScope.resolve_within()` rejects any manually entered or trace-recorded path that resolves outside the workspace; uploaded file names go through `safe_filename()`.
- **Migration** — `python dev_scripts/migrate_user_layout.py --dry-run` previews, and without the flag moves, the legacy flat layout (`data/datasets/<user>_*.parquet`, `data/traces/<user>_session_*.json`, `data/taxonomy/<user>/`, `data/knowledge_graphs/<user>_*.json`, `DOCUMENTS/`) into the per-user workspaces; a legacy `data/chroma_db/` index is not migrated (the index is rebuilt from the PDFs). Files without a username prefix are assigned to the administrator.
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

### `utils/data_analyzer.py` — DataAnalyzer

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
`utils.config_loader.create_empty_config()` and is uploaded on **Data > Validate** (sidebar, _Load Existing
Config_), which merges it into the session without overwriting existing rules. The Validate page reads
`mask_families` only:

| Family               | Operator       | A rule that matches...      | Result column                        |
| -------------------- | -------------- | --------------------------- | ------------------------------------ |
| `clinical_anomalies` | `any` (forced) | flags the row as an anomaly | `clinical_anomalies_any` (+ `_list`) |
| `study_inclusion`    | `all` (forced) | keeps the row in the study  | `study_inclusion_all` (+ `_list`)    |

Each rule is a key of its family (the rule name) with one of two shapes:

- **Expression rule:** `{"numeric": false, "expression": "<expression>"}`. The expression uses `df['column']`
  (bare names are rejected), `& | ~` with parenthesised comparisons, and the allow-listed `pd.*` / `np.*`
  functions and Series methods of `core.expressions.safe_eval` (`isin`, `between`, `.str.*`, `.dt.*`,
  `pd.to_datetime(..., dayfirst=True)`, ...).
- **Numeric rule:** `{"numeric": true, "lower_bound": x, "upper_bound": y, "strategy": "include" | "exclude"}`.
  The rule name must be the column name. With both bounds, _include_ matches values inside the range and
  _exclude_ values outside it, so use _exclude_ with the plausible range for anomalies and _include_ with the
  eligible range for inclusion. NaN never matches. Avoid leaving `strategy` empty when both bounds are set.

A rule has exactly one shape. `core.criteria.rule_kind` decides it, and it is the only place that does: the
evaluation, the columns read and the description in the participant flow all use it. A rule that holds both a
numeric range and an expression is refused rather than read one way or the other. The outlier rules (`bounds`,
`row_ids`) and threshold families (`{"threshold": k}` as the operator) have their own shapes.

Extra keys in a rule are kept; `description` is shown as the help text of the criterion in the sidebar. Do not add
keys at family level other than `operator`. Operators are read without regard to case.

`core.population.configuration_problems` lists what makes a configuration ambiguous or unusable, and an uploaded
file with problems is not loaded:

- an unknown operator, or a threshold that is not a whole number;
- a family entry that is not a rule;
- a numeric rule without bounds or with an unknown strategy;
- a rule of an unknown shape, or with both shapes.

Ticked criteria define the analysis population (`core/population.py`). A family with the `any` operator is an
exclusion family (`clinical_anomalies`, and `outliers`, which the Handle Outliers tab fills with frozen row ids):
a row is excluded when a ticked rule fires. A family with `all` is an inclusion family (`study_inclusion`): a row
is excluded when it fails a ticked criterion. A family with a threshold operator is one exclusion rule, "at least
k of these rules". A ticked criterion that cannot be evaluated (unknown column, syntax error) stops the analyses:
`core/criteria.py` raises `CriterionError`, the population is not `ok`, and the pages show the error until the
criterion is unticked or corrected. The **Run Anomaly Detection** and **Run Inclusion Criteria Check** buttons of
the Validate tabs (the result columns above) and the quality dashboard evaluate the rules with the same functions
(`core.population.family_masks`), so they count what the population counts. A rule that cannot be evaluated is
shown with its reason, and the family result is left out until it is corrected.

`assets/tutorial/tutorial_config.json` is a complete, tested example (8 anomaly rules, 4 inclusion
criteria). `ui.state.tutorial_config()` merges it into the default configuration (`folder_names` stay
machine-specific).

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
5. Add utility functions to the appropriate `utils/` module, not inline in app pages.
6. Test with the scripts in `tests/` (e.g. `test_rag_full_integration.py`, `test_trace_documenter.py`).
