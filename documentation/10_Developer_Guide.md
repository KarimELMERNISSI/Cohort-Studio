# Developer Guide

This guide covers the architecture, design patterns, and key subsystems of the Cohort Monitoring Package for developers who want to understand, extend, or maintain the application.

---

## Architecture Overview

```text
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Streamlit   │───│   App Pages      │───│  Backend         │
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
- **AI providers are pluggable connectors** in `manage/rag/` (local Ollama by default; Google Gemini, OpenAI and Mistral as cloud providers that require an explicit data-sharing consent).
- Each mixin accesses shared resources (`self.vectorstore`, `self.llm`) from the base `RAGManager`.

---

### Vector databases

| Engine | Mode | Where vectors live | Suits |
| --- | --- | --- | --- |
| LanceDB (default) | embedded | `<workspace>/vectors/lancedb/`, columnar Lance files on disk | one machine, any library size; low memory; exact search below 50 000 chunks, an IVF index above |
| Qdrant | embedded | `<workspace>/vectors/qdrant/storage/`, every vector in memory | small and medium libraries (up to ~20 000 chunks); one process per folder (the client is shared per folder) |
| Qdrant | server | a Qdrant service (`CMP_QDRANT_URL`, optional `CMP_QDRANT_API_KEY`), one collection per user | several users, very large libraries, concurrent sessions; HNSW search and payload indexes |

- **Choice** — Settings > AI engine > *Vector database*. The default comes from `CMP_VECTOR_STORE` (`auto`, `lancedb`
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

| State | Description |
| --- | --- |
| **Pending** | Account created but not yet activated. Cannot log in. |
| **Active** | Admin-approved. Full access to the application. |
| **Deactivated** | Admin-disabled. Cannot log in until reactivated. |
| **Deleted** | Permanently removed from the database. |

### Implementation
- Passwords are hashed with **bcrypt** before storage.
- User records are stored in the DuckDB database (`db_manager.py`).
- The `is_active` field controls login eligibility.
- Administrators are created from `CMP_ADMIN_USERNAME` / `CMP_ADMIN_PASSWORD(_FILE)` on first start or with `python manage_users.py create-admin <name>`; the role (not the username) grants access to the **Users Management** page.

### Data Isolation

Per-user isolation is enforced by `core/workspace.py`:

- **Workspace directories** — `UserScope.for_user(username)` maps every storage kind (`datasets`, `stats`, `traces` with `artifacts/` and `snapshots/`, `uploads`, `taxonomy`, `knowledge_graphs`, `documents`, `vectors`, `rag_eval_logs`) to `<DATA_ROOT>/users/<username>/<kind>/`. `DBManager`, `TransformationManager` and `RAGManager` resolve the signed-in user's scope from the session (`utils/session_scope.get_current_scope()`); outside a Streamlit session (CLI, tests) they fall back to the legacy shared directories and the username-prefix convention.
- **Path containment** — `UserScope.resolve_within()` rejects any manually entered or trace-recorded path that resolves outside the workspace; uploaded file names go through `safe_filename()`.
- **Migration** — `python dev_scripts/migrate_user_layout.py --dry-run` previews, and without the flag moves, the legacy flat layout (`data/datasets/<user>_*.parquet`, `data/traces/<user>_session_*.json`, `data/taxonomy/<user>/`, `data/knowledge_graphs/<user>_*.json`, `DOCUMENTS/`) into the per-user workspaces; a legacy `data/chroma_db/` index is not migrated (the index is rebuilt from the PDFs). Files without a username prefix are assigned to the administrator.
- **Login lockout** — `CMP_MAX_FAILED_LOGINS` consecutive failures lock an account for `CMP_LOCKOUT_MINUTES` (defaults 5 and 15).

---

## Pydantic Response Schemas

All structured LLM responses are validated against Pydantic models defined in `manage/rag_schemas.py`:

| Schema | Purpose |
| --- | --- |
| `TaxonomyNode` | Node in the taxonomy graph (variable, concept, formula, category). |
| `TaxonomyEdge` | Relationship between two taxonomy nodes. |
| `ComputedVariable` | AI-suggested variable with name, formula, and reasoning. |
| `ImputationFormula` | AI-suggested imputation formula for a target variable. |
| `DocumentSummary` | Structured summary extracted from a research paper. |

### Why Schemas Matter

- **Type safety** — Catch malformed LLM output before it reaches the UI.
- **Retry logic** — If parsing fails, `utils/llm_utils.py` attempts JSON repair and re-prompts the LLM.
- **Consistency** — Downstream code can rely on well-typed objects instead of raw dictionaries.

---

## LLM Utilities (`utils/llm_utils.py`)

Robust JSON parsing pipeline for LLM responses:

1. **Markdown Extraction** — Strips ```json fences from the response.
2. **Bracket Balancing** — Fixes unmatched `[`, `{`, `]`, `}`.
3. **Trailing Comma Removal** — Cleans common JSON syntax errors.
4. **Pydantic Validation** — Validates the parsed object against the expected schema.
5. **LLM Self-Repair** — If all else fails, sends the malformed output back to the LLM with an error message and asks for a corrected version.

---

## Data Analysis Utilities

### `utils/data_analyzer.py` — DataAnalyzer

Classifies DataFrame columns into semantic types:

| Type | Description |
| --- | --- |
| `numeric_cols` | Continuous numerical columns (float/int, high cardinality). |
| `categorical_cols` | String/object columns. |
| `binary_cols` | Columns with exactly 2 unique values. |
| `date_cols` | Datetime columns. |
| `low_cardinality_numeric_cols` | Numerical columns with few unique values (may be coded categories). |

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
- `add_step(function, params, description, output_dataset_path)` — Records a transformation step.
- `get_available_datasets_from_traces()` — Lists datasets referenced in past traces.

### TraceDocumenter

Generates `.docx` reports from trace JSON files using `python-docx`:
- Session metadata header.
- Step-by-step table with function, parameters, and description.
- Downloadable via Streamlit's file download button.

### ReproductionManager

Handles trace replay with path resolution for cross-platform compatibility.

---

## RAG Quality Monitor

The `rag_monitoring.py` page provides a dashboard for monitoring RAG system health:

| Metric | Description |
| --- | --- |
| **Embedding Statistics** | Document count, chunk count, and embedding dimensions. |
| **Retrieval Quality** | Relevance scores for test queries. |
| **Evaluation Results** | Structured evaluation of RAG responses against expected outputs. |

This is primarily a developer and admin tool for diagnosing RAG performance issues.

---

## Configuration

### `config/config.json`

Contains data quality rules, validation thresholds, and anomaly definitions. Key sections:

| Section | Purpose |
| --- | --- |
| `anomalies` | Clinical anomaly rules per variable (min/max bounds, impossible values). |
| `inclusion_criteria` | Mask family definitions for cohort filtering. |
| `quality_weights` | Weights for completeness, validity, and consistency in the quality score. |

### `.streamlit/secrets.toml`

Stores the `GOOGLE_API_KEY` for local development (not committed to version control).

---

## Contributing

1. Follow the existing Mixin pattern when adding new RAG capabilities.
2. Validate all LLM responses with Pydantic schemas.
3. Use `TransformationManager.add_step()` to log any data-modifying operation.
4. Add utility functions to the appropriate `utils/` module, not inline in app pages.
5. Test with the scripts in `tests/` (e.g. `test_rag_full_integration.py`, `test_trace_documenter.py`).
