# 🛠️ Developer Guide

This guide covers the architecture, design patterns, and key subsystems of the Cohort Monitoring Package for developers who want to understand, extend, or maintain the application.

---

## 🏗️ Architecture Overview

```text
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Streamlit   │───▶│   App Pages      │───▶│  Backend         │
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
- **Storage** — DuckDB databases, ChromaDB vector stores, Parquet snapshots, and trace JSON files.

---

## 🧠 RAG Manager (Mixin Pattern)

The RAG system is structured using the **Mixin composition pattern** to keep each concern in its own module:

```text
RAGManager(TaxonomyMixin, DocumentsMixin, ComputedVarsMixin)
├── rag_taxonomy.py      # Taxonomy generation & refinement
├── rag_documents.py     # PDF processing, summarisation, knowledge graphs
├── rag_computed_vars.py # Variable suggestion & imputation formulas
└── rag_manager.py       # Orchestrator: initialisation, ChromaDB, shared utilities
```

### Key Design Decisions

- **ChromaDB** is used as the vector store for document embeddings.
- **Google Gemini** is the LLM backend (via a custom LangChain adapter in `utils/custom_gemini.py`).
- Each mixin accesses shared resources (`self.vectorstore`, `self.llm`) from the base `RAGManager`.

---

## 🔐 Authentication & User Management

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
- The `admin` user is automatically activated and has access to the **Users Management** page.

### Data Isolation

Per-user isolation is enforced via:

- **Filename prefixes** — Saved datasets, traces, and configs are prefixed with the username.
- **Directory scoping** — ChromaDB collections and snapshot directories are user-specific.
- **Session state** — `st.session_state.username` gates all read/write operations.

---

## 📋 Pydantic Response Schemas

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

## 🔧 LLM Utilities (`utils/llm_utils.py`)

Robust JSON parsing pipeline for LLM responses:

1. **Markdown Extraction** — Strips ```json fences from the response.
2. **Bracket Balancing** — Fixes unmatched `[`, `{`, `]`, `}`.
3. **Trailing Comma Removal** — Cleans common JSON syntax errors.
4. **Pydantic Validation** — Validates the parsed object against the expected schema.
5. **LLM Self-Repair** — If all else fails, sends the malformed output back to the LLM with an error message and asks for a corrected version.

---

## 📊 Data Analysis Utilities

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

## 🔁 Trace & Reproduction System

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

## 📊 RAG Quality Monitor

The `rag_monitoring.py` page provides a dashboard for monitoring RAG system health:

| Metric | Description |
| --- | --- |
| **Embedding Statistics** | Document count, chunk count, and embedding dimensions. |
| **Retrieval Quality** | Relevance scores for test queries. |
| **Evaluation Results** | Structured evaluation of RAG responses against expected outputs. |

This is primarily a developer and admin tool for diagnosing RAG performance issues.

---

## 📁 Configuration

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

## 🤝 Contributing

1. Follow the existing Mixin pattern when adding new RAG capabilities.
2. Validate all LLM responses with Pydantic schemas.
3. Use `TransformationManager.add_step()` to log any data-modifying operation.
4. Add utility functions to the appropriate `utils/` module, not inline in app pages.
5. Test with the scripts in `tests/` (e.g. `test_rag_full_integration.py`, `test_trace_documenter.py`).
