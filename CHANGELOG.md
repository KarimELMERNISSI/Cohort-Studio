# Changelog

All notable changes to the Cohort Monitoring Package are documented here.

---

## [Unreleased]

### Graph rendering connector (audit phase P1; findings G1-G8)

- **New `graph/` package** — Renderer-agnostic model (`GraphSpec`), a palette/legend single source of truth, pure builders for the correlation network, the clinical knowledge graph and the document graph, and a renderer contract (`translate()` / `render()`) with three backends: Plotly + NetworkX (offline default), yFiles (opt-in extra `graph-yfiles`) and Graphviz (static). See `documentation/developer/Graph_Rendering.md`.
- **yFiles is optional** — Removed from the required dependencies (`requirements.txt`; use `requirements-yfiles.txt` or `uv sync --extra graph-yfiles`), selected with `CMP_GRAPH_RENDERER=yfiles`, refused with `CMP_OFFLINE=1`; third-party notice on the About page and under every yFiles-rendered graph (license clause 1.7).
- **Pages** — Main View, Data Insight and Documents Insight render through `graph.ui.render_graph`; graph selection now reaches the pages (the Data Insight selection panel works, a graph click selects the entity on the Documents page); a renderer failure shows an error card and falls back to the offline engine instead of blanking the remaining tabs; JSON/HTML exports and an in-page "Graph engine" selector were added.
- **Fixes** — Stable node/edge ids and ordering (no uuid per rerun, no `hash()` ids); node heat computed from the numeric correlations rather than rounded labels; one idempotent `make_formula_id()` (no `FORM_FORM_` double prefix); one variable-type vocabulary with a style for `Outcome`; document-graph relations remapped after entity merge (no spurious "Inferred" nodes); one correlation-strength threshold table (`explore/correlation_strength.py`, Schober et al. 2018) shared by the graph, the Excel export and the UI; the dead `plotly_corr_network` was removed and `explore/corr_matrix.py` no longer imports a UI library (`get_yfiles_network_data` remains as a deprecated shim).
- **Tests** — `tests/graph_layer/`: model integrity and serialisation, palette coverage, builder reference graphs, Plotly and yFiles translation (the latter replayed through the library's own mapping engine), renderer-selection policy, headless renderer contract.

### Security and stability (audit phase P0; finding IDs refer to `documentation/13_Audit_Report_2026-10.md`)

- **Audit report** — Added `documentation/13_Audit_Report_2026-10.md` with the verified findings and the remediation programme.
- **E5 Admin takeover** — Signing up with the username `admin` no longer creates an active administrator. Administrators are bootstrapped from `CMP_ADMIN_USERNAME` / `CMP_ADMIN_PASSWORD(_FILE)` on first start or with `python manage_users.py create-admin <name>`; all privilege checks now use the stored role (`manage/iam/authz.py`). Minimum password length is 12 (`CMP_MIN_PASSWORD_LENGTH`), usernames are validated. `reset_password.py` is replaced by `manage_users.py` (passwords are prompted, never passed on the command line).
- **E6 Cloud LLM** — The AI provider now defaults to local Ollama. Cloud providers require an explicit per-session confirmation listing exactly what would leave the server; `RAGManager.initialize_system` refuses cloud providers without it. Server API keys are no longer pre-filled into the sidebar and user-typed keys are no longer written to `os.environ`.
- **E7 Patient data on stdout** — Debug `print` calls that dumped data frames, rows and identifiers (`enrichment`, `monitor/changes.py`, `enrich/external_data.py`, dashboard) were replaced by `logger.debug` of shapes and column names. Logging is configured centrally (`utils/logging_config.py`, `CMP_LOG_LEVEL`); global warning suppression in `main.py` was removed.
- **E12 Tracked secrets** — `.streamlit/secrets.toml`, client study configs (`config/*.json`) and the `DOCUMENTS/` PDF library are no longer tracked (git-ignored; local copies kept). Added `.streamlit/config.toml` with usage statistics disabled.
- **E2 Visualization page** — Fixed the `ImportError` (`DataAnalyzer` now imported from `utils.data_analyzer`) that prevented the page from loading.
- **G1/G2 yFiles 2.0** — Replaced `Layout.HIERARCHIC` with `Layout.HIERARCHICAL`, moved height/directed to `show()`, removed widget attributes ignored by yfiles-graphs-for-streamlit 2.x, and isolated graph rendering so a failure no longer blanks the other tabs. The dead `app_pages/yfiles_test.py` page was removed.
- **E9 Crashes** — Fixed 16 undefined names (`compute_transformations` imports, `logger` in `utils/analysis_utils.py`, `Any` in `utils/export_utils.py`, `time` shadowing in Data Insight) and the empty-state crashes on Data Enrichment and Data Validation (clear notices instead of `KeyError` / `UnboundLocalError`).
- **U4/U8 Copy** — Fixed the comparison export always reporting failure, the duplicate dictionary key in Visualization, the empty delete-user button label, and the French label on the comparison tab.
- **E14 CI** — Added `.github/workflows/ci.yml` (ruff syntax/undefined-name check + offline pytest); `pytest` now excludes `integration` tests by default so a plain run never calls live LLM APIs.

### Owner actions still required

- Revoke and rotate the Google API key committed in `fdd29cb` (it is still the key in the local `.env`), purge it from history and enable GitHub secret scanning. Rotate the admin password that was tracked in `.streamlit/secrets.toml`.

---

## [2.2.0] - 2026-02-10

### Documentation Overhaul

- **README.md** — Added clustering, user management, RAG quality monitor, DBSCAN outlier detection, document summariser, replay modes, full configuration table, and expanded tech stack.
- **02_Data_Validation.md** — Added DBSCAN, outlier visualisation (2-D projection), handling strategy matrix, and impact analysis measures.
- **03_Data_Enrichment.md** — Added Import External Data, AI imputation modes (Free Thinking / Guided), clustering (K-Means, DBSCAN, GMM), scaling methods, and dataset versioning.
- **04_Visualization.md** — Added medical plots (Kaplan-Meier, Bland-Altman, ROC), integrated statistics table, and export formats.
- **05_Epidemiology.md** — Removed duplicate sections; consolidated into clean structure with test recommendation, post-hoc tests, and effect size interpretation.
- **06_Data_Insight.md** — Added taxonomy refinement (repair, enrich, concept nodes), versioning, graph visualisation details, and export options.
- **07_Document_Insight.md** — Added document summariser tab, concept merging, citation tracking, and coverage analysis details.
- **08_Reproduction.md** — Added Fast vs Full replay modes, cross-platform path resolution, and TraceDocumenter reporting.
- **09_Clustering.md** — Aligned with Data Enrichment naming; added cross-references and common pitfalls.
- **10_Developer_Guide.md** — Added user management architecture, trace & reproduction system, clustering utilities, RAG quality monitor, and config documentation.
- **00_Researcher_Handbook.md** — Added Knowledge & Literature phase and Reproducibility phase; aligned all phases with navigation order.
- **Tutorials.md** — Added tutorials for clustering, document analysis, and reproducing analysis sessions (9 total).

---

## [2.1.0] - 2026-01-02

### Architecture Improvements

#### RAG Manager Refactoring

- **Mixin Pattern**: Split monolithic `rag_manager.py` (2098 lines → 451 lines)
  - `rag_taxonomy.py` - Taxonomy generation
  - `rag_documents.py` - Document processing
  - `rag_computed_vars.py` - Variable suggestions

#### LLM Reliability Enhancements

- **NEW** `manage/rag_schemas.py` - Pydantic models for type-safe LLM responses
- **NEW** `utils/llm_utils.py` - Robust JSON parsing with:
- Multi-strategy parsing (markdown extraction, bracket balancing)
- Automatic repair (trailing commas, unmatched brackets)
- Schema validation with Pydantic
- Retry mechanism with LLM self-repair

### Utility Extraction

#### Home Page Modularization

- **NEW** `utils/data_analyzer.py` - Consolidated DataAnalyzer class
- **NEW** `utils/export_utils.py` - Excel export functions
- **NEW** `utils/statistics_utils.py` - Normality tests and guidelines

### Performance
- Added `@st.cache_data` for `get_statistics_dataframe()`
- Added `@functools.cache` for `generate_palette()`

### Documentation
- Updated README with architecture diagrams
- **NEW** Developer Guide (`documentation/10_Developer_Guide.md`)
- **NEW** CHANGELOG.md

---

## [2.0.0] - 2025-12-22

### Features
- RAG Quality Monitoring Dashboard
- Document Knowledge Graph
- Taxonomy Generation
- AI-powered column renaming

---

## [1.0.0] - 2025-11-01

### Initial Release
- Data loading and versioning
- Data quality checks
- Visualization suite
- Epidemiology module
