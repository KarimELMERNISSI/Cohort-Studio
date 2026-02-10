# Changelog

All notable changes to the Cohort Monitoring Package are documented here.

---

## [2.2.0] - 2026-02-10

### 📚 Documentation Overhaul

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

### 🚀 Architecture Improvements

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

### 🛠️ Utility Extraction

#### Home Page Modularization

- **NEW** `utils/data_analyzer.py` - Consolidated DataAnalyzer class
- **NEW** `utils/export_utils.py` - Excel export functions
- **NEW** `utils/statistics_utils.py` - Normality tests and guidelines

### 📈 Performance

- Added `@st.cache_data` for `get_statistics_dataframe()`
- Added `@functools.cache` for `generate_palette()`

### 📚 Documentation

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
