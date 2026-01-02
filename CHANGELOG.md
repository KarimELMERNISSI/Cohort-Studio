# Changelog

All notable changes to the Cohort Monitoring Package are documented here.

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
- **NEW** Developer Guide (`documentation/12_Developer_Guide.md`)
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
