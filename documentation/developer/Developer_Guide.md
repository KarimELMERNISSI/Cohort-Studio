# Developer Guide

## Overview

This is the central entry point for developers working on the **Cohort Monitoring Package**.
This documentation aims to provide a comprehensive view of the codebase structure, execution flow, and module details.

## Quick Links

- **[App_Pages.md](App_Pages.md)**: Feature pages in `app_pages/`.
- **[Management_Modules.md](Management_Modules.md)**: Backend logic in `manage/`.
- **[Utilities.md](Utilities.md)**: Helper functions in `utils/`.
- **[Prompts.md](Prompts.md)**: LLM prompts in `prompts/`.
- **[Enrichment_Modules.md](Enrichment_Modules.md)**: Enrichment logic in `enrich/`.
- **[Monitoring_Modules.md](Monitoring_Modules.md)**: Monitoring logic in `monitor/`.
- **[Exploration_Modules.md](Exploration_Modules.md)**: Exploration logic in `explore/`.
- **[Configuration.md](Configuration.md)**: Details on application configuration files.
- **[License_Dependencies.md](License_Dependencies.md)**: Audit of third-party libraries and license recommendations.
- **[Process_Flow.md](Process_Flow.md)**: Diagrams tracing key execution flows.

## Key Concepts

- **Session State & Data**: The app relies heavily on `st.session_state` to persist the `data_manager` (DataFrame wrapper), `enrichment_manager`, and `rag_manager` across re-runs.
- **Traceability System**: every modification to the data (filtering, cleaning, enrichment) is recorded as a "Step" in a `TransformationManager`. This trace can be exported as a JSON file for reproduction or as a DOCX report for documentation.
- **RAG Architecture**: The "Documents Insight" feature uses a Vector Database (ChromaDB) to index PDFs. It supports multiple LLM backends (OpenAI, Mistral) which are configured via the sidebar or config files.

## Project Structure

The project follows a standard Streamlit multi-page application structure:

```
app/
├── main.py                  # Application entry point
├── app_pages/               # Individual frontend page modules
├── assets/                  # Static assets (images, css)
├── config/                  # Configuration files (JSON)
├── data/                    # Local data storage
├── dev_scripts/             # Developer utility scripts
├── documentation/           # Project documentation
│   └── developer/           # Generated API docs
├── enrich/                  # Data enrichment logic
├── environments/            # Environment configs (Docker, Conda)
├── explore/                 # Data exploration modules
├── manage/                  # Backend logic: State, DB, RAG, Transformations
├── monitor/                 # Data monitoring & outlier detection
├── prompts/                 # LLM prompts for RAG
├── tests/                   # Integration and import tests
└── utils/                   # Shared utilities (including miss_forest)
    └── miss_forest/         # Split implementation of MissForest
```

## Getting Started

1. **Entry Point**: The application starts at `main.py`.
2. **Navigation**: Handled by `utils.multipage.MultiPageApp`. Pages are registered in `main.py`.
3. **Data Management**: Data is loaded into `st.session_state['data']`. Transformations are tracked by `TransformationManager`.

## Development Workflow

- **Adding a Page**: Create a new module in `app_pages/` with an `app()` function, then register it in `main.py`.
- **Updating Logic**:
  - Core data processing should go in `utils/` or `manage/`.
  - UI-specific logic remains in `app_pages/`.
- **Documentation**:
  - Add docstrings to new functions.
  - Run `dev_scripts/doc_gen.py` to update the markdown documentation.
