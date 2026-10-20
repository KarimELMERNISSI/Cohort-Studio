# Developer Guide

## Overview

This is the central entry point for developers working on **CohortStudio**.
This documentation aims to provide a comprehensive view of the codebase structure, execution flow, and module details.
The architecture and the main subsystems are described in `documentation/10_Developer_Guide.md`.

## Quick Links

- **[App_Pages.md](App_Pages.md)**: Feature pages in `app_pages/`.
- **[Management_Modules.md](Management_Modules.md)**: Backend logic in `manage/`.
- **[Utilities.md](Utilities.md)**: Helper functions in `utils/`.
- **[Prompts.md](Prompts.md)**: LLM prompts in `prompts/`.
- **[Enrichment_Modules.md](Enrichment_Modules.md)**: Enrichment logic in `enrich/`.
- **[Monitoring_Modules.md](Monitoring_Modules.md)**: Monitoring logic in `monitor/`.
- **[Exploration_Modules.md](Exploration_Modules.md)**: Exploration logic in `explore/`.
- **[Configuration.md](Configuration.md)**: The study configuration and the environment settings.
- **[Graph_Rendering.md](Graph_Rendering.md)**: The renderer-agnostic graph layer in `graph/`.
- **[License_Dependencies.md](License_Dependencies.md)**: Audit of third-party libraries and license recommendations.
- **[Process_Flow.md](Process_Flow.md)**: Diagrams tracing key execution flows.

## Key Concepts

- **Session State & Data**: `AppState` (`ui/state.py`) is the Streamlit facade over `DataSession` (`core/session.py`), which holds the committed table, the Enrich draft and the analysis population. Pages read the views `data`, `working_df` (the analysis population) and `enriched_df` (the Enrich draft) and change the data only through `AppState`. The managers (`db_manager`, `transformation_manager`, `rag_manager`) are kept in `st.session_state` across reruns.
- **Traceability System**: every modification to the data (cleaning, enrichment, imputation) is recorded as a "Step" by the `TransformationManager`, through `AppState`; the analysis population is recorded when a result or a saved version uses it. The trace is a JSON file in the user's workspace: it is replayed on Reproduce > Versions & traces and documented as a DOCX report.
- **RAG Architecture**: The Knowledge pages (Variable map, Documents) index the PDFs of the user's document library in a vector database (LanceDB by default, or Qdrant; `manage/vectorstores/`). The AI provider is local Ollama by default; Google Gemini, OpenAI, Mistral AI and Anthropic Claude are cloud options that need an API key and the user's consent. Both are chosen in Settings > AI engine.

## Project Structure

The project is a Streamlit application with one entry point and a page registry:

```
CohortStudio/
├── main.py                  # Entry point: sign-in, navigation and the sidebar shell
├── app_pages/               # Page modules
│   └── views/               # The page files run by st.navigation
├── assets/                  # Static assets (brand, sample and tutorial cohorts)
├── biostats/                # UI-free statistics engine
├── config/                  # Local study configurations (git-ignored; not read by the app)
├── core/                    # Session data, population, criteria, fingerprints, workspaces (no Streamlit)
├── data/                    # Local data storage (per-user workspaces under data/users/)
├── dev_scripts/             # Developer utility scripts
├── documentation/           # Project documentation
│   └── developer/           # Developer notes and generated API docs
├── enrich/                  # Data enrichment logic
├── environments/            # Conda environment (Python 3.14, then requirements.txt)
├── explore/                 # Data exploration modules
├── graph/                   # Renderer-agnostic graph layer
├── ingest/                  # File loaders (encoding, delimiter and decimal detection)
├── manage/                  # Backend logic: DB, IAM, RAG, vector stores, traces, replay, dossier
├── monitor/                 # Data monitoring & outlier detection
├── prompts/                 # LLM prompts for RAG
├── tests/                   # Test suite
├── ui/                      # Shell: navigation registry, AppState, sign-in, import flow, components
└── utils/                   # Shared utilities (including miss_forest)
    └── miss_forest/         # Split implementation of MissForest
```

## Getting Started

1. **Entry Point**: The application starts at `main.py`.
2. **Navigation**: `st.navigation`, built from the page registry `PAGES` in `ui/nav.py`; the page files are in `app_pages/views/`.
3. **Data Management**: `AppState` installs the dataset (`set_dataset()`) and is the only writer of the data; every change is traced by the `TransformationManager`.

## Development Workflow

- **Adding a Page**: Create the page file in `app_pages/views/` (it runs as a script) and add a `PageDef` to `PAGES` in `ui/nav.py` (key, file, title, URL path, section, icon; `admin=True` for administrator pages).
- **Updating Logic**:
  - Core data processing should go in `core/`, `utils/` or `manage/`.
  - UI-specific logic remains in `app_pages/` and `ui/`.
  - Changes of the data go through `AppState` (`commit_change()`, `stage()`).
- **Documentation**:
  - Add docstrings to new functions.
  - Run `dev_scripts/doc_gen.py` to update the generated API docs (`App_Pages.md`, `Management_Modules.md`, `Utilities.md`, `Prompts.md`, `Enrichment_Modules.md`, `Monitoring_Modules.md`, `Exploration_Modules.md`); the other files of this folder are written by hand.
