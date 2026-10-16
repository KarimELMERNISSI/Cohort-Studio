# Configuration Documentation

## Overview

CohortStudio has two kinds of configuration:

1. **The study configuration** (JSON): the clinical anomaly rules, inclusion criteria and outlier criteria of a study. It lives in the session (`st.session_state.config`).
2. **Environment settings**: storage, accounts, AI providers, vector database, graph engine and logging, read from environment variables (template: `.env.example`).

The format of the study configuration and of its rules is described in `documentation/10_Developer_Guide.md` (section *Configuration*).

## Study configuration

- **Start**: every session starts from `utils.config_loader.create_empty_config()` (empty `mask_families`, default outlier thresholds, folder names under the data root).
- **Load**: on **Data > Validate**, choose *Load Existing Config* in the sidebar and upload a JSON file; it is merged into the session configuration without overwriting existing keys.
- **Save**: **Download Configuration File**, in the results of the anomaly and inclusion checks on **Data > Validate**, downloads the session configuration as `config.json`.
- **Example**: `assets/tutorial/tutorial_config.json` (8 anomaly rules, 4 inclusion criteria), loaded with the tutorial cohort from the Start page.

The `config/` folder may hold local copies of study configurations. Its JSON files are git-ignored, and the application does not read them.

## Environment settings

`.env.example` lists the variables with their defaults. `documentation/10_Developer_Guide.md` describes the AI keys, the vector database and the accounts.
