# CohortStudio

[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/KarimELMERNISSI/Cohort-Studio/actions/workflows/ci.yml/badge.svg)](https://github.com/KarimELMERNISSI/Cohort-Studio/actions/workflows/ci.yml)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![DuckDB Version](https://img.shields.io/badge/DuckDB-1.5%2B-fff?logo=duckdb&logoColor=black)](https://duckdb.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-0ea5e9.svg)](https://karimelmernissi.github.io/Cohort-Studio/)

> **Clinical cohort data, from raw file to reproducible evidence.**
> An open-source, local-first application for observational cohort studies: data quality, a toggleable analysis
> population with its participant flow, multiple imputation pooled with Rubin's rules, guided statistics and survival
> models, and a Word study dossier that states what every result was computed on.

---

## Executive Overview

**CohortStudio** is built for clinical researchers, biostatisticians and health data scientists who work with observational cohorts, registry databases and clinical trial datasets.

Cohort analyses are often spread across ad-hoc Python or R scripts, unversioned spreadsheets and tools that do not record how the data were changed. CohortStudio brings the steps together in one local-first application built on **Python 3.14+** and **Streamlit 1.65+**: every change of the data is recorded in a replayable trace, every analysis runs on the same declared population, and every result states the data, the population and the N it was computed on.

### Core Architecture Pillars

1. **Plain files and per-user workspaces**: each saved version is a complete Parquet file (`name_v1`, `name_v2`, …) written with pyarrow, with a lineage file next to it; DuckDB holds the user accounts and the statistics cache. Every account works in its own workspace.
2. **Statistics engine** (`biostats`): Welch-first group comparisons with effect sizes and confidence intervals ($\text{Hedges' } g$, rank-biserial $r$, $\eta^2$, $\omega^2$), ANCOVA (one outcome, several covariates) with a test of parallel slopes, 2x2 measures with the Haldane-Anscombe correction, Little's MCAR test, diagnostic accuracy (ROC, PPV, NPV, likelihood ratios), survival and regression models, and multiple imputation. Normality and variance checks are diagnostics, never gates.
3. **Data quality**: a score across completeness, uniqueness, consistency, uniformity, statistical validity (IQR, z-score, quantiles, Isolation Forest, local outlier factor, DBSCAN) and the clinical anomaly rules you declare.
4. **Optional AI assistant** (the `ai` group): connectors for local Ollama (the default), Google Gemini, OpenAI, Mistral and Anthropic Claude, for questions on your documents, column-name suggestions and variable engineering. Cloud providers need an explicit consent; the suggestions come from the language model and are not checked against a terminology.
5. **Analysis Population, Recorded Lineage & Verified Replay**: Inclusion and exclusion criteria are filters you tick in the sidebar; every page analyses the same population, every result says which population and data it was computed on, and a result computed on other data is not shown as current. Every change of the data goes through one write path and is one step of a hash-chained (tamper-evident) JSON trace that records what it did to the table. The headless engine (`run_trace`) replays the trace, checks every step against the fingerprint recorded for it, and says when the replayed table is exactly the analysed one; a multiple imputation is re-run and its imputed cells are checked; the few steps it cannot recompute (the single imputation adopted from the former multiple-imputation panel) are listed. A Microsoft Word (`.docx`) report documents the trace, and a Word **study dossier** gathers the provenance, the participant flow (STROBE item 13), the data handling, the missing data (with the multiple-imputation items of Sterne et al. 2009) and the results you add to it. Neither is an electronically signed record, and neither certifies compliance with a reporting guideline.

---

## At a Glance

| Area                     | What CohortStudio does                                                                                      |
| :----------------------- | :---------------------------------------------------------------------------------------------------------- |
| **Storage**              | Parquet versions (`name_v1`, `name_v2`, …) with a lineage file, in a workspace per user                      |
| **Analysis population**  | Inclusion, anomaly and outlier criteria ticked in the sidebar; participant flow (STROBE item 13)             |
| **Statistics**           | Welch-first comparisons with effect sizes, 2x2 measures, survival and regression, diagnostic accuracy, agreement |
| **Missing data**         | Little's MCAR test, single imputation for exploration, multiple imputation pooled with Rubin's rules         |
| **Outliers**             | IQR, z-score, quantiles, Isolation Forest, local outlier factor, DBSCAN                                      |
| **Clinical anomaly rules** | Bounds and expressions declared in the study configuration                                                 |
| **Reproducibility**      | Hash-chained trace (tamper-evident, not signed), verified replay, Word report and study dossier             |
| **AI (optional)**        | Local Ollama by default; Gemini, OpenAI, Mistral and Claude after an explicit consent                        |
| **Knowledge graphs**     | Offline Plotly/NetworkX engine by default; the proprietary yFiles engine is optional                         |
| **Deployment**           | Local (uv or pip), Docker or Docker Compose; offline with the default graph engine and a local LLM          |
| **Licence**              | Apache 2.0                                                                                                  |

---

## Architecture

```mermaid
flowchart TB
    UI["<b>Interface</b><br/>app_pages, ui: Streamlit pages, state facade, population selector, result cards"]
    SVC["<b>Services</b><br/>manage: traces, replay, Word report and study dossier, accounts (IAM)<br/>enrich, explore, monitor, graph"]
    ENG["<b>Engine</b>, no Streamlit (checked by tests/test_layering.py)<br/>biostats: statistics, multiple imputation, pooling<br/>ingest: file import · core: session, population, lineage, participant flow"]
    DATA["<b>Storage</b><br/>Parquet versions with lineage files · JSON traces · DuckDB accounts and statistics cache · per-user workspaces"]
    AI["<b>AI engine</b> (optional ai group)<br/>manage.rag_manager · manage.rag.providers: Ollama, Gemini, OpenAI, Mistral, Claude<br/>manage.vectorstores: LanceDB, Qdrant"]
    UI --> SVC --> ENG --> DATA
    SVC -.-> AI
```

---

## Module Breakdown

### 1. Data Import & Persistence

- CSV, TSV and text files, Excel workbooks (`.xlsx`, `.xls`, `.xlsm`) and Parquet; automatic encoding detection (UTF-8, Latin-1, CP1252) and delimiter detection (comma, semicolon, tab, pipe), with a parse report.
- Each saved version is a complete Parquet file (`dataset_v1`, `dataset_v2`, …) written with pyarrow, with a lineage file (fingerprint, parent trace, population) that is checked when the version is loaded.
- DuckDB holds the user accounts and the cache of dataset statistics.
- Multi-user isolation: every account has its own workspace under `<DATA_ROOT>/users/<username>/` (datasets, traces, uploads, taxonomies, knowledge graphs, document library, vector store); path inputs are confined to that workspace. Accounts are managed by a role-based IAM module with bcrypt hashing and login lockout.

### 2. Clinical Biostatistics & Epidemiological Testing

- **Normality Diagnostics**: Shapiro-Wilk and D'Agostino-Pearson tests, reported next to the results; they do not decide the test on their own.
- **Two-Group & Multi-Group Inference**: Welch's t-test by default (Student's t optional), Mann-Whitney U with the Hodges-Lehmann shift, Welch ANOVA (default) and one-way ANOVA, Kruskal-Wallis; pairwise comparisons report every pair with raw and adjusted p-values (Games-Howell, Tukey HSD, Dunn with Holm). Effect sizes with confidence intervals: Hedges' g (noncentral t), rank-biserial r, eta and omega squared, bias-corrected Cramer's V. Assumption checks are diagnostics, never gates.
- **2x2 Contingency & Cohort Epidemiology**: Relative Risk (RR), Odds Ratio (OR), Risk Difference (RD) with Haldane-Anscombe small-sample / zero-cell corrections, Wald & logit confidence intervals, and rare disease guidance ($OR \approx RR$).
- **Effect Sizes**: Cohen's $d$, Hedges' $g$ (small-sample bias correction), Eta-squared ($\eta^2$), Partial $\eta^2$, and Omega-squared ($\omega^2$).
- **Missingness Diagnostics**: Little's MCAR test on EM estimates of the mean and covariance (Little 1988) and a full-family scan of missingness-versus-observed dependencies with adjusted p-values.
- **ANCOVA**: Treatment comparisons of one outcome adjusted for continuous baseline covariates, with a test of homogeneity of regression slopes.
- **Survival & regression**: Kaplan-Meier curves with pointwise Greenwood intervals, log-rank test, Cox proportional hazards (Efron ties, Grambsch-Therneau check with the optional `survival` extra), logistic regression (odds ratios, DeLong AUC) and Poisson / negative-binomial regression with person-time offsets.
- **Paired, rates & stratified**: paired t and Wilcoxon signed-rank, one-sample tests, exact McNemar, incidence rates with exact Poisson intervals and rate ratios, Mantel-Haenszel odds and risk ratios with the Breslow-Day test, standardised mortality ratios, Cohen's kappa and intraclass correlations.
- **Multiple imputation (for inference)**: chained equations like R `mice` (predictive mean matching, Bayesian normal regression, logistic and polytomous regression with augmentation, random-forest donors), m independent chains with diagnostics (traces, R-hat, observed against imputed values), fitted on the analysis population. The Survival & regression and Analyze pages pool their analyses over the imputed datasets with Rubin's rules (Barnard-Rubin degrees of freedom, D1/D2 tests of several parameters, log, logit and cloglog scales) and show the complete-case analysis next to them; an analysis without a pooling rule says why. Pooling matches R `mice` 3.19 and `mitml` 0.4.5 on reference datasets. The imputation is saved with a version, exported for R (`mice` long) and Stata (`flong`), replayed and checked, and comes with a methods paragraph (Sterne et al. 2009).
- **Diagnostic Test Accuracy**: Sensitivity, Specificity, Positive/Negative Predictive Values (PPV/NPV), Diagnostic Odds Ratio, Youden's Index, Likelihood Ratios (LR+, LR-), and ROC AUC.
- **Multiple Testing Corrections**: Holm (the default) and Bonferroni for the family-wise error rate, Benjamini-Hochberg for the false discovery rate.

### 3. Data Quality & Anomaly Profiling

- **Completeness Index**: Total non-missing data ratio with variable-level missingness ranking.
- **Uniqueness Index**: Detection of exact duplicate rows.
- **Statistical Validity**: Univariate (IQR, Z-Score, Quantiles) and multivariate (Isolation Forest, Local Outlier Factor, DBSCAN) outlier scoring.
- **Consistency & Uniformity**: Identification of mis-typed numerical/date strings, trailing whitespace, and capitalization inconsistencies.
- **Clinical Rules**: Clinical exclusion and anomaly rules declared as bounds or expressions (`mask_families` in the study configuration, built on Validate or loaded from a JSON file).

### 4. Advanced Data Enrichment

- **Imputation Suites**: Targeted formula-driven imputation and global single imputation (mean/median, KNN, chained equations via IterativeImputer, tree-based missForest with LightGBM). Single imputation does not propagate the imputation uncertainty, so standard errors and p-values computed on singly-imputed data are too small: the filled cells are remembered (also in saved versions and replays), and every result that uses them says so. Use multiple imputation for inference.
- **Dimensionality Reduction**: t-SNE; Principal Component Analysis (PCA), Factor Analysis of Mixed Data (FAMD) and UMAP with the optional `ml` group.
- **Transformations**: one-hot, label and ordinal encoding; min-max normalisation, z-score standardisation and log(x+1); row-wise statistics across columns (mean, median, sum, minimum, maximum, standard deviation).
- **Population Clustering**: K-Means, DBSCAN and Gaussian mixture models; an elbow and silhouette chart helps choose the number of clusters.

### 5. AI Assistant (optional `ai` group)

- **Provider-Agnostic Interface**: Connectors for local Ollama (the default: prompts and documents stay on your server), Google Gemini, OpenAI, Mistral and Anthropic Claude, with the model chosen in Settings (Claude answers; local Ollama or another provider embeds the documents). Cloud providers are used only after an explicit consent in Settings > AI engine.
- **Column-Name Suggestions**: The language model suggests clearer, standardised names for raw columns. The suggestions are not checked against a terminology: UMLS, LOINC and SNOMED CT are not loaded.
- **Document Intelligence**: Vectorized retrieval over uploaded clinical protocols, biomedical research PDFs, and study documentation, indexed in LanceDB or Qdrant (selectable for your machine).
- **Knowledge Graphs**: Automated variable relationship induction into interactive knowledge graphs, rendered through a library-agnostic connector (`graph/`): the offline Plotly/NetworkX engine by default, the proprietary yFiles engine as an opt-in extra (`uv sync --extra graph-yfiles`, `CMP_GRAPH_RENDERER=yfiles`; it loads its runtime from yworks.com and reports anonymised usage data).

### 6. Pipeline Reproducibility & Audit Trail

- **Trace Engine**: Logs timestamped steps with function names, serialized parameters, and output snapshot paths.
- **Pipeline Replay**: Replays the recorded steps (enrichment joins, derived variables, single and targeted imputation fitted on the recorded population, date conversion, outlier clipping, kept population) headless (`run_trace`) or from the Versions & traces page, checks each step against its recorded fingerprint, re-runs a multiple imputation and checks that it finds the same imputed cells, and lists every step it could not recompute (the single imputation adopted from the former multiple-imputation panel).
- **Analysis Population**: Inclusion, anomaly and outlier criteria are ticked or unticked in the sidebar and filter every analysis without deleting rows; Validate shows the participant flow (rows in, excluded and out per family, per-criterion counts and reasons). Imputation on Enrich is fitted on the population rows only.
- **Integrity**: Trace steps are chained by SHA-256 hashes (tamper-evident, not signed); saved versions carry a lineage file (fingerprint, parent trace, population) checked when they are loaded.
- **Replay Modes**: Fast replay loads the snapshot recorded with each step; Full replay recomputes every step from the source file. Paths are resolved inside the user's workspace, on Windows and Linux.
- **Word Trace Report**: Export a Microsoft Word (`.docx`) report of the recorded steps and their parameters (not an electronically signed record).
- **Participant flow (STROBE item 13)**: a diagram from the records read in the file, through the rows removed by recorded steps and each family of criteria (with the criteria behind each exclusion), to the analysis population and the N of each analysis; click a box for its criteria and their definitions (Validate > Analysis population, Reproduce > Study dossier).
- **Study dossier**: add any result to the dossier with the button under it; Reproduce > Study dossier downloads a Word document with the provenance, the participant flow, the data handling, the missing data and multiple imputation, the results with what they were computed on, and the references. Results computed on other data or another population are left out, or kept and marked.

### 7. Interactive Clinical Visualisation & Figure Export

- **Clinical Plots** (Explore > Visualize): distributions (histogram, box, violin), categorical charts (bar, pie, sunburst, icicle, treemap), time series, scatter plots, correlation matrices, clustermaps, Sankey diagrams and ROC curves. Bland-Altman plots and diagnostic cut-offs are on Analyze > Hypothesis tests & epidemiology > Diagnostic accuracy & agreement.
- **Significance Brackets** (optional): pairwise Welch t-tests or Mann-Whitney U tests from the statistics engine, with the same multiple-testing correction as the summary table (Holm, Benjamini-Hochberg or Bonferroni).
- **Figure Export**: SVG from the camera icon of each figure, a self-contained interactive HTML file (Plotly.js embedded, so it opens offline) and the Plotly JSON specification (`.json`).
- **Underlying Analytical Data Extraction**: One-click download of the exact plotted subset as raw tabular CSV (`.csv`) or Microsoft Excel (`.xlsx`) datasets.

### 8. Pluggable Modular IAM Architecture

- **Provider Contract**: `BaseIAMProvider` (`manage/iam/base.py`) defines sign-in, account creation, session checks and roles. The application uses two roles, administrator and researcher.
- **Default Provider**: local accounts in DuckDB with bcrypt hashing and login lockout (`LocalDuckDBIAMProvider`); the first administrator comes from `.env` or from a one-time setup token.
- **Other Providers**: a Better Auth sidecar adapter (password sign-in and session checks over HTTP), and a mode without sign-in for a single local user (`CMP_IAM_PROVIDER=disabled`).

---

## Installation & Quick Start

### Option A: Using `uv` (Recommended)

[`uv`](https://github.com/astral-sh/uv) provides rapid, deterministic dependency resolution on Python 3.14+:

```bash
# Clone the repository
git clone https://github.com/KarimELMERNISSI/Cohort-Studio.git
cd Cohort-Studio

# Sync the virtual environment: the core application (statistics, data pipeline, analysis pages)
uv sync
# ...plus the optional groups you use, or all of them:
#   ai        LLM connectors, document library and chat, vector databases (LanceDB, Qdrant)
#   ml        MissForest imputation (LightGBM), UMAP, FAMD
#   survival  proportional-hazards test and concordance of the Cox model (lifelines)
uv sync --extra all

# Run the application
uv run streamlit run main.py
```

Without an optional group the application still runs: the methods that need it are not offered, and the page says
which group to install.

### Option B: Standard Python Virtual Environment

```bash
# Clone the repository
git clone https://github.com/KarimELMERNISSI/Cohort-Studio.git
cd Cohort-Studio

# Create and activate Python 3.14+ virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies (requirements.txt holds every optional group except the proprietary
# graph-yfiles engine, which is in requirements-yfiles.txt)
pip install -r requirements.txt

# Run the application
streamlit run main.py
```

### Option C: Containerized Deployment (Docker)

```bash
# Build and start (Docker Compose v2); data, accounts and traces live in ./data
docker compose up -d --build

# First start: the log shows a one-time setup token; the login page asks for it to create the administrator
docker compose logs cohortstudio
```

Without Compose:

```bash
docker build -t cohortstudio:latest .
cp .env.example .env       # then fill in what you use
docker run -d -p 8501:8501 -v ${PWD}/data:/app/data --env-file .env --name cohortstudio cohortstudio:latest
docker logs cohortstudio   # the one-time setup token
```

The container runs as uid 1000: on a Linux host, run `chown -R 1000:1000 data` once before the first start.

The image installs every optional group except the proprietary graph-yfiles engine (the build argument
`WITH_YFILES=1` adds it). CI builds it on every push, starts a fresh container and checks its health endpoint and
the first-run setup.

---

## Environment Configuration

Copy `.env.example` to `.env` and fill in what you use (the example documents every variable the application reads). Key settings:

```env
# Application storage root (default: ./data)
DATA_ROOT=./data

# First administrator: created on first start when no active administrator exists
# (alternatively: python manage_users.py create-admin <username>)
CMP_ADMIN_USERNAME=admin
CMP_ADMIN_PASSWORD=choose-a-long-passphrase

# AI providers: Ollama is the default and keeps data on the server. Cloud providers are opt-in
# and require an explicit confirmation in the app before any data is sent.
OLLAMA_BASE_URL=http://localhost:11434
GOOGLE_API_KEY=
OPENAI_API_KEY=
MISTRAL_API_KEY=
ANTHROPIC_API_KEY=
```

### Administrator accounts and user management

- Accounts created from the sign-in page are **pending** until an administrator activates them. The username `admin` receives no special treatment.
- Bootstrap the first administrator with `CMP_ADMIN_USERNAME` / `CMP_ADMIN_PASSWORD` (or `CMP_ADMIN_PASSWORD_FILE`), or run `python manage_users.py create-admin <username>` (the password is prompted, never passed as an argument).
- Without them, the first start writes a **one-time setup token** to the server log, and the login page creates the administrator with it. Only a hash of the token is stored; it stops working once used. `python manage_users.py setup-token` issues a new one.
- `python manage_users.py list | activate | deactivate | reset-password <username>` covers day-to-day administration.

---

## Verification & Test Suite

The automated suite covers the statistics engine (reference values from R and the literature), the import engine, per-user workspaces and the expression sandbox, Parquet persistence, transformation lineage and replay, the graph connector, the AI connectors, and every page through Streamlit's headless AppTest; a layering test and source-level quality gates guard the architecture. CI runs ruff, pyright and the suite with a coverage floor on every push:

```bash
# Install the development tools (pytest, ruff, pyright) and every optional group once
uv sync --extra dev --extra all

# Run the complete test suite
uv run pytest -v

# Run with test coverage analysis
uv run pytest --cov=. --cov-report=term-missing
```

```text
776 passed, 2 deselected (live-provider integration test, slow simulations) in about 6 minutes
Line coverage about 66 % of the non-test code (CI floor 63 %), with per-package floors in CI:
biostats and core 92 %, graph 91 %, ingest 87 %, ui 81 %, explore 74 %, manage 61 %, enrich 56 %
```

CI also runs the suite in an environment without the optional groups (the AI tests are skipped), builds the Docker
image and checks a fresh container, and runs the statistical simulation suites (`pytest -m slow`) every week.

| Test Module                                                                                                      | Coverage Scope                                                                                                                                                                                                                          | Status     |
| :--------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------- |
| [`tests/biostats_layer/`](tests/biostats_layer/)                                                                 | 2x2 orientation and measures, power semantics, missingness scan, Z-scores, Welch-first inference, post-hoc tables, effect sizes with CIs, normality, EM-based Little's test, diagnostic accuracy, Bland-Altman, mixed-type associations, multiple imputation (pooling with R parity, engine, adapters, binding) | **PASSED** |
| [`tests/core_layer/`](tests/core_layer/)                                                                         | Whitelisted expression evaluator (allow/deny corpora), naming patterns, per-user workspaces, path containment, login lockout                                                                                                            | **PASSED** |
| [`tests/ingest_layer/`](tests/ingest_layer/)                                                                     | Encoding/delimiter/decimal detection, parse reports, Excel sheet merge, sample cohort properties                                                                                                                                        | **PASSED** |
| [`tests/graph_layer/`](tests/graph_layer/)                                                                       | Graph model, builders, palette, Plotly and yFiles translation, renderer policy and contract                                                                                                                                             | **PASSED** |
| [`tests/ui_layer/`](tests/ui_layer/)                                                                             | Theme tokens vs config, navigation registry, `AppState.set_dataset` invalidation and trace, test chooser, glossary, help-text coverage, AppTest smoke over every page                                                                   | **PASSED** |
| [`tests/test_layering.py`](tests/test_layering.py), [`tests/test_quality_gates.py`](tests/test_quality_gates.py) | Dependency direction (no Streamlit in domain modules), no `eval`, no sleeps, no hard-coded paths, raw-HTML and `print` ratchets                                                                                                         | **PASSED** |
| [`tests/test_authz.py`](tests/test_authz.py)                                                                     | Role checks, username and password policy                                                                                                                                                                                               | **PASSED** |
| [`tests/test_epidemiology_stats.py`](tests/test_epidemiology_stats.py)                                           | 2x2 Contingency, Effect Sizes, ANCOVA, Diagnostic Accuracy                                                                                                                                                                              | **PASSED** |
| [`tests/test_epidemiology_integration.py`](tests/test_epidemiology_integration.py)                               | Two-group, ANOVA, Little's MCAR, Multiple corrections                                                                                                                                                                                   | **PASSED** |
| [`tests/test_date_parser_and_stats.py`](tests/test_date_parser_and_stats.py)                                     | Date parsing (ISO, day-first, Excel serial numbers), normality tests, Arrow-safe frames                                                                                                                                                  | **PASSED** |
| [`tests/test_corr_matrix.py`](tests/test_corr_matrix.py)                                                         | Mixed-type Correlation Matrix (Pearson, Spearman, Cramér's V)                                                                                                                                                                           | **PASSED** |
| [`tests/test_export_utils.py`](tests/test_export_utils.py)                                                       | SVG export settings, interactive HTML (also self-contained), Plotly JSON, CSV and Excel                                                                                                                                                                | **PASSED** |
| [`tests/test_db_manager.py`](tests/test_db_manager.py)                                                           | bcrypt accounts and activation, Parquet versions with row ids and fingerprints, per-user datasets                                                                                                                                                                                     | **PASSED** |
| [`tests/test_iam_framework.py`](tests/test_iam_framework.py)                                                     | IAM providers (local, disabled, Better Auth sidecar adapter), bootstrap administrator, lockout                                                                                                                                                               | **PASSED** |
| [`tests/test_transformation_manager.py`](tests/test_transformation_manager.py)                                   | Session lineage, complex parameter serialization, traces                                                                                                                                                                                | **PASSED** |
| [`tests/test_reproduction_manager.py`](tests/test_reproduction_manager.py)                                       | Headless pipeline re-execution (`execute_trace_pipeline`), snapshot replay                                                                                                                                                              | **PASSED** |
| [`tests/test_data_quality_auditor.py`](tests/test_data_quality_auditor.py)                                       | Completeness, Uniqueness, Outlier validity, MCAR heuristics                                                                                                                                                                             | **PASSED** |
| [`tests/test_data_analyzer.py`](tests/test_data_analyzer.py)                                                     | `DatasetProfile` immutability, type categorization, charts                                                                                                                                                                              | **PASSED** |
| [`tests/test_rag_connectors.py`](tests/test_rag_connectors.py)                                                   | Gemini, Ollama, OpenAI, Mistral factory & failover                                                                                                                                                                                      | **PASSED** |
| [`tests/test_rag_evaluator.py`](tests/test_rag_evaluator.py)                                                     | LLM-as-judge evaluation (retrieval relevance, faithfulness), semantic similarity, JSON validity                                                                                                                                                              | **PASSED** |
| [`tests/test_rag_pipeline_steps.py`](tests/test_rag_pipeline_steps.py)                                           | RAG retrieval, prompt augmentation, grounding validation                                                                                                                                                                                | **PASSED** |
| [`tests/test_clustering_utils.py`](tests/test_clustering_utils.py)                                               | K-Means, DBSCAN, GMM, Silhouette score, cluster stratification                                                                                                                                                                          | **PASSED** |
| [`tests/test_trace_documenter.py`](tests/test_trace_documenter.py)                                               | Microsoft Word `.docx` transformation report generation                                                                                                                                                                                 | **PASSED** |

---

## Complete Documentation Index

For in-depth guides and methodological handbooks, consult the [`documentation/`](documentation/) directory:

- [**00_Researcher_Handbook.md**](documentation/00_Researcher_Handbook.md): Methodological workflow, clinical philosophy, and phase-by-phase execution guide.
- [**01_Home.md**](documentation/01_Home.md): The Start page, file import (encoding, delimiter and decimal detection, parse report) and saved versions.
- [**02_Data_Validation.md**](documentation/02_Data_Validation.md): Quality scorecards, clinical anomaly masks, outlier clipping, and version comparisons.
- [**03_Data_Enrichment.md**](documentation/03_Data_Enrichment.md): Merges, derived variables, targeted and global single imputation, multiple imputation for inference, dimensionality reduction, and clustering.
- [**04_Visualization.md**](documentation/04_Visualization.md): Interactive Plotly clinical graphics with integrated statistical significance overlays.
- [**05_Epidemiology.md**](documentation/05_Epidemiology.md): Parametric/non-parametric testing, ANCOVA, power analysis, and epidemiological ratios.
- [**06_Data_Insight.md**](documentation/06_Data_Insight.md): The Variable map: a knowledge graph of the study variables built with the AI engine.
- [**07_Document_Insight.md**](documentation/07_Document_Insight.md): Clinical document summarization, PDF RAG chat, and study coverage analysis.
- [**08_Reproduction.md**](documentation/08_Reproduction.md): Fast and full replay of the trace with verification, the Word report of the trace, and the study dossier.
- [**09_Clustering.md**](documentation/09_Clustering.md): Algorithmic population stratification guidelines.
- [**10_Developer_Guide.md**](documentation/10_Developer_Guide.md): Architecture specifications, plugin connectors, and development guidelines.
- [**11_UI_Style_Guide.md**](documentation/11_UI_Style_Guide.md): Page structure, components and wording rules of the interface.
- [**12_IAM_Architecture_and_BetterAuth_Evaluation.md**](documentation/12_IAM_Architecture_and_BetterAuth_Evaluation.md): Pluggable authentication architecture and the evaluation of a Better Auth sidecar.
- [**13_Audit_Report_2026-10.md**](documentation/13_Audit_Report_2026-10.md): The October 2026 audit and the follow-up programme (M0 to M4).
- [**14_Full_Tutorial.md**](documentation/14_Full_Tutorial.md): From a raw file to a reproducible report, step by step on the tutorial cohort.
- [**Tutorials.md**](documentation/Tutorials.md): Shorter walkthroughs of each page.

---

## License & Citation

Distributed under the **Apache License, Version 2.0**. See [`LICENSE`](LICENSE) for full legal text.

If you use CohortStudio in academic or clinical research, please cite:

```bibtex
@software{elmernissi2026cohortstudio,
  author  = {El Mernissi, Karim},
  title   = {CohortStudio: Clinical cohort data, from raw file to reproducible evidence},
  year    = {2026},
  version = {2.3.0},
  url     = {https://github.com/KarimELMERNISSI/Cohort-Studio},
  license = {Apache-2.0}
}
```
