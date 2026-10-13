# Cohort Monitoring Package

[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/KarimELMERNISSI/Cohort-Monitoring-Package/actions/workflows/ci.yml/badge.svg)](https://github.com/KarimELMERNISSI/Cohort-Monitoring-Package/actions/workflows/ci.yml)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![DuckDB Version](https://img.shields.io/badge/DuckDB-1.5%2B-fff?logo=duckdb&logoColor=black)](https://duckdb.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-0ea5e9.svg)](https://karimelmernissi.github.io/Cohort-Monitoring-Package/)

> **High-Performance Clinical Cohort Intelligence & Epidemiological Analytics**  
> An open-source, enterprise-grade platform uniting zero-lock-in DuckDB Parquet persistence, rigorous biostatistical inference, and model-agnostic medical RAG in an intuitive, production-ready interface.

---

## Executive Overview

The **Cohort Monitoring Package (CMP)** is engineered for clinical researchers, biostatisticians, and health data scientists managing observational cohorts, registry databases, and clinical trial datasets.

Traditional cohort data workflows often oscillate between fragmented ad-hoc Python/R scripts, unversioned CSV spreadsheets, and rigid commercial Electronic Data Capture (EDC) systems. CMP provides a unified, local-first platform built on **Python 3.14+**, **DuckDB 1.5+**, and **Streamlit 1.65+** that enforces end-to-end data integrity, statistical validity, and reproducible audit trails.

### Core Architecture Pillars

1. **Zero-Lock-In High-Performance Storage**: Native Parquet dataset versioning and caching executed via DuckDB columnar queries. Complete user data isolation with transactional stability on Windows and Linux.
2. **Statistically Rigorous Epidemiology Engine**: Automated normality profiling, parametric and non-parametric comparative hypothesis testing, multivariate ANCOVA with parallel slope diagnostics, effect size quantification ($\text{Cohen's } d$, $\text{Hedges' } g$, $\eta^2$, $\omega^2$), Little's MCAR missingness testing, Haldane-Anscombe zero-cell continuity corrections for odds ratios, and diagnostic test accuracy analytics (ROC, PPV, NPV, Likelihood Ratios).
3. **Multi-Dimensional Data Quality Auditing**: Real-time scoring across Completeness, Uniqueness, Consistency, Uniformity, Statistical Outlier Validity (IQR, Z-Score, Quantiles, Isolation Forest, Local Outlier Factor), and declared Clinical Anomaly Rules.
4. **Model-Agnostic Medical RAG**: Pluggable connector architecture supporting Google Gemini 2.5, local Ollama (air-gapped/on-premise privacy), OpenAI, and Mistral for medical document synthesis, semantic column standardization (UMLS/LOINC/SNOMED), and automated variable engineering.
5. **Deterministic Lineage, Complete Pipeline Reproduction & Audit Dossiers**: Every transformation step, imputation, and enrichment join is logged into an executable JSON session trace. The headless reproduction engine can re-execute the complete multi-step pipeline (`execute_trace_pipeline`) to recreate identical analytical datasets deterministically, while generating signed Microsoft Word (`.docx`) transformation certificates.

---

## Capability Comparison Matrix

| Capability                        | Ad-Hoc Scripts / Notebooks   | Commercial EDC (REDCap/Medidata) | Cohort Monitoring Package                                                                  |
| :-------------------------------- | :--------------------------- | :------------------------------- | :----------------------------------------------------------------------------------------- |
| **Storage Architecture**          | Flat CSVs / Memory Pandas    | Row-based SQL / Cloud RDBMS      | **Columnar DuckDB + Parquet Snapshots**                                                    |
| **Data Versioning**               | Manual copies or Git LFS     | Audit log only (no snapshotting) | **Automated Zero-Copy Versioning (`_v1`, `_v2`)**                                          |
| **Biostatistical Validation**     | Manual script assembly       | Limited / Third-party add-ons    | **Integrated Test Guidance & Effect Sizes**                                                |
| **Missingness Testing**           | Custom R / Python code       | None                             | **Automated Little's MCAR & Pairwise MAR**                                                 |
| **Outlier Detection**             | Hardcoded IQR / standard dev | Manual range validation          | **Univariate + ML (Isolation Forest, LOF, DBSCAN)**                                        |
| **Clinical Anomaly Rules**        | Custom if-else blocks        | Form validation rules            | **Declarative Expression Masks & Highlight Tables**                                        |
| **Document Intelligence**         | Disconnected LLM chats       | None                             | **Model-Agnostic RAG (Local Ollama, Gemini, OpenAI)**                                      |
| **Knowledge Graph Taxonomy**      | Manual drawing               | Controlled terminologies         | **Interactive knowledge graphs (offline Plotly engine by default, yFiles optional)**       |
| **Audit Trail & Pipeline Replay** | Fragile notebook state       | Fixed PDF audit trail            | **Full Pipeline Re-Execution (Headless & UI) + Word Dossiers**                             |
| **Deployment Model**              | Local interpreter            | Cloud SaaS subscription          | **Self-Hosted / Docker / Cloud; air-gapped with the default graph engine and a local LLM** |
| **License**                       | N/A                          | Proprietary / Commercial fees    | **Open-Source (Apache 2.0)**                                                               |

---

## Architectural Blueprint

```text
+---------------------------------------------------------------------------------------+
|                                  PRESENTATION LAYER                                   |
| Streamlit 1.65 st.navigation shell: Start / Data (Import, Validate, Enrich) / Explore |
|           Analyze / Knowledge / Reproduce / Help (Handbook, Settings) / Admin         |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|                               DOMAIN & COMPUTATION LAYER                              |
|  • utils.data_analyzer (DatasetProfile)    • explore.data_quality_auditor (Scoring)    |
|  • explore.epidemiology_stats (Inference)  • enrich.imputation_pipeline (MICE, Forest) |
|  • explore.corr_matrix (Rank/Bivariate)    • explore.outliers (Multi-method Detection) |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|                             ORCHESTRATION & STATE LAYER                               |
|  • manage.transformation_manager           • manage.reproduction_manager (Pipeline)  |
|  • manage.db_manager (DuckDB & bcrypt)     • manage.trace_documenter (.docx Dossier) |
+---------------------+-------------------------------------+---------------------------+
                      |                                     |
                      v                                     v
+----------------------------------+  +-------------------------------------------------+
|       PERSISTENCE LAYER          |  |             RAG & AGENTIC LAYER                 |
|  • DuckDB 1.5+ Parquet Engine    |  |  • manage.rag_manager (Unified Orchestration)   |
|  • User DB (bcrypt Auth)         |  |  • manage.rag_connectors (Gemini, Ollama, OpenAI)|
|  • ChromaDB Vector Store         |  |  • manage.rag_documents (PDF Chunking & Search) |
|  • Traces & Stats Parquet Cache  |  |  • manage.rag_taxonomy (Knowledge Graph Bridge) |
+----------------------------------+  +-------------------------------------------------+
```

---

## Module Breakdown

### 1. Data Ingestion & DuckDB Persistence

- Automatic encoding detection (UTF-8, Latin-1, CP1252) and dynamic delimiter parsing (comma, semicolon, tab, pipe).
- Columnar Parquet persistence with atomic version incrementation (`dataset_v1`, `dataset_v2`).
- Query acceleration: instantaneous analytical aggregations powered by embedded DuckDB.
- Multi-user isolation: every account has its own workspace under `<DATA_ROOT>/users/<username>/` (datasets, traces, uploads, taxonomies, knowledge graphs, document library, vector store); path inputs are confined to that workspace. Accounts are managed by a role-based IAM module with bcrypt hashing and login lockout.

### 2. Clinical Biostatistics & Epidemiological Testing

- **Normality & Distribution Profiling**: Shapiro-Wilk, D'Agostino-Pearson omnibus tests with automated test recommendation logic.
- **Two-Group & Multi-Group Inference**: Welch's t-test by default (Student's t optional), Mann-Whitney U with the Hodges-Lehmann shift, Welch ANOVA (default) and one-way ANOVA, Kruskal-Wallis; pairwise comparisons report every pair with raw and adjusted p-values (Games-Howell, Tukey HSD, Dunn with Holm). Effect sizes with confidence intervals: Hedges' g (noncentral t), rank-biserial r, eta and omega squared, bias-corrected Cramer's V. Assumption checks are diagnostics, never gates.
- **2x2 Contingency & Cohort Epidemiology**: Relative Risk (RR), Odds Ratio (OR), Risk Difference (RD) with Haldane-Anscombe small-sample / zero-cell corrections, Wald & logit confidence intervals, and rare disease guidance ($OR \approx RR$).
- **Effect Sizes**: Cohen's $d$, Hedges' $g$ (small-sample bias correction), Eta-squared ($\eta^2$), Partial $\eta^2$, and Omega-squared ($\omega^2$).
- **Missingness Diagnostics**: Little's MCAR test on EM estimates of the mean and covariance (Little 1988) and a full-family scan of missingness-versus-observed dependencies with adjusted p-values.
- **Multivariate ANCOVA**: Treatment comparisons adjusting for continuous baseline covariates, featuring automated test of homogeneity of regression slopes.
- **Survival & regression**: Kaplan-Meier curves with pointwise Greenwood intervals, log-rank test, Cox proportional hazards (Efron ties, Grambsch-Therneau check with the optional `survival` extra), logistic regression (odds ratios, DeLong AUC) and Poisson / negative-binomial regression with person-time offsets.
- **Paired, rates & stratified**: paired t and Wilcoxon signed-rank, one-sample tests, exact McNemar, incidence rates with exact Poisson intervals and rate ratios, Mantel-Haenszel odds and risk ratios with the Breslow-Day test, standardised mortality ratios, Cohen's kappa and intraclass correlations.
- **Multiple imputation**: m completed datasets by chained equations with posterior sampling, pooled by Rubin's rules with the fraction of missing information.
- **Diagnostic Test Accuracy**: Sensitivity, Specificity, Positive/Negative Predictive Values (PPV/NPV), Diagnostic Odds Ratio, Youden's Index, Likelihood Ratios (LR+, LR-), and ROC AUC.
- **Multiple Testing Corrections**: False Discovery Rate (Benjamini-Hochberg) and Family-Wise Error Rate (Bonferroni).

### 3. Data Quality & Anomaly Profiling

- **Completeness Index**: Total non-missing data ratio with variable-level missingness ranking.
- **Uniqueness Index**: Detection of redundant cohort entries and duplicate patient records.
- **Statistical Validity**: Univariate (IQR, Z-Score, Quantiles) and multivariate (Isolation Forest, Local Outlier Factor, DBSCAN) outlier scoring.
- **Consistency & Uniformity**: Identification of mis-typed numerical/date strings, trailing whitespace, and capitalization inconsistencies.
- **Clinical Rule Enforcement**: Declarative execution of clinical exclusion and anomaly boundaries (`mask_families` in `config/config.json`).

### 4. Advanced Data Enrichment

- **Imputation Suites**: Targeted formula-driven imputation and global single imputation (mean/median, KNN, chained equations via IterativeImputer, tree-based missForest with LightGBM). Imputation uncertainty is not propagated; multiple imputation with pooling is planned.
- **Dimensionality Reduction**: Principal Component Analysis (PCA), Factor Analysis of Mixed Data (FAMD), t-SNE, and UMAP.
- **Feature Engineering**: One-Hot, Ordinal, Label encoding, mathematical transformations (log, square root, Box-Cox), and robust/standard scaling.
- **Population Clustering**: K-Means, DBSCAN, and Gaussian Mixture Models (GMM) with automated hyperparameter sweeps.

### 5. Model-Agnostic Medical RAG

- **Provider-Agnostic Interface**: Direct connectors for Google Gemini 2.5, local Ollama (air-gapped clinical deployments), OpenAI (GPT-4o), and Mistral.
- **Medical Semantic Renaming**: Harmonizes raw electronic health record columns with standard terminologies (UMLS, LOINC, SNOMED CT).
- **Document Intelligence**: Vectorized retrieval over uploaded clinical protocols, biomedical research PDFs, and study documentation via ChromaDB.
- **Knowledge Graphs**: Automated variable relationship induction into interactive knowledge graphs, rendered through a library-agnostic connector (`graph/`): the offline Plotly/NetworkX engine by default, the proprietary yFiles engine as an opt-in extra (`uv sync --extra graph-yfiles`, `CMP_GRAPH_RENDERER=yfiles`; it loads its runtime from yworks.com and reports anonymised usage data).

### 6. Pipeline Reproducibility & Audit Trail

- **Deterministic Trace Engine**: Logs timestamped operational trees with function signatures, serialized parameters, and output snapshot paths.
- **Complete Pipeline Reproduction**: Re-executes entire multi-step transformation pipelines (enrichment joins, MICE/MissForest imputations, feature scaling, mathematical transforms) headless (`execute_trace_pipeline`) or via UI (`reproduce_trace`) to guarantee exact reproducible results across environments.
- **Replay Verification**: Full (data re-execution) and Fast (parameter audit) replay capabilities with cross-platform path resolution.
- **Word Dossier Generation**: Export publication-ready Microsoft Word (`.docx`) audit dossiers summarizing cohort transformations.

### 7. Interactive Clinical Visualisation & 1-Click Fast Figure Export

- **Multi-Modal Clinical Plots**: Distributions (KDE, box, violin, eCDF), cohort stratification, correlation heatmaps, Bland-Altman agreement plots, and ROC curves with optimal Youden's $J$ cutoffs.
- **Integrated Hypothesis Test Overlays**: Automated routing to parametric (Student's t / ANOVA) or non-parametric (Mann-Whitney U / Kruskal-Wallis) tests with Bonferroni / Benjamini-Hochberg FDR adjustments rendered directly on plots with publication brackets.
- **Publication Graphics Export**: 1-click export of figures in vector SVG, 300+ DPI raster PNG, standalone interactive HTML (`.html`), and full serialized Plotly JSON (`.json`) schemas.
- **Underlying Analytical Data Extraction**: One-click download of the exact plotted subset as raw tabular CSV (`.csv`) or Microsoft Excel (`.xlsx`) datasets.

### 8. Pluggable Modular IAM Architecture

- **Protocol-First Contract**: `AuthService` protocol defining authentication, user creation, session verification, and role-based permissions (`Role.ADMIN`, `Role.RESEARCHER`, `Role.AUDITOR`).
- **Default In-Process Adapter**: `DefaultAuthService` powered by embedded DuckDB and bcrypt hashing for zero-dependency, local-first environments.
- **Extensible Identity Adapters**: Drop-in architectural support for external enterprise IAM (Better Auth HTTP sidecar, Keycloak OIDC, LDAP) without leaking dependencies or degrading core analytical throughput.

---

## Installation & Quick Start

### Option A: Using `uv` (Recommended)

[`uv`](https://github.com/astral-sh/uv) provides rapid, deterministic dependency resolution on Python 3.14+:

```bash
# Clone the repository
git clone https://github.com/KarimELMERNISSI/Cohort-Monitoring-Package.git
cd Cohort-Monitoring-Package

# Sync virtual environment and dependencies
uv sync

# Run the application
uv run streamlit run main.py
```

### Option B: Standard Python Virtual Environment

```bash
# Clone the repository
git clone https://github.com/KarimELMERNISSI/Cohort-Monitoring-Package.git
cd Cohort-Monitoring-Package

# Create and activate Python 3.14+ virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main.py
```

### Option C: Containerized Deployment (Docker)

```bash
# Build the production image
docker build -t cohort-monitoring-package:latest .

# Run container with mounted local volume
docker run -d \
  -p 8501:8501 \
  -v ${PWD}/data:/app/data \
  -e GOOGLE_API_KEY="your-gemini-key" \
  --name cohort-platform \
  cohort-monitoring-package:latest
```

Using Docker Compose:

```bash
docker compose up -d
```

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
```

### Administrator accounts and user management

- Accounts created from the sign-in page are **pending** until an administrator activates them. The username `admin` receives no special treatment.
- Bootstrap the first administrator with `CMP_ADMIN_USERNAME` / `CMP_ADMIN_PASSWORD` (or `CMP_ADMIN_PASSWORD_FILE`), or run `python manage_users.py create-admin <username>` (the password is prompted, never passed as an argument).
- `python manage_users.py list | activate | deactivate | reset-password <username>` covers day-to-day administration.

---

## Verification & Test Suite

The automated suite covers the statistics engine (reference values from R and the literature), the import engine, per-user workspaces and the expression sandbox, DuckDB persistence, transformation lineage and replay, the graph connector, the RAG connectors, and every page through Streamlit's headless AppTest; a layering test and source-level quality gates guard the architecture. CI runs ruff, pyright and the suite with a coverage floor on every push:

```bash
# Run the complete test suite
uv run pytest -v

# Run with test coverage analysis
uv run pytest --cov=. --cov-report=term-missing
```

```text
458 passed, 1 deselected (live-provider integration test) in about 4 minutes
Line coverage 55 % (CI floor 50 %); ruff and pyright (scoped) clean
```

| Test Module                                                                                                      | Coverage Scope                                                                                                                                                                                                                          | Status     |
| :--------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------- |
| [`tests/biostats_layer/`](tests/biostats_layer/)                                                                 | 2x2 orientation and measures, power semantics, missingness scan, Z-scores, Welch-first inference, post-hoc tables, effect sizes with CIs, normality, EM-based Little's test, diagnostic accuracy, Bland-Altman, mixed-type associations | **PASSED** |
| [`tests/core_layer/`](tests/core_layer/)                                                                         | Whitelisted expression evaluator (allow/deny corpora), naming patterns, per-user workspaces, path containment, login lockout                                                                                                            | **PASSED** |
| [`tests/ingest_layer/`](tests/ingest_layer/)                                                                     | Encoding/delimiter/decimal detection, parse reports, Excel sheet merge, sample cohort properties                                                                                                                                        | **PASSED** |
| [`tests/graph_layer/`](tests/graph_layer/)                                                                       | Graph model, builders, palette, Plotly and yFiles translation, renderer policy and contract                                                                                                                                             | **PASSED** |
| [`tests/ui_layer/`](tests/ui_layer/)                                                                             | Theme tokens vs config, navigation registry, `AppState.set_dataset` invalidation and trace, test chooser, glossary, help-text coverage, AppTest smoke over every page                                                                   | **PASSED** |
| [`tests/test_layering.py`](tests/test_layering.py), [`tests/test_quality_gates.py`](tests/test_quality_gates.py) | Dependency direction (no Streamlit in domain modules), no `eval`, no sleeps, no hard-coded paths, raw-HTML and `print` ratchets                                                                                                         | **PASSED** |
| [`tests/test_authz.py`](tests/test_authz.py)                                                                     | Role checks, username and password policy                                                                                                                                                                                               | **PASSED** |
| [`tests/test_epidemiology_stats.py`](tests/test_epidemiology_stats.py)                                           | 2x2 Contingency, Effect Sizes, ANCOVA, Diagnostic Accuracy                                                                                                                                                                              | **PASSED** |
| [`tests/test_epidemiology_integration.py`](tests/test_epidemiology_integration.py)                               | Two-group, ANOVA, Little's MCAR, Multiple corrections                                                                                                                                                                                   | **PASSED** |
| [`tests/test_date_parser_and_stats.py`](tests/test_date_parser_and_stats.py)                                     | Kolmogorov-Smirnov, Shapiro-Wilk, Levene, Spearman/Pearson correlation, Date Ingestion                                                                                                                                                  | **PASSED** |
| [`tests/test_corr_matrix.py`](tests/test_corr_matrix.py)                                                         | Mixed-type Correlation Matrix (Pearson, Spearman, Cramér's V)                                                                                                                                                                           | **PASSED** |
| [`tests/test_export_utils.py`](tests/test_export_utils.py)                                                       | Vector SVG, 300+ DPI PNG, Standalone HTML, Plotly JSON, CSV/Excel export                                                                                                                                                                | **PASSED** |
| [`tests/test_db_manager.py`](tests/test_db_manager.py)                                                           | bcrypt Auth, Activation, DuckDB Parquet persistence                                                                                                                                                                                     | **PASSED** |
| [`tests/test_iam_framework.py`](tests/test_iam_framework.py)                                                     | Pluggable IAM contract, DefaultAuthService, BetterAuth sidecar evaluation                                                                                                                                                               | **PASSED** |
| [`tests/test_transformation_manager.py`](tests/test_transformation_manager.py)                                   | Session lineage, complex parameter serialization, traces                                                                                                                                                                                | **PASSED** |
| [`tests/test_reproduction_manager.py`](tests/test_reproduction_manager.py)                                       | Headless pipeline re-execution (`execute_trace_pipeline`), snapshot replay                                                                                                                                                              | **PASSED** |
| [`tests/test_data_quality_auditor.py`](tests/test_data_quality_auditor.py)                                       | Completeness, Uniqueness, Outlier validity, MCAR heuristics                                                                                                                                                                             | **PASSED** |
| [`tests/test_data_analyzer.py`](tests/test_data_analyzer.py)                                                     | `DatasetProfile` immutability, type categorization, charts                                                                                                                                                                              | **PASSED** |
| [`tests/test_rag_connectors.py`](tests/test_rag_connectors.py)                                                   | Gemini, Ollama, OpenAI, Mistral factory & failover                                                                                                                                                                                      | **PASSED** |
| [`tests/test_rag_evaluator.py`](tests/test_rag_evaluator.py)                                                     | RAGAS framework metrics, Faithfulness, Answer Relevance, Context Precision                                                                                                                                                              | **PASSED** |
| [`tests/test_rag_pipeline_steps.py`](tests/test_rag_pipeline_steps.py)                                           | RAG retrieval, prompt augmentation, grounding validation                                                                                                                                                                                | **PASSED** |
| [`tests/test_clustering_utils.py`](tests/test_clustering_utils.py)                                               | K-Means, DBSCAN, GMM, Silhouette score, cluster stratification                                                                                                                                                                          | **PASSED** |
| [`tests/test_trace_documenter.py`](tests/test_trace_documenter.py)                                               | Microsoft Word `.docx` transformation report generation                                                                                                                                                                                 | **PASSED** |

---

## Complete Documentation Index

For in-depth guides and methodological handbooks, consult the [`documentation/`](documentation/) directory:

- [**00_Researcher_Handbook.md**](documentation/00_Researcher_Handbook.md): Methodological workflow, clinical philosophy, and phase-by-phase execution guide.
- [**01_Home.md**](documentation/01_Home.md): Ingestion pipelines, automatic encoding/delimiter resolution, and DuckDB performance benchmarks.
- [**02_Data_Validation.md**](documentation/02_Data_Validation.md): Quality scorecards, clinical anomaly masks, outlier clipping, and version comparisons.
- [**03_Data_Enrichment.md**](documentation/03_Data_Enrichment.md): Targeted and global imputation (MICE, MissForest), dimensionality reduction, and clustering.
- [**04_Visualization.md**](documentation/04_Visualization.md): Interactive Plotly clinical graphics with integrated statistical significance overlays.
- [**05_Epidemiology.md**](documentation/05_Epidemiology.md): Parametric/non-parametric testing, ANCOVA, power analysis, and epidemiological ratios.
- [**06_Data_Insight.md**](documentation/06_Data_Insight.md): Knowledge graph generation, semantic taxonomy repair, and node enrichment.
- [**07_Document_Insight.md**](documentation/07_Document_Insight.md): Clinical document summarization, PDF RAG chat, and study coverage analysis.
- [**08_Reproduction.md**](documentation/08_Reproduction.md): Deterministic trace replay, session auditing, and Word report compilation.
- [**09_Clustering.md**](documentation/09_Clustering.md): Algorithmic population stratification guidelines.
- [**10_Developer_Guide.md**](documentation/10_Developer_Guide.md): Architecture specifications, plugin connectors, and development guidelines.
- [**11_RAG_Scientific_Evaluation.md**](documentation/11_RAG_Scientific_Evaluation.md): RAGAS framework metrics, Faithfulness, Answer Relevance, Context Precision, and benchmark audit procedures.
- [**12_IAM_Architecture_and_BetterAuth_Evaluation.md**](documentation/12_IAM_Architecture_and_BetterAuth_Evaluation.md): Pluggable authentication architecture, enterprise Better Auth sidecar evaluation, and latency benchmarking.
- [**Tutorials.md**](documentation/Tutorials.md): End-to-end clinical case study walkthroughs.

---

## License & Citation

Distributed under the **Apache License, Version 2.0**. See [`LICENSE`](LICENSE) for full legal text.

If you use the Cohort Monitoring Package in academic or clinical research, please cite:

```bibtex
@software{cohort_monitoring_package_2026,
  author = {Karim El Mernissi},
  title = {Cohort Monitoring Package: Intuitive Clinical Epidemiology & Cohort Data Exploration},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/KarimELMERNISSI/Cohort-Monitoring-Package}}
}
```
