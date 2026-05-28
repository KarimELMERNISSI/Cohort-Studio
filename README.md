# Cohort Monitoring Package

## Overview

The **Cohort Monitoring Package** is a professional-grade, Streamlit-based application designed for comprehensive cohort data management, monitoring, analysis, and enrichment. Tailored for medical/epidemiological research, it integrates advanced data science with an intuitive UI to streamline the data lifecycle—from ingestion to complex statistical analysis and reproducible reporting.

---

## 🚀 Key Features

### 🔐 Secure Authentication & Isolation

- **User Accounts**: Dedicated login/signup with bcrypt-hashed password storage.
- **Account Activation**: New accounts require admin approval before access is granted.
- **Data Privacy**: Complete isolation of datasets, traces, and analysis results per user.
- **Admin Panel**: Centralized user management — activate/deactivate accounts, reset passwords, delete users.

### 📂 Data Management & Preparation

- **Dataset Versioning**: Built-in version control using DuckDB and Parquet snapshots.
- **Smart Column Renaming**: AI-powered semantic renaming with RAG (UMLS/SNOMED/LOINC suggestions), plus manual mapping.
- **Cohort Filtering**: Complex inclusion/exclusion criteria via configurable mask families.
- **Performance Benchmark**: Auto-comparison between Pandas and DuckDB query performance.

### 🧠 Knowledge & Document Intelligence

- **Taxonomy Generation**: Infer variable relationships and generate interactive knowledge graphs (yFiles).
- **Taxonomy Refinement**: Repair malformed graphs, enrich nodes with AI descriptions, add missing concept nodes.
- **Document Chat**: Retrieval-augmented Q&A over uploaded PDFs.
- **Document Summarizer**: Structured extraction of objectives, methods, findings, and variables from research papers.
- **Coverage Analysis**: Check dataset variable coverage against research documents.

### 🧪 Data Enrichment

- **Advanced Imputation**: Targeted (formula-based, AI-assisted) and Global (MICE, MissForest, KNN, Mean/Median).
- **Dimensionality Reduction**: PCA, FAMD, t-SNE, UMAP (via Prince and scikit-learn).
- **Feature Engineering**: One-Hot / Label / Ordinal encoding, aggregation, scaling (Standard, MinMax, Robust, MaxAbs).
- **Population Clustering**: K-Means, DBSCAN, Gaussian Mixture with automatic standardisation.
- **AI Variable Discovery**: RAG-powered suggestions for computed variables from existing columns.
- **External Data Import**: Upload and join enrichment datasets with configurable merge logic.

### 🔍 Data Quality & Monitoring

- **Automated Quality Checks**: Global data quality score with completeness, validity, and consistency scorecards.
- **Clinical Anomaly Detection**: Rule-based flagging of clinically impossible or suspect values.
- **Outlier Detection**: Z-score, IQR, Quantile, Local Outlier Factor, Isolation Forest, DBSCAN.
- **Handling Strategies**: None, Remove, Clip, Tag — with impact analysis (flat / % of initial value / % of IQR).
- **Outlier Visualisation**: 2-D projection (PCA/FAMD/t-SNE/UMAP) with colour-coded outlier overlay.
- **Dataset Comparison**: Compare different versions side by side.

### 📊 Advanced Visualisation

Interactive Plotly visualisations with integrated statistical testing:

- Histograms, Box Plots, Violin Plots, Scatter Plots
- Correlation Matrices, Clustermaps
- Bland-Altman Plots, ROC Curves, Kaplan-Meier Curves
- Integrated significance overlays (Mann-Whitney, Kruskal-Wallis, etc.)
- Export to HTML, PNG, SVG

### 📈 Epidemiology & Statistics

- Automated parametric/non-parametric test selection with recommendation helper.
- Effect size interpretation (Cohen's d, η², etc.).
- Multiple Testing Corrections (Bonferroni, FDR).
- Multivariate Analysis (ANCOVA).
- Power & Sample Size Calculations.
- Z-Score Standardisation with reference population support.

### 🔁 Reproducibility

- **Trace Recording**: Automatic logging of every analysis step with parameters and dataset snapshots.
- **Replay Engine**: Fast (parameters only) and Full (data included) replay modes with cross-platform path resolution.
- **Automated Reports**: Generate `.docx` transformation reports from trace files.

### 📊 RAG Quality Monitor

- Dashboard for monitoring RAG system health: embedding statistics, retrieval quality metrics, and evaluation results.

---

## 📁 Project Structure

```text
Cohort Monitoring Package/
├── main.py                     # Application entry point
│
├── app_pages/                  # Streamlit page modules
│   ├── home.py                 # Dashboard & Data Loading
│   ├── data_monitoring.py      # Validation & Monitoring
│   ├── data_enrichment.py      # Imputation, Transformation & Clustering
│   ├── data_insight.py         # Knowledge Graph & Taxonomy
│   ├── document_insight.py     # Document Chat, Summariser & Graph
│   ├── epidemiology.py         # Statistical Analysis
│   ├── visualization.py        # Plotting Suite
│   ├── reproduce_analysis.py   # Trace Replay & Reporting
│   ├── rag_monitoring.py       # RAG Quality Dashboard
│   ├── users_management.py     # Admin User Management
│   └── about.py                # Application Overview
│
├── manage/                     # Backend Managers
│   ├── rag_manager.py          # RAG Orchestrator (Mixin Pattern)
│   ├── rag_taxonomy.py         # Taxonomy Generation Mixin
│   ├── rag_documents.py        # Document Processing Mixin
│   ├── rag_computed_vars.py    # Variable Suggestions Mixin
│   ├── rag_schemas.py          # Pydantic Response Schemas
│   ├── db_manager.py           # DuckDB Database Manager
│   └── reproduction_manager.py # Trace Replay Logic
│
├── utils/                      # Utility Modules
│   ├── llm_utils.py            # Robust LLM Parsing & Validation
│   ├── data_analyzer.py        # DataFrame Column Analysis
│   ├── clustering_utils.py     # Clustering Helpers (K-Means, DBSCAN, GMM)
│   ├── export_utils.py         # Excel Export Functions
│   ├── statistics_utils.py     # Normality Tests & Guidelines
│   ├── visualization_utils.py  # Plot Generators
│   └── custom_gemini.py        # Custom LangChain Wrapper
│
├── config/                     # Configuration Files
│   └── config.json             # Data quality rules & validation thresholds
├── prompts/                    # LLM Prompt Templates
├── enrich/                     # Data Enrichment Logic
├── explore/                    # Exploratory Analysis & Data Quality
├── monitor/                    # Monitoring Logic
├── data/                       # Data Storage (DuckDB, ChromaDB, Traces)
└── documentation/              # User Guides & Tutorials
```

---

## 🏗️ Architecture

### RAG Manager (Mixin Pattern)

The RAG system uses a **Mixin composition pattern** for maintainability:

```text
RAGManager
├── TaxonomyMixin      → rag_taxonomy.py
├── DocumentsMixin     → rag_documents.py
└── ComputedVarsMixin  → rag_computed_vars.py
```

### LLM Reliability

All LLM interactions use robust parsing with **Pydantic validation**:

- **Multi-strategy JSON parsing** (markdown extraction, bracket balancing)
- **Schema validation** with type-safe Pydantic models
- **Automatic retry** with LLM-based repair

### Data Isolation

Per-user isolation enforced via filename prefixes and directory scoping across datasets, traces, ChromaDB collections, and DuckDB tables.

---

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- Google API Key (for Gemini LLM features)

### Steps

```bash
# Clone repository
git clone <repository-url>
cd Cohort-Monitoring-Package

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run main.py
```

### Docker

```bash
docker build -t cohort-monitoring .
docker run -p 8501:8501 -e GOOGLE_API_KEY=your_key cohort-monitoring
```

---

## ⚙️ Configuration

| Variable | Description |
| :--- | :--- |
| `GOOGLE_API_KEY` | Required. Google Gemini API key for all AI/RAG features. Set as environment variable or in `.streamlit/secrets.toml`. |
| `config/config.json` | Data quality rules, validation thresholds, anomaly definitions, and inclusion criteria masks. Editable via the in-app Configuration page. |

---

## 📚 Documentation

See the `documentation/` folder for detailed guides:

| Guide | Description |
| :--- | :--- |
| `00_Researcher_Handbook.md` | **Start Here**. Philosophy, workflow, and phase-by-phase guide. |
| `01_Home.md` | Data loading, column renaming, dashboard statistics, performance benchmark. |
| `02_Data_Validation.md` | Clinical anomalies, inclusion criteria, outlier detection & handling, dataset comparison. |
| `03_Data_Enrichment.md` | Imputation (Targeted & Global), dimensionality reduction, feature engineering, clustering, AI variable discovery. |
| `04_Visualization.md` | Interactive plotting suite with integrated statistical testing. |
| `05_Epidemiology.md` | Hypothesis testing, ANCOVA, power analysis, Z-score standardisation. |
| `06_Data_Insight.md` | Taxonomy (knowledge graph) generation, refinement, versioning, and AI enrichment. |
| `07_Document_Insight.md` | Document summariser, chat with PDFs, literature knowledge graphs, coverage analysis. |
| `08_Reproduction.md` | Trace replay, session management, and automated reporting. |
| `09_Clustering.md` | Population clustering algorithms and interpretation guide. |
| `10_Developer_Guide.md` | Architecture, RAG system, authentication, utilities, and contribution guide. |
| `Tutorials.md` | Step-by-step workflows for common tasks. |

---

## 🧰 Technologies

| Category | Technologies |
| :--- | :--- |
| **Frontend** | Streamlit, yFiles (graph visualisation) |
| **Data** | Pandas, NumPy, DuckDB |
| **Visualisation** | Plotly |
| **Statistics/ML** | SciPy, Statsmodels, Scikit-learn, UMAP, Prince (PCA/FAMD) |
| **AI/RAG** | LangChain, ChromaDB, Google Gemini |
| **Validation** | Pydantic |
| **Reporting** | python-docx |
| **Security** | bcrypt |

---

## 📄 License

[Apache 2.0](LICENSE)
