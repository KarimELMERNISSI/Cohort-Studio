# Cohort Monitoring Package

## Overview

The **Cohort Monitoring Package** is a professional-grade, Streamlit-based application designed for comprehensive cohort data management, monitoring, analysis, and enrichment. Tailored for medical/epidemiological research, it integrates advanced data science with an intuitive UI to streamline the data lifecycle—from ingestion to complex statistical analysis.

---

## 🚀 Key Features

### 🔐 Secure Authentication & Isolation

- **User Accounts**: Dedicated login/signup with hashed password storage.
- **Data Privacy**: Complete isolation of datasets, traces, and analysis results per user.
- **Admin Oversight**: Centralized management capability.

### 📂 Data Management & Preparation

- **Dataset Versioning**: Built-in version control using DuckDB
- **Smart Column Renaming**: AI-powered renaming with RAG (UMLS/SNOMED/LOINC suggestions)
- **Cohort Filtering**: Complex inclusion/exclusion criteria

### 🧠 Knowledge & Document Intelligence

- **Taxonomy Generation**: Infer variable relationships and generate knowledge graphs
- **Document Chat**: Chat with PDFs and visualize concepts as interactive graphs
- **Coverage Analysis**: Check dataset variable coverage against research documents

### 🧪 Data Enrichment

- **Advanced Imputation**: MissForest and statistical methods
- **Dimensionality Reduction**: PCA, t-SNE, UMAP
- **Feature Engineering**: Scaling, encoding, custom transformations

### 🔍 Data Quality & Monitoring

- **Automated Quality Checks**: Data integrity, completeness, consistency
- **Anomaly Detection**: Statistical outliers and clinical anomalies
- **Longitudinal Monitoring**: Track distribution shifts over time

### 📊 Advanced Visualization

Interactive Plotly visualizations with statistical testing:

- Histograms, Box Plots, Violin Plots
- Correlation Matrices, Clustermaps
- Bland-Altman, ROC Curves

### 📈 Epidemiology & Statistics

- Automated Parametric/Non-Parametric test selection
- Multiple Testing Corrections (Bonferroni, FDR)
- Multivariate Analysis (ANCOVA)

---

## 📁 Project Structure

```text
Cohort Monitoring Package/
├── main.py                     # Application entry point
│
├── app_pages/                  # Streamlit page modules
│   ├── home.py                 # Dashboard & Data Loading
│   ├── data_enrichment.py      # Imputation & Transformation
│   ├── data_insight.py         # Knowledge Graph & Taxonomy
│   ├── document_insight.py     # Document Chat & Graph
│   ├── epidemiology.py         # Statistical Analysis
│   ├── data_monitoring.py      # Validation & Monitoring
│   └── visualization.py        # Plotting Suite
│
├── manage/                     # Backend Managers
│   ├── rag_manager.py          # RAG Orchestrator (Mixin Pattern)
│   ├── rag_taxonomy.py         # Taxonomy Generation Mixin
│   ├── rag_documents.py        # Document Processing Mixin
│   ├── rag_computed_vars.py    # Variable Suggestions Mixin
│   ├── rag_schemas.py          # Pydantic Response Schemas
│   └── db_manager.py           # DuckDB Database Manager
│
├── utils/                      # Utility Modules
│   ├── llm_utils.py            # Robust LLM Parsing & Validation
│   ├── data_analyzer.py        # DataFrame Column Analysis
│   ├── export_utils.py         # Excel Export Functions
│   ├── statistics_utils.py     # Normality Tests & Guidelines
│   ├── visualization_utils.py  # Plot Generators
│   └── custom_gemini.py        # Custom LangChain Wrapper
│
├── prompts/                    # LLM Prompt Templates
├── enrich/                     # Data Enrichment Logic
├── explore/                    # Exploratory Analysis
├── monitor/                    # Monitoring Logic
├── data/                       # Data Storage (DuckDB, ChromaDB)
└── documentation/              # User Guides & Tutorials
```

---

## 🏗️ Architecture

### RAG Manager (Mixin Pattern)

The RAG system uses a **Mixin composition pattern** for maintainability:

```text
RAGManager (451 lines)
├── TaxonomyMixin      → rag_taxonomy.py
├── DocumentsMixin     → rag_documents.py
└── ComputedVarsMixin  → rag_computed_vars.py
```

### LLM Reliability

All LLM interactions use robust parsing with **Pydantic validation**:

- **Multi-strategy JSON parsing** (markdown extraction, bracket balancing)
- **Schema validation** with type-safe Pydantic models
- **Automatic retry** with LLM-based repair

---

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- Google API Key (for Gemini LLM)

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

Set environment variables or use the Configuration page:

| Variable | Description |
|----------|-------------|
| :--- | :--- |

---

## 📚 Documentation

See the `documentation/` folder for detailed guides:

| Guide | Description |
| :--- | :--- |
| `00_Researcher_Handbook.md` | **Start Here**. Philosophy, workflow, and phase-by-phase guide. |
| `01_Home.md` | Data loading, RAG setup, and dashboard statistics |
| `02_Data_Validation.md` | Validation rules, Anomalies, and Inclusion masks |
| `03_Data_Enrichment.md` | Imputation, Computed Variables, and Dimensionality Reduction |
| `04_Visualization.md` | Interactive plotting and distribution checks |
| `05_Epidemiology.md` | Statistical analysis, Power calc, and Z-Scores |
| `06_Data_Insight.md` | Variable relationships and Taxonomy Graph |
| `07_Document_Insight.md` | Chat with PDF and Literature Knowledge Graphs |
| `08_Reproduction.md` | Trace replay and session management |
| `09_Clustering.md` | Population clustering (PCA, t-SNE, UMAP, K-Means) |
| `10_Developer_Guide.md` | Architecture, RAG utilities, and Contribution guide |
| `Tutorials.md` | Step-by-step specific workflows |

---

## 🧰 Technologies

| Category | Technologies |
| :--- | :--- |
| **Frontend** | Streamlit, yFiles |
| **Data** | Pandas, NumPy, DuckDB |
| **Visualization** | Plotly |
| **Statistics/ML** | SciPy, Statsmodels, Scikit-learn, UMAP |
| **AI/RAG** | LangChain, ChromaDB, Google Gemini |
| **Validation** | Pydantic |

---

## 📄 License

[MIT License](LICENSE)
