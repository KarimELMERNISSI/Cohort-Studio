# Cohort Monitoring Package

## Overview

The **Cohort Monitoring Package** is a professional-grade, Streamlit-based application designed for the comprehensive management, monitoring, analysis, and enrichment of cohort data. Tailored for medical and epidemiological research, this tool integrates advanced data science techniques with an intuitive user interface to streamline the entire data lifecycle—from ingestion and cleaning to complex statistical analysis and visualization.

## Key Features

### 1. 📂 Data Management & Preparation
*   **Dataset Versioning**: Built-in version control using DuckDB to track changes and manage multiple dataset iterations.
*   **Smart Column Renaming**: AI-powered column renaming assistant using RAG (Retrieval-Augmented Generation) to suggest standard medical terminologies (UMLS/SNOMED/LOINC) or context-aware names from literature.
*   **Cohort Filtering**: Define and apply complex inclusion/exclusion criteria and anomaly filters.

### 2. 🔍 Data Quality & Monitoring
*   **Automated Quality Checks**: Instantly assess data integrity, completeness, and consistency.
*   **Anomaly Detection**: Identify statistical outliers and clinical anomalies using configurable rules.
*   **Longitudinal Monitoring**: Track shifts in data distribution and quality metrics over time.

### 3. 🧪 Data Enrichment & Transformation
*   **Advanced Imputation**: Handle missing data using state-of-the-art algorithms like **MissForest**, alongside standard statistical methods.
*   **Dimensionality Reduction**: Visualize high-dimensional data using **PCA**, **t-SNE**, and **UMAP**.
*   **Feature Engineering**: Apply scaling, encoding, and custom mathematical transformations to generate new insights.

### 4. 📊 Advanced Visualization
A robust suite of interactive visualizations powered by **Plotly**, featuring:
*   **Distribution Analysis**: Histograms, Box Plots, and Violin Plots with **integrated statistical significance testing**.
*   **Relationship Exploration**: Scatter Plots, Correlation Matrices, and Hierarchical Clustermaps.
*   **Medical Statistics**: Bland-Altman Plots for method comparison and ROC Curves for diagnostic performance.
*   **Time Series**: Line plots with confidence intervals and aggregation options.

### 5. 📈 Epidemiology & Statistics
*   **Hypothesis Testing**: Automated selection of Parametric (T-test, ANOVA) or Non-Parametric (Mann-Whitney, Kruskal-Wallis) tests.
*   **Correction Methods**: Built-in support for Multiple Testing Corrections (Bonferroni, Benjamini-Hochberg FDR).
*   **Multivariate Analysis**: Perform ANCOVA to adjust for covariates.
*   **Effect Sizes**: Automatic calculation of Cohen's d, Rank-Biserial correlation, etc.

### 6. 🤖 RAG & AI Integration
*   **Document Intelligence**: Chat with your project documentation (PDFs, etc.) to retrieve context-aware answers.
*   **AI Assistance**: Leverage LLMs for data cleaning suggestions and terminology standardization.

## Project Structure

```
Cohort Monitoring Package/
├── main.py                 # Application entry point
├── app_pages/              # Streamlit modules for each functional area
│   ├── home.py             # Dashboard & Data Loading
│   ├── visualization.py    # Advanced Plotting Suite
│   ├── epidemiology.py     # Statistical Analysis Module
│   ├── data_enrichment.py  # Imputation & Transformation
│   ├── data_quality.py     # Quality Assessment
│   └── ...
├── data/                   # Data storage (Inputs, Outputs, ChromaDB)
├── enrich/                 # Core logic for data enrichment
├── explore/                # Exploratory data analysis scripts
├── manage/                 # Backend managers (DB, RAG, Files)
├── miss_forest/            # Custom MissForest implementation
├── monitor/                # Data monitoring logic
├── utils/                  # Helper functions & UI components
└── config/                 # Configuration files
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Cohort-Monitoring-Package
    ```

2.  **Install dependencies:**
    Ensure you have Python 3.8+ installed.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If `requirements.txt` is missing, install core packages: `streamlit pandas numpy scipy plotly scikit-learn fuzzywuzzy umap-learn lightgbm duckdb chromadb langchain`)*

## Usage

Run the application using Streamlit:

```bash
streamlit run main.py
```

The application will open in your default web browser. Use the sidebar to navigate between different modules (Home, Visualization, Epidemiology, etc.).

## Technologies

*   **Frontend**: Streamlit
*   **Data Processing**: Pandas, NumPy, DuckDB
*   **Visualization**: Plotly Express, Plotly Graph Objects
*   **Statistics/ML**: SciPy, Statsmodels, Scikit-learn, MissForest, UMAP
*   **AI/RAG**: LangChain, ChromaDB

## License

[Insert License Information Here]
