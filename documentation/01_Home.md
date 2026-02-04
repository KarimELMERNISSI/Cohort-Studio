# 🏠 Home / Dashboard

## Overview

The **Home** page serves as the central command center for the Cohort Monitoring Package. It handles dataset ingestion, version management, and provides an initial high-level overview of the data. It also features an AI-powered assistant for standardizing column names.

## Key Features

### 1. 📊 Dataset Statistics (Tab 1)

* **Data Preview & Filtering**:
  * **Global Filters**: Checkboxes to "Exclude Rows with Anomalies" or "Restrict to Inclusion Criteria" (requires pre-defined criteria).
  * **Column Categorization**: Automatic classification of columns into Quantitative, Binary, Categorical, Date, etc.
* **Descriptive Statistics**:
  * **Quantitative Data**: Detailed stats (Mean, Median, Std Dev, Min/Max, Quantiles) with **Normality Tests** (Shapiro-Wilk, D'Agostino's K², Kolmogorov-Smirnov). Supports **grouping** by categorical variables and Excel export.
  * **Qualitative Data**: Counts, frequencies, and top modalities. Supports **grouping** and Excel export.
* **Correlation Analysis**:
  * **Network Graph (New)**: Interactive graph visualization powered by **yFiles**.
    * **Dynamic Styling**: Edges are colored by sign (**Blue** for positive, **Red** for negative) and styled by strength (dotted to thick solid lines).
    * **Focus Mode**: Select a "Focus Variable" to isolate and explore its specific connected component.
    * **Heatmap**: Nodes glow based on their cumulative correlation strength, highlighting key drivers.
    * **Legend**: Built-in legend to explain visual encodings.
  * **Interactive Matrix**: Traditional heatmap visualization with Excel export.
  * **Advanced Options**: Support for Pearson, Spearman, and Kendall methods. Includes **Hierarchical Clustering** (dendrograms).

### 2. 🏷️ Columns Renaming (Tab 2)

* **AI-Powered Suggestions (RAG)**:
  * **Medical Literature**: Suggests names based on context found in uploaded research papers.
  * **Standard Terminologies**: Maps columns to standard medical ontologies (UMLS, SNOMED CT, LOINC).
* **Manual Interface**:
  * **Smart Filtering**: Sort by name or filter by type (Numeric, Date, Binary, etc.).
  * **Contextual Info**: Displays top 5 values (for categorical) or basic stats (Median, Min, Max) for each column to aid in identification.

### 3. ⚡ Performance Benchmark (Tab 3)

* **Engine Comparison**: Benchmarks the performance of **Pandas** vs. **DuckDB** for standard operations (Basic Stats, Grouping, Correlation).
* **Optimization**: Automatically highlights the faster engine and allows users to set their preferred computation backend for the session.

### 4. 🔐 Authentication & Security

* **Secure Login**: Sidebar-based authentication using hashed passwords.
* **User Isolation**: All data (datasets, traces, taxonomies) is isolated per user.
* **Admin Access**: Dedicated admin role for system oversight.

### 5. 💾 Dataset Management (Sidebar)

* **Versioning**: Save the current state of the dataset as a new version.
* **History**: Load previous versions of the dataset (User-Isolated).
* **Auto-Loading**: Automatically loads the most recent dataset session upon startup.

## Usage Guide

1. **Load/Import**: Use the sidebar to load a saved dataset or upload a new file.
2. **Clean & Rename**: Go to the **Columns Renaming** tab to standardize variable names using AI or manual input.
3. **Inspect Statistics**: Use the **Dataset Statistics** tab to filter anomalies, check normality, and explore correlations.
4. **Optimize**: Run the **Performance Benchmark** to ensure the application uses the fastest engine for your specific dataset size.

## Technical Details

* **File**: `app_pages/home.py`
* **Utilities**:
  * `utils/data_analyzer.py` - Column type detection
  * `utils/export_utils.py` - Excel export
  * `utils/statistics_utils.py` - Normality tests
* **Dependencies**: `manage.db_manager`, `manage.rag_manager`, `explore.corr_matrix`, `scipy.stats`, `duckdb`
