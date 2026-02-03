# 🧪 Data Enrichment

## Overview

The **Data Enrichment** module provides advanced tools for enhancing the dataset through imputation, dimensionality reduction, and feature engineering. It is designed to handle complex data challenges like missing values and high-dimensional spaces.

## Key Features

### 1. 🧹 Handle Missing Data (Tab 1)

Organized into two focused sub-tabs:

#### A. Targeted Imputation (Custom Formulas)

* **Precision**: Manually fix specific variables using known relationships (e.g., `Weight = BMI * Height^2`).
* **AI Guided Mode**: The RAG assistant can suggest valid Python formulas based on your columns and docs.

#### B. Global Imputation (Auto)

* **Numerical Strategies**:
  * **Standard**: Mean, Median.
  * **Advanced**: **KNN**, **MICE** (Iterative), **MissForest** (Random Forest).
* **Categorical Strategies**: Mode or MissForest classification.
* **Auto-Encoding**: Pipeline handles encoding/scaling automatically.

### 2. 🌌 Dimensionality Reduction

* **Mixed Data Support**:
  * **FAMD (Factor Analysis of Mixed Data)**: Specifically designed to handle datasets with both numerical and categorical variables simultaneously.
* **Linear Methods**:
  * **PCA (Principal Component Analysis)**: Uses the `prince` library (sklearn engine) with automatic centering and scaling.
* **Non-Linear / Manifold Learning**:
  * **t-SNE**: t-Distributed Stochastic Neighbor Embedding for cluster visualization.
  * **UMAP**: Uniform Manifold Approximation and Projection for scalable structure preservation.
* **Analytics**: Includes "Explained Variance" plots and "Contribution to Components" heatmaps for PCA/FAMD.

### 3. 🛠️ Feature Engineering

* **Encoding Strategies**:
  * **One-Hot Encoding**: For nominal variables.
  * **Label Encoding**: For simple integer mapping.
  * **Ordinal Encoding**: Respects specific category orders (e.g., Low < Medium < High).
* **Statistical Aggregations**:
  * **Central Tendency**: Mean, Median.
  * **Dispersion**: Standard Deviation, Min, Max.
  * **Totals**: Summation.
* **Scaling & Normalization**:
  * **Z-Score Standardization**: Centers data around 0 with unit variance.
  * **Min-Max Normalization**: Scales data to a fixed range [0, 1].
  * **Log Transformation**: Natural Logarithm (Log(x+1)) for handling skewed data.

### 4. 🤖 AI Variable Discovery (RAG)

* **Context-Aware Suggestions**: Uses Retrieval-Augmented Generation (RAG) to suggest new computed variables based on your uploaded documentation (e.g., clinical protocols, research papers).
* **Dual Modes**:
  * **Specific Search**: Ask for specific concepts (e.g., "Kidney Function", "BMI") and the AI will find relevant formulas in your documents.
  * **Open Discovery**: Let the AI analyze your dataset columns against your documents to suggest relevant derived variables automatically.
* **Source Transparency**: Each suggestion is tagged with its source type (**Document**, **Hybrid**, or **Knowledge Base**) and includes the specific formula and reasoning derived from the text.
* **One-Click Application**: Instantly apply the suggested formulas to the variable creation editor.

### 5. 📄 Data Transformation Reporting

* **Automated Documentation**: Generates a professional Microsoft Word (`.docx`) report summarizing the entire data transformation session.
* **Audit Trail**: Includes:
  * Session ID and Source Dataset name.
  * Chronological list of all applied steps (Enrichment, Imputation, etc.).
  * Detailed parameter tables for each operation.
  * Timestamps for all actions.
* **Access Points**: Available in the **Dataset History** sidebar (next to "Load Selected Version") and in the **Reproduce Analysis** tab.

## Usage Guide

1. **Impute Missing Data**: Select columns with missing values and choose an imputation method (MissForest is recommended for complex datasets).
2. **Reduce Dimensions**: Select a subset of numeric features and run PCA, t-SNE, or UMAP to visualize the data in 2D or 3D space. This is useful for identifying clusters or patterns.
3. **Transform Features**: Create new variables or scale existing ones to prepare the data for statistical modeling.
4. **Discover Variables**: Expand "🤖 AI Variable Suggestions", upload your protocol documents (in the sidebar), and click "Generate Suggestions" to find clinically relevant computed variables.
5. **Download Report**: In the **Dataset History** sidebar, select a version and click **📄 Download Report** to get a documented history of your changes.

## Technical Details

* **File**: `app_pages/data_enrichment.py`
* **Dependencies**: `utils.miss_forest`, `prince` (FAMD/PCA), `sklearn` (Preprocessing, Decomposition, Manifold), `umap`, `manage.rag_manager`
