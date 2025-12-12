# 🧪 Data Enrichment

## Overview
The **Data Enrichment** module provides advanced tools for enhancing the dataset through imputation, dimensionality reduction, and feature engineering. It is designed to handle complex data challenges like missing values and high-dimensional spaces.

## Key Features

### 1. 🧩 Advanced Imputation
*   **Numerical Methods**:
    *   **Standard**: Mean, Median.
    *   **Advanced**:
        *   **KNN**: K-Nearest Neighbors imputation.
        *   **MICE**: Multivariate Imputation by Chained Equations (iterative regression).
        *   **MissForest**: Random Forest-based imputation (robust to non-linearities).
*   **Categorical Methods**:
    *   **Most Frequent**: Mode imputation.
    *   **MissForest**: Random Forest classification for categorical missingness.
*   **Pipeline Integration**:
    *   **Auto-Encoding/Scaling**: Optional One-Hot Encoding and Scaling during the imputation process.
    *   **Remainder Handling**: Smart detection of columns to "Passthrough" or "Drop" based on missingness thresholds.

### 2. 🌌 Dimensionality Reduction
*   **Mixed Data Support**:
    *   **FAMD (Factor Analysis of Mixed Data)**: Specifically designed to handle datasets with both numerical and categorical variables simultaneously.
*   **Linear Methods**:
    *   **PCA (Principal Component Analysis)**: Uses the `prince` library (sklearn engine) with automatic centering and scaling.
*   **Non-Linear / Manifold Learning**:
    *   **t-SNE**: t-Distributed Stochastic Neighbor Embedding for cluster visualization.
    *   **UMAP**: Uniform Manifold Approximation and Projection for scalable structure preservation.
*   **Analytics**: Includes "Explained Variance" plots and "Contribution to Components" heatmaps for PCA/FAMD.

### 3. 🛠️ Feature Engineering
*   **Encoding Strategies**:
    *   **One-Hot Encoding**: For nominal variables.
    *   **Label Encoding**: For simple integer mapping.
    *   **Ordinal Encoding**: Respects specific category orders (e.g., Low < Medium < High).
*   **Statistical Aggregations**:
    *   **Central Tendency**: Mean, Median.
    *   **Dispersion**: Standard Deviation, Min, Max.
    *   **Totals**: Summation.
*   **Scaling & Normalization**:
    *   **Z-Score Standardization**: Centers data around 0 with unit variance.
    *   **Min-Max Normalization**: Scales data to a fixed range [0, 1].
    *   **Log Transformation**: Natural Logarithm (Log(x+1)) for handling skewed data.

### 4. 🤖 AI Variable Discovery (RAG)
*   **Context-Aware Suggestions**: Uses Retrieval-Augmented Generation (RAG) to suggest new computed variables based on your uploaded documentation (e.g., clinical protocols, research papers).
*   **Dual Modes**:
    *   **Specific Search**: Ask for specific concepts (e.g., "Kidney Function", "BMI") and the AI will find relevant formulas in your documents.
    *   **Open Discovery**: Let the AI analyze your dataset columns against your documents to suggest relevant derived variables automatically.
*   **Source Transparency**: Each suggestion is tagged with its source type (**Document**, **Hybrid**, or **Knowledge Base**) and includes the specific formula and reasoning derived from the text.
*   **One-Click Application**: Instantly apply the suggested formulas to the variable creation editor.

## Usage Guide
1.  **Impute Missing Data**: Select columns with missing values and choose an imputation method (MissForest is recommended for complex datasets).
2.  **Reduce Dimensions**: Select a subset of numeric features and run PCA, t-SNE, or UMAP to visualize the data in 2D or 3D space. This is useful for identifying clusters or patterns.
3.  **Transform Features**: Create new variables or scale existing ones to prepare the data for statistical modeling.
4.  **Discover Variables**: Expand "🤖 AI Variable Suggestions", upload your protocol documents (in the sidebar), and click "Generate Suggestions" to find clinically relevant computed variables.

## Technical Details
*   **File**: `app_pages/data_enrichment.py`
*   **Dependencies**: `miss_forest`, `prince` (FAMD/PCA), `sklearn` (Preprocessing, Decomposition, Manifold), `umap`, `manage.rag_manager`
