# Data Enrichment

The **Data Enrichment** page handles the complete data preparation pipeline — from importing external data sources and imputing missing values, to engineering new variables through transformations, dimensionality reduction, and clustering. All transformations are logged to the trace system for full reproducibility.

---

## Import External Data

Merge additional data sources into your main dataset.

### Workflow

1. **Upload Main Dataset** — Load from the current session, a file upload, or a saved version.
2. **Upload Enrichment Dataset** — Provide the supplementary data to merge.
3. **Select Columns** — Choose which columns to retain from each dataset.
4. **Configure Merge** — Select the join key(s) and merge type (inner, left, right, outer).
5. **Review Results** — Inspect the enriched dataset, including column overlap and row counts.

---

## Handle Missing Data

Two complementary approaches are available for imputation.

### Targeted Imputation

Manually fix specific columns using formulas or AI-assisted suggestions.

| Feature | Description |
| --- | --- |
| **Formula Editor** | Write expressions using column names and arithmetic operators (e.g. `Weight / (Height**2)`). |
| **AI Suggestions (RAG)** | The system proposes computed formulas based on related columns and uploaded documentation. |

#### AI Suggestion Modes

| Mode | Description |
| --- | --- |
| **Free Thinking (Auto)** | Scans all available columns to suggest formulas automatically. |
| **Guided (with Hints)** | User provides a clue (e.g. "unit conversion") and selects context columns for more focused suggestions. |

### Global Imputation

Apply statistical or ML-based strategies to all missing values simultaneously.

| Method | Type | Description |
| --- | --- | --- |
| **Mean / Median** | Statistical | Replace missing values with the column mean or median. |
| **Most Frequent** | Statistical | Replace with the mode (categorical columns). |
| **KNN** | ML | Impute using K-nearest neighbours. |
| **MICE** | ML | Multiple Imputation by Chained Equations — iterative, multivariate. |
| **MissForest** | ML | Random forest-based imputation for mixed data types. |

#### Global Imputation Options

| Option | Description |
| --- | --- |
| **Categorical Encoder** | Optionally one-hot encode categorical columns during imputation. |
| **Numerical Scaler** | Optionally scale numerical columns during imputation. |
| **Remainder Strategy** | Handle columns not part of the imputation (passthrough, drop). |

After imputation, an **imputation mask** highlights which values were filled, and a **before/after summary** is displayed.

---

## Create New Variables

Engineer new features via transformations, dimensionality reduction, encoding, scaling, or clustering.

### Transformation Types

#### Dimensionality Reduction

| Method | Description |
| --- | --- |
| **PCA** | Principal Component Analysis (via Prince). Handles numerical data. |
| **FAMD** | Factor Analysis of Mixed Data. Handles both numerical and categorical variables. |
| **t-SNE** | Non-linear embedding for 2-D/3-D visualisation. Parameters: perplexity, learning rate. |
| **UMAP** | Uniform Manifold Approximation. Parameters: n_neighbours, min_dist. |

#### Clustering

| Method | Description |
| --- | --- |
| **K-Means** | Partition-based clustering. Set number of clusters (K). |
| **DBSCAN** | Density-based clustering. Set epsilon and min_samples. |
| **Gaussian Mixture** | Probabilistic clustering. Set number of components. |

> **Note**: Input variables are automatically standardised (Z-score). Rows with any missing values in selected columns are excluded (complete case analysis).

#### Encoding

| Method | Description |
| --- | --- |
| **One-Hot** | Binary column per category. |
| **Label** | Integer encoding. |
| **Ordinal** | User-specified order mapping. |

#### Scaling

| Method | Description |
| --- | --- |
| **Standard (Z-Score)** | Mean = 0, Std = 1. |
| **Min-Max** | Scaled to [0, 1]. |
| **Robust** | Median-centred, IQR-scaled. Resistant to outliers. |
| **Max-Abs** | Scaled by maximum absolute value. |

#### Statistical Transformations

Log, Square Root, Box-Cox, Yeo-Johnson, Rank, Quantile, and more.

### Naming Convention

All new columns follow a configurable naming pattern (e.g. `{method_applied}_{initial_variable}`) to maintain traceability.

---

## AI Variable Discovery

The RAG system can suggest **computed variables** that are clinically or statistically meaningful based on:
- Existing column names and data types.
- Uploaded research documentation.
- Known medical formulas and indices.

Suggestions include the variable name, formula, reasoning, and expected units.

---

## Dataset Versioning & Persistence

The sidebar provides:

- **Dataset History** — Load any previously saved dataset version.
- **Automatic Snapshots** — Each transformation saves a Parquet snapshot for reproducibility.
- **Reproduce Analysis** — Upload or select a trace file to regenerate reports or replay transformations.

---

## Practical Guidance
- Run **Global Imputation** before **Dimensionality Reduction** or **Clustering** — these methods require complete data.
- Use **Targeted Imputation** with AI suggestions for domain-specific formulas (e.g. computing BMI from weight and height).
- Check the **imputation mask** after global imputation to verify which values were filled.
