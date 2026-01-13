# 🔬 Population Clustering

## Overview

The **Population Clustering** tab in Data Insight enables exploratory analysis to identify natural subgroups (phenotypes) in your cohort based on multiple variables.

---

## Features

### Dimensionality Reduction

| Method | Preserves | Speed | Best For |
| :--- | :--- | :--- | :--- |
| **PCA** | Global variance | ⚡ Fast | Initial exploration, interpretability |
| **FAMD** | Mixed (Num+Cat) variance | 🚀 Medium | Mixed data (Numeric + Categorical) |
| **t-SNE** | Local structure | 🐢 Slow | Cluster visualization (< 3000 samples) |
| **UMAP** | Local + global | 🚀 Medium | Large datasets, preserves topology |

### Clustering Algorithms

| Method | Best For | Key Parameter |
| :--- | :--- | :--- |
| **K-Means** | Well-separated spherical clusters | K (number of clusters) |
| **DBSCAN** | Arbitrary shapes, outlier detection | eps (neighborhood size) |
| **Gaussian Mixture** | Overlapping clusters, soft assignments | Number of components |

---

## Usage Guide

1. **Load data** from Main View
1. **Load data** from Main View
1. **Select variables** (Numeric or Mixed for FAMD)
1. **Choose dimensionality reduction** (PCA for numeric, FAMD for mixed)
1. **Choose clustering** (K-Means for simple, DBSCAN for outliers)
1. **Run analysis**
1. **Explore results**:
   - Scatter plot (2D projection)
   - Cluster profiles (mean values per cluster)
   - Optimal K analysis (elbow + silhouette)

---

## Interpretation Guide

### Reading the Scatter Plot

- Each **point** = one sample
- **Nearby points** = similar profiles
- **Colors** = cluster assignments
- **Axis Labels** = % of explained variance (for PCA/FAMD), helping you judge dimension importance
- **Well-separated clusters** = distinct subpopulations

### Reading Cluster Profiles

- Each **row** = one cluster
- Each **row** = one cluster
- **Values**:
  - **Numeric**: Mean value
  - **Categorical**: Mode (most frequent value)
- Compare to characterize phenotypes

### Choosing K (Number of Clusters)

| Silhouette Score | Interpretation |
| :--- | :--- |
| > 0.5 | Good clustering |
| 0.25 - 0.5 | Acceptable |
| < 0.25 | Poor separation |

---

## Caveats

- **2D projection** may distort distances
- **t-SNE/UMAP**: distances between clusters are NOT meaningful
- Always **validate with clinical knowledge**
- ~5% false positives by chance alone

---

---

## Generating Cluster Variables

You can also create a permanent variable for your clusters to use in other analysis tabs:

1. Go to **Data Enrichment** > **Create New Variables**
2. Select **Transformation Type**: `Cluster-Based`
3. Choose your numeric variables
4. Select the clustering algorithm (K-Means, DBSCAN, GMM)
5. Click **Apply Transformation**

A new column (e.g., `Cluster_K-Means`) will be added to your dataset containing the cluster labels.

## Methodological Notes

> [!IMPORTANT]
> **Standardization**: All clustering algorithms implemented here rely on distance metrics (Euclidean). Input variables are **automatically standardized** (Z-score normalization: mean=0, std=1) before analysis to ensure that variables with large ranges (e.g., Platelets) do not dominate those with small ranges (e.g., Creatinine).
> [!WARNING]
> **Missing Values**: Clustering algorithms cannot handle missing data. Rows with **any missing value** in the selected variables will be **excluded** from the analysis (Complete Case Analysis). Please impute missing data beforehand if significant data loss is a concern.

---

## Technical Details

- **File**: `utils/clustering_utils.py`
- **Dependencies**: `scikit-learn`, `umap-learn` (optional), `prince` (for FAMD)

## References

- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations.
- Ester, M., et al. (1996). A density-based algorithm for discovering clusters (DBSCAN).
- McInnes, L., et al. (2018). UMAP: Uniform Manifold Approximation and Projection.
