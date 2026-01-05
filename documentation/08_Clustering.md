# 🔬 Population Clustering

## Overview

The **Population Clustering** tab in Data Insight enables exploratory analysis to identify natural subgroups (phenotypes) in your cohort based on multiple variables.

---

## Features

### Dimensionality Reduction

| Method | Preserves | Speed | Best For |
|:-------|:----------|:------|:---------|
| **PCA** | Global variance | ⚡ Fast | Initial exploration, interpretability |
| **t-SNE** | Local structure | 🐢 Slow | Cluster visualization (< 3000 samples) |
| **UMAP** | Local + global | 🚀 Medium | Large datasets, preserves topology |

### Clustering Algorithms

| Method | Best For | Key Parameter |
|:-------|:---------|:--------------|
| **K-Means** | Well-separated spherical clusters | K (number of clusters) |
| **DBSCAN** | Arbitrary shapes, outlier detection | eps (neighborhood size) |
| **Gaussian Mixture** | Overlapping clusters, soft assignments | Number of components |

---

## Usage Guide

1. **Load data** from Main View
2. **Select variables** (numeric only, ≥ 2)
3. **Choose dimensionality reduction** (PCA for speed, UMAP for accuracy)
4. **Choose clustering** (K-Means for simple, DBSCAN for outliers)
5. **Run analysis**
6. **Explore results**:
   - Scatter plot (2D projection)
   - Cluster profiles (mean values per cluster)
   - Optimal K analysis (elbow + silhouette)

---

## Interpretation Guide

### Reading the Scatter Plot

- Each **point** = one sample
- **Nearby points** = similar profiles
- **Colors** = cluster assignments
- **Well-separated clusters** = distinct subpopulations

### Reading Cluster Profiles

- Each **row** = one cluster
- **Values** = mean of variables in that cluster
- Compare to characterize phenotypes

### Choosing K (Number of Clusters)

| Silhouette Score | Interpretation |
|:-----------------|:---------------|
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

## Technical Details

- **File**: `utils/clustering_utils.py`
- **Dependencies**: `scikit-learn`, `umap-learn` (optional)

## References

- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations.
- Ester, M., et al. (1996). A density-based algorithm for discovering clusters (DBSCAN).
- McInnes, L., et al. (2018). UMAP: Uniform Manifold Approximation and Projection.
