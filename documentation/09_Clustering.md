# 🔬 Population Clustering

Population clustering helps identify natural subgroups within your cohort based on shared characteristics. This guide covers the clustering algorithms available in the **Data Enrichment** page (under **Create New Variables → Clustering**) and how to interpret the results.

---

## 🔄 Preprocessing

Before clustering, the pipeline applies two automatic preprocessing steps:

1. **Standardisation** — All numerical variables are Z-score standardised to ensure equal weight in distance calculations.
2. **Complete Case Analysis** — Rows with any missing values in the selected columns are excluded. Impute missing data first if needed (see `03_Data_Enrichment.md`).

---

## 📐 Dimensionality Reduction (Optional Pre-step)

For high-dimensional data, reducing dimensions before clustering can improve results and enable visualisation.

| Method | Best For | Key Parameters |
| --- | --- | --- |
| **PCA** | Numerical data; preserving variance. | `n_components` |
| **FAMD** | Mixed data (numerical + categorical). | `n_components` |
| **t-SNE** | 2-D/3-D visualisation of clusters. | `perplexity`, `learning_rate` |
| **UMAP** | 2-D/3-D visualisation; preserves global structure better than t-SNE. | `n_neighbours`, `min_dist` |

> **Tip**: Apply dimensionality reduction as a separate step first, then cluster on the resulting components.

---

## 🧩 Clustering Algorithms

### K-Means

| Property | Value |
| --- | --- |
| **Type** | Partition-based |
| **Parameter** | Number of clusters (K) |
| **Best For** | Well-separated, spherical clusters of similar size. |
| **Output** | Cluster label (0 to K−1) per row. |

**How to choose K**: Use the elbow method or silhouette analysis. Start with K = 2–5 and inspect the results visually.

### DBSCAN

| Property | Value |
| --- | --- |
| **Type** | Density-based |
| **Parameters** | `eps` (neighbourhood radius), `min_samples` (minimum points per cluster) |
| **Best For** | Irregularly shaped clusters; detecting noise/outliers. |
| **Output** | Cluster label per row; noise points labelled as −1. |

**How to tune**: Smaller `eps` creates more, tighter clusters. Increase `min_samples` to require denser regions.

### Gaussian Mixture Model (GMM)

| Property | Value |
| --- | --- |
| **Type** | Probabilistic |
| **Parameter** | Number of components |
| **Best For** | Overlapping clusters; soft (probabilistic) assignments. |
| **Output** | Most likely cluster label per row. |

**When to use GMM over K-Means**: When clusters overlap or have different shapes/sizes.

---

## 📊 Interpreting Results

After clustering, a new column is appended to the dataset with the cluster label. To understand what each cluster represents:

1. **Compare cluster means** — For each variable, compute the mean per cluster and identify distinguishing features.
2. **Visualise in 2-D** — Use t-SNE or UMAP to project the data and colour by cluster label.
3. **Cross-tabulate** — Check cluster composition against known clinical categories (e.g. disease stage, treatment group).
4. **Statistical tests** — Run ANOVA or Kruskal-Wallis across clusters to find significantly different variables.

---

## ⚠️ Common Pitfalls

| Pitfall | Recommendation |
| --- | --- |
| **Missing values** | Impute before clustering — the algorithm will exclude incomplete rows. |
| **Unstandardised data** | The pipeline standardises automatically, but verify that extreme outliers are handled first. |
| **Too many variables** | Consider dimensionality reduction (PCA/FAMD) to reduce noise. |
| **Choosing K arbitrarily** | Use quantitative metrics (silhouette, inertia) alongside domain knowledge. |

---

## 🔗 Related Documentation

- [03_Data_Enrichment.md](03_Data_Enrichment.md) — Where clustering is configured and applied.
- [02_Data_Validation.md](02_Data_Validation.md) — Outlier detection before clustering.
- [04_Visualization.md](04_Visualization.md) — Plotting cluster distributions and comparisons.
