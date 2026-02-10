# 📋 Data Validation & Monitoring

The **Data Validation & Monitoring** page provides a comprehensive data quality framework. It detects clinical anomalies, applies inclusion/exclusion criteria, identifies statistical and ML-based outliers, and compares dataset versions — all tracked through an integrated Data Quality Score.

---

## 🏥 Clinical Anomalies

The system checks your dataset for values that are clinically impossible or suspect based on rules defined in `config/config.json`.

### How It Works

1. **Rule Evaluation** — Each variable is checked against configured ranges (e.g. `Age < 0` or `BMI > 100`).
2. **Anomaly Flagging** — Rows with anomalies are flagged, with the violated rule displayed per variable.
3. **Impact Score** — Anomalies contribute to the overall Data Quality Score.

### Anomaly Types

| Type | Description |
| --- | --- |
| **Impossible** | Values that cannot exist clinically (e.g. negative age). |
| **Suspect** | Values that are plausible but warrant manual review. |

---

## ✅ Inclusion Criteria

Define and apply inclusion/exclusion criteria to filter your cohort. Criteria are organised into **mask families** — groups of related conditions combined with logical operators.

### Key Concepts

- **Mask Family**: A named group of conditions (e.g. "Adult Patients").
- **Mask**: An individual condition within a family (numeric threshold or expression).
- **Family Operator**: How masks within a family combine — `AND` (all must pass) or `OR` (any may pass).

### Mask Types

| Type | Example |
| --- | --- |
| **Numeric** | `Age >= 18 AND Age <= 90` |
| **Expression** | `Gender == 'Female'` |

### Workflow

1. Create a mask family and name it.
2. Add individual masks (numeric or expression).
3. Apply the family to filter your dataset.
4. View flagged vs retained rows.

---

## 📉 Outlier Detection & Handling

Detect and manage outliers using six methods spanning statistical and machine learning approaches.

### Detection Methods

| Method | Type | Key Parameter | Description |
| --- | --- | --- | --- |
| **Z-Score** | Statistical | `threshold` (default 3.0) | Flags values beyond ±N standard deviations from the mean. |
| **IQR** | Statistical | `multiplier` (default 1.5) | Flags values outside Q1 − k·IQR and Q3 + k·IQR. |
| **Quantile** | Statistical | `lower` / `upper` percentiles | Flags values below or above specified percentiles. |
| **Local Outlier Factor** | ML | `n_neighbors`, `contamination` | Density-based, flags points in sparse regions. |
| **Isolation Forest** | ML | `n_estimators`, `max_samples` | Tree-based, isolates anomalous points via random partitions. |
| **DBSCAN** | ML | `eps`, `min_samples` | Clustering-based, flags noise points not assigned to any cluster. |

### Handling Strategies

| Strategy | Available For | Description |
| --- | --- | --- |
| **None** | All methods | Detect only — no modification. |
| **Remove** | All methods | Delete rows identified as outliers. |
| **Clip** | Z-Score, IQR, Quantile | Cap values at the detection threshold. |
| **Tag** | All methods | Append tag/score columns to mark outliers. |

### Impact Analysis

After applying a handling strategy, the system reports the impact on each variable using selectable measures:

- **Flat** — Absolute change in mean, std, min, max.
- **% of Initial Value** — Percentage change relative to original value.
- **% of IQR** — Change normalised by the interquartile range.

### Outlier Visualisation

A 2-D projection scatter plot overlays detected outliers onto a reduced representation of the data. Supported projection methods:

- PCA, FAMD, t-SNE, UMAP

Outlier points are colour-coded for easy visual inspection.

---

## 🔄 Dataset Comparison

Compare two dataset versions side by side. Data sources include:

- **Current Session Data** — the dataset loaded in the application.
- **Upload File** — CSV, Excel, or Parquet upload.
- **Enter File Path** — direct path to a file on disk.
- **Select from History** — pick a snapshot from past trace sessions.

The comparison highlights differences in shape, column overlap, and value-level changes.

---

## 📊 Data Quality Dashboard

The Data Quality Score aggregates results from all validation checks into a single overview:

| Scorecard | What It Measures |
| --- | --- |
| **Completeness** | Percentage of non-missing values across the dataset. |
| **Validity (Outliers)** | Proportion of values within the expected statistical range. |
| **Consistency** | Adherence to anomaly rules and clinical logic. |

> **Tip**: Focus on resolving high-impact issues first — anomalies and outliers that meaningfully shift the overall score.
