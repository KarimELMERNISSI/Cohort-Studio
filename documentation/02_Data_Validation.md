# Data Validation & Monitoring

The **Data Validation & Monitoring** page provides a comprehensive data quality framework. It detects clinical anomalies, applies inclusion/exclusion criteria, identifies statistical and ML-based outliers, and compares dataset versions — all tracked through an integrated Data Quality Score.

---

## Completeness

Analyzes missing values and helps diagnose the mechanism of missingness (MCAR, MAR, MNAR).

### Missing Data Analysis

- **Basic Counts**: Bar chart and percentage of missing values per column.
- **Advanced Diagnosis**:
  - **Nullity Matrix**: A clustered heatmap visualizing patterns of missing data. Useful for spotting blocks of missingness across rows or correlations between variables.
  - **Distribution**: Histogram of row missingness (percentage of missing values per row).

### Statistical Diagnosis

- **Little's MCAR Test**: A statistical test to determine if data is Missing Completely At Random (MCAR).
  - *p > 0.05*: Likely MCAR (random).
  - *p < 0.05*: Likely MAR or MNAR (systematic).
- **MAR Indicator**: Checks if missingness in one variable is correlated with values in another, helping identify Missing At Random (MAR) dependencies.

---

## Clinical Anomalies

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

## Yes Inclusion Criteria

Define inclusion and exclusion criteria, then tick the ones that define your **analysis population**. Criteria are
**filters**: ticking or unticking them changes which rows the analyses use, never the dataset itself. Criteria are
organised into **mask families** — groups of related conditions combined with logical operators.

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
3. Run the check to see the flagged and retained rows.
4. Click **Exclude these anomalies from the analysis population** or **Restrict the analysis population to these
   criteria**: the family is ticked in the sidebar (*Population*). You can also tick or untick single criteria there.

### The analysis population

- The sidebar shows the population on every page (for example *928 of 1,206 rows (12 criteria)*), and each analysis
  page repeats it in a banner. Every result says which population it was computed on; a result computed on another
  population is not shown as current.
- Families are applied in the order of the configuration: an exclusion family (operator *any*) excludes a row when a
  ticked rule fires; an inclusion family (operator *all*) excludes a row that fails a ticked criterion. A missing
  value never fires a rule: it does not flag an anomaly and does not meet an inclusion criterion.
- A ticked criterion that cannot be evaluated (a typo in a column name, a text column in a numeric rule) stops the
  analyses with an explanation instead of silently keeping or dropping rows.
- The **Analysis population** tab shows the participant flow: rows in, excluded and out for each family, what each
  criterion excludes (on the whole table, at its step, and as the first reason), and the excluded rows with their
  reasons. These are the numbers of a STROBE participant-flow diagram (item 13).
- **Keep only these rows** deletes the excluded rows from the current data (for an export or a hand-over). It is
  the one action that removes rows: it asks for confirmation, it is one step of the trace, saved versions are not
  changed, and the criteria are then unticked.

---

## Outlier Detection & Handling

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

### Acting on the outliers

- **Remove** or **Tag**: *Exclude these outlier rows from the analysis population* adds the flagged rows as a
  criterion of the *outliers* family (the row ids are frozen when you add it, so the criterion keeps meaning the
  same rows) and ticks it. No row is deleted, and the criterion can be unticked in the sidebar.
- **Clip** (z-score, IQR, quantile): *Clip the values in the dataset* sets the values outside the method's bounds
  to the bounds, as one step of the trace that records the bounds and is replayed with them.

---

## Dataset Comparison

Compare two dataset versions side by side. Data sources include:

- **Current Session Data** — the dataset loaded in the application.
- **Upload File** — CSV, Excel, or Parquet upload.
- **Enter File Path** — direct path to a file on disk.
- **Select from History** — pick a snapshot from past trace sessions.

The comparison highlights differences in shape, column overlap, and value-level changes.

---

## Conformity & Consistency

Checks for data structure and formatting issues beyond simple value errors.

### Uniqueness

Detects **Duplicate Rows** in the dataset.

- **Metric**: Percentage of unique rows.
- **Action**: Review and remove duplicates that may skew analysis.

### Type Consistency

Identifies columns with **mixed data types**, specifically text (Object) columns that appear to contain numeric or datetime data.

- **Goal**: Ensure variables are cast to their correct types for accurate analysis.

### Uniformity

Detects specific string formatting issues in text columns:

- **Whitespace**: Leading or trailing spaces.
- **Capitalization**: Inconsistent casing (e.g., "Male" vs "male").

## Data Quality Dashboard

The Data Quality Score aggregates results from all validation checks into a single overview:

| Scorecard | What It Measures |
| --- | --- |
| **Completeness** | Percentage of non-missing values across the dataset. |
| **Statistical Validity** | Proportion of non-outlier values (based on selected method). |
| **Clinical Validity** | Compliance with defined clinical anomaly rules. |
| **Consistency** | Percentage of columns with correct data type definitions. |
| **Uniformity** | Percentage of text values with consistent formatting (case, whitespace). |
| **Uniqueness** | Percentage of unique (non-duplicate) rows. |

The **global score** is the weighted mean of the dimensions (Score Weights Configuration; clinical validity counts
double by default, and is left out when no anomaly rule is declared). It is drawn as three linked charts:

- **The ring** — each coloured segment is one dimension, sized by its weight; the grey gap after a segment is the
  points that dimension loses, so the score in the centre is 100 minus the gaps. Hover a segment for its score,
  weight, points added and points lost. Under the score: the change since the previous check of this dataset.
- **Where the score loses points** — one bar per dimension that loses points, the most costly on top. Click a bar to
  list where the points go: the columns with empty cells, the columns with outliers (or the rows flagged by a
  multivariate method), the text columns that hold numbers or dates, the columns with spacing or capitalisation
  issues, the duplicated rows, or the clinical rules that flag rows.
- **Score across the checks of this dataset** — one point per version of the data: as loaded, then after each change
  (merge, computed or transformed variable, imputation, renamed column, date conversion, kept population). The line
  uses the current weights; click a
  dimension in the legend to add its own line. Loading another dataset starts a new history.

> **Tip**: Focus on resolving high-impact issues first — the top bar of "Where the score loses points" — then come
> back after each change to see its effect on the score.
