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

- **Little's MCAR Test**: A chi-square test of whether the selected numeric variables are consistent with Missing Completely At Random (MCAR).
  - *p ≥ 0.05*: No evidence against MCAR. This does not prove MCAR (the test has little power in small samples), and the test cannot distinguish MAR from MNAR.
  - *p < 0.05*: Evidence against MCAR: the mechanism is MAR or MNAR (the test cannot tell them apart).
- **MAR Indicator**: Checks if missingness in one variable is correlated with values in another, helping identify Missing At Random (MAR) dependencies.

---

## Clinical Anomalies

The system checks your dataset for values that are clinically impossible or suspect, using the anomaly rules of your
study configuration. Declare the rules on the **Handle Clinical Anomalies** tab, or load a configuration file (JSON)
from the sidebar (**Action** > *Load Existing Config*); **Download Configuration File** saves it for a later session.
A loaded file adds the families and rules this session lacks, each whole. A rule this session already holds keeps its
definition here, and the page names those whose definition in the file differs.
For someone else, use **Download for Sharing** (`config_for_sharing.json`). It holds the criteria and the derived
variables, and leaves out what only means something here, naming it under the button:
- the rules that list the ids of the rows they exclude (outliers frozen as rows, by any method), also inside a rule
  "at least k of", and a family counted "at least k of" whole when one of its rules lists rows;
- the notes of the derived variables;
- this installation's settings: its folders and the paths of uploaded files.

The other rules are kept as written, with their descriptions: check that none names a participant (an expression
such as `df['patient_id'] == 'P-0042'`).

### How It Works

1. **Rule Evaluation** — Each rule is checked on every row: a plausible range on one column (for example potassium
   outside 2.5–7.0) or an expression (for example `df['BMI'] > 100`). See *Mask Types* below.
2. **Anomaly Flagging** — Rows with anomalies are flagged, with the violated rule displayed per variable.
3. **Impact Score** — Anomalies contribute to the overall Data Quality Score.

### Anomaly Rules

Every anomaly rule belongs to the *clinical_anomalies* family (operator `any`): a row is flagged when at least one
rule fires. The rules have no severity classes (such as impossible or suspect).

---

## Inclusion Criteria

Define inclusion and exclusion criteria, then tick the ones that define your **analysis population**. Criteria are
**filters**: ticking or unticking them changes which rows the analyses use, never the dataset itself. Criteria are
organised into **mask families** — groups of related conditions combined with logical operators.

### Key Concepts

- **Mask Family**: A group of conditions. The page fills three families: *clinical_anomalies* (Handle Clinical
  Anomalies tab), *study_inclusion* (Handle Inclusion Criteria tab) and *outliers* (Handle Outliers tab); a loaded
  configuration file can add others.
- **Mask**: An individual condition within a family (numeric range or expression).
- **Family Operator**: How masks within a family combine — `all` (inclusion: a row must meet every ticked criterion)
  or `any` (exclusion: a row is excluded when a ticked rule fires). A family a tab creates gets the tab's operator; a
  family that a loaded configuration gives another operator, such as "at least 2 rules", keeps it when its tab opens.

### Mask Types

| Type | Example |
| --- | --- |
| **Numeric** | Tick *Is Numeric*; mask name `Age` (the exact column name), lower bound 18, upper bound 90, strategy *include* |
| **Expression** | `df['Sex'] == 'Female'` or `(df['SBP'] > 140) & (df['DBP'] > 90)` |

A numeric mask applies to the column named like the mask. With both bounds, the strategy *include* selects the
values inside the range and *exclude* the values outside it: for an inclusion criterion, use *include* with the
eligible range (age 18–85); for an anomaly rule, use *exclude* with the plausible range (potassium 2.5–7.0 flags 9.8).
With a single bound, the values at or beyond it are selected (lower bound 18: age ≥ 18). In a configuration file, a
single bound with the strategy *exclude* selects the values short of it instead (age < 18), and the participant
flow says so. An expression is a Python condition on `df`; combine conditions with `&` (and) or `|` (or), each in
brackets.

### Workflow

1. Open the tab of the family: **Handle Clinical Anomalies** or **Handle Inclusion Criteria**.
2. Add individual masks (numeric or expression), by hand or with the *AI Assistant* (AI engine required).
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
- **The checks, the quality dashboard and the population read the rules the same way**, so they count the same rows.
  A rule that cannot be evaluated is shown with its reason on the tab that checks it.
- **Loading a configuration file.** A file that would be read in more than one way is refused, with the list of its
  problems:
  - an unknown family operator (operators are read without regard to case, so `All` is `all`);
  - a threshold that is not a whole number;
  - a family entry that is not a rule;
  - a numeric rule without bounds, with a bound that is not a number (`"18 years"`), or with an unknown strategy;
  - frozen bounds that are not `[lower, upper]` pairs of numbers (or `null`), listed rows that are not
    `[first, last]` pairs of row numbers, and a threshold rule without its rules or with a malformed one;
  - a rule that holds both a numeric range and an expression.

  A rule like these that is already in the session (loaded before this check) is shown as one that cannot be
  evaluated: the pages still open.
- The **Analysis population** tab draws the participant flow (STROBE item 13): the records read in the file,
  the rows removed by recorded steps, the rows excluded by each family of criteria and the analysis population.
  Click a box, or choose it in the menu under the diagram, to see the criteria behind an exclusion, the rows
  each excluded first and their definitions. Each definition says what the criterion does: "Implausible age:
  excluded when df['age'] > 110", or for an inclusion criterion, which is a requirement, "Adult: excluded unless
  18 <= age <= 85". The note under them says how a missing value counts: a participant whose value is missing meets
  no condition on it, so an exclusion criterion keeps them and an inclusion criterion excludes them (in an
  expression, a missing value meets `.isna()`, and may also meet a condition written with `!=` or `~`). Below, the
  tables give the rows in, excluded and out for each family, what each criterion excludes (on the whole table, at
  its step, and as the first reason), and the excluded rows with their reasons.
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
| **Quantile** | Statistical | `lower` / `upper` percentiles (default 0.01 / 0.99) | Flags values below or above specified percentiles (linear interpolation). |
| **Local Outlier Factor** | ML | `n_neighbors` (default 50), `contamination` | Density-based, flags points in sparse regions. |
| **Isolation Forest** | ML | `n_estimators`, `max_samples` | Tree-based, isolates anomalous points via random partitions (seeded). |
| **DBSCAN** | ML | `eps`, `min_samples` | Clustering-based, flags noise points not assigned to any cluster. |

**The same rows on every screen.** This tab, the quality dashboard's outlier tab and the Statistical Validity score
use one detector (`biostats/outliers.py`) with the same defaults, so they flag the same rows for the same method
and parameters.
- **Statistical methods** analyse the continuous columns: numeric, nullable integers included, with more than two
  values. A 0/1 column has no outliers, and the page lists the columns it did not analyse.
- **ML methods** use the rows that are complete on the chosen columns: numbers standardised, categories one-hot
  encoded, identifier-like text columns left out. A row with a missing value is not assessed, and the page says
  how many. Missing values are not imputed first, because imputed values would then decide which rows are
  flagged: impute them on Enrich if you want those rows assessed.

### Handling Strategies

| Strategy | Available For | Description |
| --- | --- | --- |
| **None** | All methods | Detect only — no modification. |
| **Remove** | All methods | The impact analysis shows the statistics without the flagged rows; the action excludes those rows from the analysis population (no row is deleted). |
| **Clip** | Z-Score, IQR, Quantile | The impact analysis shows the values capped at the detection bounds, exactly as the action clips them in the dataset (and as the replay of the trace does). |
| **Tag** | All methods | Flags the rows without changing the statistics; the action is the same as for Remove. |

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

A text file, uploaded or given by its path, is read with the detected delimiter and decimal separator. **How the
file is read** lets you choose others, as on the Import page.

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

**Recommendations & best practices**, under the score, lists what each dimension that loses points suggests. With
derived variables declared (Data > Enrich > Handle Missing Data > Derived variables), it also checks each one against
every row of the dataset, as Kahn et al. (2016) describe computational conformance: the rows where the recorded value
is not the one its formula gives (probable entry errors, or another unit when a published factor explains them), the
missing values its formula can compute (Fill exactly computes those of the analysis population), and a formula that
cannot be used, cannot be checked (too few rows, or a variable taking one value) or that the data disagree with, also
for one value of a code or text variable, which it does not name. It names the variables and counts the rows, never
shows a row's value, and does not compare the values an imputation filled. Its link opens the Derived variables tab.
It is advice only: the score does not change.
