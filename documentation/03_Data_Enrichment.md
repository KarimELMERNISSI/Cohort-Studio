# Data Enrichment

The **Data Enrichment** page handles the complete data preparation pipeline — from importing external data sources and imputing missing values, to engineering new variables through transformations, dimensionality reduction, and clustering. All transformations are logged to the trace system and can be replayed.

## The Enrich draft

Changes made on this page — merges, derived variables, transformations, imputations — go into a **draft**. A banner at
the top of the page lists them; **Apply to the dataset** makes each one a step of the trace and the analyses use them,
**Discard the draft** forgets them. The main dataset is the one loaded from Data > Import or the sidebar.

A draft can only be applied over the data it was built on. If the data change while it is open (a rename on
Explore, a date conversion on Validate), the banner offers to **redo** the changes on the current data from their
recorded settings, or to discard the draft. An imputation is fitted on, and fills, the rows of the **analysis
population** only (the ticked inclusion and exclusion criteria); if the population changes before the draft is
applied, the banner offers to refit it on the new population. When an applied imputation was fitted on other rows
than the current population, the analysis pages say so.

---

## Import External Data

Merge additional data sources into your main dataset.

### Workflow

1. **Main Dataset** — The current dataset, with the changes of the Enrich draft (load another dataset from Data > Import or the sidebar).
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
| **Chained equations** (shown as "MICE") | ML | Single imputation by chained equations (scikit-learn IterativeImputer, one deterministic pass). It is **not** multiple imputation: the uncertainty of the imputation is not propagated to later tests. |
| **MissForest** | ML | Iterative tree-based imputation (missForest algorithm with LightGBM learners) for mixed data types; models are fitted on originally observed rows only. |

#### Global Imputation Options

| Option | Description |
| --- | --- |
| **Categorical Encoder** | Optionally one-hot encode categorical columns during imputation. |
| **Numerical Scaler** | Optionally scale numerical columns during imputation. |
| **Remainder Strategy** | Handle columns not part of the imputation (passthrough, drop). |

After imputation, an **imputation mask** highlights which values were filled, and a **before/after summary** is displayed.

Values filled here are single imputations: fine for exploration, clustering and figures, but the analyses treat them as observed, so their standard errors and p-values are too small. CohortStudio remembers which cells were filled (also in saved versions and replays), and every result that uses them shows a warning.

### Multiple imputation (for inference)

The **Multiple imputation (for inference)** tab creates m completed versions of the analysis population. The analysis pages run each analysis on every version and pool the results with Rubin's rules (chapter 5). The dataset itself is not changed. The panel has five steps.

1. **Population and missingness**: rows, share of incomplete rows, the suggested m (at least the percentage of incomplete rows; White, Royston & Wood 2011) and the missing values of each variable. Values filled earlier by a single imputation are flagged: the multiple imputation would treat them as observed.
2. **Imputation model**: the variables. Identifiers, dates, free text and categories with more than 20 levels are left out by default. Keep every variable of the planned analyses, the outcome included: leaving it out biases the associations towards zero (Moons et al. 2006). Each incomplete variable gets a method:

   | Method | For | How |
   | --- | --- | --- |
   | `pmm` | numeric | predictive mean matching: a Bayesian regression draw, then an observed value of one of the 5 closest donors (default) |
   | `norm` | numeric | Bayesian linear regression with a random residual |
   | `logreg` | two categories | Bayesian logistic regression, augmented against perfect prediction (White, Daniel & Royston 2010) |
   | `polyreg` | several categories | multinomial logistic regression on a bootstrap sample of the observed rows |
   | `rf` | any | random-forest donors (Doove, van Buuren & Dusseldorp 2014) |
   | *do not impute* | structurally missing values | the variable stays missing and is not used as a predictor (for example a 6-month measurement of a patient who died before 6 months) |

3. **Settings**: m, iterations, seed (the same settings give the same imputations), imputation within the levels of a variable (`by`), a planned Cox model (the event and the Nelson-Aalen cumulative hazard then replace the follow-up time as predictors; White & Royston 2009), and several processor cores.
4. **Diagnostics**: chain traces (mean and SD of each variable per iteration: the chains should mix, without a trend), an indicative R-hat, observed against imputed values, and the predictors left out (constant or collinear).
5. **Use, save, export**: the status of the imputation against the current population (fresh, a union of whole strata, fitted on more rows, or out of date), saving a version with the imputation, mice long and Stata flong files, and a methods paragraph to adapt for the paper (Sterne et al. 2009).

The imputation is recorded in the trace (`mi_impute`, with a fingerprint of the imputed cells); a full replay re-runs it and checks that it finds the same cells.

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

> **Note**: Input variables are automatically standardised (Z-score). Dimensionality reduction and clustering need complete data: if a selected column has a missing value, Enrich stops and asks you to impute it first (Handle Missing Data).

#### Encoding

| Method | Description |
| --- | --- |
| **One-Hot** | Binary column per category. |
| **Label** | Integer encoding. |
| **Ordinal** | User-specified order mapping. |

#### Scaling

| Method | Description |
| --- | --- |
| **Z-Score Standardization** | Mean = 0, Std = 1. |
| **Min-Max Normalization** | Scaled to [0, 1]. |
| **Natural Logarithm Transformation** | log(x + 1). |

#### Statistics

Mean, Median, Summation, Minimum Value, Maximum Value or Standard Deviation of a selected column, stored as a new
column that repeats the value on every row.

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

- **Saved versions** — The sidebar dataset card: **Save** stores the current dataset as a new version, **Change** loads a saved one.
- **Automatic Snapshots** — Each merge, transformation and imputation saves a Parquet snapshot in your workspace; the Fast replay mode uses them.
- **Reproduce Analysis** — On Reproduce > Versions & traces, upload or select a trace file to replay it or download its Word report.

---

## Practical Guidance
- Run **Global Imputation** before **Dimensionality Reduction** or **Clustering** — these methods require complete data.
- Use **Targeted Imputation** with AI suggestions for domain-specific formulas (e.g. computing BMI from weight and height).
- Check the **imputation mask** after global imputation to verify which values were filled.
