# Exploration Modules Documentation

## corr_matrix.py

_No module description._

**Imports**:
`re`, `warnings`, `matplotlib.pyplot`, `numpy`, `pandas`, `plotly.express`, `plotly.figure_factory`, `plotly.graph_objects`, `seaborn`, `matplotlib.colors.to_hex`, `openpyxl.styles.Font`, `openpyxl.styles.PatternFill`, `explore.correlation_strength.STRENGTH_ORDER`, `explore.correlation_strength.STRENGTH_THRESHOLDS`, `explore.correlation_strength.CorrelationMethod`, `explore.correlation_strength.categorize_strength`, `functools.cache`

### def `plotly_corr_mat` (corr_matrix.py)

- **Arguments**: `data, targets, predictors, method, selected_color, corr_method_name, triangle, cluster_rows, cluster_cols, show_row_dendrogram, show_col_dendrogram, correlation_matrix`

- **Returns**: `None`

Generate an interactive correlation matrix using Plotly, with optional triangular masking.

Parameters:
- data (pd.DataFrame): The input data containing the variables.

- targets (list): List of target variable names.

- predictors (list): List of predictor variable names.

- method (str): Correlation method ('pearson', 'spearman', 'kendall').

- selected_color (str): Colormap for the heatmap.

- corr_method_name (str or None): Display name for the correlation method (optional).

- triangle (str): 'lower', 'upper', or 'full' to control the displayed triangle.

- cluster_rows (bool): Whether to apply hierarchical clustering to reorder rows.

- cluster_cols (bool): Whether to apply hierarchical clustering to reorder columns.

- show_row_dendrogram (bool): Whether to show the row dendrogram tree.

- show_col_dendrogram (bool): Whether to show the column dendrogram tree.

- correlation_matrix (pd.DataFrame): Pre-calculated correlation matrix (optional).

Returns:
- plotly.graph_objs._figure.Figure: Plotly figure object.

### def `add_correlation_strength_table` (corr_matrix.py)

- **Arguments**: `writer, method, sheet_name`

- **Returns**: `None`

Add the correlation strength information table to a new sheet in the Excel file,
with the row corresponding to the selected method bolded.

Parameters:
writer (pd.ExcelWriter): The Pandas ExcelWriter object.
method (str): The correlation method ('pearson', 'spearman', 'kendall').
sheet_name (str): The name of the sheet to add the information table.

Returns:
None

### def `add_correlation_chart_openpyxl` (corr_matrix.py)

- **Arguments**: `writer, correlations, method, sheet_name, threshold, top_n`

- **Returns**: `None`

Add an Excel sheet showing correlations sorted from greatest to lowest,
with correlation strength labels based on the selected method (Pearson, Spearman, Kendall).
Filters correlations based on the strength threshold (default: 'strong' or more) and limits the display to the top N correlations.

Parameters:
writer (pd.ExcelWriter): The Pandas ExcelWriter object.
correlations (pd.DataFrame): The correlation matrix.
method (str): The correlation method ('pearson', 'spearman', 'kendall').
sheet_name (str): The name of the sheet where the chart will be added.
threshold (str): Minimum strength category to include in the chart ('negligible', 'weak', 'moderate', 'strong', 'very strong').
top_n (int): Number of top correlations to display (default: 20).

Returns:
None

### def `generate_palette` (corr_matrix.py)

- **Arguments**: `style: str`

- **Returns**: `list`

Generate a diverging color palette for correlation matrix visualization.

Parameters:
style (str): Color style to use:
    - "blue_white_red": Clear blue (negative) → white (zero) → red (positive)

    - "coolwarm": Matplotlib's coolwarm colormap (color-blind friendly)

    - "RdBu_r": Red-Blue reversed (classic scientific)

    - "legacy": Original seaborn palette

Returns:
list of str: List of 256 color hex codes.

### def `get_color_from_value` (corr_matrix.py)

- **Arguments**: `value: float, min_val: float, max_val: float, palette: list`

- **Returns**: `str`

Get the fill color based on the correlation value using an existing color palette.
Uses symmetric scaling centered at zero for correlation matrices.

Parameters:
value (float): The correlation value to determine the color.
min_val (float): Minimum value in the correlation matrix for scaling.
max_val (float): Maximum value in the correlation matrix for scaling.
palette (list of str): List of color hex codes.

Returns:
str: The color code in hexadecimal format.

### def `_write_corr_to_sheet` (corr_matrix.py)

- **Arguments**: `writer: pd.ExcelWriter, correlations: pd.DataFrame, counts: pd.DataFrame, sheet_name: str, method: str`

- **Returns**: `None`

Helper function to write the correlation matrix to a sheet and apply formatting.

Parameters:
writer (pd.ExcelWriter): The Pandas ExcelWriter object.
correlations (pd.DataFrame): The correlation matrix.
counts (pd.DataFrame): The matrix containing counts of rows used for each correlation.
sheet_name (str): The name of the sheet in the Excel file.
method (str): Correlation method ('pearson', 'spearman', etc.).

Returns:
None

### def `custom_corr_mat_to_excel` (corr_matrix.py)

- **Arguments**: `data, targets, predictors, group_column, file_name, method, threshold`

- **Returns**: `None`

Calculate the correlation matrix for the given targets and predictors, and write it to an Excel file
with conditional formatting based on float values for the correlation matrix.
The formatted values with counts (x.xx (count)) are updated after applying the color formatting.

Parameters:
data (pd.DataFrame): The DataFrame containing the target and predictor columns.
targets (list): List of target column names.
predictors (list): List of predictor column names.
group_column (str): Optional. Column name to group data by. A sheet will be created for each group.
file_name (str or BytesIO): Name of the Excel file or BytesIO buffer to write to.
method (str): Correlation method ('pearson', 'spearman', etc.).
threshold (str): Correlation strength threshold for the summary chart.

Returns:
None

### def `_compute_corr_and_counts` (corr_matrix.py)

- **Arguments**: `data, targets, predictors, method, min_periods`

- **Returns**: `None`

Compute the correlation matrix and count of valid (non-NaN) observations for each pair of variables.

Parameters:
data (pd.DataFrame): The DataFrame containing the target and predictor columns.
targets (list): List of target column names.
predictors (list): List of predictor column names.
method (str): Correlation method ('pearson', 'spearman', etc.).

Returns:
pd.DataFrame: Correlation matrix (with NaN for missing correlations).
pd.DataFrame: Matrix of counts (number of valid rows used for each correlation).

---

## correlation_strength.py

_Correlation strength categories and thresholds (single source of truth).

The absolute-value cut-offs for Pearson correlations follow Schober, Boer & Schwarte (2018),
*Correlation Coefficients: Appropriate Use and Interpretation*, Anesthesia & Analgesia 126(5):
negligible < 0.10, weak < 0.40, moderate < 0.70, strong < 0.90, very strong >= 0.90.
Spearman and Kendall cut-offs are the values historically used by this application
(Kendall's tau is converted approximately with tau ~ (2/pi) * asin(r)).

This module has no UI dependency; the correlation graph builder, the Excel export and the
legend all read the same table._

**Imports**:
`__future__.annotations`, `enum.StrEnum`, `pandas`

### class `CorrelationMethod` (correlation_strength.py)

No description available.

### class `StrengthCategory` (correlation_strength.py)

No description available.

### def `coerce_method` (correlation_strength.py)

- **Arguments**: `method: ComplexType`

- **Returns**: `CorrelationMethod`

Accept enum members or strings (case-insensitive); unknown values fall back to Pearson.

### def `coerce_category` (correlation_strength.py)

- **Arguments**: `category: ComplexType`

- **Returns**: `StrengthCategory`

Accept enum members, display labels ('Very Strong') or slugs ('very_strong').

### def `thresholds_for` (correlation_strength.py)

- **Arguments**: `method: ComplexType`

- **Returns**: `tuple[ComplexType]`

No description available.

### def `categorize_strength` (correlation_strength.py)

- **Arguments**: `abs_value: float, method: ComplexType`

- **Returns**: `StrengthCategory`

Map |r| to its strength category for the given method.

### def `min_abs_for` (correlation_strength.py)

- **Arguments**: `category: ComplexType, method: ComplexType`

- **Returns**: `float`

Smallest |r| that belongs to ``category`` (or stronger) for the given method.

### def `strength_table` (correlation_strength.py)

- **Arguments**: `method: ComplexType`

- **Returns**: `pd.DataFrame`

Table of categories and lower bounds (all methods, or one) for exports and legends.

---

## data_quality_auditor.py

_No module description._

**Imports**:
`logging`, `collections.abc.Mapping`, `typing.Any`, `numpy`, `pandas`, `biostats.outliers.UNIVARIATE_METHODS`, `biostats.outliers.detect_outliers`, `core.derived.from_config`, `core.population.family_masks`, `core.registry.as_code`, `enrich.derived.CANNOT`, `enrich.derived.DISAGREES`, `enrich.derived.MIN_ROWS`, `enrich.derived.MOSTLY`, `enrich.derived.TOO_FEW`, `enrich.derived.check_derived`

### class `DataQualityAuditor` (data_quality_auditor.py)

Audits a pandas DataFrame to assess data quality across multiple dimensions:
- Completeness: Presence of missing values.

- Uniqueness: Presence of duplicate rows.

- Validity: Presence of outliers in numerical data.

- Consistency: Basic type consistency checks.

- Clinical Validity: Compliance with declared anomaly criteria.

**Methods:**

- **__init__**(`self, df: pd.DataFrame, config: ComplexType, imputed: ComplexType`) -> `None`
  > No description available.

- **compute_completeness**(`self`) -> `float`
  > Calculates the percentage of non-missing values.

- **compute_uniqueness**(`self`) -> `float`
  > Calculates the percentage of unique rows.

- **compute_validity**(`self, method: ComplexType, params: ComplexType`) -> `float`
  > Calculates a validity score based on the absence of outliers in numerical columns.
  >
  > The outliers are those of biostats.outliers.detect_outliers, the detector of the Validate page and of the
  > dashboard's outlier tab, with the same parameters, so the score counts what they flag.
  > - Univariate methods ('iqr', the default, 'zscore', 'quantile'): the share of the values of the continuous
  >   numeric columns that are outliers.
  > - Multivariate methods ('Local Outlier Factor', 'Isolation Forest', 'DBSCAN'): the share of the rows
  >   complete on the numeric columns that are flagged (at least 11 rows are needed).
  >
  >
  > **Parameters**:
  >
  > method : str, optional
  >     Detection method: 'iqr' (default), 'zscore', 'quantile',
  >     'Local Outlier Factor', 'Isolation Forest', 'DBSCAN'
  > params : dict, optional
  >     Method-specific parameters (e.g., multiplier, threshold, etc.); the defaults are those of
  >     biostats.outliers.DEFAULT_PARAMS

- **compute_consistency**(`self`) -> `float`
  > Approximates consistency by checking if text/object columns could be converted to numeric or datetime.
  > If a column is text but contains mostly numbers, it might be inconsistent formatting.

- **compute_clinical_validity**(`self`) -> `ComplexType`
  > Calculates a score based on declared anomaly criteria (if available).
  > Score = 100 - (% of rows the clinical-anomalies family flags). The rules are read as the analysis
  > population reads them (core.population.family_masks), with the family's operator; a rule that cannot be
  > evaluated is listed in the details, and the score uses the rules that can.
  > Returns None if no anomaly rules are defined (shows as N/A in dashboard).

- **compute_uniformity**(`self`) -> `float`
  > Checks for string uniformity issues:
  > - Leading/trailing whitespace.
  > - Inconsistent capitalization (e.g., 'Male' vs 'male').

- **compute_nullity_correlation**(`self`) -> `None`
  > Calculates the correlation between the missingness of variables.
  > Returns a DataFrame where 1 means variables tend to be missing together.

- **check_mar_dependency**(`self, target_col`) -> `None`
  > Checks if missingness in `target_col` is dependent on other observed variables (MAR).
  > Returns a list of dependencies found.

- **perform_mcar_test**(`self`) -> `None`
  > Performs a heuristic Little's MCAR test by aggregating pairwise MAR checks.
  > If we find significant dependencies between missingness and observed values,
  > we reject the MCAR hypothesis.

- **run_audit**(`self, validity_method: ComplexType, validity_params: ComplexType`) -> `dict[ComplexType]`
  > Runs all checks and populates metrics and advice.
  >
  >
  > **Parameters**:
  >
  > validity_method : str, optional
  >     Method to use for Statistical Validity calculation
  > validity_params : dict, optional
  >     Parameters for the validity method

- **generate_advice**(`self`) -> `None`
  > Generates actionable advice based on metrics.

- **derived_advice**(`self`) -> `ComplexType`
  > The declared derived variables (Enrich > Handle Missing Data > Derived variables) against the data, every row:
  > the recorded values their formula does not give, and the missing values it can compute ("computational
  > conformance", Kahn et al. 2016). Variable names (as code: a name can come from an uploaded file), counts and the
  > check's tolerance, never a row's value nor a check's reason (it can quote one); none when nothing is declared
  > or there is nothing to say. Shown on screen only: an export would need the study package's disclosure rule
  > (k = 5 for each count and difference).

### def `_details` (data_quality_auditor.py)

- **Arguments**: `rows: list[tuple[ComplexType]]`

- **Returns**: `pd.DataFrame`

No description available.

---

## quality_score.py

_The global data-quality score as a weighted mean of the audit dimensions, with its decomposition.

Pure Python (no Streamlit, no plotting): the dashboard draws the decomposition, the tests check it.
Each dimension scores 0-100; its weight decides its share of the global score; the points a
dimension "loses" are ``(100 - score) * share``, so the global score is ``100 - sum(lost)``._

**Imports**:
`__future__.annotations`, `collections.abc.Mapping`, `dataclasses.dataclass`

### class `DimensionScore` (quality_score.py)

No description available.

**Methods:**

- **earned**(`self`) -> `float`
  > Points this dimension contributes to the global score.

- **lost**(`self`) -> `float`
  > Points of the global score this dimension costs.

### class `QualityScore` (quality_score.py)

No description available.

**Methods:**

- **by_loss**(`self`) -> `list[DimensionScore]`
  > Dimensions from the most to the least costly (ties in display order).

- **lost**(`self`) -> `float`
  > No description available.

### def `ordered` (quality_score.py)

- **Arguments**: `names`

- **Returns**: `list[str]`

Known dimensions first, in display order, then any other metric.

### def `weighted_score` (quality_score.py)

- **Arguments**: `metrics: Mapping[ComplexType], weights: ComplexType`

- **Returns**: `QualityScore`

Weighted mean of the available dimensions (a ``None`` metric, e.g. clinical validity without rules, is left out).

With every weight at zero the global score is 0, as before.

### def `band` (quality_score.py)

- **Arguments**: `score: float`

- **Returns**: `str`

``excellent`` (>= 90), ``good`` (>= 80), ``fair`` (>= 50) or ``poor``.

### def `interpretation` (quality_score.py)

- **Arguments**: `score: float`

- **Returns**: `str`

No description available.

---
