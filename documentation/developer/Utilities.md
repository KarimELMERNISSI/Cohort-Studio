# Utility Functions

## _errors.py

_No module description._

### class `MultipleDataTypesError` (_errors.py)

Raised when any column of the input argument `x` has more than one
datatype when calling the function `_validate_single_datatype_features`.

### class `NotFittedError` (_errors.py)

Raised when attempting to call the class method `transform` before the
`MissForest` model has been trained.

---

## _label_encoding.py

_No module description._

**Imports**:
`pandas`

### def `_label_encoding` (_label_encoding.py)

- **Arguments**: `x: pd.DataFrame, mappings: dict`

- **Returns**: `pd.DataFrame`

Performs label encoding on given features and the input mappings.

**Parameters**:

x : pd.DataFrame of shape (n_samples, n_features)
    Dataset (features only) that needs to be encoded.
mappings : dict
    Dictionary that contains the categorical variables as keys and their
    corresponding encodings as values.

**Returns**:

x : pd.DataFrame of shape (n_samples, n_features)
    Label-encoded dataset (features only).

### def `_rev_label_encoding` (_label_encoding.py)

- **Arguments**: `x: pd.DataFrame, rev_mappings: dict`

- **Returns**: `pd.DataFrame`

Performs reverse label encoding on given features and the input
reverse mappings.

**Parameters**:

x : pd.DataFrame of shape (n_samples, n_features)
    Dataset (features only) that needs to be imputed.
rev_mappings : dict
    Dictionary that contains the categorical variables as keys and their
    corresponding encodings as values.

**Returns**:

x : pd.DataFrame of shape (n_samples, n_features)
    Reverse label-encoded dataset (features only).

---

## _metrics.py

_No module description._

**Imports**:
`numpy`, `pandas`

### def `pfc` (_metrics.py)

- **Arguments**: `x_imp_cat_curr: pd.DataFrame, x_imp_cat_prev: pd.DataFrame`

- **Returns**: `float`

Compute and return the proportion of falsely classified (PFC) of two
different categorical imputation matrices.

**Parameters**:

x_imp_cat_curr : pd.DataFrame
    Latest categorical imputation matrix.
x_imp_cat_prev : pd.DataFrame
    Before last categorical imputation matrix.

**Returns**:

float
    Proportion of falsely classified (PFC) of two different categorical
    imputation matrices.

### def `nrmse` (_metrics.py)

- **Arguments**: `x_imp_num_curr: pd.DataFrame, x_imp_num_prev: pd.DataFrame`

- **Returns**: `float`

Compute and return the normalized root mean squared error (NRMSE) two
different numerical imputation matrices.

**Parameters**:

x_imp_num_curr : pd.DataFrame
    Latest numerical imputation matrix.
x_imp_num_prev : pd.DataFrame
    Before last numerical imputation matrix.

**Returns**:

float
    Normalized root mean squared error (NRMSE) two different numerical
    imputation matrices.

---

## _validate.py

_No module description._

**Imports**:
`typing.Any`, `numpy`, `pandas`, `sklearn.base.BaseEstimator`, `utils.miss_forest._errors.MultipleDataTypesError`

### def `_is_estimator` (_validate.py)

- **Arguments**: `estimator: ComplexType`

- **Returns**: `bool`

Checks if the argument `estimator` is an object that implements the
scikit-learn estimator API.

**Parameters**:

estimator : estimator object
    This object is assumed to implement the scikit-learn estimator API.

**Returns**:

bool
    Returns True if the argument `estimator` is None or has class
    methods `fit` and `predict`. Otherwise, returns False.

### def `_validate_single_datatype_features` (_validate.py)

- **Arguments**: `x: pd.DataFrame`

- **Returns**: `None`

Checks if all values in the features belong to the same datatype.

**Parameters**:

x : pd.DataFrame of shape (n_samples, n_features)
    Dataset (features only) that needs to be checked.

**Raises**:

MultipleDataTypesError
    Raised if not all values in the features belong to the same datatype.

---

## analysis_utils.py

_No module description._

**Imports**:
`logging`, `numpy`, `pandas`, `scipy.stats`, `biostats.inference.LEGACY_NAMES`, `biostats.inference.compare_groups`, `biostats.inference.groups_from_frame`, `biostats.inference.recommend`, `biostats.posthoc.default_method`, `biostats.posthoc.pairwise_comparisons`

### def `render_analysis_configuration` (analysis_utils.py)

- **Arguments**: ``

- **Returns**: `None`

Compatibility shim: the widgets live in :mod:`app_pages.analysis_config`.

### def `analyze_variable` (analysis_utils.py)

- **Arguments**: `df, group_col, target, numeric_cols, categorical_cols, binary_cols, manual_test`

- **Returns**: `None`

Analyzes a single variable against the group column, selecting the appropriate test.

For binary outcomes the 2x2 measures use an explicit orientation (audit S1): ``exposure_ref`` is the
reference level of ``group_col`` and ``event_level`` the outcome level counted as the event; both are
auto-detected (0/No/control vs 1/Yes/case) when omitted.

### def `recommend_statistical_test` (analysis_utils.py)

- **Arguments**: `df, group_col, target_col, numeric_cols, categorical_cols, binary_cols`

- **Returns**: `None`

Recommended test and reasoning: Welch-first for numeric outcomes (audit S7); chi-square family otherwise.

### def `compare_variable_result` (analysis_utils.py)

- **Arguments**: `df, group_col, target, manual_test, reference`

- **Returns**: `None`

Full :class:`biostats.TestResult` for a numeric variable (used by the result card).

### def `perform_post_hoc` (analysis_utils.py)

- **Arguments**: `df, group_col, target_col, test_type, adjust_method`

- **Returns**: `None`

Every pairwise comparison with raw and adjusted p-values; the method follows the omnibus test (audit S7).

### def `compute_cohens_d` (analysis_utils.py)

- **Arguments**: `group1, group2`

- **Returns**: `None`

Compute Cohen's d for two independent groups.

### def `compute_eta_squared` (analysis_utils.py)

- **Arguments**: `groups_data`

- **Returns**: `None`

Compute Eta Squared for ANOVA.

### def `compute_cramers_v` (analysis_utils.py)

- **Arguments**: `confusion_matrix`

- **Returns**: `None`

Bias-corrected Cramer's V (Bergsma 2013) from the Pearson chi-square without Yates correction.

### def `littles_mcar_test` (analysis_utils.py)

- **Arguments**: `df`

- **Returns**: `None`

Little's MCAR test on EM estimates of the mean and covariance (audit S15); see :mod:`biostats.mcar`.

### def `scan_missingness_dependencies` (analysis_utils.py)

- **Arguments**: `df, target_cols, predictor_cols, test_preference, progress_callback, correction, alpha`

- **Returns**: `None`

Deprecated shim over :func:`biostats.missing.scan_missingness_dependencies`.

Returns EVERY performed test (not only p < 0.05, audit S3) so that a multiplicity correction applied by
the caller controls the real family. With ``correction`` set, adjusted p-values are included.

---

## clustering_utils.py

_Clustering and Dimensionality Reduction Utilities

This module provides functions for:
- Dimensionality reduction (PCA, t-SNE, UMAP, FAMD)

- Clustering (K-Means, DBSCAN, Gaussian Mixture)

- Optimal cluster analysis (Elbow, Silhouette)

All heavy imports are done lazily to avoid slowing app startup._

**Imports**:
`typing.Any`, `numpy`, `pandas`

### def `standardize_data` (clustering_utils.py)

- **Arguments**: `df: pd.DataFrame`

- **Returns**: `pd.DataFrame`

Standardize numeric columns to mean=0, std=1.

### def `prepare_data_for_clustering` (clustering_utils.py)

- **Arguments**: `df: pd.DataFrame, numeric_cols: list[str], categorical_cols: ComplexType, handle_missing: str`

- **Returns**: `tuple[ComplexType]`

Prepare data for clustering: handle missing values, standardize.

Returns:
    (prepared_df, valid_indices)

### def `run_pca` (clustering_utils.py)

- **Arguments**: `data: pd.DataFrame, n_components: int`

- **Returns**: `tuple[ComplexType]`

Run PCA on the data.

Returns:
    (embeddings, info_dict with explained_variance)

### def `run_tsne` (clustering_utils.py)

- **Arguments**: `data: pd.DataFrame, n_components: int, perplexity: float, max_iter: int, max_samples: int`

- **Returns**: `tuple[ComplexType]`

Run t-SNE on the data.
Note: t-SNE is slow for large datasets, so we sample if needed.

Returns:
    (embeddings, info_dict)

### def `run_umap` (clustering_utils.py)

- **Arguments**: `data: pd.DataFrame, n_components: int, n_neighbors: int, min_dist: float, max_samples: int`

- **Returns**: `tuple[ComplexType]`

Run UMAP on the data.

Returns:
    (embeddings, info_dict)

### def `run_famd` (clustering_utils.py)

- **Arguments**: `data: pd.DataFrame, n_components: int`

- **Returns**: `tuple[ComplexType]`

Run FAMD (Factor Analysis of Mixed Data) for mixed numeric/categorical data.

Returns:
    (embeddings, info_dict)

### def `fit_kmeans` (clustering_utils.py)

- **Arguments**: `embeddings: np.ndarray, n_clusters: int`

- **Returns**: `tuple[ComplexType]`

Fit K-Means clustering.

Returns:
    (cluster_labels, info_dict)

### def `fit_dbscan` (clustering_utils.py)

- **Arguments**: `embeddings: np.ndarray, eps: float, min_samples: int`

- **Returns**: `tuple[ComplexType]`

Fit DBSCAN clustering.

Returns:
    (cluster_labels, info_dict) - Note: -1 means noise

### def `fit_gaussian_mixture` (clustering_utils.py)

- **Arguments**: `embeddings: np.ndarray, n_components: int`

- **Returns**: `tuple[ComplexType]`

Fit Gaussian Mixture Model.

Returns:
    (cluster_labels, info_dict with probabilities)

### def `optimal_k_analysis` (clustering_utils.py)

- **Arguments**: `embeddings: np.ndarray, k_range: range`

- **Returns**: `dict[ComplexType]`

Compute elbow and silhouette metrics for different K values.

Returns:
    dict with inertias, silhouette_scores, optimal_k_elbow, optimal_k_silhouette

### def `compute_cluster_profiles` (clustering_utils.py)

- **Arguments**: `df: pd.DataFrame, labels: np.ndarray, numeric_cols: list[str], categorical_cols: ComplexType`

- **Returns**: `pd.DataFrame`

Compute mean/mode of variables per cluster.

Returns:
    DataFrame with cluster statistics

---

## config_loader.py

_No module description._

**Imports**:
`re`, `typing.Any`, `utils.data_paths.get_data_root`

### def `create_empty_config` (config_loader.py)

- **Arguments**: ``

- **Returns**: `dict[ComplexType]`

Create an empty configuration structure.

**Returns**:

dict
    A dictionary containing the default configuration structure for the application,
    including sections for masks, transformations, thresholds, and folder paths.

### def `transform_expression` (config_loader.py)

- **Arguments**: `expression, df_name`

- **Returns**: `None`

Transform a user-friendly expression into a valid Python expression for DataFrame filtering.

**Parameters**:

expression : str
    The input expression (e.g., "Age > 18").
df_name : str, optional
    The name of the dataframe variable to use in the expression (default is 'df').

**Returns**:

str
    The transformed expression executable in Python (e.g., "df['Age'] > 18").

---

## data_analyzer.py

_Data Analyzer Module.

Provides the DataAnalyzer class for analyzing and categorizing DataFrame columns.
This is a consolidated module combining functionality from home.py and visualization_utils.py._

**Imports**:
`dataclasses.dataclass`, `dataclasses.field`, `typing.Any`, `numpy`, `pandas`

### class `DatasetProfile` (data_analyzer.py)

Immutable representation of categorized DataFrame columns.

### class `DataAnalyzer` (data_analyzer.py)

Helper class to analyze and validate data columns.

Categorizes columns into:
- numeric_cols: Numeric columns (excluding low cardinality)

- categorical_cols: Categorical and object columns

- date_cols: Date/datetime columns

- binary_cols: Columns with exactly 2 unique values

- low_cardinality_numeric_cols: Numeric columns with <= 10 unique values

- high_cardinality_cat_cols: Categorical columns with many unique values

- timedelta_cols: Time interval columns

**Methods:**

- **__init__**(`self, df: pd.DataFrame`) -> `None`
  > Initialize the analyzer with a DataFrame.
  >
  > Args:
  >     df: pandas DataFrame to analyze

- **refresh**(`self, df: pd.DataFrame`) -> `None`
  > Refresh the column metadata based on the latest DataFrame state.
  >
  > Args:
  >     df: Updated pandas DataFrame

- **profile**(`self`) -> `DatasetProfile`
  > Returns an immutable snapshot of the current dataset profile.

- **analyze_columns**(`self`) -> `None`
  > Analyze and categorize columns by data type.

- **_detect_date_columns**(`self`) -> `list[str]`
  > Detect columns that are likely dates, with additional checks for accuracy.

- **_detect_binary_columns**(`self`) -> `tuple[ComplexType]`
  > Detect columns with only two unique values, including numeric columns.
  >
  > Returns:
  >     tuple: (all_binary, numeric_binary, non_numeric_binary)

- **_detect_low_cardinality_numeric_columns**(`self`) -> `list[str]`
  > Detect numeric columns with <= 10 unique values to treat as categorical.

- **_detect_high_cardinality_cat_columns**(`self`) -> `list[str]`
  > Detect categorical columns with many unique values.

- **get_suitable_columns**(`self, plot_type: str`) -> `dict[ComplexType]`
  > Get suitable columns for different plot types.
  >
  > Args:
  >     plot_type: Type of plot (e.g., 'Histogram', 'Box Plot', 'Scatter Plot')
  >
  > Returns:
  >     dict: Mapping of axis/parameter to suitable columns

---

## data_paths.py

_Centralized Data Path Configuration

All data paths are configured relative to a single DATA_ROOT environment variable.
This allows Docker deployments to use mounted volumes while keeping local development simple.

Usage:
    from utils.data_paths import get_datasets_dir, get_vectors_dir

Environment Variables:
    DATA_ROOT: Base directory for all data storage (default: "./data")_

**Imports**:
`os`

### def `get_data_root` (data_paths.py)

- **Arguments**: ``

- **Returns**: `str`

Get the base data directory.

### def `get_datasets_dir` (data_paths.py)

- **Arguments**: ``

- **Returns**: `str`

Get the datasets directory path.

### def `get_stats_dir` (data_paths.py)

- **Arguments**: ``

- **Returns**: `str`

Get the statistics cache directory path.

### def `get_vectors_dir` (data_paths.py)

- **Arguments**: ``

- **Returns**: `str`

Vector indexes outside a signed-in session (one folder per engine, see manage.vectorstores).

### def `get_traces_dir` (data_paths.py)

- **Arguments**: ``

- **Returns**: `str`

Get the analysis traces directory path.

### def `get_uploads_dir` (data_paths.py)

- **Arguments**: ``

- **Returns**: `str`

Get the uploads directory path.

### def `get_taxonomy_dir` (data_paths.py)

- **Arguments**: ``

- **Returns**: `str`

Get the taxonomy directory path.

### def `get_enrichment_dir` (data_paths.py)

- **Arguments**: ``

- **Returns**: `str`

Get the enrichment input directory path.

### def `get_enriched_dir` (data_paths.py)

- **Arguments**: ``

- **Returns**: `str`

Get the enriched output directory path.

### def `get_comparisons_dir` (data_paths.py)

- **Arguments**: ``

- **Returns**: `str`

Get the comparisons directory path.

### def `get_knowledge_graphs_dir` (data_paths.py)

- **Arguments**: ``

- **Returns**: `str`

Get the knowledge graphs directory path.

### def `get_rag_eval_logs_dir` (data_paths.py)

- **Arguments**: ``

- **Returns**: `str`

Get the RAG evaluation logs directory path.

---

## date_parser.py

_No module description._

**Imports**:
`pandas`

### def `smart_parse_dates` (date_parser.py)

- **Arguments**: `series`

- **Returns**: `None`

Parses dates by enforcing a single consistent format across the column.

Detects components (Day, Month, Year) by analyzing global numeric ranges.

Strategy:
1. Identify Separator & Split Strings
2. Analyze each column position (0, 1, 2)
3. Deduction Rules:
   - Value > 31 => YEAR

   - Value > 12 (and not Year) => DAY

   - Remaining => MONTH
4. Reconstruct Date

**Parameters**:

series : pd.Series
    The pandas Series to parse.

**Returns**:

tuple
    (parsed_series, notes) - The parsed datetime series and a list of notes explaining the parsing logic.

---

## environment.py

_The software a dataset was loaded and analysed with: CohortStudio's version and commit, Python and the key libraries.
It is recorded in the trace when a dataset is loaded (the ``initial_load`` step) and printed in the study dossier,
so that a reader knows which versions produced the results._

**Imports**:
`__future__.annotations`, `platform`, `tomllib`, `importlib.metadata.PackageNotFoundError`, `importlib.metadata.version`, `pathlib.Path`

### def `app_version` (environment.py)

- **Arguments**: ``

- **Returns**: `str`

CohortStudio's version: the installed package, else pyproject.toml (a source checkout).

### def `git_commit` (environment.py)

- **Arguments**: `root: Path`

- **Returns**: `ComplexType`

The short commit of a source checkout (read from .git, without running git), or None.

### def `environment` (environment.py)

- **Arguments**: ``

- **Returns**: `dict[ComplexType]`

CohortStudio, its commit (when known), Python and the versions of the key libraries that are installed.

### def `environment_text` (environment.py)

- **Arguments**: ``

- **Returns**: `str`

The environment in one line, for reports: "CohortStudio 2.4.1 (commit abc1234), Python 3.14.3, numpy 2.4, ...".

---

## epidemiology_utils.py

_Epidemiology and Biostatistics Utility Module (legacy entry points).

What is implemented here:
- 2x2 contingency measures (OR with Woolf CI, RR with Katz CI, RD with Newcombe CI, NNT, attributable
  fraction) - delegated to :mod:`biostats.contingency`, which also carries the explicit table orientation

- Cohen's d (pooled SD) with a normal-approximation CI and the Hedges' g small-sample factor (approximate J)

- Eta-squared / omega-squared point estimates for one-way designs

- ANCOVA diagnostics (homogeneity of slopes, residual normality, Breusch-Pagan homoscedasticity)

- Diagnostic test accuracy (sensitivity, specificity, PPV, NPV with Wilson score CIs; LR+/LR-, Youden J)

No compliance with reporting guidelines is implied by this module; see documentation/05_Epidemiology.md._

**Imports**:
`logging`, `math`, `typing.Any`, `numpy`, `pandas`, `pydantic.BaseModel`, `pydantic.Field`, `scipy.stats`, `scipy.stats.norm`

### class `ConfidenceInterval` (epidemiology_utils.py)

Container for lower and upper confidence bounds.

### class `Contingency2x2Result` (epidemiology_utils.py)

Complete epidemiological evaluation for 2x2 contingency tables.

### class `EffectSizeResult` (epidemiology_utils.py)

Standardized effect size estimates with confidence intervals.

### class `ANCOVADiagnosticsResult` (epidemiology_utils.py)

Diagnostic tests and assumptions verification for ANCOVA.

### class `DiagnosticTestResult` (epidemiology_utils.py)

Clinical diagnostic accuracy metrics with Wilson score confidence intervals.

### def `calculate_2x2_epidemiology_metrics` (epidemiology_utils.py)

- **Arguments**: `table_or_counts: ComplexType, exposure_label: str, outcome_label: str, alpha: float`

- **Returns**: `Contingency2x2Result`

Calculate comprehensive epidemiological metrics from a 2x2 table.

Format:
             Outcome + (Case)    Outcome - (Control)
Exposed +           a                    b
Exposed -           c                    d

Args:
    table_or_counts: Either a 2x2 matrix/DataFrame or tuple (a, b, c, d).
    exposure_label: Descriptive label for the exposure factor.
    outcome_label: Descriptive label for the health outcome.
    alpha: Significance level for 100*(1-alpha)% confidence intervals.

Returns:
    Contingency2x2Result with OR, RR, RD, NNT, Attributable Fraction, tests, and guidance.

### def `compute_cohens_d_with_ci` (epidemiology_utils.py)

- **Arguments**: `group1: ComplexType, group2: ComplexType, alpha: float`

- **Returns**: `EffectSizeResult`

Compute Cohen's d with Hedges-Olkin 95% Confidence Interval and Hedges' g small-sample bias correction.

Args:
    group1: Observations for reference or exposed group.
    group2: Observations for comparator group.
    alpha: Significance level (default 0.05 for 95% CI).

Returns:
    EffectSizeResult with Cohen's d, CI, Hedges' g, and interpretation.

### def `compute_eta_and_omega_squared` (epidemiology_utils.py)

- **Arguments**: `groups_data: list[ComplexType]`

- **Returns**: `dict[ComplexType]`

Compute Eta-squared (η²) and Omega-squared (ω²) for One-Way ANOVA.
Omega-squared provides an unbiased population effect size for clinical studies.

### def `check_ancova_assumptions` (epidemiology_utils.py)

- **Arguments**: `df: pd.DataFrame, target: str, group: str, covariates: list[str]`

- **Returns**: `ANCOVADiagnosticsResult`

Rigorously verify ANCOVA epidemiological assumptions:
1. Homogeneity of regression slopes (Group x Covariates interaction test)
2. Normality of residuals (Shapiro-Wilk or Jarque-Bera)
3. Homoscedasticity of residuals (Breusch-Pagan)

Args:
    df: Dataset containing variables.
    target: Continuous dependent variable.
    group: Categorical factor.
    covariates: Continuous covariates adjusted for in the model.

Returns:
    ANCOVADiagnosticsResult with test statistics, warnings, and clinical recommendations.

### def `_wilson_score_interval` (epidemiology_utils.py)

- **Arguments**: `p: float, n: int, alpha: float`

- **Returns**: `tuple[ComplexType]`

Compute Wilson score interval for proportions.

### def `calculate_diagnostic_accuracy` (epidemiology_utils.py)

- **Arguments**: `tp: int, fp: int, fn: int, tn: int, alpha: float`

- **Returns**: `DiagnosticTestResult`

Calculate clinical diagnostic accuracy parameters with 95% Wilson score confidence intervals.

Args:
    tp: True Positives
    fp: False Positives
    fn: False Negatives
    tn: True Negatives
    alpha: Significance level (default 0.05 for 95% CI)

---

## export_utils.py

_Export Utilities.

Functions for exporting DataFrames to various formats (Excel, etc.)_

**Imports**:
`re`, `io.BytesIO`, `typing.Any`, `pandas`

### def `to_excel` (export_utils.py)

- **Arguments**: `df, index: bool, sheet_name: str`

- **Returns**: `bytes`

Excel bytes for a download button (the one implementation used by every page).

### def `to_excel_sheets` (export_utils.py)

- **Arguments**: `dataframes_dict: dict[ComplexType]`

- **Returns**: `bytes`

Converts multiple DataFrames into an Excel file with separate sheets.

Args:
    dataframes_dict: Dictionary mapping sheet names to pandas DataFrames.

Returns:
    bytes: Multi-sheet Excel workbook as bytes.

### def `to_csv_bytes` (export_utils.py)

- **Arguments**: `df: pd.DataFrame, index: bool`

- **Returns**: `bytes`

Converts a pandas DataFrame to UTF-8 encoded CSV bytes.

Args:
    df: The pandas DataFrame to convert.
    index: Whether to include the DataFrame index column.

Returns:
    bytes: CSV content encoded as UTF-8 bytes.

### def `fig_to_html_str` (export_utils.py)

- **Arguments**: `fig: Any, include_plotlyjs: ComplexType`

- **Returns**: `str`

Exports a Plotly figure object into an interactive HTML page.

Args:
    fig: Plotly Graph Object figure.
    include_plotlyjs: True embeds Plotly.js (a self-contained file of about 4.5 MB that opens offline);
        'cdn' links it from the internet (a small file that needs a connection to open).

Returns:
    str: HTML markup containing the interactive figure.

### def `fig_to_json_str` (export_utils.py)

- **Arguments**: `fig: Any`

- **Returns**: `str`

Exports a Plotly figure object to a JSON specification string.

Args:
    fig: Plotly Graph Object figure.

Returns:
    str: Serialized JSON representation of figure traces and layout.

### def `get_publication_plot_config` (export_utils.py)

- **Arguments**: `filename: str, format_type: str, width: int, height: int, scale: int`

- **Returns**: `dict[ComplexType]`

Generates a Plotly modebar configuration dict for publication-quality exports.

Configures the in-browser camera icon to export vector SVG or 300+ DPI PNG
directly without requiring server-side rendering dependencies.

Args:
    filename: Base filename for downloaded graphic.
    format_type: Image format ('svg', 'png', 'jpeg', 'webp').
    width: Image width in pixels.
    height: Image height in pixels.
    scale: Resolution scaling multiplier (e.g. 3 for publication print quality).

Returns:
    dict: Configuration dictionary for plotly_chart(fig, config=config).

---

## llm_utils.py

_LLM Utilities for Reliable Structured Output.

Provides robust JSON parsing, schema validation, and structured LLM invocation
with retry and repair capabilities._

**Imports**:
`json`, `logging`, `re`, `typing.Any`, `typing.TypeVar`, `pydantic.BaseModel`, `pydantic.ValidationError`, `utils.logging_config.redact_secrets`

### class `ProviderError` (llm_utils.py)

The language model answered with an error instead of a completion (a missing model, a refused key...).

### class `StructuredOutputHelper` (llm_utils.py)

Helper class for managing structured LLM outputs with schemas.

Usage:
    helper = StructuredOutputHelper(llm)
    result, error = helper.invoke_with_schema(prompt, MySchema)

**Methods:**

- **__init__**(`self, llm, max_retries: int`) -> `None`
  > No description available.

- **invoke_with_schema**(`self, prompt: str, schema: type[T], repair_on_fail: bool`) -> `tuple[ComplexType]`
  > Invoke LLM and parse response with schema validation.
  >
  > Args:
  >     prompt: The prompt to send
  >     schema: Pydantic model for validation
  >     repair_on_fail: If True, attempt LLM-based repair on parse failure
  >
  > Returns:
  >     Tuple of (parsed_model, error_message)

- **invoke_json**(`self, prompt: str, default: Any`) -> `tuple[ComplexType]`
  > Invoke LLM and parse as generic JSON without schema.
  >
  > Args:
  >     prompt: The prompt to send
  >     default: Default value if parsing fails
  >
  > Returns:
  >     Tuple of (parsed_json, error_message)

### def `provider_error` (llm_utils.py)

- **Arguments**: `text: Any`

- **Returns**: `ComplexType`

The error a connector answered with instead of a completion, without API keys; None for a completion.

### def `clean_json_response` (llm_utils.py)

- **Arguments**: `text: str`

- **Returns**: `str`

Clean LLM response to extract valid JSON.

Uses multiple strategies:
1. Extract from markdown code blocks
2. Find JSON by brace/bracket matching
3. Return cleaned text

### def `repair_json` (llm_utils.py)

- **Arguments**: `broken_json: str`

- **Returns**: `str`

Attempt to repair common JSON issues.

Fixes:
- Trailing commas

- Single quotes to double quotes (outside of values)

- Missing closing brackets

### def `parse_json_safe` (llm_utils.py)

- **Arguments**: `text: str, default: Any`

- **Returns**: `Any`

Safely parse JSON with multiple fallback strategies.

Args:
    text: Raw text containing JSON
    default: Default value if parsing fails completely

Returns:
    Parsed JSON or default value

### def `validate_and_parse` (llm_utils.py)

- **Arguments**: `text: str, schema: type[T], strict: bool`

- **Returns**: `tuple[ComplexType]`

Parse JSON and validate against Pydantic schema.

Args:
    text: Raw LLM response text
    schema: Pydantic model class to validate against
    strict: If True, raise on validation error

Returns:
    Tuple of (parsed_model, error_message)

### def `create_json_repair_prompt` (llm_utils.py)

- **Arguments**: `broken_json: str, expected_schema: str`

- **Returns**: `str`

Create a prompt to ask LLM to repair broken JSON.

Args:
    broken_json: The malformed JSON string
    expected_schema: Description of expected schema

Returns:
    Prompt for LLM to fix the JSON

### def `extract_json_from_text` (llm_utils.py)

- **Arguments**: `text: str`

- **Returns**: `list[dict]`

Extract all JSON objects from text.

Useful when LLM returns multiple JSON blocks or explanatory text.

Returns:
    List of parsed JSON objects

---

## logging_config.py

_Central logging configuration. Call ``configure_logging()`` once from the application entry point.

Every record is written without API keys (:func:`redact_secrets`): a provider's error can quote the key it refused,
as Google does for a suspended key ("Consumer 'api_key:...' has been suspended")._

**Imports**:
`__future__.annotations`, `logging`, `os`, `re`

### class `RedactSecrets` (logging_config.py)

Rewrites each record without API keys, its traceback included.

**Methods:**

- **filter**(`self, record: logging.LogRecord`) -> `bool`
  > No description available.

### def `redact_secrets` (logging_config.py)

- **Arguments**: `text: str`

- **Returns**: `str`

The text without API keys: the values of the configured secrets (variables named ``*API_KEY``,
``*_SECRET`` or ``*_PASSWORD``), Google, OpenAI and Anthropic keys, and the value of an ``api_key: ...``,
``x-api-key: ...`` or ``Authorization: ...`` field.

### def `single_line` (logging_config.py)

- **Arguments**: `value: object`

- **Returns**: `str`

``value`` on one line, for a log entry: a typed name, path or formula cannot forge another entry.

### def `configure_logging` (logging_config.py)

- **Arguments**: `level: ComplexType`

- **Returns**: `None`

Configure the root logger from ``CMP_LOG_LEVEL`` (default INFO). Safe to call repeatedly.

---

## missforest.py

_No module description._

**Imports**:
`warnings`, `collections.abc.Iterable`, `copy.deepcopy`, `typing.Any`, `numpy`, `pandas`, `sklearn.base.BaseEstimator`, `tqdm.tqdm`, `biostats._types.RANDOM_STATE`, `utils.miss_forest._errors.NotFittedError`, `utils.miss_forest._label_encoding._label_encoding`, `utils.miss_forest._label_encoding._rev_label_encoding`, `utils.miss_forest._metrics.nrmse`, `utils.miss_forest._metrics.pfc`, `utils.miss_forest._validate._is_estimator`, `utils.miss_forest._validate._validate_single_datatype_features`

### class `MissForest` (missforest.py)

**Attributes**:

classifier : Union[Any, BaseEstimator]
    Estimator that predicts missing values of categorical columns.
regressor : Union[Any, BaseEstimator]
    Estimator that predicts missing values of numerical columns.
initial_guess : str
    Determines the method of initial imputation.
max_iter : int
    Maximum iterations of imputing.
early_stopping : bool
    Determines if early stopping will be executed.
categorical_columns : list
    All categorical columns of given dataframe `x`.
numerical_columns : list
    All numerical columns of given dataframe `x`.
_is_fitted : bool
    A state that determines if an instance of `MissForest` is fitted.

**Methods**:

_get_missing_rows(x: pd.DataFrame)
    Gather the indices of any rows that have missing values.
_get_map_and_rev_map(self, x: pd.DataFrame)
    Gets the encodings and the reverse encodings of categorical variables.
_compute_initial_imputations(self, x: pd.DataFrame,
                                 categorical: Iterable[Any])
    Computes and stores the initial imputation values for each feature
    in `x`.
_initial_impute(x: pd.DataFrame,
                    initial_imputations: Dict[Any, Union[str, np.float64]])
    Imputes the values of features using the mean or median for
    numerical variables; otherwise, uses the mode for imputation.
_add_unseen_categories(x, mappings)
    Updates mappings and reverse mappings based on any unseen categories
    encountered.
fit(self, x: pd.DataFrame, categorical: Iterable[Any] = None)
    Checks if the arguments are valid and initializes different class
    attributes.
transform(self, x: pd.DataFrame)
    Imputes all missing values in `x`.
fit_transform(self, x: pd.DataFrame, categorical: Iterable[Any] = None)
    Calls class methods `fit` and `transform` on `x`.

**Methods:**

- **__init__**(`self, clf: ComplexType, rgr: ComplexType, initial_guess: str, max_iter: int, early_stopping, progress_bar`) -> `None`
  >
  > **Parameters**:
  >
  > clf : estimator object, default=None.
  >     This object is assumed to implement the scikit-learn estimator api.
  > rgr : estimator object, default=None.
  >     This object is assumed to implement the scikit-learn estimator api.
  > max_iter : int, default=5
  >     Determines the number of iteration.
  > initial_guess : str, default=`median`
  >     If `mean`, initial imputation will be the mean of the features.
  >     If `median`, initial imputation will be the median of the features.
  > early_stopping : bool
  >     Determines if early stopping will be executed.
  >
  >
  > **Raises**:
  >
  > ValueError
  >     - If argument `clf` is not an estimator.
  >     - If argument `rgr` is not an estimator.
  >     - If argument `initial_guess` is not a str.
  >     - If argument `initial_guess` is neither `mean` nor `median`.
  >     - If argument `max_iter` is not an int.
  >     - If argument `early_stopping` is not a bool.

- **_get_missing_rows**(`x: pd.DataFrame`) -> `dict[ComplexType]`
  > Gather the indices of any rows that have missing values.
  >
  >
  > **Parameters**:
  >
  > x : pd.DataFrame of shape (n_samples, n_features)
  >     Dataset (features only) that needs to be imputed.
  >
  >
  > **Returns**:
  >
  > missing_rows : dict
  >     Dictionary containing features with missing values as keys,
  >     and their corresponding indices as values.

- **_get_map_and_rev_map**(`self, x: pd.DataFrame`) -> `ComplexType`
  > Gets the encodings and the reverse encodings of categorical
  > variables.
  >
  >
  > **Parameters**:
  >
  > x : pd.DataFrame of shape (n_samples, n_features)
  >     Dataset (features only) that needs to be encoded.
  >
  >
  > **Returns**:
  >
  > mappings : dict
  >     Dictionary containing the categorical variables as keys and
  >     their corresponding encodings as values.
  > rev_mappings : dict
  >     Dictionary containing the categorical variables as keys and
  >     their corresponding reverse encodings as values.

- **_compute_initial_imputations**(`self, x: pd.DataFrame, categorical: Iterable[Any]`) -> `dict[ComplexType]`
  > Computes and stores the initial imputation values for each feature
  > in `x`.
  >
  >
  > **Parameters**:
  >
  > x : pd.DataFrame of shape (n_samples, n_features)
  >     The dataset consisting solely of features that require imputation.
  > categorical : Iterable[Any]
  >     An iterable containing identifiers for all categorical features
  >     present in `x`.
  >
  >
  > **Raises**:
  >
  > ValueError
  >     - Raised if any feature specified in the `categorical` argument
  >     does not exist within the columns of `x`.
  >     - Raised if the `initial_guess` argument is provided and its
  >     value is neither 'mean' nor 'median'.

- **_initial_impute**(`x: pd.DataFrame, initial_imputations: dict[ComplexType]`) -> `pd.DataFrame`
  > Imputes the values of features using the mean or median for
  > numerical variables; otherwise, uses the mode for imputation.
  >
  >
  > **Parameters**:
  >
  > x : pd.DataFrame of shape (n_samples, n_features)
  >     Dataset (features only) that needs to be imputed.
  > initial_imputations : dict
  >     Dictionary containing initial imputation values for each feature.
  >
  >
  > **Returns**:
  >
  > x : pd.DataFrame of shape (n_samples, n_features)
  >     Imputed dataset (features only).

- **_add_unseen_categories**(`x, mappings`) -> `ComplexType`
  > Updates mappings and reverse mappings based on any unseen
  > categories encountered.
  >
  >
  > **Parameters**:
  >
  > x : pd.DataFrame of shape (n_samples, n_features)
  >     The dataset consisting solely of features that require imputation.
  > mappings : dict
  >     A dictionary mapping categorical variables to their encoded
  >     representations.
  >
  >
  > **Returns**:
  >
  > rev_mappings : dict
  >     A dictionary mapping categorical variables to their original
  >     values, effectively serving as the reverse of the `mappings`
  >     parameter.
  > updated_mappings : dict
  >     An updated dictionary reflecting the latest mappings between
  >     categorical variables and their encoded representations,
  >     incorporating any new categories encountered during processing.

- **_is_stopping_criterion_satisfied**(`self, pfc_score: list[float], nrmse_score: list[float]`) -> `bool`
  > Checks if stopping criterion satisfied. If satisfied, return True.
  > Otherwise, return False.
  >
  >
  > **Parameters**:
  >
  > pfc_score : list[float]
  >     Latest 2 PFC scores.
  > nrmse_score : list[float]
  >     Latest 2 NRMSE scores.
  >
  >
  > **Returns**:
  >
  > bool
  >     True, if stopping criterion satisfied.
  >     False, if stopping criterion not satisfied.

- **_update_progress**(`self`) -> `None`
  > No description available.

- **fit**(`self, x: pd.DataFrame, categorical: Iterable[Any]`) -> `None`
  > Checks if the arguments are valid and initializes different class
  > attributes.
  >
  >
  > **Parameters**:
  >
  > x : pd.DataFrame of shape (n_samples, n_features)
  >     Dataset (features only) that needs to be imputed.
  > categorical : Iterable[Any], default=None
  >     All categorical features of x.
  >
  >
  > **Returns**:
  >
  > x : pd.DataFrame of shape (n_samples, n_features)
  >     Reverse label-encoded dataset (features only).
  >
  >
  > **Raises**:
  >
  > ValueError
  >     - If argument `x` is not a pandas DataFrame, NumPy array, or
  >       list of lists.
  >     - If argument `categorical` is not a list of strings or NoneType.
  >     - If argument `categorical` is NoneType and has a length of
  >       less than one.
  >     - If there are inf values present in argument `x`.
  >     - If there are one or more columns with all rows missing.

- **transform**(`self, x: pd.DataFrame`) -> `pd.DataFrame`
  > Imputes all missing values in `x`.
  >
  >
  > **Parameters**:
  >
  > x : pd.DataFrame of shape (n_samples, n_features)
  >     Dataset (features only) that needs to be imputed.
  >
  >
  > **Returns**:
  >
  > pd.DataFrame
  >     - Before last imputation matrix, if stopping criterion is
  >       triggered.
  >     - Last imputation matrix, if all iterations are done.
  >
  >
  > **Raises**:
  >
  > NotFittedError
  >     If `MissForest` is not fitted.
  > ValueError
  >     If there are no missing values in `x`.

- **fit_transform**(`self, x: pd.DataFrame, categorical: Iterable[Any]`) -> `pd.DataFrame`
  > Calls class methods `fit` and `transform` on `x`.
  >
  >
  > **Parameters**:
  >
  > x : pd.DataFrame of shape (n_samples, n_features)
  >     Dataset (features only) that needs to be imputed.
  > categorical : Iterable[Any], default=None
  >     All categorical features of `x`.
  >
  >
  > **Returns**:
  >
  > pd.DataFrame of shape (n_samples, n_features)
  >     Imputed dataset (features only).

### def `default_learners` (missforest.py)

- **Arguments**: ``

- **Returns**: `tuple[ComplexType]`

Deterministic LightGBM learners (audit S16): seeded, single-threaded histogram construction. LightGBM is in the
optional `ml` extra (follow-up programme M4), so it is imported only when MissForest is used.

---

## multipage.py

_Loader entry points used by the pages. Parsing lives in :mod:`ingest.loaders`, the engine the import and the
replay use, so a file read here, by upload or by path, gives the table the replay will read again. Each table
carries the options it was parsed with in ``df.attrs["parse_options"]``, for the step that records it. This module
has no Streamlit dependency (pages render their own messages)._

**Imports**:
`__future__.annotations`, `hashlib`, `logging`, `typing.Any`, `pandas`, `core.state.ParseOptions`, `ingest.loaders.LoadResult`, `ingest.loaders.excel_sheet_names`, `ingest.loaders.file_kind`, `ingest.loaders.read_path`, `ingest.loaders.read_table`, `ingest.loaders.detect_encoding`, `utils.config_loader.create_empty_config`

### def `detect_encoding` (multipage.py)

- **Arguments**: `file_content: bytes, num_lines: int`

- **Returns**: `str`

Encoding of ``file_content`` (UTF-8, then Windows-1252, then charset-normalizer).

### def `_with_options` (multipage.py)

- **Arguments**: `result: LoadResult`

- **Returns**: `pd.DataFrame`

The table, carrying the options it was parsed with (recorded by the step that uses it, for the replay).

### def `load_dataframe` (multipage.py)

- **Arguments**: `uploaded_file: Any, options: ComplexType`

- **Returns**: `ComplexType`

DataFrame from a Streamlit upload or a file path, read with ``options`` (the ones the user chose in
``ui.components.csv_options``; detected when omitted). A multi-sheet workbook uploaded returns the sentinel
``{"__multi_sheet__": True, "sheet_names": [...]}`` so the caller can ask which sheets to load; a workbook read
by path gives its first sheet, as the replay does. A file that cannot be read by path raises.

### def `_load_excel_with_selection` (multipage.py)

- **Arguments**: `file_bytes: bytes, sheet_names: list, selected_sheets: list, merge_key: ComplexType`

- **Returns**: `ComplexType`

One sheet, or several merged on ``merge_key`` (outer join) or stacked when no key is given.

### def `get_file_hash` (multipage.py)

- **Arguments**: `uploaded_file: Any`

- **Returns**: `str`

MD5 of the upload's content.

---

## optional.py

_Optional dependency groups (the extras of pyproject.toml, follow-up programme M4): which are installed, and what to
tell the user when a feature needs one that is not.

- ``ai``: LLM connectors, document library and vector databases (variable map, documents, AI suggestions);

- ``ml``: MissForest imputation (LightGBM), UMAP and FAMD (prince);

- ``survival``: the proportional-hazards test and the concordance of the Cox model (lifelines).

The statistics engine, the data pipeline and the analysis pages need none of them._

**Imports**:
`__future__.annotations`, `importlib.util`, `functools.cache`

### def `available` (optional.py)

- **Arguments**: `module: str`

- **Returns**: `bool`

True when ``module`` can be imported (checked without importing it).

### def `extra_installed` (optional.py)

- **Arguments**: `extra: str`

- **Returns**: `bool`

No description available.

### def `method_available` (optional.py)

- **Arguments**: `method: str`

- **Returns**: `bool`

False for a method whose optional library is missing (MissForest, UMAP, FAMD).

### def `available_methods` (optional.py)

- **Arguments**: `methods: list[str]`

- **Returns**: `tuple[ComplexType]`

The methods that can run here, and those left out because their optional library is missing.

### def `missing_message` (optional.py)

- **Arguments**: `feature: str, extra: str`

- **Returns**: `str`

No description available.

### def `missing_methods_message` (optional.py)

- **Arguments**: `methods: list[str]`

- **Returns**: `str`

No description available.

---

## path_utils.py

_No module description._

**Imports**:
`logging`, `os`, `utils.logging_config.single_line`

### def `normalize_path` (path_utils.py)

- **Arguments**: `path_str: str`

- **Returns**: `str`

Normalizes a file path by replacing backslashes with forward slashes.
Crucial for handling Windows paths in a Linux (Docker) environment,
ensuring os.path.basename returns the correct filename.

### def `resolve_path` (path_utils.py)

- **Arguments**: `original_path: str, search_dirs: list`

- **Returns**: `ComplexType`

Attempts to resolve a file path.

1. Checks if original_path exists validly.
2. If not, extracts the filename and searches in `search_dirs`.

Args:
    original_path (str): The absolute or relative path from the trace.
    search_dirs (list, optional): List of directories to search in. Defaults to ['data'].

Returns:
    str | None: The valid path if found, or None if not found.

---

## session_scope.py

_The current user's storage scope, derived from the Streamlit session (None outside a session)._

**Imports**:
`__future__.annotations`, `core.workspace.UserScope`

### def `get_current_scope` (session_scope.py)

- **Arguments**: ``

- **Returns**: `ComplexType`

Scope of the signed-in user, or ``None`` when there is no Streamlit session or no user.

---

## statistics_utils.py

_Statistics Utilities.

Functions for statistical analysis including normality tests._

**Imports**:
`datetime`, `numpy`, `pandas`, `pyarrow`, `scipy.stats`

### def `sanitize_dataframe_for_arrow` (statistics_utils.py)

- **Arguments**: `df: ComplexType`

- **Returns**: `ComplexType`

Sanitize a pandas DataFrame so it can be converted to PyArrow without ArrowInvalid errors.

Ensures columns containing mixed types (e.g. numeric values mixed with Timestamp or Timedelta objects)
are safely converted to uniform types (such as strings), while preserving purely numeric and boolean dtypes.

Args:
    df: Input pandas DataFrame to sanitize.

Returns:
    pd.DataFrame | None: Sanitized DataFrame guaranteed to be compatible with PyArrow and Streamlit.

### def `normality_test` (statistics_utils.py)

- **Arguments**: `column, method`

- **Returns**: `None`

Perform specified normality test on a column and return the p-value.

Args:
    column: pandas Series with numeric data
    method: One of 'shapiro', 'dagostino', 'ks', or 'anderson'

Returns:
    float: p-value or np.nan if test fails

### def `show_test_guidelines` (statistics_utils.py)

- **Arguments**: ``

- **Returns**: `None`

Compatibility shim: the guide widgets live in :mod:`app_pages.statistics_guides`.

---

## visualization_utils.py

_No module description._

**Imports**:
`logging`, `re`, `numpy`, `pandas`, `plotly.colors`, `plotly.figure_factory`, `plotly.graph_objects`, `matplotlib.colors.to_rgba`, `scipy.stats`, `scipy.stats.zscore`

### def `compute_pairwise_stats` (visualization_utils.py)

- **Arguments**: `data_list, group_labels, test_type, correction_method, alpha`

- **Returns**: `None`

Pairwise comparisons for plot overlays, through the biostats engine so that brackets use the same
family and correction as the summary table (audit S6). Parametric = pairwise Welch t, non-parametric =
pairwise Mann-Whitney; the automatic strategy follows the Welch-first policy.

### def `compute_significant_pairs` (visualization_utils.py)

- **Arguments**: `data_list, test_type, correction_method, alpha, p_value_func`

- **Returns**: `None`

Wrapper for backward compatibility. Returns list of (i, j, p_adj).

### def `create_clustermap_streamlit` (visualization_utils.py)

- **Arguments**: `df, value_cols, sample_col, group_col, row_group_map, transformation, cluster_cols, cluster_rows, show_row_dendrogram, show_col_dendrogram, show_row_labels, show_col_labels, figsize, font_scale, cmap`

- **Returns**: `None`

Generates a clustermap figure for Streamlit using Plotly.
Supports wide format:
- value_cols: List of numeric columns (Features)

- sample_col: Column identifying samples (Columns of heatmap)

- group_col: Column for grouping/coloring samples

- row_group_map: Dict or Series mapping feature names to groups (for row coloring)

- figsize: Tuple (width, height) or None for auto-size (interpreted as relative scale for Plotly)

- font_scale: Scaling factor for fonts

- cmap: Colormap name

### def `add_significance_annotations` (visualization_utils.py)

- **Arguments**: `fig, data_list, significant_pairs, row, col`

- **Returns**: `None`

Adds significance brackets and stars to an existing figure.

### def `create_boxplot_with_significance_streamlit` (visualization_utils.py)

- **Arguments**: `df, group_col, value_col, significant_pairs, title, y_label, palette, width, height, font_family, font_size, template`

- **Returns**: `None`

Create box plot with significance stars for Streamlit.
Uses df, group_col, value_col instead of lists.
Significant pairs are pre-computed externally.

### def `adjust_color_to_rgba` (visualization_utils.py)

- **Arguments**: `color, transparency`

- **Returns**: `None`

Converts a color (hex or rgb format) to an RGBA string with the specified transparency.

### def `get_figure_column_names` (visualization_utils.py)

- **Arguments**: `fig, df`

- **Returns**: `None`

Extracts the column names or data-related attributes from a Plotly figure
and checks if they match any columns in the provided DataFrame (df).

### def `add_custom_hovertemplate` (visualization_utils.py)

- **Arguments**: `fig, df, selected_cols`

- **Returns**: `None`

Adds a custom hovertemplate to an existing Plotly figure, showing ``selected_cols`` on hover
(the page renders the column picker; this helper stays UI-free).

### def `convert_to_datetime_or_timedelta` (visualization_utils.py)

- **Arguments**: `df, col`

- **Returns**: `None`

No description available.

### def `resample_data` (visualization_utils.py)

- **Arguments**: `df, x_col, y_col, color_col, time_granularity, agg`

- **Returns**: `None`

No description available.

### def `create_qq_plot` (visualization_utils.py)

- **Arguments**: `data, title`

- **Returns**: `None`

Generates a Q-Q plot using Plotly to check for normality.

### def `create_bland_altman_plot` (visualization_utils.py)

- **Arguments**: `df, method1, method2, title`

- **Returns**: `None`

Bland-Altman plot with the bias, the limits of agreement (SD with ddof=1) and their confidence bands (audit S10).

### def `create_roc_curve` (visualization_utils.py)

- **Arguments**: `df, true_col, score_col, pos_label`

- **Returns**: `None`

Generates a ROC Curve.

### def `create_correlation_matrix_streamlit` (visualization_utils.py)

- **Arguments**: `data, cols, method, cmap, triangle, cluster_rows, cluster_cols, show_row_dendrogram, show_col_dendrogram`

- **Returns**: `None`

Generate an interactive correlation matrix using Plotly, with optional triangular masking and clustering.

### def `plot_nullity_matrix` (visualization_utils.py)

- **Arguments**: `df, row_id_col, cluster_cols, cluster_rows, show_dendrograms, show_row_labels, show_col_labels, figsize`

- **Returns**: `None`

Generates a Nullity Matrix (Clustermap style).

- Adapts figure size and font size to data density for readability.

- Zero gaps between components.

- Missing Count bar in %.

---
