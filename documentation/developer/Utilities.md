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
`typing.Any`, `typing.Union`, `sklearn.base.BaseEstimator`, `pandas`, `numpy`, `utils.miss_forest._errors.MultipleDataTypesError`

### def `_is_estimator` (_validate.py)

- **Arguments**: `estimator: Union[ComplexType]`

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
`streamlit`, `pandas`, `numpy`, `scipy.stats`, `statsmodels.stats.multitest`, `statsmodels.stats.multicomp.pairwise_tukeyhsd`, `utils.visualization_utils`

### def `render_analysis_configuration` (analysis_utils.py)

- **Arguments**: `df, numeric_cols, categorical_cols, binary_cols, group_col_options, default_group_col, target_col_options, key_prefix`

- **Returns**: `None`

Renders the Analysis Configuration expander.
Returns a dictionary with the selected configuration.

### def `analyze_variable` (analysis_utils.py)

- **Arguments**: `df, group_col, target, numeric_cols, categorical_cols, binary_cols, manual_test`

- **Returns**: `None`

Analyzes a single variable against the group column, selecting the appropriate test.

### def `recommend_statistical_test` (analysis_utils.py)

- **Arguments**: `df, group_col, target_col, numeric_cols, categorical_cols, binary_cols`

- **Returns**: `None`

No description available.

### def `perform_post_hoc` (analysis_utils.py)

- **Arguments**: `df, group_col, target_col, test_type`

- **Returns**: `None`

No description available.

### def `compute_cohens_d` (analysis_utils.py)

- **Arguments**: `group1, group2`

- **Returns**: `None`

No description available.

### def `compute_eta_squared` (analysis_utils.py)

- **Arguments**: `groups_data`

- **Returns**: `None`

No description available.

### def `compute_cramers_v` (analysis_utils.py)

- **Arguments**: `confusion_matrix`

- **Returns**: `None`

No description available.

### def `littles_mcar_test` (analysis_utils.py)

- **Arguments**: `df`

- **Returns**: `None`

No description available.

### def `scan_missingness_dependencies` (analysis_utils.py)

- **Arguments**: `df, target_cols, predictor_cols, test_preference, progress_callback`

- **Returns**: `None`

Systematically tests for dependencies between missingness of targets and values/missingness of predictors.

Args:
 test_preference: "Auto-Detect", "Force Parametric", "Force Non-Parametric"

Returns:
- pd.DataFrame with significant results.

---

## clustering_utils.py

_Clustering and Dimensionality Reduction Utilities

This module provides functions for:
- Dimensionality reduction (PCA, t-SNE, UMAP, FAMD)

- Clustering (K-Means, DBSCAN, Gaussian Mixture)

- Optimal cluster analysis (Elbow, Silhouette)

All heavy imports are done lazily to avoid slowing app startup._

**Imports**:
`numpy`, `pandas`, `typing.Tuple`, `typing.Optional`, `typing.Dict`, `typing.Any`, `typing.List`

### def `standardize_data` (clustering_utils.py)

- **Arguments**: `df: pd.DataFrame`

- **Returns**: `pd.DataFrame`

Standardize numeric columns to mean=0, std=1.

### def `prepare_data_for_clustering` (clustering_utils.py)

- **Arguments**: `df: pd.DataFrame, numeric_cols: List[str], categorical_cols: Optional[List[str]], handle_missing: str`

- **Returns**: `Tuple[ComplexType]`

Prepare data for clustering: handle missing values, standardize.

Returns:
 (prepared_df, valid_indices)

### def `run_pca` (clustering_utils.py)

- **Arguments**: `data: pd.DataFrame, n_components: int`

- **Returns**: `Tuple[ComplexType]`

Run PCA on the data.

Returns:
 (embeddings, info_dict with explained_variance)

### def `run_tsne` (clustering_utils.py)

- **Arguments**: `data: pd.DataFrame, n_components: int, perplexity: float, max_iter: int, max_samples: int`

- **Returns**: `Tuple[ComplexType]`

Run t-SNE on the data.
Note: t-SNE is slow for large datasets, so we sample if needed.

Returns:
 (embeddings, info_dict)

### def `run_umap` (clustering_utils.py)

- **Arguments**: `data: pd.DataFrame, n_components: int, n_neighbors: int, min_dist: float, max_samples: int`

- **Returns**: `Tuple[ComplexType]`

Run UMAP on the data.

Returns:
 (embeddings, info_dict)

### def `run_famd` (clustering_utils.py)

- **Arguments**: `data: pd.DataFrame, n_components: int`

- **Returns**: `Tuple[ComplexType]`

Run FAMD (Factor Analysis of Mixed Data) for mixed numeric/categorical data.

Returns:
 (embeddings, info_dict)

### def `fit_kmeans` (clustering_utils.py)

- **Arguments**: `embeddings: np.ndarray, n_clusters: int`

- **Returns**: `Tuple[ComplexType]`

Fit K-Means clustering.

Returns:
 (cluster_labels, info_dict)

### def `fit_dbscan` (clustering_utils.py)

- **Arguments**: `embeddings: np.ndarray, eps: float, min_samples: int`

- **Returns**: `Tuple[ComplexType]`

Fit DBSCAN clustering.

Returns:
 (cluster_labels, info_dict) - Note: -1 means noise

### def `fit_gaussian_mixture` (clustering_utils.py)

- **Arguments**: `embeddings: np.ndarray, n_components: int`

- **Returns**: `Tuple[ComplexType]`

Fit Gaussian Mixture Model.

Returns:
 (cluster_labels, info_dict with probabilities)

### def `optimal_k_analysis` (clustering_utils.py)

- **Arguments**: `embeddings: np.ndarray, k_range: range`

- **Returns**: `Dict[ComplexType]`

Compute elbow and silhouette metrics for different K values.

Returns:
 dict with inertias, silhouette_scores, optimal_k_elbow, optimal_k_silhouette

### def `compute_cluster_profiles` (clustering_utils.py)

- **Arguments**: `df: pd.DataFrame, labels: np.ndarray, numeric_cols: List[str], categorical_cols: Optional[List[str]]`

- **Returns**: `pd.DataFrame`

Compute mean/mode of variables per cluster.

Returns:
 DataFrame with cluster statistics

---

## config_loader.py

_No module description._

**Imports**:
`re`, `typing.Dict`, `typing.Any`

### def `create_empty_config` (config_loader.py)

- **Arguments**: ``

- **Returns**: `Dict[ComplexType]`

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

## custom_gemini.py

_No module description._

**Imports**:
`typing.Any`, `typing.List`, `typing.Optional`, `typing.Dict`, `time`, `random`, `langchain_core.language_models.chat_models.BaseChatModel`, `langchain_core.messages.BaseMessage`, `langchain_core.messages.HumanMessage`, `langchain_core.messages.AIMessage`, `langchain_core.messages.SystemMessage`, `langchain_core.outputs.ChatResult`, `langchain_core.outputs.ChatGeneration`, `langchain_core.callbacks.manager.CallbackManagerForLLMRun`, `google.genai`, `google.genai.types`, `langchain_core.embeddings.Embeddings`

### class `CustomGeminiEmbeddings` (custom_gemini.py)

Custom embedding class that uses the modern Google GenAI SDK (v1.0+).

**Methods:**

- **__init__**(`self, api_key: str, model: str`) -> `None`
 > No description available.

- **embed_documents**(`self, texts: List[str]`) -> `List[List[float]]`
 > Embed a list of documents.

- **embed_query**(`self, text: str`) -> `List[float]`
 > Embed a single query.

### class `CustomGeminiChat` (custom_gemini.py)

A custom LangChain wrapper for Google's new 'google-genai' SDK (v1.0+).
Bypasses 'langchain-google-genai' to avoid dependency hell.

**Methods:**

- **__init__**(`self, api_key: str, model: str, temperature: float`) -> `None`
 > No description available.

- **_llm_type**(`self`) -> `str`
 > No description available.

- **_generate**(`self, messages: List[BaseMessage], stop: Optional[List[str]], run_manager: Optional[CallbackManagerForLLMRun]`) -> `ChatResult`
 > No description available.

---

## data_analyzer.py

_Data Analyzer Module.

Provides the DataAnalyzer class for analyzing and categorizing DataFrame columns.
This is a consolidated module combining functionality from home.py and visualization_utils.py._

**Imports**:
`pandas`, `numpy`

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

- **__init__**(`self, df`) -> `None`
 > Initialize the analyzer with a DataFrame.
 >
 > Args:
 > df: pandas DataFrame to analyze

- **refresh**(`self, df`) -> `None`
 > Refresh the column metadata based on the latest DataFrame state.
 >
 > Args:
 > df: Updated pandas DataFrame

- **analyze_columns**(`self`) -> `None`
 > Analyze and categorize columns by data type.

- **_detect_date_columns**(`self`) -> `None`
 > Detect columns that are likely dates, with additional checks for accuracy.

- **_detect_binary_columns**(`self`) -> `None`
 > Detect columns with only two unique values, including numeric columns.
 >
 > Returns:
 > tuple: (all_binary, numeric_binary, non_numeric_binary)

- **_detect_low_cardinality_numeric_columns**(`self`) -> `None`
 > Detect numeric columns with <= 10 unique values to treat as categorical.

- **_detect_high_cardinality_cat_columns**(`self`) -> `None`
 > Detect categorical columns with many unique values.

- **get_suitable_columns**(`self, plot_type`) -> `None`
 > Get suitable columns for different plot types.
 >
 > Args:
 > plot_type: Type of plot (e.g., 'Histogram', 'Box Plot', 'Scatter Plot')
 >
 > Returns:
 > dict: Mapping of axis/parameter to suitable columns

---

## date_parser.py

_No module description._

**Imports**:
`pandas`, `numpy`

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

## export_utils.py

_Export Utilities.

Functions for exporting DataFrames to various formats (Excel, etc.)_

**Imports**:
`io.BytesIO`, `pandas`

### def `to_excel` (export_utils.py)

- **Arguments**: `df`

- **Returns**: `None`

Convert a DataFrame to Excel bytes for download.

Args:
 df: pandas DataFrame to export

Returns:
 bytes: Excel file as bytes

### def `to_excel_sheets` (export_utils.py)

- **Arguments**: `dataframes_dict`

- **Returns**: `None`

Converts multiple DataFrames into an Excel file with separate sheets.

Args:
 dataframes_dict: dict mapping sheet names to DataFrames

Returns:
 bytes: Excel file as bytes

---

## llm_utils.py

_LLM Utilities for Reliable Structured Output.

Provides robust JSON parsing, schema validation, and structured LLM invocation
with retry and repair capabilities._

**Imports**:
`json`, `re`, `logging`, `typing.TypeVar`, `typing.Type`, `typing.Optional`, `typing.Any`, `typing.Dict`, `pydantic.BaseModel`, `pydantic.ValidationError`

### class `StructuredOutputHelper` (llm_utils.py)

Helper class for managing structured LLM outputs with schemas.

Usage:
 helper = StructuredOutputHelper(llm)
 result, error = helper.invoke_with_schema(prompt, MySchema)

**Methods:**

- **__init__**(`self, llm, max_retries: int`) -> `None`
 > No description available.

- **invoke_with_schema**(`self, prompt: str, schema: Type[T], repair_on_fail: bool`) -> `tuple[ComplexType]`
 > Invoke LLM and parse response with schema validation.
 >
 > Args:
 > prompt: The prompt to send
 > schema: Pydantic model for validation
 > repair_on_fail: If True, attempt LLM-based repair on parse failure
 >
 > Returns:
 > Tuple of (parsed_model, error_message)

- **invoke_json**(`self, prompt: str, default: Any`) -> `tuple[ComplexType]`
 > Invoke LLM and parse as generic JSON without schema.
 >
 > Args:
 > prompt: The prompt to send
 > default: Default value if parsing fails
 >
 > Returns:
 > Tuple of (parsed_json, error_message)

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

- **Arguments**: `text: str, schema: Type[T], strict: bool`

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

## missforest.py

_No module description._

**Imports**:
`copy.deepcopy`, `typing.Union`, `numpy`, `pandas`, `lightgbm.LGBMClassifier`, `lightgbm.LGBMRegressor`, `utils.miss_forest._errors.NotFittedError`, `utils.miss_forest._validate._is_estimator`, `utils.miss_forest._validate._validate_single_datatype_features`, `utils.miss_forest._label_encoding._label_encoding`, `utils.miss_forest._label_encoding._rev_label_encoding`, `utils.miss_forest._metrics.pfc`, `utils.miss_forest._metrics.nrmse`, `typing.Any`, `typing.Tuple`, `typing.Iterable`, `typing.Dict`, `sklearn.base.BaseEstimator`, `tqdm.tqdm`, `warnings`

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

- **__init__**(`self, clf: Union[ComplexType], rgr: Union[ComplexType], initial_guess: str, max_iter: int, early_stopping, progress_bar`) -> `None`
 >
 > **Parameters**:
 >
 > clf : estimator object, default=None.
 > This object is assumed to implement the scikit-learn estimator api.
 > rgr : estimator object, default=None.
 > This object is assumed to implement the scikit-learn estimator api.
 > max_iter : int, default=5
 > Determines the number of iteration.
 > initial_guess : str, default=`median`
 > If `mean`, initial imputation will be the mean of the features.
 > If `median`, initial imputation will be the median of the features.
 > early_stopping : bool
 > Determines if early stopping will be executed.
 >
 >
 > **Raises**:
 >
 > ValueError
 > - If argument `clf` is not an estimator.
 > - If argument `rgr` is not an estimator.
 > - If argument `initial_guess` is not a str.
 > - If argument `initial_guess` is neither `mean` nor `median`.
 > - If argument `max_iter` is not an int.
 > - If argument `early_stopping` is not a bool.

- **_get_missing_rows**(`x: pd.DataFrame`) -> `Dict[ComplexType]`
 > Gather the indices of any rows that have missing values.
 >
 >
 > **Parameters**:
 >
 > x : pd.DataFrame of shape (n_samples, n_features)
 > Dataset (features only) that needs to be imputed.
 >
 >
 > **Returns**:
 >
 > missing_rows : dict
 > Dictionary containing features with missing values as keys,
 > and their corresponding indices as values.

- **_get_map_and_rev_map**(`self, x: pd.DataFrame`) -> `Union[ComplexType]`
 > Gets the encodings and the reverse encodings of categorical
 > variables.
 >
 >
 > **Parameters**:
 >
 > x : pd.DataFrame of shape (n_samples, n_features)
 > Dataset (features only) that needs to be encoded.
 >
 >
 > **Returns**:
 >
 > mappings : dict
 > Dictionary containing the categorical variables as keys and
 > their corresponding encodings as values.
 > rev_mappings : dict
 > Dictionary containing the categorical variables as keys and
 > their corresponding reverse encodings as values.

- **_compute_initial_imputations**(`self, x: pd.DataFrame, categorical: Iterable[Any]`) -> `Dict[ComplexType]`
 > Computes and stores the initial imputation values for each feature
 > in `x`.
 >
 >
 > **Parameters**:
 >
 > x : pd.DataFrame of shape (n_samples, n_features)
 > The dataset consisting solely of features that require imputation.
 > categorical : Iterable[Any]
 > An iterable containing identifiers for all categorical features
 > present in `x`.
 >
 >
 > **Raises**:
 >
 > ValueError
 > - Raised if any feature specified in the `categorical` argument
 > does not exist within the columns of `x`.
 > - Raised if the `initial_guess` argument is provided and its
 > value is neither 'mean' nor 'median'.

- **_initial_impute**(`x: pd.DataFrame, initial_imputations: Dict[ComplexType]`) -> `pd.DataFrame`
 > Imputes the values of features using the mean or median for
 > numerical variables; otherwise, uses the mode for imputation.
 >
 >
 > **Parameters**:
 >
 > x : pd.DataFrame of shape (n_samples, n_features)
 > Dataset (features only) that needs to be imputed.
 > initial_imputations : dict
 > Dictionary containing initial imputation values for each feature.
 >
 >
 > **Returns**:
 >
 > x : pd.DataFrame of shape (n_samples, n_features)
 > Imputed dataset (features only).

- **_add_unseen_categories**(`x, mappings`) -> `Union[ComplexType]`
 > Updates mappings and reverse mappings based on any unseen
 > categories encountered.
 >
 >
 > **Parameters**:
 >
 > x : pd.DataFrame of shape (n_samples, n_features)
 > The dataset consisting solely of features that require imputation.
 > mappings : dict
 > A dictionary mapping categorical variables to their encoded
 > representations.
 >
 >
 > **Returns**:
 >
 > rev_mappings : dict
 > A dictionary mapping categorical variables to their original
 > values, effectively serving as the reverse of the `mappings`
 > parameter.
 > updated_mappings : dict
 > An updated dictionary reflecting the latest mappings between
 > categorical variables and their encoded representations,
 > incorporating any new categories encountered during processing.

- **_is_stopping_criterion_satisfied**(`self, pfc_score: list[float], nrmse_score: list[float]`) -> `bool`
 > Checks if stopping criterion satisfied. If satisfied, return True.
 > Otherwise, return False.
 >
 >
 > **Parameters**:
 >
 > pfc_score : list[float]
 > Latest 2 PFC scores.
 > nrmse_score : list[float]
 > Latest 2 NRMSE scores.
 >
 >
 > **Returns**:
 >
 > bool
 > True, if stopping criterion satisfied.
 > False, if stopping criterion not satisfied.

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
 > Dataset (features only) that needs to be imputed.
 > categorical : Iterable[Any], default=None
 > All categorical features of x.
 >
 >
 > **Returns**:
 >
 > x : pd.DataFrame of shape (n_samples, n_features)
 > Reverse label-encoded dataset (features only).
 >
 >
 > **Raises**:
 >
 > ValueError
 > - If argument `x` is not a pandas DataFrame, NumPy array, or
 > list of lists.
 > - If argument `categorical` is not a list of strings or NoneType.
 > - If argument `categorical` is NoneType and has a length of
 > less than one.
 > - If there are inf values present in argument `x`.
 > - If there are one or more columns with all rows missing.

- **transform**(`self, x: pd.DataFrame`) -> `pd.DataFrame`
 > Imputes all missing values in `x`.
 >
 >
 > **Parameters**:
 >
 > x : pd.DataFrame of shape (n_samples, n_features)
 > Dataset (features only) that needs to be imputed.
 >
 >
 > **Returns**:
 >
 > pd.DataFrame
 > - Before last imputation matrix, if stopping criterion is
 > triggered.
 > - Last imputation matrix, if all iterations are done.
 >
 >
 > **Raises**:
 >
 > NotFittedError
 > If `MissForest` is not fitted.
 > ValueError
 > If there are no missing values in `x`.

- **fit_transform**(`self, x: pd.DataFrame, categorical: Iterable[Any]`) -> `pd.DataFrame`
 > Calls class methods `fit` and `transform` on `x`.
 >
 >
 > **Parameters**:
 >
 > x : pd.DataFrame of shape (n_samples, n_features)
 > Dataset (features only) that needs to be imputed.
 > categorical : Iterable[Any], default=None
 > All categorical features of `x`.
 >
 >
 > **Returns**:
 >
 > pd.DataFrame of shape (n_samples, n_features)
 > Imputed dataset (features only).

---

## multipage.py

_No module description._

**Imports**:
`streamlit`, `typing.Dict`, `typing.Any`, `typing.Callable`, `typing.Optional`, `typing.Union`, `dataclasses.dataclass`, `pandas`, `io`, `hashlib`, `manage.file_handling`, `os`, `pathlib.Path`, `requests`, `io.BytesIO`, `PIL.Image`, `base64`, `manage.transformation_manager.TransformationManager`, `app_pages.rag_sidebar.render_rag_sidebar`

### class `Page` (multipage.py)

No description available.

**Methods:**

- **__init__**(`self, title: str, function: Callable, icon: Union[ComplexType]`) -> `None`
 > No description available.

- **_process_icon**(`self, icon: Union[ComplexType]`) -> `Union[ComplexType]`
 > Process the icon, supporting emojis, file paths, URLs, and image bytes
 >
 > Args:
 > icon: Icon source (emoji, file path, URL, or image bytes)
 >
 > Returns:
 > Processed icon (emoji or PIL Image)

### class `MultiPageApp` (multipage.py)

No description available.

**Methods:**

- **__init__**(`self`) -> `None`
 > No description available.

- **initialize_session_state**(`self`) -> `None`
 > Initialize session state variables.

- **add_page**(`self, title: str, function: Callable, icon: Union[ComplexType]`) -> `None`
 > Add a new page to the app with a custom display label and icon.
 >
 > Args:
 > title (str): The title of the page
 > function (Callable): The function to render the page
 > icon (Union[str, Path], optional): An emoji or path to an image file. Defaults to None.

- **_render_sidebar_icon**(`self, icon: Union[ComplexType], page_name: str, is_widget: bool`) -> `str`
 > Render icons for the sidebar using Markdown or plain text for widgets.
 >
 > Args:
 > icon: Icon to render (emoji or PIL Image).
 > page_name: Name of the page.
 > is_widget: If True, return plain text for use in widgets like `checkbox`.
 >
 > Returns:
 > A string suitable for use in Markdown or plain text, depending on `is_widget`.

- **_render_icon**(`self, icon: Union[ComplexType], width: int`) -> `str`
 > Render icons for the page title (HTML-supported).
 >
 > Args:
 > icon: Icon to render (emoji or PIL Image)
 > width: Width of the icon in pixels
 >
 > Returns:
 > Rendered HTML for display

- **_handle_data_upload**(`self`) -> `None`
 > Handle data upload in sidebar.

- **run**(`self`) -> `None`
 > Run the multi-page app.

### def `detect_encoding` (multipage.py)

- **Arguments**: `file_content: bytes, num_lines: int`

- **Returns**: `str`

Detects the encoding of file bytes using charset-normalizer.

Parameters:
- file_content (bytes): The file content in bytes.

- num_lines (int): The number of lines to read from the file for encoding detection. Default is 100.

Returns:
- str: The detected encoding of the file.

Note:
The function attempts to detect the encoding by analyzing a portion of the file's content.
It reads the specified number of lines and uses the charset-normalizer library to determine the encoding.

### def `detect_separator` (multipage.py)

- **Arguments**: `text: str`

- **Returns**: `str`

Detect the most likely separator in a CSV text.

Parameters:
- text (str): Sample of the CSV content

Returns:
- str: Detected separator

### def `load_csv_with_separator` (multipage.py)

- **Arguments**: `file_bytes: bytes, encoding: str, num_lines: int`

- **Returns**: `Optional[pd.DataFrame]`

Load a CSV from bytes with automatic delimiter detection.

Parameters:
- file_bytes (bytes): The file content in bytes

- encoding (str): File encoding

- num_lines (int): Number of lines to analyze for delimiter detection

Returns:
- Optional[pd.DataFrame]: Loaded DataFrame or None if error

### def `load_dataframe` (multipage.py)

- **Arguments**: `uploaded_file`

- **Returns**: `Optional[pd.DataFrame]`

Load DataFrame from uploaded Streamlit file.

Parameters:
- uploaded_file: Streamlit UploadedFile object

Returns:
- Optional[pd.DataFrame]: Loaded DataFrame or None if error

### def `create_empty_config` (multipage.py)

- **Arguments**: ``

- **Returns**: `Dict[ComplexType]`

Create an empty configuration structure.

### def `get_file_hash` (multipage.py)

- **Arguments**: `uploaded_file`

- **Returns**: `str`

Generate a hash of the file content using hashlib.

### def `handle_data_upload` (multipage.py)

- **Arguments**: ``

- **Returns**: `None`

Handle data upload and perform file comparison.

---

## path_utils.py

_No module description._

**Imports**:
`os`, `streamlit`

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

## statistics_utils.py

_Statistics Utilities.

Functions for statistical analysis including normality tests._

**Imports**:
`streamlit`, `numpy`, `scipy.stats`

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

Displays a detailed explanation of the normality tests,
including algorithms, formulas, and recommendations.

### def `_show_overview_guide` (statistics_utils.py)

- **Arguments**: ``

- **Returns**: `None`

Display normality test overview and selection guide.

### def `_show_algorithm_details` (statistics_utils.py)

- **Arguments**: ``

- **Returns**: `None`

Display detailed algorithm explanations.

---

## visualization_utils.py

_No module description._

**Imports**:
`plotly.graph_objects`, `pandas`, `numpy`, `scipy.stats`, `itertools.combinations`, `re`, `matplotlib.colors.to_rgba`, `streamlit`, `statsmodels.api`, `sklearn.preprocessing.LabelEncoder`, `seaborn`, `matplotlib.pyplot`, `scipy.stats.zscore`, `matplotlib.patches.Patch`, `scipy.cluster.hierarchy`, `plotly.subplots.make_subplots`, `plotly.colors`, `plotly.figure_factory`, `statsmodels.stats.multitest`, `utils.data_analyzer.DataAnalyzer`, `plotly.graph_objects`, `plotly.figure_factory`, `pandas`, `numpy`, `streamlit`

### def `compute_pairwise_stats` (visualization_utils.py)

- **Arguments**: `data_list, group_labels, test_type, correction_method, alpha`

- **Returns**: `None`

Compute pairwise comparisons and return a detailed DataFrame.

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

- **Arguments**: `fig, df`

- **Returns**: `None`

Adds a custom hovertemplate to an existing Plotly figure, showing only selected columns on hover.

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

Generates a Bland-Altman plot.

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
