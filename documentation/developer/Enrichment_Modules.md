# Enrichment Modules Documentation

## custom_metrics_and_filters.py

_No module description._

**Imports**:
`logging`, `pandas`, `core.expressions.UnsafeExpressionError`, `core.expressions.evaluate`

### def `create_mask` (custom_metrics_and_filters.py)

- **Arguments**: `df, condition, mask_name`

- **Returns**: `None`

Create a mask based on conditions specified in the config file.

Parameters:
    df (pd.DataFrame): The DataFrame to create the mask from.
    condition (dict): A dictionary containing the conditions for creating the mask.
    mask_name (str): The name of the mask.

Returns:
    pd.Series or None: The mask as a boolean Series or None if an error occurs.

### def `create_expression_mask` (custom_metrics_and_filters.py)

- **Arguments**: `df, expression`

- **Returns**: `None`

Create a boolean mask based on the evaluation of the given expression using the DataFrame.

Parameters:
    df (pd.DataFrame): The DataFrame to evaluate the expression on.
    expression (str): The expression to evaluate.

Returns:
    pd.Series: A boolean Series representing the result of the expression evaluation.

Raises:
    ValueError: If an error occurs during expression evaluation.

### def `create_numeric_mask` (custom_metrics_and_filters.py)

- **Arguments**: `series, lower_bound, upper_bound, strategy, exclude_na`

- **Returns**: `None`

Create a boolean mask based on numeric conditions applied to a Series.

Parameters:
    series (pd.Series): The Series to create the mask from.
    lower_bound (float, optional): The lower bound for the numeric condition. Defaults to None.
    upper_bound (float, optional): The upper bound for the numeric condition. Defaults to None.
    strategy (str, optional): The strategy to apply when creating the mask. Can be 'include' if you want to keep the values inside the boundaries as outliers, 'exclude' if you want to keep the values outside the boundaries as outliers, or None if you use only one boundary. Defaults to None.
    exclude_na (bool, optional): Whether to exclude NaN values from the mask. Defaults to True.

Returns:
    pd.Series: A boolean Series representing the numeric mask.

Raises:
    ValueError: If an invalid strategy is provided.

### def `zip_masks` (custom_metrics_and_filters.py)

- **Arguments**: `df, config, mask_type`

- **Returns**: `None`

Zip masks created from the configuration with their respective names.

Parameters:
    df (pd.DataFrame): The DataFrame to create masks from.
    config (dict from json data file): The configuration containing mask definitions.
    mask_type (str, optional): The type of masks to create from the configuration. If None, it will take all the masks of the config data given in parameter.

Returns:
    List[Tuple[str, pd.Series]]: A list of tuples containing the mask names and their corresponding boolean Series.

Raises:
    ValueError: If an error occurs while creating a mask.

### def `tag_masks` (custom_metrics_and_filters.py)

- **Arguments**: `row`

- **Returns**: `None`

Create a comma-separated list of expert tests columns based on specified conditions in a row.

Parameters:
- row (pd.Series): A row in the DataFrame.

Returns:
- str: Comma-separated list of columns with expert test outliers.

### def `generate_computed_column` (custom_metrics_and_filters.py)

- **Arguments**: `df, config`

- **Returns**: `None`

Generate and append computed columns to a DataFrame based on the provided configuration and the DataFrame itself.
This is a part of the process of derivating new computed features.
Parameters:
    df (pd.DataFrame): The DataFrame to which computed columns will be added.
    config (Dict[str, Any]): The configuration dictionary containing computed column information.

Returns:
    None

### def `apply_computation` (custom_metrics_and_filters.py)

- **Arguments**: `df, column_name, computation_expr`

- **Returns**: `None`

Apply a computation expression to a DataFrame and create a new column based on the result.
This is a part of the process of derivating new computed features.

Parameters:
    df (pd.DataFrame): The DataFrame to which the computation will be applied.
    column_name (str): The name of the new column to be created.
    computation_expr (str): The computation expression to be applied.

Returns:
    None

### def `create_computation` (custom_metrics_and_filters.py)

- **Arguments**: `df, expression`

- **Returns**: `None`

Parse and evaluate a given expression within the context of a DataFrame.
This is a part of the process of derivating new computed features.

Parameters:
    df (pd.DataFrame): The DataFrame used as the context for the computation.
    expression (str): The expression to be parsed and evaluated.

Returns:
    Any: The result of the evaluated expression.

---

## data_imputation.py

_No module description._

**Imports**:
`pandas`, `sklearn.base.TransformerMixin`, `sklearn.compose.ColumnTransformer`, `sklearn.experimental.enable_iterative_imputer`, `sklearn.impute.IterativeImputer`, `sklearn.impute.KNNImputer`, `sklearn.impute.SimpleImputer`, `sklearn.pipeline.Pipeline`, `sklearn.preprocessing.OneHotEncoder`, `sklearn.preprocessing.StandardScaler`, `biostats._types.RANDOM_STATE`, `utils.data_analyzer.DataAnalyzer`, `utils.miss_forest.missforest.MissForest`

### class `MissForestTransformer` (data_imputation.py)

No description available.

**Methods:**

- **__init__**(`self, categorical_cols, debug, progress_bar`) -> `None`
  > A transformer wrapper for MissForest imputation for both numerical and categorical data.
  >
  >
  > **Parameters**:
  >
  > categorical_cols : list, default=None
  >     List of categorical column names for imputation. If None, assumes no categorical columns.
  > debug : bool, default=False
  >     If True, enables debug information output.

- **debug_print**(`self`) -> `None`
  > Helper function for controlled debug printing.

- **fit**(`self, X, y`) -> `None`
  > Fit the MissForest imputer on X.
  >
  >
  > **Parameters**:
  >
  > X : pd.DataFrame
  >     The input dataframe to fit the imputer.
  > y : Ignored
  >
  >
  > **Returns**:
  >
  > self : object
  >     Returns self.

- **transform**(`self, X`) -> `None`
  > Impute missing values in X based on the fitted MissForest imputer.
  >
  >
  > **Parameters**:
  >
  > X : pd.DataFrame
  >     The input dataframe to impute.
  >
  >
  > **Returns**:
  >
  > X_imputed : pd.DataFrame
  >     Dataframe with imputed values.

- **fit_transform**(`self, X, y`) -> `None`
  > Fit the MissForest imputer and transform X.
  >
  >
  > **Parameters**:
  >
  > X : pd.DataFrame
  >     The input dataframe to fit and transform.
  > progress_placeholder : st.empty, optional
  >     Streamlit placeholder for the progress bar. If None, a new progress bar is created.
  > y : Ignored
  >
  >
  > **Returns**:
  >
  > X_imputed : pd.DataFrame
  >     Dataframe with imputed values.

### def `_is_text` (data_imputation.py)

- **Arguments**: `series: pd.Series`

- **Returns**: `bool`

Text-like column: object, pandas string ('str' in pandas 3) or categorical dtype.

### def `detect_remainder_columns` (data_imputation.py)

- **Arguments**: `data, threshold, date_like_columns`

- **Returns**: `None`

Columns that should not be imputed or encoded: dates (true date/time dtypes and text columns
recognised as dates) and high-cardinality text columns such as identifiers.

Parameters:
- data: pd.DataFrame
    The input DataFrame.

- threshold: float, default=0.8
    The proportion of distinct values to non-null values to qualify as a remainder column.
    If the value is greater than 1, it defines the exact number of distinct values used as the threshold.

- date_like_columns: iterable of str, optional
    Text columns already recognised as dates (e.g. ``DataAnalyzer.date_cols``).

Returns:
- list: Names of columns to be included in the remainder, preserving the order in the input DataFrame.

### def `debug_print` (data_imputation.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `get_feature_names` (data_imputation.py)

- **Arguments**: `preprocessor, column_names, num_scaler, cat_encoder, debug`

- **Returns**: `None`

Retrieve the new feature names after applying transformations in the preprocessor.

Parameters:
- preprocessor: The fitted ColumnTransformer or Pipeline that has been applied to the data.

- column_names (list): List of original column names.

- num_scaler (bool): Whether numerical columns are scaled.

- cat_encoder (bool): Whether categorical columns are encoded.

- debug (bool): Whether to print debug information.

Returns:
- new_cols (list): List of transformed column names.

### def `reorder_data_columns` (data_imputation.py)

- **Arguments**: `data, numerical_columns, categorical_columns`

- **Returns**: `None`

Numerical columns first, then categorical ones, then every other column in its original order.
No column is ever dropped (true/false, date and text columns used to disappear here).

### def `partition_columns` (data_imputation.py)

- **Arguments**: `data, remainder_columns, analyzer`

- **Returns**: `None`

Assign every column to exactly one group: (numerical, categorical, passthrough).

- categorical: text and categorical columns, true/false columns, binary and low-cardinality numeric columns;

- numerical: the other numeric columns;

- passthrough: everything else (requested remainder columns, dates, identifiers, any other dtype), kept in
  the input order, which is the order in which a ColumnTransformer appends its remainder.

### def `get_imputer` (data_imputation.py)

- **Arguments**: `numerical_imputation_method, categorical_imputation_method, cat_encoder, num_scaler, data, remainder_columns, remainder_strategy, remainder_threshold, total_steps, progress_placeholder, debug`

- **Returns**: `None`

Returns the appropriate imputer based on the specified imputation methods for numerical and categorical columns.

Parameters:
- numerical_imputation_method (str): Imputation method for numerical columns ('mean', 'median', 'knn', 'missforest').

- categorical_imputation_method (str): Imputation method for categorical columns ('most_frequent', 'missforest').

- cat_encoder (bool): Whether to apply one-hot encoding to categorical columns.

- num_scaler (bool): Whether to scale numerical columns.

- data (pandas.DataFrame): Input DataFrame used to identify categorical and numerical columns.

- remainder_columns: list, default=None: Columns to pass through without transformation (e.g., ID columns), will be excluded from imputation, applying remainder strategy.
    NB : If it is set to 'auto', it will automatically guess the remainder columns based on your input data.

- remainder_threshold (float): Default (0.8), from 0 to 1, filter out non-numeric columns having a ratio distinct modalities / non empty rows higher than the fixed threshold if ramainder columns is set to 'auto'

- remainder_strategy (str): 'passthrough' (keep as it is) or 'drop' (remove from the final dataset)

- progress_placeholder: Streamlit empty container for showing the progress bar, default=None.

- debug (bool): Whether to print debug information.

Returns:
- A ColumnTransformer object or a MissForestTransformer if using MissForest for both.

- A list of all output columns in the final transformed DataFrame.

---

## external_data.py

_No module description._

**Imports**:
`logging`, `os`, `pandas`, `manage.file_handling`, `core.fingerprint.ROW_ID`, `core.fingerprint.extend_row_ids`

### def `add_data` (external_data.py)

- **Arguments**: `df, additional_df, left_id_names, right_id_names, additional_cols, strategy, conflict_resolution`

- **Returns**: `None`

Enrich the dataset by either merging on common identifiers (columns) or appending rows.

Rows of ``df`` keep their row id (``__row_id``, the index) through the merge; rows added by an outer or
right join, appended rows and repeats of a row (one-to-many join) get fresh ids. The merged frame's
``attrs["join_counts"]`` says how many rows came from the main dataset, were added, dropped or repeated.
Same parameters and result as :func:`_merge`.

### def `_merge` (external_data.py)

- **Arguments**: `df, additional_df, left_id_names, right_id_names, additional_cols, strategy, conflict_resolution`

- **Returns**: `None`

Enrich the dataset by either merging on common identifiers (columns) or appending rows.

Parameters:
    df (DataFrame): The main dataset to augment.
    additional_df (DataFrame): The additional dataset to merge or append.
    left_id_names (list): List of identifier column(s) in the main dataset.
    right_id_names (list): List of identifier column(s) in the additional dataset.
    additional_cols (list, optional): List of columns to include from the additional dataset. If None, all columns are included.
    strategy (str, optional): Enrichment strategy. Options are 'left', 'right', 'outer', 'inner', 'cross' for column merges, or 'rows' for row append. Defaults to 'left'.
    conflict_resolution (str, optional): Handle row conflicts based on identifiers with 'replace', 'keep', or 'ignore'. Defaults to 'keep' if None.

Returns:
    DataFrame: The enriched DataFrame or the original if an error occurs.

### def `process_data_enrichment` (external_data.py)

- **Arguments**: `df, additional_data_info`

- **Returns**: `None`

Process requested comparisons from JSON config files and write results to Excel.

Parameters:
- df (DataFrame): The dataset to augment.

- additional_data_info (dict): JSON configuration containing merging details.

Returns:
- DataFrame: The dataset df enriched with data from the additional data file. Returns the original df if an error occurs.

### def `process_data_enrichments_from_config` (external_data.py)

- **Arguments**: `df, data_file_name, config`

- **Returns**: `None`

Process requested data enrichments from JSON config files and write results to Excel.

Parameters:
- df (DataFrame): The dataset to augment.

- config (dict): JSON configuration containing comparison details.

Returns:
- DataFrame: The dataset df enriched with data from all the additional data files. Returns the original df if an error occurs.

---

## formulas.py

_Formulas for derived variables and targeted imputation (moved from the Enrich page, no Streamlit).

``evaluate_formula_safely`` validates a formula against the dataset's columns, rewrites date subtractions to
give days, and evaluates it through the whitelisted evaluator (``core.expressions``). The replay engine uses
it as well, so a derived variable is recomputed exactly as it was created._

**Imports**:
`ast`, `datetime`, `logging`, `re`, `typing.Any`, `numpy`, `pandas`, `core.expressions.evaluate_tree`, `utils.data_analyzer.DataAnalyzer`, `utils.date_parser.smart_parse_dates`, `utils.logging_config.single_line`

### def `validate_variable_name` (formulas.py)

- **Arguments**: `name, existing_columns`

- **Returns**: `None`

Validates the variable name and returns (is_valid, error_message).

### def `is_number` (formulas.py)

- **Arguments**: `token`

- **Returns**: `None`

Check if a string token represents a valid number.

**Parameters**:

token : str
    The token to check.

**Returns**:

bool
    True if the token can be converted to a float, False otherwise.

### def `validate_and_extract_columns_ast` (formulas.py)

- **Arguments**: `formula: str, dataset_columns: list`

- **Returns**: `tuple[ComplexType]`

Parses formula using Python AST to safely extract variables and constants
regardless of spacing.

### def `normalize_formula` (formulas.py)

- **Arguments**: `formula: str, column_mapping: dict`

- **Returns**: `str`

Replace column names in the formula with normalized names (e.g., var_1, var_2).

**Parameters**:

formula : str
    The computation formula to be normalized.
column_mapping : dict
    A mapping of original column names to normalized names.

**Returns**:

str
    The formula with normalized column names.

### def `extract_constant_value` (formulas.py)

- **Arguments**: `constant_token: str`

- **Returns**: `str`

Extract the value from a constant token like constant(val1) -> val1

### def `create_variables_dict` (formulas.py)

- **Arguments**: `dataset: pd.DataFrame, identified_columns: list[str], constant_tokens: list[str]`

- **Returns**: `dict[ComplexType]`

Create a dictionary mapping variable names to their corresponding Series
for both regular columns and constants.

**Parameters**:

dataset : pd.DataFrame
    The input dataset
identified_columns : List[str]
    List of valid column names used in the formula
constant_tokens : List[str]
    List of constant tokens found in the formula

**Returns**:

Dict[str, pd.Series]:
    Dictionary mapping variable names to their Series

### def `evaluate_formula_safely` (formulas.py)

- **Arguments**: `formula: str, variable_name: str, dataset: pd.DataFrame, return_series: bool`

- **Returns**: `tuple[ComplexType]`

Safely evaluate a computation formula to create a new variable in the dataset.

**Parameters**:

formula : str
    The computation formula entered by the user.
variable_name : str
    The name of the new variable to create.
dataset : pd.DataFrame
    The dataset containing columns to use in the formula.
return_series : bool, optional
    If True, returns the computed Series instead of a success message, and does not modify the dataset.

**Returns**:

tuple[bool, Any, pd.DataFrame]
    A tuple containing:
    - A boolean indicating success or failure.

    - A success message OR the computed Series (if return_series=True), OR an error message.

    - The modified DataFrame (or the original DataFrame on failure).

### def `numeric_summary` (formulas.py)

- **Arguments**: `values: pd.Series`

- **Returns**: `pd.DataFrame`

Classical and robust statistics of a numeric variable: N, missing, mean, SD, median, quartiles, min and max.

The median and the quartiles are not moved by a few extreme values (entry errors), the mean and the SD are;
infinite values (a division by zero) are counted apart and left out of the other statistics.

### def `far_outside` (formulas.py)

- **Arguments**: `values: pd.Series, k: float`

- **Returns**: `tuple[ComplexType]`

Rows whose value lies beyond Tukey's far-out fences, Q1 - k IQR and Q3 + k IQR (k = 3), with the fences:
values that are probably entry errors (a height in metres among centimetres, a weight of 0) rather than
the tail of the distribution. Infinite values are always included.

---

## imputation_service.py

_Single imputation of the session data (moved from the Enrich page, no Streamlit).

``apply_imputer`` fills every missing value of a table (mean, median, KNN, chained equations, missForest);
``impute_rows`` fills only the rows of the analysis population with an imputer fitted on those rows, so
excluded patients neither inform nor receive imputed values; ``fill_with_formula`` is the targeted
imputation of one column. The replay engine calls the same functions._

**Imports**:
`logging`, `dataclasses.dataclass`, `dataclasses.field`, `typing.Any`, `pandas`, `enrich.data_imputation`, `enrich.formulas.evaluate_formula_safely`

### class `ImputationOutcome` (imputation_service.py)

No description available.

### def `debug_print` (imputation_service.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `match_null_mask_to_encoded_columns` (imputation_service.py)

- **Arguments**: `initial_null_mask, output_cols, original_cols`

- **Returns**: `None`

Align initial null mask with output columns, including deducing one-hot encoded versions of categorical variables.
Ensures the mask matches the order of output_cols.

Parameters:
    initial_null_mask (pd.DataFrame): Original null mask for input data.
    output_cols (list): Ordered list of columns after imputation and encoding.
    original_cols (list): List of original columns from the input data.

Returns:
    pd.DataFrame: Aligned null mask matching the column order in output_cols.

### def `apply_imputer` (imputation_service.py)

- **Arguments**: `data, numerical_imputation_method, categorical_imputation_method, cat_encoder, num_scaler, remainder_columns, remainder_strategy, remainder_threshold, progress_placeholder, debug`

- **Returns**: `None`

Applies an imputer to a pandas DataFrame, with options for numerical and categorical imputations,
and handling of remainder columns. Also tracks imputed values.

Parameters:
    data (pd.DataFrame): The input data for imputation.
    numerical_imputation_method (str): Method for imputing numerical columns ('mean', 'median', 'knn', etc.).
    categorical_imputation_method (str): Method for imputing categorical columns ('most_frequent', 'missforest', etc.).
    cat_encoder (bool): Whether to apply one-hot encoding to categorical columns.
    num_scaler (bool): Whether to scale numerical columns.
    remainder_columns (list or str): Columns to pass through without transformation ('auto' for automatic detection).
    remainder_strategy (str): Strategy for handling remainder columns ('passthrough', 'drop', etc.).
    remainder_threshold (float): Threshold for automatic remainder column detection.
    progress_placeholder (object): Placeholder for showing progress in Streamlit.
    debug (bool): Flag to enable debug print statements.

Returns:
    pd.DataFrame: The imputed DataFrame with updated column names.
    pd.DataFrame: A mask DataFrame indicating which values were imputed.

### def `impute_rows` (imputation_service.py)

- **Arguments**: `frame: pd.DataFrame, rows: Any`

- **Returns**: `ImputationOutcome`

Impute the missing values of ``rows`` (row ids; every row when None) with an imputer fitted on them.

Rows outside ``rows`` are returned untouched. One-hot encoding, scaling and dropping remainder columns
change the columns of the table; they are refused when only part of the rows is imputed.

### def `fill_with_formula` (imputation_service.py)

- **Arguments**: `frame: pd.DataFrame, target: str, formula: str`

- **Returns**: `tuple[ComplexType]`

Fill the missing values of ``target`` with a formula of the other columns; returns (table, filled).

---

## transformation_logic.py

_No module description._

**Imports**:
`numpy`, `pandas`, `sklearn.preprocessing.LabelEncoder`, `sklearn.preprocessing.MinMaxScaler`, `sklearn.preprocessing.OneHotEncoder`, `sklearn.preprocessing.OrdinalEncoder`, `sklearn.preprocessing.StandardScaler`, `core.expressions.safe_format`

### def `apply_variable_transformation` (transformation_logic.py)

- **Arguments**: `dataframe, params`

- **Returns**: `None`

No description available.

---
