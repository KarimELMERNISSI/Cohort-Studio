# Enrichment Modules Documentation

## conditional_transformations.py

_No module description._

**Imports**:
`logging`, `ast`, `pandas`, `numpy`

### def `global_column_mapping` (conditional_transformations.py)

- **Arguments**: `unit_columns, gl_columns, mmoll_columns, unit_column_suffix, gl_column_suffix, mmoll_column_suffix`

- **Returns**: `None`

Creates a global column mapping for different units based on the provided column lists and suffixes.
This function is a part of a process for handling multiple columns with same names but different content.
The aim of this process is to remap column names based on column contents before using them for filling columns empty values with the transformation function (units or g/L or mmol/L).

Parameters:
- unit_columns (list): List of column names for units.

- gl_columns (list): List of column names for g/L units.

- mmoll_columns (list): List of column names for mmol/L units.

- unit_column_suffix (str): Suffix for columns related to units.

- gl_column_suffix (str): Suffix for columns related to g/L units.

- mmoll_column_suffix (str): Suffix for columns related to mmol/L units.

Returns:
- dict: A dictionary mapping original column names to their corresponding global column names.

### def `create_column_mapping` (conditional_transformations.py)

- **Arguments**: `column_tuples`

- **Returns**: `None`

Creates a mapping between original column names and their corresponding global column names.
This function is a part of a process for handling multiple columns with same names but different content.
The aim of this process is to remap column names based on column contents before using them for filling columns empty values with the transformation function (units or g/L or mmol/L).

Parameters:
    column_tuples (List[Tuple[List[str], str]]): A list of tuples, where each tuple contains a list of
        column names and a column suffix.

Returns:
    dict: A dictionary mapping original column names to their corresponding global column names.

### def `count_non_null_values` (conditional_transformations.py)

- **Arguments**: `df, unit_column, unit_value`

- **Returns**: `None`

Counts non-null values in specific columns of a DataFrame that match a given unit value in a specified unit column.

Parameters:
    df (pd.DataFrame): The DataFrame to search for non-null values.
    unit_column (str): The name of the unit column.
    unit_value (str, optional): The unit value to filter rows (default: 'mmol/L').

Returns:
    Union[pd.Series, None]: A Series containing counts of non-null values for each base column in the filtered rows,
        or None if no base columns are found.

### def `find_measures_with_unit_column` (conditional_transformations.py)

- **Arguments**: `columns, unit_column`

- **Returns**: `None`

Finds base columns associated with a given unit column in a list of column names.

Parameters:
    columns (List[str]): The list of column names to search.
    unit_column (str): The name of the unit column.

Returns:
    List[str]: A list of base column names associated with the unit column.

### def `find_unit_columns` (conditional_transformations.py)

- **Arguments**: `df, units, columns`

- **Returns**: `None`

Finds columns containing unit information in a DataFrame.

Parameters:
    df (pd.DataFrame): The DataFrame to search.
    units (List[str]): A list of unit strings to search for.
    columns (List[str]): A list of column names to search.

Returns:
    List[str]: A list of column names that contain unit information.

### def `retrieve_transform_column_names` (conditional_transformations.py)

- **Arguments**: `config`

- **Returns**: `None`

Retrieves column names involved in transformations from the configuration json data.

Parameters:
    config : The configuration json data containing transformation information.

Returns:
    List[str]: The list of column names involved in transformations.

### def `transform_column` (conditional_transformations.py)

- **Arguments**: `df, config`

- **Returns**: `None`

Apply column transformations to a DataFrame based on the provided configuration.

Parameters:
    df (pd.DataFrame): The DataFrame to be transformed.
    config (Dict[str, Any]): The configuration json data containing transformation information.

Returns:
    None

### def `create_condition_mask` (conditional_transformations.py)

- **Arguments**: `df, expression`

- **Returns**: `None`

Parse and evaluate a given expression within the context of a DataFrame,
generating a boolean mask based on the evaluation result.
This is a part of the process of applying transformation, it'll drive where the transformation is applied.

Parameters:
    df (pd.DataFrame): The DataFrame used as the context for the condition mask.
    expression (str): The expression to be parsed and evaluated.

Returns:
    pd.Series: A boolean mask indicating the evaluation result for each row in the DataFrame.

### def `create_transformation` (conditional_transformations.py)

- **Arguments**: `df, expression, condition_column, parameter_column, condition, conversion_rate`

- **Returns**: `None`

Create a transformation to apply to a DataFrame based on an expression, with optional conditional application and conversion rates.
This is a part of the process of applying transformation.

Parameters:
    df (pd.DataFrame): The DataFrame to be transformed.
    expression (str): The expression defining the transformation.
    condition_column (str): The name of the column used for conditional application of the transformation.
    parameter_column (str): The name of the column containing parameters for the transformation.
    condition (pd.Series): A boolean Series representing the condition for applying the transformation.
    conversion_rate (float): The conversion rate used in the transformation.

Returns:
    pd.Series: The transformed column as a Series.

### def `apply_transformation` (conditional_transformations.py)

- **Arguments**: `df, condition, parameter_column, transformed_column, conversion_rate, transformation_expr`

- **Returns**: `None`

Apply the created transformation to a DataFrame based on specified conditions and expressions.
This is a part of the process of applying transformation.

Parameters:
    df (pd.DataFrame): The DataFrame to be transformed.
    condition (pd.Series): A boolean Series representing the condition for applying the transformation.
    parameter_column (str): The name of the column containing parameters for the transformation.
    transformed_column (str): The name of the column to be transformed.
    conversion_rate (float): The conversion rate used in the transformation.
    transformation_expr (str): The expression defining the transformation.

Raises:
    ValueError: If an error occurs during the transformation.

Returns:
    None

---

## custom_metrics_and_filters.py

_No module description._

**Imports**:
`pandas`, `ast`, `logging`

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
`pandas`, `sklearn.base.TransformerMixin`, `sklearn.experimental.enable_iterative_imputer`, `sklearn.impute.IterativeImputer`, `lightgbm.LGBMClassifier`, `lightgbm.LGBMRegressor`, `utils.miss_forest.missforest.MissForest`, `sklearn.impute.SimpleImputer`, `sklearn.impute.KNNImputer`, `sklearn.preprocessing.OneHotEncoder`, `sklearn.preprocessing.StandardScaler`, `sklearn.compose.ColumnTransformer`, `sklearn.pipeline.Pipeline`, `streamlit`, `utils.data_analyzer.DataAnalyzer`

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

### def `detect_remainder_columns` (data_imputation.py)

- **Arguments**: `data, threshold`

- **Returns**: `None`

Detects columns to be included in the remainder based on distinct value criteria for object or categorical types
or if they are date-related columns.

Parameters:
- data: pd.DataFrame
    The input DataFrame.

- threshold: float, default=0.8
    The proportion of distinct values to non-null values to qualify as a remainder column.
    If the value is greater than 1, it defines the exact number of distinct values used as the threshold.

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

- **Arguments**: `data, remainder_columns`

- **Returns**: `None`

Reorder columns in a DataFrame by placing numerical, categorical, and remainder columns in a specific order.

Parameters:
- data: The DataFrame to reorder.

- remainder_columns: List of columns to keep at the end of the DataFrame.

Returns:
- data: The reordered DataFrame.

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

### def `get_imputer_in_progress` (data_imputation.py)

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

### def `get_imputer_classic` (data_imputation.py)

- **Arguments**: `numerical_imputation_method, categorical_imputation_method, cat_encoder, num_scaler, data, remainder_columns, remainder_strategy, remainder_threshold, debug`

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

- debug (bool): Whether to print debug information.

Returns:
- A ColumnTransformer object or a MissForestTransformer if using MissForest for both.

- A list of all output columns in the final transformed DataFrame.

---

## external_data.py

_No module description._

**Imports**:
`os`, `manage.file_handling`, `pandas`, `logging`

### def `add_data` (external_data.py)

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
