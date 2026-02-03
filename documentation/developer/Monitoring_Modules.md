# Monitoring Modules Documentation

## changes.py

_No module description._

**Imports**:
`os`, `pandas`, `logging`, `manage.file_handling`

### def `highlight_modifications` (changes.py)

- **Arguments**: `row, dataset, common_id_only_in_df1, common_id_only_in_df2, common_id_modified, common_id`

- **Returns**: `None`

Highlight rows where 'modification' column is True, or visit_id is only in df1 or df2.

### def `highlight_modifications_tuples` (changes.py)

- **Arguments**: `row, dataset, common_id_only_in_df1, common_id_only_in_df2, common_id_modified, common_id`

- **Returns**: `None`

Highlight rows where 'modification' column is True, or common_id is only in df1 or df2.

### def `keep_modifications` (changes.py)

- **Arguments**: `row, common_id_only_in_df1, common_id_only_in_df2, common_id`

- **Returns**: `None`

Filter out rows where 'modification' column is False and where common_id (tuple) is not in
common_id_only_in_df1 or common_id_only_in_df2.

### def `compare_dataframes` (changes.py)

- **Arguments**: `df1, df2, id_column, exception_list`

- **Returns**: `None`

Compare two DataFrames based on a common row identifier and identify modifications.
This is a part of the process of comparing two DataFrames.

Parameters:
- df1 (pd.DataFrame): First DataFrame.

- df2 (pd.DataFrame): Second DataFrame.

- id_column (str): Column name containing the IDs for comparison.

Returns:
- pd.DataFrame: DataFrame with differing values between df1 and df2 identified.

### def `process_comparison` (changes.py)

- **Arguments**: `comparison_name, comparison_info, config, exclude_cols`

- **Returns**: `None`

Process the comparison of two dataframes based on a common row identifier.
This is a part of the process of comparing two DataFrames.

Parameters:
    comparison_name (str): The name of the comparison.
    comparison_info (): Information about the comparison including input files, output file, folder paths, and common row identifier.

Returns:
    A tuple containing the results of the comparison, including DataFrame 1, DataFrame 2, common row identifier name, filtered comparison result,
    rows only in DataFrame 1, rows only in DataFrame 2, modified common row identifiers, and the output file path. Returns None if an error occurs.

Raises:
    None

### def `process_comparisons_from_config` (changes.py)

- **Arguments**: `config, exclude_cols_stats`

- **Returns**: `None`

Process requested comparisons from JSON config files and write results to Excel.

Parameters:
- config (dict): JSON configuration containing comparison details.

- exclude_cols_stats (list): List of columns to exclude from statistics or processing.

Returns:
- None

### def `process_comparison_st` (changes.py)

- **Arguments**: `comparison_name, df1, df2, common_id, common_cols, output_file`

- **Returns**: `None`

Compare two DataFrames based on a common row identifier and return comparison results.

Parameters:
    comparison_name (str): Name of the comparison task.
    df1 (dataframe):
    df2 (dataframe):
    common_id (list):
    common_cols (list, optional): Columns to exclude from both DataFrames. Defaults to None.

Returns:
    tuple: (df1, df2, common_id, comparison_result_filtered,
            rows_only_in_df1, rows_only_in_df2, modified_common_ids, output_file)

Notes:
    - Handles missing folders and file paths.

    - Logs steps and handles errors gracefully.

---

## outliers bckp.py

_No module description._

**Imports**:
`sys`, `os`, `pandas`, `numpy`, `sklearn.neighbors.LocalOutlierFactor`, `sklearn.ensemble.IsolationForest`, `sklearn.impute.SimpleImputer`, `sklearn.pipeline.Pipeline`

### def `find_outliers_zscore` (outliers bckp.py)

- **Arguments**: `series, threshold`

- **Returns**: `None`

Finds outliers in a numerical Series using Z-score.

Parameters:
- series (pandas.Series): The Series to find outliers in.

- threshold (float, optional): The Z-score threshold for identifying outliers.
  Values with absolute Z-score greater than this threshold are considered outliers.
  Default is 3.

Returns:
- pandas.Series: A boolean Series indicating outliers, with True for outliers and False otherwise.

This function computes the Z-score for each value in the input Series. For numerical columns
(int64 or float64 data types), it calculates the Z-score as (value - mean) / standard deviation.
Outliers are identified as values with absolute Z-score greater than the specified threshold.
For non-numeric columns, it returns an empty Series with dtype=bool.

### def `find_outliers_boxplot` (outliers bckp.py)

- **Arguments**: `series, tolerance`

- **Returns**: `None`

Finds potential outliers in a numerical Series using boxplot statistics.

Parameters:
- series (pandas.Series): The Series to find potential outliers in.

- tolerance (float, optional): The IQR tolerance for identifying outliers.
  Higher IQR tolerance leads to less sensitivity in detecting outliers. Increase this value if too much outliers are found with this method.
  Default is 1.5.

Returns:
- pandas.Series: A boolean Series indicating potential outliers, with True for outliers and False otherwise.

This function calculates the lower and upper bounds for potential outliers using the
interquartile range (IQR) method based on boxplot statistics. For numerical columns
(int64 or float64 data types), it calculates the lower bound as the first quartile (Q1)
minus tolerance times the IQR, and the upper bound as the third quartile (Q3) plus tolerance times the IQR.
Values outside this range are considered potential outliers. For non-numeric columns,
it returns an empty Series with dtype=bool.

### def `find_categorical_outliers` (outliers bckp.py)

- **Arguments**: `series, threshold`

- **Returns**: `None`

Finds potential outliers in a categorical Series based on the frequency of categories.

Parameters:
- series (pandas.Series): The categorical Series to find potential outliers in.

- threshold (int, optional): The threshold for identifying potential outliers based on z-scores.
  Categories with z-scores below the negative threshold are considered potential outliers.
  Default is 1.

Returns:
- pandas.Series: A Series with potential outlier categories marked as None, and non-outlier categories unchanged.

This function identifies potential outliers in a categorical Series based on the frequency
of categories. It calculates the z-scores for each category count and identifies categories
with z-scores below the negative threshold as potential outliers. The original Series is
mapped, marking potential outlier categories as None and leaving non-outlier categories unchanged.

### def `find_lof_outliers` (outliers bckp.py)

- **Arguments**: `data`

- **Returns**: `None`

Detects outliers in the given dataset using Local Outlier Factor (LOF).

Parameters:
- data (pandas.DataFrame): The input DataFrame containing numerical data.

Returns:
- numpy.ndarray: An array of boolean values indicating whether each data point is an outlier.

### def `compute_lof_scores` (outliers bckp.py)

- **Arguments**: `data`

- **Returns**: `None`

Computes the Local Outlier Factor (LOF) scores for the given dataset.

Parameters:
- data (pandas.DataFrame): The input DataFrame containing numerical data.

Returns:
- numpy.ndarray: An array of LOF scores for each data point in the DataFrame.

### def `find_isolation_forest_outliers` (outliers bckp.py)

- **Arguments**: `data, n_estimators, max_samples, contamination`

- **Returns**: `None`

Detects outliers in the given dataset using Isolation Forest algorithm.

Parameters:
- data (pandas.DataFrame): The input DataFrame containing numerical data.

Returns:
- numpy.ndarray: An array of boolean values indicating whether each data point is an outlier.

### def `compute_isolation_forest_scores` (outliers bckp.py)

- **Arguments**: `data, n_estimators, max_samples, contamination`

- **Returns**: `None`

Computes the Isolation Forest scores for the given dataset.

Parameters:
- data (pandas.DataFrame): The input DataFrame containing numerical data.

Returns:
- numpy.ndarray: An array of LOF scores for each data point in the DataFrame.

### def `tag_cat_columns` (outliers bckp.py)

- **Arguments**: `row`

- **Returns**: `None`

Create a comma-separated list of categorical columns with outliers in a row.

Parameters:
- row (pd.Series): A row in the DataFrame.

Returns:
- str: Comma-separated list of columns with outliers.

### def `tag_num_columns` (outliers bckp.py)

- **Arguments**: `row`

- **Returns**: `None`

Create a comma-separated list of numerical columns with outliers in a row.

Parameters:
- row (pd.Series): A row in the DataFrame.

Returns:
- str: Comma-separated list of columns with outliers.

---

## outliers.py

_No module description._

**Imports**:
`sys`, `os`, `pandas`, `numpy`, `sklearn.neighbors.LocalOutlierFactor`, `sklearn.cluster.DBSCAN`, `sklearn.ensemble.IsolationForest`, `sklearn.impute.SimpleImputer`, `sklearn.pipeline.Pipeline`, `enrich.data_imputation`, `sklearn.preprocessing.OneHotEncoder`, `sklearn.preprocessing.StandardScaler`, `sklearn.compose.ColumnTransformer`

### def `debug_print` (outliers.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `find_outliers_zscore` (outliers.py)

- **Arguments**: `series, threshold`

- **Returns**: `None`

Finds outliers in a numerical Series using Z-score.

Parameters:
- series (pandas.Series): The Series to find outliers in.

- threshold (float, optional): The Z-score threshold for identifying outliers.
  Values with absolute Z-score greater than this threshold are considered outliers.
  Default is 3.

Returns:
- pandas.Series: A boolean Series indicating outliers, with True for outliers and False otherwise.

This function computes the Z-score for each value in the input Series. For numerical columns
(int64 or float64 data types), it calculates the Z-score as (value - mean) / standard deviation.
Outliers are identified as values with absolute Z-score greater than the specified threshold.
For non-numeric columns, it returns an empty Series with dtype=bool.

### def `find_outliers_boxplot` (outliers.py)

- **Arguments**: `series, tolerance`

- **Returns**: `None`

Finds potential outliers in a numerical Series using boxplot statistics.

Parameters:
- series (pandas.Series): The Series to find potential outliers in.

- tolerance (float, optional): The IQR tolerance for identifying outliers.
  Higher IQR tolerance leads to less sensitivity in detecting outliers. Increase this value if too much outliers are found with this method.
  Default is 1.5.

Returns:
- pandas.Series: A boolean Series indicating potential outliers, with True for outliers and False otherwise.

This function calculates the lower and upper bounds for potential outliers using the
interquartile range (IQR) method based on boxplot statistics. For numerical columns
(int64 or float64 data types), it calculates the lower bound as the first quartile (Q1)
minus tolerance times the IQR, and the upper bound as the third quartile (Q3) plus tolerance times the IQR.
Values outside this range are considered potential outliers. For non-numeric columns,
it returns an empty Series with dtype=bool.

### def `find_categorical_outliers` (outliers.py)

- **Arguments**: `series, threshold`

- **Returns**: `None`

Finds potential outliers in a categorical Series based on the frequency of categories.

Parameters:
- series (pandas.Series): The categorical Series to find potential outliers in.

- threshold (int, optional): The threshold for identifying potential outliers based on z-scores.
  Categories with z-scores below the negative threshold are considered potential outliers.
  Default is 1.

Returns:
- pandas.Series: A Series with potential outlier categories marked as None, and non-outlier categories unchanged.

This function identifies potential outliers in a categorical Series based on the frequency
of categories. It calculates the z-scores for each category count and identifies categories
with z-scores below the negative threshold as potential outliers. The original Series is
mapped, marking potential outlier categories as None and leaving non-outlier categories unchanged.

### def `find_lof_outliers` (outliers.py)

- **Arguments**: `data, n_neighbors, contamination, numerical_imputation_method, categorical_imputation_method, remainder_columns, remainder_threshold, debug`

- **Returns**: `None`

Detects outliers in a DataFrame using Local Outlier Factor (LOF) with preprocessing.

Parameters:
- data: pd.DataFrame
    The input DataFrame containing both numerical and categorical features.

- n_neighbors: int, default=20
    Number of neighbors to use for LOF.

- contamination: str or float, default='auto'
    Proportion of outliers in the dataset.

- numerical_imputation_method: str, default='mean'
    Imputation method for numerical columns ('mean', 'knn', or 'missforest').

- categorical_imputation_method: str, default='most_frequent'
    Imputation method for categorical columns ('most_frequent' or 'missforest').

- remainder_columns: list, default=None
    Columns to pass through without transformation (e.g., ID columns).
remainder_threshold (float): Default (0.4), from 0 to 1, filter out non-numeric columns having a ratio distinct modalities / non empty rows higher than the fixed threshold if ramainder columns is set to 'auto'. If it is a 'higher than 1' number, then it defines a constant number which is directly used as threshold to evaluate the number of modalities.

- debug: bool, default=False
    If True, prints debug information.

Returns:
- outlier_tags: np.ndarray
    Boolean array where True indicates an outlier.

- lof_scores: np.ndarray
    An array of LOF scores for each data point.

- full_pipeline: sklearn.pipeline.Pipeline
    The fitted pipeline for reuse and inspection.

### def `find_isolation_forest_outliers` (outliers.py)

- **Arguments**: `data, n_estimators, max_samples, contamination, numerical_imputation_method, categorical_imputation_method, remainder_columns, remainder_threshold, debug`

- **Returns**: `None`

Detects outliers in a DataFrame using Isolation Forest with preprocessing.

Parameters:
- data: pd.DataFrame
    The input DataFrame containing both numerical and categorical features.

- n_estimators: int, default=100
    Number of base estimators in the ensemble.

- max_samples: str or int, default='auto'
    Number of samples to draw from data for training each base estimator.

- contamination: str or float, default='auto'
    Proportion of outliers in the data set.

- numerical_imputation_method: str, default='missforest'
    Imputation method for numerical columns ('mean', 'knn', or 'missforest').

- categorical_imputation_method: str, default='missforest'
    Imputation method for categorical columns ('most_frequent' or 'missforest').

- remainder_columns: list, default=None
    Columns to pass through without transformation (e.g., ID columns).

- remainder_threshold: float, default=0.8
    Threshold for identifying high cardinality categorical columns.

- debug: bool, default=False
    If True, prints debug information.

Returns:
- outlier_tags: np.ndarray
    Boolean array where True indicates an outlier.

- outlier_scores: np.ndarray
    An array of anomaly scores for each data point.

- full_pipeline: sklearn.pipeline.Pipeline
    The fitted pipeline for reuse and inspection.

### def `tag_cat_columns` (outliers.py)

- **Arguments**: `row`

- **Returns**: `None`

Create a comma-separated list of categorical columns with outliers in a row.

Parameters:
- row (pd.Series): A row in the DataFrame.

Returns:
- str: Comma-separated list of columns with outliers.

### def `tag_num_columns` (outliers.py)

- **Arguments**: `row`

- **Returns**: `None`

Create a comma-separated list of numerical columns with outliers in a row.

Parameters:
- row (pd.Series): A row in the DataFrame.

Returns:
- str: Comma-separated list of columns with outliers.

### def `find_dbscan_outliers` (outliers.py)

- **Arguments**: `data, eps, min_samples, numerical_imputation_method, categorical_imputation_method, remainder_columns, remainder_threshold, debug`

- **Returns**: `None`

Detects outliers in a DataFrame using DBSCAN with preprocessing.

Parameters:
- data: pd.DataFrame

- eps: float, default=0.5
    The maximum distance between two samples for one to be considered as in the neighborhood of the other.

- min_samples: int, default=5
    The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

Returns:
- outlier_tags: np.ndarray (True for outliers/noise)

- outlier_scores: np.ndarray (Cluster labels, where -1 is noise/outlier)

- full_pipeline: sklearn.pipeline.Pipeline

---
