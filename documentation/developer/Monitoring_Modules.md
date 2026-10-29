# Monitoring Modules Documentation

## changes.py

_No module description._

**Imports**:
`logging`, `os`, `pandas`, `ingest.loaders.read_path`

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

### def `_differs` (changes.py)

- **Arguments**: `a, b`

- **Returns**: `bool`

Two cells differ: one is missing and the other is not, or both hold values that differ.

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
