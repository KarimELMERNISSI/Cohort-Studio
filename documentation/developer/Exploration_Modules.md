# Exploration Modules Documentation

## corr_matrix.py

_No module description._

**Imports**:
`numpy`, `pandas`, `seaborn`, `matplotlib.pyplot`, `openpyxl.styles.PatternFill`, `openpyxl.styles.Font`, `matplotlib.colors.to_hex`, `plotly.express`, `scipy.cluster.hierarchy`, `scipy.spatial.distance`, `plotly.figure_factory`, `plotly.graph_objects`, `networkx`, `yfiles_graphs_for_streamlit.Node`, `yfiles_graphs_for_streamlit.Edge`, `yfiles_graphs_for_streamlit.EdgeStyle`, `yfiles_graphs_for_streamlit.DashStyle`, `yfiles_graphs_for_streamlit.NodeStyle`, `yfiles_graphs_for_streamlit.NodeShape`, `functools.cache`

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

### def `plotly_corr_network` (corr_matrix.py)

- **Arguments**: `data, targets, predictors, method, threshold_category, corr_method_name, node_color`

- **Returns**: `None`

Generate an interactive correlation network graph using Plotly and NetworkX.
Nodes represent variables, and edges represent correlations.
Edge styles/attributes depend on correlation strength.

Parameters:
- data (pd.DataFrame): Input data.

- targets (list): Target variables.

- predictors (list): Predictor variables.

- method (str): Correlation method.

- threshold_category (str): Minimum strength category to display edge ('Negligible', 'Weak', 'Moderate', 'Strong', 'Very Strong').

- corr_method_name (str): Display name for method.

- node_color (str): Color of nodes.

Returns:
- plotly.graph_objects.Figure: The network graph.

### def `get_yfiles_network_data` (corr_matrix.py)

- **Arguments**: `data, targets, predictors, method, threshold_category, correlation_matrix, focus_node`

- **Returns**: `None`

Generate nodes and edges for yFiles correlation network.

Parameters:
- data (pd.DataFrame): Input data.

- targets (list): Target variables.

- predictors (list): Predictor variables.

- method (str): Correlation method.

- threshold_category (str): Minimum strength category.

- correlation_matrix (pd.DataFrame): Pre-calculated correlation matrix (optional).

- focus_node (str): Variable to focus on (optional). Filters graph to connected component.

Returns:
- nodes (list): List of yFiles Node objects.

- edges (list): List of yFiles Edge objects.

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
file_name (str): Name of the Excel file to write to (default is 'correlation_matrix.xlsx').
method (str): Correlation method ('pearson', 'spearman', etc.).

Returns:
None

### def `_compute_corr_and_counts` (corr_matrix.py)

- **Arguments**: `data, targets, predictors, method`

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

## data_quality_auditor.py

_No module description._

**Imports**:
`pandas`, `numpy`, `pandas.api.types.is_numeric_dtype`, `pandas.api.types.is_datetime64_any_dtype`, `enrich.custom_metrics_and_filters`

### class `DataQualityAuditor` (data_quality_auditor.py)

Audits a pandas DataFrame to assess data quality across multiple dimensions:
- Completeness: Presence of missing values.

- Uniqueness: Presence of duplicate rows.

- Validity: Presence of outliers in numerical data.

- Consistency: Basic type consistency checks.

- Clinical Validity: Compliance with declared anomaly criteria.

**Methods:**

- **__init__**(`self, df, config`) -> `None`
  > No description available.

- **compute_completeness**(`self`) -> `None`
  > Calculates the percentage of non-missing values.

- **compute_uniqueness**(`self`) -> `None`
  > Calculates the percentage of unique rows.

- **compute_validity**(`self, method, params`) -> `None`
  > Calculates a validity score based on the absence of outliers in numerical columns.
  >
  >
  > **Parameters**:
  >
  > method : str, optional
  >     Detection method: 'iqr' (default), 'zscore', 'quantile',
  >     'Local Outlier Factor', 'Isolation Forest', 'DBSCAN'
  > params : dict, optional
  >     Method-specific parameters (e.g., multiplier, threshold, etc.)

- **compute_consistency**(`self`) -> `None`
  > Approximates consistency by checking if object columns could be converted to numeric or datetime.
  > If a column is 'object' but contains mostly numbers, it might be inconsistent formatting.

- **compute_clinical_validity**(`self`) -> `None`
  > Calculates a score based on declared anomaly criteria (if available).
  > Score = 100 - (% of rows triggering at least one anomaly).
  > Returns None if no anomaly rules are defined (shows as N/A in dashboard).

- **compute_uniformity**(`self`) -> `None`
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

- **run_audit**(`self, validity_method, validity_params`) -> `None`
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

---

## statistics.py

_No module description._

**Imports**:
`logging`, `streamlit`, `scipy.stats`, `numpy`, `pandas`, `subprocess`, `os`, `sklearn.ensemble.RandomForestRegressor`, `sklearn.ensemble.RandomForestClassifier`, `sklearn.impute.SimpleImputer`, `enrich.data_imputation`, `sklearn.impute.KNNImputer`, `seaborn`, `openpyxl.styles.PatternFill`, `openpyxl.styles.Font`, `matplotlib.colors.to_hex`, `datetime`

### def `get_statistics_dataframe` (statistics.py)

- **Arguments**: `df, _analyzer, nb_top_categories, exclude_columns, multi_index, super_column, qual_var_threshold, dataset_name, _db_manager`

- **Returns**: `None`

Creates a new DataFrame with statistics for each column.

Parameters:
- df (pd.DataFrame): The DataFrame for which to calculate statistics.

- nb_top_categories (int): The number of top categories to consider for categorical columns.

- exclude_columns (list): A list of column names to exclude from the statistics.

Returns:
- pd.DataFrame: A new DataFrame containing statistics for each column.

### def `calculate_group_stats` (statistics.py)

- **Arguments**: `group, group_name`

- **Returns**: `None`

Calculate descriptive statistics for a group.

Parameters:
- group (pandas.Series): The group for which to calculate statistics.

- group_name (str, Optional): The name of the group.

Returns:
- dict: Dictionary containing descriptive statistics with group name as prefix.

### def `calculate_group_stats_for_multiple_groups` (statistics.py)

- **Arguments**: `groups, group_names`

- **Returns**: `None`

Calculate descriptive statistics for multiple groups.

Parameters:
- groups (list of pandas.Series): A list of groups for which to calculate statistics.

- group_names (list, Optional): A list of group names. If None, generic names will be generated.

Returns:
- dict: Dictionary containing descriptive statistics for all groups.

### def `color_cells` (statistics.py)

- **Arguments**: `val`

- **Returns**: `None`

No description available.

### def `format_ltx_p_value` (statistics.py)

- **Arguments**: `val`

- **Returns**: `None`

Custom formatter for p-values.

### def `format_html_p_value` (statistics.py)

- **Arguments**: `val`

- **Returns**: `None`

Custom formatter for p-values.

### def `format_xlsx_p_value` (statistics.py)

- **Arguments**: `val`

- **Returns**: `None`

Custom formatter for p-values.

### def `produce_latex_table` (statistics.py)

- **Arguments**: `df, columns, filename, unicode_latex_mapping, title, footnotes, float_nb_digits, has_multi_index`

- **Returns**: `None`

Convert DataFrame to LaTeX table and write it to a .tex file.

Parameters:
    df (DataFrame): The DataFrame containing data.
    columns (list, optional): The list of columns to include in the table. Defaults to None.
    filename (str, optional): The name of the .tex file to write. Defaults to "output.tex".
    unicode_latex_mapping (dict, optional): A dictionary mapping Unicode characters to LaTeX representations. Defaults to None.
    title (str, optional): The title of the table. Defaults to None.
    footnotes (dict, optional): A dictionary mapping column or index names to their associated footnotes. Defaults to None.
    float_nb_digits (int, optional): Number of digits to display for floating-point numbers. Defaults to 2.

### def `produce_xlsx_table` (statistics.py)

- **Arguments**: `df, columns, filename, title, float_nb_digits, has_multi_index`

- **Returns**: `None`

Convert DataFrame to an excel table and write it to a .xlsx file.

Parameters:
    df (DataFrame): The DataFrame containing data.
    columns (list, optional): The list of columns to include in the table. Defaults to None.
    filename (str, optional): The name of the .tex file to write. Defaults to "output.xlsx".
    title (str, optional): The title of the table. Defaults to None.
    float_nb_digits (int, optional): Number of digits to display for floating-point numbers. Defaults to 2.

### def `produce_html_table` (statistics.py)

- **Arguments**: `df, columns, filename, title, float_nb_digits, has_multi_index`

- **Returns**: `None`

Convert DataFrame to an html table and write it to a .html file.

Parameters:
    df (DataFrame): The DataFrame containing data.
    columns (list, optional): The list of columns to include in the table. Defaults to None.
    filename (str, optional): The name of the .tex file to write. Defaults to "output.xlsx".
    title (str, optional): The title of the table. Defaults to None.
    float_nb_digits (int, optional): Number of digits to display for floating-point numbers. Defaults to 2.

### def `compile_latex_to_pdf` (statistics.py)

- **Arguments**: `tex_file, output_folder`

- **Returns**: `None`

Compile LaTeX to PDF using pdflatex and return the path of the created PDF file.

Parameters:
    tex_file (str): The path to the.tex file.
    output_folder (str): The path to the folder where the PDF file will be saved.
Returns:
    str: The path of the created PDF file.

### def `pdf_to_png` (statistics.py)

- **Arguments**: `pdf_file, output_dir`

- **Returns**: `None`

Convert PDF to PNG using pdftoppm with high resolution and return the path of the created PNG file.

Parameters:
    pdf_file (str): The path to the PDF file.
    output_dir (str, optional): The directory to save the output PNG file.
Returns:
    str: The path of the created PNG file.

### def `handle_missing_values` (statistics.py)

- **Arguments**: `df, variables, target_variable, strategy`

- **Returns**: `None`

Handle missing values in the DataFrame based on the specified strategy.

Parameters:
- df (pandas.DataFrame): The DataFrame containing the data.

- variables (list): A list of variable names for which to handle missing values.

- target_variable (str): The name of the target variable.

- strategy (str): Strategy to handle missing values. Options: 'remove', 'median', 'mean', 'missforest', 'knn', 'mode'.

Returns:
- pandas.DataFrame: DataFrame with missing values handled based on the specified strategy.

### def `chi2_test` (statistics.py)

- **Arguments**: `df, variables, target_variable, missing_strategy, test_name`

- **Returns**: `None`

Perform a chi-square test of independence between categorical variables and a target variable.

Parameters:
- df (pandas.DataFrame): The DataFrame containing the data.

- variables (list): A list of column names representing categorical variables.

- target_variable (str): The name of the target variable.

- missing_strategy (str): Strategy to handle missing values. Options: 'remove', 'impute_median', 'impute_mean', 'missforest', 'knn'.

Returns:
- dict: A dictionary containing the chi-square test results for each categorical variable.

### def `fisher_exact_test` (statistics.py)

- **Arguments**: `df, variables, target_variable, missing_strategy, test_name`

- **Returns**: `None`

Perform Fisher's exact test for binary variables in a DataFrame.

Parameters:
- df (pandas.DataFrame): The DataFrame containing the data.

- variables (list): A list of names of binary variables for which to perform the test.

- target_variable (str): The name of the target variable for the contingency table.

- missing_strategy (str): Strategy to handle missing values. Options: 'remove', 'impute_median', 'impute_mean', 'missforest', 'knn'.

Returns:
- dict: Dictionary containing the results of the Fisher's exact test for each variable.

### def `wilcoxon_rank_sum_test_with_descriptive` (statistics.py)

- **Arguments**: `df, variables, target_variable, control_group, target_group, missing_strategy, descriptive, group_names, test_name`

- **Returns**: `None`

Perform Wilcoxon rank sum test for continuous variables in a DataFrame.

'scipy.stats.mannwhitneyu' and 'scipy.stats.ranksums' will be applied depending on the context.
Mann-Whitney U test ('scipy.stats.mannwhitneyu'): The test statistic is based on the ranks of the observations from both samples.
It calculates the Mann-Whitney U statistic, which represents the probability of one randomly selected observation from one sample being greater than a randomly selected observation from the other sample.
Wilcoxon rank-sum test: It also calculates a U statistic, but it's slightly different from the Mann-Whitney U statistic. The Wilcoxon rank-sum test is a special case of the Mann-Whitney U test when the sample sizes are equal.

Parameters:
- df (pandas.DataFrame): The DataFrame containing the data.

- variables (list): A list of names of continuous variables for which to perform the test.

- target_variable (str): The name of the target variable that is used to infer target (True) and control (False) groups. It should be a binary variable.

- control_group (pandas.DataFrame, optional) and target_group (pandas.DataFrame, optional): Instead of using the target variable, it is possible to provide explicitly the control and target groups.

- missing_strategy (str): Strategy to handle missing values. Options: 'remove', 'impute_median', 'impute_mean', 'missforest', 'knn'.

- descriptive (bool): Whether to include descriptive statistics for control and target groups.

Returns:
- dict: Dictionary containing the results of the Wilcoxon rank sum test for each variable.

### def `t_test_with_descriptive` (statistics.py)

- **Arguments**: `df, variables, target_variable, control_group, target_group, missing_strategy, descriptive, group_names, test_name`

- **Returns**: `None`

Compute a t-test between two groups based on given variables.

Parameters:
- df (pandas.DataFrame): The DataFrame containing the data.

- variables (list): A list of names of continuous variables for which to perform the test.

- target_variable (str): The name of the target variable that is used to infer target (True) and control (False) groups. It should be a binary variable. If provided, the function will split the data into two groups based on this variable. Either this parameter or both control_group and target_group must be provided.

- control_group (pandas.DataFrame, optional): A DataFrame representing the control group. If provided, this group will be used as the control group in the t-test. This parameter must be used in conjunction with target_group.

- target_group (pandas.DataFrame, optional): A DataFrame representing the target group. If provided, this group will be used as the target group in the t-test. This parameter must be used in conjunction with control_group.

- missing_strategy (str): Strategy to handle missing values. Options: 'remove', 'impute_median', 'impute_mean', 'missforest', 'knn'. Default is 'remove'.

- descriptive (bool): Whether to include descriptive statistics for control and target groups. If True, the function will calculate and include descriptive statistics such as mean, standard deviation, quartiles, and min/max for each group. Default is False.

Returns:
- dict: Dictionary containing the results of the t-test for each variable. The dictionary keys are the variable names, and the values are dictionaries containing the t-statistic, p-value, and optionally, descriptive statistics for the control and target groups.

### def `kruskal_wallis_test_with_descriptive` (statistics.py)

- **Arguments**: `groups, variables, missing_strategy, descriptive, group_names, test_name`

- **Returns**: `None`

Perform Kruskal-Wallis test for continuous variables in a DataFrame.

Parameters:
- df (pandas.DataFrame): The DataFrame containing the data.

- variables (list): A list of names of continuous variables for which to perform the test.

- groups (list of pandas.DataFrame): List of DataFrames representing different groups.

- missing_strategy (str): Strategy to handle missing values. Options: 'remove', 'impute_median', 'impute_mean', 'missforest', 'knn'.

- descriptive (bool): Whether to include descriptive statistics for groups.

Returns:
- dict: Dictionary containing the results of the Kruskal-Wallis test for each variable.

### def `anova_test_with_descriptive` (statistics.py)

- **Arguments**: `groups, variables, missing_strategy, descriptive, group_names, test_name`

- **Returns**: `None`

Perform ANOVA test for continuous variables in a list of DataFrames representing different groups.

Parameters:
- df (pandas.DataFrame): The DataFrame containing the data.

- variables (list): A list of names of continuous variables for which to perform the test.

- groups (list of pandas.DataFrame): List of DataFrames representing different groups.

- missing_strategy (str): Strategy to handle missing values. Options: 'remove', 'impute_median', 'impute_mean', 'missforest', 'knn'.

- descriptive (bool): Whether to include descriptive statistics for groups.

Returns:
- dict: Dictionary containing the results of the ANOVA test for each variable.

### def `create_multiindex_dataframe` (statistics.py)

- **Arguments**: `result, group_keys, test_name, super_column`

- **Returns**: `None`

No description available.

### def `process_group` (statistics.py)

- **Arguments**: `info_group, df`

- **Returns**: `None`

Process a group defined by information in 'info_group' dictionary.

Parameters:
- info_group (dict): Dictionary containing information about the group.

- df (pandas.DataFrame): DataFrame containing the data.

Returns:
- pandas.DataFrame: DataFrame representing the processed group.

### def `make_test` (statistics.py)

- **Arguments**: `test_type, df, variables, target_variable, control_group, target_group, groups, missing_strategy, descriptive, group_names, test_name`

- **Returns**: `None`

Select the test to compute between two groups based on given variables.

Parameters:
- test_type(str): the name of the test to process

- df (pandas.DataFrame): The DataFrame containing the data.

- variables (list): A list of names of continuous variables for which to perform the test.

- target_variable (str, optional): The name of the target variable that is used to infer target (True) and control (False) groups. It should be a binary variable. If provided, the function will split the data into two groups based on this variable. Either this parameter or both control_group and target_group must be provided.

- control_group (pandas.DataFrame, optional): A DataFrame representing the control group. If provided, this group will be used as the control group in the t-test. This parameter must be used in conjunction with target_group.

- target_group (pandas.DataFrame, optional): A DataFrame representing the target group. If provided, this group will be used as the target group in the t-test. This parameter must be used in conjunction with control_group.

- missing_strategy (str): Strategy to handle missing values. Options: 'remove', 'impute_median', 'impute_mean', 'missforest', 'knn'. Default is 'remove'.

- descriptive (bool): Whether to include descriptive statistics for control and target groups. If True, the function will calculate and include descriptive statistics such as mean, standard deviation, quartiles, and min/max for each group. Default is False.

Returns:
- dict: Dictionary containing the results of the corresponding test for each variable.

### def `process_test` (statistics.py)

- **Arguments**: `df, test_name, test_info, config, multi_index`

- **Returns**: `None`

Process the statistical test based on the config file data.

Parameters:
- df (pandas.DataFrame): DataFrame containing the data.

- test_name (str): Name of the hypothesis test being processed.

- test_info (dict): Information about the hypothesis test, including test type, variables, groups, etc.

- config (dict, optional): Additional configuration parameters. Default is None.

Returns:
- tuple: A dictionary containing the results (statistics and p-value for each required and eligible variable) for the processed statistical test.

Raises:
- None

---

## stats_examples.py

_No module description._

---
