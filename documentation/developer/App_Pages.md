# App Pages Documentation

## about.py

_No module description._

**Imports**:
`streamlit`

### def `app` (about.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

---

## about.py

_About page._

**Imports**:
`app_pages.about.app`

---

## ai_quality.py

_AI quality page (administrators): RAG evaluation and embedding health._

**Imports**:
`app_pages.rag_monitoring.render_rag_monitoring`

---

## analysis_config.py

_Analysis configuration widgets of the Analyze page (moved out of utils so the engine side stays UI-free)._

**Imports**:
`streamlit`, `biostats.multiplicity.UI_OPTIONS`

### def `render_analysis_configuration` (analysis_config.py)

- **Arguments**: `df, numeric_cols, categorical_cols, binary_cols, group_col_options, default_group_col, target_col_options, key_prefix`

- **Returns**: `None`

Renders the Analysis Configuration expander.
Returns a dictionary with the selected configuration.

---

## analysis_diagnostic.py

_Diagnostic accuracy and method agreement tab of the Analyze page (audit S10)._

**Imports**:
`__future__.annotations`, `numpy`, `pandas`, `plotly.graph_objects`, `streamlit`, `biostats.agreement.bland_altman`, `biostats.diagnostic.confusion_at`, `biostats.diagnostic.delong_auc`, `biostats.diagnostic.diagnostic_accuracy`, `biostats.diagnostic.youden_cutoff`, `ui.components.error_card`, `ui.components.help_popover`, `ui.components.section`, `ui.components.status_chip`

### def `_ci_text` (analysis_diagnostic.py)

- **Arguments**: `ci`

- **Returns**: `str`

No description available.

### def `_roc_figure` (analysis_diagnostic.py)

- **Arguments**: `y_true: pd.Series, y_score: pd.Series, pos_label, auc: dict, cutoff: ComplexType`

- **Returns**: `go.Figure`

No description available.

### def `_bland_altman_figure` (analysis_diagnostic.py)

- **Arguments**: `ba, label_a: str, label_b: str`

- **Returns**: `go.Figure`

No description available.

### def `run_diagnostic_accuracy` (analysis_diagnostic.py)

- **Arguments**: `df: pd.DataFrame, analyzer`

- **Returns**: `None`

No description available.

---

## analyze.py

_Analyze page: hypothesis tests, epidemiological measures, ANCOVA, power, z-scores._

**Imports**:
`app_pages.epidemiology.app`, `ui.components.page_header`

---

## data_enrichment.py

_No module description._

**Imports**:
`ast`, `io`, `json`, `logging`, `os`, `re`, `datetime.datetime`, `typing.Any`, `numpy`, `pandas`, `streamlit`, `matplotlib.pyplot`, `rapidfuzz.fuzz`, `enrich.data_imputation`, `enrich.external_data`, `manage.db_manager.DBManager`, `manage.transformation_manager.TransformationManager`, `utils.data_analyzer.DataAnalyzer`, `utils.date_parser.smart_parse_dates`, `core.expressions.evaluate_tree`, `core.expressions.safe_format`, `utils.data_paths.get_traces_dir`, `utils.session_scope.get_current_scope`, `utils.multipage._load_excel_with_selection`, `utils.multipage.get_file_hash`, `utils.multipage.load_dataframe`, `ui.components.no_dataset`, `utils.export_utils.to_excel`

### def `_workspace_path` (data_enrichment.py)

- **Arguments**: `path_text: str`

- **Returns**: `str`

Only files inside the signed-in user's workspace may be referenced by path (audit E4).

### def `_artifact_dir` (data_enrichment.py)

- **Arguments**: `session_id: str`

- **Returns**: `str`

No description available.

### def `_handle_multisheet_inline` (data_enrichment.py)

- **Arguments**: `uploaded_file, key_prefix: str`

- **Returns**: `ComplexType`

Handle multi-sheet Excel files inline with a selection UI.

Persists the loaded DataFrame in session state so it survives Streamlit re-runs.
Returns the loaded DataFrame, or None if the user hasn't made a selection yet.

### def `handle_main_data_upload` (data_enrichment.py)

- **Arguments**: ``

- **Returns**: `ComplexType`

Handle the upload or path input for the main dataset.

### def `handle_enrichment_data_upload` (data_enrichment.py)

- **Arguments**: ``

- **Returns**: `ComplexType`

Handle the upload or path input for the enrichment dataset.

### def `configure_enrichment` (data_enrichment.py)

- **Arguments**: `main_df: pd.DataFrame, enrichment_df: pd.DataFrame, enrichment_file_path: str`

- **Returns**: `ComplexType`

Configure and perform the data enrichment.

### def `display_category_box` (data_enrichment.py)

- **Arguments**: `title, columns`

- **Returns**: `None`

Display a box containing column names for a specific category.

**Parameters**:

title : str
    The title of the category box.
columns : list
    List of column names to display.

### def `display_results` (data_enrichment.py)

- **Arguments**: `enriched_df: pd.DataFrame, title: str, key_base`

- **Returns**: `None`

Display the results of the enrichment process.

### def `display_external_data_results` (data_enrichment.py)

- **Arguments**: `enriched_df: pd.DataFrame, filtered_main_data: pd.DataFrame`

- **Returns**: `None`

Display the results of the enrichment process.

### def `display_new_variables_results` (data_enrichment.py)

- **Arguments**: `enriched_df: pd.DataFrame, main_data: pd.DataFrame`

- **Returns**: `None`

Display the results of the enrichment process.

### def `display_imputation_results` (data_enrichment.py)

- **Arguments**: `imputed_data: pd.DataFrame, main_data: pd.DataFrame, cols_per_row: int`

- **Returns**: `None`

Display the results of the enrichment process.

### def `save_enrichment_config` (data_enrichment.py)

- **Arguments**: `enrichment_name: str, strategy: str, input_file_folder: str, input_file: str, identifier: str, left_identifier: str, right_identifier: str, additional_columns: str, conflict_resolution: str`

- **Returns**: `None`

Save the enrichment configuration to session state.

### def `calculate_prefix_score` (data_enrichment.py)

- **Arguments**: `input_token: str, column_name: str`

- **Returns**: `float`

Calculate a score based on how well the input matches the beginning of the column name.

**Parameters**:

input_token : str
    The user's input token
column_name : str
    The column name to compare against

**Returns**:

float
    Prefix matching score between 0 and 1

### def `calculate_similarity_score` (data_enrichment.py)

- **Arguments**: `input_token: str, column_name: str`

- **Returns**: `int`

Calculate a similarity score between input token and column name with
prefix-weighted scoring and length-aware adjustments.

**Parameters**:

input_token : str
    The user's input token
column_name : str
    The column name to compare against

**Returns**:

int
    Adjusted similarity score

### def `tokenize_formula` (data_enrichment.py)

- **Arguments**: `formula: str`

- **Returns**: `list[str]`

Tokenizes the formula, treating quoted variables as single tokens and splitting others by operators.

**Parameters**:

formula : str
    The formula input by the user.

**Returns**:

List[str]
    The list of tokens extracted from the formula.

### def `suggest_columns` (data_enrichment.py)

- **Arguments**: `computation_formula: str, column_names: list[str], threshold: int, max_suggestions: int`

- **Returns**: `list[tuple[ComplexType]]`

Suggest column names based on similarity to the input formula with enhanced
prefix matching, including handling quoted variable names as unique tokens.

**Parameters**:

computation_formula : str
    The formula input by the user.
column_names : List[str]
    List of available column names in the dataset.
threshold : int, optional
    Minimum similarity score to include in suggestions (default: 60).
max_suggestions : int, optional
    Maximum number of suggestions to return (default: 5).

**Returns**:

List[Tuple[str, int]]
    List of tuples containing (column_name, similarity_score).

### def `update_formula_callback` (data_enrichment.py)

- **Arguments**: `col, formula_key`

- **Returns**: `None`

Callback to update formula when a suggestion is clicked.

### def `display_column_suggestions` (data_enrichment.py)

- **Arguments**: `suggestions, formula_key`

- **Returns**: `None`

Displays column suggestions as buttons. Clicking a button replaces the last token
in the formula input with the selected column name.

### def `validate_variable_name` (data_enrichment.py)

- **Arguments**: `name, existing_columns`

- **Returns**: `None`

Validates the variable name and returns (is_valid, error_message).

### def `is_number` (data_enrichment.py)

- **Arguments**: `token`

- **Returns**: `None`

Check if a string token represents a valid number.

**Parameters**:

token : str
    The token to check.

**Returns**:

bool
    True if the token can be converted to a float, False otherwise.

### def `validate_and_extract_columns_ast` (data_enrichment.py)

- **Arguments**: `formula: str, dataset_columns: list`

- **Returns**: `tuple[ComplexType]`

Parses formula using Python AST to safely extract variables and constants
regardless of spacing.

### def `normalize_formula` (data_enrichment.py)

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

### def `extract_constant_value` (data_enrichment.py)

- **Arguments**: `constant_token: str`

- **Returns**: `str`

Extract the value from a constant token like constant(val1) -> val1

### def `create_variables_dict` (data_enrichment.py)

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

### def `evaluate_formula_safely` (data_enrichment.py)

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

### def `generate_stats` (data_enrichment.py)

- **Arguments**: `temp_main_data, variable_name`

- **Returns**: `None`

No description available.

### def `define_new_variables` (data_enrichment.py)

- **Arguments**: `main_data`

- **Returns**: `None`

Enhanced version of define_new_variables with improved preview and validation.

### def `merge_transformed_columns` (data_enrichment.py)

- **Arguments**: `main_data, transformed_data, handle_duplicates`

- **Returns**: `None`

Merge transformed_data into main_data securely.

Parameters:
- main_data (pd.DataFrame): The original DataFrame.

- transformed_data (pd.DataFrame): The DataFrame with new columns.

- handle_duplicates (str): How to handle duplicate column names.
                           Options: "replace" or "ignore".

Returns:
- pd.DataFrame: The merged DataFrame.

### def `compute_transformations` (data_enrichment.py)

- **Arguments**: `dataframe`

- **Returns**: `None`

No description available.

### def `debug_print` (data_enrichment.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `match_null_mask_to_encoded_columns` (data_enrichment.py)

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

### def `apply_imputer` (data_enrichment.py)

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

### def `save_snapshot` (data_enrichment.py)

- **Arguments**: `df, step_name, base_dir`

- **Returns**: `None`

Saves a snapshot of the dataframe in the user's workspace and returns the path.

### def `extract_column_stats` (data_enrichment.py)

- **Arguments**: `df, columns`

- **Returns**: `None`

Extracts summary statistics (Min, Max, Median) and potential units for selected columns.
Used to give context to the RAG model for unit conversion logic.

### def `app` (data_enrichment.py)

- **Arguments**: ``

- **Returns**: `None`

Main application function for the Data Enrichment page.

Handles:
1. External Data Upload and Enrichment
2. Targeted Imputation (Manual/AI-assisted)
3. Global Imputation (MICE, KNN, etc.)
4. Dataset Versioning and Persistence

---

## data_insight.py

_No module description._

**Imports**:
`json`, `os`, `re`, `pandas`, `streamlit`, `core.workspace.UserScope`, `graph.builders.taxonomy.CATEGORY_LABELS`, `graph.builders.taxonomy.RELATION_LABELS`, `graph.builders.taxonomy.build_taxonomy_graph`, `graph.builders.taxonomy.make_formula_id`, `graph.builders.taxonomy.parse_formula_id`, `graph.model.LayoutHint`, `graph.model.RenderOptions`, `graph.ui.current_theme`, `graph.ui.render_graph`, `graph.ui.render_graph_exports`, `graph.ui.renderer_selector`, `utils.clustering_utils.compute_cluster_profiles`, `utils.clustering_utils.fit_dbscan`, `utils.clustering_utils.fit_gaussian_mixture`, `utils.clustering_utils.fit_kmeans`, `utils.clustering_utils.prepare_data_for_clustering`, `utils.clustering_utils.run_famd`, `utils.clustering_utils.run_pca`, `utils.clustering_utils.run_tsne`, `utils.clustering_utils.run_umap`, `ui.components.no_dataset`

### def `get_user_taxonomy_dir` (data_insight.py)

- **Arguments**: `username`

- **Returns**: `None`

Taxonomy directory of the user's workspace (no shared fallback; audit E4).

### def `get_taxonomy_versions` (data_insight.py)

- **Arguments**: `username`

- **Returns**: `None`

Returns list of (version_int, filepath) sorted descending.

### def `repair_taxonomy_links` (data_insight.py)

- **Arguments**: `taxonomy, formulas_registry`

- **Returns**: `None`

Ensures taxonomy variables have correct related_formula_ids based on registry.
Run this on load to fix desynchronized files.

### def `load_and_repair_taxonomy` (data_insight.py)

- **Arguments**: `target_path, rag_manager`

- **Returns**: `None`

Robust loading function:
1. Loads Taxonomy & Formulas
2. Repairs Links immediately
3. Updates Session State
4. Syncs with RAG Manager (if active)

### def `app` (data_insight.py)

- **Arguments**: ``

- **Returns**: `None`

Main application function for the Data Insight (Knowledge Graph) page.

Handles:
1. Loading and saving taxonomy versions.
2. Visualizing variables and formulas as a graph.
3. Generating new taxonomies using RAG.
4. Enriching existing taxonomies with external knowledge.

---

## data_monitoring.py

_No module description._

**Imports**:
`json`, `os`, `io.BytesIO`, `typing.Any`, `numpy`, `pandas`, `streamlit`, `scipy.stats`, `monitor.changes`, `monitor.outliers`, `utils.clustering_utils`, `app_pages.data_quality_dashboard.render_dashboard`, `enrich.custom_metrics_and_filters`, `manage.db_manager.DBManager`, `manage.transformation_manager.TransformationManager`, `utils.config_loader.create_empty_config`, `utils.config_loader.transform_expression`, `utils.data_analyzer.DataAnalyzer`, `utils.date_parser.smart_parse_dates`, `utils.multipage.load_dataframe`, `utils.session_scope.get_current_scope`, `ui.components.confirm`, `ui.components.no_dataset`, `utils.export_utils.to_excel`

### class `OutlierHandler` (data_monitoring.py)

Helper class to detect and handle outliers using various methods

**Methods:**

- **__init__**(`self, df, config`) -> `None`
  > No description available.

- **detect_outliers_zscore**(`self, column, threshold`) -> `None`
  > Detect outliers using Z-score method

- **detect_outliers_iqr**(`self, column, multiplier`) -> `None`
  > Detect outliers using IQR method

- **detect_outliers_quantile**(`self, column, lower, upper`) -> `None`
  > Detect outliers using quantile method

- **detect_lof_outliers**(`self, data, n_neighbors, contamination, numerical_imputation_method, categorical_imputation_method, remainder_columns, remainder_threshold, debug`) -> `None`
  > Detect outliers using Local Outlier Factor (LOF)
  >
  >
  > **Parameters**:
  >
  > columns : list
  >     List of column names to use for outlier detection
  > n_neighbors : int, optional (default=20)
  >     Number of neighbors to use for LOF
  > contamination : str or float, optional (default='auto')
  >     Expected proportion of outliers in the dataset
  > numerical_imputation_method : str, optional (default='missforest')
  >     Method for imputing numerical missing values
  > categorical_imputation_method : str, optional (default='missforest')
  >     Method for imputing categorical missing values
  > remainder_columns : str or list, optional (default='auto')
  >     Columns to keep if not in the selected columns
  > remainder_threshold : float, optional (default=0.4)
  >     Threshold for keeping columns with missing values
  > debug : bool, optional (default=True)
  >     Enable debug mode
  >
  >
  > **Returns**:
  >
  > tuple
  >     Outlier tags, outlier scores, and additional information

- **detect_isolation_forest_outliers**(`self, data, n_estimators, max_samples, contamination, numerical_imputation_method, categorical_imputation_method, remainder_columns, remainder_threshold, debug`) -> `None`
  > Detect outliers using Isolation Forest method
  >
  >
  > **Parameters**:
  >
  > columns : list
  >     List of column names to use for outlier detection
  > n_estimators : int, optional (default=100)
  >     Number of trees in the forest
  > max_samples : str or int, optional (default='auto')
  >     Number of samples to draw for each base estimator
  > contamination : str or float, optional (default='auto')
  >     Expected proportion of outliers in the dataset
  > numerical_imputation_method : str, optional (default='missforest')
  >     Method for imputing numerical missing values
  > categorical_imputation_method : str, optional (default='missforest')
  >     Method for imputing categorical missing values
  > remainder_columns : str or list, optional (default='auto')
  >     Columns to keep if not in the selected columns
  > remainder_threshold : float, optional (default=0.4)
  >     Threshold for keeping columns with missing values
  > debug : bool, optional (default=True)
  >     Enable debug mode
  >
  >
  > **Returns**:
  >
  > tuple
  >     Outlier tags, outlier scores, and additional information

- **detect_dbscan_outliers**(`self, data, eps, min_samples, numerical_imputation_method, categorical_imputation_method, remainder_columns, remainder_threshold, debug`) -> `None`
  > Detect outliers using DBSCAN method
  >
  >
  > **Parameters**:
  >
  > data : pd.DataFrame
  >     Input data
  > eps : float
  >     The maximum distance between two samples for one to be considered as in the neighborhood of the other.
  > min_samples : int
  >     The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
  > numerical_imputation_method : str
  >     Imputation method for numerical columns
  > categorical_imputation_method : str
  >     Imputation method for categorical columns
  > remainder_columns : str
  >     How to handle remainder columns
  > remainder_threshold : float
  >     Threshold for remainder columns
  > debug : bool
  >     Enable debug printing
  >
  >
  > **Returns**:
  >
  > tuple
  >     (outliers_tag, outliers_scores, additional_info)

- **handle_outliers**(`self, columns, method, handling_strategy`) -> `None`
  > Handle outliers in the DataFrame and generate an outlier matrix.
  >
  > Parameters:
  >     columns (list): List of columns to analyze for outliers.
  >     method (str): Method to detect outliers ('zscore', 'iqr', 'quantile', 'Local Outlier Factor', 'Isolation Forest').
  >     handling_strategy (str): Strategy for handling outliers ('none', 'remove', 'clip', 'tag').
  >     **kwargs: Additional parameters for specific outlier detection methods.
  >
  > Returns:
  >     DataFrame: Processed DataFrame with outliers handled.
  >     DataFrame: Outlier matrix with boolean values indicating outliers.

- **_apply_handling_strategy**(`self, columns, outlier_matrix, strategy, method, is_outlier, outlier_score`) -> `None`
  > Apply a handling strategy to the outliers.
  >
  > Parameters:
  >     columns (list): Columns analyzed for outliers.
  >     outlier_matrix (DataFrame): Boolean matrix indicating outliers.
  >     strategy (str): Handling strategy ('none', 'remove', 'clip', 'tag').
  >     method (str): Outlier detection method.
  >     is_outlier (ndarray): Boolean array for dataset-wide methods.
  >     outlier_score (ndarray): Array of outlier scores for dataset-wide methods.
  >     **kwargs: Additional parameters for handling strategies.
  >
  > Returns:
  >     DataFrame: Processed DataFrame with outliers handled.

- **clip_outliers**(`self, column, is_outlier, method`) -> `None`
  > Helper function to clip outliers based on the specified method.
  >
  > Parameters:
  >     column (str): Column name to process.
  >     is_outlier (pd.Series): Boolean mask indicating outliers.
  >     method (str): Method for clipping ('zscore', 'iqr', 'quantile').
  >     kwargs: Additional parameters for each method.
  >
  > Returns:
  >     pd.Series: Updated column with outliers clipped.

- **get_outliers_masks**(`self`) -> `None`
  > Get the masks of outliers detected so far.
  >
  >
  > **Returns**:
  >
  > dict or None
  >     Dictionary of outlier masks if available, else None.

- **get_outlier_summary**(`self, columns, method`) -> `None`
  > Get a summary of outliers for each column or the dataset as a whole.
  >
  > Parameters:
  >     columns (list): List of columns to analyze for outliers.
  >     method (str): Method to detect outliers ('zscore', 'iqr', 'quantile', 'Local Outlier Factor', 'Isolation Forest').
  >     **kwargs: Additional parameters for specific outlier detection methods.
  >
  > Returns:
  >     dict: Summary of outliers for each column or the dataset.

### def `export_comparison_results` (data_monitoring.py)

- **Arguments**: `df1, df2, comparison_result_filtered, rows_only_in_df1, rows_only_in_df2, modified_common_ids, common_id, key_base`

- **Returns**: `None`

Efficiently export comparison results to an Excel file

### def `data_selection` (data_monitoring.py)

- **Arguments**: `tmp: str, label: str`

- **Returns**: `tuple[ComplexType]`

Handle data selection from various sources.

**Parameters**:

tmp : str
    Unique identifier for the session keys.
label : str
    Label to display in the UI selector.

**Returns**:

Tuple[Optional[pd.DataFrame], Optional[str]]
    The selected dataframe and its source name/path, or (None, None) if selection failed or nothing selected.

### def `add_outlier_handling_ui` (data_monitoring.py)

- **Arguments**: `df, numeric_cols`

- **Returns**: `None`

Add outlier handling UI components with improved organization and summary display.

### def `configure_method_params` (data_monitoring.py)

- **Arguments**: `detection_method`

- **Returns**: `None`

Configure parameters based on the selected outlier detection method.

### def `select_columns_for_outlier_handling` (data_monitoring.py)

- **Arguments**: `df, numeric_cols, detection_method`

- **Returns**: `None`

Select columns for outlier handling based on the detection method.

### def `format_change` (data_monitoring.py)

- **Arguments**: `change, change_measure`

- **Returns**: `None`

No description available.

### def `display_combined_information` (data_monitoring.py)

- **Arguments**: `df, df_processed, summary, selected_cols, detection_method, handling_strategy`

- **Returns**: `None`

Display outlier summary, outlier impact (percentage change), and before/after statistics side by side for each variable.

### def `calculate_vmin_vmax` (data_monitoring.py)

- **Arguments**: `df, change_measure`

- **Returns**: `None`

No description available.

### def `calculate_outlier_impact` (data_monitoring.py)

- **Arguments**: `df, df_processed, selected_cols, change_measure`

- **Returns**: `None`

Calculate the impact of outlier handling on each selected column by computing
the percentage change in mean, std, min, and max values.

### def `generate_outlier_summary_from_predictions` (data_monitoring.py)

- **Arguments**: `df, outlier_predictions, selected_cols, method`

- **Returns**: `None`

Generate outlier summary based on raw predictions.

Parameters:
    df (DataFrame): Original DataFrame.
    outlier_predictions (dict): Raw outlier predictions for each column.
    selected_cols (list): Columns analyzed for outliers.
    method (str): Outlier detection method.

Returns:
    dict: Summary of outliers for each column or overall.

### def `display_outlier_visualization` (data_monitoring.py)

- **Arguments**: `df, outlier_matrix, selected_cols, method`

- **Returns**: `None`

Display 2D visualization of outliers using PCA, FAMD, t-SNE, or UMAP.

### def `outlier_values_dialog` (data_monitoring.py)

- **Arguments**: `outlier_summary`

- **Returns**: `None`

Display outlier values in a dialog

### def `display_results_with_outliers` (data_monitoring.py)

- **Arguments**: `filtered_df: pd.DataFrame, st`

- **Returns**: `None`

Display the results of the enrichment process.

### def `ensure_family_exists` (data_monitoring.py)

- **Arguments**: `config: dict[ComplexType], family_name: str`

- **Returns**: `None`

Ensure the specified family exists in the configuration.

### def `validate_mask_name` (data_monitoring.py)

- **Arguments**: `config: dict[ComplexType], family_name: str, mask_name: str, st_container`

- **Returns**: `None`

Validate the mask name and handle duplicates.

### def `handle_numeric_mask_input` (data_monitoring.py)

- **Arguments**: `st_container, base_key: str`

- **Returns**: `None`

Collect numeric mask inputs from the user.

### def `handle_expression_mask_input` (data_monitoring.py)

- **Arguments**: `st_container, base_key: str`

- **Returns**: `None`

Collect expression mask inputs from the user.

### def `handle_family_operator` (data_monitoring.py)

- **Arguments**: `st_container, base_key: str`

- **Returns**: `None`

Handle family operator selection and its parameters.

### def `add_mask_family` (data_monitoring.py)

- **Arguments**: `config: dict[ComplexType], st_container`

- **Returns**: `None`

No description available.

### def `display_mask_family` (data_monitoring.py)

- **Arguments**: `config`

- **Returns**: `None`

Create a color-coded display of mask families configuration.

### def `display_mask_family_alt` (data_monitoring.py)

- **Arguments**: `config`

- **Returns**: `None`

Display mask families configuration in a single top-level expander without nesting.

### def `apply_family_mask` (data_monitoring.py)

- **Arguments**: `df: pd.DataFrame, family_name: str, family_config: dict, st_container`

- **Returns**: `None`

Apply a single mask family from JSON configuration to the dataset.

Parameters:
- df: DataFrame to which the mask family is applied.

- family_name: Name of the mask family to apply.

- family_config: Configuration for the specific mask family.

- st_container: Streamlit container for displaying results.

Returns:
- Updated DataFrame with mask family tags and lists.

- The names of the columns related to the masks family

### def `display_results` (data_monitoring.py)

- **Arguments**: `df: pd.DataFrame, bool_column: str, title: str, key_base: str, st`

- **Returns**: `None`

Display the results of the enrichment process.

### def `color_columns` (data_monitoring.py)

- **Arguments**: `s, columns_to_highlight`

- **Returns**: `None`

Applies a background color to specific columns.

### def `deep_merge_dicts` (data_monitoring.py)

- **Arguments**: `original, new`

- **Returns**: `None`

Deeply merges two dictionaries. Adds new keys and values from `new` to `original`
without modifying existing keys in `original`.

### def `render_ai_criteria_assistant` (data_monitoring.py)

- **Arguments**: `st_container, config, family_name, mode`

- **Returns**: `None`

Renders the AI assistant for generating criteria.

### def `add_domain_expert_based_anomaly_mask` (data_monitoring.py)

- **Arguments**: `config: dict[ComplexType], st_container`

- **Returns**: `None`

No description available.

### def `display_results_with_anomaly` (data_monitoring.py)

- **Arguments**: `df: pd.DataFrame, anomaly_col: str, st`

- **Returns**: `None`

Display the results of the enrichment process.

### def `add_study_inclusion_mask` (data_monitoring.py)

- **Arguments**: `config: dict[ComplexType], st_container`

- **Returns**: `None`

No description available.

### def `display_results_with_inclusion` (data_monitoring.py)

- **Arguments**: `df: pd.DataFrame, inclusion_col: str, st`

- **Returns**: `None`

Display the results of the enrichment process.

### def `process_dataset` (data_monitoring.py)

- **Arguments**: `df: pd.DataFrame`

- **Returns**: `pd.DataFrame`

Process dataset using DataAnalyzer to fix types.

### def `app` (data_monitoring.py)

- **Arguments**: ``

- **Returns**: `None`

Main application function for the Data Monitoring page.

Handles:
1. Data Quality Dashboard (Data Validity, Completeness, etc.).
2. Clinical Anomalies Management (Add/Apply Masks).
3. Study Inclusion Criteria Management (Add/Apply Masks).
4. Outlier Handling (Univariate & Multivariate).
5. Dataset Comparison (Changes tracking).

---

## data_quality_dashboard.py

_No module description._

**Imports**:
`numpy`, `pandas`, `plotly.express`, `plotly.figure_factory`, `plotly.graph_objects`, `scipy.cluster.hierarchy`, `statsmodels.stats.multitest`, `streamlit`, `utils.analysis_utils`, `utils.visualization_utils`, `explore.data_quality_auditor.DataQualityAuditor`

### def `render_dashboard` (data_quality_dashboard.py)

- **Arguments**: `df, config`

- **Returns**: `None`

Renders the Data Quality Dashboard.

---

## document_insight.py

_No module description._

**Imports**:
`json`, `streamlit`, `graph.builders.document.build_document_graph`, `graph.builders.document.filter_by_documents`, `graph.model.LayoutHint`, `graph.model.RenderOptions`, `graph.ui.current_theme`, `graph.ui.render_graph`, `graph.ui.render_graph_exports`, `graph.ui.renderer_selector`, `datetime`, `core.workspace.UserScope`, `core.workspace.resolve_within`, `core.workspace.safe_filename`, `ui.components.no_dataset`

### def `_graph_dir` (document_insight.py)

- **Arguments**: `username`

- **Returns**: `None`

Knowledge-graph directory of the user's workspace (audit E4).

### def `get_saved_graphs` (document_insight.py)

- **Arguments**: `username`

- **Returns**: `None`

Returns the saved graph names of the user's workspace.

### def `save_graph` (document_insight.py)

- **Arguments**: `name, graph_json, source_docs, username`

- **Returns**: `None`

Saves graph + metadata into the user's workspace.

### def `load_graph` (document_insight.py)

- **Arguments**: `name, username`

- **Returns**: `None`

Loads a graph payload from the user's workspace.

### def `app` (document_insight.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

---

## documents.py

_Documents page: literature knowledge graph and document chat._

**Imports**:
`app_pages.document_insight.app`, `ui.components.page_header`

---

## enrich.py

_Enrich page: external data merge, computed variables, imputation, transformations._

**Imports**:
`app_pages.data_enrichment.app`, `ui.components.page_header`

---

## epidemiology.py

_No module description._

**Imports**:
`re`, `matplotlib.pyplot`, `numpy`, `pandas`, `plotly.express`, `plotly.graph_objects`, `statsmodels.api`, `statsmodels.formula.api`, `streamlit`, `scipy.stats`, `scipy.stats.t`, `statsmodels.stats.power.TTestIndPower`, `biostats.power.ARE_NORMAL`, `biostats.power.ARE_WORST_CASE`, `biostats.power.anova_power`, `biostats.power.anova_sample_size`, `biostats.power.chi2_independence_power`, `biostats.power.chi2_independence_sample_size`, `biostats.power.kruskal_power`, `biostats.power.kruskal_sample_size`, `biostats.zscores.reference_zscores`, `utils.analysis_utils`, `utils.visualization_utils`, `utils.data_analyzer.DataAnalyzer`, `app_pages.analysis_diagnostic.run_diagnostic_accuracy`, `biostats.multiplicity.adjust`, `ui.components.no_dataset`, `ui.components.result_card`, `utils.export_utils.to_excel`

### def `_render_twobytwo_controls` (epidemiology.py)

- **Arguments**: `df, group_col, outcome_col, default_epi`

- **Returns**: `None`

Reference-group / event-level / design selectors; recomputes the 2x2 analysis with explicit orientation.

### def `_render_twobytwo_panel` (epidemiology.py)

- **Arguments**: `epi`

- **Returns**: `None`

Labelled 2x2 table, measures of association (comparison vs reference) and guidance.

### def `app` (epidemiology.py)

- **Arguments**: ``

- **Returns**: `None`

Main application function for the Epidemiology Analysis page.

Handles:
1. Univariate Group Comparisons (T-tests, ANOVA, Chi-Square, etc.).
2. Multivariate Analysis (ANCOVA) to control for confounders.
3. Power Analysis & Sample Size Calculation.
4. Z-Score Calculation (Reference Standardization).
5. Educational Content display.

### def `run_univariate_analysis` (epidemiology.py)

- **Arguments**: `df, analyzer`

- **Returns**: `None`

No description available.

### def `run_multivariate_analysis` (epidemiology.py)

- **Arguments**: `df, analyzer`

- **Returns**: `None`

No description available.

### def `perform_ancova` (epidemiology.py)

- **Arguments**: `df, target, group, covariates`

- **Returns**: `None`

No description available.

### def `visualize_result` (epidemiology.py)

- **Arguments**: `df, group_col, target, analyzer`

- **Returns**: `None`

No description available.

### def `run_zscore_analysis` (epidemiology.py)

- **Arguments**: `df, analyzer`

- **Returns**: `None`

No description available.

### def `show_educational_content` (epidemiology.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `run_power_analysis` (epidemiology.py)

- **Arguments**: `df, analyzer`

- **Returns**: `None`

No description available.

---

## explore.py

_Explore page: descriptive statistics, column renaming, correlations, performance benchmark._

**Imports**:
`app_pages.home.app`, `ui.components.page_header`

---

## handbook.py

_Handbook page: the researcher documentation rendered in the app (works offline)._

**Imports**:
`re`, `pathlib.Path`, `streamlit`, `ui.components.page_header`

### def `_chapters` (handbook.py)

- **Arguments**: ``

- **Returns**: `list[tuple[ComplexType]]`

(stem, title) for every top-level chapter, in file order.

### def `_anchor` (handbook.py)

- **Arguments**: `text: str`

- **Returns**: `str`

No description available.

---

## home.py

_No module description._

**Imports**:
`io.BytesIO`, `json`, `time`, `duckdb`, `numpy`, `pandas`, `streamlit`, `explore.corr_matrix`, `graph.builders.correlation.build_correlation_graph`, `graph.legend.render_legend`, `graph.model.LayoutHint`, `graph.model.RenderOptions`, `graph.ui.current_theme`, `graph.ui.render_graph`, `graph.ui.render_graph_exports`, `graph.ui.renderer_selector`, `manage.db_manager.DBManager`, `manage.rag_manager.RAGManager`, `utils.data_analyzer.DataAnalyzer`, `utils.export_utils.to_excel`, `utils.export_utils.to_excel_sheets`, `utils.statistics_utils.normality_test`, `utils.statistics_utils.sanitize_dataframe_for_arrow`, `ui.components.confirm`, `ui.components.no_dataset`, `hashlib`

### class `RenameColumnsComponent` (home.py)

A Streamlit component for renaming DataFrame columns with sorting, searching, type filtering, and top values display.

**Methods:**

- **__init__**(`self, df, analyzer`) -> `None`
  > No description available.

- **_get_columns_info**(`self, columns_to_process`) -> `None`
  > Gather detailed information about columns for RAG context.

- **display**(`self`) -> `None`
  > Displays the column renaming UI with sorting, searching, type filtering, and a compact layout.

### def `_q` (home.py)

- **Arguments**: `identifier`

- **Returns**: `str`

Quote a DuckDB identifier (column names come from uploaded files; audit E4).

### def `_frame_fingerprint` (home.py)

- **Arguments**: `df`

- **Returns**: `str`

Short content hash so cached statistics follow the frame (filters, renames), not only its name (audit E8).

### def `_top_correlations` (home.py)

- **Arguments**: `corr_matrix: pd.DataFrame, node_ids, limit: int`

- **Returns**: `pd.DataFrame`

Strongest correlations of the selected variables (for the graph selection panel).

### def `display_category_box` (home.py)

- **Arguments**: `title, columns`

- **Returns**: `None`

No description available.

### def `move_columns_to_front` (home.py)

- **Arguments**: `df, columns_to_move`

- **Returns**: `None`

Move specified columns to the beginning of the DataFrame.

Parameters:
df (pd.DataFrame): The original DataFrame.
columns_to_move (list): List of column names to move to the front.

Returns:
pd.DataFrame: DataFrame with specified columns moved to the front.

### def `get_statistics_dataframe` (home.py)

- **Arguments**: `df, _analyzer, nb_top_categories, exclude_columns, qual_var_threshold, dataset_name, _db_manager`

- **Returns**: `None`

Creates new DataFrames with statistics for numerical and non-numerical columns.
Tries to load from DuckDB/Parquet cache if dataset_name is provided.

Note: Parameters starting with _ are excluded from hashing by Streamlit.

### def `_cached_corr_excel_bytes` (home.py)

- **Arguments**: `data, targets, predictors, group_column, method`

- **Returns**: `None`

No description available.

### def `run_benchmark` (home.py)

- **Arguments**: `df, analyzer`

- **Returns**: `None`

No description available.

### def `app` (home.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

---

## import_data.py

_Import page: the three-step import flow._

**Imports**:
`ui.components.page_header`, `ui.import_flow.render_import_flow`, `ui.state.get_state`

---

## rag_monitoring.py

_RAG Monitoring Dashboard

Streamlit page for monitoring and visualizing RAG system quality metrics.
Provides dashboards for:
- Quality overview with trends

- Evaluation details and drill-down

- Embedding health visualization

- Manual evaluation triggers_

**Imports**:
`json`, `sys`, `datetime.datetime`, `pathlib.Path`, `streamlit`

### def `render_rag_monitoring` (rag_monitoring.py)

- **Arguments**: ``

- **Returns**: `None`

Main entry point for the RAG Monitoring page.

### def `render_quality_overview` (rag_monitoring.py)

- **Arguments**: `evaluator`

- **Returns**: `None`

Render the quality overview dashboard.

### def `render_evaluation_details` (rag_monitoring.py)

- **Arguments**: `evaluator`

- **Returns**: `None`

Render the evaluation details view.

### def `render_embedding_health` (rag_monitoring.py)

- **Arguments**: `evaluator`

- **Returns**: `None`

Render embedding health visualization.

### def `render_run_evaluation` (rag_monitoring.py)

- **Arguments**: `evaluator`

- **Returns**: `None`

Render the manual evaluation trigger with progress tracking.

### def `run_single_evaluation` (rag_monitoring.py)

- **Arguments**: `evaluator, rag, query: str, eval_function: str`

- **Returns**: `None`

Run a single evaluation with progress tracking.

### def `run_test_suite_with_progress` (rag_monitoring.py)

- **Arguments**: `evaluator, rag, test_cases: list, selected_ids: list`

- **Returns**: `None`

Run test suite with progress tracking.

### def `run_demo_evaluation` (rag_monitoring.py)

- **Arguments**: `evaluator`

- **Returns**: `None`

Run a demo evaluation without the full RAG system.

### def `render_export_options` (rag_monitoring.py)

- **Arguments**: `evaluator`

- **Returns**: `None`

Render export buttons.

---

## rag_sidebar.py

_RAG and AI Configuration Sidebar Component.

Provides a unified, provider-agnostic interface for configuring:
- AI Providers: Google Gemini, Ollama (Local Open Source), OpenAI, Mistral AI

- Model selection and dynamic model discovery

- Local and cloud embedding models

- Document retrieval parameters and clinical adherence calibration_

**Imports**:
`os`, `streamlit`, `dotenv.load_dotenv`, `core.workspace.safe_filename`, `manage.rag.ConnectorFactory`, `manage.rag.ProviderConfig`, `manage.rag.ProviderType`, `utils.session_scope.get_current_scope`

### def `_documents_dir` (rag_sidebar.py)

- **Arguments**: ``

- **Returns**: `str`

The signed-in user's document library (audit E4); the legacy shared folder outside a session.

### def `_discover_ollama_models` (rag_sidebar.py)

- **Arguments**: `base_url: str`

- **Returns**: `list[str]`

Ollama model list, probed at most once a minute per endpoint (audit E11).

### def `_discover_cloud_models` (rag_sidebar.py)

- **Arguments**: `_manager, provider: str, api_key: str`

- **Returns**: `list[str]`

Cloud model list, fetched at most once an hour per key (audit E11).

### def `_api_key_input` (rag_sidebar.py)

- **Arguments**: `env_var: str, label: str, key: str, help_text: str`

- **Returns**: `ComplexType`

Session-scoped API key entry.

The server-side key is never shown or pre-filled and nothing is written to ``os.environ``
(audit E6). Leaving the field empty falls back to the server-configured key, if any.

### def `_render_cloud_consent` (rag_sidebar.py)

- **Arguments**: `selected_provider: str, display_names: dict`

- **Returns**: `bool`

Explicit, session-scoped confirmation before any data can be sent to a cloud provider (audit E6).

### def `render_ai_settings` (rag_sidebar.py)

- **Arguments**: ``

- **Returns**: `None`

AI provider, keys, consent, document library and retrieval parameters (Settings page).

### def `render_rag_sidebar` (rag_sidebar.py)

- **Arguments**: ``

- **Returns**: `None`

Legacy wrapper: the same controls inside a sidebar expander (the Settings page is the primary home).

---

## reproduce.py

_Reproduce page: traces, artifacts, replay and reports._

**Imports**:
`app_pages.reproduce_analysis.app`, `ui.components.page_header`

---

## reproduce_analysis.py

_No module description._

**Imports**:
`difflib`, `json`, `os`, `pandas`, `streamlit`, `manage.trace_documenter.TraceDocumenter`, `utils.path_utils.normalize_path`, `utils.path_utils.resolve_path`, `core.workspace.safe_filename`, `utils.data_paths.get_traces_dir`, `utils.data_paths.get_uploads_dir`, `utils.session_scope.get_current_scope`, `manage.reproduction_manager.run_trace`

### def `_uploads_dir` (reproduce_analysis.py)

- **Arguments**: ``

- **Returns**: `str`

No description available.

### def `_traces_dir` (reproduce_analysis.py)

- **Arguments**: ``

- **Returns**: `str`

No description available.

### def `_artifacts_dir` (reproduce_analysis.py)

- **Arguments**: ``

- **Returns**: `str`

No description available.

### def `_resolve_user_path` (reproduce_analysis.py)

- **Arguments**: `text: ComplexType`

- **Returns**: `ComplexType`

A manually entered path must exist inside the user's workspace (audit E4).

### def `app` (reproduce_analysis.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `jump_to_step` (reproduce_analysis.py)

- **Arguments**: `trace_data, target_step_index, rerun_mode`

- **Returns**: `None`

Reproduces the analysis up to the target step index.

---

## settings.py

_Settings page: AI provider and consent, graph engine, import defaults, workspace maintenance._

**Imports**:
`streamlit`, `app_pages.rag_sidebar.render_ai_settings`, `ui.components.confirm`, `ui.components.page_header`, `ui.components.section`, `ui.shell.clear_workspace_statistics`, `ui.state.get_state`

---

## start.py

_Start page: the three-step workflow, saved versions, the sample cohort and the AI status._

**Imports**:
`streamlit`, `ui.components.page_header`, `ui.components.page_link`, `ui.components.section`, `ui.components.status_chip`, `ui.state.SAMPLE_DICTIONARY_PATH`, `ui.state.get_state`

---

## statistics_guides.py

_Normality-test guidance widgets (moved out of utils so the statistics helpers stay UI-free)._

**Imports**:
`streamlit`

### def `show_test_guidelines` (statistics_guides.py)

- **Arguments**: ``

- **Returns**: `None`

Displays a detailed explanation of the normality tests,
including algorithms, formulas, and recommendations.

### def `_show_overview_guide` (statistics_guides.py)

- **Arguments**: ``

- **Returns**: `None`

Display normality test overview and selection guide.

### def `_show_algorithm_details` (statistics_guides.py)

- **Arguments**: ``

- **Returns**: `None`

Display detailed algorithm explanations.

---

## transformation_logic.py

_Compatibility shim: the transformation logic lives in :mod:`enrich.transformation_logic` (no UI code)._

**Imports**:
`enrich.transformation_logic.*`, `enrich.transformation_logic.apply_variable_transformation`

---

## users.py

_Users page (administrators)._

**Imports**:
`app_pages.users_management.app`, `ui.components.page_header`

---

## users_management.py

_Users Management Page - Admin Only

This page allows administrators to:
- View all registered users with their status

- Activate/Deactivate user accounts

- Reset user passwords

- Delete user accounts_

**Imports**:
`streamlit`, `manage.db_manager.DBManager`, `manage.iam.authz.MIN_PASSWORD_LENGTH`, `manage.iam.authz.is_admin_role`

### def `app` (users_management.py)

- **Arguments**: ``

- **Returns**: `None`

Render the Users Management page.

---

## validate.py

_Validate page: data quality dashboard, anomalies, inclusion criteria, outliers, version comparison._

**Imports**:
`app_pages.data_monitoring.app`, `ui.components.page_header`

---

## variable_map.py

_Variable map page: taxonomy of variables, formulas, clustering and the knowledge graph._

**Imports**:
`app_pages.data_insight.app`, `ui.components.page_header`

---

## visualization.py

_No module description._

**Imports**:
`numpy`, `pandas`, `plotly.express`, `plotly.figure_factory`, `plotly.graph_objects`, `streamlit`, `plotly.subplots.make_subplots`, `scipy.stats`, `utils.export_utils`, `utils.visualization_utils`, `app_pages.data_monitoring`, `manage.db_manager.DBManager`, `utils.data_analyzer.DataAnalyzer`, `ui.components.no_dataset`

### def `app` (visualization.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

---

## visualize.py

_Visualize page: distributions, group comparisons, time series, ROC, Bland-Altman._

**Imports**:
`app_pages.visualization.app`, `ui.components.page_header`

---

## which_test.py

_Which test? Four questions -> the recommended analysis, why, and where to run it._

**Imports**:
`streamlit`, `ui.components.glossary_popover`, `ui.components.page_header`, `ui.components.page_link`, `ui.components.section`, `ui.components.status_chip`, `ui.guidance.DESIGNS`, `ui.guidance.DISTRIBUTIONS`, `ui.guidance.GROUPS`, `ui.guidance.OUTCOMES`, `ui.guidance.recommend`

---
