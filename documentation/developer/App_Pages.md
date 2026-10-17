# App Pages Documentation

## about.py

_About page body: what the platform does, third-party notices and contact._

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
`app_pages.about.app`, `ui.components.page_header`

---

## ai_quality.py

_AI quality page (administrators): RAG evaluation and embedding health._

**Imports**:
`app_pages.rag_monitoring.render_rag_monitoring`, `ui.components.page_header`

---

## analysis_config.py

_Analysis configuration widgets of the Analyze page (moved out of utils so the engine side stays UI-free)._

**Imports**:
`streamlit`, `biostats.multiplicity.UI_OPTIONS`, `ui.guidance.analysis.HELP`

### def `render_analysis_configuration` (analysis_config.py)

- **Arguments**: `df, numeric_cols, categorical_cols, binary_cols, group_col_options, default_group_col, target_col_options, key_prefix`

- **Returns**: `None`

Renders the Analysis Configuration expander.
Returns a dictionary with the selected configuration.

---

## analysis_diagnostic.py

_Diagnostic accuracy and method agreement tab of the Analyze page (audit S10)._

**Imports**:
`__future__.annotations`, `numpy`, `pandas`, `plotly.graph_objects`, `streamlit`, `biostats.agreement.bland_altman`, `biostats.diagnostic.confusion_at`, `biostats.diagnostic.delong_auc`, `biostats.diagnostic.diagnostic_accuracy`, `biostats.diagnostic.youden_cutoff`, `ui.components.error_card`, `ui.components.help_popover`, `ui.components.section`, `ui.components.status_chip`, `ui.results.provenance`, `ui.results.recall`, `ui.results.remember`

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

## analysis_extras.py

_Paired tests, incidence rates, stratified tables, standardised ratios and reliability (audit S11)._

**Imports**:
`__future__.annotations`, `pandas`, `streamlit`, `biostats.paired.mcnemar_test`, `biostats.paired.one_sample_proportion`, `biostats.paired.one_sample_t`, `biostats.paired.one_sample_wilcoxon`, `biostats.paired.paired_t`, `biostats.paired.wilcoxon_signed_rank`, `biostats.rates.incidence_rate`, `biostats.rates.mantel_haenszel`, `biostats.rates.rate_ratio`, `biostats.rates.standardised_ratio`, `biostats.reliability.cohens_kappa`, `biostats.reliability.icc`, `ui.components.error_card`, `ui.components.help_popover`, `ui.components.result_card`, `ui.components.section`, `ui.components.status_chip`, `ui.components.mi_mode.mi_token`, `ui.components.mi_mode.show_mi_notes`, `ui.progress.mark_step`, `ui.results.provenance`, `ui.results.recall`, `ui.results.remember`, `ui.state.get_state`

### def `_ci` (analysis_extras.py)

- **Arguments**: `ci`

- **Returns**: `str`

No description available.

### def `_not_pooled` (analysis_extras.py)

- **Arguments**: `analysis: str`

- **Returns**: `None`

The reason a test has no pooling rule, and what to run instead (biostats.mi.not_poolable).

### def `_show_test` (analysis_extras.py)

- **Arguments**: `res, key: str`

- **Returns**: `None`

A stored test result: the pooled test with its notes and complete-case comparison, or the observed one.

### def `run_paired_and_rates` (analysis_extras.py)

- **Arguments**: `df: pd.DataFrame, analyzer`

- **Returns**: `None`

Paired and one-sample tests (pooled over the imputed datasets ``mi`` when given), McNemar, rates,
Mantel-Haenszel and standardised ratios.

### def `run_reliability` (analysis_extras.py)

- **Arguments**: `df: pd.DataFrame, analyzer`

- **Returns**: `None`

No description available.

---

## analysis_guidance_ui.py

_Rendering of the guidance of Analyze > Hypothesis tests & epidemiology (rules in ``ui.guidance.analysis``).

Everything here is shown only while the page's Guidance switch is on, so experienced users can keep
the page compact._

**Imports**:
`__future__.annotations`, `pandas`, `streamlit`, `ui.components.page_link`, `ui.components.status_chip`, `ui.guidance.analysis.HELP`, `ui.guidance.analysis.TAB_GUIDE`, `ui.guidance.analysis.TEST_NOTES`, `ui.guidance.analysis.AnalysisPlan`, `ui.guidance.analysis.AncovaCheck`

### def `guidance_toggle` (analysis_guidance_ui.py)

- **Arguments**: ``

- **Returns**: `bool`

No description available.

### def `guidance_on` (analysis_guidance_ui.py)

- **Arguments**: ``

- **Returns**: `bool`

No description available.

### def `render_orientation` (analysis_guidance_ui.py)

- **Arguments**: ``

- **Returns**: `None`

Which tab answers which question, and a way to the test chooser.

### def `_advice` (analysis_guidance_ui.py)

- **Arguments**: `level: str, text: str`

- **Returns**: `None`

No description available.

### def `render_plan` (analysis_guidance_ui.py)

- **Arguments**: `plan: AnalysisPlan`

- **Returns**: `None`

What Run will do, checked against the data, before anything is computed.

### def `render_ancova_check` (analysis_guidance_ui.py)

- **Arguments**: `check: AncovaCheck`

- **Returns**: `None`

No description available.

### def `render_test_note` (analysis_guidance_ui.py)

- **Arguments**: `test: str`

- **Returns**: `None`

No description available.

---

## analysis_pooled.py

_Analyze page with multiply imputed data (follow-up programme M2): group comparisons and ANCOVA run on the m
imputed datasets of the analysis population and pooled with Rubin's rules (biostats.mi.analyze_mi).

Group comparison, per variable:
- numeric, two groups: Welch t-test, the mean difference pooled (Barnard-Rubin df);

- numeric, three or more groups: a linear model on the groups, the group means pooled and the group effect
  tested by D1 (pooled Wald F; equal variances, as in a one-way ANOVA);

- two categories, two groups: odds ratio and risk ratio (log scale) and risk difference pooled, the
  chi-square by D2;

- other categorical variables: the chi-square test of independence by D2.
Rank tests and Fisher's exact test have no pooling rule: the page gives the reason and the alternative._

**Imports**:
`__future__.annotations`, `dataclasses.dataclass`, `typing.Any`, `pandas`, `streamlit`, `biostats.contingency.TwoByTwo`, `biostats.contingency.analyze_twobytwo`, `biostats.contingency.chi_square_independence`, `biostats.contingency.detect_event_level`, `biostats.contingency.detect_reference_level`, `biostats.inference.compare_groups`, `biostats.linear.ancova`, `biostats.mi.analyze_mi`, `biostats.mi.not_poolable`, `biostats.multiplicity.UI_OPTIONS`, `biostats.multiplicity.adjust`, `ui.components.error_card`, `ui.components.result_card`, `ui.components.mi_mode.mi_token`, `ui.components.mi_mode.show_mi_notes`, `ui.components.mi_mode.show_pooled_terms`, `ui.components.mi_mode.show_tests`, `ui.progress.mark_step`, `ui.results.provenance`, `ui.results.recall`, `ui.results.remember`, `ui.state.get_state`

### class `Pooled` (analysis_pooled.py)

One variable compared across the groups: the kind of analysis, the pooled result and its orientation.

### def `_levels` (analysis_pooled.py)

- **Arguments**: `series: pd.Series`

- **Returns**: `list[str]`

No description available.

### def `_in_groups` (analysis_pooled.py)

- **Arguments**: `data: pd.DataFrame, group: str, levels: list[str]`

- **Returns**: `pd.DataFrame`

No description available.

### def `_columns` (analysis_pooled.py)

- **Arguments**: `mi, analyzer`

- **Returns**: `tuple[ComplexType]`

Numeric and categorical columns of the imputed data, and the grouping candidates.

### def `compare_pooled` (analysis_pooled.py)

- **Arguments**: `mi, group: str, levels: list[str], target: str`

- **Returns**: `Pooled`

Compare ``target`` across the ``levels`` of ``group`` on every imputed dataset and pool the result.

### def `_summary_row` (analysis_pooled.py)

- **Arguments**: `target: str, item: Pooled, group: str`

- **Returns**: `dict[ComplexType]`

No description available.

### def `_show_summary` (analysis_pooled.py)

- **Arguments**: `results: dict[ComplexType], errors: dict[ComplexType], group: str, correction: str`

- **Returns**: `None`

No description available.

### def `_twobytwo_frame` (analysis_pooled.py)

- **Arguments**: `res`

- **Returns**: `pd.DataFrame`

No description available.

### def `_show_complete_twobytwo` (analysis_pooled.py)

- **Arguments**: `cc`

- **Returns**: `None`

No description available.

### def `_show_means` (analysis_pooled.py)

- **Arguments**: `means: pd.DataFrame`

- **Returns**: `None`

No description available.

### def `_show_detail` (analysis_pooled.py)

- **Arguments**: `target: str, item: Pooled, group: str`

- **Returns**: `None`

No description available.

### def `run_pooled_comparison` (analysis_pooled.py)

- **Arguments**: `mi, analyzer`

- **Returns**: `None`

No description available.

### def `run_pooled_ancova` (analysis_pooled.py)

- **Arguments**: `mi, analyzer`

- **Returns**: `None`

No description available.

---

## analysis_survival.py

_Survival & regression page body: Kaplan-Meier, log-rank, Cox, logistic and count regression (audit S11).

With multiply imputed data (``mi``), every analysis is run on the m imputed datasets and pooled
(biostats.mi.analyze_mi): Kaplan-Meier curves on the cloglog scale with the median read from the pooled
curve, the log-rank test by D2, hazard, odds and rate ratios with Rubin's rules and a D1 test per factor._

**Imports**:
`__future__.annotations`, `numpy`, `pandas`, `plotly.graph_objects`, `streamlit`, `biostats._types.TestResult`, `biostats.regression.count_regression`, `biostats.regression.logistic_regression`, `biostats.survival.compare_survival`, `biostats.survival.cox_regression`, `ui.components.error_card`, `ui.components.help_popover`, `ui.components.result_card`, `ui.components.section`, `ui.components.mi_mode.mi_token`, `ui.components.mi_mode.show_mi_notes`, `ui.components.mi_mode.show_pooled_terms`, `ui.components.mi_mode.show_tests`, `ui.progress.mark_step`, `ui.results.provenance`, `ui.results.recall`, `ui.results.remember`, `ui.state.get_state`

### def `_group_candidates` (analysis_survival.py)

- **Arguments**: `df: pd.DataFrame, analyzer, max_levels: int`

- **Returns**: `list[str]`

No description available.

### def `_km_figure` (analysis_survival.py)

- **Arguments**: `curves`

- **Returns**: `go.Figure`

No description available.

### def `_pooled_km_figure` (analysis_survival.py)

- **Arguments**: `curves: dict[ComplexType]`

- **Returns**: `go.Figure`

No description available.

### def `_median_text` (analysis_survival.py)

- **Arguments**: `median: ComplexType`

- **Returns**: `str`

The median survival time, or 'not reached' when the curve stays above 0.5.

### def `_table` (analysis_survival.py)

- **Arguments**: `df: pd.DataFrame, formats: dict`

- **Returns**: `None`

No description available.

### def `_default_index` (analysis_survival.py)

- **Arguments**: `options: list[str], hints: tuple[ComplexType]`

- **Returns**: `int`

Index of the first option whose name contains a hint (e.g. 'time', 'died'), else 0.

### def `_show_survival` (analysis_survival.py)

- **Arguments**: `res`

- **Returns**: `None`

No description available.

### def `_show_pooled_survival` (analysis_survival.py)

- **Arguments**: `res`

- **Returns**: `None`

No description available.

### def `run_survival` (analysis_survival.py)

- **Arguments**: `df: pd.DataFrame, analyzer`

- **Returns**: `None`

No description available.

---

## analyze.py

_Analyze page: hypothesis tests, epidemiological measures, ANCOVA, power, z-scores._

**Imports**:
`app_pages.epidemiology.app`, `ui.components.page_header`, `ui.components.mi_mode.mi_banner`, `ui.components.population.population_banner`, `ui.state.get_state`

---

## data_enrichment.py

_No module description._

**Imports**:
`ast`, `io`, `json`, `logging`, `os`, `re`, `datetime.datetime`, `numpy`, `pandas`, `streamlit`, `matplotlib.pyplot`, `rapidfuzz.fuzz`, `enrich.external_data`, `app_pages.enrichment_mi.render_multiple_imputation`, `core.expressions.safe_format`, `core.population.catalogue`, `enrich.formulas.evaluate_formula_safely`, `enrich.formulas.extract_constant_value`, `enrich.formulas.far_outside`, `enrich.formulas.numeric_summary`, `enrich.formulas.validate_and_extract_columns_ast`, `enrich.formulas.validate_variable_name`, `enrich.imputation_service.impute_rows`, `ingest.loaders.UPLOAD_TYPES`, `manage.db_manager.DBManager`, `ui.components.error_card`, `ui.components.no_dataset`, `ui.components.staging.render_draft_banner`, `ui.state.get_state`, `utils.data_analyzer.DataAnalyzer`, `utils.data_paths.get_traces_dir`, `utils.date_parser.smart_parse_dates`, `utils.export_utils.to_excel`, `utils.logging_config.single_line`, `utils.multipage._load_excel_with_selection`, `utils.multipage.get_file_hash`, `utils.multipage.load_dataframe`, `utils.optional.available_methods`, `utils.optional.method_available`, `utils.optional.missing_methods_message`, `utils.session_scope.get_current_scope`

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

The main dataset of the enrichment: the session dataset with the Enrich draft applied.

Datasets are loaded from Data > Import or the sidebar (AppState.set_dataset), so every load starts a
trace; this page no longer reads files of its own.

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

### def `generate_stats` (data_enrichment.py)

- **Arguments**: `temp_main_data, variable_name`

- **Returns**: `None`

No description available.

### def `render_new_variable_statistics` (data_enrichment.py)

- **Arguments**: `temp_data, var_name, used_cols`

- **Returns**: `None`

Statistics of a new variable on the analysis population, robust (median, quartiles) and classical (mean, SD),
and the values far outside the others with their source values: a few entry errors (a height in metres, a
weight of 0) can move the mean and the SD far from what the median shows.

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

Main application function of the Enrich page (Data > Enrich).

Handles:
1. External Data Upload and Enrichment
2. Targeted Imputation (Manual/AI-assisted)
3. Global Imputation (MICE, KNN, etc.)
4. Dataset Versioning and Persistence

---

## data_insight.py

_No module description._

**Imports**:
`json`, `os`, `re`, `pandas`, `streamlit`, `core.workspace.UserScope`, `graph.builders.taxonomy.CATEGORY_LABELS`, `graph.builders.taxonomy.RELATION_LABELS`, `graph.builders.taxonomy.build_taxonomy_graph`, `graph.builders.taxonomy.make_formula_id`, `graph.builders.taxonomy.parse_formula_id`, `graph.model.LayoutHint`, `graph.model.RenderOptions`, `graph.ui.current_theme`, `graph.ui.render_graph`, `graph.ui.render_graph_exports`, `graph.ui.renderer_selector`, `ui.components.no_dataset`, `utils.clustering_utils.compute_cluster_profiles`, `utils.clustering_utils.fit_dbscan`, `utils.clustering_utils.fit_gaussian_mixture`, `utils.clustering_utils.fit_kmeans`, `utils.clustering_utils.prepare_data_for_clustering`, `utils.clustering_utils.run_famd`, `utils.clustering_utils.run_pca`, `utils.clustering_utils.run_tsne`, `utils.clustering_utils.run_umap`, `utils.optional.available_methods`

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

Main application function of the Variable map page (Knowledge > Variable map).

Handles:
1. Loading and saving taxonomy versions.
2. Visualizing variables and formulas as a graph.
3. Generating new taxonomies using RAG.
4. Enriching existing taxonomies with external knowledge.

---

## data_monitoring.py

_No module description._

**Imports**:
`json`, `os`, `io.BytesIO`, `typing.Any`, `numpy`, `pandas`, `streamlit`, `scipy.stats`, `monitor.changes`, `monitor.outliers`, `utils.clustering_utils`, `app_pages.data_quality_dashboard.render_dashboard`, `core.fingerprint.ids_to_ranges`, `core.population.catalogue`, `core.population.criterion_id`, `enrich.custom_metrics_and_filters`, `ingest.loaders.UPLOAD_TYPES`, `manage.db_manager.DBManager`, `ui.components.no_dataset`, `ui.components.population.activate_family`, `ui.components.population.render_population_panel`, `ui.state.get_state`, `utils.config_loader.create_empty_config`, `utils.config_loader.transform_expression`, `utils.data_analyzer.DataAnalyzer`, `utils.date_parser.smart_parse_dates`, `utils.export_utils.to_excel`, `utils.multipage.load_dataframe`, `utils.optional.available_methods`, `utils.session_scope.get_current_scope`

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

### def `_clip_bounds` (data_monitoring.py)

- **Arguments**: `df: pd.DataFrame, columns, method: str, params: dict`

- **Returns**: `dict`

The bounds a clipping uses, per column (recorded in the trace and replayed).

### def `_exclude_outliers` (data_monitoring.py)

- **Arguments**: `detection: dict`

- **Returns**: `None`

Callback: the flagged rows become an exclusion criterion (row ids frozen now), ticked at once.

### def `_clip_outliers` (data_monitoring.py)

- **Arguments**: `detection: dict`

- **Returns**: `None`

Callback: clip the values to the method's bounds in the dataset, as one traced, replayable step.

### def `render_outlier_actions` (data_monitoring.py)

- **Arguments**: `detection: ComplexType`

- **Returns**: `None`

What to do with the detected outliers: exclude them from the population, or clip the values.

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

### def `render_date_conversion` (data_monitoring.py)

- **Arguments**: `state, df: pd.DataFrame, analyzer`

- **Returns**: `None`

Text columns that look like dates: preview the parsing, convert them as one traced step on request.

Dates used to be re-parsed in place on every visit of this page, untraced and invisible to the replay.

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
`numpy`, `pandas`, `plotly.express`, `plotly.figure_factory`, `plotly.graph_objects`, `scipy.cluster.hierarchy`, `statsmodels.stats.multitest`, `streamlit`, `utils.analysis_utils`, `utils.visualization_utils`, `explore.data_quality_auditor.DataQualityAuditor`, `explore.quality_score.DEFAULT_WEIGHTS`, `explore.quality_score.DIMENSION_HELP`, `explore.quality_score.DIMENSIONS`, `explore.quality_score.interpretation`, `explore.quality_score.ordered`, `explore.quality_score.weighted_score`, `ui.progress.mark_step`, `ui.progress.record_quality_check`, `ui.quality_charts.chart_config`, `ui.quality_charts.history_figure`, `ui.quality_charts.loss_figure`, `ui.quality_charts.ring_figure`

### def `_selected_dimension` (data_quality_dashboard.py)

- **Arguments**: `event`

- **Returns**: `ComplexType`

Dimension of the bar clicked in the "points lost" chart, if any.

### def `_score_breakdown` (data_quality_dashboard.py)

- **Arguments**: `quality, details, checks, weights`

- **Returns**: `None`

Where the score loses points (click a bar to see the columns or rules) and the score across checks.

### def `render_dashboard` (data_quality_dashboard.py)

- **Arguments**: `df, config`

- **Returns**: `None`

Renders the Data Quality Dashboard.

---

## document_insight.py

_No module description._

**Imports**:
`datetime`, `json`, `streamlit`, `core.workspace.UserScope`, `core.workspace.resolve_within`, `core.workspace.safe_filename`, `graph.builders.document.build_document_graph`, `graph.builders.document.filter_by_documents`, `graph.model.LayoutHint`, `graph.model.RenderOptions`, `graph.ui.current_theme`, `graph.ui.render_graph`, `graph.ui.render_graph_exports`, `graph.ui.renderer_selector`, `ui.components.no_dataset`

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
`streamlit`, `app_pages.document_insight.app`, `ui.components.page_header`, `utils.optional.extra_installed`, `utils.optional.missing_message`

---

## dossier.py

_Study dossier page: the participant flow, the results added to the report and the Word dossier._

**Imports**:
`app_pages.study_dossier.app`, `ui.components.page_header`

---

## enrich.py

_Enrich page: external data merge, computed variables, imputation, transformations._

**Imports**:
`app_pages.data_enrichment.app`, `ui.components.page_header`

---

## enrichment_mi.py

_Multiple imputation on the Enrich page (follow-up programme M2): a guided panel in five steps.

1. Population and missingness: what will be imputed, on which rows, how many imputations.
2. Imputation model: the variables (the outcome of the planned analyses always in) and a method each.
3. Settings: m, iterations, seed, strata, the auxiliary of a Cox analysis, parallel processes.
4. Run and diagnose: chain traces, R-hat, observed against imputed values.
5. Use, save, export: status against the analysis population, versions, mice and Stata files, methods text.

The imputation never changes the dataset: it is kept beside it (``AppState.set_mi``), recorded in the trace
and re-computed by the replay. The panel runs as a fragment so its widgets do not rerun the whole page._

**Imports**:
`__future__.annotations`, `os`, `numpy`, `pandas`, `plotly.graph_objects`, `streamlit`, `biostats._types.RANDOM_STATE`, `biostats.mi.report.default_model_columns`, `biostats.mi.report.incomplete_share`, `biostats.mi.report.methods_paragraph`, `biostats.mi.report.missingness_summary`, `biostats.mi.spec.ImputationSpec`, `biostats.mi.spec.default_method`, `biostats.mi.spec.is_numeric`, `biostats.mi.spec.suggest_m`, `core.session.PopulationError`, `ui.components.error_card`, `ui.components.help_popover`, `ui.state.get_state`

### def `_status_badge` (enrichment_mi.py)

- **Arguments**: `state`

- **Returns**: `None`

No description available.

### def `render_multiple_imputation` (enrichment_mi.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `_trace_figure` (enrichment_mi.py)

- **Arguments**: `trace: dict, name: str`

- **Returns**: `go.Figure`

No description available.

### def `_distribution_figure` (enrichment_mi.py)

- **Arguments**: `observed: pd.Series, imputed: pd.Series, numeric: bool`

- **Returns**: `go.Figure`

No description available.

### def `_diagnostics` (enrichment_mi.py)

- **Arguments**: `stored`

- **Returns**: `None`

No description available.

### def `_use` (enrichment_mi.py)

- **Arguments**: `state, stored`

- **Returns**: `None`

No description available.

---

## epidemiology.py

_No module description._

**Imports**:
`matplotlib.pyplot`, `numpy`, `pandas`, `plotly.express`, `plotly.graph_objects`, `streamlit`, `scipy.stats`, `scipy.stats.t`, `statsmodels.stats.power.TTestIndPower`, `utils.analysis_utils`, `utils.visualization_utils`, `app_pages.analysis_diagnostic.run_diagnostic_accuracy`, `app_pages.analysis_extras.run_paired_and_rates`, `app_pages.analysis_extras.run_reliability`, `app_pages.analysis_guidance_ui.guidance_on`, `app_pages.analysis_guidance_ui.guidance_toggle`, `app_pages.analysis_guidance_ui.render_ancova_check`, `app_pages.analysis_guidance_ui.render_orientation`, `app_pages.analysis_guidance_ui.render_plan`, `app_pages.analysis_guidance_ui.render_test_note`, `biostats.multiplicity.adjust`, `biostats.power.ARE_NORMAL`, `biostats.power.ARE_WORST_CASE`, `biostats.power.anova_power`, `biostats.power.anova_sample_size`, `biostats.power.chi2_independence_power`, `biostats.power.chi2_independence_sample_size`, `biostats.power.kruskal_power`, `biostats.power.kruskal_sample_size`, `biostats.zscores.reference_zscores`, `core.fingerprint.frame_fingerprint`, `ui.components.error_card`, `ui.components.no_dataset`, `ui.components.result_card`, `ui.guidance.analysis.HELP`, `ui.guidance.analysis.ancova_check`, `ui.guidance.analysis.cohens_d_from_means`, `ui.guidance.analysis.cohens_f_from_means`, `ui.guidance.analysis.cohens_w_from_proportions`, `ui.guidance.analysis.p_superiority_from_d`, `ui.guidance.analysis.parse_numbers`, `ui.guidance.analysis.plan_group_comparison`, `ui.guidance.analysis.power_sentence`, `ui.guidance.analysis.sample_size_sentence`, `ui.progress.mark_step`, `ui.results.CURRENT`, `ui.results.provenance`, `ui.results.recall`, `ui.results.remember`, `ui.results.status`, `utils.data_analyzer.DataAnalyzer`, `utils.export_utils.to_excel`

### def `_effect_ok` (epidemiology.py)

- **Arguments**: `effect_size`

- **Returns**: `bool`

A planning effect must be positive: with no difference to detect, no sample is large enough.

### def `_prefilled_number_input` (epidemiology.py)

- **Arguments**: `label, default`

- **Returns**: `None`

A number input that the results above may prefill through the session state ("Load Parameters into
Calculator", the group sizes of the dataset). Streamlit warns when a widget gets both a default value and a
session-state value, so the default is passed only while the key is unset.

### def `_expected_means_inputs` (epidemiology.py)

- **Arguments**: `prefix: str`

- **Returns**: `None`

Expected means of two groups and the common SD; returns (mean1, mean2, sd).

### def `_render_twobytwo_controls` (epidemiology.py)

- **Arguments**: `df, group_col, outcome_col, default_epi`

- **Returns**: `None`

Reference-group / event-level / design selectors; recomputes the 2x2 analysis with explicit orientation.

### def `_render_twobytwo_panel` (epidemiology.py)

- **Arguments**: `epi`

- **Returns**: `None`

Labelled 2x2 table, measures of association (comparison vs reference) and guidance.

### def `app` (epidemiology.py)

- **Arguments**: `mi`

- **Returns**: `None`

Main application function for the Epidemiology Analysis page.

Handles:
1. Univariate Group Comparisons (T-tests, ANOVA, Chi-Square, etc.).
2. Multivariate Analysis (ANCOVA) to control for confounders.
3. Power Analysis & Sample Size Calculation.
4. Z-Score Calculation (Reference Standardization).
5. Educational Content display.

With multiply imputed data (``mi``, see ui.components.mi_mode) the group comparison and the ANCOVA are
pooled over the imputed datasets (app_pages.analysis_pooled).

### def `_observed_agreement_note` (epidemiology.py)

- **Arguments**: ``

- **Returns**: `None`

Diagnostic accuracy and agreement with imputed data selected: what is analysed, and why it is not pooled.

### def `run_univariate_analysis` (epidemiology.py)

- **Arguments**: `df, analyzer, transformed`

- **Returns**: `None`

No description available.

### def `run_multivariate_analysis` (epidemiology.py)

- **Arguments**: `df, analyzer`

- **Returns**: `None`

No description available.

### def `show_ancova` (epidemiology.py)

- **Arguments**: `df, res, target, group, covariates`

- **Returns**: `None`

ANCOVA results (biostats.linear): Type II table, interpretation, adjusted means, coefficients, diagnostics.

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
`app_pages.home.app`, `ui.components.page_header`, `ui.components.population.population_banner`

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
`json`, `time`, `io.BytesIO`, `duckdb`, `numpy`, `pandas`, `streamlit`, `explore.corr_matrix`, `core.fingerprint.frame_fingerprint`, `graph.builders.correlation.build_correlation_graph`, `graph.legend.render_legend`, `graph.model.LayoutHint`, `graph.model.RenderOptions`, `graph.ui.current_theme`, `graph.ui.render_graph`, `graph.ui.render_graph_exports`, `graph.ui.renderer_selector`, `manage.db_manager.DBManager`, `manage.rag_manager.RAGManager`, `ui.components.confirm`, `ui.components.no_dataset`, `ui.progress.mark_step`, `ui.state.get_state`, `utils.data_analyzer.DataAnalyzer`, `utils.export_utils.to_excel`, `utils.export_utils.to_excel_sheets`, `utils.statistics_utils.normality_test`, `utils.statistics_utils.sanitize_dataframe_for_arrow`

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

_AI engine configuration (Settings page).

Provider-agnostic controls for the AI providers (local Ollama by default; Google Gemini, OpenAI, Mistral
and Anthropic Claude as cloud providers that require an explicit data-sharing consent; Claude has no
embedding model, so another provider embeds the documents), model and embedding
selection, the document library, retrieval parameters and engine initialisation. Laid out as cards:
provider and models, data sharing and document library on the left; retrieval, the vector database
(LanceDB or Qdrant, see ``manage.vectorstores``) and the engine on the right. Widget keys are stable
because other pages read them from the session._

**Imports**:
`os`, `shutil`, `pathlib.Path`, `streamlit`, `dotenv.load_dotenv`, `core.workspace.remove_pdf`, `core.workspace.safe_filename`, `manage.rag.ConnectorFactory`, `manage.rag.ProviderConfig`, `manage.rag.ProviderType`, `manage.vectorstores.ENGINE_SUMMARIES`, `manage.vectorstores.ENGINE_TITLES`, `manage.vectorstores.IndexManifest`, `manage.vectorstores.MachineProfile`, `manage.vectorstores.VectorEngine`, `manage.vectorstores.VectorStoreSettings`, `manage.vectorstores.estimate_chunks`, `manage.vectorstores.probe_machine`, `manage.vectorstores.qdrant_embedded_warnings`, `manage.vectorstores.recommend`, `manage.vectorstores.resolve_engine`, `manage.vectorstores.qdrant_index.QdrantIndex`, `manage.vectorstores.qdrant_index.user_collection`, `ui.components.confirm`, `utils.data_paths.get_vectors_dir`, `utils.session_scope.get_current_scope`

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

### def `_ensure_manager` (rag_sidebar.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `_provider_card` (rag_sidebar.py)

- **Arguments**: ``

- **Returns**: `dict`

Provider, connection and model choice; returns the engine configuration.

### def `_claude_embedding_block` (rag_sidebar.py)

- **Arguments**: `cfg: dict`

- **Returns**: `tuple[ComplexType]`

Claude has no embedding model: choose the provider that embeds the documents (local Ollama by default).

### def `_consent_card` (rag_sidebar.py)

- **Arguments**: `provider: str`

- **Returns**: `bool`

Explicit, session-scoped confirmation before any data can be sent to a cloud provider (audit E6).

### def `_library_card` (rag_sidebar.py)

- **Arguments**: ``

- **Returns**: `ComplexType`

Upload PDFs into the user's document library, choose the ones to index and remove the others (the
index is rebuilt from the remaining documents at the next initialisation: ``manage.vectorstores.index_matches``).

### def `_retrieval_card` (rag_sidebar.py)

- **Arguments**: ``

- **Returns**: `tuple[ComplexType]`

No description available.

### def `_machine` (rag_sidebar.py)

- **Arguments**: `storage: str`

- **Returns**: `MachineProfile`

CPU, memory and free disk, probed at most every five minutes.

### def `_vectors_dir` (rag_sidebar.py)

- **Arguments**: ``

- **Returns**: `Path`

No description available.

### def `_vector_card` (rag_sidebar.py)

- **Arguments**: `selected_files: ComplexType`

- **Returns**: `dict`

Choice of the vector database for this machine and library; returns the engine settings.

### def `_engine_card` (rag_sidebar.py)

- **Arguments**: `cfg: dict, consent: bool, adherence: float, temperature: float, selected_files: ComplexType, vector: dict`

- **Returns**: `None`

No description available.

### def `render_ai_settings` (rag_sidebar.py)

- **Arguments**: ``

- **Returns**: `None`

AI provider, keys, consent, document library, retrieval parameters and engine (Settings page).

---

## reproduce.py

_Reproduce page: traces, artifacts, replay and reports._

**Imports**:
`app_pages.reproduce_analysis.app`, `ui.components.page_header`

---

## reproduce_analysis.py

_No module description._

**Imports**:
`difflib`, `json`, `os`, `pandas`, `streamlit`, `core.lineage.verify_trace`, `core.workspace.safe_filename`, `ingest.loaders.UPLOAD_TYPES`, `manage.reproduction_manager.run_trace`, `manage.trace_documenter.TraceDocumenter`, `ui.progress.current_trace_session`, `ui.progress.mark_step`, `ui.state.get_state`, `utils.data_paths.get_traces_dir`, `utils.data_paths.get_uploads_dir`, `utils.path_utils.normalize_path`, `utils.path_utils.resolve_path`, `utils.session_scope.get_current_scope`

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

### def `render_trace_integrity` (reproduce_analysis.py)

- **Arguments**: `trace: dict`

- **Returns**: `None`

The hash chain of the trace (schema 2): verified, not verifiable (older traces) or broken.

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

_Settings page: AI engine and consent, graph engine, import defaults, workspace maintenance._

**Imports**:
`streamlit`, `ui.components.confirm`, `ui.components.page_header`, `ui.shell.clear_workspace_statistics`, `ui.state.get_state`, `utils.optional.extra_installed`, `utils.optional.missing_message`

---

## start.py

_Start page: welcome banner, dataset figures, the three-step workflow, saved versions, sample cohort, AI status._

**Imports**:
`streamlit`, `core.progress.DONE`, `core.progress.STALE`, `ui.components.page_header`, `ui.components.page_link`, `ui.components.status_chip`, `ui.components.switch_to`, `ui.components.workflow_map.workflow_map`, `ui.progress.current_progress`, `ui.state.SAMPLE_DICTIONARY_PATH`, `ui.state.TUTORIAL_COHORT_PATH`, `ui.state.TUTORIAL_CONFIG_PATH`, `ui.state.TUTORIAL_DICTIONARY_PATH`, `ui.state.TUTORIAL_FOLLOWUP_PATH`, `ui.state.get_state`, `ui.workflow.BRANCHES`

### def `progress_chip` (start.py)

- **Arguments**: `step: str`

- **Returns**: `None`

The step's status for the current data, as on the map.

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

## study_dossier.py

_Reproduce > Study dossier (follow-up programme M3): the participant flow of the analysis population with the
analyses added to the report, the list of those results (current, or computed on other data or another
population), and the Word study dossier that documents them._

**Imports**:
`__future__.annotations`, `platform`, `tomllib`, `importlib.metadata.PackageNotFoundError`, `importlib.metadata.version`, `pathlib.Path`, `typing.Any`, `pandas`, `streamlit`, `core.attrition.AnalysisCount`, `core.attrition.describe_definition`, `core.lineage.verify_trace`, `ui.components.no_dataset`, `ui.components.section`, `ui.components.status_chip`, `ui.components.flow.current_flow`, `ui.components.flow.render_flow`, `ui.results.CURRENT`, `ui.results.STALE_DATA`, `ui.results.ReportItem`, `ui.results.remove_from_report`, `ui.results.report_items`, `ui.results.report_status`, `ui.state.get_state`

### def `imputations_of` (study_dossier.py)

- **Arguments**: `value: Any`

- **Returns**: `ComplexType`

m when the result pools multiply imputed datasets (biostats.mi.analyze.MIResult), else None.

### def `analysis_counts` (study_dossier.py)

- **Arguments**: `items: list[ReportItem]`

- **Returns**: `list[AnalysisCount]`

What each result of the report analysed, for the boxes under the population.

### def `_stale_text` (study_dossier.py)

- **Arguments**: `status: str`

- **Returns**: `str`

No description available.

### def `_app_version` (study_dossier.py)

- **Arguments**: ``

- **Returns**: `str`

No description available.

### def `_versions` (study_dossier.py)

- **Arguments**: ``

- **Returns**: `str`

No description available.

### def `_provenance_rows` (study_dossier.py)

- **Arguments**: `state, trace: dict`

- **Returns**: `list[tuple[ComplexType]]`

No description available.

### def `_criteria_table` (study_dossier.py)

- **Arguments**: `population`

- **Returns**: `pd.DataFrame`

No description available.

### def `_data_steps` (study_dossier.py)

- **Arguments**: `trace: dict`

- **Returns**: `list[tuple[ComplexType]]`

No description available.

### def `_missing_table` (study_dossier.py)

- **Arguments**: `frame: pd.DataFrame`

- **Returns**: `pd.DataFrame`

No description available.

### def `_mi_section` (study_dossier.py)

- **Arguments**: `state`

- **Returns**: `ComplexType`

No description available.

### def `session_parts` (study_dossier.py)

- **Arguments**: `state, items: list[ReportItem]`

- **Returns**: `dict[ComplexType]`

What the dossier needs from the session, read during the page run: the download builds the document later,
outside the run, where the session is not available (finish_input and build_dossier are pure).

### def `finish_input` (study_dossier.py)

- **Arguments**: `parts: dict[ComplexType]`

- **Returns**: `None`

The DossierInput from the session parts: result tables, figures and references (no session needed).

### def `dossier_input` (study_dossier.py)

- **Arguments**: `state, items: list[ReportItem]`

- **Returns**: `None`

Everything the Word dossier needs (manage.dossier.DossierInput), read from the session now.

### def `app` (study_dossier.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `_word_bytes` (study_dossier.py)

- **Arguments**: `parts: dict[ComplexType]`

- **Returns**: `bytes`

No description available.

### def `build_dossier_bytes` (study_dossier.py)

- **Arguments**: `state, items: list[ReportItem]`

- **Returns**: `bytes`

No description available.

---

## survival.py

_Survival & regression page: Kaplan-Meier, log-rank, Cox, logistic and count regression._

**Imports**:
`streamlit`, `app_pages.analysis_survival.run_survival`, `ui.components.no_dataset`, `ui.components.page_header`, `ui.components.mi_mode.mi_banner`, `ui.components.population.population_banner`, `ui.state.get_state`, `utils.data_analyzer.DataAnalyzer`

---

## transformation_logic.py

_Compatibility shim: the transformation logic lives in :mod:`enrich.transformation_logic` (no UI code)._

**Imports**:
`enrich.transformation_logic.apply_variable_transformation`

---

## users.py

_Users page (administrators)._

**Imports**:
`app_pages.users_management.app`, `ui.components.page_header`

---

## users_management.py

_Users page (Admin > Users), for administrators only

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

Render the Users page.

---

## validate.py

_Validate page: data quality dashboard, anomalies, inclusion criteria, outliers, version comparison._

**Imports**:
`app_pages.data_monitoring.app`, `ui.components.page_header`

---

## variable_map.py

_Variable map page: taxonomy of variables, formulas, clustering and the knowledge graph._

**Imports**:
`streamlit`, `app_pages.data_insight.app`, `ui.components.page_header`, `utils.optional.extra_installed`, `utils.optional.missing_message`

---

## visualization.py

_No module description._

**Imports**:
`functools`, `numpy`, `pandas`, `plotly.express`, `plotly.figure_factory`, `plotly.graph_objects`, `streamlit`, `plotly.subplots.make_subplots`, `scipy.stats`, `utils.export_utils`, `utils.visualization_utils`, `app_pages.data_monitoring`, `manage.db_manager.DBManager`, `ui.components.no_dataset`, `utils.data_analyzer.DataAnalyzer`

### def `app` (visualization.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

---

## visualize.py

_Visualize page: distributions, categorical charts, time series, relationships (correlations, clustermap, ROC)._

**Imports**:
`app_pages.visualization.app`, `ui.components.page_header`, `ui.components.population.population_banner`

---

## which_test.py

_Which test? Four questions -> the recommended analysis, why, and where to run it._

**Imports**:
`streamlit`, `ui.components.glossary_popover`, `ui.components.page_header`, `ui.components.page_link`, `ui.components.section`, `ui.components.status_chip`, `ui.guidance.DESIGNS`, `ui.guidance.DISTRIBUTIONS`, `ui.guidance.GROUPS`, `ui.guidance.OUTCOMES`, `ui.guidance.recommend`

---
