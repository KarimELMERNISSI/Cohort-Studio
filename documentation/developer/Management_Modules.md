# Backend Management Logic

## anonymize.py

_No module description._

---

## db_manager.py

_No module description._

**Imports**:
`duckdb`, `glob`, `re`, `os`, `logging`, `pandas`

### class `DBManager` (db_manager.py)

Helper class to manage DuckDB connections and data persistence using Parquet with versioning

**Methods:**

- ****init****(`self, db_path`) -> `None`
  > No description available.

- **_get_connection**(`self`) -> `None`
  > Get a new DuckDB connection.

- **_get_next_version**(`self, base_name`) -> `None`
  > Determines the next version number for a given base name.

- **save_dataframe**(`self, df, name, folder`) -> `None`
  > Persists a pandas DataFrame to a Parquet file via DuckDB (Low level).

- **save_dataset**(`self, df, base_name`) -> `None`
  > Persists a pandas DataFrame to a Parquet file with automatic versioning.

- **load_dataframe**(`self, name, folder`) -> `None`
  > Loads a Parquet file into a pandas DataFrame via DuckDB (Low level).

- **load_dataset**(`self, file_name`) -> `None`
  > Loads a Parquet file into a pandas DataFrame via DuckDB.
  >
  >
  > **Parameters**:
  >
  > file_name : str
  > The name of the file to load (without extension)
  >
  >
  > **Returns**:
  >
  > tuple
  > (DataFrame, message) - The loaded DataFrame and a success/error message

- **get_available_tables**(`self`) -> `None`
  > List available parquet files in the current directory.

- **get_available_datasets**(`self`) -> `None`
  > List available parquet files in the current directory, sorted by modification time.

- **save_stats**(`self, df, dataset_name, stats_type`) -> `None`
  > Save statistical results to parquet cache.

- **load_stats**(`self, dataset_name, stats_type`) -> `None`
  > Load statistical results from parquet cache.

- **clear_stats**(`self, dataset_name`) -> `None`
  > Delete cached statistics for a given dataset to force recalculation.

- **clear_all_stats**(`self`) -> `None`
  > Delete ALL cached statistics files.

---

## file_handling.py

_No module description._

**Imports**:
`os`, `pandas`, `detect_delimiter.detect`, `logging`

### def `create_folders` (file_handling.py)

- **Arguments**: `config`

- **Returns**: `None`

Create folders based on the configuration.

Parameters:
    config (dict): Configuration dictionary containing folder names.

Returns:
    None

### def `list_files_in_directory` (file_handling.py)

- **Arguments**: `directory, file_extensions`

- **Returns**: `None`

Lists files with specified extensions in a directory.

Parameters:

- directory (str): The path to the directory.

- file_extensions (list of str, optional): List of file extensions to filter. Default is None.

Returns:

- list of str: List of file paths that match the specified extensions.

If no file_extensions are provided, all files in the directory are listed.

### def `detect_encoding` (file_handling.py)

- **Arguments**: `file_path, num_lines`

- **Returns**: `None`

Detects the encoding of a file using charset-normalizer.

Parameters:

- file_path (str): The path to the file.

- num_lines (int): The number of lines to read from the file for encoding detection. Default is 100.

Returns:

- str: The detected encoding of the file.

Note:
The function attempts to detect the encoding of the file by analyzing a portion of its content.
It reads the specified number of lines and uses the charset-normalizer library to determine the encoding.

### def `load_csv_with_separator` (file_handling.py)

- **Arguments**: `file_path, encoding, num_lines`

- **Returns**: `None`

Load a CSV file with automatic delimiter detection.

- file_path (str): The path to the CSV file.

- encoding (str): The encoding of the file.

- num_lines (int): The number of lines to read for delimiter detection. Default is 10.

Returns:

- pd.DataFrame: The loaded DataFrame.

### def `load_dataframe` (file_handling.py)

- **Arguments**: `file_path, encoding`

- **Returns**: `None`

Loads a DataFrame from a file.

Parameters:

- file_path (str): The path to the file to load.

- encoding (str, optional): The encoding of the file. Default is 'utf-8'.

Returns:

- DataFrame or None: The loaded DataFrame if successful, otherwise None.

This function attempts to load a DataFrame from the specified file. It first detects the file extension
to determine the file type. For CSV files, it uses a custom function to handle loading with the specified
encoding. For Excel files, it uses pandas read_excel method. If the file format is unsupported or an error
occurs during loading, it logs an error message and returns None.

### def `auto_name` (file_handling.py)

- **Arguments**: `input_file_path, config, output_extension, descriptive`

- **Returns**: `None`

Generate an output file path based on the input file path.

Parameters:
    input_file_path (str): The path to the input file.
    output_extension (str): The extension of the output file. Default is 'csv'.
    descriptive (bool): Whether to use a descriptive folder. Default is False.

Returns:
    str: The output file path.

Raises:
    None

---

## rag_computed_vars.py

_Computed Variables Mixin for RAGManager.

Contains all computed variable suggestion and validation methods.
This is part of the RAGManager class composition pattern._

**Imports**:
`json`, `re`, `ast`, `hashlib`, `logging`, `rag_schemas.TheoreticalConcept`, `rag_schemas.TheoreticalConceptList`, `rag_schemas.ComputedVariableSuggestion`, `rag_schemas.SuggestionResponse`, `rag_schemas.ProxyVariable`, `rag_schemas.AlternativeFormula`, `rag_schemas.FormulaCorrection`, `utils.llm_utils.parse_json_safe`, `utils.llm_utils.validate_and_parse`

### class `ComputedVarsMixin` (rag_computed_vars.py)

Mixin class containing computed variable functionality for RAGManager.

Methods:
    - _expand_search_hint

    - suggest_computed_variables

    - _get_theoretical_formulas

    - _map_formulas_to_data

    - suggest_computed_variables_with_validation

    - suggest_proxy_variable

    - suggest_alternative_formula

**Methods:**

- **_expand_search_hint**(`self, search_hint`) -> `None`
  > Pre-processing: Expand acronyms or ambiguous terms to standard medical concepts.

- **suggest_computed_variables**(`self, columns, search_hint, num_suggestions, suggestion_mode, allow_missing_variables, use_taxonomy, existing_categories, progress_callback`) -> `None`
  > Suggests computed variables based on dataset columns and medical knowledge.

- **_get_theoretical_formulas**(`self, search_hint, suggestion_mode, context_text, count, available_columns, use_taxonomy`) -> `None`
  > Step 1: Identify WHAT to calculate (Standard Medical Knowledge).

- **_map_formulas_to_data**(`self, concepts, columns, allow_missing, limit, use_taxonomy, existing_categories`) -> `None`
  > Step 2: Map theoretical inputs to actual dataset columns.

- **suggest_computed_variables_with_validation**(`self, columns, search_hint, num_suggestions, suggestion_mode, allow_missing_variables, use_taxonomy, existing_categories, progress_callback`) -> `None`
  > Wrapper that calls the pipeline and performs final AST validation with Auto-Correction.

- **suggest_proxy_variable**(`self, target_variable, available_columns`) -> `None`
  > Suggests a proxy variable when the target is missing.

- **suggest_alternative_formula**(`self, target_concept, missing_variable, available_columns`) -> `None`
  > Suggests an alternative formula avoiding a missing variable.

- **suggest_imputation_formulas**(`self, target_variable, available_columns, search_hint, context_columns, context_stats, num_suggestions`) -> `None`
  > Suggests formulas to impute missing values for a specific target variable
  > based on relationships with other available columns.
  >
  > Parameters:
  > - target_variable (str): Name of the variable to impute.
  > - available_columns (list): List of all available column names.
  > - search_hint (str): Optional user-provided hint (e.g. "Reverse BMI").
  > - context_columns (list): Optional list of specific columns selected by the user.
  > - context_stats (dict): Optional dictionary of statistics for the context columns.
  > - num_suggestions (int): Number of suggestions to generate.

---

## rag_documents.py

_Documents Mixin for RAGManager.

Contains all document processing, knowledge graph extraction and chat methods.
This is part of the RAGManager class composition pattern._

**Imports**:
`os`, `json`, `time`

### class `DocumentsMixin` (rag_documents.py)

Mixin class containing document processing functionality for RAGManager.

Methods:
    - get_available_documents

    - extract_custom_graph_from_doc

    - extract_merged_graph_from_docs

    - match_columns_to_graph

    - chat_with_specific_doc

**Methods:**

- **get_available_documents**(`self`) -> `None`
  > Returns a list of PDF files in the documents directory.

- **extract_custom_graph_from_doc**(`self, file_name, progress_callback`) -> `None`
  > Uses Gemini Native File API to extract a knowledge graph from a specific document.

- **extract_merged_graph_from_docs**(`self, doc_list, progress_callback`) -> `None`
  > Extracts and merges graphs from multiple documents.
  > Returns (merged_json, error).

- **match_columns_to_graph**(`self, columns, graph_json`) -> `None`
  > Matches dataset columns to Knowledge Graph nodes using fuzzy string matching.
  > Returns dict: {column_name: {match_found: bool, node: data, confidence: float}}

- **chat_with_specific_doc**(`self, doc_names, query`) -> `None`
  > Chat with specific document(s) using the vector store.
  > doc_names: string or list of strings.

---

## rag_evaluator.py

_RAG Evaluator Module

Provides comprehensive evaluation metrics for the RAG system including:

- Retrieval quality (context relevance, semantic similarity)

- Generation quality (faithfulness, answer relevance, hallucination detection)

- Embedding quality assessment

- Logging and tracking of evaluation results_

**Imports**:
`json`, `os`, `datetime.datetime`, `pathlib.Path`, `typing.Dict`, `typing.List`, `typing.Optional`, `typing.Any`, `typing.Tuple`, `hashlib`, `prompts.evaluation.evaluate_context_relevance`, `prompts.evaluation.evaluate_faithfulness`, `prompts.evaluation.evaluate_answer_relevance`, `prompts.evaluation.detect_hallucination`

### class `RAGEvaluator` (rag_evaluator.py)

Evaluates RAG system outputs for quality monitoring.

Uses LLM-as-judge approach for semantic evaluation metrics
and direct validation for structural metrics.

**Methods:**

- ****init****(`self, llm, log_dir: str`) -> `None`
  > Initialize the RAG Evaluator.
  >
  > Args:
  > llm: The LLM instance to use for evaluation (same as RAG system)
  > log_dir: Directory to store evaluation logs

- **_init_summary**(`self`) -> `None`
  > Initialize the metrics summary file.

- **_save_summary**(`self, summary: dict`) -> `None`
  > Save the metrics summary to file.

- **_load_summary**(`self`) -> `dict`
  > Load the metrics summary from file.

- **_generate_eval_id**(`self, query: str, timestamp: str`) -> `str`
  > Generate a unique evaluation ID.

- **evaluate_retrieval_quality**(`self, query: str, retrieved_chunks: List[str], k: int`) -> `Dict[ComplexType]`
  > Evaluate the quality of retrieved context.
  >
  > Args:
  > query: The original query
  > retrieved_chunks: List of retrieved text chunks
  > k: Number of top chunks to consider
  >
  > Returns:
  > Dictionary with relevance scores and analysis

- **calculate_semantic_similarity**(`self, query_embedding: List[float], chunk_embeddings: List[List[float]]`) -> `Dict[ComplexType]`
  > Calculate semantic similarity between query and chunks.
  >
  > Args:
  > query_embedding: The query's embedding vector
  > chunk_embeddings: List of chunk embedding vectors
  >
  > Returns:
  > Dictionary with similarity metrics

- **evaluate_generation_quality**(`self, query: str, context: str, response: str, function_name: str`) -> `Dict[ComplexType]`
  > Evaluate the quality of generated response.
  >
  > Args:
  > query: The original query
  > context: The retrieved context used
  > response: The generated response
  > function_name: Name of the RAG function being evaluated
  >
  > Returns:
  > Dictionary with quality scores

- **validate_json_output**(`self, response: str`) -> `Dict[ComplexType]`
  > Validate if the response contains valid JSON.
  >
  > Args:
  > response: The generated response
  >
  > Returns:
  > Dictionary with validation results

- **evaluate**(`self, query: str, context: str, response: str, function_name: str, retrieved_chunks: Optional[List[str]]`) -> `Dict[ComplexType]`
  > Run full evaluation on a RAG query-response pair.
  >
  > Args:
  > query: The original query
  > context: The retrieved context
  > response: The generated response
  > function_name: Name of the RAG function
  > retrieved_chunks: Optional list of individual chunks
  >
  > Returns:
  > Complete evaluation results

- **log_result**(`self, result: Dict[ComplexType]`) -> `None`
  > Log evaluation result to file and update summary.
  >
  > Args:
  > result: Evaluation result dictionary

- **_parse_evaluation_response**(`self, response: str, metric_type: str`) -> `Dict[ComplexType]`
  > Parse LLM evaluation response into structured format.

- **load_test_suite**(`self`) -> `List[Dict[ComplexType]]`
  > Load the test suite from file.

- **save_test_suite**(`self, test_cases: List[Dict[ComplexType]]`) -> `None`
  > Save test cases to file.

- **add_test_case**(`self, query: str, expected_concepts: List[str], ground_truth: Optional[str], expected_sources: Optional[List[str]]`) -> `None`
  > Add a new test case to the suite.

- **run_test_suite**(`self, rag_manager`) -> `Dict[ComplexType]`
  > Run all test cases against the RAG system.
  >
  > Args:
  > rag_manager: The RAGManager instance to test
  >
  > Returns:
  > Test suite results with pass/fail for each case

- **get_summary**(`self`) -> `Dict[ComplexType]`
  > Get the current metrics summary.

- **get_recent_evaluations**(`self, limit: int`) -> `List[Dict[ComplexType]]`
  > Get recent evaluation results.

- **get_metrics_by_date**(`self, days: int`) -> `Dict[ComplexType]`
  > Get metrics aggregated by date for the last N days.

---

## rag_manager.py

_RAG Manager - Core Module.

This is the main RAGManager class that composes functionality from mixin classes:

- TaxonomyMixin: Variable taxonomy generation and enrichment

- DocumentsMixin: Document processing and knowledge graph extraction

- ComputedVarsMixin: Computed variable suggestions

The mixin pattern allows splitting a large class into focused modules
while maintaining a single class interface for existing code._

**Imports**:
`os`, `shutil`, `streamlit`, `pathlib.Path`, `ast`, `json`, `hashlib`, `time`, `logging`, `re`, `rapidfuzz.process`, `prompts.context_analysis`, `prompts.column_renaming`, `prompts.taxonomy_simple`, `prompts.formula_enrichment`, `prompts.anomaly_criteria_prompt`, `rag_taxonomy.TaxonomyMixin`, `rag_documents.DocumentsMixin`, `rag_computed_vars.ComputedVarsMixin`, `utils.llm_utils.parse_json_safe`, `utils.llm_utils.validate_and_parse`, `utils.llm_utils.StructuredOutputHelper`, `utils.llm_utils.clean_json_response`

### class `RAGManager` (rag_manager.py)

Main RAG Manager class that composes functionality from mixins.

Core Methods (defined here):
    - **init**

    - is_available

    - get_available_models

    - _clean_json_response

    - _analyze_global_context

    - initialize_system

    - update_adherence_score

    - _update_internal_state

Taxonomy Methods (from TaxonomyMixin):
    - suggest_column_renaming

    - generate_variable_taxonomy

    - enrich_variable_taxonomy

    - refine_variable_taxonomy

Document Methods (from DocumentsMixin):
    - get_available_documents

    - extract_custom_graph_from_doc

    - extract_merged_graph_from_docs

    - match_columns_to_graph

    - chat_with_specific_doc

Computed Variable Methods (from ComputedVarsMixin):
    - suggest_computed_variables

    - suggest_computed_variables_with_validation

    - suggest_proxy_variable

    - suggest_alternative_formula

**Methods:**

- ****init****(`self, documents_dir, api_key`) -> `None`
  > No description available.

- **is_available**(`self`) -> `None`
  > Check if RAG dependencies are available.

- **get_available_models**(`self`) -> `None`
  > Get list of available Gemini models.

- **_clean_json_response**(`self, text`) -> `None`
  > Clean JSON output from LLM using robust parsing utilities.
  >
  > This method is a wrapper around llm_utils.clean_json_response
  > for backward compatibility with existing code.

- **_parse_json_safe**(`self, text, default`) -> `None`
  > Safely parse JSON with multiple fallback strategies.
  >
  > Uses llm_utils.parse_json_safe for robust parsing with:
  > - Automatic JSON extraction from markdown
  > - Trailing comma removal
  > - Bracket balancing
  > - AST literal_eval fallback

- **_get_structured_output_helper**(`self`) -> `None`
  > Get a StructuredOutputHelper instance for schema-validated LLM calls.

- **_analyze_global_context**(`self, texts, dataset_columns`) -> `None`
  > Analyzes a sample of the documents and dataset columns to determine the global context.

- **initialize_system**(`self, model_name, adherence_score, temperature, dataset_columns, use_existing_db, progress_callback, selected_files`) -> `None`
  > Initialize the RAG system with documents and LLM.

- **update_adherence_score**(`self, adherence_score`) -> `None`
  > Updates the adherence score and rebuilds the chain without full re-initialization.

- **_update_internal_state**(`self, adherence_score`) -> `None`
  > Updates internal state using the Modern LangChain 0.3 Architecture.

- **enrich_variable_taxonomy**(`self, current_taxonomy, columns_info, distance, progress_callback`) -> `None`
  > Enriches an existing taxonomy using 'Wise Enrichment' strategy (Formula-centric).
  > Returns (new_candidates, new_formulas, error) where new_candidates is a dict of proposed variables.
  > Does NOT merge automatically.

- **refine_variable_taxonomy**(`self, current_taxonomy, feedback_dict, columns_info`) -> `None`
  > Refines specific variables in the taxonomy based on user feedback.

- **suggest_anomaly_criteria**(`self, description, columns, sample_data, mode`) -> `None`
  > Suggests anomaly or inclusion criteria based on natural language description.
  > Returns a list of criteria dictionaries.

---

## rag_schemas.py

_RAG Response Schemas.

Pydantic models for structured LLM outputs, enabling type-safe parsing
and Gemini's native response_schema enforcement._

**Imports**:
`typing.List`, `typing.Optional`, `pydantic.BaseModel`, `pydantic.Field`

### class `TheoreticalConcept` (rag_schemas.py)

Schema for a theoretical medical concept/formula.

### class `TheoreticalConceptList` (rag_schemas.py)

Container for multiple theoretical concepts.

### class `ComputedVariableSuggestion` (rag_schemas.py)

Schema for a computed variable suggestion mapped to dataset columns.

### class `SuggestionResponse` (rag_schemas.py)

Container for computed variable suggestions.

### class `ProxyVariable` (rag_schemas.py)

Schema for proxy variable suggestion.

### class `AlternativeFormula` (rag_schemas.py)

Schema for alternative formula suggestion.

### class `ExpandedTerm` (rag_schemas.py)

Schema for expanded medical term.

### class `FormulaCorrection` (rag_schemas.py)

Schema for corrected formula.

### class `MarkdownFormulas` (rag_schemas.py)

Schema for LaTeX formula conversions.

---

## rag_taxonomy.py

_Taxonomy Mixin for RAGManager.

Contains all variable taxonomy generation and enrichment methods.
This is part of the RAGManager class composition pattern._

**Imports**:
`json`

### class `TaxonomyMixin` (rag_taxonomy.py)

Mixin class containing taxonomy-related functionality for RAGManager.

Methods:
    - suggest_column_renaming

    - generate_variable_taxonomy

    - _generate_simple_taxonomy

    - _generate_advanced_taxonomy

    - _merge_stats

    - _identify_standard_concepts

    - _enrich_with_formulas

    - _contextualize_variables

    - _resolve_formula_links

    - _unify_synonyms

    - _enrich_graph_metadata

    - enrich_variable_taxonomy

    - refine_variable_taxonomy

**Methods:**

- **suggest_column_renaming**(`self, columns, strategy, progress_callback`) -> `None`
  > Suggests standardized column names based on medical literature.

- **generate_variable_taxonomy**(`self, columns_info, existing_mapping, deep_analysis, progress_callback`) -> `None`
  > Generates a taxonomy mapping for the provided columns.
  > If deep_analysis is True, uses a multi-step pipeline for richer results.

- **_merge_stats**(`self, taxonomy, columns_info`) -> `None`
  > Merges statistical info from columns_info into the taxonomy.

- **_generate_simple_taxonomy**(`self, columns_info, existing_mapping, deep_analysis, progress_callback`) -> `None`
  > Generates a taxonomy mapping for the provided columns.
  > columns_info: List of dicts with 'name', 'type', 'stats', etc.
  > existing_mapping: Dict of manual renames (original -> new) to use as ground truth.

- **_generate_advanced_taxonomy**(`self, columns_info, existing_mapping, progress_callback`) -> `None`
  > Orchestrates the multi-step taxonomy generation pipeline.

- **_enrich_graph_metadata**(`self, taxonomy, columns_info`) -> `None`
  > Step 4: Identify node types and relationships for graph visualization.

- **_identify_standard_concepts**(`self, columns_info, existing_mapping`) -> `None`
  > Step 1: Map variables to standard medical concepts.

- **_enrich_with_formulas**(`self, taxonomy, columns_info`) -> `None`
  > Step 2: Identify formula relationships (Structured).

- **_contextualize_variables**(`self, taxonomy, columns_info`) -> `None`
  > Step 3: Add clinical usage and interpretation context.

- **_resolve_formula_links**(`self, taxonomy`) -> `None`
  > Step 4: Resolve 'External' formula inputs to internal keys via LLM.

- **_unify_synonyms**(`self, new_candidates, existing_keys`) -> `None`
  > Uses LLM to identify and merge synonyms within the new candidates
  > and against existing taxonomy keys.
  > Returns a mapping { 'alias_id': 'canonical_id' }.

---

## reproduction_manager.py

_No module description._

**Imports**:
`streamlit`, `pandas`, `json`, `enrich.external_data`, `app_pages.transformation_logic.apply_variable_transformation`, `manage.db_manager.DBManager`, `utils.path_utils.resolve_path`

### def `reproduce_trace` (reproduction_manager.py)

- **Arguments**: `trace_file`

- **Returns**: `None`

No description available.

---

## trace_documenter.py

_No module description._

**Imports**:
`os`, `json`, `datetime.datetime`, `docx.Document`, `docx.shared.Pt`, `docx.shared.Inches`, `docx.shared.RGBColor`, `docx.enum.text.WD_ALIGN_PARAGRAPH`, `docx.oxml.ns.qn`, `docx.oxml.OxmlElement`

### class `TraceDocumenter` (trace_documenter.py)

No description available.

**Methods:**

- ****init****(`self, trace_source`) -> `None`
  > Initialize with either a dictionary trace or a path to a json trace file.

- **generate_report**(`self`) -> `None`
  > Generates a Word document report of the transformation trace.
  > Returns the BytesIO object of the document.

- **_add_title**(`self, doc`) -> `None`
  > No description available.

- **_add_session_info**(`self, doc`) -> `None`
  > No description available.

- **_add_step**(`self, doc, index, step`) -> `None`
  > No description available.

- **_render_params**(`self, doc, params, indent`) -> `None`
  > No description available.

- **_add_row**(`self, table, label, value`) -> `None`
  > No description available.

- **_get_readable_function_name**(`self, func_name`) -> `None`
  > No description available.

- **_add_footer**(`self, doc`) -> `None`
  > No description available.

---

## transformation_manager.py

_No module description._

**Imports**:
`json`, `logging`, `datetime.datetime`, `pandas`, `streamlit`, `os`

### class `TransformationManager` (transformation_manager.py)

No description available.

**Methods:**

- ****init****(`self, trace_dir`) -> `None`
  > Initialize the TransformationManager.
  >
  >
  > **Parameters**:
  >
  > trace_dir : str
  > Directory where transformation traces will be stored.

- **initialize_session**(`self, dataset_name`) -> `None`
  > Initializes a new session trace.

- **get_trace_path**(`self`) -> `None`
  > Generates the trace file path based on the session ID.

- **add_step**(`self, function_name, params, description, output_dataset_path, stats_cat_path, stats_num_path, corr_matrix_path`) -> `None`
  > Records a transformation step with optional references to intermediate data.
  >
  >
  > **Parameters**:
  >
  > function_name : str
  > Name of the function/transformation applied.
  > params : dict
  > Dictionary of parameters used in the transformation.
  > description : str
  > Human-readable description of the step.
  > output_dataset_path : str, optional
  > Path to the output dataset (snapshot).
  > stats_cat_path : str, optional
  > Path to categorical statistics file.
  > stats_num_path : str, optional
  > Path to numerical statistics file.
  > corr_matrix_path : str, optional
  > Path to correlation matrix file.

- **_make_serializable**(`self, obj`) -> `None`
  > Recursively converts objects to JSON-serializable formats.

- **get_trace**(`self`) -> `None`
  > Returns the current transformation trace.

- **save_trace**(`self, filepath`) -> `None`
  > Saves the trace to a JSON file.

- **load_trace**(`self, filepath`) -> `None`
  > Loads a trace from a JSON file.

- **get_available_datasets_from_traces**(`self`) -> `None`
  > Scans all trace files in the trace directory and returns a list of available datasets.
  > Each entry contains: session_id, timestamp, step_description, file_path.

---
