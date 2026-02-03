# LLM Prompts and Templates

## anomaly_criteria.py

_Prompt for generating anomaly and inclusion criteria._

### def `anomaly_criteria_prompt` (anomaly_criteria.py)

- **Arguments**: `description: str, columns_info: str, sample_data: str, mode: str`

- **Returns**: `str`

Generates a prompt for converting natural language to boolean expressions.

Args:
    description: User's natural language description of the criteria.
    columns_info: List of available columns in the dataframe.
    mode: 'anomaly' or 'inclusion'.

Returns:
    Formatted prompt string.

---

## column_renaming.py

_Prompt for suggesting standard medical terminology for column names._

### def `column_renaming` (column_renaming.py)

- **Arguments**: `current_role: str, adherence_guidance: str, task: str, columns_str: str, context_text: str`

- **Returns**: `str`

Suggests standard medical terminology for dataset columns.

Args:
    current_role: The current expert role (e.g., "Medical Researcher")
    adherence_guidance: Guidance based on adherence score
    task: The specific task description (standardization or literature-based)
    columns_str: Comma-separated list of column names
    context_text: Context retrieved from documents

Returns:
    Formatted prompt string

---

## context_analysis.py

_Prompt for analyzing document context and defining expert personas._

### def `context_analysis` (context_analysis.py)

- **Arguments**: `sample_text: str, columns_context: str, columns_suffix: str`

- **Returns**: `str`

Analyzes documents to determine domain and define expert roles.

Args:
    sample_text: Sample text from documents (first 3 chunks)
    columns_context: Optional dataset variables context
    columns_suffix: Suffix for column reference in prompt

Returns:
    Formatted prompt string

---

## contextualize_variables.py

_Prompt for adding clinical context to variables._

### def `contextualize_variables` (contextualize_variables.py)

- **Arguments**: `vars_desc: str`

- **Returns**: `str`

Provides clinical context and detailed descriptions for variables.

Args:
    vars_desc: Description of variables (name, standard_name, values)

Returns:
    Formatted prompt string

---

## document_graph.py

_Prompt for extracting knowledge graph from documents._

### def `document_graph` (document_graph.py)

- **Arguments**: ``

- **Returns**: `str`

Extracts a knowledge graph of entities and relationships from a document.

Returns:
    Formatted prompt string (no parameters needed - uses file context)

---

## evaluation.py

_Evaluation prompts for RAG quality assessment using LLM-as-judge._

### def `evaluate_context_relevance` (evaluation.py)

- **Arguments**: `query: str, context: str`

- **Returns**: `str`

Evaluate how relevant the retrieved context is to the query.

Args:
    query: The original user query
    context: The retrieved context chunks

Returns:
    Prompt for LLM-as-judge

### def `evaluate_faithfulness` (evaluation.py)

- **Arguments**: `context: str, response: str`

- **Returns**: `str`

Evaluate if the response is faithful/grounded in the provided context.

Args:
    context: The source context used for generation
    response: The generated response

Returns:
    Prompt for LLM-as-judge

### def `evaluate_answer_relevance` (evaluation.py)

- **Arguments**: `query: str, response: str`

- **Returns**: `str`

Evaluate how relevant the response is to the original query.

Args:
    query: The original user query
    response: The generated response

Returns:
    Prompt for LLM-as-judge

### def `detect_hallucination` (evaluation.py)

- **Arguments**: `context: str, response: str`

- **Returns**: `str`

Detect if the response contains hallucinated (fabricated) information.

Args:
    context: The source context
    response: The generated response

Returns:
    Prompt for hallucination detection

### def `evaluate_json_structure` (evaluation.py)

- **Arguments**: `expected_keys: list, response: str`

- **Returns**: `str`

Evaluate if the JSON response has the expected structure.

Args:
    expected_keys: List of expected top-level keys
    response: The response containing JSON

Returns:
    Prompt for structure evaluation

### def `evaluate_formula_correctness` (evaluation.py)

- **Arguments**: `formula: str, expected_variables: list`

- **Returns**: `str`

Evaluate if a generated formula is mathematically correct.

Args:
    formula: The generated formula string
    expected_variables: List of variables that should be in the formula

Returns:
    Prompt for formula evaluation

---

## expand_search_hint.py

_Prompt for expanding medical acronyms._

### def `expand_search_hint` (expand_search_hint.py)

- **Arguments**: `search_hint: str`

- **Returns**: `str`

Expands medical acronyms to their full standard names.

Args:
    search_hint: The search term to expand (may be an acronym)

Returns:
    Formatted prompt string

---

## fix_formula_variables.py

_Prompt for fixing formula variable names to match dataset._

### def `fix_formula_variables` (fix_formula_variables.py)

- **Arguments**: `formula: str, missing: list, columns: list`

- **Returns**: `str`

Fixes formula variables that don't match the dataset.

Args:
    formula: The formula string with incorrect variable names
    missing: List of variable names not found in dataset
    columns: List of available column names

Returns:
    Formatted prompt string

---

## formula_enrichment.py

_Prompt for identifying formula relationships between variables._

### def `formula_enrichment` (formula_enrichment.py)

- **Arguments**: `current_role: str, vars_desc: str`

- **Returns**: `str`

Identifies standard medical formulas and scores related to variables.

Args:
    current_role: The current expert role
    vars_desc: Description of variables (name, standard_name, values)

Returns:
    Formatted prompt string

---

## graph_metadata.py

_Prompt for identifying semantic relationships for knowledge graph._

### def `graph_metadata` (graph_metadata.py)

- **Arguments**: `vars_desc: str`

- **Returns**: `str`

Analyzes variables to identify semantic relationships for a Knowledge Graph.

Args:
    vars_desc: Description of variables with standard names

Returns:
    Formatted prompt string

---

## map_formulas.py

_Prompt for mapping theoretical formulas to dataset columns._

### def `map_formulas` (map_formulas.py)

- **Arguments**: `columns_str: str, concepts_str: str, missing_instr: str, limit: int, taxonomy_context: str`

- **Returns**: `str`

Maps theoretical formula concepts to actual dataset columns.

Args:
    columns_str: Comma-separated list of available columns
    concepts_str: JSON string of theoretical concepts
    missing_instr: Instruction for handling missing variables
    limit: Number of top suggestions to return
    taxonomy_context: Optional taxonomy context for variable understanding

Returns:
    Formatted prompt string

---

## markdown_formula.py

_Prompt for converting Python formulas to LaTeX notation._

### def `markdown_formula` (markdown_formula.py)

- **Arguments**: `formulas_json: str`

- **Returns**: `str`

Converts Python formulas to LaTeX/Markdown notation.

Args:
    formulas_json: JSON string of {name: formula} pairs

Returns:
    Formatted prompt string

---

## refine_taxonomy.py

_Prompt for refining taxonomy based on user feedback._

### def `refine_taxonomy` (refine_taxonomy.py)

- **Arguments**: `context_block: str`

- **Returns**: `str`

Refines taxonomy variables based on user feedback.

Args:
    context_block: Block of context for each variable to refine

Returns:
    Formatted prompt string

---

## resolve_formula_links.py

_Prompt for mapping external formula variables to dataset keys._

### def `resolve_formula_links` (resolve_formula_links.py)

- **Arguments**: `unresolved_vars: str, dataset_desc: str`

- **Returns**: `str`

Maps external variables in formulas to existing dataset variables.

Args:
    unresolved_vars: List of unresolved variable names
    dataset_desc: Dataset dictionary with key: standard name pairs

Returns:
    Formatted prompt string

---

## taxonomy_simple.py

_Prompt for creating variable taxonomy mapping._

### def `taxonomy_simple` (taxonomy_simple.py)

- **Arguments**: `columns_str: str, context_text: str, manual_renames_str: str, deep_instructions: str, json_structure_extra: str`

- **Returns**: `str`

Creates a taxonomy mapping for dataset variables.

Args:
    columns_str: JSON string of column information
    context_text: Context from documents (data dictionaries/protocols)
    manual_renames_str: User-provided renamings as ground truth
    deep_instructions: Additional instructions for deep analysis
    json_structure_extra: Additional JSON fields for deep analysis

Returns:
    Formatted prompt string

---

## theoretical_formulas.py

_Prompt for identifying theoretical medical formulas._

### def `theoretical_formulas` (theoretical_formulas.py)

- **Arguments**: `current_role: str, adherence_guidance: str, context_text: str, task_desc: str, count: int, scoping_instruction: str`

- **Returns**: `str`

Identifies standard medical formulas based on context.

Args:
    current_role: The current expert role
    adherence_guidance: Guidance based on adherence score
    context_text: Context from documents
    task_desc: Description of the specific task
    count: Number of concepts to generate
    scoping_instruction: Optional constraint on available variables

Returns:
    Formatted prompt string

---

## unify_synonyms.py

_Prompt for identifying and merging synonym variables._

### def `unify_synonyms` (unify_synonyms.py)

- **Arguments**: `new_keys_str: str, exist_sample: str`

- **Returns**: `str`

Identifies synonyms and maps them to canonical IDs.

Args:
    new_keys_str: Comma-separated list of new variable keys
    exist_sample: Sample of existing variable keys for context

Returns:
    Formatted prompt string

---

## wise_enrichment.py

_Prompt for wise enrichment LLM inference step._

### def `wise_enrichment` (wise_enrichment.py)

- **Arguments**: `level: int, rag_context: str, vars_context: str, adherence_instruction: str`

- **Returns**: `str`

Identifies new formulas and variables for taxonomy enrichment.

Args:
    level: Current enrichment level (1, 2, etc.)
    rag_context: Context retrieved from documents
    vars_context: Current variables with standard names
    adherence_instruction: Instruction based on adherence score

Returns:
    Formatted prompt string

---
