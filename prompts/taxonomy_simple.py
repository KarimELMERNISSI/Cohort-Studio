"""Prompt for creating variable taxonomy mapping."""


def taxonomy_simple(
    columns_str: str,
    context_text: str,
    manual_renames_str: str = "",
    deep_instructions: str = "",
    json_structure_extra: str = ""
) -> str:
    """
    Creates a taxonomy mapping for dataset variables.
    
    Args:
        columns_str: JSON string of column information
        context_text: Context from documents (data dictionaries/protocols)
        manual_renames_str: User-provided renamings as ground truth
        deep_instructions: Additional instructions for deep analysis
        json_structure_extra: Additional JSON fields for deep analysis
    
    Returns:
        Formatted prompt string
    """
    return f"""Role: Medical Data Standardizer.

Task: Create a taxonomy mapping for the provided dataset variables.
The goal is to map potentially cryptic or non-standard variable names to their Standard Medical Concept.

Input Variables:
{columns_str}

Context from Documents (Data Dictionaries / Protocols):
{context_text}

{manual_renames_str}

Instructions:
1. Analyze each variable name and its statistics/values to infer its meaning.
2. USE THE CONTEXT from documents to find exact definitions if available.
3. Map it to a Standard Medical Concept (e.g. "sbp_val" -> "Systolic Blood Pressure").
4. Provide a brief description.
5. Return a JSON object where keys are the ORIGINAL variable names.
{deep_instructions}

JSON Output Format:
{{
    "original_var_name": {{
        "standard_name": "Standard Concept Name",
        "node_type": "Input",
        "description": "Brief description of what this variable represents.",
        "category": "Demographics/Vitals/Labs/etc"{json_structure_extra}
    }},
    ...
}}"""
