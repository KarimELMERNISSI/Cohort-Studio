"""Prompt for mapping external formula variables to dataset keys."""


def resolve_formula_links(
    unresolved_vars: str,
    dataset_desc: str
) -> str:
    """
    Maps external variables in formulas to existing dataset variables.
    
    Args:
        unresolved_vars: List of unresolved variable names
        dataset_desc: Dataset dictionary with key: standard name pairs
    
    Returns:
        Formatted prompt string
    """
    return f"""Role: Data Mapping Expert.

Task: Map 'External' variables found in formulas to existing variables in the dataset.

External Variables (Unresolved):
{unresolved_vars}

Dataset Dictionary (Key: Standard Name):
{dataset_desc}

Instructions:
For each External Variable, determine if it corresponds to an existing Dataset Key (likely via synonym or abbreviation).

Return JSON mapping:
{{
    "External_Name_1": "Dataset_Key_X", 
    "External_Name_2": null
}}"""
