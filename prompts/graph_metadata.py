"""Prompt for identifying semantic relationships for knowledge graph."""


def graph_metadata(vars_desc: str) -> str:
    """
    Analyzes variables to identify semantic relationships for a Knowledge Graph.
    
    Args:
        vars_desc: Description of variables with standard names
    
    Returns:
        Formatted prompt string
    """
    return f"""Role: Medical Data Scientist.

Task: Analyze the variables to identify semantic relationships (correlations, risk factors) for a Knowledge Graph.

Variables:
{vars_desc}

Instructions:
For each variable, identify OTHER variables from the list that are directly related (e.g., risk factors, co-morbidities).
Do NOT analyze formulas (we already have those). Focus on clinical associations.

JSON Output Format:
{{
    "original_var_name": ["related_var_1", "related_var_2"],
    ...
}}"""
