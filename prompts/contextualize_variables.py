"""Prompt for adding clinical context to variables."""


def contextualize_variables(vars_desc: str) -> str:
    """
    Provides clinical context and detailed descriptions for variables.
    
    Args:
        vars_desc: Description of variables (name, standard_name, values)
    
    Returns:
        Formatted prompt string
    """
    return f"""Role: Medical Expert.

Task: Provide clinical context and detailed descriptions for the provided variables.

Variables:
{vars_desc}

Instructions:
For each variable, provide:
1. "description": A clear, medical definition.
2. "clinical_usage": How this variable is used in clinical practice (e.g., diagnosis, monitoring, prognosis).
3. "category": A broad category (e.g., Demographics, Vitals, Lab Test, Comorbidity).
4. "topic": A specific medical topic (e.g., "Cardiovascular Health", "Renal Function", "Diabetes Management").
5. "proxy_variables": A list of potential proxy variables or synonyms often used interchangeably or as surrogates.

Return JSON:
{{
    "original_var_name": {{
        "description": "...",
        "clinical_usage": "...",
        "category": "...",
        "topic": "...",
        "proxy_variables": ["...", "..."]
    }}
}}"""
