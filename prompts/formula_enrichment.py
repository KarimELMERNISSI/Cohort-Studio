"""Prompt for identifying formula relationships between variables."""


def formula_enrichment(
    current_role: str,
    vars_desc: str
) -> str:
    """
    Identifies standard medical formulas and scores related to variables.
    
    Args:
        current_role: The current expert role
        vars_desc: Description of variables (name, standard_name, values)
    
    Returns:
        Formatted prompt string
    """
    return f"""Role: {current_role} & Expert System.

Task: Identify standard medical formulas and scores that relate to the provided variables.

Variables:
{vars_desc}

Instructions:
1. Identify known medical formulas (e.g., BMI, eGFR, HAS-BLED, CHA2DS2-VASc, Unit Conversions) that involve these variables.
2. Create a "Formula Registry" entry for each.
3. Link variables to these formulas explicitly.

Return JSON structure:
{{
    "formulas": [
        {{
            "id": "unique_slug_id", 
            "name": "Display Name",
            "description": "Brief description",
            "expression": "Mathematical expression or rule description",
            "input_variables": ["original_var_name_1", "original_var_name_2"],
            "output_variable": "original_var_name_result"
        }}
    ],
    "variable_updates": {{
        "original_var_name": {{
            "involved_in_formulas": ["formula_id_1"] 
        }}
    }}
}}"""
