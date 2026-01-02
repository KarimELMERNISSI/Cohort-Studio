"""Prompt for wise enrichment LLM inference step."""


def wise_enrichment(
    level: int,
    rag_context: str,
    vars_context: str,
    adherence_instruction: str
) -> str:
    """
    Identifies new formulas and variables for taxonomy enrichment.
    
    Args:
        level: Current enrichment level (1, 2, etc.)
        rag_context: Context retrieved from documents
        vars_context: Current variables with standard names
        adherence_instruction: Instruction based on adherence score
    
    Returns:
        Formatted prompt string
    """
    return f"""Role: Clinical Knowledge Expert.

Task: "Wise Enrichment" of a Clinical Knowledge Graph (Level {level}).

Goal: Identify potential NEW formulas or scores that could be calculated from the Current Variables.

Context from Documents:
{rag_context}

Current Variables (Format: ID: "Standard Name"):
{vars_context}

Instructions:
1. {adherence_instruction}
2. **Synonym Check**: Check the 'Current Variables' list CAREFULLY. If a concept exists (e.g. 'Body Height'), USE THAT ID. Do NOT create a duplicate (e.g. 'height_cm').
3. **ID Format**: Use snake_case for IDs (e.g. `body_mass_index`, NOT `calc_bmi`). Avoid prefixes like `calc_` or `derived_`.
4. **Formula Check**: Do not propose formulas that strictly duplicate existing ones. Variants are OK (e.g. BSA DuBois vs BSA Mosteller), but exact duplicates (BMI vs Body Mass Index) are NOT.
5. Look for standard medical scores (e.g. BMI, eGFR) where we have SOME of the variables.
6. Propose the MISSING external variables needed.

Return JSON with the NEW variables and NEW formulas:
{{
    "variables": {{
        "new_variable_id_snake_case": {{
            "standard_name": "Standard Name",
            "description": "Why this is needed",
            "node_type": "Input-External", 
            "category": "Suggested Category",
            "clinical_usage": "Reason for inclusion"
        }}
    }},
    "formulas": {{
        "new_formula_id": {{
            "name": "Formula Name (e.g. BMI)",
            "description": "Calculation logic",
            "output_variable": "output_var_id",
            "input_variables": ["input_var_id_1", "input_var_id_2"]
        }}
    }}
}}"""
