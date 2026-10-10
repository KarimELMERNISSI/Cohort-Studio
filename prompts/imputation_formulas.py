"""Prompt for suggesting physiological/clinical imputation formulas."""


def imputation_formulas(
    scope_desc: str,
    columns_str: str,
    target_variable: str,
    stats_context: str = "",
    hint_instruction: str = "",
    num_suggestions: int = 3
) -> str:
    """
    Suggests physiological formulas or regression-like heuristics to estimate missing values.
    
    Args:
        scope_desc: Description of column scope (e.g., 'Available Columns')
        columns_str: Comma-separated list of column names
        target_variable: Name of the variable to impute
        stats_context: Optional statistics context string for unit/range matching
        hint_instruction: Optional user hint instruction string
        num_suggestions: Number of distinct formula options to request
    
    Returns:
        Formatted prompt string
    """
    return f"""Role: Medical Data Expert.
Task: Suggest physiological formulas or regression-like heuristics to ESTIMATE missing values for '{target_variable}'.

{scope_desc} for Input: [{columns_str}]{stats_context}
Target Variable to Impute: {target_variable}
{hint_instruction}

Instructions:
1. Identify standard relationships where '{target_variable}' is the OUTPUT.
2. Example: If Target is 'Weight', suggest 'BMI * (Height**2)'.
3. If statistics show mismatched units (e.g. g/L vs mg/dL), include the conversion factor in the formula.
4. Only suggest formulas using the provided {scope_desc}.
5. Provide {num_suggestions} distinct options.
6. **CRITICAL SYNTAX RULES**:
   - Use **Python/Pandas** syntax ONLY. 
   - **DO NOT** use SQL (No `CASE WHEN`).
   - **DO NOT** use `df['col']` or `df.col`. Refer to columns directly.
   - For column names with **spaces, dots (.), or special characters**, you MUST enclose them in **double double quotes** (e.g. `""LDLc.2""`, `""My Var""`).
   - For simple column names (letters/numbers/underscores only), use them directly (e.g. `Weight`, `BMI_2`).
   - For conditional logic, use `np.where(condition, value_if_true, value_if_false)`.
   - Example: `np.where(""LDLc.2"" == 'mmol/L', ""LDLc.1"" * 0.387, ""LDLc.1"")`
   - Use `**` for power (e.g. `Height**2`), not `^`.

Return JSON Object:
{{
    "suggestions": [
        {{
            "name": "Imputed_{target_variable}",
            "formula": "Formula using available columns",
            "reasoning": "Why this relationship holds (e.g. standard Definition)",
            "confidence": "High/Medium/Low"
        }}
    ]
}}"""
