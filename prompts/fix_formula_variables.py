"""Prompt for fixing formula variable names to match dataset."""


def fix_formula_variables(
    formula: str,
    missing: list,
    columns: list
) -> str:
    """
    Fixes formula variables that don't match the dataset.
    
    Args:
        formula: The formula string with incorrect variable names
        missing: List of variable names not found in dataset
        columns: List of available column names
    
    Returns:
        Formatted prompt string
    """
    return f"""Role: Code Fixer.
The formula "{formula}" contains variables that do not match the dataset: {missing}.

Available Variables: {columns}

Task: Rewrite the formula by replacing the missing variables with their exact counterparts from the Available Variables list.
- Fix typos, case sensitivity, or slight naming variations (e.g. 'Weight' -> 'Weight_kg').
- Ensure Python/NumPy syntax (use 'np.' for functions).
- If a variable name contains spaces, enclose it in double double-quotes (e.g. ""Variable Name"").
- If a variable is truly missing and has no match, keep it as is.

Return ONLY the corrected formula string."""
