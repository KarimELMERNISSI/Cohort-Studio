"""Prompt for converting Python formulas to LaTeX notation."""


def markdown_formula(formulas_json: str) -> str:
    """
    Converts Python formulas to LaTeX/Markdown notation.
    
    Args:
        formulas_json: JSON string of {name: formula} pairs
    
    Returns:
        Formatted prompt string
    """
    return f"""Task: Convert these Python formulas into standard LaTeX/Markdown mathematical notation.

Input Formulas:
{formulas_json}

Instructions:
1. Return a JSON object where keys are the variable names and values are the LaTeX strings.
2. Use standard LaTeX notation (e.g. \\frac{{}}, \\sqrt{{}}, \\times).
3. Remove 'np.' prefixes.
4. Use readable variable names (remove underscores if it improves readability).
5. Do NOT wrap in $$ or $.

Example:
Input: "np.sqrt( Weight_kg / (Height_m ** 2) )"
Output: "\\sqrt{{\\frac{{Weight}}{{Height^2}}}}" """
