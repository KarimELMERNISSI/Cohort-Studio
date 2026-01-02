"""Prompt for suggesting standard medical terminology for column names."""


def column_renaming(
    current_role: str,
    adherence_guidance: str,
    task: str,
    columns_str: str,
    context_text: str
) -> str:
    """
    Suggests standard medical terminology for dataset columns.
    
    Args:
        current_role: The current expert role (e.g., "Medical Researcher")
        adherence_guidance: Guidance based on adherence score
        task: The specific task description (standardization or literature-based)
        columns_str: Comma-separated list of column names
        context_text: Context retrieved from documents
    
    Returns:
        Formatted prompt string
    """
    return f"""Role: {current_role}
Constraint: {adherence_guidance}

Task: {task}

Input Data (Columns & Stats): 
{columns_str}

Context from Documents:
{context_text}

Instructions:
1. Analyze the Input Data and Context.
2. Suggest a new name ONLY if the current name is ambiguous, non-standard, or can be improved.
3. Return a JSON object where keys are the ORIGINAL names and values are the NEW names.
4. Return ONLY valid JSON. No markdown formatting, no explanations outside the JSON.

JSON Output:"""
