"""Prompt for identifying theoretical medical formulas."""


def theoretical_formulas(
    current_role: str,
    adherence_guidance: str,
    context_text: str,
    task_desc: str,
    count: int,
    scoping_instruction: str = ""
) -> str:
    """
    Identifies standard medical formulas based on context.
    
    Args:
        current_role: The current expert role
        adherence_guidance: Guidance based on adherence score
        context_text: Context from documents
        task_desc: Description of the specific task
        count: Number of concepts to generate
        scoping_instruction: Optional constraint on available variables
    
    Returns:
        Formatted prompt string
    """
    return f"""Role: {current_role}
Constraint: {adherence_guidance}

Context from Documents:
{context_text}

{scoping_instruction}

Task: {task_desc}

Generate {count} distinct medical concepts. For each, provide the standard mathematical formula using standard variable names (e.g. 'Weight', 'Height').

Return JSON List:
[
    {{
        "concept_name": "Name of the variable",
        "standard_formula": "Mathematical formula using standard terms",
        "required_inputs": ["List", "of", "standard", "inputs"],
        "clinical_relevance": "Why is this useful?",
        "source": "Exact filename and page number if from context, else 'Standard Medical Knowledge'",
        "logic": "Brief explanation of the formula derivation"
    }}
]"""
