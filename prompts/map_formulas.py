"""Prompt for mapping theoretical formulas to dataset columns."""


def map_formulas(
    columns_str: str,
    concepts_str: str,
    missing_instr: str,
    limit: int,
    taxonomy_context: str = "",
    category_instruction: str = ""
) -> str:
    """
    Maps theoretical formula concepts to actual dataset columns.
    
    Args:
        columns_str: Comma-separated list of available columns
        concepts_str: JSON string of theoretical concepts
        missing_instr: Instruction for handling missing variables
        limit: Number of top suggestions to return
        taxonomy_context: Optional taxonomy context for variable understanding
        category_instruction: Optional instruction for classifying into existing categories
    
    Returns:
        Formatted prompt string
    """
    return f"""Role: Expert Data Engineer.
Task: Implement the following Theoretical Concepts using the Available Dataset Columns.

Available Columns: [{columns_str}]
{taxonomy_context}
Theoretical Concepts: {concepts_str}

CRITICAL SYNTAX RULES (Interpreter Constraints):
1. **AUTHORIZED OPERATORS**: You may use: +, -, *, /, ** (for power), (, ).
2. **FUNCTIONS**: You MUST use the 'np.' prefix for mathematical functions.
   - Correct: np.sqrt(x), np.log(x), np.exp(x), np.abs(x)
   - Wrong: sqrt(x), log(x), ln(x), square_root(x)
3. **SPACING**: You MUST put a single space around every operator.
   - Correct: " ( Weight / Height ) ** 2 "
   - Wrong: "Weight/Height**2"
4. **VARIABLE NAMES**: 
   - If a variable name contains spaces or special characters, you MUST enclose it in double double-quotes.
   - Example: ""Weight (kg)"" / ""Height (m)""
   - If it is a simple name, you can use it directly: Weight / Height

Instructions:
1. Map 'required_inputs' to 'Available Columns'.
2. Handle Unit Conversions (e.g. m to cm, lbs to kg) directly in the formula.
3. {missing_instr}
4. Select the top {limit} feasible suggestions.
5. **CRITICAL**: You MUST preserve the 'source' and 'logic' information from the Theoretical Concepts into 'source_citation' and 'source_explanation'.
{category_instruction}

Return JSON Object:
{{
    "suggestions": [
        {{
            "name": "snake_case_name",
            "title": "Readable Title",
            "formula": " Spaced Formula ",
            "missing_variables": ["list", "if", "any"],
            "description": "Clinical relevance",
            "suggestion_category": "Category Name",
            "source_type": "Document" or "Model Knowledge" or "Hybrid",
            "source_citation": "Filename.pdf (Pages X, Y) or 'Model Knowledge'",
            "source_explanation": "Briefly explain the logic or source of the formula."
        }}
    ]
}}"""
