"""Prompt for refining taxonomy based on user feedback."""


def refine_taxonomy(context_block: str) -> str:
    """
    Refines taxonomy variables based on user feedback.
    
    Args:
        context_block: Block of context for each variable to refine
    
    Returns:
        Formatted prompt string
    """
    return f"""Role: Medical Data Expert & Taxonomy Refiner.

Task: Refine the taxonomy for the following variables based on specific USER FEEDBACK.

Instructions:
1. Review the "Current Taxonomy" and "USER FEEDBACK" for each variable.
2. Update the taxonomy fields (standard_name, description, category, clinical_usage, related_formulas, topic, proxy_variables) to address the feedback.
3. If the user corrects a unit, update the description and related_formulas (conversion) accordingly.
4. If the user corrects the meaning, update the standard_name and description.
5. Keep existing valid information if it doesn't conflict with the feedback.

Variables to Refine:
{context_block}

Return JSON:
{{
    "variable_name": {{
        "standard_name": "...",
        "description": "...",
        "category": "...",
        "clinical_usage": "...",
        "related_formulas": ["..."],
        "topic": "...",
        "proxy_variables": ["..."]
    }}
}}"""
