"""Prompt for expanding medical acronyms."""


def expand_search_hint(search_hint: str) -> str:
    """
    Expands medical acronyms to their full standard names.
    
    Args:
        search_hint: The search term to expand (may be an acronym)
    
    Returns:
        Formatted prompt string
    """
    return f"""Role: Medical Terminology Expert.
Input: "{search_hint}"

Task: 
1. If the input is a medical acronym or abbreviation (e.g. BSA, BMI, eGFR, SBP), return the STANDARD FULL NAME (e.g. Body Surface Area).
2. If it is already a full name or not a known medical acronym, return it exactly as is.

Return ONLY the expanded/standard name. No bolding, no extra text."""
