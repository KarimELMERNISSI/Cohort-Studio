"""Prompt for identifying and merging synonym variables."""


def unify_synonyms(
    new_keys_str: str,
    exist_sample: str
) -> str:
    """
    Identifies synonyms and maps them to canonical IDs.
    
    Args:
        new_keys_str: Comma-separated list of new variable keys
        exist_sample: Sample of existing variable keys for context
    
    Returns:
        Formatted prompt string
    """
    return f"""Role: Clinical Data Standardizer.
Task: Identify synonyms in a list of variable IDs and map them to a single canonical ID.

New Variables: {new_keys_str}
Existing Variables (Context): {exist_sample}

Instructions:
1. Look for synonyms among "New Variables" (e.g. 'bmi', 'body_mass_index').
2. Look for synonyms between "New Variables" and "Existing Variables".
3. If a synonym exists, choose the MOST STANDARD medical acronym or name as the 'canonical_id'.
4. If the canonical ID is already in "Existing Variables", map to that.
5. If duplicates found (e.g. 'bsa_dubois' and 'bsa_mosteller' are NOT synonyms, keep both), do NOT map distinct variants.

Return JSON mapping {{ "alias_id": "canonical_id" }}
Only include entries that need re-mapping."""
