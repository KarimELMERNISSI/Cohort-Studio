"""Prompt for proposing which values of a categorical variable are one category written another way."""


def category_harmonisation(column: str, values_text: str, max_proposals: int = 30) -> str:
    """
    Asks which values as written are the same category: a synonym, a translation, an abbreviation, a brand and a
    generic name. Nothing is applied: each proposal is checked here, then offered to the user.

    Args:
        column: the variable's name
        values_text: one line per value as written (the values the payload policy lets be sent; no count, no row)
        max_proposals: the most proposals to return

    Returns:
        The prompt.
    """
    return f"""Role: Clinical data manager.
Task: The variable "{column}" holds the values below, each written as it is in the data. Some may be ONE category
written another way: a synonym, a translation (Femme and Female; Oui and Yes), an abbreviation (MI and myocardial
infarction), a brand and its generic name (Doliprane and paracetamol), British and American spellings. Propose, for
such a value, the value it is the same category as.

Values:
{values_text}

Rules:
1. Join two values only when they name the SAME category. Never join two categories, even one letter apart:
   Normal and Abnormal, Stable and Unstable, Hypotension and Hypertension, Dysphasia and Dysphagia, HFrEF and HFmrEF,
   Stage IIIA and IIIB, Type 1 and Type 2, NYHA II and III are different categories.
2. "value" is one of the values listed, exactly as written. "clean" is preferably one of the values listed (the
   most standard spelling); otherwise the standard name of the category.
3. Never propose that a value means missing or unknown, never group categories into a broader one (no "other",
   no class), and never correct a value you are not sure of.
4. "reason" says in a few words why they are the same category ("French for female", "brand of paracetamol").
5. At most {max_proposals} proposals; none when every value is its own category.

Return JSON:
{{"proposals": [{{"value": "as written", "clean": "what it becomes", "reason": "a few words"}}]}}"""
