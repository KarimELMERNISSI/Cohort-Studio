"""Prompt for analyzing document context and defining expert personas."""


def context_analysis(
    sample_text: str,
    columns_context: str = "",
    columns_suffix: str = ""
) -> str:
    """
    Analyzes documents to determine domain and define expert roles.
    
    Args:
        sample_text: Sample text from documents (first 3 chunks)
        columns_context: Optional dataset variables context
        columns_suffix: Suffix for column reference in prompt
    
    Returns:
        Formatted prompt string
    """
    return f"""Analyze the following text samples from a set of documents{columns_suffix}. 

Your task is to identify the context and define expert personas that would be best suited to answer questions about this data, depending on how strictly they must adhere to the documents.

Return the result strictly as a JSON object with the following structure:
{{
    "domain": "The specific medical or scientific domain (e.g. Cardiology, Oncology)",
    "context_description": "A concise description (max 2 sentences) of the study type and document nature.",
    "roles": {{
        "strict": "A role title for high adherence (e.g. Clinical Data Auditor)",
        "balanced": "A role title for balanced adherence (e.g. Principal Investigator)",
        "creative": "A role title for low adherence/high creativity (e.g. Senior Medical Consultant)"
    }}
}}

Text Samples:
{sample_text}
{columns_context}"""
