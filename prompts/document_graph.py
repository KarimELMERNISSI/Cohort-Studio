"""Prompt for extracting knowledge graph from documents."""


def document_graph() -> str:
    """
    Extracts a knowledge graph of entities and relationships from a document.
    
    Returns:
        Formatted prompt string (no parameters needed - uses file context)
    """
    return """Role: Expert Information Architect.
Task: Analyze this document and extract a Knowledge Graph of key entities and their relationships.

Instructions:
1. Identify core entities (Concepts, Methods, Metrics, Findings, Diseases, Treatments).
2. Identify relationships between them.
3. NAMING CONVENTION: Use the **Canonical/Standard** name for each entity. 
   - E.g., Use "Heart Failure" instead of "HF". 
   - Deduplicate within the document (do not create separate nodes for acronyms).
4. EXHAUSTIVE EXTRACTION: For each entity, scan the ENTIRE document.
   - Collect ALL page numbers.
   - Select the BEST definition and representative quote.
5. SCIENTIFIC SUMMARY: Analyze the document type (e.g. Clinical Study, Review, Protocol) and generate a structured summary.

Return JSON:
{
    "summary": {
        "title": "Inferred Document Title",
        "doc_type": "Study Type (e.g. Cohort Study, Review)",
        "objective": "Primary goal/hypothesis of the study",
        "methods": "Key methodology, population, study design",
        "key_findings": "Primary results and outcomes",
        "significance": "Clinical or scientific implications",
        "top_concepts": ["List of 3-5 most important concepts"]
    },
    "nodes": [
        {
            "id": "Canonical Name",
            "type": "Concept/Metric/Finding/etc",
            "description": "Comprehensive Definition",
            "source_text": "Representative quote...",
            "page_reference": "1, 3, 5"
        }
    ],
    "edges": [
        {
            "source": "Source Node ID",
            "target": "Target Node ID",
            "relation": "relationship_type",
            "description": "Context of relationship"
        }
    ],
    "formulas": [
        {
            "name": "Formula Name",
            "expression": "Math expression",
            "page": "Page X",
            "description": "Explanation"
        }
    ]
}"""
