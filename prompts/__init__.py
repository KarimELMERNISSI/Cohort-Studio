"""
Prompt Templates for the RAG System.

This module provides all LLM prompt templates as Python functions with f-strings.
Each function returns a formatted prompt string ready to be sent to the LLM.

Usage:
    from prompts import context_analysis, column_renaming
    prompt = context_analysis(sample_text="...", columns_context="...")
"""

from .anomaly_criteria import anomaly_criteria_prompt
from .column_renaming import column_renaming
from .context_analysis import context_analysis
from .contextualize_variables import contextualize_variables
from .document_graph import document_graph

# Evaluation prompts
from .evaluation import (
    detect_hallucination,
    evaluate_answer_relevance,
    evaluate_context_relevance,
    evaluate_faithfulness,
    evaluate_formula_correctness,
    evaluate_json_structure,
)
from .expand_search_hint import expand_search_hint
from .fix_formula_variables import fix_formula_variables
from .formula_enrichment import formula_enrichment
from .graph_metadata import graph_metadata
from .imputation_formulas import imputation_formulas
from .map_formulas import map_formulas
from .markdown_formula import markdown_formula
from .refine_taxonomy import refine_taxonomy
from .resolve_formula_links import resolve_formula_links
from .taxonomy_simple import taxonomy_simple
from .theoretical_formulas import theoretical_formulas
from .unify_synonyms import unify_synonyms
from .wise_enrichment import wise_enrichment

__all__ = [
    "context_analysis",
    "column_renaming", 
    "taxonomy_simple",
    "formula_enrichment",
    "contextualize_variables",
    "resolve_formula_links",
    "unify_synonyms",
    "graph_metadata",
    "document_graph",
    "wise_enrichment",
    "expand_search_hint",
    "theoretical_formulas",
    "map_formulas",
    "imputation_formulas",
    "fix_formula_variables",
    "markdown_formula",
    "refine_taxonomy",
    # Evaluation
    "evaluate_context_relevance",
    "evaluate_faithfulness",
    "evaluate_answer_relevance",
    "detect_hallucination",
    "evaluate_json_structure",
    "evaluate_formula_correctness",
]

