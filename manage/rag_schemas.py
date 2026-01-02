"""
RAG Response Schemas.

Pydantic models for structured LLM outputs, enabling type-safe parsing
and Gemini's native response_schema enforcement.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class TheoreticalConcept(BaseModel):
    """Schema for a theoretical medical concept/formula."""
    concept_name: str = Field(..., description="Name of the medical variable or score")
    standard_formula: str = Field(..., description="Mathematical formula using standard terms")
    required_inputs: List[str] = Field(default_factory=list, description="List of required input variables")
    clinical_relevance: str = Field(default="", description="Why this is clinically useful")
    source: str = Field(default="Standard Medical Knowledge", description="Citation or source")
    logic: str = Field(default="", description="Brief explanation of formula derivation")


class TheoreticalConceptList(BaseModel):
    """Container for multiple theoretical concepts."""
    concepts: List[TheoreticalConcept] = Field(default_factory=list)


class ComputedVariableSuggestion(BaseModel):
    """Schema for a computed variable suggestion mapped to dataset columns."""
    name: str = Field(..., description="snake_case variable name")
    title: str = Field(..., description="Human-readable title")
    formula: str = Field(..., description="Python/NumPy formula with proper spacing")
    missing_variables: List[str] = Field(default_factory=list, description="Variables not found in dataset")
    description: str = Field(default="", description="Clinical relevance")
    source_type: str = Field(default="Model Knowledge", description="Document, Model Knowledge, or Hybrid")
    source_citation: str = Field(default="Model Knowledge", description="Filename.pdf (Pages X, Y) or Model Knowledge")
    source_explanation: str = Field(default="", description="Logic or source explanation")
    markdown_formula: Optional[str] = Field(default=None, description="LaTeX representation")
    original_formula: Optional[str] = Field(default=None, description="Formula before correction")


class SuggestionResponse(BaseModel):
    """Container for computed variable suggestions."""
    suggestions: List[ComputedVariableSuggestion] = Field(default_factory=list)
    error: Optional[str] = Field(default=None, description="Error message if any")


class ProxyVariable(BaseModel):
    """Schema for proxy variable suggestion."""
    proxy_found: bool = Field(default=False)
    proxy_name: Optional[str] = Field(default=None)
    formula: Optional[str] = Field(default=None)
    explanation: Optional[str] = Field(default=None)


class AlternativeFormula(BaseModel):
    """Schema for alternative formula suggestion."""
    alternative_found: bool = Field(default=False)
    alternative_name: Optional[str] = Field(default=None)
    formula: Optional[str] = Field(default=None)
    explanation: Optional[str] = Field(default=None)


class ExpandedTerm(BaseModel):
    """Schema for expanded medical term."""
    expanded_name: str = Field(..., description="Full standard name of the term")
    original_term: str = Field(..., description="Original input term")


class FormulaCorrection(BaseModel):
    """Schema for corrected formula."""
    corrected_formula: str = Field(..., description="The corrected formula string")
    changes_made: List[str] = Field(default_factory=list, description="List of changes made")


class MarkdownFormulas(BaseModel):
    """Schema for LaTeX formula conversions."""
    formulas: dict = Field(default_factory=dict, description="Map of name to LaTeX formula")
