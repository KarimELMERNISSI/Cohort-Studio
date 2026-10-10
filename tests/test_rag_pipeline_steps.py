"""
Comprehensive Unit Tests for RAG Pipeline Steps.

Covers:
1. Formula AST parsing, variable extraction, normalization (^ to **), and quoting resilience.
2. Document knowledge graph node and edge formatting (yFiles map compatibility).
3. Citation page parsing resilience (int, list, string).
4. Multi-document graph extraction and merging with entity deduplication.
5. Modular prompt generation functions across the pipeline.
6. Pydantic validation schemas for formulas and provider configurations.
"""

import json
from unittest.mock import MagicMock

import pytest

from manage.rag.schemas import ProviderConfig, ProviderType, RAGConfig
from manage.rag_computed_vars import parse_formula_vars
from manage.rag_documents import DocumentsMixin
from manage.rag_schemas import (
    AlternativeFormula,
    ComputedVariableSuggestion,
    FormulaCorrection,
    ProxyVariable,
    SuggestionResponse,
    TheoreticalConcept,
    TheoreticalConceptList,
)
from prompts import (
    contextualize_variables,
    document_graph,
    graph_metadata,
    imputation_formulas,
    map_formulas,
    refine_taxonomy,
    resolve_formula_links,
    unify_synonyms,
    wise_enrichment,
)


class TestFormulaParsingAndValidation:
    """Test AST parsing, variable extraction, and normalization for formulas."""

    def test_parse_simple_arithmetic(self):
        formula = "weight / (height ** 2)"
        vars_found, valid = parse_formula_vars(formula)
        assert valid is True
        assert vars_found == {"weight", "height"}

    def test_parse_caret_power_normalization(self):
        formula = "weight / (height ^ 2)"
        vars_found, valid = parse_formula_vars(formula)
        assert valid is True
        assert vars_found == {"weight", "height"}

    def test_parse_double_double_quoted_variables(self):
        formula = '""Systolic BP"" - ""Diastolic BP""'
        vars_found, valid = parse_formula_vars(formula)
        assert valid is True
        assert vars_found == {"Systolic BP", "Diastolic BP"}

    def test_parse_backtick_quoted_variables(self):
        formula = "`Heart Rate` / 60 + `Age (yr)` * 0.1"
        vars_found, valid = parse_formula_vars(formula)
        assert valid is True
        assert vars_found == {"Heart Rate", "Age (yr)"}

    def test_parse_numpy_mathematical_calls(self):
        formula = "np.log(creatinine) * 1.5 + np.sqrt(bun)"
        vars_found, valid = parse_formula_vars(formula)
        assert valid is True
        assert "creatinine" in vars_found
        assert "bun" in vars_found
        assert "np" in vars_found

    def test_parse_syntax_error_resilience(self):
        formula = "weight / (height ** )"
        vars_found, valid = parse_formula_vars(formula)
        assert valid is False
        assert vars_found == set()

    def test_parse_empty_formula(self):
        vars_found, valid = parse_formula_vars("")
        assert valid is False or len(vars_found) == 0


from unittest.mock import MagicMock, patch

class TestDocumentKnowledgeGraphFormatting:
    """Test graph structure compatibility with yFiles visual specifications."""

    @patch("manage.rag_documents.os.path.exists", return_value=True)
    @patch("manage.rag.PDFDocumentLoader")
    def test_document_graph_extraction_formatting(self, mock_loader_cls, mock_exists):
        class DummyDocManager(DocumentsMixin):
            def __init__(self):
                self.initialized = True
                self.documents_dir = "DOCUMENTS"
                self.llm = MagicMock()
                self.llm.client = None

            def _clean_json_response(self, text):
                return text

            def _parse_json_safe(self, text):
                return json.loads(text)

        doc_mgr = DummyDocManager()

        sample_graph = {
            "nodes": [
                {
                    "id": "creatinine",
                    "label": "Serum Creatinine",
                    "type": "Input-Internal",
                    "role": "input-internal",
                    "citation": "kidney.pdf (Page 2)",
                },
                {
                    "id": "egfr",
                    "label": "eGFR CKD-EPI",
                    "type": "Derived-Internal",
                    "role": "derived-internal",
                    "citation": "kidney.pdf (Page 4)",
                },
            ],
            "edges": [
                {
                    "source": "creatinine",
                    "target": "egfr",
                    "label": "input",
                    "style": "solid",
                    "color": "#4285F4",
                    "directed": True,
                }
            ],
        }

        doc_mgr.llm.invoke.return_value = MagicMock(content=json.dumps(sample_graph))
        mock_doc = MagicMock()
        mock_doc.metadata = {"page": 2}
        mock_doc.page_content = "Serum Creatinine is used to compute eGFR CKD-EPI."
        mock_loader_instance = MagicMock()
        mock_loader_instance.load_single_pdf.return_value = [mock_doc]
        mock_loader_cls.return_value = mock_loader_instance

        graph, err = doc_mgr.extract_custom_graph_from_doc("kidney.pdf")
        assert err is None
        assert len(graph["nodes"]) == 2
        assert len(graph["edges"]) == 1

        node_types = {n["type"] for n in graph["nodes"]}
        assert "Input-Internal" in node_types
        assert "Derived-Internal" in node_types

        edge = graph["edges"][0]
        assert edge["source"] == "creatinine"
        assert edge["target"] == "egfr"
        assert edge["label"] == "input"
        assert edge["directed"] is True

    def test_citation_page_handling_resilience(self):
        class DummyDocManager(DocumentsMixin):
            def __init__(self):
                self.initialized = True
                self.documents_dir = "DOCUMENTS"
                self.llm = MagicMock()

            def _clean_json_response(self, text):
                return text

            def _parse_json_safe(self, text):
                return json.loads(text)

        doc_mgr = DummyDocManager()

        # Mock extract_custom_graph_from_doc to return heterogeneous citation page types
        def mock_extract(doc_name, progress_callback=None):
            if doc_name == "doc1.pdf":
                return {
                    "nodes": [
                        {"id": "var_a", "label": "Variable A", "type": "Input-Internal", "page_reference": 3}
                    ],
                    "edges": [],
                }, None
            elif doc_name == "doc2.pdf":
                return {
                    "nodes": [
                        {"id": "var_b", "label": "Variable B", "type": "Derived-Internal", "page_reference": [1, 5]}
                    ],
                    "edges": [],
                }, None
            else:
                return {
                    "nodes": [
                        {"id": "var_c", "label": "Variable C", "type": "Derived-External", "page_reference": "2, 4"}
                    ],
                    "edges": [],
                }, None

        doc_mgr.extract_custom_graph_from_doc = mock_extract

        graph, err = doc_mgr.extract_merged_graph_from_docs(["doc1.pdf", "doc2.pdf", "doc3.pdf"])
        assert err is None
        assert "nodes" in graph
        assert len(graph["nodes"]) == 3
        # Check citations are aggregated in the 'citations' list without type errors
        for node in graph["nodes"]:
            assert "citations" in node
            assert len(node["citations"]) > 0
            for cit in node["citations"]:
                assert isinstance(cit["page"], str)


class TestPromptModularityAndContract:
    """Verify modular prompt functions produce structured, non-empty text."""

    def test_imputation_formulas_prompt(self):
        prompt = imputation_formulas(
            scope_desc="Available Columns",
            columns_str="height, weight, age",
            target_variable="bmi",
            stats_context="",
            hint_instruction="Focus on adult population",
            num_suggestions=3,
        )
        assert "target_variable" in prompt or "bmi" in prompt
        assert "height, weight, age" in prompt
        assert "suggestions" in prompt

    def test_map_formulas_prompt(self):
        prompt = map_formulas(
            columns_str="height_m, weight_kg, age",
            concepts_str='[{"concept_name": "BMI", "standard_formula": "weight / height^2"}]',
            missing_instr="Allow missing variables.",
            limit=5,
            taxonomy_context="Context",
            category_instruction="Assign categories to formulas.",
        )
        assert "BMI" in prompt
        assert "height_m, weight_kg, age" in prompt
        assert "suggestions" in prompt

    def test_wise_enrichment_prompt(self):
        prompt = wise_enrichment(
            level=1,
            rag_context="Clinical trials context",
            vars_context="var1: Systolic BP",
            adherence_instruction="Adhere strictly to evidence",
        )
        assert "Level 1" in prompt
        assert "new_variables" in prompt
        assert "new_formulas" in prompt

    def test_refine_taxonomy_prompt(self):
        prompt = refine_taxonomy(context_block="Variable: sbp\nFeedback: change unit to mmHg")
        assert "USER FEEDBACK" in prompt
        assert "sbp" in prompt
        assert "standard_name" in prompt

    def test_document_graph_prompt(self):
        prompt = document_graph()
        assert "Knowledge Graph" in prompt
        assert "nodes" in prompt
        assert "edges" in prompt

    def test_graph_enrichment_prompts(self):
        meta_prompt = graph_metadata(
            vars_desc="creatinine (Serum Creatinine): lab marker\negfr (Estimated GFR): kidney function"
        )
        assert "creatinine" in meta_prompt
        assert "Knowledge Graph" in meta_prompt

        context_prompt = contextualize_variables(
            vars_desc="sbp (Systolic BP): 120\ndbp (Diastolic BP): 80"
        )
        assert "sbp" in context_prompt
        assert "clinical_usage" in context_prompt

        links_prompt = resolve_formula_links(
            unresolved_vars="creatinine_clearance",
            dataset_desc="cr: Serum Creatinine\nage: Patient Age",
        )
        assert "creatinine_clearance" in links_prompt
        assert "Dataset Dictionary" in links_prompt

        syn_prompt = unify_synonyms(
            new_keys_str="bmi, body_mass_index",
            exist_sample="height, weight",
        )
        assert "bmi, body_mass_index" in syn_prompt
        assert "canonical_id" in syn_prompt


class TestPydanticSchemas:
    """Verify type safety and validation of RAG models and provider configurations."""

    def test_theoretical_concept_validation(self):
        concept = TheoreticalConcept(
            concept_name="BMI",
            standard_formula="weight / (height ** 2)",
            required_inputs=["weight", "height"],
            clinical_relevance="Body mass index screening",
        )
        assert concept.concept_name == "BMI"
        assert len(concept.required_inputs) == 2

        container = TheoreticalConceptList(concepts=[concept])
        dumped = container.model_dump()
        assert len(dumped["concepts"]) == 1

    def test_computed_variable_suggestion_validation(self):
        suggestion = ComputedVariableSuggestion(
            name="map_score",
            title="Mean Arterial Pressure",
            formula="""(2 * ""Diastolic BP"" + ""Systolic BP"") / 3""",
            missing_variables=[],
            description="Cardiovascular perfusion estimation",
            source_type="Document",
            source_citation="icu_guidelines.pdf (Page 12)",
        )
        assert suggestion.name == "map_score"
        assert suggestion.source_type == "Document"

        response = SuggestionResponse(suggestions=[suggestion])
        assert len(response.suggestions) == 1
        assert response.error is None

    def test_provider_config_sanitization(self):
        cfg = ProviderConfig(
            provider=ProviderType.GEMINI,
            model_name="gemini-1.5-flash",
            base_url="https://generativelanguage.googleapis.com/",
        )
        assert cfg.base_url == "https://generativelanguage.googleapis.com"
        assert cfg.temperature == 0.3

    def test_rag_config_defaults(self):
        rag_cfg = RAGConfig()
        assert rag_cfg.chunk_size == 1000
        assert rag_cfg.search_type == "mmr"
        assert rag_cfg.adherence_score == 0.5
