"""Reference behaviour of the clinical knowledge-graph builder."""

import pytest

from graph.builders.taxonomy import (
    VariableNodeType,
    build_taxonomy_graph,
    make_category_id,
    make_formula_id,
    normalize_node_type,
    parse_formula_id,
)
from graph.model import GraphKind


@pytest.fixture
def taxonomy() -> dict:
    return {
        "ht_cm": {
            "standard_name": "Height",
            "node_type": "Input",
            "category": "Anthropometry",
            "description": "Height in cm",
        },
        "wt_kg": {
            "standard_name": "Weight",
            "node_type": "Input-Internal",
            "category": "Anthropometry",
            "relationships": ["ht_cm"],
        },
        "bmi": {
            "standard_name": "BMI",
            "node_type": "Derived-Internal",
            "category": "Anthropometry",
        },
        "death": {
            "standard_name": "Death",
            "node_type": "Outcome",
            "category": "Outcomes",
            "relationships": ["bmi"],
        },
        "ldl": {
            "standard_name": "LDL cholesterol",
            "node_type": "External-Derived",
            "category": "Lipids",
        },
    }


@pytest.fixture
def formulas() -> dict:
    return {
        "bmi_formula": {
            "name": "Body mass index",
            "input_variables": ["Height", "Weight"],
            "output_variable": "BMI",
            "expression": "wt_kg / (ht_cm/100)**2",
        },
        "FORM_risk": {
            "name": "Risk score",
            "input_variables": ["BMI", "age"],
            "output_variable": "death",
        },
        "unrelated": {"name": "Other", "input_variables": ["x"], "output_variable": "y"},
    }


class TestVocabulary:
    def test_node_type_aliases(self) -> None:
        assert normalize_node_type("Input") is VariableNodeType.INPUT_INTERNAL
        assert normalize_node_type("derived") is VariableNodeType.DERIVED_INTERNAL
        assert normalize_node_type("External-Derived") is VariableNodeType.DERIVED_EXTERNAL
        assert normalize_node_type("Input-External") is VariableNodeType.INPUT_EXTERNAL
        assert normalize_node_type("OUTCOME") is VariableNodeType.OUTCOME
        assert normalize_node_type(None) is VariableNodeType.INPUT_INTERNAL
        assert normalize_node_type("something external") is VariableNodeType.INPUT_EXTERNAL

    def test_formula_id_is_idempotent(self) -> None:
        assert make_formula_id("x") == "FORM_x"
        assert make_formula_id("FORM_x") == "FORM_x"  # regression: no FORM_FORM_x
        assert parse_formula_id("FORM_x") == "x" and parse_formula_id("x") is None
        assert make_category_id("Lipids") == "CAT_Lipids"


class TestBuilder:
    def test_structured_graph(self, taxonomy: dict, formulas: dict) -> None:
        spec = build_taxonomy_graph(taxonomy, formulas)
        assert spec.kind == GraphKind.TAXONOMY and spec.directed
        ids = spec.node_ids()
        # categories first (sorted), then variables (taxonomy order + externals), then formulas (sorted)
        assert ids[:3] == ("CAT_Anthropometry", "CAT_Lipids", "CAT_Outcomes")
        assert ids[3:8] == ("ht_cm", "wt_kg", "bmi", "death", "ldl")
        assert ids[8] == "age"  # external input created by the risk formula
        assert ids[9:] == ("FORM_bmi_formula", "FORM_risk")
        assert spec.node("ldl").category == "derived_external"
        assert (
            spec.node("death").category == "outcome"
        )  # outcome is preserved even though a formula yields it
        assert spec.node("bmi").category == "derived"
        assert spec.node("age").category == "input_external"
        assert spec.node("FORM_bmi_formula").data["formula_key"] == "bmi_formula"
        assert spec.node("FORM_risk").data["registry_key"] == "risk"

    def test_edges_resolved_by_standard_name_and_deduplicated(
        self, taxonomy: dict, formulas: dict
    ) -> None:
        spec = build_taxonomy_graph(taxonomy, formulas)
        edge_ids = {e.id for e in spec.edges}
        assert (
            "input:ht_cm->FORM_bmi_formula" in edge_ids
            and "input:wt_kg->FORM_bmi_formula" in edge_ids
        )
        assert "yields:FORM_bmi_formula->bmi" in edge_ids
        assert "yields:FORM_risk->death" in edge_ids and "input:age->FORM_risk" in edge_ids
        assert "category_member:CAT_Anthropometry|bmi" in edge_ids
        assert "related:ht_cm|wt_kg" in edge_ids and "related:bmi|death" in edge_ids
        assert not any(e.id.startswith("input:x") for e in spec.edges)  # unrelated formula dropped
        directed = {e.id: e.directed for e in spec.edges}
        assert directed["input:age->FORM_risk"] and not directed["related:ht_cm|wt_kg"]
        assert spec.stats().edges_by_category == {
            "category_member": 5,
            "input": 4,
            "related": 2,
            "yields": 2,
        }

    def test_hidden_variables_are_restored_from_the_full_taxonomy(
        self, taxonomy: dict, formulas: dict
    ) -> None:
        filtered = {"bmi": taxonomy["bmi"]}
        spec = build_taxonomy_graph(filtered, formulas, full_taxonomy=taxonomy)
        assert {"ht_cm", "wt_kg", "bmi", "FORM_bmi_formula"} <= set(spec.node_ids())
        assert spec.node("ht_cm").category == "input"
        assert "category_member:CAT_Anthropometry|ht_cm" in {e.id for e in spec.edges}
        assert spec.meta["restored_hidden_nodes"] >= 2
        assert "FORM_risk" in spec.node_ids()  # bmi is one of its inputs

    def test_formula_query_filters_graph(self, taxonomy: dict, formulas: dict) -> None:
        spec = build_taxonomy_graph(taxonomy, formulas, formula_query="MASS")
        assert set(spec.node_ids()) >= {"ht_cm", "wt_kg", "bmi", "FORM_bmi_formula"}
        assert "FORM_risk" not in spec.node_ids() and "death" not in spec.node_ids()

    def test_legacy_free_text_formulas_are_stable(self, taxonomy: dict) -> None:
        legacy = {
            "ht_cm": {
                **taxonomy["ht_cm"],
                "related_formulas": ["Parameter in: Body Mass Index (BMI)"],
            },
            "bmi": {
                **taxonomy["bmi"],
                "node_type": "Input",
                "related_formulas": "Result of: Body Mass Index.",
            },
        }
        first = build_taxonomy_graph(legacy)
        second = build_taxonomy_graph(legacy)
        assert first.fingerprint() == second.fingerprint()
        assert "FORM_Body_Mass_Index" in first.node_ids()
        assert first.node("bmi").category == "derived"
        assert {"input:ht_cm->FORM_Body_Mass_Index", "yields:FORM_Body_Mass_Index->bmi"} <= {
            e.id for e in first.edges
        }

    def test_empty_inputs(self) -> None:
        spec = build_taxonomy_graph({}, {})
        assert spec.nodes == () and spec.edges == () and spec.legend
        assert build_taxonomy_graph(None).nodes == ()

    def test_payload_is_json_native(self, taxonomy: dict, formulas: dict) -> None:
        import numpy as np

        taxonomy["ht_cm"]["stats"] = {"mean": np.float64(170.2)}
        spec = build_taxonomy_graph(taxonomy, formulas)
        assert spec.to_json()  # would raise if data were not JSON-native
