"""Reference behaviour of the correlation network builder."""

import numpy as np
import pandas as pd
import pytest

from explore.correlation_strength import StrengthCategory, categorize_strength, min_abs_for, strength_table
from graph.builders.correlation import build_correlation_graph, to_legacy_dicts
from graph.model import GraphKind, LineStyle


class TestStrengthTable:
    def test_pearson_categories(self) -> None:
        assert categorize_strength(0.05, "pearson") == StrengthCategory.NEGLIGIBLE
        assert categorize_strength(0.10, "pearson") == StrengthCategory.WEAK
        assert categorize_strength(0.39, "pearson") == StrengthCategory.WEAK
        assert categorize_strength(0.40, "pearson") == StrengthCategory.MODERATE
        assert categorize_strength(0.90, "PEARSON") == StrengthCategory.VERY_STRONG

    def test_method_specific_thresholds(self) -> None:
        assert categorize_strength(0.39, "spearman") == StrengthCategory.MODERATE
        assert categorize_strength(0.39, "kendall") == StrengthCategory.MODERATE
        assert min_abs_for("Very Strong", "kendall") == 0.71
        assert min_abs_for("very_strong", "pearson") == 0.90
        assert min_abs_for("unknown", "unknown") == 0.10  # falls back to Weak / Pearson

    def test_strength_table_shape(self) -> None:
        table = strength_table()
        assert set(table["Method"]) == {"pearson", "spearman", "kendall"}
        assert len(table) == 15


class TestBuilder:
    def test_reference_graph(self, corr_matrix: pd.DataFrame) -> None:
        spec = build_correlation_graph(corr_matrix, method="pearson", min_strength="Weak")
        assert spec.kind == GraphKind.CORRELATION and not spec.directed
        assert spec.node_ids() == ("a", "b", "c", "d")
        assert [e.id for e in spec.edges] == ["corr:a|b", "corr:a|c", "corr:b|d"]
        assert [e.category for e in spec.edges] == ["corr.pos.very_strong", "corr.neg.moderate", "corr.pos.weak"]
        assert [e.label for e in spec.edges] == ["0.95", "-0.50", "0.20"]
        assert [e.style.width for e in spec.edges] == [5.0, 2.5, 1.5]
        assert [e.style.line for e in spec.edges] == [LineStyle.SOLID, LineStyle.SOLID, LineStyle.DASHED]
        assert all(not e.directed for e in spec.edges)
        assert spec.meta["n_pairs_total"] == 6 and spec.meta["n_pairs_kept"] == 3
        assert spec.legend  # legend travels with the spec

    def test_heat_uses_numeric_values(self, corr_matrix: pd.DataFrame) -> None:
        spec = build_correlation_graph(corr_matrix, min_strength="Weak")
        weights = {n.id: n.weight for n in spec.nodes}
        assert weights["a"] == pytest.approx(1.0)
        assert weights["b"] == pytest.approx(1.15 / 1.45, rel=1e-5)
        assert weights["c"] == pytest.approx(0.5 / 1.45, rel=1e-5)
        assert weights["d"] == pytest.approx(0.2 / 1.45, rel=1e-5)

    def test_min_strength_filters_edges_but_keeps_nodes(self, corr_matrix: pd.DataFrame) -> None:
        spec = build_correlation_graph(corr_matrix, min_strength="Strong")
        assert [e.id for e in spec.edges] == ["corr:a|b"]
        assert spec.node_ids() == ("a", "b", "c", "d")
        weights = {n.id: n.weight for n in spec.nodes}
        assert weights["c"] == 0.0 and weights["d"] == 0.0

    def test_focus_keeps_connected_component_and_highlights(self, corr_matrix: pd.DataFrame) -> None:
        spec = build_correlation_graph(corr_matrix, min_strength="Weak", focus="d")
        assert spec.node_ids() == ("a", "b", "c", "d")
        assert spec.node("d").category == "focus" and spec.node("a").category == "variable"
        strong_focus = build_correlation_graph(corr_matrix, min_strength="Strong", focus="c")
        assert strong_focus.node_ids() == ("c",)  # c has no strong edges: only itself remains
        assert strong_focus.edges == ()

    def test_focus_unknown_variable_is_ignored(self, corr_matrix: pd.DataFrame) -> None:
        spec = build_correlation_graph(corr_matrix, focus="nope")
        assert spec.node_ids() == ("a", "b", "c", "d")

    def test_nan_pairs_are_skipped(self, corr_matrix: pd.DataFrame) -> None:
        matrix = corr_matrix.copy()
        matrix.loc["a", "b"] = np.nan
        matrix.loc["b", "a"] = np.nan
        spec = build_correlation_graph(matrix, min_strength="Weak")
        assert [e.id for e in spec.edges] == ["corr:a|c", "corr:b|d"]
        assert spec.meta["n_pairs_total"] == 5

    def test_rectangular_matrix(self) -> None:
        matrix = pd.DataFrame([[0.8, -0.3]], index=["outcome"], columns=["age", "bmi"])
        spec = build_correlation_graph(matrix, min_strength="Weak")
        assert spec.node_ids() == ("age", "bmi", "outcome")
        assert sorted(e.id for e in spec.edges) == ["corr:age|outcome", "corr:bmi|outcome"]

    def test_empty_matrix(self) -> None:
        spec = build_correlation_graph(pd.DataFrame())
        assert spec.nodes == () and spec.edges == () and spec.legend
        assert build_correlation_graph(None).nodes == ()

    def test_spec_is_deterministic(self, corr_matrix: pd.DataFrame) -> None:
        assert build_correlation_graph(corr_matrix).fingerprint() == build_correlation_graph(corr_matrix).fingerprint()

    def test_legacy_dict_shape(self, corr_matrix: pd.DataFrame) -> None:
        nodes, edges = to_legacy_dicts(build_correlation_graph(corr_matrix, min_strength="Weak"))
        assert nodes[0]["id"] == "a" and set(nodes[0]["properties"]) == {"label", "color", "shape", "heat"}
        assert edges[0]["start"] == "a" and edges[0]["end"] == "b"
        assert edges[0]["properties"]["dash"] == "solid" and edges[2]["properties"]["dash"] == "dashed"
        assert edges[0]["properties"]["directed"] is False
