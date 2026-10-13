"""Shared fixtures for the graph layer tests."""

import pandas as pd
import pytest

from graph.model import EdgeSpec, GraphKind, GraphSpec, LayoutHint, NodeSpec, edge_id
from graph.palette import edge_style_for, legend_entries, node_style_for


@pytest.fixture
def corr_matrix() -> pd.DataFrame:
    """4x4 symmetric correlation matrix with one pair per strength band."""
    labels = ["a", "b", "c", "d"]
    values = [
        [1.00, 0.95, -0.50, 0.00],
        [0.95, 1.00, 0.00, 0.20],
        [-0.50, 0.00, 1.00, 0.05],
        [0.00, 0.20, 0.05, 1.00],
    ]
    return pd.DataFrame(values, index=labels, columns=labels)


@pytest.fixture
def taxonomy_spec() -> GraphSpec:
    """Small directed graph: category -> variables -> formula -> output."""
    kind = GraphKind.TAXONOMY

    def node(node_id: str, category: str, label: str | None = None) -> NodeSpec:
        return NodeSpec(
            id=node_id,
            label=label or node_id,
            category=category,
            style=node_style_for(kind, category),
        )

    def edge(
        category: str, source: str, target: str, directed: bool = True, label: str = ""
    ) -> EdgeSpec:
        return EdgeSpec(
            id=edge_id(category, source, target, undirected=not directed),
            source=source,
            target=target,
            category=category,
            directed=directed,
            label=label,
            style=edge_style_for(kind, category),
        )

    nodes = (
        node("CAT_vitals", "category", "Vitals"),
        node("height", "input"),
        node("weight", "input"),
        node("FORM_bmi", "formula", "BMI formula"),
        node("bmi", "derived"),
    )
    edges = (
        edge("category_member", "CAT_vitals", "height", directed=False),
        edge("category_member", "CAT_vitals", "weight", directed=False),
        edge("input", "height", "FORM_bmi", label="input"),
        edge("input", "weight", "FORM_bmi", label="input"),
        edge("yields", "FORM_bmi", "bmi", label="yields"),
    )
    return GraphSpec(
        kind=kind,
        title="Taxonomy",
        nodes=nodes,
        edges=edges,
        directed=True,
        layout=LayoutHint.HIERARCHICAL,
        legend=legend_entries(kind),
    )
