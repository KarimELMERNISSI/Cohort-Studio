"""
Formula diagram builder: the variables a formula uses, and the variable it computes.

Drawn from the formula as the application reads it (:func:`enrich.formulas.read_formula` gives the names), never
from a description of it: each variable is marked present in the data or missing, so the diagram shows what
computing the formula will use. Pure: no Streamlit, no rendering library.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable

from graph.model import EdgeSpec, GraphKind, GraphSpec, LayoutHint, NodeSpec, edge_id
from graph.palette import edge_style_for, legend_entries, node_style_for


def build_formula_graph(
    target: str,
    inputs: Iterable[str],
    available: Collection[str],
    *,
    theme: str = "light",
    title: str = "",
) -> GraphSpec:
    """
    One node per variable the formula uses (in the order given, once each), an arrow from each to the computed
    variable ``target``; a variable that is not in ``available`` (the data's columns) is marked missing.
    """
    kind = GraphKind.FORMULA
    target_id = f"target:{target}"
    nodes: list[NodeSpec] = []
    edges: list[EdgeSpec] = []
    missing: list[str] = []
    for name in dict.fromkeys(inputs):
        category = "input" if name in available else "missing"
        if category == "missing":
            missing.append(name)
        node_id = f"input:{name}"
        nodes.append(
            NodeSpec(
                id=node_id,
                label=name if category == "input" else f"{name} (missing)",
                category=category,
                style=node_style_for(kind, category, theme),
                tooltip=f"{name}: in the data"
                if category == "input"
                else f"{name}: not in the data",
            )
        )
        edges.append(
            EdgeSpec(
                id=edge_id(category, node_id, target_id),
                source=node_id,
                target=target_id,
                category=category,
                directed=True,
                style=edge_style_for(kind, category, theme),
            )
        )
    nodes.append(
        NodeSpec(
            id=target_id,
            label=target,
            category="target",
            style=node_style_for(kind, "target", theme),
            tooltip=f"{target}: the new variable",
        )
    )
    return GraphSpec(
        kind=kind,
        title=title,
        directed=True,
        layout=LayoutHint.HIERARCHICAL,
        nodes=tuple(nodes),
        edges=tuple(edges),
        legend=legend_entries(kind, theme),
        meta={"target": target, "missing": missing},
    )


__all__ = ["build_formula_graph"]
