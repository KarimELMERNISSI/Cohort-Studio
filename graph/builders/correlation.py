"""
Correlation network builder.

Turns a correlation matrix (square, or rectangular targets x predictors) into a
:class:`~graph.model.GraphSpec`: one node per variable, one undirected edge per pair whose
|r| reaches the requested strength category. Node ``weight`` ("heat") is the normalised sum of
|r| over the *visible* edges, computed from the numeric values (never from rounded labels).
Pure pandas/numpy code: no Streamlit, no rendering library.
"""

from __future__ import annotations

from collections import deque

import pandas as pd

from explore.correlation_strength import (
    CorrelationMethod,
    StrengthCategory,
    categorize_strength,
    coerce_category,
    coerce_method,
    min_abs_for,
)
from graph.model import EdgeSpec, GraphKind, GraphSpec, LayoutHint, LineStyle, NodeSpec, edge_id
from graph.palette import edge_style_for, legend_entries, node_style_for

_LEGACY_DASH = {
    LineStyle.SOLID: "solid",
    LineStyle.DASHED: "dashed",
    LineStyle.DOTTED: "dotted",
    LineStyle.DASH_DOT: "dashed",
}


def build_correlation_graph(
    corr_matrix: pd.DataFrame | None,
    *,
    method: CorrelationMethod | str = CorrelationMethod.PEARSON,
    min_strength: StrengthCategory | str = StrengthCategory.WEAK,
    focus: str | None = None,
    theme: str = "light",
    title: str = "Correlation network",
) -> GraphSpec:
    """Build the correlation network.

    Args:
        corr_matrix: correlation coefficients; rows and columns are variable names.
        method: correlation method, selects the strength thresholds.
        min_strength: weakest category to keep (edges below it are dropped).
        focus: when given and present, only the connected component of that variable is kept
            and the node is highlighted.
        theme: ``"light"`` or ``"dark"`` palette.
    """
    method_enum = coerce_method(method)
    category_enum = coerce_category(min_strength)
    threshold = min_abs_for(category_enum, method_enum)
    kind = GraphKind.CORRELATION
    meta = {
        "method": method_enum.value,
        "min_strength": category_enum.value,
        "threshold": float(threshold),
        "focus": focus,
        "n_pairs_total": 0,
        "n_pairs_kept": 0,
    }

    if corr_matrix is None or corr_matrix.empty:
        return GraphSpec(
            kind=kind,
            title=title,
            directed=False,
            layout=LayoutHint.ORGANIC,
            legend=legend_entries(kind, theme),
            meta=meta,
        )

    rows = [str(r) for r in corr_matrix.index]
    cols = [str(c) for c in corr_matrix.columns]
    values = corr_matrix.to_numpy()

    edges: list[EdgeSpec] = []
    seen_pairs: set[tuple[str, str]] = set()
    n_pairs_total = 0
    for i, row_name in enumerate(rows):
        for j, col_name in enumerate(cols):
            if row_name == col_name:
                continue
            pair = (row_name, col_name) if row_name <= col_name else (col_name, row_name)
            if pair in seen_pairs:
                continue
            raw = values[i, j]
            if pd.isna(raw):
                continue
            seen_pairs.add(pair)
            n_pairs_total += 1
            value = float(raw)
            magnitude = abs(value)
            if magnitude < threshold:
                continue
            strength = categorize_strength(magnitude, method_enum)
            category = f"corr.{'pos' if value >= 0 else 'neg'}.{strength.name.lower()}"
            edges.append(
                EdgeSpec(
                    id=edge_id("corr", pair[0], pair[1], undirected=True),
                    source=pair[0],
                    target=pair[1],
                    label=f"{value:.2f}",
                    category=category,
                    directed=False,
                    style=edge_style_for(kind, category, theme),
                    weight=min(1.0, magnitude),
                    tooltip=f"{pair[0]} - {pair[1]}: r = {value:.3f} ({strength.value})",
                    data={
                        "r": value,
                        "abs_r": magnitude,
                        "strength": strength.value,
                        "method": method_enum.value,
                    },
                )
            )

    all_ids = sorted(set(rows) | set(cols))
    visible = set(all_ids)
    if focus is not None and focus in visible:
        adjacency: dict[str, set[str]] = {node_id: set() for node_id in all_ids}
        for e in edges:
            adjacency[e.source].add(e.target)
            adjacency[e.target].add(e.source)
        visible = {focus}
        queue: deque[str] = deque([focus])
        while queue:
            current = queue.popleft()
            for neighbour in adjacency[current]:
                if neighbour not in visible:
                    visible.add(neighbour)
                    queue.append(neighbour)

    visible_edges = [e for e in edges if e.source in visible and e.target in visible]
    heat: dict[str, float] = {node_id: 0.0 for node_id in all_ids if node_id in visible}
    for e in visible_edges:
        heat[e.source] += float(e.data["abs_r"])
        heat[e.target] += float(e.data["abs_r"])
    max_heat = max(heat.values(), default=0.0)

    nodes: list[NodeSpec] = []
    for node_id in all_ids:
        if node_id not in visible:
            continue
        node_category = "focus" if node_id == focus else "variable"
        weight = heat[node_id] / max_heat if max_heat > 0 else 0.0
        nodes.append(
            NodeSpec(
                id=node_id,
                label=node_id,
                category=node_category,
                style=node_style_for(kind, node_category, theme),
                weight=round(weight, 6),
                tooltip=f"{node_id}: cumulative |r| = {heat[node_id]:.2f}",
                data={
                    "degree": sum(1 for e in visible_edges if node_id in (e.source, e.target)),
                    "heat": heat[node_id],
                },
            )
        )

    meta.update({"n_pairs_total": n_pairs_total, "n_pairs_kept": len(visible_edges)})
    return GraphSpec(
        kind=kind,
        title=title,
        nodes=tuple(nodes),
        edges=tuple(visible_edges),
        directed=False,
        layout=LayoutHint.ORGANIC,
        legend=legend_entries(kind, theme),
        meta=meta,
    )


def to_legacy_dicts(spec: GraphSpec) -> tuple[list[dict], list[dict]]:
    """Plain-dict nodes/edges in the shape the pre-connector code (and yFiles) expected.

    Kept for the deprecated ``explore.corr_matrix.get_yfiles_network_data`` shim and its tests.
    """
    nodes = [
        {
            "id": n.id,
            "properties": {
                "label": n.label,
                "color": n.style.color,
                "shape": n.style.shape.value,
                "heat": n.weight if n.weight is not None else 0.0,
            },
        }
        for n in spec.nodes
    ]
    edges = [
        {
            "start": e.source,
            "end": e.target,
            "id": e.id,
            "properties": {
                "label": e.label,
                "color": e.style.color,
                "dash": _LEGACY_DASH[e.style.line],
                "thickness": e.style.width,
                "directed": e.directed,
            },
        }
        for e in spec.edges
    ]
    return nodes, edges


__all__ = ["build_correlation_graph", "to_legacy_dicts"]
