"""
Plotly + NetworkX renderer: the offline default.

Positions come from NetworkX (spring / circular / shell layouts, and a small layered layout for
hierarchical graphs); drawing uses plotly.graph_objects; selection uses Streamlit's native
``st.plotly_chart(on_select="rerun")``. No network access, MIT/BSD licences, no telemetry.
Degradations compared with a dedicated graph library: nodes cannot be dragged and edges are
straight lines.
"""

from __future__ import annotations

import logging
import math
from typing import Any, NamedTuple

from graph.base import BaseGraphRenderer, RendererCapabilities, RenderError
from graph.model import (
    GraphKind,
    GraphSpec,
    LayoutHint,
    LineStyle,
    NodeShape,
    RenderOptions,
    SelectionResult,
)

logger = logging.getLogger(__name__)

SYMBOL_MAP: dict[NodeShape, str] = {
    NodeShape.ELLIPSE: "circle",
    NodeShape.RECTANGLE: "square",
    NodeShape.ROUND_RECTANGLE: "square",
    NodeShape.HEXAGON: "hexagon",
    NodeShape.OCTAGON: "octagon",
    NodeShape.PILL: "circle",
    NodeShape.TRIANGLE: "triangle-up",
}
DASH_MAP: dict[LineStyle, str] = {
    LineStyle.SOLID: "solid",
    LineStyle.DASHED: "dash",
    LineStyle.DOTTED: "dot",
    LineStyle.DASH_DOT: "dashdot",
}
DEFAULT_LAYOUT: dict[GraphKind, LayoutHint] = {
    GraphKind.CORRELATION: LayoutHint.ORGANIC,
    GraphKind.TAXONOMY: LayoutHint.HIERARCHICAL,
    GraphKind.DOCUMENT: LayoutHint.HIERARCHICAL,
    GraphKind.FORMULA: LayoutHint.HIERARCHICAL,
}
NODE_BASE_SIZE = 22.0
LAYER_SPACING = 1.6
COLUMN_SPACING = 1.0


class PlotlyPayload(NamedTuple):
    figure: Any
    node_curve: int | None
    edge_curve: int | None
    positions: dict[str, tuple[float, float]]
    layout: LayoutHint


Positions = dict[str, tuple[float, float]]


# --------------------------------------------------------------------------- layouts
def _resolve_layout(spec: GraphSpec, options: RenderOptions) -> LayoutHint:
    hint = options.layout or spec.layout
    if hint == LayoutHint.AUTO:
        hint = DEFAULT_LAYOUT.get(spec.kind, LayoutHint.ORGANIC)
    return hint


def _layered_positions(spec: GraphSpec) -> Positions:
    """Simple Sugiyama-style layout: topological layers (top-down) + barycentre ordering."""
    import networkx as nx

    digraph = nx.DiGraph()
    digraph.add_nodes_from(n.id for n in spec.nodes)
    for e in spec.edges:
        if e.source != e.target:
            digraph.add_edge(e.source, e.target)
    # Break cycles so that topological layering is defined.
    while True:
        try:
            cycle = nx.find_cycle(digraph)
        except nx.NetworkXNoCycle:
            break
        u, v = cycle[-1][0], cycle[-1][1]
        digraph.remove_edge(u, v)

    layers: dict[str, int] = {}
    for depth, generation in enumerate(nx.topological_generations(digraph)):
        for node_id in generation:
            layers[node_id] = depth
    by_layer: dict[int, list[str]] = {}
    for node_id in digraph.nodes:
        by_layer.setdefault(layers.get(node_id, 0), []).append(node_id)

    order: dict[str, float] = {}
    depths = sorted(by_layer)
    for depth in depths:  # initial order: stable (builder order)
        for idx, node_id in enumerate(by_layer[depth]):
            order[node_id] = float(idx)
    for _ in range(2):  # two barycentre sweeps (down then up) reduce crossings noticeably
        for depth in depths[1:]:
            _barycentre_sort(by_layer[depth], order, lambda n: list(digraph.predecessors(n)))
        for depth in reversed(depths[:-1]):
            _barycentre_sort(by_layer[depth], order, lambda n: list(digraph.successors(n)))

    positions: Positions = {}
    for depth in depths:
        nodes_in_layer = by_layer[depth]
        width = (len(nodes_in_layer) - 1) * COLUMN_SPACING
        for idx, node_id in enumerate(nodes_in_layer):
            positions[node_id] = (idx * COLUMN_SPACING - width / 2.0, -depth * LAYER_SPACING)
    return positions


def _barycentre_sort(layer_nodes: list[str], order: dict[str, float], neighbours) -> None:
    def key(node_id: str) -> float:
        nbs = [order[n] for n in neighbours(node_id) if n in order]
        return sum(nbs) / len(nbs) if nbs else order[node_id]

    layer_nodes.sort(key=key)
    for idx, node_id in enumerate(layer_nodes):
        order[node_id] = float(idx)


def compute_positions(spec: GraphSpec, layout: LayoutHint, focus: str | None = None) -> Positions:
    """Deterministic 2-D positions for every node of the spec."""
    import networkx as nx

    if not spec.nodes:
        return {}
    if len(spec.nodes) == 1:
        return {spec.nodes[0].id: (0.0, 0.0)}

    graph = spec.to_networkx(directed=False)
    n = graph.number_of_nodes()
    if layout in (LayoutHint.HIERARCHICAL, LayoutHint.TREE, LayoutHint.ORTHOGONAL):
        positions = _layered_positions(spec)
    elif layout == LayoutHint.CIRCULAR:
        positions = nx.circular_layout(graph)
    elif layout == LayoutHint.RADIAL:
        centre = focus if focus in graph else max(graph.degree, key=lambda item: item[1])[0]
        shells: list[list[str]] = [
            list(layer_nodes) for layer_nodes in nx.bfs_layers(graph, centre)
        ]
        placed = {node for shell in shells for node in shell}
        remaining = [node for node in graph.nodes if node not in placed]
        if remaining:
            shells.append(remaining)
        positions = nx.shell_layout(graph, nlist=shells)
    else:  # ORGANIC / AUTO
        positions = nx.spring_layout(graph, seed=42, k=1.5 / math.sqrt(n), iterations=100)
    return {str(node_id): (float(xy[0]), float(xy[1])) for node_id, xy in positions.items()}


# --------------------------------------------------------------------------- renderer
class PlotlyGraphRenderer(BaseGraphRenderer):
    name = "plotly"
    display_name = "Plotly (offline)"

    @property
    def capabilities(self) -> RendererCapabilities:
        return RendererCapabilities(
            supports_selection=True,
            supports_edge_selection=True,
            supports_heat=True,
            supports_node_shapes=True,
            supports_edge_labels=True,
            supports_mixed_direction=True,
            supports_hierarchical=True,
            supports_drag=False,
            supports_html_export=True,
            requires_internet=False,
            telemetry=False,
            license_note="plotly (MIT) and networkx (BSD-3); rendered locally, no telemetry.",
        )

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        try:
            import networkx  # noqa: F401
            import plotly.graph_objects  # noqa: F401
        except ImportError as exc:
            return False, f"plotly/networkx not installed: {exc}"
        return True, ""

    # ------------------------------------------------------------------ translate
    def translate(self, spec: GraphSpec, options: RenderOptions) -> PlotlyPayload:
        import plotly.graph_objects as go

        theme = "dark" if options.theme == "dark" else "light"
        text_color = "#E6EDF3" if theme == "dark" else "#1F2328"
        layout = _resolve_layout(spec, options)
        positions = compute_positions(spec, layout, options.focus_node)
        fig = go.Figure()
        node_curve: int | None = None
        edge_curve: int | None = None

        if positions:
            xs = [p[0] for p in positions.values()]
            ys = [p[1] for p in positions.values()]
            span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
        else:
            span = 1.0
        shorten = 0.035 * span

        # Edge lines grouped by style (one trace per distinct colour/dash/width).
        groups: dict[tuple[str, str, float], list[tuple[float, float, float, float]]] = {}
        for e in spec.edges:
            x0, y0 = positions[e.source]
            x1, y1 = positions[e.target]
            style_key = (e.style.color or "#999999", DASH_MAP[e.style.line], float(e.style.width))
            groups.setdefault(style_key, []).append((x0, y0, x1, y1))
        for (colour, dash, width), segments in groups.items():
            gx: list[float | None] = []
            gy: list[float | None] = []
            for x0, y0, x1, y1 in segments:
                gx.extend([x0, x1, None])
                gy.extend([y0, y1, None])
            fig.add_trace(
                go.Scatter(
                    x=gx,
                    y=gy,
                    mode="lines",
                    line=dict(color=colour, width=width, dash=dash),
                    hoverinfo="none",
                    showlegend=False,
                )
            )

        # Arrow heads for directed edges (drawn as annotations ending at the node boundary).
        for e in spec.edges:
            if not e.directed:
                continue
            x0, y0 = positions[e.source]
            x1, y1 = positions[e.target]
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            if length <= 1e-9:
                continue
            ex, ey = x1 - dx / length * shorten, y1 - dy / length * shorten
            fig.add_annotation(
                x=ex,
                y=ey,
                ax=x0,
                ay=y0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=max(1.0, float(e.style.width) * 0.6),
                arrowcolor=e.style.color or "#999999",
                text="",
                opacity=0.95,
            )

        # Edge midpoints: hover, selection and labels.
        if spec.edges:
            mid_x, mid_y, labels, hovers, ids = [], [], [], [], []
            for e in spec.edges:
                x0, y0 = positions[e.source]
                x1, y1 = positions[e.target]
                mid_x.append((x0 + x1) / 2.0)
                mid_y.append((y0 + y1) / 2.0)
                labels.append(e.label if options.show_edge_labels else "")
                hovers.append(
                    e.tooltip or f"{e.source} -> {e.target}"
                    if e.directed
                    else e.tooltip or f"{e.source} - {e.target}"
                )
                ids.append(e.id)
            fig.add_trace(
                go.Scatter(
                    x=mid_x,
                    y=mid_y,
                    mode="markers+text",
                    text=labels,
                    textposition="middle center",
                    textfont=dict(size=10, color=text_color),
                    marker=dict(size=9, opacity=0.0),
                    hovertext=hovers,
                    hoverinfo="text",
                    customdata=ids,
                    showlegend=False,
                    name="edges",
                )
            )
            edge_curve = len(fig.data) - 1

        # Nodes.
        if spec.nodes:
            nx_, ny_, labels, hovers, ids, colours, symbols, sizes, borders, weights = (
                [] for _ in range(10)
            )
            for n in spec.nodes:
                x, y = positions[n.id]
                nx_.append(x)
                ny_.append(y)
                labels.append(n.label or n.id)
                hovers.append(n.tooltip or n.label or n.id)
                ids.append(n.id)
                colours.append(n.style.color or "#87CEEB")
                symbols.append(SYMBOL_MAP.get(n.style.shape, "circle"))
                sizes.append(NODE_BASE_SIZE * float(n.size or 1.0))
                borders.append(
                    n.style.border_color or ("#FFFFFF" if theme == "dark" else "#333333")
                )
                weights.append(float(n.weight) if n.weight is not None else 0.0)
            marker: dict[str, Any] = dict(
                size=sizes, symbol=symbols, line=dict(width=1.5, color=borders)
            )
            if options.heat and any(w > 0 for w in weights):
                marker.update(
                    color=weights,
                    colorscale="YlOrRd",
                    cmin=0.0,
                    cmax=1.0,
                    showscale=True,
                    colorbar=dict(title="Heat", thickness=12, len=0.5),
                )
            else:
                marker.update(color=colours)
            fig.add_trace(
                go.Scatter(
                    x=nx_,
                    y=ny_,
                    mode="markers+text",
                    text=labels,
                    textposition="top center",
                    textfont=dict(size=11, color=text_color),
                    marker=marker,
                    hovertext=hovers,
                    hoverinfo="text",
                    customdata=ids,
                    showlegend=False,
                    name="nodes",
                )
            )
            node_curve = len(fig.data) - 1

        fig.update_layout(
            height=options.height,
            margin=dict(l=10, r=10, t=36 if spec.title else 10, b=10),
            title=dict(text=spec.title, x=0.0, font=dict(size=14)) if spec.title else None,
            showlegend=False,
            hovermode="closest",
            dragmode="pan",
            template="plotly_dark" if theme == "dark" else "plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False, showgrid=False, zeroline=False),
            yaxis=dict(
                visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1
            ),
            uirevision=spec.fingerprint(),
        )
        return PlotlyPayload(fig, node_curve, edge_curve, positions, layout)

    # ------------------------------------------------------------------ render
    def render(self, spec: GraphSpec, options: RenderOptions) -> SelectionResult:
        import streamlit as st

        try:
            payload = self.translate(spec, options)
        except Exception as exc:
            raise RenderError(f"Plotly renderer could not translate the graph: {exc}") from exc

        chart_kwargs: dict[str, Any] = dict(
            key=f"{options.key}__plotly",
            width="stretch",
            theme=None,
            config={"displaylogo": False, "scrollZoom": True, "displayModeBar": True},
        )
        if options.selectable:
            chart_kwargs.update(on_select="rerun", selection_mode=("points", "box", "lasso"))
        event = st.plotly_chart(payload.figure, **chart_kwargs)
        if not options.selectable:
            return SelectionResult(renderer=self.name)
        return self._selection_from_event(event, payload)

    def _selection_from_event(self, event: Any, payload: PlotlyPayload) -> SelectionResult:
        node_ids: list[str] = []
        edge_ids: list[str] = []
        try:
            points = event.selection.points  # type: ignore[attr-defined]
        except Exception:
            points = []
        for point in points or []:
            curve = point.get("curve_number")
            custom = point.get("customdata")
            if isinstance(custom, (list, tuple)):
                custom = custom[0] if custom else None
            if custom is None:
                continue
            if curve == payload.node_curve:
                node_ids.append(str(custom))
            elif curve == payload.edge_curve:
                edge_ids.append(str(custom))
        return SelectionResult(
            node_ids=tuple(dict.fromkeys(node_ids)),
            edge_ids=tuple(dict.fromkeys(edge_ids)),
            renderer=self.name,
        )

    def export_html(self, spec: GraphSpec, options: RenderOptions) -> bytes | None:
        payload = self.translate(spec, options)
        return payload.figure.to_html(include_plotlyjs=True, full_html=True).encode("utf-8")


__all__ = ["PlotlyGraphRenderer", "PlotlyPayload", "compute_positions"]
