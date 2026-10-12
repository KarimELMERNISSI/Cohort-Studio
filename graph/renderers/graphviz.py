"""
Graphviz renderer: static DOT drawing through ``st.graphviz_chart`` (no Python dependency, no
network access). Useful for small dependency diagrams (formula inputs -> target) and as a last
resort when neither Plotly nor yFiles is available. No selection support.
"""

from __future__ import annotations

import html
from typing import Any

from graph.base import BaseGraphRenderer, RendererCapabilities, RenderError
from graph.model import GraphKind, GraphSpec, LayoutHint, LineStyle, NodeShape, RenderOptions, SelectionResult

DOT_SHAPES: dict[NodeShape, str] = {
    NodeShape.ELLIPSE: "ellipse",
    NodeShape.RECTANGLE: "box",
    NodeShape.ROUND_RECTANGLE: "box",
    NodeShape.HEXAGON: "hexagon",
    NodeShape.OCTAGON: "octagon",
    NodeShape.PILL: "ellipse",
    NodeShape.TRIANGLE: "triangle",
}
DOT_LINES: dict[LineStyle, str] = {
    LineStyle.SOLID: "solid",
    LineStyle.DASHED: "dashed",
    LineStyle.DOTTED: "dotted",
    LineStyle.DASH_DOT: "dashed",
}
ENGINES: dict[LayoutHint, str] = {
    LayoutHint.AUTO: "dot",
    LayoutHint.HIERARCHICAL: "dot",
    LayoutHint.TREE: "dot",
    LayoutHint.ORTHOGONAL: "dot",
    LayoutHint.ORGANIC: "neato",
    LayoutHint.CIRCULAR: "circo",
    LayoutHint.RADIAL: "twopi",
}


def _quote(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def to_dot(spec: GraphSpec, options: RenderOptions) -> str:
    """DOT source for the spec (deterministic; node/edge order preserved)."""
    dark = options.theme == "dark"
    font_color = "#E6EDF3" if dark else "#1F2328"
    lines = ["digraph G {" if spec.directed else "graph G {"]
    lines.append("  graph [bgcolor=transparent, rankdir=TB, splines=true, overlap=false, pad=0.2];")
    lines.append(f"  node [style=filled, fontname=Helvetica, fontsize=11, fontcolor={_quote(font_color)}, color={_quote('#777777')}];")
    lines.append(f"  edge [fontname=Helvetica, fontsize=9, fontcolor={_quote(font_color)}];")
    if spec.title:
        lines.append(f"  label={_quote(spec.title)}; labelloc=t; fontcolor={_quote(font_color)};")
    for n in spec.nodes:
        attrs = {
            "label": n.label or n.id,
            "shape": DOT_SHAPES.get(n.style.shape, "ellipse"),
            "fillcolor": n.style.color or "#CCCCCC",
            "fontcolor": n.style.text_color or font_color,
            "tooltip": n.tooltip or n.label or n.id,
        }
        if n.style.shape == NodeShape.ROUND_RECTANGLE:
            attrs["style"] = "filled,rounded"
        attr_text = ", ".join(f"{k}={_quote(v)}" for k, v in attrs.items())
        lines.append(f"  {_quote(n.id)} [{attr_text}];")
    connector = "->" if spec.directed else "--"
    for e in spec.edges:
        attrs = {
            "color": e.style.color or "#999999",
            "style": DOT_LINES.get(e.style.line, "solid"),
            "penwidth": f"{max(0.5, float(e.style.width)):g}",
            "tooltip": e.tooltip or e.label,
        }
        if options.show_edge_labels and e.label:
            attrs["label"] = e.label
        if spec.directed:
            attrs["dir"] = "forward" if e.directed else "none"
        attr_text = ", ".join(f"{k}={_quote(v)}" for k, v in attrs.items())
        lines.append(f"  {_quote(e.source)} {connector} {_quote(e.target)} [{attr_text}];")
    lines.append("}")
    return "\n".join(lines)


class GraphvizGraphRenderer(BaseGraphRenderer):
    name = "graphviz"
    display_name = "Graphviz (static)"

    @property
    def capabilities(self) -> RendererCapabilities:
        return RendererCapabilities(
            supports_selection=False,
            supports_edge_selection=False,
            supports_heat=False,
            supports_node_shapes=True,
            supports_edge_labels=True,
            supports_mixed_direction=True,
            supports_hierarchical=True,
            supports_drag=False,
            supports_html_export=False,
            requires_internet=False,
            telemetry=False,
            license_note="Graphviz DOT rendered by Streamlit's bundled viz.js (MIT); no telemetry.",
        )

    def translate(self, spec: GraphSpec, options: RenderOptions) -> str:
        return to_dot(spec, options)

    def render(self, spec: GraphSpec, options: RenderOptions) -> SelectionResult:
        import streamlit as st

        try:
            dot = self.translate(spec, options)
        except Exception as exc:
            raise RenderError(f"Graphviz renderer could not translate the graph: {exc}") from exc
        layout = options.layout or spec.layout
        engine = ENGINES.get(layout, "dot")
        if spec.kind == GraphKind.CORRELATION and layout == LayoutHint.AUTO:
            engine = "neato"
        try:
            st.graphviz_chart(dot, engine=engine, width="stretch")
        except TypeError:  # older Streamlit without the engine/width arguments
            st.graphviz_chart(dot)
        return SelectionResult(renderer=self.name)

    def export_html(self, spec: GraphSpec, options: RenderOptions) -> bytes | None:
        dot = html.escape(self.translate(spec, options))
        return f"<pre>{dot}</pre>".encode("utf-8")


__all__: list[Any] = ["GraphvizGraphRenderer", "to_dot"]
