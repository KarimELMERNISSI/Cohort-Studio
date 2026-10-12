"""Legend rendering (Streamlit) generated from :class:`~graph.model.LegendEntry` items."""

from __future__ import annotations

import html
from collections.abc import Iterable

import streamlit as st

from graph.model import GraphSpec, LegendEntry, LineStyle, NodeShape

_CSS_LINE = {
    LineStyle.SOLID: "solid",
    LineStyle.DASHED: "dashed",
    LineStyle.DOTTED: "dotted",
    LineStyle.DASH_DOT: "dashed",
}


def _node_swatch(entry: LegendEntry) -> str:
    shape = entry.shape or NodeShape.ELLIPSE
    radius = {
        NodeShape.ELLIPSE: "50%",
        NodeShape.PILL: "50%",
        NodeShape.ROUND_RECTANGLE: "4px",
        NodeShape.RECTANGLE: "0",
        NodeShape.OCTAGON: "4px",
        NodeShape.HEXAGON: "3px",
        NodeShape.TRIANGLE: "0",
    }[shape]
    clip = ""
    if shape == NodeShape.HEXAGON:
        clip = "clip-path:polygon(25% 5%,75% 5%,100% 50%,75% 95%,25% 95%,0 50%);"
    elif shape == NodeShape.TRIANGLE:
        clip = "clip-path:polygon(50% 0,100% 100%,0 100%);"
    return (
        f'<span style="display:inline-block;width:16px;height:16px;border-radius:{radius};{clip}'
        f'background:{entry.color};border:1px solid rgba(128,128,128,0.6);vertical-align:middle"></span>'
    )


def _edge_swatch(entry: LegendEntry) -> str:
    width = max(1.0, float(entry.width or 2.0))
    line = _CSS_LINE[entry.line or LineStyle.SOLID]
    return (
        f'<span style="display:inline-block;width:48px;vertical-align:middle;'
        f'border-top:{width:g}px {line} {entry.color}"></span>'
    )


def legend_html(entries: Iterable[LegendEntry], *, title: str | None = None) -> str:
    """HTML for a grouped legend; uses Streamlit CSS variables so it follows light/dark themes."""
    groups: dict[str, list[LegendEntry]] = {}
    for entry in entries:
        groups.setdefault(entry.group or "", []).append(entry)
    parts = ['<div class="cmp-graph-legend" style="font-size:0.88em;color:var(--text-color)">']
    if title:
        parts.append(f'<div style="font-weight:600;margin-bottom:4px">{html.escape(title)}</div>')
    for group, items in groups.items():
        if group:
            parts.append(f'<div style="font-weight:600;margin:6px 0 2px 0">{html.escape(group)}</div>')
        parts.append('<table style="border-collapse:collapse"><tbody>')
        for entry in items:
            swatch = _node_swatch(entry) if entry.swatch == "node" else _edge_swatch(entry)
            parts.append(
                f'<tr><td style="padding:2px 10px 2px 0;width:56px">{swatch}</td>'
                f'<td style="padding:2px 0">{html.escape(entry.label)}</td></tr>'
            )
        parts.append("</tbody></table>")
    parts.append("</div>")
    return "".join(parts)


def render_legend(spec_or_entries: GraphSpec | Iterable[LegendEntry], *, title: str | None = None) -> None:
    """Render the legend of a spec (or of explicit entries) in the current Streamlit container."""
    entries = spec_or_entries.legend if isinstance(spec_or_entries, GraphSpec) else tuple(spec_or_entries)
    if not entries:
        st.caption("No legend available for this graph.")
        return
    st.markdown(legend_html(entries, title=title), unsafe_allow_html=True)


__all__ = ["legend_html", "render_legend"]
