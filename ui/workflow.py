"""
The research workflow shown on the Start page: a core path (load, validate, explore and analyse,
report) and optional branches (enrich, map variables to the field, learn from documents).

Pure Python: :func:`workflow_model` returns the geometry and an SVG drawing; the interactive map
component (``ui.components.workflow_map``) renders it with theme colours and handles clicks.
"""

from __future__ import annotations

import hashlib
import html
from dataclasses import dataclass

CANVAS = (1200, 346)
BOX_W, BOX_H = 176, 60


@dataclass(frozen=True)
class Branch:
    key: str  # navigation key of the page that runs it
    icon: str
    title: str
    when: str  # where the branch attaches to the core path
    text: str
    links: tuple[tuple[str, str], ...]
    needs_ai: bool = False


BRANCHES: tuple[Branch, ...] = (
    Branch(
        "enrich",
        ":material/auto_fix_high:",
        "Enrich the cohort",
        "after validation, as often as needed",
        "Merge external sources, derive variables from formulas or AI suggestions, impute missing values "
        "(single imputation for exploration, multiple imputation for inference). "
        "Each change can be validated and analysed again before the next one.",
        (("enrich", "Open Enrich"),),
    ),
    Branch(
        "variable_map",
        ":material/account_tree:",
        "Map your variables to your field",
        "any time · uses the AI engine",
        "Build a taxonomy of your variables grounded in your clinical domain and your documents: definitions, "
        "units, relationships and derived formulas, saved as versions.",
        (("variable_map", "Open Variable map"),),
        needs_ai=True,
    ),
    Branch(
        "documents",
        ":material/description:",
        "Learn from your documents",
        "any time · uses the AI engine",
        "Index your literature, protocols or data dictionaries, chat with them and extract a knowledge graph that "
        "also informs the variable map and the enrichment suggestions.",
        (("documents", "Open Documents"), ("settings", "Document library")),
        needs_ai=True,
    ),
    Branch(
        "reproduce",
        ":material/history:",
        "Report and reproduce",
        "at the end, or any time",
        "Every step is traced: reopen saved versions, replay an analysis step by step and export the Word report.",
        (("reproduce", "Open Versions & traces"),),
    ),
)


@dataclass(frozen=True)
class Node:
    id: str
    page: str  # navigation key opened on click
    title: str
    sub: str
    kind: str  # "core" | "optional"
    x: int
    y: int
    description: str
    needs_ai: bool = False


NODES: tuple[Node, ...] = (
    Node(
        "load",
        "import",
        "1  Load data",
        "import · versions · sample",
        "core",
        12,
        34,
        "Import a CSV, Excel or Parquet file, reopen a saved version, or try the sample cohort.",
    ),
    Node(
        "validate",
        "validate",
        "2  Validate",
        "quality · anomalies · outliers",
        "core",
        236,
        34,
        "Quality score, missingness pattern, clinical anomalies, inclusion criteria and outliers.",
    ),
    Node(
        "analyse",
        "explore",
        "3  Explore & analyse",
        "statistics · tests · survival",
        "core",
        796,
        34,
        "Descriptive statistics and correlations, then hypothesis tests, epidemiological measures and survival.",
    ),
    Node(
        "report",
        "reproduce",
        "Report & reproduce",
        "versions · traces · report",
        "core",
        1010,
        34,
        "Saved versions, the trace of every step, step-by-step replay and the Word report.",
    ),
    Node(
        "enrich",
        "enrich",
        "Enrich",
        "merge · derive · impute",
        "optional",
        516,
        150,
        "Merge external sources, derive variables from formulas or AI suggestions, impute missing values. "
        "Incremental: validate and analyse again after each change, then come back for the next one.",
    ),
    Node(
        "docs",
        "documents",
        "Your documents",
        "literature · protocols",
        "optional",
        516,
        270,
        "Index your literature, protocols or data dictionaries, chat with them and extract a knowledge graph.",
        needs_ai=True,
    ),
    Node(
        "varmap",
        "variable_map",
        "Variable map",
        "your field's taxonomy",
        "optional",
        796,
        270,
        "A taxonomy of your variables grounded in your clinical domain and documents: definitions, units, relationships.",
        needs_ai=True,
    ),
)


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    src_anchor: str  # "<side>:<fraction along that side>", side in l, r, t, b
    dst_anchor: str
    dashed: bool = False
    label: str = ""
    both: bool = False  # two-way link, an arrow head at each end
    label_t: float = 0.5  # where along the curve the label sits (0 = source, 1 = target)
    label_dx: float = 0.0
    label_dy: float = -8.0
    label_anchor: str = "middle"  # SVG text-anchor


# Each link has its own lane: no two links share an anchor or cross, and labels sit beside their line.
# Enrichment is incremental (derive or impute, validate again, analyse, come back for the next variable),
# so its links with Validate and Explore & analyse are two-way.
EDGES: tuple[Edge, ...] = (
    Edge("load", "validate", "r:0.5", "l:0.5"),
    Edge("validate", "analyse", "r:0.5", "l:0.5"),
    Edge("analyse", "report", "r:0.5", "l:0.5"),
    Edge(
        "validate",
        "enrich",
        "b:0.75",
        "l:0.5",
        dashed=True,
        label="iterate",
        both=True,
        label_t=0.75,
    ),
    Edge(
        "enrich",
        "analyse",
        "r:0.5",
        "b:0.25",
        dashed=True,
        label="iterate",
        both=True,
        label_t=0.25,
    ),
    Edge(
        "docs",
        "enrich",
        "t:0.5",
        "b:0.5",
        dashed=True,
        label="suggests variables",
        label_dx=10,
        label_dy=4,
        label_anchor="start",
    ),
    Edge("docs", "varmap", "r:0.5", "l:0.5", dashed=True, label="grounds"),
    Edge(
        "varmap",
        "analyse",
        "t:0.7",
        "b:0.7",
        dashed=True,
        label="informs",
        label_dx=10,
        label_dy=4,
        label_anchor="start",
    ),
)

# Legend drawn in the free bottom-left corner: (x, y) of its first row; box samples, then line samples.
LEGEND_ORIGIN = (24, 284)
LEGEND_BOXES = (("core", "core step"), ("optional", "optional path"))
LEGEND_LINES = (
    ("", "main flow", False),
    ("dashed", "optional link", False),
    ("dashed", "loop: go back and forth as needed", True),
)
LEGEND_COLUMN_2 = 176  # x offset of the line samples
LABEL_CHAR_W, LABEL_H = (
    6.2,
    12.0,
)  # generous text metrics (11-11.5 px font) used by the layout checks

_NODE_BY_ID = {n.id: n for n in NODES}
_NORMALS = {"l": (-1.0, 0.0), "r": (1.0, 0.0), "t": (0.0, -1.0), "b": (0.0, 1.0)}

Point = tuple[float, float]
Box = tuple[float, float, float, float]  # x0, y0, x1, y1


def _anchor(node: Node, spec: str) -> tuple[Point, Point]:
    """Point on the box border and the outward normal there."""
    side, _, frac = spec.partition(":")
    f = float(frac or 0.5)
    x, y, w, h = node.x, node.y, BOX_W, BOX_H
    point = {
        "l": (x, y + h * f),
        "r": (x + w, y + h * f),
        "t": (x + w * f, y),
        "b": (x + w * f, y + h),
    }[side]
    return point, _NORMALS[side]


def _controls(edge: Edge) -> tuple[Point, Point, Point, Point]:
    """Cubic Bezier that leaves and enters each box perpendicular to its border."""
    p0, n0 = _anchor(_NODE_BY_ID[edge.src], edge.src_anchor)
    p3, n3 = _anchor(_NODE_BY_ID[edge.dst], edge.dst_anchor)
    pull = max(28.0, (abs(p3[0] - p0[0]) + abs(p3[1] - p0[1])) * 0.42)
    c1 = (p0[0] + n0[0] * pull, p0[1] + n0[1] * pull)
    c2 = (p3[0] + n3[0] * pull, p3[1] + n3[1] * pull)
    return p0, c1, c2, p3


def _bezier(c: tuple[Point, Point, Point, Point], t: float) -> Point:
    u = 1 - t
    a, b, d, e = u**3, 3 * u * u * t, 3 * u * t * t, t**3
    return (
        a * c[0][0] + b * c[1][0] + d * c[2][0] + e * c[3][0],
        a * c[0][1] + b * c[1][1] + d * c[2][1] + e * c[3][1],
    )


def edge_path(edge: Edge) -> str:
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = _controls(edge)
    return f"M{x0:.1f} {y0:.1f} C{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {x3:.1f} {y3:.1f}"


def edge_points(edge: Edge, samples: int = 64) -> list[Point]:
    """Points along the drawn curve (used by the layout checks)."""
    c = _controls(edge)
    return [_bezier(c, i / samples) for i in range(samples + 1)]


def label_position(edge: Edge) -> Point:
    x, y = _bezier(_controls(edge), edge.label_t)
    return x + edge.label_dx, y + edge.label_dy


def text_box(x: float, y: float, text: str, anchor: str = "start") -> Box:
    """Approximate extent of one line of text drawn at baseline (x, y)."""
    width = LABEL_CHAR_W * len(text)
    x0 = {"start": x, "middle": x - width / 2, "end": x - width}[anchor]
    return x0, y - LABEL_H + 3, x0 + width, y + 3


def label_box(edge: Edge) -> Box | None:
    if not edge.label:
        return None
    x, y = label_position(edge)
    return text_box(x, y, edge.label, edge.label_anchor)


def node_box(node: Node) -> Box:
    return node.x, node.y, node.x + BOX_W, node.y + BOX_H


def next_label_box(node: Node) -> Box:
    """Extent of the "suggested next" caption drawn above a box."""
    return text_box(node.x + BOX_W / 2, node.y - 12, "suggested next", "middle")


def legend_box() -> Box:
    ox, oy = LEGEND_ORIGIN
    longest = max(len(text) for _, text, _ in LEGEND_LINES)
    rows = max(len(LEGEND_BOXES), len(LEGEND_LINES))
    return (
        ox - 4,
        oy - 12,
        ox + LEGEND_COLUMN_2 + 60 + LABEL_CHAR_W * longest,
        oy + (rows - 1) * 24 + 8,
    )


def _legend_svg() -> str:
    ox, oy = LEGEND_ORIGIN
    parts = ['<g class="legend" aria-hidden="true">']
    for i, (kind, text) in enumerate(LEGEND_BOXES):
        y = oy + i * 24
        parts.append(f'<rect class="{kind}" x="{ox}" y="{y - 8}" width="30" height="16" rx="5"/>')
        parts.append(f'<text x="{ox + 40}" y="{y + 4}">{html.escape(text)}</text>')
    lx = ox + LEGEND_COLUMN_2
    for i, (cls, text, both) in enumerate(LEGEND_LINES):
        y = oy + i * 24
        start = ' marker-start="url(#wf-arrow)"' if both else ""
        parts.append(
            f'<path class="{cls}" d="M{lx + 2} {y} L{lx + 46} {y}"{start} marker-end="url(#wf-arrow)"/>'
        )
        parts.append(f'<text x="{lx + 60}" y="{y + 4}">{html.escape(text)}</text>')
    parts.append("</g>")
    return "".join(parts)


def suggested_next(has_data: bool) -> str:
    return "validate" if has_data else "load"


def workflow_svg(has_data: bool, ai_on: bool) -> str:
    """The workflow drawing; colours come from CSS classes so the component can apply the active theme."""
    width, height = CANVAS
    nxt = suggested_next(has_data)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="group" aria-label="Research workflow" xmlns="http://www.w3.org/2000/svg">',
        "<defs>",
        '<marker id="wf-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path class="arrow" d="M0 0 L10 5 L0 10 z"/></marker>',
        '<marker id="wf-arrow-hi" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path class="arrow-hi" d="M0 0 L10 5 L0 10 z"/></marker>',
        "</defs>",
    ]
    # the pulse ring sits under the arrows so an arrow head is never crossed by it
    ring = _NODE_BY_ID[nxt]
    parts.append(
        f'<rect class="ring" data-ring-for="{ring.id}" x="{ring.x - 5}" y="{ring.y - 5}" '
        f'width="{BOX_W + 10}" height="{BOX_H + 10}" rx="16"/>'
    )
    for edge in EDGES:
        cls = "edge" + (" dashed" if edge.dashed else "") + (" both" if edge.both else "")
        start = ' marker-start="url(#wf-arrow)"' if edge.both else ""
        parts.append(f'<g class="{cls}" data-from="{edge.src}" data-to="{edge.dst}">')
        parts.append(f'<path d="{edge_path(edge)}"{start} marker-end="url(#wf-arrow)"/>')
        if edge.label:
            lx, ly = label_position(edge)
            parts.append(
                f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{edge.label_anchor}">'
                f"{html.escape(edge.label)}</text>"
            )
        parts.append("</g>")
    parts.append(_legend_svg())
    for node in NODES:
        classes = [node.kind]
        if node.id == "load" and has_data:
            classes.append("done")
        if node.id == nxt:
            classes.append("next")
        if node.needs_ai and not ai_on:
            classes.append("ai-off")
        label = f"{node.title}: {node.description} Opens the {node.page.replace('_', ' ')} page."
        x, y = node.x, node.y
        parts.append(
            f'<g class="node {" ".join(classes)}" data-node="{node.id}" tabindex="0" role="link" '
            f'aria-label="{html.escape(label, quote=True)}">'
        )
        parts.append(
            f'<rect class="box" x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="12"/>'
        )
        parts.append(
            f'<text class="title" x="{x + 14}" y="{y + 26}">{html.escape(node.title)}</text>'
        )
        parts.append(f'<text class="sub" x="{x + 14}" y="{y + 45}">{html.escape(node.sub)}</text>')
        if node.id == "load" and has_data:
            cx, cy = x + BOX_W - 14, y + 14
            parts.append(f'<circle class="check" cx="{cx}" cy="{cy}" r="8"/>')
            parts.append(f'<path class="tick" d="M{cx - 4} {cy} l3 3 l5 -6"/>')
        if node.needs_ai and not ai_on:
            parts.append(
                f'<rect class="pill" x="{x + BOX_W - 56}" y="{y + 7}" width="48" height="17" rx="8.5"/>'
            )
            parts.append(
                f'<text class="pill-text" x="{x + BOX_W - 32}" y="{y + 19.5}" text-anchor="middle">AI off</text>'
            )
        if node.id == nxt:
            parts.append(
                f'<text class="next-text" x="{x + BOX_W / 2}" y="{y - 12}" text-anchor="middle">suggested next</text>'
            )
        parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)


def workflow_model(has_data: bool, ai_on: bool) -> dict:
    """Everything the interactive map needs: the drawing, per-node details for the hint line, and the edges."""
    svg = workflow_svg(has_data, ai_on)
    return {
        "svg": svg,
        # The browser redraws only when the drawing changes (state or a new release of the diagram).
        "version": hashlib.sha1(svg.encode("utf-8"), usedforsecurity=False).hexdigest()[:12],
        "nodes": {
            n.id: {
                "title": n.title.lstrip("0123456789 ").strip(),
                "description": n.description,
                "needs_ai": n.needs_ai,
            }
            for n in NODES
        },
        "edges": [[e.src, e.dst] for e in EDGES],
        "hint": "Hover a box to see what it does; click it (or press Enter) to open that step.",
        "ai_on": ai_on,
    }


def node_page(node_id: str) -> str | None:
    node = _NODE_BY_ID.get(node_id)
    return node.page if node else None


__all__ = [
    "BRANCHES",
    "EDGES",
    "NODES",
    "Branch",
    "Edge",
    "Node",
    "edge_path",
    "edge_points",
    "label_box",
    "legend_box",
    "next_label_box",
    "node_box",
    "node_page",
    "suggested_next",
    "workflow_model",
    "workflow_svg",
]
