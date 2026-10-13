"""
Colour tokens, style rules and legend entries for every graph kind (single source of truth).

Builders resolve styles through :func:`node_style_for` / :func:`edge_style_for`; legends are
generated from the same rules with :func:`legend_entries`, so a picture and its legend cannot
drift apart. Renderers never choose colours. This module has no UI dependency.

The hex values reproduce the colours the application used before the connector refactor, so the
change is visually neutral; adjust them here (and only here) to re-theme every graph.
"""

from __future__ import annotations

from graph.model import EdgeStyle, GraphKind, LegendEntry, LineStyle, NodeShape, NodeStyle

PALETTE: dict[str, str] = {
    # correlation network
    "corr.node.variable": "#87CEEB",
    "corr.node.focus": "#FFA500",
    "corr.pos.negligible": "#D3D3D3",
    "corr.pos.weak": "#ADD8E6",
    "corr.pos.moderate": "#4169E1",
    "corr.pos.strong": "#0000FF",
    "corr.pos.very_strong": "#00008B",
    "corr.neg.negligible": "#D3D3D3",
    "corr.neg.weak": "#F08080",
    "corr.neg.moderate": "#CD5C5C",
    "corr.neg.strong": "#FF0000",
    "corr.neg.very_strong": "#8B0000",
    # variable taxonomy (clinical knowledge graph)
    "tax.node.category": "#E0E0E0",
    "tax.node.input": "#4285F4",
    "tax.node.derived": "#FBBC05",
    "tax.node.outcome": "#34A853",
    "tax.node.formula": "#EA4335",
    "tax.node.input_external": "#AECBFA",
    "tax.node.derived_external": "#FEEFC3",
    "tax.edge.category_member": "#BDBDBD",
    "tax.edge.input": "#4285F4",
    "tax.edge.yields": "#EA4335",
    "tax.edge.related": "#9E9E9E",
    # document knowledge graph
    "doc.node.Concept": "#4285F4",
    "doc.node.Metric": "#34A853",
    "doc.node.Finding": "#EA4335",
    "doc.node.Disease": "#FBBC05",
    "doc.node.Treatment": "#AA46BB",
    "doc.node.Method": "#00ACC1",
    "doc.node.Other": "#9E9E9E",
    "doc.node.Inferred": "#CFD8DC",
    "doc.edge.relation": "#BDBDBD",
    # formula dependency mini-graph
    "formula.node.input": "#4285F4",
    "formula.node.missing": "#EA4335",
    "formula.node.target": "#34A853",
    "formula.edge.input": "#9E9E9E",
    "formula.edge.missing": "#EA4335",
}

DARK_OVERRIDES: dict[str, str] = {
    "tax.node.category": "#5F6368",
    "corr.pos.negligible": "#6B6B6B",
    "corr.neg.negligible": "#6B6B6B",
    "tax.edge.category_member": "#8A8A8A",
    "doc.edge.relation": "#8A8A8A",
    "doc.node.Inferred": "#607D8B",
    "formula.edge.input": "#B0B0B0",
}

STRENGTH_SLUGS: tuple[str, ...] = ("negligible", "weak", "moderate", "strong", "very_strong")
_STRENGTH_LINES: dict[str, tuple[LineStyle, float]] = {
    "negligible": (LineStyle.DOTTED, 1.0),
    "weak": (LineStyle.DASHED, 1.5),
    "moderate": (LineStyle.SOLID, 2.5),
    "strong": (LineStyle.SOLID, 3.5),
    "very_strong": (LineStyle.SOLID, 5.0),
}
_STRENGTH_LABELS: dict[str, str] = {
    "negligible": "Negligible",
    "weak": "Weak",
    "moderate": "Moderate",
    "strong": "Strong",
    "very_strong": "Very strong",
}

# category -> (colour token, shape, legend label, legend group)
_NODE_RULES: dict[GraphKind, dict[str, tuple[str, NodeShape, str, str]]] = {
    GraphKind.CORRELATION: {
        "variable": ("corr.node.variable", NodeShape.ELLIPSE, "Variable", "Nodes"),
        "focus": ("corr.node.focus", NodeShape.ELLIPSE, "Focus variable", "Nodes"),
    },
    GraphKind.TAXONOMY: {
        "category": ("tax.node.category", NodeShape.ROUND_RECTANGLE, "Category", "Nodes"),
        "input": ("tax.node.input", NodeShape.ELLIPSE, "Input variable", "Nodes"),
        "derived": ("tax.node.derived", NodeShape.ELLIPSE, "Derived variable", "Nodes"),
        "outcome": ("tax.node.outcome", NodeShape.ELLIPSE, "Outcome variable", "Nodes"),
        "formula": ("tax.node.formula", NodeShape.HEXAGON, "Formula", "Nodes"),
        "input_external": (
            "tax.node.input_external",
            NodeShape.ELLIPSE,
            "External input variable",
            "Nodes",
        ),
        "derived_external": (
            "tax.node.derived_external",
            NodeShape.ELLIPSE,
            "External derived variable",
            "Nodes",
        ),
    },
    GraphKind.DOCUMENT: {
        "Concept": ("doc.node.Concept", NodeShape.ELLIPSE, "Concept", "Entities"),
        "Metric": ("doc.node.Metric", NodeShape.ELLIPSE, "Metric", "Entities"),
        "Finding": ("doc.node.Finding", NodeShape.ELLIPSE, "Finding", "Entities"),
        "Disease": ("doc.node.Disease", NodeShape.ELLIPSE, "Disease", "Entities"),
        "Treatment": ("doc.node.Treatment", NodeShape.ELLIPSE, "Treatment", "Entities"),
        "Method": ("doc.node.Method", NodeShape.ELLIPSE, "Method", "Entities"),
        "Other": ("doc.node.Other", NodeShape.ELLIPSE, "Other", "Entities"),
        "Inferred": (
            "doc.node.Inferred",
            NodeShape.ROUND_RECTANGLE,
            "Inferred (mentioned only in a relation)",
            "Entities",
        ),
    },
    GraphKind.FORMULA: {
        "input": ("formula.node.input", NodeShape.ELLIPSE, "Input available", "Nodes"),
        "missing": ("formula.node.missing", NodeShape.ELLIPSE, "Input missing", "Nodes"),
        "target": ("formula.node.target", NodeShape.ROUND_RECTANGLE, "Target variable", "Nodes"),
    },
}

_DEFAULT_NODE_RULE: dict[GraphKind, str] = {
    GraphKind.CORRELATION: "variable",
    GraphKind.TAXONOMY: "input",
    GraphKind.DOCUMENT: "Other",
    GraphKind.FORMULA: "input",
}


def _correlation_edge_rules() -> dict[str, tuple[str, LineStyle, float, str, str]]:
    rules: dict[str, tuple[str, LineStyle, float, str, str]] = {}
    for sign, group in (("pos", "Positive correlation"), ("neg", "Negative correlation")):
        for slug in STRENGTH_SLUGS:
            token = f"corr.{sign}.{slug}"
            line, width = _STRENGTH_LINES[slug]
            rules[token] = (token, line, width, _STRENGTH_LABELS[slug], group)
    return rules


# category -> (colour token, line style, width, legend label, legend group)
_EDGE_RULES: dict[GraphKind, dict[str, tuple[str, LineStyle, float, str, str]]] = {
    GraphKind.CORRELATION: _correlation_edge_rules(),
    GraphKind.TAXONOMY: {
        "category_member": (
            "tax.edge.category_member",
            LineStyle.DASHED,
            1.5,
            "Belongs to category",
            "Relations",
        ),
        "input": ("tax.edge.input", LineStyle.SOLID, 2.0, "Input of a formula", "Relations"),
        "yields": ("tax.edge.yields", LineStyle.SOLID, 2.0, "Formula yields variable", "Relations"),
        "related": ("tax.edge.related", LineStyle.DOTTED, 1.5, "Related variables", "Relations"),
    },
    GraphKind.DOCUMENT: {
        "relation": (
            "doc.edge.relation",
            LineStyle.SOLID,
            2.0,
            "Relation (label shows its type)",
            "Relations",
        ),
    },
    GraphKind.FORMULA: {
        "input": ("formula.edge.input", LineStyle.SOLID, 2.0, "Input available", "Relations"),
        "missing": ("formula.edge.missing", LineStyle.DASHED, 2.0, "Input missing", "Relations"),
    },
}

_DEFAULT_EDGE_RULE: dict[GraphKind, str] = {
    GraphKind.CORRELATION: "corr.pos.weak",
    GraphKind.TAXONOMY: "related",
    GraphKind.DOCUMENT: "relation",
    GraphKind.FORMULA: "input",
}


def color(token: str, theme: str = "light") -> str:
    """Resolve a colour token for the given theme ('light' or 'dark')."""
    if theme == "dark" and token in DARK_OVERRIDES:
        return DARK_OVERRIDES[token]
    return PALETTE[token]


def text_color_on(background: str | None) -> str:
    """Black or white text, whichever contrasts better with the background colour."""
    if not background:
        return "#000000"
    hex_value = background.lstrip("#")[:6]
    try:
        r, g, b = (int(hex_value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return "#000000"

    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    luminance = 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)
    return "#000000" if luminance > 0.4 else "#FFFFFF"


def node_categories(kind: GraphKind) -> tuple[str, ...]:
    return tuple(_NODE_RULES[kind])


def edge_categories(kind: GraphKind) -> tuple[str, ...]:
    return tuple(_EDGE_RULES[kind])


def node_style_for(kind: GraphKind, category: str, theme: str = "light") -> NodeStyle:
    """Style for a node category; unknown categories fall back to the kind's default rule."""
    rules = _NODE_RULES[kind]
    token, shape, _, _ = rules.get(category, rules[_DEFAULT_NODE_RULE[kind]])
    fill = color(token, theme)
    return NodeStyle(color=fill, shape=shape, text_color=text_color_on(fill))


def edge_style_for(kind: GraphKind, category: str, theme: str = "light") -> EdgeStyle:
    """Style for an edge category; unknown categories fall back to the kind's default rule."""
    rules = _EDGE_RULES[kind]
    token, line, width, _, _ = rules.get(category, rules[_DEFAULT_EDGE_RULE[kind]])
    return EdgeStyle(color=color(token, theme), line=line, width=width)


def node_label(kind: GraphKind, category: str) -> str:
    """Human-readable label of a node category (for legends and filters)."""
    rules = _NODE_RULES[kind]
    return rules[category][2] if category in rules else category


def edge_label(kind: GraphKind, category: str) -> str:
    """Human-readable label of an edge category (for legends and filters)."""
    rules = _EDGE_RULES[kind]
    return rules[category][3] if category in rules else category


def legend_entries(kind: GraphKind, theme: str = "light") -> tuple[LegendEntry, ...]:
    """Legend derived from the same rules that style the graph."""
    entries: list[LegendEntry] = []
    for token, shape, label, group in _NODE_RULES[kind].values():
        entries.append(
            LegendEntry(
                label=label, swatch="node", color=color(token, theme), shape=shape, group=group
            )
        )
    for token, line, width, label, group in _EDGE_RULES[kind].values():
        entries.append(
            LegendEntry(
                label=label,
                swatch="edge",
                color=color(token, theme),
                line=line,
                width=width,
                group=group,
            )
        )
    return tuple(entries)


__all__ = [
    "DARK_OVERRIDES",
    "PALETTE",
    "STRENGTH_SLUGS",
    "color",
    "edge_categories",
    "edge_label",
    "edge_style_for",
    "legend_entries",
    "node_categories",
    "node_label",
    "node_style_for",
    "text_color_on",
]
