"""
Library-agnostic graph rendering layer.

* :mod:`graph.model` - the semantic graph model (``GraphSpec``) shared by builders and renderers.
* :mod:`graph.palette` - colours, style rules and legends (single source of truth).
* :mod:`graph.builders` - pure functions turning domain data into a ``GraphSpec``.
* :mod:`graph.base` / :mod:`graph.manager` - the renderer contract and backend selection.
* :mod:`graph.renderers` - backends (Plotly/NetworkX offline default, yFiles opt-in, Graphviz), registered in
  ``GRAPH_RENDERERS``, a connector family on :mod:`core.registry`.
* :mod:`ui.graph_view` - the Streamlit entry point ``render_graph`` (in the ui layer, M5b).

Importing this package never imports Streamlit or a rendering library.
"""

from graph.model import (
    EdgeSpec,
    EdgeStyle,
    GraphKind,
    GraphSpec,
    GraphStats,
    LayoutHint,
    LegendEntry,
    LineStyle,
    NodeShape,
    NodeSpec,
    NodeStyle,
    RenderOptions,
    SelectionResult,
    edge_id,
)

__all__ = [
    "EdgeSpec",
    "EdgeStyle",
    "GraphKind",
    "GraphSpec",
    "GraphStats",
    "LayoutHint",
    "LegendEntry",
    "LineStyle",
    "NodeShape",
    "NodeSpec",
    "NodeStyle",
    "RenderOptions",
    "SelectionResult",
    "edge_id",
]
