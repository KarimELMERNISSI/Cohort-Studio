"""
Rendering backends, a connector family (core.registry): each is registered here as ``"module:Class"`` and imported
only when used, so an uninstalled optional library (yFiles) never breaks the others. :mod:`graph.manager` chooses
among them.

The names are stored in the users' preferences and in ``CMP_GRAPH_RENDERER``: they never change.
"""

from __future__ import annotations

from core.registry import ConnectorRegistry
from graph.base import BaseGraphRenderer

OFFLINE_DEFAULT = "plotly"  # first of the automatic order, and the fallback when a drawing fails

GRAPH_RENDERERS: ConnectorRegistry[BaseGraphRenderer] = ConnectorRegistry(
    "graph engine",
    setting="CMP_GRAPH_RENDERER",
    auto=(
        OFFLINE_DEFAULT,
        "graphviz",
    ),  # yFiles is never chosen automatically (internet, telemetry)
)
GRAPH_RENDERERS.register("plotly", "graph.renderers.plotly:PlotlyGraphRenderer")
GRAPH_RENDERERS.register(
    "yfiles", "graph.renderers.yfiles:YFilesGraphRenderer", extra="graph-yfiles"
)
GRAPH_RENDERERS.register("graphviz", "graph.renderers.graphviz:GraphvizGraphRenderer")

__all__ = ["GRAPH_RENDERERS", "OFFLINE_DEFAULT"]
