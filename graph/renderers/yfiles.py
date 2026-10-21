"""
yFiles Graphs for Streamlit renderer (optional, proprietary backend).

Translation of the semantic model to ``yfiles_graphs_for_streamlit`` 2.x primitives. Pitfalls of
the 2.x API that this adapter encapsulates:

* ``Layout.HIERARCHIC`` no longer exists (``HIERARCHICAL``); layouts are resolved by member name.
* ``height``, ``directed``, ``sync_selection``, ``key`` and ``neighborhood`` are ``show()`` arguments;
  setting them as widget attributes is silently ignored.
* ``EdgeStyle(directed=False)`` is dropped by the serializer; per-edge direction is passed through
  ``directed_mapping`` and the global ``show(directed=...)`` flag instead.
* Label styling goes through ``LabelStyle`` objects returned by the label mappings.
* Mapping errors are swallowed into ``widget._error``; the adapter re-raises them as
  :class:`~graph.base.RenderError` so the UI can fall back to the offline renderer.

The component imports Streamlit, so it is loaded only to translate and to draw (R3): the availability check finds it
and reads its version without importing it, and the API this adapter needs is checked when drawing, where a mismatch
is a :class:`~graph.base.RenderError` with the explained fallback.

License: yWorks "yFiles Streamlit Component Free License"; the component loads its runtime from
yworks.com at render time (internet required) and reports anonymised graph-structure usage data.
Install with the optional extra ``graph-yfiles``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
from collections.abc import Callable
from typing import Any, NamedTuple

from core.registry import Availability
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

LICENSE_NOTE = (
    "yFiles Graphs for Streamlit (yWorks GmbH, free license): the graph runtime is loaded from "
    "yworks.com at render time (internet required) and anonymised graph-structure usage data is "
    "reported to yWorks."
)
SHAPE_MEMBERS: dict[NodeShape, str] = {
    NodeShape.ELLIPSE: "ELLIPSE",
    NodeShape.RECTANGLE: "RECTANGLE",
    NodeShape.ROUND_RECTANGLE: "ROUND_RECTANGLE",
    NodeShape.HEXAGON: "HEXAGON",
    NodeShape.OCTAGON: "OCTAGON",
    NodeShape.PILL: "PILL",
    NodeShape.TRIANGLE: "TRIANGLE",
}
DASH_MEMBERS: dict[LineStyle, str] = {
    LineStyle.SOLID: "SOLID",
    LineStyle.DASHED: "DASH",
    LineStyle.DOTTED: "DOT",
    LineStyle.DASH_DOT: "DASH_DOT",
}
LAYOUT_MEMBERS: dict[LayoutHint, str] = {
    LayoutHint.ORGANIC: "ORGANIC",
    LayoutHint.HIERARCHICAL: "HIERARCHICAL",
    LayoutHint.CIRCULAR: "CIRCULAR",
    LayoutHint.RADIAL: "RADIAL",
    LayoutHint.TREE: "TREE",
    LayoutHint.ORTHOGONAL: "ORTHOGONAL",
}
DEFAULT_LAYOUT: dict[GraphKind, LayoutHint] = {
    GraphKind.CORRELATION: LayoutHint.ORGANIC,
    GraphKind.TAXONOMY: LayoutHint.HIERARCHICAL,
    GraphKind.DOCUMENT: LayoutHint.HIERARCHICAL,
    GraphKind.FORMULA: LayoutHint.HIERARCHICAL,
}
REQUIRED_SHOW_PARAMS = (
    "directed",
    "graph_layout",
    "sync_selection",
    "height",
    "key",
    "neighborhood",
)
REQUIRED_LABEL_FIELDS = ("text", "position", "wrapping", "color")


class YFilesPayload(NamedTuple):
    nodes: list[Any]
    edges: list[Any]
    mappings: dict[str, Callable[[Any], Any]]
    show_kwargs: dict[str, Any]


def _member(enum_cls: Any, name: str, fallback: str) -> Any:
    return getattr(enum_cls, name, getattr(enum_cls, fallback))


def _api_problems(y: Any) -> list[str]:
    """What the installed component lacks of the 2.x API this adapter uses (none when it matches)."""
    problems: list[str] = []
    layouts = {member.value for member in getattr(y, "Layout", [])}
    if "hierarchical" not in layouts:
        problems.append("Layout.HIERARCHICAL missing")
    try:
        params = inspect.signature(y.StreamlitGraphWidget.show).parameters
        problems.extend(f"show() lacks '{p}'" for p in REQUIRED_SHOW_PARAMS if p not in params)
    except Exception as exc:  # pragma: no cover - defensive
        problems.append(f"cannot inspect show(): {exc}")
    fields = getattr(getattr(y, "LabelStyle", None), "__dataclass_fields__", {})
    problems.extend(f"LabelStyle lacks '{f}'" for f in REQUIRED_LABEL_FIELDS if f not in fields)
    return problems


class YFilesGraphRenderer(BaseGraphRenderer):
    name = "yfiles"
    display_name = "yFiles (interactive, needs internet)"
    capabilities = RendererCapabilities(
        supports_selection=True,
        supports_edge_selection=True,
        supports_heat=True,
        supports_node_shapes=True,
        supports_edge_labels=True,
        supports_mixed_direction=True,
        supports_hierarchical=True,
        supports_drag=True,
        supports_html_export=False,
        requires_internet=True,
        telemetry=True,
        license_note=LICENSE_NOTE,
    )

    @classmethod
    def availability(cls) -> Availability:
        """Installed, in the major version this adapter targets; found without importing it (it imports Streamlit)."""
        found = Availability.of_module("yfiles_graphs_for_streamlit", extra="graph-yfiles")
        if not found.ok:
            return found
        try:
            version = importlib.metadata.version("yfiles-graphs-for-streamlit")
            major = int(version.split(".")[0])
        except importlib.metadata.PackageNotFoundError, ValueError:
            return Availability(False, "its package metadata is missing", "graph-yfiles")
        if major != 2:
            return Availability(
                False, f"version {version} is installed and this adapter needs 2.x", "graph-yfiles"
            )
        return Availability(True)

    # ------------------------------------------------------------------ translate
    def translate(self, spec: GraphSpec, options: RenderOptions) -> YFilesPayload:
        y = importlib.import_module("yfiles_graphs_for_streamlit")
        by_id = {n.id: n for n in spec.nodes}
        edges_by_id = {e.id: e for e in spec.edges}

        nodes = [
            y.Node(
                id=n.id,
                properties={
                    **n.data,
                    "label": n.label or n.id,
                    "category": n.category,
                    "tooltip": n.tooltip,
                },
            )
            for n in spec.nodes
        ]
        edges = [
            y.Edge(
                start=e.source,
                end=e.target,
                id=e.id,
                properties={**e.data, "label": e.label, "category": e.category},
            )
            for e in spec.edges
        ]

        # Mapping callables receive the user's Node/Edge object (single, unannotated parameter).
        def node_label(element):
            n = by_id[str(element["id"])]
            return y.LabelStyle(
                text=n.label or n.id,
                color=n.style.text_color or "#000000",
                position=y.LabelPosition.CENTER,
                wrapping=y.TextWrapping.WRAP_WORD_ELLIPSIS,
                maximum_width=160,
            )

        def node_style(element):
            n = by_id[str(element["id"])]
            return y.NodeStyle(
                color=n.style.color,
                shape=_member(y.NodeShape, SHAPE_MEMBERS[n.style.shape], "ELLIPSE"),
            )

        def node_type(element):
            return by_id[str(element["id"])].category or None

        def node_scale(element):
            n = by_id[str(element["id"])]
            return float(n.size) if n.size else 1.0

        def heat(element):  # applied to nodes and edges; edges get no heat
            n = by_id.get(str(element["id"]))
            return float(n.weight) if n is not None and n.weight is not None else None

        def edge_style(element):
            e = edges_by_id[str(element["id"])]
            return y.EdgeStyle(
                color=e.style.color,
                thickness=float(e.style.width),
                dash_style=_member(y.DashStyle, DASH_MEMBERS[e.style.line], "SOLID"),
                directed=True if e.directed else None,  # False would be dropped by the serializer
            )

        def edge_label(element):
            e = edges_by_id[str(element["id"])]
            if not options.show_edge_labels or not e.label:
                return ""
            return y.LabelStyle(
                text=e.label, font_size=10, color="#333333", background_color="#FFFFFFCC"
            )

        def directed(element):
            return bool(edges_by_id[str(element["id"])].directed)

        mappings: dict[str, Callable[[Any], Any]] = {
            "node_label_mapping": node_label,
            "node_styles_mapping": node_style,
            "node_type_mapping": node_type,
            "node_scale_factor_mapping": node_scale,
            "edge_styles_mapping": edge_style,
            "edge_label_mapping": edge_label,
            "directed_mapping": directed,
        }
        if options.heat:
            mappings["heat_mapping"] = heat

        layout = options.layout or spec.layout
        if layout == LayoutHint.AUTO:
            layout = DEFAULT_LAYOUT.get(spec.kind, LayoutHint.ORGANIC)
        global_directed = spec.directed and (
            all(e.directed for e in spec.edges) if spec.edges else True
        )
        overrides = options.renderer_overrides.get(self.name, {})
        show_kwargs: dict[str, Any] = {
            "directed": global_directed,
            "graph_layout": _member(y.Layout, LAYOUT_MEMBERS.get(layout, "ORGANIC"), "ORGANIC"),
            "sync_selection": bool(options.selectable),
            "height": int(options.height),
            "key": f"{options.key}__yfiles",
            "overview": overrides.get("overview", len(spec.nodes) > 30),
            "sidebar": overrides.get("sidebar", {"enabled": False}),
            "neighborhood": {
                "max_distance": int(options.focus_hops),
                "selected_nodes": [options.focus_node] if options.focus_node else [],
            },
        }
        return YFilesPayload(nodes, edges, mappings, show_kwargs)

    # ------------------------------------------------------------------ render
    def render(self, spec: GraphSpec, options: RenderOptions) -> SelectionResult:
        try:
            y = importlib.import_module("yfiles_graphs_for_streamlit")
            problems = _api_problems(y)
            if problems:
                raise RenderError(
                    f"the installed yFiles component does not match this adapter: {'; '.join(problems)}"
                )
            payload = self.translate(spec, options)
            widget = y.StreamlitGraphWidget(
                nodes=payload.nodes, edges=payload.edges, **payload.mappings
            )
            selected_nodes, selected_edges = widget.show(**payload.show_kwargs)
        except RenderError:
            raise
        except Exception as exc:
            raise RenderError(f"yFiles renderer failed: {exc}") from exc
        error = getattr(widget, "_error", None)
        if error is not None:
            message = getattr(widget, "_errorMessage", "") or repr(error)
            raise RenderError(f"yFiles mapping error: {message}")
        return SelectionResult(
            node_ids=tuple(str(n.get("id")) for n in selected_nodes or []),
            edge_ids=tuple(str(e.get("id")) for e in selected_edges or []),
            renderer=self.name,
        )


__all__ = ["LICENSE_NOTE", "YFilesGraphRenderer", "YFilesPayload"]
