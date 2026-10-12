"""
Renderer-agnostic graph model.

A :class:`GraphSpec` describes *what* to draw: nodes and edges with semantic categories, styles
already resolved from the palette, a layout hint and a legend. It has no dependency on any
rendering library or on Streamlit. Renderers (``graph.renderers``) translate it into their own
primitives, and builders (``graph.builders``) produce it from domain data.

Design rules
------------
* Models are immutable (``frozen=True``) and JSON round-trippable (``to_json`` / ``from_json``).
* Identifiers are deterministic: builders derive edge ids with :func:`edge_id` and keep node order
  stable, so ``fingerprint()`` is stable across reruns and can serve as a cache or widget key.
* ``data`` payloads carry domain information for detail panels; they are never drawn.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from collections.abc import Iterable
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

HEX_COLOR_PATTERN = r"^#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$"


class GraphKind(StrEnum):
    """Which kind of graph a spec describes (drives palette rules and default layout)."""

    CORRELATION = "correlation"
    TAXONOMY = "taxonomy"
    DOCUMENT = "document"
    FORMULA = "formula"


class NodeShape(StrEnum):
    """Shapes every renderer can draw or approximate."""

    ELLIPSE = "ellipse"
    RECTANGLE = "rectangle"
    ROUND_RECTANGLE = "round_rectangle"
    HEXAGON = "hexagon"
    OCTAGON = "octagon"
    PILL = "pill"
    TRIANGLE = "triangle"


class LineStyle(StrEnum):
    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"
    DASH_DOT = "dash_dot"


class LayoutHint(StrEnum):
    """Layout families; each renderer maps them onto its own algorithms."""

    AUTO = "auto"
    ORGANIC = "organic"
    HIERARCHICAL = "hierarchical"
    CIRCULAR = "circular"
    RADIAL = "radial"
    TREE = "tree"
    ORTHOGONAL = "orthogonal"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class NodeStyle(_Frozen):
    color: str | None = Field(default=None, pattern=HEX_COLOR_PATTERN)
    shape: NodeShape = NodeShape.ELLIPSE
    border_color: str | None = Field(default=None, pattern=HEX_COLOR_PATTERN)
    text_color: str | None = Field(default=None, pattern=HEX_COLOR_PATTERN)


class EdgeStyle(_Frozen):
    color: str | None = Field(default=None, pattern=HEX_COLOR_PATTERN)
    line: LineStyle = LineStyle.SOLID
    width: float = Field(default=2.0, ge=0.5, le=8.0)


class NodeSpec(_Frozen):
    id: str = Field(min_length=1)
    label: str = ""
    category: str = ""
    style: NodeStyle = NodeStyle()
    weight: float | None = Field(default=None, ge=0.0, le=1.0)
    size: float | None = Field(default=None, gt=0.0)
    tooltip: str = ""
    group: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class EdgeSpec(_Frozen):
    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    label: str = ""
    category: str = ""
    directed: bool = True
    style: EdgeStyle = EdgeStyle()
    weight: float | None = Field(default=None, ge=0.0, le=1.0)
    tooltip: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class LegendEntry(_Frozen):
    label: str
    swatch: Literal["node", "edge"]
    color: str = Field(pattern=HEX_COLOR_PATTERN)
    shape: NodeShape | None = None
    line: LineStyle | None = None
    width: float | None = None
    group: str = ""


class GraphStats(_Frozen):
    node_count: int
    edge_count: int
    nodes_by_category: dict[str, int]
    edges_by_category: dict[str, int]
    component_count: int
    isolated_nodes: int


class GraphSpec(_Frozen):
    kind: GraphKind
    title: str = ""
    nodes: tuple[NodeSpec, ...] = ()
    edges: tuple[EdgeSpec, ...] = ()
    directed: bool = True
    layout: LayoutHint = LayoutHint.AUTO
    legend: tuple[LegendEntry, ...] = ()
    meta: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = 1

    # ----------------------------------------------------------------- validation
    @model_validator(mode="after")
    def _check_integrity(self) -> GraphSpec:
        node_counts = Counter(n.id for n in self.nodes)
        dupes = sorted(i for i, c in node_counts.items() if c > 1)
        if dupes:
            raise ValueError(f"duplicate node ids: {dupes[:5]}")
        edge_counts = Counter(e.id for e in self.edges)
        dupes = sorted(i for i, c in edge_counts.items() if c > 1)
        if dupes:
            raise ValueError(f"duplicate edge ids: {dupes[:5]}")
        dangling = [e.id for e in self.edges if e.source not in node_counts or e.target not in node_counts]
        if dangling:
            raise ValueError(f"edges reference unknown nodes: {dangling[:5]}")
        try:
            json.dumps([n.data for n in self.nodes] + [e.data for e in self.edges] + [self.meta])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"node/edge data and meta must be JSON-native: {exc}") from exc
        return self

    # ----------------------------------------------------------------- queries
    def node(self, node_id: str) -> NodeSpec | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def edge(self, edge_id_: str) -> EdgeSpec | None:
        for e in self.edges:
            if e.id == edge_id_:
                return e
        return None

    def node_ids(self) -> tuple[str, ...]:
        return tuple(n.id for n in self.nodes)

    def adjacency(self) -> dict[str, set[str]]:
        """Undirected adjacency (used for focus, components and statistics)."""
        adj: dict[str, set[str]] = {n.id: set() for n in self.nodes}
        for e in self.edges:
            adj[e.source].add(e.target)
            adj[e.target].add(e.source)
        return adj

    def neighbors(self, node_id: str) -> set[str]:
        return set(self.adjacency().get(node_id, set()))

    def subgraph(self, keep: Iterable[str]) -> GraphSpec:
        """Nodes whose id is in ``keep`` and the edges between them; order is preserved."""
        keep_set = set(keep)
        nodes = tuple(n for n in self.nodes if n.id in keep_set)
        edges = tuple(e for e in self.edges if e.source in keep_set and e.target in keep_set)
        return self.model_copy(update={"nodes": nodes, "edges": edges})

    def neighborhood(self, node_id: str, hops: int = 1) -> GraphSpec:
        """The node plus everything within ``hops`` edges of it (empty spec if the node is unknown)."""
        adj = self.adjacency()
        if node_id not in adj:
            return self.subgraph(())
        seen = {node_id}
        frontier = {node_id}
        for _ in range(max(0, int(hops))):
            frontier = {nb for n in frontier for nb in adj[n]} - seen
            if not frontier:
                break
            seen |= frontier
        return self.subgraph(seen)

    def component_of(self, node_id: str) -> GraphSpec:
        """The connected component containing ``node_id`` (empty spec if the node is unknown)."""
        adj = self.adjacency()
        if node_id not in adj:
            return self.subgraph(())
        seen = {node_id}
        queue: deque[str] = deque([node_id])
        while queue:
            current = queue.popleft()
            for nb in adj[current]:
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        return self.subgraph(seen)

    def filter(
        self,
        *,
        node_categories: Iterable[str] | None = None,
        edge_categories: Iterable[str] | None = None,
    ) -> GraphSpec:
        """Keep only the given node/edge categories (``None`` keeps everything)."""
        node_set = None if node_categories is None else set(node_categories)
        edge_set = None if edge_categories is None else set(edge_categories)
        nodes = tuple(n for n in self.nodes if node_set is None or n.category in node_set)
        keep = {n.id for n in nodes}
        edges = tuple(
            e
            for e in self.edges
            if e.source in keep and e.target in keep and (edge_set is None or e.category in edge_set)
        )
        return self.model_copy(update={"nodes": nodes, "edges": edges})

    def stats(self) -> GraphStats:
        adj = self.adjacency()
        seen: set[str] = set()
        components = 0
        for start in adj:
            if start in seen:
                continue
            components += 1
            seen.add(start)
            stack = [start]
            while stack:
                current = stack.pop()
                for nb in adj[current]:
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        return GraphStats(
            node_count=len(self.nodes),
            edge_count=len(self.edges),
            nodes_by_category=dict(sorted(Counter(n.category for n in self.nodes).items())),
            edges_by_category=dict(sorted(Counter(e.category for e in self.edges).items())),
            component_count=components,
            isolated_nodes=sum(1 for neighbours in adj.values() if not neighbours),
        )

    # ----------------------------------------------------------------- conversions
    def to_networkx(self, directed: bool | None = None):
        """Build a NetworkX graph (``DiGraph`` when any edge is directed unless overridden)."""
        import networkx as nx

        use_directed = any(e.directed for e in self.edges) if directed is None else directed
        g = nx.DiGraph() if use_directed else nx.Graph()
        for n in self.nodes:
            g.add_node(n.id, label=n.label, category=n.category, weight=n.weight)
        for e in self.edges:
            g.add_edge(e.source, e.target, id=e.id, category=e.category, weight=e.weight, directed=e.directed)
            if use_directed and not e.directed:
                g.add_edge(e.target, e.source, id=e.id, category=e.category, weight=e.weight, directed=False)
        return g

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, text: str) -> GraphSpec:
        return cls.model_validate_json(text)

    def fingerprint(self) -> str:
        """Stable short digest of the full spec (same content, same digest)."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:16]


def edge_id(category: str, source: str, target: str, *, undirected: bool = False) -> str:
    """Deterministic edge identifier; undirected edges sort their endpoints."""
    if undirected:
        a, b = sorted((source, target))
        return f"{category}:{a}|{b}"
    return f"{category}:{source}->{target}"


class RenderOptions(_Frozen):
    """Per-call rendering options understood by every renderer."""

    key: str = Field(min_length=1)
    height: int = Field(default=500, ge=200, le=2000)
    layout: LayoutHint | None = None
    selectable: bool = False
    heat: bool = False
    show_edge_labels: bool = True
    show_legend: bool = True
    focus_node: str | None = None
    focus_hops: int = Field(default=1, ge=1, le=3)
    theme: Literal["light", "dark", "auto"] = "auto"
    renderer_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)


class SelectionResult(_Frozen):
    """What the user selected in the rendered graph (empty when the renderer has no selection)."""

    node_ids: tuple[str, ...] = ()
    edge_ids: tuple[str, ...] = ()
    renderer: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.node_ids or self.edge_ids)


__all__ = [
    "EdgeSpec",
    "EdgeStyle",
    "GraphKind",
    "GraphSpec",
    "GraphStats",
    "HEX_COLOR_PATTERN",
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
