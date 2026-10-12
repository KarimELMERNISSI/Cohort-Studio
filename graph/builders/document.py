"""
Document knowledge graph: tolerant schema, cross-document merge and the graph builder.

The LLM returns per-document JSON (``nodes[{id,type,description,source_text,page_reference}]``,
``edges[{source,target,relation,description}]``). :func:`merge_document_graphs` unifies entities
across documents (case-insensitive and fuzzy matching, as before) **and remaps edge endpoints to the
merged entity ids**, which the previous implementation did not do; that omission produced the
spurious "Inferred" nodes in the graph. Genuinely dangling endpoints are still shown as flagged
``Inferred`` nodes so gaps in the extraction stay visible.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Collection, Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from graph.model import EdgeSpec, GraphKind, GraphSpec, LayoutHint, NodeSpec, edge_id
from graph.palette import edge_style_for, legend_entries, node_style_for

ENTITY_TYPES: tuple[str, ...] = ("Concept", "Metric", "Finding", "Disease", "Treatment", "Method", "Other")
INFERRED = "Inferred"
DEFAULT_RELATION = "relates to"


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    return str(value)


class Citation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    doc: str = ""
    page: str = ""
    text: str = ""

    @field_validator("doc", "page", "text", mode="before")
    @classmethod
    def _coerce_text(cls, value: Any) -> str:
        return _as_text(value)


class DocNode(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    type: str = "Concept"
    description: str = ""
    citations: list[Citation] = Field(default_factory=list)
    source_text: str = ""
    page_reference: str = ""

    @field_validator("id", "type", "description", "source_text", "page_reference", mode="before")
    @classmethod
    def _coerce_text(cls, value: Any) -> str:
        return _as_text(value).strip()

    @field_validator("citations", mode="before")
    @classmethod
    def _coerce_citations(cls, value: Any) -> list[Any]:
        return [c for c in (value or []) if isinstance(c, (dict, Citation))]


class DocEdge(BaseModel):
    model_config = ConfigDict(extra="ignore")
    source: str
    target: str
    relation: str = DEFAULT_RELATION
    description: str = ""

    @field_validator("source", "target", "relation", "description", mode="before")
    @classmethod
    def _coerce_text(cls, value: Any) -> str:
        return _as_text(value).strip()


class DocumentGraph(BaseModel):
    model_config = ConfigDict(extra="ignore")
    nodes: list[DocNode] = Field(default_factory=list)
    edges: list[DocEdge] = Field(default_factory=list)
    formulas: list[dict[str, Any]] = Field(default_factory=list)
    summaries: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def parse(cls, graph_json: Mapping[str, Any] | DocumentGraph | None) -> DocumentGraph:
        """Tolerant parsing of LLM / persisted JSON: malformed entries are skipped, never fatal."""
        if graph_json is None:
            return cls()
        if isinstance(graph_json, DocumentGraph):
            return graph_json
        nodes: list[DocNode] = []
        for raw in graph_json.get("nodes", []) or []:
            if not isinstance(raw, Mapping) or not _as_text(raw.get("id")).strip():
                continue
            nodes.append(DocNode.model_validate(dict(raw)))
        edges: list[DocEdge] = []
        for raw in graph_json.get("edges", []) or []:
            if not isinstance(raw, Mapping):
                continue
            if not _as_text(raw.get("source")).strip() or not _as_text(raw.get("target")).strip():
                continue
            edges.append(DocEdge.model_validate(dict(raw)))
        formulas = [dict(f) for f in (graph_json.get("formulas", []) or []) if isinstance(f, Mapping)]
        summaries = [dict(s) for s in (graph_json.get("summaries", []) or []) if isinstance(s, Mapping)]
        return cls(nodes=nodes, edges=edges, formulas=formulas, summaries=summaries)

    def to_json_dict(self) -> dict[str, Any]:
        """Plain dict in the persisted contract (what the pages and files expect)."""
        return self.model_dump(mode="json")


def normalize_entity_type(raw: str | None) -> str:
    """Map a free-text entity type to one of :data:`ENTITY_TYPES` (substring match, default Other)."""
    text = _as_text(raw).strip().lower()
    if not text:
        return "Other"
    if text == INFERRED.lower():
        return INFERRED
    for known in ENTITY_TYPES:
        if known.lower() in text:
            return known
    return "Other"


def canonical_key(name: str) -> str:
    return re.sub(r"\s+", " ", _as_text(name).strip().casefold())


def build_alias_map(ids: Iterable[str], *, cutoff: float = 0.85) -> dict[str, str]:
    """``canonical_key -> canonical id`` for every id, plus fuzzy aliases between close ids."""
    aliases: dict[str, str] = {}
    canonical_ids: dict[str, str] = {}
    for node_id in sorted(set(ids), key=lambda s: (canonical_key(s), s)):
        key = canonical_key(node_id)
        if key in canonical_ids:
            aliases[key] = canonical_ids[key]
            continue
        match = difflib.get_close_matches(key, list(canonical_ids), n=1, cutoff=cutoff)
        if match:
            aliases[key] = canonical_ids[match[0]]
        else:
            canonical_ids[key] = node_id
            aliases[key] = node_id
    return aliases


def resolve_endpoint(name: str, aliases: Mapping[str, str], *, cutoff: float = 0.85) -> str:
    """Resolve an edge endpoint to a known node id (exact, case-insensitive, then fuzzy)."""
    key = canonical_key(name)
    if key in aliases:
        return aliases[key]
    match = difflib.get_close_matches(key, list(aliases), n=1, cutoff=cutoff)
    return aliases[match[0]] if match else _as_text(name).strip()


def merge_document_graphs(per_document: list[tuple[str, Mapping[str, Any] | DocumentGraph]], *, cutoff: float = 0.85) -> DocumentGraph:
    """Merge per-document graphs: unify entities, accumulate citations, remap and de-duplicate edges."""
    merged: dict[str, DocNode] = {}
    key_to_merged: dict[str, str] = {}
    pending_edges: list[DocEdge] = []
    formulas: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for doc_name, graph_json in per_document:
        graph = DocumentGraph.parse(graph_json)
        for node in graph.nodes:
            key = canonical_key(node.id)
            if not key:
                continue
            match_key = key if key in key_to_merged else None
            if match_key is None:
                close = difflib.get_close_matches(key, list(key_to_merged), n=1, cutoff=cutoff)
                match_key = close[0] if close else None
            citations = list(node.citations) or [
                Citation(doc=doc_name, page=node.page_reference or "Unknown", text=node.source_text)
            ]
            for citation in citations:
                if not citation.doc:
                    citation.doc = doc_name
            if match_key is None:
                merged[key] = DocNode(id=node.id, type=node.type or "Concept", description=node.description, citations=citations)
                key_to_merged[key] = key
            else:
                existing = merged[key_to_merged[match_key]]
                existing.citations.extend(citations)
                if len(node.description) > len(existing.description):
                    existing.description = node.description
                key_to_merged[key] = key_to_merged[match_key]
        pending_edges.extend(graph.edges)
        for formula in graph.formulas:
            formulas.append({**formula, "doc": doc_name})
        for summary in graph.summaries:
            summaries.append({**summary, "doc": doc_name})
        summary_single = graph_json.get("summary") if isinstance(graph_json, Mapping) else None
        if isinstance(summary_single, Mapping):
            summaries.append({**dict(summary_single), "doc": doc_name})

    aliases = {key: merged[target].id for key, target in key_to_merged.items()}
    edges: list[DocEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in pending_edges:
        source = resolve_endpoint(edge.source, aliases, cutoff=cutoff)
        target = resolve_endpoint(edge.target, aliases, cutoff=cutoff)
        relation = edge.relation or DEFAULT_RELATION
        signature = (canonical_key(source), canonical_key(target), relation.lower())
        if signature in seen or canonical_key(source) == canonical_key(target):
            continue
        seen.add(signature)
        edges.append(DocEdge(source=source, target=target, relation=relation, description=edge.description))
    return DocumentGraph(nodes=list(merged.values()), edges=edges, formulas=formulas, summaries=summaries)


def filter_by_documents(graph: Mapping[str, Any] | DocumentGraph | None, visible_docs: Collection[str]) -> DocumentGraph:
    """Keep the nodes cited by at least one visible document, the edges between them, and their formulas/summaries."""
    parsed = DocumentGraph.parse(graph)
    visible = set(visible_docs or [])
    if not visible:
        return DocumentGraph()
    nodes: list[DocNode] = []
    for node in parsed.nodes:
        relevant = [c for c in node.citations if c.doc in visible]
        if relevant:
            nodes.append(node.model_copy(update={"citations": relevant}))
    keep = {canonical_key(n.id) for n in nodes}
    edges = [e for e in parsed.edges if canonical_key(e.source) in keep and canonical_key(e.target) in keep]
    formulas = [f for f in parsed.formulas if f.get("doc") in visible]
    summaries = [s for s in parsed.summaries if s.get("doc") in visible]
    return DocumentGraph(nodes=nodes, edges=edges, formulas=formulas, summaries=summaries)


def build_document_graph(
    graph_json: Mapping[str, Any] | DocumentGraph | None,
    *,
    focus: str | None = None,
    hops: int = 1,
    theme: str = "light",
    title: str = "Document knowledge graph",
) -> GraphSpec:
    """Build the entity/relation graph of one or more documents.

    ``focus`` keeps only the entity and its neighbourhood (``hops`` edges away). Edge endpoints are
    resolved to node ids case-insensitively; unresolved endpoints become ``Inferred`` nodes.
    """
    kind = GraphKind.DOCUMENT
    graph = DocumentGraph.parse(graph_json)
    nodes: list[NodeSpec] = []
    seen_ids: set[str] = set()
    for node in graph.nodes:
        if node.id in seen_ids:
            continue
        seen_ids.add(node.id)
        category = normalize_entity_type(node.type)
        nodes.append(
            NodeSpec(
                id=node.id, label=node.id, category=category, style=node_style_for(kind, category, theme),
                tooltip=node.description,
                data={
                    "type": node.type or category,
                    "description": node.description,
                    "citations": [c.model_dump() for c in node.citations],
                    "source_text": node.source_text,
                    "page_reference": node.page_reference,
                    "inferred": False,
                },
            )
        )
    aliases = build_alias_map(seen_ids)
    edges: dict[str, EdgeSpec] = {}
    inferred: dict[str, NodeSpec] = {}
    for edge in graph.edges:
        source = resolve_endpoint(edge.source, aliases)
        target = resolve_endpoint(edge.target, aliases)
        for endpoint in (source, target):
            if endpoint not in seen_ids and endpoint not in inferred:
                inferred[endpoint] = NodeSpec(
                    id=endpoint, label=endpoint, category=INFERRED, style=node_style_for(kind, INFERRED, theme),
                    tooltip="Mentioned only in a relation; no entity description was extracted.",
                    data={"type": INFERRED, "description": "", "citations": [], "inferred": True},
                )
        if source == target:
            continue
        relation = edge.relation or DEFAULT_RELATION
        eid = edge_id(f"rel:{re.sub(r'[^a-z0-9]+', '_', relation.lower()).strip('_') or 'relates_to'}", source, target)
        if eid in edges:
            continue
        edges[eid] = EdgeSpec(
            id=eid, source=source, target=target, label=relation, category=relation, directed=True,
            style=edge_style_for(kind, "relation", theme), tooltip=edge.description or relation,
            data={"relation": relation, "description": edge.description},
        )
    all_nodes = tuple(nodes) + tuple(inferred[k] for k in sorted(inferred))
    spec = GraphSpec(
        kind=kind, title=title, nodes=all_nodes, edges=tuple(edges.values()), directed=True,
        layout=LayoutHint.HIERARCHICAL, legend=legend_entries(kind, theme),
        meta={"n_entities": len(nodes), "n_inferred": len(inferred), "n_relations": len(edges), "focus": focus},
    )
    if focus:
        focus_id = resolve_endpoint(focus, aliases)
        if spec.node(focus_id) is not None:
            spec = spec.neighborhood(focus_id, hops=max(1, int(hops)))
    return spec


__all__ = [
    "DEFAULT_RELATION",
    "ENTITY_TYPES",
    "INFERRED",
    "Citation",
    "DocEdge",
    "DocNode",
    "DocumentGraph",
    "build_alias_map",
    "build_document_graph",
    "canonical_key",
    "filter_by_documents",
    "merge_document_graphs",
    "normalize_entity_type",
    "resolve_endpoint",
]
