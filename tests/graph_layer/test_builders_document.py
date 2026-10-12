"""Document knowledge graph: tolerant parsing, cross-document merge with edge remapping, focus."""

from graph.builders.document import (
    DocumentGraph,
    build_document_graph,
    filter_by_documents,
    merge_document_graphs,
    normalize_entity_type,
)
from graph.model import GraphKind

DOC_A = {
    "nodes": [
        {"id": "Heart failure", "type": "Disease", "description": "Reduced ejection fraction", "source_text": "...", "page_reference": 3},
        {"id": "LVEF", "type": "Metric (echo)", "description": "Left ventricular ejection fraction", "page_reference": [3, 4]},
        {"id": "", "type": "Concept"},
    ],
    "edges": [
        {"source": "LVEF", "target": "heart failure", "relation": "diagnoses"},
        {"source": "LVEF", "target": "Beta blockers", "relation": "improved by"},
        {"source": "LVEF", "target": "LVEF", "relation": "self"},
    ],
    "formulas": [{"name": "LVEF", "expression": "(EDV-ESV)/EDV"}],
    "summary": {"title": "Doc A"},
}
DOC_B = {
    "nodes": [
        {"id": "heart Failure", "type": "disease", "description": "Reduced ejection fraction with symptoms and signs"},
        {"id": "Beta blockers", "type": "Treatment", "description": "Standard therapy"},
    ],
    "edges": [
        {"source": "Beta blockers", "target": "Heart Failure", "relation": "treats"},
        {"source": "beta blockers", "target": "HEART FAILURE", "relation": "Treats"},
    ],
}


def test_parse_is_tolerant() -> None:
    graph = DocumentGraph.parse(DOC_A)
    assert [n.id for n in graph.nodes] == ["Heart failure", "LVEF"]  # empty id skipped
    assert graph.nodes[0].page_reference == "3" and graph.nodes[1].page_reference == "3, 4"
    assert len(graph.edges) == 3 and graph.formulas
    assert DocumentGraph.parse(None).nodes == [] and DocumentGraph.parse({"nodes": "junk"}).nodes == []


def test_entity_type_normalisation() -> None:
    assert normalize_entity_type("Metric (echo)") == "Metric"
    assert normalize_entity_type("disease") == "Disease"
    assert normalize_entity_type("weird") == "Other" and normalize_entity_type(None) == "Other"
    assert normalize_entity_type("Inferred") == "Inferred"


def test_merge_unifies_entities_and_remaps_edges() -> None:
    merged = merge_document_graphs([("a.pdf", DOC_A), ("b.pdf", DOC_B)])
    ids = [n.id for n in merged.nodes]
    assert ids == ["Heart failure", "LVEF", "Beta blockers"]
    hf = merged.nodes[0]
    assert [c.doc for c in hf.citations] == ["a.pdf", "b.pdf"]
    assert hf.citations[0].page == "3"
    assert hf.description == "Reduced ejection fraction with symptoms and signs"  # longer wins
    relations = sorted((e.source, e.target, e.relation) for e in merged.edges)
    assert relations == [
        ("Beta blockers", "Heart failure", "treats"),
        ("LVEF", "Beta blockers", "improved by"),
        ("LVEF", "Heart failure", "diagnoses"),
    ]
    assert [s["doc"] for s in merged.summaries] == ["a.pdf"]
    assert merged.formulas[0]["doc"] == "a.pdf"


def test_build_from_merged_graph_has_no_spurious_inferred_nodes() -> None:
    merged = merge_document_graphs([("a.pdf", DOC_A), ("b.pdf", DOC_B)])
    spec = build_document_graph(merged)
    assert spec.kind == GraphKind.DOCUMENT and spec.directed
    assert spec.meta["n_inferred"] == 0
    assert spec.node_ids() == ("Heart failure", "LVEF", "Beta blockers")
    assert spec.node("Heart failure").category == "Disease" and spec.node("LVEF").category == "Metric"
    assert spec.node("Heart failure").data["citations"][1]["doc"] == "b.pdf"
    assert sorted(e.category for e in spec.edges) == ["diagnoses", "improved by", "treats"]
    assert all(e.directed for e in spec.edges)


def test_dangling_endpoints_become_flagged_inferred_nodes() -> None:
    spec = build_document_graph({"nodes": [{"id": "A", "type": "Concept"}],
                                 "edges": [{"source": "a", "target": "Zeta", "relation": "relates to"}]})
    assert spec.node_ids() == ("A", "Zeta")
    zeta = spec.node("Zeta")
    assert zeta.category == "Inferred" and zeta.data["inferred"] is True
    assert spec.edges[0].source == "A"  # case-insensitive endpoint resolution


def test_filter_by_documents_and_focus() -> None:
    merged = merge_document_graphs([("a.pdf", DOC_A), ("b.pdf", DOC_B)])
    only_b = filter_by_documents(merged.to_json_dict(), ["b.pdf"])
    assert [n.id for n in only_b.nodes] == ["Heart failure", "Beta blockers"]
    assert [(e.source, e.target) for e in only_b.edges] == [("Beta blockers", "Heart failure")]
    assert filter_by_documents(merged, []).nodes == []

    focused = build_document_graph(merged, focus="lvef", hops=1)
    assert set(focused.node_ids()) == {"LVEF", "Heart failure", "Beta blockers"}
    focused_hf = build_document_graph(merged, focus="Heart failure", hops=1)
    assert set(focused_hf.node_ids()) == {"Heart failure", "LVEF", "Beta blockers"}
    assert build_document_graph(merged, focus="unknown").node_ids() == ("Heart failure", "LVEF", "Beta blockers")


def test_builder_is_deterministic() -> None:
    merged = merge_document_graphs([("a.pdf", DOC_A), ("b.pdf", DOC_B)])
    assert build_document_graph(merged).fingerprint() == build_document_graph(merged.to_json_dict()).fingerprint()
