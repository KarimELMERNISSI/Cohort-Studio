"""Integrity, serialisation and query behaviour of the renderer-agnostic graph model."""

import pytest
from pydantic import ValidationError

from graph.model import (
    EdgeSpec,
    GraphKind,
    GraphSpec,
    NodeSpec,
    RenderOptions,
    SelectionResult,
    edge_id,
)


def _spec(**overrides) -> GraphSpec:
    nodes = (
        NodeSpec(id="a", label="A", category="x"),
        NodeSpec(id="b", label="B", category="y"),
        NodeSpec(id="c", label="C", category="x"),
        NodeSpec(id="lonely", label="L", category="y"),
    )
    edges = (
        EdgeSpec(id="e:a->b", source="a", target="b", category="rel"),
        EdgeSpec(id="e:b->c", source="b", target="c", category="other", directed=False),
    )
    base = dict(kind=GraphKind.TAXONOMY, nodes=nodes, edges=edges)
    base.update(overrides)
    return GraphSpec(**base)


class TestIntegrity:
    def test_duplicate_node_ids_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate node ids"):
            GraphSpec(kind=GraphKind.TAXONOMY, nodes=(NodeSpec(id="a"), NodeSpec(id="a")))

    def test_duplicate_edge_ids_rejected(self) -> None:
        nodes = (NodeSpec(id="a"), NodeSpec(id="b"))
        edges = (EdgeSpec(id="e", source="a", target="b"), EdgeSpec(id="e", source="b", target="a"))
        with pytest.raises(ValidationError, match="duplicate edge ids"):
            GraphSpec(kind=GraphKind.TAXONOMY, nodes=nodes, edges=edges)

    def test_dangling_edge_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown nodes"):
            GraphSpec(
                kind=GraphKind.TAXONOMY,
                nodes=(NodeSpec(id="a"),),
                edges=(EdgeSpec(id="e", source="a", target="zz"),),
            )

    def test_non_json_payload_rejected(self) -> None:
        with pytest.raises(ValidationError, match="JSON-native"):
            GraphSpec(kind=GraphKind.TAXONOMY, nodes=(NodeSpec(id="a", data={"x": object()}),))

    def test_models_are_frozen(self) -> None:
        spec = _spec()
        with pytest.raises(ValidationError):
            spec.title = "changed"  # type: ignore[misc]


class TestSerialisation:
    def test_json_round_trip(self) -> None:
        spec = _spec(title="t", meta={"k": [1, 2]})
        assert GraphSpec.from_json(spec.to_json()) == spec

    def test_fingerprint_is_stable_and_content_sensitive(self) -> None:
        assert _spec().fingerprint() == _spec().fingerprint()
        assert _spec().fingerprint() != _spec(title="other").fingerprint()
        assert len(_spec().fingerprint()) == 16

    def test_edge_id_is_deterministic(self) -> None:
        assert edge_id("rel", "a", "b") == "rel:a->b"
        assert (
            edge_id("rel", "b", "a", undirected=True)
            == edge_id("rel", "a", "b", undirected=True)
            == "rel:a|b"
        )


class TestQueries:
    def test_node_lookup_and_order(self) -> None:
        spec = _spec()
        assert spec.node("b").label == "B"
        assert spec.node("missing") is None
        assert spec.node_ids() == ("a", "b", "c", "lonely")

    def test_neighbors_are_undirected(self) -> None:
        spec = _spec()
        assert spec.neighbors("b") == {"a", "c"}
        assert spec.neighbors("lonely") == set()

    def test_subgraph_preserves_order_and_drops_crossing_edges(self) -> None:
        sub = _spec().subgraph(["c", "a"])
        assert sub.node_ids() == ("a", "c")
        assert sub.edges == ()

    def test_neighborhood_and_component(self) -> None:
        spec = _spec()
        assert spec.neighborhood("a", hops=1).node_ids() == ("a", "b")
        assert spec.neighborhood("a", hops=2).node_ids() == ("a", "b", "c")
        assert spec.component_of("c").node_ids() == ("a", "b", "c")
        assert spec.component_of("nope").node_ids() == ()

    def test_filter_by_categories(self) -> None:
        spec = _spec()
        only_x = spec.filter(node_categories=["x"])
        assert only_x.node_ids() == ("a", "c")
        assert only_x.edges == ()
        rel_only = spec.filter(edge_categories=["rel"])
        assert [e.id for e in rel_only.edges] == ["e:a->b"]

    def test_stats(self) -> None:
        stats = _spec().stats()
        assert stats.node_count == 4 and stats.edge_count == 2
        assert stats.nodes_by_category == {"x": 2, "y": 2}
        assert stats.edges_by_category == {"other": 1, "rel": 1}
        assert stats.component_count == 2 and stats.isolated_nodes == 1

    def test_to_networkx(self) -> None:
        g = _spec().to_networkx()
        assert g.is_directed()
        assert g.has_edge("a", "b") and g.has_edge("b", "c") and g.has_edge("c", "b")
        assert not g.has_edge("b", "a")


class TestOptions:
    def test_render_options_defaults_and_bounds(self) -> None:
        opts = RenderOptions(key="k")
        assert opts.height == 500 and opts.theme == "auto" and not opts.selectable
        with pytest.raises(ValidationError):
            RenderOptions(key="k", height=50)

    def test_selection_result_empty(self) -> None:
        assert SelectionResult().is_empty
        assert not SelectionResult(node_ids=("a",)).is_empty
