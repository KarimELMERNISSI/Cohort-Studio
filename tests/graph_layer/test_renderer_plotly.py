"""The offline Plotly/NetworkX renderer translates a spec into an inspectable figure."""

from types import SimpleNamespace

import pandas as pd

from graph.builders.correlation import build_correlation_graph
from graph.model import GraphSpec, LayoutHint, RenderOptions
from graph.renderers.plotly import PlotlyGraphRenderer, compute_positions


def _renderer() -> PlotlyGraphRenderer:
    found = PlotlyGraphRenderer.availability()
    assert found.ok, found.reason
    return PlotlyGraphRenderer()


def test_capabilities_are_offline() -> None:
    caps = _renderer().capabilities
    assert not caps.requires_internet and not caps.telemetry
    assert caps.supports_selection and caps.supports_html_export


def test_translate_correlation_graph(corr_matrix: pd.DataFrame) -> None:
    spec = build_correlation_graph(corr_matrix, min_strength="Weak")
    payload = _renderer().translate(spec, RenderOptions(key="k", heat=True, theme="light"))
    fig = payload.figure
    node_trace = fig.data[payload.node_curve]
    assert list(node_trace.customdata) == list(spec.node_ids())
    assert list(node_trace.text) == ["a", "b", "c", "d"]
    edge_trace = fig.data[payload.edge_curve]
    assert list(edge_trace.customdata) == [e.id for e in spec.edges]
    assert list(edge_trace.text) == ["0.95", "-0.50", "0.20"]
    assert len(fig.layout.annotations) == 0  # undirected: no arrow heads
    assert fig.layout.height == 500
    assert node_trace.marker.showscale is True  # heat colour scale


def test_translate_directed_graph_adds_arrows_and_layers(taxonomy_spec: GraphSpec) -> None:
    payload = _renderer().translate(taxonomy_spec, RenderOptions(key="k", theme="dark"))
    directed = sum(1 for e in taxonomy_spec.edges if e.directed)
    assert len(payload.figure.layout.annotations) == directed
    assert payload.layout == LayoutHint.HIERARCHICAL
    pos = payload.positions
    assert pos["CAT_vitals"][1] > pos["height"][1] > pos["FORM_bmi"][1] > pos["bmi"][1]
    assert payload.figure.layout.template.layout.paper_bgcolor is not None


def test_positions_are_deterministic(corr_matrix: pd.DataFrame) -> None:
    spec = build_correlation_graph(corr_matrix)
    first = compute_positions(spec, LayoutHint.ORGANIC)
    second = compute_positions(spec, LayoutHint.ORGANIC)
    assert first == second
    assert set(first) == set(spec.node_ids())
    assert set(compute_positions(spec, LayoutHint.CIRCULAR)) == set(spec.node_ids())
    assert set(compute_positions(spec, LayoutHint.RADIAL, focus="a")) == set(spec.node_ids())


def test_single_node_and_empty_specs(taxonomy_spec: GraphSpec) -> None:
    one = taxonomy_spec.subgraph(["bmi"])
    assert compute_positions(one, LayoutHint.ORGANIC) == {"bmi": (0.0, 0.0)}
    empty = taxonomy_spec.subgraph([])
    payload = _renderer().translate(empty, RenderOptions(key="k"))
    assert payload.node_curve is None and payload.edge_curve is None


def test_export_html_contains_labels(taxonomy_spec: GraphSpec) -> None:
    html = _renderer().export_html(taxonomy_spec, RenderOptions(key="k", theme="light"))
    assert html is not None and b"BMI formula" in html and b"<html" in html.lower()


def test_selection_from_event(corr_matrix: pd.DataFrame) -> None:
    renderer = _renderer()
    spec = build_correlation_graph(corr_matrix, min_strength="Weak")
    payload = renderer.translate(spec, RenderOptions(key="k", selectable=True, theme="light"))
    event = SimpleNamespace(
        selection=SimpleNamespace(
            points=[
                {"curve_number": payload.node_curve, "point_index": 0, "customdata": "a"},
                {"curve_number": payload.node_curve, "point_index": 1, "customdata": ["b"]},
                {"curve_number": payload.edge_curve, "point_index": 0, "customdata": "corr:a|b"},
                {"curve_number": 0, "point_index": 0},
            ]
        )
    )
    result = renderer._selection_from_event(event, payload)
    assert (
        result.node_ids == ("a", "b")
        and result.edge_ids == ("corr:a|b",)
        and result.renderer == "plotly"
    )
