"""Offline verification of the yFiles adapter: translation is replayed through the library's own
mapping engine without a browser, and the installed library is checked for the API this adapter needs."""

import copy
import importlib.metadata

import pytest

from graph.base import RenderError
from graph.model import GraphSpec, LayoutHint, RenderOptions

yfiles = pytest.importorskip("yfiles_graphs_for_streamlit")

from graph.renderers import yfiles as adapter
from graph.renderers.yfiles import YFilesGraphRenderer


def _apply(payload):
    from yfiles_graphs_for_streamlit.apply_mappings import MappingClass

    widget = yfiles.StreamlitGraphWidget(
        nodes=payload.nodes, edges=payload.edges, **payload.mappings
    )
    mapper = MappingClass(widget, widget._data_importer, payload.show_kwargs["directed"])
    nodes, edges = mapper.apply_mappings(
        copy.deepcopy(widget._user_nodes),
        copy.deepcopy(widget._user_edges),
        widget._user_nodes,
        widget._user_edges,
    )
    to_dict = lambda item: item.to_dict() if hasattr(item, "to_dict") else dict(item)
    return [to_dict(n) for n in nodes], [to_dict(e) for e in edges], widget


def test_installed_library_matches_adapter_expectations() -> None:
    found = YFilesGraphRenderer.availability()
    assert found.ok, found.reason
    assert adapter._api_problems(yfiles) == []
    assert "hierarchical" in {m.value for m in yfiles.Layout}


def test_another_major_version_is_refused_with_its_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "3.0.1")
    found = YFilesGraphRenderer.availability()
    assert not found.ok and "3.0.1" in found.reason and "2.x" in found.reason
    assert found.install_hint == "uv sync --extra graph-yfiles"


def test_an_api_mismatch_is_a_render_error_so_the_page_falls_back(
    monkeypatch: pytest.MonkeyPatch, taxonomy_spec: GraphSpec
) -> None:
    monkeypatch.setattr(adapter, "_api_problems", lambda y: ["show() lacks 'key'"])
    with pytest.raises(RenderError, match=r"does not match this adapter: show\(\) lacks 'key'"):
        YFilesGraphRenderer().render(taxonomy_spec, RenderOptions(key="k"))


def test_capabilities_declare_internet_and_telemetry() -> None:
    caps = YFilesGraphRenderer.capabilities  # read from the class, without drawing
    assert caps.requires_internet and caps.telemetry and caps.supports_drag


def test_translate_taxonomy_graph(taxonomy_spec: GraphSpec) -> None:
    renderer = YFilesGraphRenderer()
    payload = renderer.translate(
        taxonomy_spec, RenderOptions(key="kg", height=750, selectable=True, theme="light")
    )

    assert payload.show_kwargs["graph_layout"] == yfiles.Layout.HIERARCHICAL
    assert (
        payload.show_kwargs["directed"] is False
    )  # mixed graph: arrows come from the per-edge mapping
    assert payload.show_kwargs["sync_selection"] is True
    assert payload.show_kwargs["height"] == 750 and payload.show_kwargs["key"] == "kg__yfiles"
    assert [n["id"] for n in payload.nodes] == list(taxonomy_spec.node_ids())
    assert [e["id"] for e in payload.edges] == [e.id for e in taxonomy_spec.edges]

    nodes, edges, widget = _apply(payload)
    assert getattr(widget, "_error", None) is None

    formula = next(n for n in nodes if n["id"] == "FORM_bmi")
    assert formula["label"] == "BMI formula"
    assert formula["styles"]["shape"] == yfiles.NodeShape.HEXAGON.value
    assert formula["styles"]["label_styles"]["position"] == "center"
    assert formula["styles"]["color"].lower() == taxonomy_spec.node("FORM_bmi").style.color.lower()

    member = next(e for e in edges if e["id"].startswith("category_member"))
    yields = next(e for e in edges if e["id"].startswith("yields"))
    assert member["directed"] is False and yields["directed"] is True
    assert member["styles"]["dashStyle"] == yfiles.DashStyle.DASH.value
    assert "directed" not in member["styles"] and yields["styles"]["directed"] is True
    assert yields["label"] == "yields"


def test_translate_heat_and_layout_override(taxonomy_spec: GraphSpec) -> None:
    spec = taxonomy_spec.model_copy(
        update={"nodes": tuple(n.model_copy(update={"weight": 0.5}) for n in taxonomy_spec.nodes)}
    )
    payload = YFilesGraphRenderer().translate(
        spec, RenderOptions(key="k", heat=True, layout=LayoutHint.CIRCULAR, show_edge_labels=False)
    )
    assert payload.show_kwargs["graph_layout"] == yfiles.Layout.CIRCULAR
    nodes, edges, widget = _apply(payload)
    assert getattr(widget, "_error", None) is None
    assert all(n["heat"] == 0.5 for n in nodes)
    assert all(e["label"] == "" for e in edges)
