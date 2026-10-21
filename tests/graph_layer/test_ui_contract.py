"""Renderer contract: every registered backend renders a small spec headlessly without raising,
and the Streamlit entry point isolates failures and falls back to the offline engine."""

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from graph.builders.correlation import build_correlation_graph
from graph.model import GraphSpec, RenderOptions
from graph.renderers import GRAPH_RENDERERS
from graph.renderers.graphviz import GraphvizGraphRenderer, to_dot


def _corr_spec() -> GraphSpec:
    labels = ["a", "b", "c"]
    matrix = pd.DataFrame(
        [[1, 0.8, -0.5], [0.8, 1, 0.1], [-0.5, 0.1, 1]], index=labels, columns=labels
    )
    return build_correlation_graph(matrix, min_strength="Weak")


def _render_with(spec_json: str, rtype: str) -> None:
    """Page body executed by AppTest (must be self-contained: imports inside)."""
    import streamlit as st

    from graph.manager import GraphRendererManager
    from graph.model import GraphSpec, RenderOptions

    spec = GraphSpec.from_json(spec_json)
    renderer = GraphRendererManager().get(rtype)
    result = renderer.render(
        spec, RenderOptions(key="t", selectable=True, heat=True, theme="light")
    )
    st.write(f"rendered-by:{result.renderer}")


def _render_with_fallback(spec_json: str) -> None:
    import streamlit as st

    from graph.base import BaseGraphRenderer, RendererCapabilities, RenderError
    from graph.manager import GraphConfig, get_graph_renderer_manager
    from graph.model import GraphSpec, RenderOptions, SelectionResult
    from ui.graph_view import render_graph

    class Exploding(BaseGraphRenderer):
        name = "exploding"
        display_name = "Exploding"
        capabilities = RendererCapabilities(
            supports_selection=False,
            supports_edge_selection=False,
            supports_heat=False,
            supports_node_shapes=False,
            supports_edge_labels=False,
            supports_mixed_direction=False,
            supports_hierarchical=False,
            supports_drag=False,
            supports_html_export=False,
            requires_internet=False,
            telemetry=False,
            license_note="test",
        )

        def translate(self, spec, options):
            raise RenderError("boom")

        def render(self, spec, options) -> SelectionResult:
            raise RenderError("boom")

    manager = get_graph_renderer_manager(GraphConfig())
    manager.set_renderer(Exploding())
    result = render_graph(GraphSpec.from_json(spec_json), RenderOptions(key="t"))
    st.write(f"rendered-by:{result.renderer}")
    st.write("after-graph")
    get_graph_renderer_manager(GraphConfig())  # reset the global manager


def _render_empty() -> None:
    import streamlit as st

    from graph.model import GraphKind, GraphSpec, RenderOptions
    from ui.graph_view import render_graph

    result = render_graph(GraphSpec(kind=GraphKind.CORRELATION), RenderOptions(key="t"))
    st.write(f"empty:{result.is_empty}")


@pytest.mark.parametrize("name", GRAPH_RENDERERS.names())
def test_each_available_renderer_renders_headlessly(name: str) -> None:
    found = GRAPH_RENDERERS.availability(name)
    if not found.ok:
        pytest.skip(f"{name} not available: {found.reason}")
    at = AppTest.from_function(
        _render_with,
        kwargs={"spec_json": _corr_spec().to_json(), "rtype": name},
        default_timeout=60,
    ).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any(f"rendered-by:{name}" in m.value for m in at.markdown)


def test_graphviz_translation_is_deterministic_dot() -> None:
    spec = _corr_spec()
    dot = to_dot(spec, RenderOptions(key="t"))
    assert dot.startswith("graph G {") and '"a" -- "b"' in dot and "penwidth" in dot
    assert dot == to_dot(spec, RenderOptions(key="t"))
    assert GraphvizGraphRenderer().capabilities.supports_selection is False


def test_render_graph_isolates_failures_and_falls_back() -> None:
    at = AppTest.from_function(
        _render_with_fallback, kwargs={"spec_json": _corr_spec().to_json()}, default_timeout=60
    ).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("could not render" in e.value for e in at.error)
    assert any("Falling back" in i.value for i in at.info)
    texts = [m.value for m in at.markdown]
    assert any("rendered-by:plotly" in t for t in texts) and any("after-graph" in t for t in texts)


def test_render_graph_handles_empty_spec() -> None:
    at = AppTest.from_function(_render_empty, default_timeout=60).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("Nothing to draw" in i.value for i in at.info)
