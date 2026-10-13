"""Renderer selection policy: offline Plotly by default, yFiles opt-in, safe fallbacks."""

import pytest

from graph.base import BaseGraphRenderer, RendererCapabilities
from graph.manager import (
    GraphConfig,
    GraphRendererManager,
    GraphRendererType,
    get_graph_renderer_manager,
)
from graph.model import GraphSpec, RenderOptions, SelectionResult


class _FakeRenderer(BaseGraphRenderer):
    name = "fake"
    display_name = "Fake"

    @property
    def capabilities(self) -> RendererCapabilities:
        return RendererCapabilities(
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

    def translate(self, spec: GraphSpec, options: RenderOptions):
        return {"nodes": len(spec.nodes)}

    def render(self, spec: GraphSpec, options: RenderOptions) -> SelectionResult:
        return SelectionResult(renderer=self.name)


def test_auto_resolves_to_the_offline_plotly_renderer() -> None:
    manager = GraphRendererManager(GraphConfig(renderer=GraphRendererType.AUTO))
    assert manager.renderer.name == "plotly"
    assert not manager.renderer.capabilities.requires_internet


def test_offline_mode_never_picks_an_internet_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = GraphRendererManager(GraphConfig(renderer=GraphRendererType.YFILES, offline=True))
    assert not manager.renderer.capabilities.requires_internet
    for rtype, ok, reason in manager.available_renderers():
        if rtype == GraphRendererType.YFILES and ok:
            pytest.fail(f"yFiles reported available in offline mode: {reason}")


def test_environment_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CMP_GRAPH_RENDERER", "nonsense")
    monkeypatch.setenv("CMP_OFFLINE", "yes")
    manager = GraphRendererManager()
    assert manager.config.renderer == GraphRendererType.AUTO and manager.config.offline is True
    monkeypatch.setenv("CMP_GRAPH_RENDERER", "plotly")
    monkeypatch.delenv("CMP_OFFLINE")
    manager.reset_renderer()
    assert manager.config.renderer == GraphRendererType.PLOTLY and manager.renderer.name == "plotly"


def test_set_renderer_injection_and_get_override() -> None:
    manager = GraphRendererManager(GraphConfig())
    fake = _FakeRenderer()
    manager.set_renderer(fake)
    assert manager.renderer is fake
    assert manager.get(None) is fake
    assert manager.get("plotly").name == "plotly"


def test_global_accessor_singleton() -> None:
    first = get_graph_renderer_manager()
    assert get_graph_renderer_manager() is first
    replaced = get_graph_renderer_manager(GraphConfig(renderer=GraphRendererType.PLOTLY))
    assert replaced is not first and get_graph_renderer_manager() is replaced
