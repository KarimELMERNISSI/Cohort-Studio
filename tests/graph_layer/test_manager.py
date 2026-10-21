"""Renderer selection policy: offline Plotly by default, yFiles opt-in, safe fallbacks."""

import logging

import pytest

from graph.base import BaseGraphRenderer, RendererCapabilities
from graph.manager import (
    GraphConfig,
    GraphRendererManager,
    get_graph_renderer_manager,
)
from graph.model import GraphSpec, RenderOptions, SelectionResult


class _FakeRenderer(BaseGraphRenderer):
    name = "fake"
    display_name = "Fake"
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

    def translate(self, spec: GraphSpec, options: RenderOptions):
        return {"nodes": len(spec.nodes)}

    def render(self, spec: GraphSpec, options: RenderOptions) -> SelectionResult:
        return SelectionResult(renderer=self.name)


def test_auto_resolves_to_the_offline_plotly_renderer() -> None:
    manager = GraphRendererManager(GraphConfig(renderer="auto"))
    assert manager.renderer.name == "plotly"
    assert not manager.renderer.capabilities.requires_internet


def test_offline_mode_never_picks_an_internet_renderer() -> None:
    manager = GraphRendererManager(GraphConfig(renderer="yfiles", offline=True))
    assert not manager.renderer.capabilities.requires_internet
    assert not manager.availability("yfiles").ok
    assert all(found.ok for name, found in manager.select().alternatives if name != "yfiles")


def test_environment_configuration(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # until start-up refuses it (M5b), a name that is not an engine gives the automatic one, and the log never shows
    # the value (it can be a pasted key or address)
    monkeypatch.setenv("CMP_GRAPH_RENDERER", "zz9-nonsense")
    monkeypatch.setenv("CMP_OFFLINE", "yes")
    with caplog.at_level(logging.WARNING, logger="graph.manager"):
        manager = GraphRendererManager()
    assert manager.config.renderer == "auto" and manager.config.offline is True
    assert "CMP_GRAPH_RENDERER is not a graph engine" in caplog.text and "zz9" not in caplog.text
    monkeypatch.setenv("CMP_GRAPH_RENDERER", "Plotly")
    monkeypatch.delenv("CMP_OFFLINE")
    manager.reset_renderer()
    assert manager.config.renderer == "plotly" and manager.renderer.name == "plotly"


def test_the_choice_follows_the_session_the_preference_then_the_configuration() -> None:
    manager = GraphRendererManager(GraphConfig(renderer="graphviz"))
    assert (manager.select().chosen, manager.select().source) == ("graphviz", "setting")
    assert manager.select(preference="plotly").chosen == "plotly"
    chosen = manager.select(session="graphviz", preference="plotly")
    assert (chosen.chosen, chosen.source) == ("graphviz", "session")
    stale = manager.select(session="cohere")
    assert stale.chosen == "graphviz" and "`cohere` is not offered here" in stale.warnings[0]
    # the session's "auto" leaves the configuration in charge, in the manager and in the kit alike
    assert manager.get("auto").name == manager.select(session="auto").chosen == "graphviz"


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
    replaced = get_graph_renderer_manager(GraphConfig(renderer="plotly"))
    assert replaced is not first and get_graph_renderer_manager() is replaced
