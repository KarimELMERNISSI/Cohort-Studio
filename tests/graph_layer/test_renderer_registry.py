"""The graph engines as a connector family (M5b): the registry keeps the connector contract, and the manager adds the
offline mode."""

import subprocess
import sys
from pathlib import Path

import pytest

from core.registry import Availability
from graph.manager import GraphConfig, GraphRendererManager
from graph.renderers import GRAPH_RENDERERS, OFFLINE_DEFAULT
from tests import connector_contract as contract

ROOT = Path(__file__).resolve().parents[2]


def test_the_stored_names_never_change() -> None:
    # kept in the users' preferences and in CMP_GRAPH_RENDERER
    contract.check_names(GRAPH_RENDERERS, ("plotly", "yfiles", "graphviz"))
    assert GRAPH_RENDERERS.valid() == ("auto", "plotly", "yfiles", "graphviz")


def test_the_registry_keeps_the_connector_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    contract.check_availability(GRAPH_RENDERERS, monkeypatch)
    contract.check_classes(GRAPH_RENDERERS)
    contract.check_precedence(GRAPH_RENDERERS)
    contract.check_unknown_setting(GRAPH_RENDERERS)


def test_a_missing_yfiles_names_its_group(monkeypatch: pytest.MonkeyPatch) -> None:
    contract.check_missing_package(
        GRAPH_RENDERERS, "yfiles", "yfiles_graphs_for_streamlit", monkeypatch
    )
    assert GRAPH_RENDERERS.availability("yfiles").extra == "graph-yfiles"


def test_the_automatic_order_is_offline_first() -> None:
    def only(*names: str):
        return lambda name: Availability(name in names, "a test")

    assert OFFLINE_DEFAULT == "plotly"
    assert GRAPH_RENDERERS.select(check=only("plotly", "yfiles", "graphviz")).chosen == "plotly"
    assert GRAPH_RENDERERS.select(check=only("yfiles", "graphviz")).chosen == "graphviz"
    assert GRAPH_RENDERERS.select(check=only("yfiles")).chosen is None  # yFiles only when chosen


def test_checking_the_engines_loads_neither_streamlit_nor_the_yfiles_component() -> None:
    """The yFiles component imports Streamlit: its check finds it without importing it (R3)."""
    code = (
        "import sys\n"
        "from graph.manager import GraphConfig, GraphRendererManager\n"
        "manager = GraphRendererManager(GraphConfig())\n"
        "[manager.availability(name) for name in ('plotly', 'yfiles', 'graphviz')]\n"
        "print(sorted(m for m in ('streamlit', 'yfiles_graphs_for_streamlit') if m in sys.modules))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=120
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "[]"


def test_the_offline_mode_refuses_an_engine_that_needs_internet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = GraphRendererManager(GraphConfig(renderer="yfiles", offline=True))
    monkeypatch.setattr(GRAPH_RENDERERS, "availability", lambda name: Availability(True))
    refused = manager.availability("yfiles")
    assert not refused.ok and "offline mode (CMP_OFFLINE=1)" in refused.reason
    chosen = manager.select()
    assert chosen.chosen == "plotly" and "`yfiles` is not available here" in chosen.warnings[0]
    assert manager.availability("graphviz").ok


def test_the_fallback_is_the_offline_engine_never_the_one_that_failed() -> None:
    manager = GraphRendererManager(GraphConfig())
    fallback = manager.fallback("yfiles")
    assert fallback is not None and fallback.name == OFFLINE_DEFAULT
    assert manager.fallback(OFFLINE_DEFAULT) is None


def test_a_configuration_names_a_registered_engine() -> None:
    assert GraphConfig(renderer=" YFiles ").renderer == "yfiles"
    with pytest.raises(ValueError, match="use auto, plotly, yfiles or graphviz"):
        GraphConfig(renderer="nonsense")
