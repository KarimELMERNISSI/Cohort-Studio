"""
Graph renderer manager.

Chooses the rendering backend through the family's registry (:data:`graph.renderers.GRAPH_RENDERERS`): the session's
choice, then the user's saved preference, then the configuration (environment variables ``CMP_GRAPH_RENDERER`` and
``CMP_OFFLINE``, or an explicit :class:`GraphConfig`), then the offline-first order. It instantiates the renderer
lazily and exposes ``set_renderer`` / ``reset_renderer`` for tests and custom backends. The structure mirrors
``manage.iam.manager.IAMManager``.

Default policy (owner decision, October 2026): the offline Plotly/NetworkX renderer is the default;
the proprietary yFiles component is opt-in (``CMP_GRAPH_RENDERER=yfiles`` or the in-app selector)
because its free license loads a runtime from yworks.com and reports anonymised usage data.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.registry import AUTO, Availability, Selection
from graph.base import BaseGraphRenderer
from graph.renderers import GRAPH_RENDERERS, OFFLINE_DEFAULT
from utils.settings import Settings, get_settings

logger = logging.getLogger(__name__)

OFFLINE = Availability(False, "disabled in offline mode (CMP_OFFLINE=1)")


class GraphConfig(BaseModel):
    # a refused name is not repeated in the error: it can come from a setting, where it can be a pasted key
    model_config = ConfigDict(hide_input_in_errors=True)

    renderer: str = Field(default=AUTO, description="A registered engine, or auto (offline first).")
    offline: bool = Field(
        default=False, description="Never choose a renderer that needs internet access."
    )
    fallback_on_error: bool = Field(
        default=True, description="Fall back to the offline renderer when rendering fails."
    )
    show_license_notice: bool = Field(
        default=True, description="Show the third-party notice for proprietary renderers."
    )

    @field_validator("renderer")
    @classmethod
    def _registered(cls, value: str) -> str:
        if not value or not value.strip():
            return AUTO  # unset, as in the settings
        name = GRAPH_RENDERERS.canonical(value)
        if name is None:
            raise ValueError(GRAPH_RENDERERS.setting_problem(value))
        return name


class GraphRendererManager:
    """Central facade selecting and caching the active graph renderer."""

    def __init__(self, config: GraphConfig | None = None) -> None:
        self._config = config or self._resolve_config_from_env()
        self._renderer: BaseGraphRenderer | None = None

    # ----------------------------------------------------------------- configuration
    @classmethod
    def _resolve_config_from_env(cls, settings: Settings | None = None) -> GraphConfig:
        """The renderer's configuration (utils.settings; the current settings by default)."""
        graph = (settings or get_settings()).graph
        problem = GRAPH_RENDERERS.setting_problem(graph.renderer)
        if problem:  # never the value: it can be a pasted key or address
            logger.warning("%s; the automatic engine is used", problem)
        return GraphConfig(
            renderer=AUTO if problem else graph.renderer or AUTO,
            offline=graph.offline,
            fallback_on_error=not graph.no_fallback,
        )

    @property
    def config(self) -> GraphConfig:
        return self._config

    # ----------------------------------------------------------------- choice
    def availability(self, name: str) -> Availability:
        """Whether the engine ``name`` can draw here: allowed by the offline mode, installed, and compatible. The
        offline mode is asked first: installing an engine that needs internet access would not make it usable."""
        if self._config.offline:
            try:  # an engine's module loads without its library (R3): its needs are known all the same
                online = GRAPH_RENDERERS.get(name).capabilities.requires_internet
            except Exception:  # not loadable here: the registry says why
                online = False
            if online:
                return OFFLINE
        return GRAPH_RENDERERS.availability(name)

    def select(self, *, session: str | None = None, preference: str | None = None) -> Selection:
        """The engine to use: the session's choice, the saved preference, the configuration, then offline first."""
        return GRAPH_RENDERERS.select(
            session=session,
            preference=preference,
            setting=self._config.renderer,
            check=self.availability,
        )

    def _build(self, selection: Selection) -> BaseGraphRenderer:
        for warning in selection.warnings:
            logger.warning("%s", warning)
        if selection.chosen is None:
            raise RuntimeError(
                "No graph renderer is available (plotly and networkx are required for the default renderer)."
            )
        return GRAPH_RENDERERS.get(selection.chosen)()

    # ----------------------------------------------------------------- access
    @property
    def renderer(self) -> BaseGraphRenderer:
        if self._renderer is None:
            self._renderer = self._build(self.select())
        return self._renderer

    def get(self, override: str | None = None) -> BaseGraphRenderer:
        """The configured renderer, or the session's choice ``override`` when it can draw here (``auto``: no choice,
        the configured one)."""
        name = GRAPH_RENDERERS.canonical(override)
        if not override or name == AUTO:
            return self.renderer
        if self._renderer is not None and self._renderer.name == name:
            return self._renderer  # the cached one: nothing to choose again
        selection = self.select(session=override)
        if self._renderer is not None and self._renderer.name == selection.chosen:
            return self._renderer
        return self._build(selection)

    def fallback(self, failed: str) -> BaseGraphRenderer | None:
        """The built-in offline engine, for a drawing that failed (None when it is the one that failed, or cannot
        draw here)."""
        if failed == OFFLINE_DEFAULT or not self.availability(OFFLINE_DEFAULT).ok:
            return None
        return GRAPH_RENDERERS.get(OFFLINE_DEFAULT)()

    def set_renderer(self, renderer: BaseGraphRenderer) -> None:
        """Inject a custom or mock renderer."""
        self._renderer = renderer
        logger.info("Swapped active graph renderer to %s", renderer.name)

    def reset_renderer(self) -> None:
        """Re-read the environment configuration and drop the cached renderer."""
        self._config = self._resolve_config_from_env()
        self._renderer = None


_GLOBAL_MANAGER: GraphRendererManager | None = None


def get_graph_renderer_manager(config: GraphConfig | None = None) -> GraphRendererManager:
    """Retrieve or initialise the global manager (a new config replaces the global instance)."""
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None or config is not None:
        _GLOBAL_MANAGER = GraphRendererManager(config=config)
    return _GLOBAL_MANAGER


__all__ = ["GraphConfig", "GraphRendererManager", "get_graph_renderer_manager"]
