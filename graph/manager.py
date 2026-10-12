"""
Graph renderer manager and registry.

Resolves which rendering backend to use from configuration (environment variables ``CMP_GRAPH_RENDERER``
and ``CMP_OFFLINE``, or an explicit :class:`GraphConfig`), instantiates it lazily and exposes
``set_renderer`` / ``reset_renderer`` for tests and custom backends. The structure mirrors
``manage.iam.manager.IAMManager``.

Default policy (owner decision, October 2026): the offline Plotly/NetworkX renderer is the default;
the proprietary yFiles component is opt-in (``CMP_GRAPH_RENDERER=yfiles`` or the in-app selector)
because its free license loads a runtime from yworks.com and reports anonymised usage data.
"""

from __future__ import annotations

import importlib
import logging
import os
from enum import StrEnum

from pydantic import BaseModel, Field

from graph.base import BaseGraphRenderer

logger = logging.getLogger(__name__)


class GraphRendererType(StrEnum):
    AUTO = "auto"
    PLOTLY = "plotly"
    YFILES = "yfiles"
    GRAPHVIZ = "graphviz"


class GraphConfig(BaseModel):
    renderer: GraphRendererType = GraphRendererType.AUTO
    offline: bool = Field(default=False, description="Never choose a renderer that needs internet access.")
    fallback_on_error: bool = Field(default=True, description="Fall back to the offline renderer when rendering fails.")
    show_license_notice: bool = Field(default=True, description="Show the third-party notice for proprietary renderers.")


_REGISTRY_SPEC: dict[GraphRendererType, tuple[str, str]] = {
    GraphRendererType.PLOTLY: ("graph.renderers.plotly", "PlotlyGraphRenderer"),
    GraphRendererType.YFILES: ("graph.renderers.yfiles", "YFilesGraphRenderer"),
    GraphRendererType.GRAPHVIZ: ("graph.renderers.graphviz", "GraphvizGraphRenderer"),
}

# Preference order when AUTO is requested or a fallback is needed (offline-first).
_AUTO_PREFERENCE: tuple[GraphRendererType, ...] = (
    GraphRendererType.PLOTLY,
    GraphRendererType.GRAPHVIZ,
)


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


class GraphRendererManager:
    """Central facade selecting and caching the active graph renderer."""

    def __init__(self, config: GraphConfig | None = None) -> None:
        self._config = config or self._resolve_config_from_env()
        self._renderer: BaseGraphRenderer | None = None
        self._classes: dict[GraphRendererType, type[BaseGraphRenderer]] = {}

    # ----------------------------------------------------------------- configuration
    @classmethod
    def _resolve_config_from_env(cls) -> GraphConfig:
        raw = os.getenv("CMP_GRAPH_RENDERER", "auto").strip().lower()
        try:
            renderer = GraphRendererType(raw)
        except ValueError:
            logger.warning("Unknown CMP_GRAPH_RENDERER=%r; using auto", raw)
            renderer = GraphRendererType.AUTO
        return GraphConfig(
            renderer=renderer,
            offline=_truthy(os.getenv("CMP_OFFLINE")),
            fallback_on_error=not _truthy(os.getenv("CMP_GRAPH_NO_FALLBACK")),
        )

    @property
    def config(self) -> GraphConfig:
        return self._config

    # ----------------------------------------------------------------- registry
    def registry(self) -> dict[GraphRendererType, type[BaseGraphRenderer]]:
        """Renderer classes, imported lazily so importing ``graph`` never imports a UI library."""
        for rtype, (module_name, class_name) in _REGISTRY_SPEC.items():
            if rtype in self._classes:
                continue
            try:
                module = importlib.import_module(module_name)
                self._classes[rtype] = getattr(module, class_name)
            except Exception as exc:  # missing optional dependency or broken backend
                logger.info("Graph renderer %s unavailable: %s", rtype.value, exc)
        return dict(self._classes)

    def available_renderers(self) -> list[tuple[GraphRendererType, bool, str]]:
        """``(type, available, reason)`` for every known renderer, honouring offline mode."""
        result: list[tuple[GraphRendererType, bool, str]] = []
        classes = self.registry()
        for rtype in _REGISTRY_SPEC:
            cls = classes.get(rtype)
            if cls is None:
                result.append((rtype, False, "backend not installed"))
                continue
            ok, reason = cls.availability()
            if ok and self._config.offline and cls.capabilities.fget(cls).requires_internet:  # type: ignore[union-attr]
                ok, reason = False, "disabled in offline mode (CMP_OFFLINE=1)"
            result.append((rtype, ok, reason))
        return result

    def _instantiate(self, rtype: GraphRendererType) -> BaseGraphRenderer | None:
        cls = self.registry().get(rtype)
        if cls is None:
            return None
        ok, reason = cls.availability()
        if not ok:
            logger.warning("Graph renderer %s requested but unavailable: %s", rtype.value, reason)
            return None
        instance = cls()
        if self._config.offline and instance.capabilities.requires_internet:
            logger.warning("Graph renderer %s needs internet access; refused in offline mode", rtype.value)
            return None
        return instance

    def _build_renderer(self, rtype: GraphRendererType) -> BaseGraphRenderer:
        candidates: list[GraphRendererType] = []
        if rtype != GraphRendererType.AUTO:
            candidates.append(rtype)
        candidates.extend(r for r in _AUTO_PREFERENCE if r not in candidates)
        for candidate in candidates:
            instance = self._instantiate(candidate)
            if instance is not None:
                if candidate != rtype and rtype != GraphRendererType.AUTO:
                    logger.warning("Falling back to graph renderer %s", candidate.value)
                return instance
        raise RuntimeError("No graph renderer is available (plotly and networkx are required for the default renderer).")

    # ----------------------------------------------------------------- access
    @property
    def renderer(self) -> BaseGraphRenderer:
        if self._renderer is None:
            self._renderer = self._build_renderer(self._config.renderer)
        return self._renderer

    def get(self, override: GraphRendererType | str | None = None) -> BaseGraphRenderer:
        """The configured renderer, or a specific one when ``override`` is given (with fallback)."""
        if override is None or str(override).lower() in ("", "auto", "default"):
            return self.renderer
        rtype = GraphRendererType(str(override).lower())
        if self._renderer is not None and self._renderer.name == rtype.value:
            return self._renderer
        return self._build_renderer(rtype)

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


__all__ = ["GraphConfig", "GraphRendererManager", "GraphRendererType", "get_graph_renderer_manager"]
