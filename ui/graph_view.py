"""
Streamlit entry points for graph rendering (in the ui layer since M5b: the graph package below it loads Streamlit
only inside a renderer's ``translate`` and ``render``).

Pages call :func:`render_graph` with a :class:`~graph.model.GraphSpec` and
:class:`~graph.model.RenderOptions`; the active backend is chosen by :mod:`graph.manager`
(configuration, in-app selector) and failures are isolated so that the rest of the page still
renders. Nothing here knows about yFiles or Plotly specifics.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import streamlit as st

from core.registry import AUTO, as_code
from graph.base import BaseGraphRenderer
from graph.manager import get_graph_renderer_manager
from graph.model import GraphSpec, RenderOptions, SelectionResult
from graph.renderers import GRAPH_RENDERERS
from ui.graph_legend import render_legend

logger = logging.getLogger(__name__)

RENDERER_OVERRIDE_KEY = "graph_renderer_override"


def current_theme() -> str:
    """'dark' or 'light' from the viewer's Streamlit theme (falls back to light)."""
    try:
        theme = st.context.theme
        kind = getattr(theme, "type", None)
        return "dark" if kind == "dark" else "light"
    except Exception:
        return "light"


def _resolve_theme(options: RenderOptions) -> RenderOptions:
    if options.theme != "auto":
        return options
    return options.model_copy(update={"theme": current_theme()})


def active_renderer(renderer: BaseGraphRenderer | None = None) -> BaseGraphRenderer:
    """The renderer that :func:`render_graph` will use (explicit > session override > configuration)."""
    if renderer is not None:
        return renderer
    manager = get_graph_renderer_manager()
    return manager.get(st.session_state.get(RENDERER_OVERRIDE_KEY))


def render_graph(
    spec: GraphSpec, options: RenderOptions, *, renderer: BaseGraphRenderer | None = None
) -> SelectionResult:
    """Render a graph spec with the active backend; never raises (errors are shown in place)."""
    manager = get_graph_renderer_manager()
    chosen = active_renderer(renderer)
    resolved = _resolve_theme(options)

    if resolved.show_legend and spec.legend:
        with st.expander("Legend", expanded=False):
            render_legend(spec)

    if not spec.nodes:
        st.info("Nothing to draw: the graph has no nodes for the current filters.")
        return SelectionResult(renderer=chosen.name)

    drawn = chosen
    try:
        result = chosen.render(spec, resolved)
    except Exception as exc:
        _failed("Graph rendering failed", chosen, spec, exc)
        st.error(
            f"The graph engine '{chosen.display_name}' could not render this graph: {_brief(exc)}"
        )
        if not manager.config.fallback_on_error:
            return SelectionResult(renderer=chosen.name)
        try:
            fallback = manager.fallback(chosen.name)
            if fallback is None:  # the offline engine itself failed, or cannot draw here
                return SelectionResult(renderer=chosen.name)
            st.info(f"Falling back to the built-in offline engine, {fallback.display_name}.")
            result = fallback.render(
                spec, resolved.model_copy(update={"key": f"{resolved.key}__fallback"})
            )
            drawn = fallback
        except Exception as exc2:
            _failed("Fallback graph rendering failed", chosen, spec, exc2)
            st.error(f"The fallback engine failed as well: {_brief(exc2)}")
            return SelectionResult(renderer=chosen.name)

    caps = drawn.capabilities  # the engine that drew, after a fallback too
    if caps.requires_internet and manager.config.show_license_notice:
        st.caption(
            f"Rendered with {drawn.display_name}. {caps.license_note} "
            "See About > Third-party components for details, or switch to the offline engine in the sidebar."
        )
    return result


def _failed(what: str, renderer: BaseGraphRenderer, spec: GraphSpec, exc: Exception) -> None:
    """The server log gets the engine and the kind of error only: a library's message can quote the graph's labels,
    which are the user's data (the details, at the debug level an administrator turns on)."""
    logger.warning(
        "%s (renderer=%s, kind=%s): %s", what, renderer.name, spec.kind, type(exc).__name__
    )
    logger.debug("%s", what, exc_info=exc)


def _brief(exc: Exception) -> str:
    """A library's error on the page: as code (it can quote the graph's labels), and cut."""
    return as_code(exc, limit=200)


def renderer_selector(
    *,
    container=None,
    key: str = RENDERER_OVERRIDE_KEY,
    label: str = "Graph engine",
    on_change: Callable[[], None] | None = None,
) -> None:
    """
    Selectbox listing the available renderers; the choice applies to the whole session. ``on_change`` runs when the
    user changes it (Settings keeps it for the next session; graph imports nothing from ui).
    """
    manager = get_graph_renderer_manager()
    target = container if container is not None else st
    # no session choice: the server's setting when it names an engine that can draw here, else offline first
    server = manager.select()
    default = "Automatic (offline first)"
    if manager.config.renderer != AUTO and server.chosen:
        default = f"Server default: {GRAPH_RENDERERS.display_name(server.chosen)}"
    labels: dict[str, str] = {AUTO: default}
    for name, found in server.alternatives:  # every engine, and whether it can draw here
        display = GRAPH_RENDERERS.display_name(name)
        labels[name] = display if found.ok else f"{display} (unavailable: {found.reason})"
    choices = [AUTO] + [name for name, found in server.alternatives if found.ok]
    current = st.session_state.get(key) or AUTO
    index = choices.index(current) if current in choices else 0
    chosen = target.selectbox(
        label,
        options=choices,
        index=index,
        format_func=lambda value: labels.get(value, value),
        help="The default is the engine the administrator set (CMP_GRAPH_RENDERER), else the offline Plotly "
        "engine. The yFiles engine needs internet access (it loads its runtime from yworks.com) and reports "
        "anonymised usage data to yWorks.",
        key=f"{key}__selectbox",
        on_change=on_change,
    )
    st.session_state[key] = None if chosen == AUTO else chosen


def render_graph_exports(
    spec: GraphSpec,
    options: RenderOptions,
    *,
    renderer: BaseGraphRenderer | None = None,
    file_stem: str = "graph",
) -> None:
    """Download buttons: JSON always, standalone HTML when the renderer supports it."""
    chosen = active_renderer(renderer)
    cols = st.columns(2)
    with cols[0]:
        st.download_button(
            "Download graph (JSON)",
            data=spec.to_json(),
            file_name=f"{file_stem}.json",
            mime="application/json",
            key=f"{options.key}__dl_json",
            width="stretch",
        )
    if chosen.capabilities.supports_html_export:
        with cols[1]:
            try:
                payload = chosen.export_html(spec, _resolve_theme(options))
            except Exception as exc:  # export must never break the page
                logger.warning("HTML export failed: %s", exc)
                payload = None
            if payload:
                st.download_button(
                    "Download graph (HTML)",
                    data=payload,
                    file_name=f"{file_stem}.html",
                    mime="text/html",
                    key=f"{options.key}__dl_html",
                    width="stretch",
                )


__all__ = [
    "RENDERER_OVERRIDE_KEY",
    "active_renderer",
    "current_theme",
    "render_graph",
    "render_graph_exports",
    "renderer_selector",
]
