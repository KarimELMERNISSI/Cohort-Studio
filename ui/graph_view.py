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

import streamlit as st

from core.registry import AUTO, Selection, as_code
from graph.base import BaseGraphRenderer
from graph.manager import get_graph_renderer_manager
from graph.model import GraphSpec, RenderOptions, SelectionResult
from graph.renderers import GRAPH_RENDERERS
from ui import preferences
from ui.components.connector_selector import connector_selector
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
    """The renderer that :func:`render_graph` will use (explicit > the session's choice > the saved one >
    configuration)."""
    if renderer is not None:
        return renderer
    manager = get_graph_renderer_manager()
    if RENDERER_OVERRIDE_KEY in st.session_state:  # set by the engine selector
        return manager.get(st.session_state[RENDERER_OVERRIDE_KEY])
    # no selector shown yet in this session: the saved engine when it can draw here (the name is not logged)
    kept = manager.select(preference=preferences.current().graph_renderer)
    return manager.get(kept.chosen if kept.source == "preference" else None)


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


def renderer_selector(*, container=None, explain: bool = False) -> Selection:
    """
    The graph engine of the whole session, offered on Settings and beside each graph; the engine chosen is kept for
    the next session. A page that draws with the session's engine shows it first: it brings back the saved choice.
    With ``explain`` (Settings), why the engine is used and which engines cannot draw here.
    """
    manager = get_graph_renderer_manager()
    key = f"{RENDERER_OVERRIDE_KEY}__selectbox"
    selection = connector_selector(
        GRAPH_RENDERERS,
        key=key,
        label="Graph engine",
        preference=preferences.current().graph_renderer,
        preference_name="graph_renderer",
        setting=manager.config.renderer,
        seed=st.session_state.get(RENDERER_OVERRIDE_KEY),  # chosen on another page
        check=manager.availability,  # the offline mode turns yFiles off
        help_text="The default is the engine the administrator set (CMP_GRAPH_RENDERER), else the offline Plotly "
        "engine. The yFiles engine needs internet access (it loads its runtime from yworks.com) and reports "
        "anonymised usage data to yWorks.",
        explain=explain,
        container=container,
    )
    # what render_graph draws with (active_renderer): None is the server's default
    automatic = st.session_state.get(key) == AUTO
    st.session_state[RENDERER_OVERRIDE_KEY] = None if automatic else selection.chosen
    return selection


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
