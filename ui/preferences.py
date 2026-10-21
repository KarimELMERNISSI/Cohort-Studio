"""
The user's preferences in the session (M5b, step 3): the choices of Settings, kept for the next session.

``seed_session()`` runs after sign-in, once per user: it fills the keys the session has not set yet with the choices
kept in the user's file (manage.preferences). The Settings page saves a choice from the control's ``on_change``
callback, which seeding never fires. A choice that is no longer offered (a model the provider no longer lists, an
engine this server cannot run) is not used: the page says which, and the file keeps it until the user chooses again.

The order of a choice is the session's (the control, or the running engine), then the preference, then the server's
setting, then the default. No key, password or consent is ever saved (audit E6).
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from core.registry import as_code
from manage.preferences import Preferences, load_preferences, update_preferences
from utils.session_scope import get_current_scope

PREFERENCES_KEY = "_preferences"  # the user's preferences, read once per session
SEEDED_KEY = "_preferences_seeded"  # the user whose preferences seeded this session


def current() -> Preferences:
    """The signed-in user's preferences (none outside a signed-in session)."""
    prefs = st.session_state.get(PREFERENCES_KEY)
    if isinstance(prefs, Preferences):
        return prefs
    scope = get_current_scope()
    prefs = load_preferences(scope) if scope is not None else Preferences()
    st.session_state[PREFERENCES_KEY] = prefs
    return prefs


def seed_session() -> None:
    """After sign-in, once per user: the choices kept from the last session, for the keys not set yet."""
    scope = get_current_scope()
    ss = st.session_state
    if scope is None or ss.get(SEEDED_KEY) == scope.username:
        return
    ss.pop(PREFERENCES_KEY, None)  # another user signed in: their own file
    ss[SEEDED_KEY] = scope.username
    prefs = current()
    if prefs.import_dayfirst is not None and "import_default_dayfirst" not in ss:
        ss["import_default_dayfirst"] = prefs.import_dayfirst
    if prefs.graph_renderer:
        from graph.manager import get_graph_renderer_manager
        from ui.graph_view import RENDERER_OVERRIDE_KEY

        available = {
            rtype.value for rtype, ok, _ in get_graph_renderer_manager().available_renderers() if ok
        }
        # an engine this server cannot run (yFiles offline, say) stays unused
        if prefs.graph_renderer in available and RENDERER_OVERRIDE_KEY not in ss:
            ss[RENDERER_OVERRIDE_KEY] = prefs.graph_renderer


def save(**changes: Any) -> None:
    """Keep these choices for the next session (an ``on_change`` callback). Never raises: see update_preferences."""
    scope = get_current_scope()
    if scope is None:
        return
    saved = update_preferences(scope, **changes)
    if saved is not None:
        st.session_state[PREFERENCES_KEY] = saved


def save_renderer() -> None:
    """The ``on_change`` of every graph engine selector: the engine chosen is kept for the next session."""
    from ui.graph_view import RENDERER_OVERRIDE_KEY

    save_from(f"{RENDERER_OVERRIDE_KEY}__selectbox", "graph_renderer", auto="auto")


def shown(name: str) -> str:
    """A saved name, as code in a notice: the file is the user's, and its text must not become a link or a format."""
    return as_code(name)


def renderer_notice() -> str | None:
    """Why the graph engine kept from the last session is not used on this server, or None."""
    saved = current().graph_renderer
    if not saved:
        return None
    from graph.manager import get_graph_renderer_manager

    for rtype, ok, reason in get_graph_renderer_manager().available_renderers():
        if rtype.value == saved:
            if ok:
                return None
            return (
                f"Your saved graph engine {shown(saved)} is not available here ({reason}): the automatic engine "
                "is used."
            )
    return (
        f"Your saved graph engine {shown(saved)} is not offered here: the automatic engine is used."
    )


def save_from(key: str, name: str, *, provider: str | None = None, auto: str | None = None) -> None:
    """
    An ``on_change`` callback: keep the value of the control ``key`` as the preference ``name`` (for ``provider``,
    for the models). The value ``auto`` (the server's default) forgets the preference.
    """
    value = st.session_state.get(key)
    if value is not None and value == auto:
        value = None
    save(**{name: {provider: value} if provider is not None else value})


__all__ = [
    "PREFERENCES_KEY",
    "SEEDED_KEY",
    "current",
    "renderer_notice",
    "save",
    "shown",
    "save_from",
    "save_renderer",
    "seed_session",
]
