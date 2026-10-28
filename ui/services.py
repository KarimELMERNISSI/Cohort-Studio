"""
The services of a session, built in one place and never by a page (M5b step 5, the composition root).

The AI engine is heavy and optional. It is built on first use and kept in the session, under the key the pages read
(``rag_manager``); a page that only uses a running engine reads that key and builds nothing. A dropped engine is
stopped first, so that it sends nothing more.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from ui.session_scope import get_current_scope

ENGINE_KEY = "rag_manager"


def ai_engine() -> Any:
    """
    The session's AI engine, built on first use for the signed-in user's workspace. An engine of a class reloaded
    since it was built (a code change while the server runs) is stopped and built again with the key typed for it; an
    engine a test put there is kept.
    """
    from manage.rag_manager import RAGManager

    engine = st.session_state.get(ENGINE_KEY)
    if engine is None:
        engine = st.session_state[ENGINE_KEY] = RAGManager(scope=get_current_scope())
    elif type(engine) is not RAGManager and _same_class(type(engine), RAGManager):
        key = getattr(engine, "api_key", None)
        _stop(engine, "the AI engine was reloaded after a code change")
        engine = st.session_state[ENGINE_KEY] = RAGManager(api_key=key, scope=get_current_scope())
    return engine


def drop_ai_engine(reason: str) -> None:
    """Stop the session's AI engine and forget it: nothing more is sent, and the next use builds another."""
    _stop(st.session_state.pop(ENGINE_KEY, None), reason)


def _same_class(found: type, current: type) -> bool:
    """The same class reloaded: another object with the same module and name."""
    return (found.__module__, found.__qualname__) == (current.__module__, current.__qualname__)


def _stop(engine: Any, reason: str) -> None:
    stop = getattr(engine, "stop", None)
    if callable(stop):
        stop(reason)


__all__ = ["ENGINE_KEY", "ai_engine", "drop_ai_engine"]
