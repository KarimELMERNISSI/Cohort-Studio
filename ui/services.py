"""
The services of a session, built in one place and never by a page (M5b step 5, the composition root).

- :func:`shared`: the services of the server process (``manage.services.Shared``), cached for the process; nothing of
  a user's.
- :func:`services`: the session's (``manage.services.Services``), for the signed-in user, kept in the session. The
  dataset store and the trace are also kept under the keys the pages read (``db_manager``, ``transformation_manager``),
  the same objects. A session that holds anything of another user (their services, a store, an engine) is ended as a
  sign-out ends it: the user changes only by signing in again, and nothing of the last one may stay.
- :func:`ai_engine`: the session's AI engine. It is heavy and optional: built on first use and kept under the key the
  pages read (``rag_manager``); a page that only uses a running engine reads that key and builds nothing. A dropped
  engine is stopped first, so that it sends nothing more.

Not ``st.cache_resource(scope="session")``: Streamlit drops those entries when the browser's connection drops, while
the session state survives, so a network blip would replace the trace under the data.
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from manage.paths import PathResolver
from manage.services import Services, Shared, build_services, build_shared, same_user
from ui.session_scope import get_current_scope

logger = logging.getLogger(__name__)

ENGINE_KEY = "rag_manager"
SERVICES_KEY = "_services"
# the stores of the services, under the keys the pages read
STORE_KEYS = {"datasets": "db_manager", "trace": "transformation_manager"}


@st.cache_resource(show_spinner=False)
def shared() -> Shared:
    """The services of the server process: the settings, the identity provider, the graph renderers, the budget."""
    return build_shared()


def services() -> Services:
    """
    The session's services, for the signed-in user, built on first use. A session that holds anything of another user
    is ended, as a sign-out ends it (the next run asks for a sign-in). Services built by the code before a change keep
    their stores, with the trace, and get the current classes.
    """
    scope = get_current_scope()
    found = st.session_state.get(SERVICES_KEY)
    held = {
        name: st.session_state[key] for name, key in STORE_KEYS.items() if key in st.session_state
    }
    if _holds_anothers(scope, found, held):
        end_session("another user was signed in to this session")
    current = type(found) is Services and type(found.paths) is PathResolver
    if found is not None and current and same_user(found.scope, scope):
        return found
    # the stores of the same user only: one built before the sign-in uses the shared folders
    kept = {
        name: store
        for name, store in held.items()
        if same_user(getattr(store, "scope", None), scope)
    }
    built = build_services(shared(), scope, PathResolver(scope=scope), **kept)
    st.session_state[SERVICES_KEY] = built
    for name, key in STORE_KEYS.items():
        st.session_state[key] = getattr(built, name)
    return built


def ai_engine() -> Any:
    """
    The session's AI engine, built on first use for the signed-in user's workspace. An engine of a class reloaded
    since it was built (a code change while the server runs) is stopped and built again with the key typed for it; an
    engine a test put there is kept.
    """
    from manage.rag_manager import RAGManager

    scope = services().scope  # first: another user's engine is stopped there
    engine = st.session_state.get(ENGINE_KEY)
    if engine is None:
        engine = st.session_state[ENGINE_KEY] = RAGManager(scope=scope)
    elif type(engine) is not RAGManager and _same_class(type(engine), RAGManager):
        key = getattr(engine, "api_key", None)
        _stop(engine, "the AI engine was reloaded after a code change")
        engine = st.session_state[ENGINE_KEY] = RAGManager(api_key=key, scope=scope)
    return engine


def running_engine() -> Any:
    """The session's AI engine when it runs (Settings > AI engine), else None; nothing is built."""
    engine = st.session_state.get(ENGINE_KEY)
    return engine if engine is not None and getattr(engine, "initialized", False) else None


def engine_model(engine: Any) -> str:
    """The chat model an engine runs: each connector keeps it in its configuration (``llm.config.model_name``)."""
    llm = getattr(engine, "llm", None)
    model = getattr(getattr(llm, "config", None), "model_name", None) or getattr(
        llm, "model_name", None
    )
    return str(model or "")[:80]


def end_session(reason: str) -> None:
    """
    End the session, as a sign-out does: the AI engine is stopped first, so that it sends nothing more, then every key
    is cleared, even if stopping failed, and the app runs again from the sign-in screen with nothing of the last user.
    """
    try:
        drop_ai_engine(reason)
    finally:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
    st.rerun()


def drop_ai_engine(reason: str) -> None:
    """Stop the session's AI engine and forget it: nothing more is sent, and the next use builds another."""
    _stop(st.session_state.pop(ENGINE_KEY, None), reason)


def _holds_anothers(scope: Any, found: Any, held: dict[str, Any]) -> bool:
    """Whether the session holds services, a store or an AI engine built for another user than ``scope``'s. An owner
    of None (built before a sign-in, or put there by a test) is no other user."""
    engine = st.session_state.get(ENGINE_KEY)
    owners = [getattr(found, "scope", None), getattr(engine, "scope", None)]
    owners += [getattr(store, "scope", None) for store in held.values()]
    return any(owner is not None and not same_user(owner, scope) for owner in owners)


def _same_class(found: type, current: type) -> bool:
    """The same class reloaded: another object with the same module and name."""
    return (found.__module__, found.__qualname__) == (current.__module__, current.__qualname__)


def _stop(engine: Any, reason: str) -> None:
    """Stop an engine; one that fails to stop is dropped all the same (never keeping a user signed in)."""
    stop = getattr(engine, "stop", None)
    if callable(stop):
        try:
            stop(reason)
        except Exception as exc:
            logger.warning(
                "The AI engine did not stop cleanly (%s); it is dropped", type(exc).__name__
            )


__all__ = [
    "ENGINE_KEY",
    "SERVICES_KEY",
    "STORE_KEYS",
    "ai_engine",
    "drop_ai_engine",
    "end_session",
    "services",
    "shared",
]
