"""
The current user's storage scope, derived from the Streamlit session (None outside a session or without a user).

Nothing below ui calls it (layering rules R1 and R3): the UI passes the scope down, and ui.services builds the
session's services with it (M5b step 5).
"""

from __future__ import annotations

from core.workspace import UserScope


def get_current_scope() -> UserScope | None:
    """Scope of the signed-in user, or ``None`` when there is no Streamlit session or no user."""
    try:
        import streamlit as st

        username = st.session_state.get("username")
        role = st.session_state.get("user_role")
    except Exception:
        return None
    if not username:
        return None
    try:
        return UserScope.for_user(str(username), role)
    except ValueError:
        return None


__all__ = ["get_current_scope"]
