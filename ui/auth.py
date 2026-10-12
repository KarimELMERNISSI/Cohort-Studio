"""Login screen (centred in the main area) and session bootstrap for the IAM subsystem."""

from __future__ import annotations

import streamlit as st

from manage.iam import UserCredentials, UserRole, get_iam_manager
from manage.iam.authz import MIN_PASSWORD_LENGTH, validate_new_credentials

LOGO = "assets/karim-app-logo.png"


def _reset_session() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def logout() -> None:
    """Drop everything tied to the user (managers bound to their workspace, datasets, caches)."""
    _reset_session()
    st.rerun()


def require_login() -> bool:
    """True when a user is signed in; otherwise renders the login/sign-up screen and returns False."""
    iam = get_iam_manager()

    if not iam.is_enabled:  # single-researcher workstation mode
        st.session_state.user_authenticated = True
        st.session_state.username = "local_researcher"
        st.session_state.user_role = UserRole.ADMIN.value
        return True

    if "user_authenticated" not in st.session_state:
        st.session_state.user_authenticated = False
        st.session_state.username = None
        st.session_state.user_role = None

    if st.session_state.user_authenticated:
        return True

    _, centre, _ = st.columns([1, 1.2, 1])
    with centre:
        try:
            st.image(LOGO, width=180)
        except Exception:
            pass
        st.title("Cohort Monitoring Platform")
        st.caption("Sign in to open your workspace. Your datasets, traces and documents are private to your account.")

        tab_login, tab_signup = st.tabs(["Sign in", "Create an account"])

        with tab_login, st.form("login_form"):
            username = st.text_input("Username", autocomplete="username")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            if st.form_submit_button("Sign in", type="primary", width="stretch"):
                result = iam.authenticate(UserCredentials(username=username.strip(), password=password))
                if result.success:
                    st.session_state.user_authenticated = True
                    st.session_state.username = username.strip()
                    st.session_state.user_role = (
                        str(result.user.role) if result.user is not None else UserRole.RESEARCHER.value
                    )
                    st.rerun()
                else:
                    st.error(result.message)

        with tab_signup, st.form("signup_form"):
            new_user = st.text_input("New username", help="3-32 characters: letters, digits, '.', '_' or '-'")
            new_pass = st.text_input("New password", type="password", help=f"At least {MIN_PASSWORD_LENGTH} characters")
            confirm_pass = st.text_input("Confirm password", type="password")
            if st.form_submit_button("Create account", width="stretch"):
                problem = validate_new_credentials(new_user, new_pass, confirm_pass)
                if problem:
                    st.error(problem)
                else:
                    success, msg = iam.create_user(new_user.strip(), new_pass)
                    (st.info if success else st.error)(msg)
            st.caption("New accounts are activated by an administrator.")

    return False


__all__ = ["LOGO", "logout", "require_login"]
