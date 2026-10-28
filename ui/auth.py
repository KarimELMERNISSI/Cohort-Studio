"""Login screen (a centred, branded card) and session bootstrap for the IAM subsystem."""

from __future__ import annotations

import logging

import streamlit as st

from core.workspace import earlier_workspaces, move_earlier_workspace
from manage.iam import UserCredentials, UserRole, get_iam_manager
from manage.iam.authz import MIN_PASSWORD_LENGTH, validate_new_credentials
from manage.iam.schemas import IAMProviderType
from ui.theme import APP_NAME, APP_TAGLINE, brand_asset
from utils.logging_config import single_line

logger = logging.getLogger(__name__)

LOGO = brand_asset("icon.png")


def logout() -> None:
    """Drop everything tied to the user (managers bound to their workspace, datasets, caches; ui.services)."""
    from ui.services import end_session

    end_session("the user signed out")


def _first_admin_card(iam) -> None:
    """First run: create the administrator with the one-time setup token written to the server log."""
    with st.container(border=True, key="cmp-login"):
        with st.container(key="cmp-hero", horizontal_alignment="center"):
            st.image(LOGO, width=84, alt=f"{APP_NAME} logo")
            st.title(APP_NAME, anchor=False, text_alignment="center")
            st.markdown("**First run: create the administrator**", text_alignment="center")
        st.info(
            "No administrator account exists yet. The server wrote a one-time setup token to its log: for Docker, "
            "`docker compose logs cohortstudio`; otherwise the terminal that started the application. Enter it "
            "with the administrator's username and password.",
            icon=":material/key:",
        )
        with st.form("first_admin_form", border=False):
            token = st.text_input("Setup token", type="password", icon=":material/key:")
            username = st.text_input(
                "Administrator username",
                help="3-32 characters: letters, digits, '.', '_' or '-'",
                icon=":material/person:",
            )
            password = st.text_input(
                "Password",
                type="password",
                help=f"At least {MIN_PASSWORD_LENGTH} characters",
                icon=":material/lock:",
            )
            confirm = st.text_input("Confirm password", type="password", icon=":material/lock:")
            if st.form_submit_button("Create the administrator", type="primary", width="stretch"):
                problem = validate_new_credentials(username, password, confirm)
                if problem:
                    st.error(problem, icon=":material/error:")
                else:
                    ok, message = iam.create_first_admin(token, username.strip(), password)
                    if ok:
                        st.session_state["_first_admin_created"] = username.strip()
                        st.rerun()
                    st.error(message, icon=":material/error:")


def _move_earlier_workspace(iam, username: str) -> None:
    """
    The workspace folder an earlier version gave this account, moved to its folder now when it is clearly this
    account's (core.workspace.move_earlier_workspace): with local accounts only, whose list is complete and names each
    account as it signs in. Another identity provider may list its accounts in pages, or under another name: the
    folder stays, and the log names it for an administrator (dev_scripts/move_workspaces.py).
    """
    try:
        if iam.provider_type == IAMProviderType.LOCAL:
            moved, outcome = move_earlier_workspace(
                username, lambda: [user.username for user in iam.list_users()]
            )
        else:
            moved, left = False, [folder.name for folder in earlier_workspaces(username)]
            outcome = (
                f"the folder {', '.join(left)} an earlier version gave this account is left for an "
                "administrator (dev_scripts/move_workspaces.py)"
                if left
                else ""
            )
    except Exception as exc:  # never in the way of a sign-in
        logger.warning(
            "An earlier workspace folder was not moved (%s: %s)",
            type(exc).__name__,
            single_line(str(exc)),
        )
        return
    if outcome:
        level = logging.INFO if moved else logging.WARNING
        logger.log(level, "Workspace folder at sign-in: %s", single_line(outcome))


def _login_card(iam) -> None:
    if iam.setup_pending():
        _first_admin_card(iam)
        return
    created = st.session_state.pop("_first_admin_created", None)
    if created:
        st.success(
            f"Administrator '{created}' created. Sign in to continue.",
            icon=":material/check_circle:",
        )
    with st.container(border=True, key="cmp-login"):
        with st.container(key="cmp-hero", horizontal_alignment="center"):
            st.image(LOGO, width=84, alt=f"{APP_NAME} logo")
            with st.container(key="cmp-hero-eyebrow"):
                st.markdown("Clinical cohort platform", text_alignment="center")
            st.title(APP_NAME, anchor=False, text_alignment="center")
            st.markdown(f"**{APP_TAGLINE}**", text_alignment="center")
            st.caption(
                "Sign in to open your workspace. Your datasets, traces and documents are private to your account.",
                text_alignment="center",
            )

        tab_login, tab_signup = st.tabs(
            [":material/login: Sign in", ":material/person_add: Create an account"]
        )

        with tab_login, st.form("login_form", border=False):
            username = st.text_input("Username", autocomplete="username", icon=":material/person:")
            password = st.text_input(
                "Password", type="password", autocomplete="current-password", icon=":material/lock:"
            )
            if st.form_submit_button("Sign in", type="primary", width="stretch"):
                result = iam.authenticate(
                    UserCredentials(username=username.strip(), password=password)
                )
                if result.success:
                    _move_earlier_workspace(iam, username.strip())
                    st.session_state.user_authenticated = True
                    st.session_state.username = username.strip()
                    st.session_state.user_role = (
                        str(result.user.role)
                        if result.user is not None
                        else UserRole.RESEARCHER.value
                    )
                    st.rerun()
                else:
                    st.error(result.message, icon=":material/error:")

        with tab_signup, st.form("signup_form", border=False):
            new_user = st.text_input(
                "New username",
                help="3-32 characters: letters, digits, '.', '_' or '-'",
                icon=":material/person:",
            )
            new_pass = st.text_input(
                "New password",
                type="password",
                help=f"At least {MIN_PASSWORD_LENGTH} characters",
                icon=":material/lock:",
            )
            confirm_pass = st.text_input(
                "Confirm password", type="password", icon=":material/lock:"
            )
            if st.form_submit_button("Create account", width="stretch"):
                problem = validate_new_credentials(new_user, new_pass, confirm_pass)
                if problem:
                    st.error(problem, icon=":material/error:")
                else:
                    success, msg = iam.create_user(new_user.strip(), new_pass)
                    (st.info if success else st.error)(msg)
            st.caption("New accounts are activated by an administrator.")

    st.caption(
        ":material/lock: Data stays on this server  ·  :material/badge: Role-based access  ·  :material/history: Full audit trail",
        text_alignment="center",
    )


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

    st.space("large")
    _, centre, _ = st.columns([1, 1.1, 1])
    with centre:
        _login_card(iam)
    return False


__all__ = ["LOGO", "logout", "require_login"]
