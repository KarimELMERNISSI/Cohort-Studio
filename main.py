# app/
# ├── main.py
# ├── multipage.py
# ├── pages/
# │   ├── __init__.py
# │   ├── config_form.py
# │   ├── home.py
# │   ├── data_preparation.py
# │   ├── statistical_tests.py
# │   └── visualization.py
# └── utils/
#     ├── __init__.py
#     └── data_operations.py

# main.py
import importlib

#import requests
from collections.abc import Callable

import streamlit as st

from utils.multipage import MultiPageApp

def lazy_page(module_name: str, func_name: str = "app") -> Callable:
    """Returns a callable that lazily imports and executes a page module only when accessed."""
    def _page_runner():
        mod = importlib.import_module(module_name)
        page_func = getattr(mod, func_name)
        page_func()
    return _page_runner

from manage.db_manager import DBManager
from manage.iam import UserCredentials, UserRole, get_iam_manager
from manage.iam.authz import MIN_PASSWORD_LENGTH, is_admin_role, validate_new_credentials
from utils.logging_config import configure_logging


def check_auth():
    """Manages authentication with the modular IAM subsystem; privileges come from the stored role."""
    iam = get_iam_manager()

    # If IAM is explicitly disabled, bypass authentication (single-researcher workstation mode)
    if not iam.is_enabled:
        st.session_state.user_authenticated = True
        st.session_state.username = "local_researcher"
        st.session_state.user_role = UserRole.ADMIN.value
        return True

    if "user_authenticated" not in st.session_state:
        st.session_state.user_authenticated = False
        st.session_state.username = None
        st.session_state.user_role = None

    if st.session_state.user_authenticated:
        with st.sidebar:
            st.divider()
            st.write(f"User: **{st.session_state.username}**")
            if st.button("Logout", width="stretch"):
                # Drop everything tied to the user: managers bound to their workspace, datasets, caches.
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        return True

    with st.sidebar:
        try:
            st.image("assets/karim-app-logo.png", width="stretch")
        except Exception:
            pass

        st.title("User Access")

        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

        with tab_login, st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Login", width="stretch")

            if submit_login:
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
            new_user = st.text_input("New Username", help="3-32 characters: letters, digits, '.', '_' or '-'")
            new_pass = st.text_input("New Password", type="password", help=f"At least {MIN_PASSWORD_LENGTH} characters")
            confirm_pass = st.text_input("Confirm Password", type="password")
            submit_signup = st.form_submit_button("Create Account", width="stretch")

            if submit_signup:
                problem = validate_new_credentials(new_user, new_pass, confirm_pass)
                if problem:
                    st.error(problem)
                else:
                    success, msg = iam.create_user(new_user.strip(), new_pass)
                    if success:
                        st.info(msg)
                    else:
                        st.error(msg)

    return False

@st.cache_resource(show_spinner=False)
def _bootstrap_admin_once() -> str:
    """Create the initial administrator from CMP_ADMIN_* settings, once per server process."""
    _, message = get_iam_manager().ensure_bootstrap_admin()
    return message


def main():
    configure_logging()
    imrb_icon = "assets/karim-app-logo.png"

    st.set_page_config(
        page_title="Cohort Monitoring & Analytics Platform",
        page_icon=imrb_icon,
        layout="wide"
    )

    _bootstrap_admin_once()

    if not check_auth():
        st.stop()  # Do not continue if check_auth is not True.

    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DBManager()

    # Role-based routing (never derived from the username string)
    is_admin = is_admin_role(st.session_state.get('user_role'))
    default_page = "Users Management" if is_admin else None
    app = MultiPageApp(default_page=default_page)
    
    # Register pages lazily with clean typography
    app.add_page("Main View", lazy_page("app_pages.home"), icon="")
    app.add_page("Data Validation & Monitoring", lazy_page("app_pages.data_monitoring"), icon="")
    app.add_page("Data Enrichment", lazy_page("app_pages.data_enrichment"), icon="")
    app.add_page("Data Insight", lazy_page("app_pages.data_insight"), icon="")
    app.add_page("Documents Insight", lazy_page("app_pages.document_insight"), icon="")
    app.add_page("Epidemiology & Hypothesis", lazy_page("app_pages.epidemiology"), icon="")
    app.add_page("Visualization", lazy_page("app_pages.visualization"), icon="")
    app.add_page("Reproduce Analysis", lazy_page("app_pages.reproduce_analysis"), icon="")
    app.add_page("RAG Quality Monitor", lazy_page("app_pages.rag_monitoring", "render_rag_monitoring"), icon="")
    
    # Admin-only pages
    if is_admin:
        app.add_page("Users Management", lazy_page("app_pages.users_management"), icon="")
    
    app.add_page("About", lazy_page("app_pages.about"), icon="")
    
    app.run()

if __name__ == "__main__":
    main()
