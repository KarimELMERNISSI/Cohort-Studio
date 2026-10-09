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
import warnings

#import requests
from collections.abc import Callable

import streamlit as st

from utils.multipage import MultiPageApp

# Filter expected statistical edge-case warnings
warnings.filterwarnings("ignore", message=".*sample arguments is too small.*")
warnings.filterwarnings("ignore", message=".*SmallSampleWarning.*")
warnings.filterwarnings("ignore", message=".*An input array is constant.*")

def lazy_page(module_name: str, func_name: str = "app") -> Callable:
    """Returns a callable that lazily imports and executes a page module only when accessed."""
    def _page_runner():
        mod = importlib.import_module(module_name)
        page_func = getattr(mod, func_name)
        page_func()
    return _page_runner

import time

from manage.db_manager import DBManager


def check_auth():
    """Manages authentication with sidebar login/signup."""
    if "user_authenticated" not in st.session_state:
        st.session_state.user_authenticated = False
        st.session_state.username = None

    if st.session_state.user_authenticated:
        with st.sidebar:
            st.divider()
            st.write(f"User: **{st.session_state.username}**")
            if st.button("Logout", width="stretch"):
                st.session_state.user_authenticated = False
                st.session_state.username = None
                st.rerun()
        return True

    # Initialize DB Manager for Auth
    if 'db_manager' not in st.session_state or not hasattr(st.session_state.db_manager, 'verify_user'):
        st.session_state.db_manager = DBManager()
    
    db = st.session_state.db_manager

    with st.sidebar:
        # Centered Logo
        try:
             st.image("assets/karim-app-logo.png", width="stretch")
        except:
             pass
             
        st.title("User Access")
        
        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])
        
        with tab_login, st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Login", width="stretch")
            
            if submit_login:
                success, msg = db.verify_user(username, password)
                if success:
                    st.session_state.user_authenticated = True
                    st.session_state.username = username
                    st.success(msg)
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(msg)
        
        with tab_signup, st.form("signup_form"):
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")
            submit_signup = st.form_submit_button("Create Account", width="stretch")
            
            if submit_signup:
                if new_pass != confirm_pass:
                    st.error("Passwords do not match")
                elif len(new_pass) < 4:
                    st.error("Password must be at least 4 characters")
                else:
                    success, msg = db.create_user(new_user, new_pass)
                    if success:
                        st.success("Account created. Please log in.")
                    else:
                        st.error(msg)
                            
    return False

def main():

    imrb_icon = "assets/karim-app-logo.png"
    
    st.set_page_config(
        page_title="Cohort Monitoring & Analytics Platform",
        page_icon=imrb_icon,
        layout="wide"
    )

    if not check_auth():
        st.stop()  # Do not continue if check_auth is not True.
    
    # Set default page based on user type
    default_page = "Users Management" if st.session_state.get('username') == 'admin' else None
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
    if st.session_state.get('username') == 'admin':
        app.add_page("Users Management", lazy_page("app_pages.users_management"), icon="")
    
    app.add_page("About", lazy_page("app_pages.about"), icon="")
    
    app.run()

if __name__ == "__main__":
    main()
