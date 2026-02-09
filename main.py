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
import streamlit as st
#import requests
from io import BytesIO
from PIL import Image
from utils.multipage import MultiPageApp
from app_pages import home, data_preparation, statistical_tests, visualization, data_enrichment, data_monitoring, reproduce_analysis, epidemiology, data_insight, yfiles_test, document_insight, about, users_management

# Lazy import for RAG monitoring to avoid performance impact
def load_rag_monitoring():
    """Lazy loader for RAG monitoring page - only imported when accessed."""
    from app_pages import rag_monitoring
    return rag_monitoring.render_rag_monitoring

from manage.db_manager import DBManager
import time

def check_auth():
    """Manages authentication with sidebar login/signup."""
    if "user_authenticated" not in st.session_state:
        st.session_state.user_authenticated = False
        st.session_state.username = None

    if st.session_state.user_authenticated:
        with st.sidebar:
            st.divider()
            st.write(f"👤 User: **{st.session_state.username}**")
            if st.button("Logout", use_container_width=True):
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
             st.image("assets/karim-app-logo.png", use_container_width=True)
        except:
             pass
             
        st.title("🔐 Access")
        
        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])
        
        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit_login = st.form_submit_button("Login", use_container_width=True)
                
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
        
        with tab_signup:
            with st.form("signup_form"):
                new_user = st.text_input("New Username")
                new_pass = st.text_input("New Password", type="password")
                confirm_pass = st.text_input("Confirm Password", type="password")
                submit_signup = st.form_submit_button("Create Account", use_container_width=True)
                
                if submit_signup:
                    if new_pass != confirm_pass:
                        st.error("Passwords do not match")
                    elif len(new_pass) < 4:
                        st.error("Password must be at least 4 characters")
                    else:
                        success, msg = db.create_user(new_user, new_pass)
                        if success:
                            st.success("Account created! Please login.")
                        else:
                            st.error(msg)
                            
    return False

def main():

    # Fetch icons from URLs
    #imrb_icon = requests.get("https://github.com/KarimELMERNISSI/MetaboSign/blob/main/images/metabosign_icon.png?raw=true").content
    imrb_icon = "assets/karim-app-logo.png" #Image.open("imrb-logo.png")
    
    
    st.set_page_config(
        page_title="Statistical Analysis Dashboard",
        page_icon="📊",
        layout="wide"
    )

    if not check_auth():
        st.stop()  # Do not continue if check_auth is not True.
    
    

    app = MultiPageApp()
    
    # Register pages
    app.add_page("Main View", home.app, imrb_icon ) #"🏠") # INGESTION & DESCRIPTION
    app.add_page("Data Validation & Monitoring", data_monitoring.app, "🔍") #👁️ CONTROL & VALIDATION
    app.add_page("Data Enrichment", data_enrichment.app, "🔄") # ENRICHMENT
    app.add_page("Data Insight", data_insight.app, "🧠") # INTELLIGENCE -> DATA INSIGHT
    app.add_page("Documents Insight", document_insight.app, "📄") # INTELLIGENCE -> DOC INSIGHT
    app.add_page("Epidemiology & Hypothesis", epidemiology.app, "🧬") # ANALYSIS -> HYPOTHESIS TESTING, POWER ANALYSIS, etc

    app.add_page("Visualization", visualization.app, "📈") # DIFFUSION -> VISUALIZATION / REPORT
    app.add_page("Reproduce Analysis", reproduce_analysis.app, "🔁") # REPRODUCIBILITY

    app.add_page("RAG Quality Monitor", load_rag_monitoring(), "📊") # RAG QUALITY MONITOR (OPTIONAL - more for myself)
    
    # Admin-only pages
    if st.session_state.get('username') == 'admin':
        app.add_page("Users Management", users_management.app, "👥") # ADMIN ONLY
    
    app.add_page("About", about.app, "ℹ️") # ABOUT
    #app.add_page("yFiles Test", yfiles_test.app, "🧪")
    #app.add_page("Statistical Tests", statistical_tests.app, "📊🔧")
    
    app.run()

if __name__ == "__main__":
    main()
