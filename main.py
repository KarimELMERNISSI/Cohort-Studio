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
from app_pages import home, data_preparation, statistical_tests, visualization, data_enrichment, data_monitoring, reproduce_analysis, epidemiology, data_insight, yfiles_test, document_insight, about

# Lazy import for RAG monitoring to avoid performance impact
def load_rag_monitoring():
    """Lazy loader for RAG monitoring page - only imported when accessed."""
    from app_pages import rag_monitoring
    return rag_monitoring.render_rag_monitoring

def main():

    # Fetch icons from URLs
    #imrb_icon = requests.get("https://github.com/KarimELMERNISSI/MetaboSign/blob/main/images/metabosign_icon.png?raw=true").content
    imrb_icon = "assets/karim-app-logo.png" #Image.open("imrb-logo.png")
    
    
    st.set_page_config(
        page_title="Statistical Analysis Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    

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
    app.add_page("About", about.app, "ℹ️") # ABOUT
    #app.add_page("yFiles Test", yfiles_test.app, "🧪")
    #app.add_page("Statistical Tests", statistical_tests.app, "📊🔧")
    
    app.run()

if __name__ == "__main__":
    main()
