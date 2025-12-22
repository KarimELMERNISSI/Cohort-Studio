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
from app_pages import home, data_preparation, statistical_tests, visualization, config_form, data_enrichment, data_monitoring, reproduce_analysis, epidemiology, interpretation, yfiles_test, document_insight

def main():

    # Fetch icons from URLs
    #imrb_icon = requests.get("https://github.com/KarimELMERNISSI/MetaboSign/blob/main/images/metabosign_icon.png?raw=true").content
    imrb_icon = "assets/imrb-logo.png" #Image.open("imrb-logo.png")
    
    
    st.set_page_config(
        page_title="Statistical Analysis Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    

    app = MultiPageApp()
    
    # Register pages
    app.add_page("Main View", home.app, imrb_icon ) #"🏠")
    #app.add_page("Data Quality Dashboard", data_quality.app, "✅")
    app.add_page("Data Enrichment", data_enrichment.app, "🔄")
    app.add_page("Data Insight", data_insight.app, "🧠")
    app.add_page("Documents Insight", document_insight.app, "📄")
    app.add_page("Epidemiology & Hypothesis", epidemiology.app, "🧬")
    app.add_page("Data Validation & Monitoring", data_monitoring.app, "🔍") #👁️‍🗨️
    app.add_page("Visualization", visualization.app, "📈")
    app.add_page("Reproduce Analysis", reproduce_analysis.app, "🔁")
    app.add_page("Configuration Form", config_form.app, "⚙️")
    #app.add_page("yFiles Test", yfiles_test.app, "🧪")
    #app.add_page("Statistical Tests", statistical_tests.app, "📊🔧")
    
    app.run()

if __name__ == "__main__":
    main()