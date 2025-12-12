# pages/data_preparation.py
import streamlit as st
import pandas as pd
from explore.data_quality import DataQualityAuditor

def app():
    if st.session_state.data is None:
        st.warning("Please upload data first!")
        return
    
    st.subheader("Data Preparation & Quality Assessment")

    # --- Data Quality Scorecard ---
    with st.expander("📊 Data Quality Scorecard", expanded=True):
        # Pass config if available
        config = st.session_state.get("config", None)
        auditor = DataQualityAuditor(st.session_state.data, config=config)
        metrics = auditor.run_audit()
        advice_list = auditor.generate_advice()

        # Display Metrics
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            help_text = ""
            if name == "Completeness": help_text = "Percentage of non-missing cells."
            elif name == "Uniqueness": help_text = "Percentage of unique rows."
            elif name == "Statistical Validity": help_text = "Score based on absence of statistical outliers (IQR method)."
            elif name == "Consistency": help_text = "Score based on correct data type usage."
            elif name == "Clinical Validity": help_text = "Percentage of rows compliant with declared anomaly criteria."
            
            col.metric(name, f"{value}%", help=help_text)

        # Display Advice
        st.markdown("### 💡 Improvement Advice")
        for item in advice_list:
            if item['severity'] == 'high':
                st.error(f"**{item['category']}:** {item['message']}")
            elif item['severity'] == 'medium':
                st.warning(f"**{item['category']}:** {item['message']}")
            elif item['severity'] == 'low':
                st.info(f"**{item['category']}:** {item['message']}")
            else:
                st.success(f"**{item['category']}:** {item['message']}")

    st.divider()
    
    with st.expander("Data Type Conversion"):
        selected_cols = st.multiselect(
            "Select columns to convert",
            st.session_state.data.columns
        )
        if selected_cols:
            new_type = st.selectbox(
                "Convert to:",
                ["int", "float", "string", "category", "datetime"]
            )
            if st.button("Convert"):
                try:
                    for col in selected_cols:
                        if new_type == "datetime":
                            st.session_state.data[col] = pd.to_datetime(st.session_state.data[col])
                        else:
                            st.session_state.data[col] = st.session_state.data[col].astype(new_type)
                    st.success("Conversion successful!")
                except Exception as e:
                    st.error(f"Error during conversion: {str(e)}")
    
    with st.expander("Handle Missing Values"):
        handle_missing_values()

def handle_missing_values():
    cols_with_missing = st.session_state.data.columns[
        st.session_state.data.isnull().any()
    ].tolist()
    
    if not cols_with_missing:
        st.info("No missing values found in the dataset.")
        return
        
    selected_cols = st.multiselect(
        "Select columns with missing values",
        cols_with_missing
    )
    
    if selected_cols:
        strategy = st.selectbox(
            "Handling strategy",
            ["Drop rows", "Fill with mean", "Fill with median", 
             "Fill with mode", "Fill with value"]
        )
        
        if strategy == "Fill with value":
            fill_value = st.text_input("Fill value")
        
        if st.button("Apply"):
            try:
                if strategy == "Drop rows":
                    st.session_state.data.dropna(subset=selected_cols, inplace=True)
                elif strategy == "Fill with mean":
                    for col in selected_cols:
                        st.session_state.data[col].fillna(
                            st.session_state.data[col].mean(), inplace=True
                        )
                elif strategy == "Fill with median":
                    for col in selected_cols:
                        st.session_state.data[col].fillna(
                            st.session_state.data[col].median(), inplace=True
                        )
                elif strategy == "Fill with mode":
                    for col in selected_cols:
                        st.session_state.data[col].fillna(
                            st.session_state.data[col].mode()[0], inplace=True
                        )
                elif strategy == "Fill with value":
                    for col in selected_cols:
                        st.session_state.data[col].fillna(fill_value, inplace=True)
                st.success("Missing values handled successfully!")
            except Exception as e:
                st.error(f"Error handling missing values: {str(e)}")