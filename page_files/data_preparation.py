# pages/data_preparation.py
import streamlit as st
import pandas as pd

def app():
    if st.session_state.data is None:
        st.warning("Please upload data first!")
        return
    
    st.subheader("Data Preparation")
    
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