import streamlit as st
import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from io import BytesIO
from utils.multipage import load_dataframe, get_file_hash
import enrich.external_data as eed
from utils.data_analyzer import DataAnalyzer
from fuzzywuzzy import fuzz
import os
import re
from pandas.api.types import is_numeric_dtype
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.manifold import TSNE
import umap
import enrich.data_imputation as edi
import prince
import logging
from manage.db_manager import DBManager
from manage.transformation_manager import TransformationManager
from datetime import datetime
import ast

# Note: DataTransformationEngine class was removed (dead code)
# See git history if needed

def handle_main_data_upload() -> Optional[pd.DataFrame]:
    """Handle the upload or path input for the main dataset."""
    if 'data' in st.session_state and st.session_state.data is not None:
        st.success("✅ Main dataset loaded from session")
        return st.session_state.data

    upload_method = st.radio("Choose upload method for Main Dataset:", ["Upload File", "Enter File Path"], key='main_data_method')
    
    if upload_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Drag and drop your main dataset here",
            type=['csv', 'xlsx'],
            key='main_data_uploader'
        )
        if uploaded_file is not None:
            try:
                st.write(f"file_name: {uploaded_file.name}")
                file_path = uploaded_file.name
                df = load_dataframe(uploaded_file)
                current_file_hash = get_file_hash(uploaded_file)
                
                if 'last_main_file_hash' not in st.session_state or \
                   current_file_hash != st.session_state['last_main_file_hash']:
                    st.session_state['last_main_file_hash'] = current_file_hash
                    st.success("✅ Main dataset successfully loaded from upload.")
                return df
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return None
    else:
        file_path = st.text_input("Enter file path for the main dataset:", key='main_data_path')
        
        if file_path:
            try:
                df = load_dataframe(file_path)
                st.success("✅ Main dataset successfully loaded from provided path.")
                return df
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return None
    return None

def handle_enrichment_data_upload() -> Optional[Union[pd.DataFrame, str]]:
    """Handle the upload or path input for the enrichment dataset."""
    upload_method = st.radio("Choose upload method for Enrichment Dataset:", ["Upload File", "Enter File Path"], key='enrichment_data_method')
    
    if upload_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Drag and drop your enrichment dataset here",
            type=['csv', 'xlsx'],
            key='enrichment_data_uploader'
        )
        if uploaded_file is not None:
            try:
                st.write(f"file_name: {uploaded_file.name}")
                file_path = uploaded_file.name
                
                # Save to artifacts if session is active
                if 'transformation_manager' in st.session_state and st.session_state.transformation_manager.session_id:
                    session_id = st.session_state.transformation_manager.session_id
                    artifact_dir = os.path.join("data", "traces", "artifacts", session_id)
                    os.makedirs(artifact_dir, exist_ok=True)
                    
                    # Create a safe filename
                    safe_name = "".join([c for c in uploaded_file.name if c.isalnum() or c in (' ', '.', '_', '-')]).strip()
                    saved_path = os.path.join(artifact_dir, safe_name)
                    
                    # Save the file
                    uploaded_file.seek(0)
                    with open(saved_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Use the saved path as the file_path for the trace
                    file_path = saved_path
                    uploaded_file.seek(0) # Reset for loading

                df = load_dataframe(uploaded_file)
                current_file_hash = get_file_hash(uploaded_file)
                
                if 'last_enrichment_file_hash' not in st.session_state or \
                   current_file_hash != st.session_state['last_enrichment_file_hash']:
                    st.session_state['last_enrichment_file_hash'] = current_file_hash
                    st.success("✅ Enrichment dataset successfully loaded from upload.")
                return df, file_path
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return None, None
    else:
        # Get folder path and file name from user input
        folder_path = st.text_input("Enter folder path for the enrichment dataset:", key='enrichment_folder_path')
        file_name = st.text_input("Enter file name for the enrichment dataset:", key='enrichment_data_name')

        # Dynamically set file_path based on folder_path and file_name
        file_path = os.path.join(folder_path, file_name) if folder_path and file_name else ""

        # Display file path input field with computed value
        file_path_input = st.text_input("Enter file path for the enrichment dataset:", value=file_path, key='enrichment_data_path')
        if file_path_input:
            try:
                df = load_dataframe(file_path_input)
                st.success("✅ Enrichment dataset successfully loaded from provided path.")
                return df, file_name
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return None, None
    return None, None


def configure_enrichment(main_df: pd.DataFrame, enrichment_df: pd.DataFrame, enrichment_file_path: str) -> Optional[pd.DataFrame]:
    """Configure and perform the data enrichment."""
    st.subheader("Enrichment Configuration")
    
    # Configuration form
    enrichment_name = st.text_input(
        "Enrichment Name", 
        "New Enrichment",
        help="Give a name to this enrichment configuration"
    )

    input_file_folder = st.text_input(
        "Input File Folder", 
        "./data/enrichment/",
        help="Give the path of the folder containing the data file"
    )

    input_file = st.text_input(
        "Input File Name", 
        enrichment_file_path,
        help="Give the name of the data file"
    )
    
    add_choice = st.radio(
        "What do you want to add?",
        ["Columns", "Rows"],
        help="Choose how to augment the datasets",
        horizontal=True
    )
    if add_choice == 'Columns':
        strategy = st.selectbox(
            "Join Strategy",
            ["left", "right", "outer", "inner", "cross"],
            help="Select how to join the datasets"
        )
    else:
        strategy = "rows"
    
    # Display strategy explanation
    strategy_explanations = {
        "left": ":grey_question: Keep all rows from main dataset, only matching rows from enrichment dataset",
        "right": ":grey_question: Keep all rows from enrichment dataset, only matching rows from main dataset",
        "outer": ":grey_question: Keep all rows from both datasets",
        "inner": ":grey_question: Keep only rows that match in both datasets",
        "cross": ":grey_question: Create all possible combinations of rows",
        "rows": ":grey_question: Stack the rows from both datasets"
    }
    st.info(strategy_explanations[strategy])

    # Select conflict resolution strategy
    conflict_resolution = st.selectbox(
        "Conflict Resolution",
        ['keep', 'replace', 'ignore'],
        index=0,
        help="Choose how to handle columns with the same name in the datasets: \
            'keep' to append suffixes to duplicates, 'replace' to overwrite with enrichment data, \
            'ignore' to keep original data columns."
    )
    
    row_id_approach = st.radio(
        "Row Identifier Approach",
        ["Common Row Identifier", "Distinct Row Identifiers"],
        help="Choose how to match rows between datasets",
        horizontal=True
    )
    
    enriched_df = None
    
    if row_id_approach == "Common Row Identifier":
        common_columns = list(set(main_df.columns) & set(enrichment_df.columns))
        if not common_columns:
            st.warning("No common columns found between datasets!")
            return None
        
        identifier = st.multiselect(
            "Select Common Row Identifier",
            options=common_columns,
            help="Choose the column that exists in both datasets"
        )
        
        if st.button("Perform Enrichment"):
            try:
                # Ensure right_identifier is a set for compatibility with the subtraction operation
                identifier_set = {identifier} if isinstance(identifier, str) else set(identifier)

                # Compute additional columns
                additional_columns = list(set(enrichment_df.columns) - identifier_set)

                enriched_df = eed.add_data(
                    df=main_df, additional_df=enrichment_df, 
                    left_id_names=identifier, right_id_names=identifier, 
                    additional_cols=additional_columns,
                    strategy=strategy, 
                    conflict_resolution=conflict_resolution)
                
                if enriched_df is not None:
                    # Save snapshot
                    snapshot_path = save_snapshot(enriched_df, "enrichment")
                    
                    # Log transformation step
                    params = {
                        "enrichment_file_path": enrichment_file_path,
                        "strategy": strategy,
                        "conflict_resolution": conflict_resolution,
                        "identifier": identifier,
                        "add_choice": add_choice
                    }
                    st.session_state.transformation_manager.add_step(
                        "enrichment", 
                        params, 
                        "Enriched with external data",
                        output_dataset_path=snapshot_path
                    )
                st.success("✅ Enrichment completed successfully!")
            except Exception as e:
                st.error(f"Error performing enrichment: {str(e)}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            left_identifier = st.multiselect(
                "Select Main Dataset Row Identifier",
                options=main_df.columns,
                help="Choose the column from main dataset"
            )
        with col2:
            right_identifier = st.multiselect(
                "Select Enrichment Dataset Row Identifier",
                options=enrichment_df.columns,
                help="Choose the column from enrichment dataset"
            )
        
        if st.button("Perform Enrichment"):
            try:
                # Ensure right_identifier is a set for compatibility with the subtraction operation
                right_identifier_set = {right_identifier} if isinstance(right_identifier, str) else set(right_identifier)

                # Compute additional columns
                additional_columns = list(set(enrichment_df.columns) - right_identifier_set)

                enriched_df = eed.add_data(
                    df=main_df, additional_df=enrichment_df, 
                    left_id_names=left_identifier, right_id_names=right_identifier, 
                    additional_cols=additional_columns,
                    strategy=strategy, 
                    conflict_resolution=conflict_resolution)

                save_enrichment_config(enrichment_name=enrichment_name, 
                                       strategy=strategy,
                                       input_file_folder=input_file_folder,
                                       input_file=input_file,
                                       left_identifier=left_identifier,
                                       right_identifier=right_identifier,
                                       additional_columns=additional_columns,
                                       conflict_resolution=conflict_resolution
                                       )
                st.success("✅ Enrichment completed successfully!")
            except Exception as e:
                st.error(f"Error performing enrichment: {str(e)}")
    
    return enriched_df


def display_category_box(title, columns):
    if len(columns)>0:
        st.markdown(
            f"""
            <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <h4 style="margin: 0; padding: 0; color: #333;">{title} ({len(columns)} column(s))</h4>
                <p style="margin: 0; color: #666;">{', '.join(columns) if columns is not None else 'No columns found.'}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def display_results(enriched_df: pd.DataFrame, title: str=None ,key_base='k'):
    """Display the results of the enrichment process."""

    # Preview the enriched dataset
    if title is not None:
        st.subheader(title)
    st.dataframe(enriched_df.head(), width='stretch')
    
    # Column information
    st.subheader("Columns Information")
    # Analyze the dataset using DataAnalyzer
    enr_analyzer = DataAnalyzer(enriched_df)
    column_categories = {
                "Quantitative: Columns with numerical quantitative data": sorted(enr_analyzer.numeric_cols),
                "Binary: Columns with 2 unique values (e.g., 0/1 or Yes/No)": sorted(enr_analyzer.binary_cols),
                "Non Binary Low-Cardinality Numeric: Numerical columns with few distinct values": sorted(enr_analyzer.non_binary_low_cardinality_numeric_cols),
                "Categorical: Columns with non-numerical data": sorted(enr_analyzer.categorical_cols),
                "High-Cardinality Categorical: Categorical columns with many unique values (e.g., id)": sorted(enr_analyzer.high_cardinality_cat_cols),
                "Date: Columns containing date or time data": sorted(enr_analyzer.date_cols),
                "Time Delta: Columns containing time interval data (e.g., age or period)": sorted(enr_analyzer.timedelta_cols)
            }
    for category, columns in column_categories.items():
        display_category_box(category, columns)

    col1, col2 = st.columns(2)
    with col1:
        # Download button
        st.download_button(
            label="Download Enriched Dataset",
            key= f"{key_base}_dl_bt",
            data=to_excel(enriched_df),
            file_name=f"{key_base}_dataset.xlsx",
            mime="text/xlsx",
            width='stretch'
        )
    
    with col2:
        # Save to session state and persist automatically
        if st.button("Use This Dataset for Further Analysis", width='stretch', key=f"{key_base}_save_bt",):
            st.session_state.data = enriched_df
            st.session_state.working_df = enriched_df
            
            # Automatic persistence
            if 'db_manager' in st.session_state:
                # Determine a base name based on context
                base_name = "cohort_data" 
                if "external" in key_base: base_name = "enriched_data"
                elif "imputation" in key_base: base_name = "imputed_data"
                elif "new_variables" in key_base: base_name = "calculated_data"
                
                success, msg, saved_name = st.session_state.db_manager.save_dataset(enriched_df, base_name=base_name)
                if success:
                    st.session_state['current_dataset_name'] = saved_name
                    
                    # Save transformation trace
                    if 'transformation_manager' in st.session_state:
                        trace_path = f"{saved_name}_trace.json"
                        st.session_state.transformation_manager.save_trace(trace_path)
                        st.info(f"Transformation trace saved to {trace_path}")

                    st.success(f"✅ Dataset saved as {saved_name} and ready for analysis!")
                else:
                    st.error(f"Failed to save dataset: {msg}")
            else:
                 st.success("✅ Dataset loaded to session (Persistence unavailable)!")


def display_external_data_results(enriched_df: pd.DataFrame, filtered_main_data: pd.DataFrame):
    """Display the results of the enrichment process."""
    st.subheader("Enrichment Results")
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total Rows", value=len(enriched_df), delta=len(enriched_df)-len(filtered_main_data))
    with col2:
        st.metric("Total Columns", len(enriched_df.columns), delta=len(enriched_df.columns)-len(filtered_main_data.columns))
    # Preview the enriched dataset
    display_results(enriched_df=enriched_df, title="Enriched Dataset Preview", key_base="external_data")


def display_new_variables_results(enriched_df: pd.DataFrame, main_data: pd.DataFrame):
    """Display the results of the enrichment process."""
    st.subheader("Enrichment Results")
    
    # Display metrics
    col1, col2 = st.columns([1,3],vertical_alignment='center')
    with col1:
        st.metric(label="Total Columns", value=len(enriched_df.columns), delta=len(enriched_df.columns)-len(main_data.columns))
    with col2:
        st.write(list(set(enriched_df.columns) - set(main_data.columns)))
        
    # Preview the enriched dataset
    display_results(enriched_df=enriched_df, title="Enriched Dataset Preview", key_base="new_variables")


def display_imputation_results(imputed_data: pd.DataFrame, main_data: pd.DataFrame, cols_per_row: int = 8):
    """Display the results of the enrichment process."""
    st.subheader("Imputation Results")
    
    # List to hold column metrics
    columns = imputed_data.columns.tolist()

    # Iterate over columns in chunks based on the number of columns per row
    for i in range(0, len(columns), cols_per_row):
        # Create a row of columns
        cols = st.columns(cols_per_row)
        
        # Fill each column in the row with a metric
        for j, column in enumerate(columns[i:i + cols_per_row]):
            # Calculate non-null counts for the column
            total_non_null_imputed = int(imputed_data[column].notna().sum())
            total_non_null_main = int(main_data[column].notna().sum()) if column in main_data.columns else 0
            delta_non_null = total_non_null_imputed - total_non_null_main

            # Display metric
            cols[j].metric(
                label=f"{column} Non-Null",
                value=total_non_null_imputed,
                delta=delta_non_null  # Conversion handled by int()
            )
        
    # Preview the enriched dataset
    display_results(enriched_df=imputed_data, title="Imputed Dataset Preview", key_base="imputation")


def save_enrichment_config(enrichment_name: str, strategy: str, 
                         input_file_folder: str = "./data/enrichment/",
                         input_file: str = None,
                         identifier: str = None, 
                         left_identifier: str = None, 
                         right_identifier: str = None,
                         additional_columns: str = None,
                         conflict_resolution: str = None
                         ):
    """Save the enrichment configuration to session state."""
    if 'config' not in st.session_state:
        st.session_state.config = {'data_enrichments': {}}
    
    enrichment_config = {
        "STRATEGY": strategy,
        "INPUT_FILES_FOLDER": input_file_folder,
        "INPUT_FILE": input_file,
        "CONFLICT_RESOLUTION": conflict_resolution
    }

    if additional_columns is not None:
        enrichment_config["ADDITIONAL_COLUMNS"] = additional_columns
    
    if identifier:
        enrichment_config["COMMON_ROW_IDENTIFIER"] = identifier
    else:
        enrichment_config["LEFT_ROW_IDENTIFIER"] = left_identifier
        enrichment_config["RIGHT_ROW_IDENTIFIER"] = right_identifier
    
    st.session_state.config['data_enrichments'][enrichment_name] = enrichment_config


# Convert DataFrame to Excel for download using openpyxl
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=True, sheet_name='Sheet1')
    return output.getvalue()


def calculate_prefix_score(input_token: str, column_name: str) -> float:
    """
    Calculate a score based on how well the input matches the beginning of the column name.
    
    Parameters:
    -----------
    input_token : str
        The user's input token
    column_name : str
        The column name to compare against
        
    Returns:
    --------
    float
        Prefix matching score between 0 and 1
    """
    input_lower = input_token.lower()
    col_lower = column_name.lower()
    
    # Perfect prefix match
    if col_lower.startswith(input_lower):
        return 1.0
    
    # Calculate longest common prefix
    common_prefix_length = 0
    for i in range(min(len(input_lower), len(col_lower))):
        if input_lower[i] == col_lower[i]:
            common_prefix_length += 1
        else:
            break
    
    # Return score based on prefix match quality
    if common_prefix_length == 0:
        return 0.0
    return common_prefix_length / len(input_lower)


def calculate_similarity_score(input_token: str, column_name: str) -> int:
    """
    Calculate a similarity score between input token and column name with
    prefix-weighted scoring and length-aware adjustments.
    
    Parameters:
    -----------
    input_token : str
        The user's input token
    column_name : str
        The column name to compare against
        
    Returns:
    --------
    int
        Adjusted similarity score
    """
    input_len = len(input_token)
    col_len = len(column_name)
    
    # Convert both to lowercase for comparison
    input_lower = input_token.lower()
    col_lower = column_name.lower()
    
    # Calculate base similarity scores
    ratio = fuzz.ratio(input_lower, col_lower)
    partial_ratio = fuzz.partial_ratio(input_lower, col_lower)
    token_sort_ratio = fuzz.token_sort_ratio(input_lower, col_lower)
    
    # Calculate prefix score (0 to 1)
    prefix_score = calculate_prefix_score(input_lower, col_lower)
    
    # Weight the scores (giving more weight to prefix matches)
    prefix_weight = 0.4  # 40% of the score comes from prefix matching
    fuzzy_weight = 0.6   # 60% from fuzzy matching
    
    # Calculate weighted fuzzy score
    fuzzy_score = max(ratio, partial_ratio, token_sort_ratio)
    
    # Combine scores with weights
    base_score = (prefix_score * 100 * prefix_weight) + (fuzzy_score * fuzzy_weight)
    
    # Apply length-based adjustments
    if col_len < input_len:
        # Penalize columns shorter than input unless it's an exact match
        if input_lower.startswith(col_lower):
            base_score = base_score * 0.9  # Small penalty for exact prefix match
        else:
            base_score = base_score * 0.5  # Heavy penalty for shorter non-matching columns
    
    # Penalize very short column names (less than 2 characters)
    if col_len <= 2:
        if not (input_lower == col_lower):  # Don't penalize exact matches
            base_score = base_score * 0.6
    
    # Additional penalty for single-character columns that don't exactly match
    if col_len == 1 and input_lower != col_lower:
        base_score = base_score * 0.3
    
    # Bonus for exact word beginnings in multi-word columns
    if "_" in col_lower or " " in col_lower:
        words = col_lower.replace("_", " ").split()
        if any(word.startswith(input_lower) for word in words):
            base_score = min(base_score * 1.2, 100)  # 20% bonus, capped at 100
    
    return int(base_score)


def tokenize_formula(formula: str) -> List[str]:
    """
    Tokenizes the formula, treating quoted variables as single tokens and splitting others by operators.
    
    Parameters:
    ----------
    formula : str
        The formula input by the user.
        
    Returns:
    -------
    List[str]
        The list of tokens extracted from the formula.
    """
    # Remove leading and trailing spaces from the formula
    formula = formula.strip()

    # Regex to capture anything within double quotes as a single token
    quoted_tokens = re.findall(r'\"\"[^\"]+\"\"', formula)
    
    # Replace quoted tokens in the formula with placeholders
    for idx, token in enumerate(quoted_tokens):
        placeholder = f"__quoted_{idx}__"
        formula = formula.replace(token, placeholder)
    
    # Now split the remaining formula by spaces and operators
    #tokens = re.findall(r'\S+', formula)
    tokens = (
        formula.replace(' + ', ' ; ').replace(' - ', ' ; ').replace(' * ', ' ; ').replace(' / ', ' ; ').replace(' ** ', ' ; ') # arithmetic operators
              .replace(' == ', ' ; ').replace(' >= ', ' ; ').replace(' <= ', ' ; ').replace(' < ', ' ; ').replace(' > ', ' ; ') # comparison operators
              .replace(' | ', ' ; ').replace(' & ', ' ; ') # logical operators
              .replace('( ', '').replace(' )', '') # priority operators
              .split(sep=" ; ")
              )
    tokens = [token.strip() for token in tokens]
    print(f"tokens:{tokens}")
    
    # Replace placeholders with the original quoted tokens
    for idx, token in enumerate(quoted_tokens):
        placeholder = f"__quoted_{idx}__"
        stripped_token = token[2:-2].strip()
        tokens = [stripped_token if t.strip() == placeholder else t.strip() for t in tokens]
    print(tokens)
    return tokens


def suggest_columns(computation_formula: str, column_names: List[str], threshold: int = 40, max_suggestions: int = 6) -> List[Tuple[str, int]]:
    """
    Suggest column names based on similarity to the input formula with enhanced
    prefix matching, including handling quoted variable names as unique tokens.
    
    Parameters:
    -----------
    computation_formula : str
        The formula input by the user.
    column_names : List[str]
        List of available column names in the dataset.
    threshold : int, optional
        Minimum similarity score to include in suggestions (default: 60).
    max_suggestions : int, optional
        Maximum number of suggestions to return (default: 5).
        
    Returns:
    --------
    List[Tuple[str, int]]
        List of tuples containing (column_name, similarity_score).
    """
    if not computation_formula or not column_names:
        return []
    
    # Tokenize the formula while treating quoted variables as single tokens
    tokens = tokenize_formula(computation_formula)
    
    last_token = tokens[-1] if tokens else ''
    
    # Skip suggestions if the last token is too short (unless it's the only token)
    if len(last_token) < 2 and len(tokens) > 1:
        return []
    
    # Calculate adjusted similarity scores for each column name
    suggestions = []
    for col in column_names:
        score = calculate_similarity_score(last_token, col)
        if score > threshold:
            suggestions.append((col, score))
    
    # Sort by score, prefix match, and length
    ranked_suggestions = sorted(
        suggestions,
        key=lambda x: (
            -x[1],  # Score (descending)
            -x[0].lower().startswith(last_token.lower()),  # Prefix match (True first)
            len(x[0])  # Length (ascending)
        )
    )
    
    # Return only the top N suggestions (max_suggestions)
    return ranked_suggestions[:max_suggestions]


def display_column_suggestions(suggestions):
    """
    Displays column suggestions as buttons and enables users to copy column names to the clipboard by clicking.

    Parameters:
    ----------
    suggestions : list of tuples
        List of (column name, score) tuples to display.
    """
    if suggestions:
        cols = st.columns(min(3, len(suggestions)))
        
        for idx, (col, score) in enumerate(suggestions):
            with cols[idx % 3]:
                # HTML code for the button with copy-to-clipboard functionality and temporary notification
                html_code = f"""
                <div style="margin: 5px 0; position: relative;">
                    <button 
                        onclick="
                            navigator.clipboard.writeText('{col}').then(() => {{
                                const notification = document.createElement('div');
                                notification.innerText = 'Copied: {col}';
                                notification.style = `
                                    position: absolute;
                                    top: -30px;
                                    left: 50%;
                                    transform: translateX(-50%);
                                    background-color: #4CAF50;
                                    color: white;
                                    padding: 5px 10px;
                                    border-radius: 4px;
                                    font-size: 12px;
                                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
                                    z-index: 1000;
                                `;
                                this.parentElement.appendChild(notification);
                                
                                setTimeout(() => {{
                                    notification.remove();
                                }}, 800);
                            }});
                        "
                        style="
                            background-color: #f0f2f6;
                            border: 1px solid #e0e0e0;
                            border-radius: 4px;
                            padding: 8px;
                            cursor: pointer;
                            width: 100%;
                            text-align: left;
                            box-shadow: 0px 1px 3px rgba(0,0,0,0.1);
                            font-size: 14px;"
                        onmouseover="this.style.backgroundColor='#e0e2e6'"
                        onmouseout="this.style.backgroundColor='#f0f2f6'">
                        <span style="color: #0066cc; font-weight: bold;">{col}</span>
                        <br/>
                        <small style="color: #666;">Score: {score}</small>
                    </button>
                </div>
                """
                st.components.v1.html(html_code, height=60)
    else:
        st.info("No matching columns found. Try typing part of a column name.")


def validate_variable_name(name, existing_columns):
    """
    Validates the variable name and returns (is_valid, error_message).
    """
    if not name:
        return False, "Variable name cannot be empty"
    
    if name in existing_columns:
        return False, f"Variable name '{name}' already exists"
    
    if not re.match(r'^[a-zA-Z_][ a-zA-Z0-9_\[\]\(\)]*$', name):
        return False, "Variable name must start with a letter or underscore and contain only letters, numbers, some special characters, spaces and underscores"
    
    return True, None


# Helper function to check if a string represents a number
def is_number(token):
    try:
        float(token)  # Try converting to float
        return True
    except ValueError:
        return False
    

import ast

def validate_and_extract_columns_ast(formula: str, dataset_columns: list) -> Tuple[bool, str, List[str], List[str]]:
    """
    Parses formula using Python AST to safely extract variables and constants 
    regardless of spacing.
    """
    formula = formula.strip()
    valid_columns = []
    used_columns = []
    error_message = None

    try:
        # Handle quoted variables for AST parsing
        temp_formula = formula
        quoted_vars = {}
        # Find ""Variable Name"" pattern
        quoted_matches = re.findall(r'""([^"]+)""', temp_formula)
        for i, match in enumerate(quoted_matches):
            placeholder = f"__quoted_var_{i}__"
            quoted_vars[placeholder] = match
            temp_formula = temp_formula.replace(f'""{match}""', placeholder)

        # Parse the string into an AST
        tree = ast.parse(temp_formula, mode='eval')
        
        # Traverse the tree to find variable names (nodes of type ast.Name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                # Resolve placeholder if it exists
                var_name = quoted_vars.get(node.id, node.id)
                
                # Check if it's a known library alias or function (np, pd, log, etc.)
                if var_name in ['np', 'pd', 'log', 'sqrt', 'exp', 'abs', 'min', 'max', 'constant']: 
                    continue
                
                # Check if it matches a dataset column
                if var_name in dataset_columns:
                    used_columns.append(var_name)
                else:
                    # Logic for "quoted" columns if you keep your specific syntax
                    # or handle mismatched names here
                    return False, f"Unknown variable: '{var_name}'", [], []

        return True, None, list(set(used_columns)), []

    except SyntaxError as e:
        return False, f"Syntax Error: {e}", [], []
    
    
# def validate_and_extract_columns(formula: str, dataset_columns: List[str]) -> Tuple[bool, str, List[str], List[str]]:
#     """
#     Extract and validate column names from a formula string.
    
#     Parameters:
#     -----------
#     formula : str
#         The computation formula.
#     dataset_columns : List[str] or pd.Index
#         List of valid column names from the dataset.
        
#     Returns:
#     --------
#     tuple:
#         - is_valid (bool): True if the formula is valid, False otherwise.
#         - error_message (str): Error message if validation fails, None if valid.
#         - used_columns (List[str]): List of valid column names used in the formula.
#     """

#     # Ensure consistent formatting of dataset columns
#     dataset_columns = sorted(dataset_columns, key=len, reverse=True)  # Sort by length to avoid partial matches
#     valid_columns = []
#     constant_tokens = []

#     try:
#         # Tokenize the formula while treating quoted variables as single tokens
#         tokens = tokenize_formula(formula)

#         # Check for column usage and extract valid columns from the formula
#         for column in dataset_columns:
#             if column in tokens:
#                 valid_columns.append(column)
#                 # Remove all occurrences of the column (.remove just act one the 1st occ)
#                 tokens = [token for token in tokens if token != column]
        
#         constant_tokens = [token for token in tokens if (token.startswith('constant(') and token.endswith(')') )]
#         # Handle constant values with simple string checks
#         tokens = [token for token in tokens if not ((token.startswith('constant(') and token.endswith(')')) or is_number(token))]

#         # Validate the formula syntax by trying to evaluate it
#         if len(tokens)>0:
#             raise ValueError(f"Invalid column reference detected in the formula: {tokens}")
#         # Return success if valid, with used columns
#         return True, None, valid_columns, constant_tokens
#     except Exception as e:
#         return False, str(e), valid_columns, constant_tokens


# Replace column names in the formula with their corresponding normalized names
def normalize_formula(formula: str, column_mapping: dict) -> str:
    """
    Replace column names in the formula with normalized names (e.g., var_1, var_2).
    
    Parameters:
    ----------
    formula : str
        The computation formula to be normalized.
    column_mapping : dict
        A mapping of original column names to normalized names.
    
    Returns:
    -------
    str
        The formula with normalized column names.
    """
    print("original formula:", formula)
    
    # Sort mapping by length of column name descending to prevent substring replacement issues
    # column_mapping values are Series, so we access .name
    sorted_mapping = sorted(column_mapping.items(), key=lambda x: len(str(x[1].name)), reverse=True)
    
    for normalized, original in sorted_mapping:
        col_name = str(original.name)
        print(f"normalized {normalized}, original {col_name}")
        # Replace each column name with its corresponding normalized variable name
        formula = formula.replace(f'""{col_name}""', f'{normalized}')
        formula = formula.replace(col_name, normalized)  # in case the column name is without quotes
        
    print("norm formula:", formula)
    return formula


def extract_constant_value(constant_token: str) -> str:
    """
    Extract the value from a constant token like constant(val1) -> val1
    """
    # Remove 'constant(' prefix and ')' suffix and return the inner value
    return constant_token[9:-1]  # len('constant(') == 9


def create_variables_dict(dataset: pd.DataFrame, identified_columns: List[str], constant_tokens: List[str]) -> Dict[str, pd.Series]:
    """
    Create a dictionary mapping variable names to their corresponding Series
    for both regular columns and constants.
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        The input dataset
    identified_columns : List[str]
        List of valid column names used in the formula
    constant_tokens : List[str]
        List of constant tokens found in the formula
        
    Returns:
    --------
    Dict[str, pd.Series]:
        Dictionary mapping variable names to their Series
    """
    local_vars = {}
    
    # Handle regular columns
    for idx, col in enumerate(identified_columns):
        local_vars[f"var_{idx}"] = dataset[col]
    
    # Handle constants
    start_idx = len(identified_columns)
    for idx, constant_token in enumerate(constant_tokens, start=start_idx):
        constant_value = extract_constant_value(constant_token)
        # Create a Series with the constant value repeated for each row
        constant_series = pd.Series([constant_value] * len(dataset), index=dataset.index, name=constant_token)
        local_vars[f"var_{idx}"] = constant_series
    
    return local_vars


def evaluate_formula_safely(formula: str, variable_name: str, dataset: pd.DataFrame) -> tuple[bool, str, pd.DataFrame]:
    """
    Safely evaluate a computation formula to create a new variable in the dataset.

    Parameters:
    -----------
    formula : str
        The computation formula entered by the user.
    variable_name : str
        The name of the new variable to create.
    dataset : pd.DataFrame
        The dataset containing columns to use in the formula.

    Returns:
    --------
    tuple[bool, str, pd.DataFrame]
        A tuple containing:
        - A boolean indicating success or failure.
        - A success or error message.
        - The modified DataFrame (or the original DataFrame on failure).
    """
    # Extract dataset column names
    column_names = dataset.columns.tolist()
    
    # Remove leading/trailing whitespace
    formula = formula.strip()
    
    # Validate formula and extract variables
    is_valid, error_message, identified_columns, constants = validate_and_extract_columns_ast(formula, column_names)
    if not is_valid:
        return False, f"Invalid formula: {error_message}", dataset

    # Ensure date columns are properly converted to datetime objects
    try:
        analyzer = DataAnalyzer(dataset)
        for col in identified_columns:
            if col in analyzer.date_cols:
                # Convert to datetime, coercing errors to NaT
                dataset[col] = pd.to_datetime(dataset[col], errors='coerce')
    except Exception as e:
        print(f"Error during date conversion: {e}")

    # Map column names to dataset references
    #local_vars = {col: dataset[col] for col in identified_columns}
    #local_vars = {f"var_{idx}": dataset[col] for idx, col in enumerate(identified_columns)}
    local_vars = create_variables_dict(dataset, identified_columns, constants)

    #normalize the formula
    print(f"local_vars dict: {local_vars} \nInitial formula: {formula}")
    normalized_formula = normalize_formula(formula, local_vars)
    try:
        # Safely evaluate the formula
        print(f"Normalized formula: {normalized_formula}")
        result = eval(normalized_formula, {"np":np, "pd":pd}, local_vars)
        print(result)
        # Check if result is a pandas Series with the correct length
        if isinstance(result, pd.Series) and result.shape[0] == dataset.shape[0]:
            # Add the new variable to the dataset
            if variable_name in dataset.columns:
                return False, f"Variable '{variable_name}' already exists.", dataset

            dataset[variable_name] = result
            return True, f"✨ Variable '{variable_name}' successfully added.", dataset

        # Handle cases where result is not a valid pandas Series
        return False, (
            f"Formula evaluation resulted in invalid output.\n"
            f"Expected a pandas Series with {dataset.shape[0]} rows.\n"
            f"Check for mismatched types or operations in your formula."
        ), dataset

    except Exception as e:
        return False, (
            f"Error evaluating formula: {str(e)}\n"
            f"Make sure your formula uses valid column names and operators (+, -, *, /, etc.)."
        ), dataset


# Create a statistics DataFrame based on the type of the column
def generate_stats(temp_main_data, variable_name):
    column_data = temp_main_data[variable_name]
    
    # Determine the column type
    if pd.api.types.is_numeric_dtype(column_data):
        # For numeric columns
        stats_df = pd.DataFrame({
            'Statistic': ['Mean', 'Std', 'Min', 'Max', 'Non-null Count'],
            'Value': [
                f"{column_data.mean():.2f}" if not column_data.isna().all() else "NaN",
                f"{column_data.std():.2f}" if not column_data.isna().all() else "NaN",
                f"{column_data.min():.2f}" if not column_data.isna().all() else "NaN",
                f"{column_data.max():.2f}" if not column_data.isna().all() else "NaN",
                f"{column_data.count()}"
            ]
        })
    elif pd.api.types.is_timedelta64_dtype(column_data):
        # For timedelta columns
        stats_df = pd.DataFrame({
            'Statistic': ['Mean', 'Std', 'Min', 'Max', 'Non-null Count'],
            'Value': [
                str(column_data.mean()) if not column_data.isna().all() else "NaT",
                str(column_data.std()) if not column_data.isna().all() else "NaT",
                str(column_data.min()) if not column_data.isna().all() else "NaT",
                str(column_data.max()) if not column_data.isna().all() else "NaT",
                f"{column_data.count()}"
            ]
        })
    elif pd.api.types.is_string_dtype(column_data):
        # For string columns, only non-null count is relevant
        stats_df = pd.DataFrame({
            'Statistic': ['Unique', 'Most Frequent', 'Non-null Count'],
            'Value': [
                column_data.nunique(),
                column_data.mode().iloc[0] if not column_data.mode().empty else "No Mode",
                f"{column_data.count()}"
            ]
        })
    else:
        # For other types, provide generic information
        stats_df = pd.DataFrame({
            'Statistic': ['Non-null Count'],
            'Value': [f"{column_data.count()}"]
        })
    
    return stats_df
  

def define_new_variables(main_data):
    """
    Enhanced version of define_new_variables with improved preview and validation.
    """
    if main_data is None:
        st.warning("Please load data first.")
        return main_data
    
    st.subheader("Define New Variable")

    # AI Suggestions Section
    with st.expander("🤖 AI Variable Suggestions (RAG)", expanded=False):
        if 'rag_manager' in st.session_state and st.session_state.rag_manager.initialized:
            
            # Create columns for Search | Mode | Button
            col_search, col_options, col_btn = st.columns([2, 2, 1])
            
            with col_search:
                use_specific = st.toggle("🔍 Specific Search", value=True)
                if use_specific:
                    search_hint = st.text_input("Concept:", placeholder="e.g. 'BSA' or 'Diabetes Risk'", label_visibility="collapsed")
                else:
                    search_hint = None
                    st.caption("Exploratory mode: Finds general high-value features.")
            
            with col_options:
                # DYNAMIC MODE SELECTION
                if use_specific:
                    suggestion_mode = st.selectbox(
                        "Suggestion Type",
                        ["Go To Target", "Go From Target", "Around Target"],
                        help="Go To Target: Find formulas to compute the searched concept from your data (Target is Output).\nGo From Target: Find new variables that use the searched concept as a component (Target is Input).\nAround Target: Find interesting metrics close to this concept and clinically relevant."
                    )
                else:
                    # In exploratory mode, these modes don't apply, so we disable or set to default
                    suggestion_mode = "Comprehensive" 
                    st.caption("Standard Medical Indices")

                num_suggestions = st.slider("Count", min_value=1, max_value=10, value=5)
                
                # NEW: Allow missing variables
                allow_missing = st.checkbox("Include suggestions with missing variables", value=False, help="Allow AI to suggest formulas even if some variables are not in the dataset.")
                
                # NEW: Use Taxonomy
                use_taxonomy = st.checkbox("Use Variable Taxonomy", value=True, help="Use the generated variable taxonomy to understand cryptic column names. (Requires Taxonomy to be generated in Data Preparation)")

                # Debug mode
                debug_mode = st.checkbox("Debug Mode", value=False, help="Show raw LLM output for debugging.")
            
            with col_btn:
                st.write("") # Spacer
                if use_specific:
                    st.write("") # Extra spacer for alignment
                
                if st.button("Generate", width='stretch'):
                    progress_bar = st.progress(0, text="Starting analysis...")
                    
                    def update_progress(percent, text):
                        progress_bar.progress(percent, text=text)

                    try:
                        # Use validation method to ensure quality and filter useless suggestions
                        suggestions_text, error = st.session_state.rag_manager.suggest_computed_variables_with_validation(
                            list(main_data.columns),
                            search_hint=search_hint,
                            num_suggestions=num_suggestions,
                            suggestion_mode=suggestion_mode,
                            allow_missing_variables=allow_missing,
                            use_taxonomy=use_taxonomy,
                            progress_callback=update_progress
                        )
                        progress_bar.empty()
                        
                        if debug_mode:
                            with st.expander("🕵️ Debug: Raw LLM Output", expanded=True):
                                st.code(suggestions_text, language="json")

                    except TypeError:
                        st.error("Session outdated. Please refresh the page (F5) to apply the latest updates.")
                        return

                    if error:
                        st.error(f"Error: {error}")
                    else:
                            try:
                                # Clean up potential markdown code blocks
                                if "```json" in suggestions_text:
                                    suggestions_text = suggestions_text.split("```json")[1].split("```")[0]
                                elif "```" in suggestions_text:
                                    suggestions_text = suggestions_text.split("```")[1].split("```")[0]
                                
                                response_data = json.loads(suggestions_text)
                                
                                # Handle both new object format and old list format
                                if isinstance(response_data, list):
                                    st.session_state['ai_variable_suggestions'] = response_data
                                    st.session_state['ai_domain_analysis'] = None
                                else:
                                    st.session_state['ai_variable_suggestions'] = response_data.get('suggestions', [])
                                    st.session_state['ai_domain_analysis'] = response_data.get('domain_analysis', None)
                                    
                                st.success("Suggestions received!")
                            except Exception as e:
                                st.error(f"Error parsing suggestions: {e}")
                                st.write(suggestions_text) # Fallback

            # Display Domain Analysis
            if 'ai_domain_analysis' in st.session_state and st.session_state['ai_domain_analysis']:
                da = st.session_state['ai_domain_analysis']
                st.info(f"**Dataset Domain:** {da.get('dataset_domain', 'N/A')} | **Document Domain:** {da.get('document_domain', 'N/A')}")
                if 'relevant_domains' in da:
                    st.write("**Relevant Domains:** " + ", ".join([f"`{d}`" for d in da['relevant_domains']]))
                st.markdown("---")

            # Display Suggestions
            if 'ai_variable_suggestions' in st.session_state:
                st.write("### Suggestions:")
                for i, sugg in enumerate(st.session_state['ai_variable_suggestions']):
                    with st.container(border=True):
                        col_head, col_badge = st.columns([3, 1])
                        with col_head:
                            st.markdown(f"#### {sugg.get('title', sugg.get('name', 'Unknown'))}")
                        with col_badge:
                            # Display Category badge if available, otherwise Source Type
                            category = sugg.get('suggestion_category', None)
                            source_type = sugg.get('source_type', 'Unknown')
                            
                            if category:
                                st.caption(f"_{category}_")
                            
                            color = "green" if source_type == "Document" else "orange" if source_type == "Hybrid" else "blue"
                            st.markdown(f":{color}[**{source_type}**]")
                        
                        st.markdown(f"**Description:** {sugg.get('description', sugg.get('reason', ''))}")
                        
                        # Layout: Visualization (Left) | Formula (Right)
                        col_viz, col_formula = st.columns([1, 4], vertical_alignment="center")
                        
                        with col_viz:
                            # Generate mini-graph for this specific suggestion
                            target_name = sugg.get('name', 'Target')
                            formula = sugg.get('formula', '')
                            
                            # Compact graph settings - Adjusted for visibility
                            dot_code = 'digraph G {\n  rankdir="LR";\n  bgcolor="transparent";\n  ranksep=0.2;\n  nodesep=0.1;\n  node [fontname="Arial", fontsize=9, height=0.25, margin=0.05];\n  edge [arrowsize=0.5, penwidth=0.6];\n'
                            dot_code += f'  "{target_name}" [shape=box, style=filled, fillcolor="#e1f5fe"];\n'
                            
                            try:
                                # Handle quoted variables for AST parsing
                                temp_formula = formula
                                quoted_vars = {}
                                # Find ""Variable Name"" pattern
                                quoted_matches = re.findall(r'""([^"]+)""', temp_formula)
                                for i, match in enumerate(quoted_matches):
                                    placeholder = f"__quoted_var_{i}__"
                                    quoted_vars[placeholder] = match
                                    temp_formula = temp_formula.replace(f'""{match}""', placeholder)

                                tree = ast.parse(temp_formula, mode='eval')
                                inputs = set()
                                for node in ast.walk(tree):
                                    if isinstance(node, ast.Name) and node.id != target_name:
                                        var_name = quoted_vars.get(node.id, node.id)
                                        inputs.add(var_name)

                                has_inputs = False
                                for inp in inputs:
                                    if inp in ['np', 'pd', 'log', 'exp', 'sqrt', 'abs', 'min', 'max', 'constant']: continue
                                    has_inputs = True
                                    if inp in main_data.columns:
                                        dot_code += f'  "{inp}" [shape=ellipse, style=filled, fillcolor="#f0f4c3"];\n'
                                    else:
                                        dot_code += f'  "{inp}" [shape=ellipse, style=dashed, color="red"];\n'
                                    dot_code += f'  "{inp}" -> "{target_name}";\n'
                                
                                if not has_inputs:
                                     dot_code += f'  "No Inputs" [shape=plaintext];\n'
                            except Exception as e:
                                dot_code += f'  "Error" [shape=plaintext];\n'
                            
                            dot_code += '}'
                            st.graphviz_chart(dot_code, width='stretch')

                        with col_formula:
                            if 'markdown_formula' in sugg and sugg['markdown_formula']:
                                # Clean formula for st.latex (remove $$ wrappers if present)
                                raw_formula = sugg['markdown_formula']
                                clean_formula = raw_formula.replace('$$', '').replace('$', '').strip()
                                
                                # Heuristic: If it looks like a text explanation rather than a formula, render as markdown
                                if len(clean_formula.split()) > 10 and not any(op in clean_formula for op in ['=', '\\', '^', '_']):
                                    st.warning(clean_formula)
                                else:
                                    # Ensure full equality display: variable = expression
                                    if '=' not in clean_formula:
                                        target_var = sugg.get('name', 'Variable').replace('_', '\\_')
                                        clean_formula = f"\\text{{{target_var}}} = {clean_formula}"
                                    
                                    st.latex(clean_formula)
                            else:
                                st.code(sugg.get('formula', ''), language="python")
                        
                        with st.expander("View Computation Details"):
                            st.caption("Computation Formula (for interpreter):")
                            st.code(sugg.get('formula', ''), language="text")
                            
                            # Check for missing variables
                            missing_vars = sugg.get('missing_variables', [])
                            if missing_vars:
                                st.error(f"⚠️ Missing required variables: {', '.join(missing_vars)}")
                                
                                col_fix1, col_fix2 = st.columns(2)
                                with col_fix1:
                                    if st.button(f"Find Proxy for {missing_vars[0]}", key=f"proxy_{i}_{missing_vars[0]}"):
                                        with st.spinner(f"Finding proxy for {missing_vars[0]}..."):
                                            proxy_res = st.session_state.rag_manager.suggest_proxy_variable(missing_vars[0], list(main_data.columns))
                                            if proxy_res.get('proxy_found'):
                                                st.success(f"Proxy Found: {proxy_res.get('proxy_name')}")
                                                st.info(proxy_res.get('explanation'))
                                                st.code(proxy_res.get('formula'), language="text")
                                            else:
                                                st.warning("No valid proxy found.")
                                                st.caption(proxy_res.get('explanation'))
                                
                                with col_fix2:
                                    if st.button(f"Find Alternative Formula", key=f"alt_{i}_{sugg.get('name', 'unknown')}"):
                                        with st.spinner(f"Finding alternative for {sugg.get('title')}..."):
                                            alt_res = st.session_state.rag_manager.suggest_alternative_formula(sugg.get('title'), missing_vars[0], list(main_data.columns))
                                            if alt_res.get('alternative_found'):
                                                st.success(f"Alternative Found: {alt_res.get('alternative_name')}")
                                                st.info(alt_res.get('explanation'))
                                                st.code(alt_res.get('formula'), language="text")
                                            else:
                                                st.warning("No alternative formula found.")
                                                st.caption(alt_res.get('explanation'))
                            
                            source_type = sugg.get('source_type', 'Unknown')
                            
                            citation = sugg.get('source_citation', 'N/A')
                            explanation = sugg.get('source_explanation', '')
                            
                            # Fallback for legacy structure (if session state persists)
                            if citation == 'N/A' and 'source_info' in sugg:
                                info = sugg['source_info']
                                if isinstance(info, dict):
                                    doc = info.get('document_name')
                                    page = info.get('page_number')
                                    if doc: citation = f"{doc} (Page {page})"
                                    explanation = info.get('verbatim_extract') or info.get('inference_logic') or ''

                            st.markdown("---")
                            st.markdown("**📚 Source Information**")
                            
                            if source_type == "Document":
                                st.info(f"**Source:** {citation}", icon="📄")
                                if explanation:
                                    st.caption(f"_{explanation}_")
                                    
                            elif source_type == "Hybrid":
                                st.warning(f"**Hybrid Source:** {citation}", icon="⚠️")
                                st.markdown(f"**Logic:** {explanation}")
                                
                            elif source_type == "Model Knowledge":
                                st.info(f"**Model Knowledge:** {explanation}", icon="🧠")
                                
                            else:
                                st.info(explanation or citation, icon="ℹ️")

                        if st.button("Apply to Editor", key=f"apply_var_{i}_{sugg.get('name', 'unknown')}", width='stretch'):
                            st.session_state.variable_name_input = sugg.get('name', '')
                            st.session_state.formula_input = sugg.get('formula', '')
                            st.rerun()
                st.markdown("---")

        else:
            st.info("RAG System not initialized. Please configure it in the Home page sidebar.")
    
    # Get column names
    column_names = main_data.columns.tolist()
    
    # Create two columns for input fields
    col1, col2 = st.columns([1, 2])
    
    with col1:
        variable_name = st.text_input(
            "Enter new variable name",
            help="Variable name must start with a letter or underscore and contain only letters, numbers, and underscores",
            key="variable_name_input"
        ).strip()
        
        # Real-time variable name validation
        if variable_name:
            is_valid, error_msg = validate_variable_name(variable_name, column_names)
            if not is_valid:
                st.error(error_msg)
    
    with col2:
        if st.checkbox("Show available columns"):
            st.write("Available columns:", ", ".join(f"`{col}`" for col in column_names))
    
    # Initialize session state
    if 'formula' not in st.session_state:
        st.session_state.formula = ""
    
    # Formula input with enhanced autocompletion
    computation_formula = st.text_area(
        label="Enter computation formula",
        value=st.session_state.formula,
        help=(
            "Type your formula using column names. For example: `column1 + column2 * 2`. "
            "If a column name contains special characters (e.g., 'Type 2 diabetes diagnosis - Year'), "
            "enclose the column name in double quotes like this: `\"\"Type 2 diabetes diagnosis - Year\"\"`."
            "\n\n"
            "You can use basic operators like `+`, `-`, `*`, `/` for arithmetic operations."
        ),
        key="formula_input",
        height=100
    )
    
    # Show real-time formula validation and preview
    if computation_formula:
        # Show suggestions
        st.markdown("### Column Suggestions")
        suggestions = suggest_columns(computation_formula, column_names, threshold=25)
        display_column_suggestions(suggestions)

        # Retrieve column names from the formula
        formula_valid, error_msg, used_columns, constants = validate_and_extract_columns_ast(computation_formula, column_names)
        
        if not formula_valid:
            st.error(f"Formula error: {error_msg}")

        if used_columns or constants:
            st.markdown("### Source Variables and Constants Preview")
        if used_columns:
            preview_data = main_data[used_columns].head()
            styled_preview = (preview_data.style
                            .format(precision=2)
                            .highlight_null(props='color: red;')
                            .set_table_styles([
                                {'selector': 'th', 'props': [('background-color', '#f0f2f6')]},
                                {'selector': 'td', 'props': [('padding', '8px')]}
                            ]))
            st.dataframe(styled_preview, width='stretch', hide_index=True)
        if constants:
            st.write(f"Constants list: {[extract_constant_value(constant) for constant in constants]}")
        
        
    
    # Initialize session state for computed variable preview
    if 'preview_new_variable' not in st.session_state:
        st.session_state.preview_new_variable = None

    # Compute button
    if st.button("Compute & Preview", disabled=not (variable_name and computation_formula)):
        # Final validation
        name_valid, name_error = validate_variable_name(variable_name, column_names)
        formula_valid, formula_error, used_columns, used_constants = validate_and_extract_columns_ast(computation_formula, column_names)
        
        if not name_valid:
            st.error(f"Invalid variable name: {name_error}")
        elif not formula_valid:
            st.error(f"Invalid formula: {formula_error}")
        else:
            try:
                # Compute the new variable
                succ, e, temp_main_data = evaluate_formula_safely(computation_formula, variable_name, main_data)
                
                if succ:
                    st.success(f"✨ Variable '{variable_name}' successfully computed! Review the preview below.")
                    
                    # Store preview info in session state
                    st.session_state.preview_new_variable = {
                        'name': variable_name,
                        'formula': computation_formula,
                        'used_columns': used_columns,
                        'temp_data': temp_main_data  # Storing for immediate preview
                    }
                else:
                    st.error(f"Computation failed: {e}")
            
            except Exception as e:
                st.error(f"Error computing variable: {str(e)}")
                st.info("Make sure your formula uses valid column names and operators (+, -, *, /, etc.).")

    # Display Preview and Add Button
    if st.session_state.preview_new_variable:
        preview_info = st.session_state.preview_new_variable
        
        # Check if the current inputs match the preview (optional, but good UX)
        if variable_name != preview_info['name'] or computation_formula != preview_info['formula']:
            st.warning("⚠️ Inputs have changed since last computation. Please click 'Compute & Preview' again.")
        else:
            st.write("### New Variable Preview")
            
            temp_data = preview_info['temp_data']
            var_name = preview_info['name']
            used_cols = preview_info['used_columns']
            
            # Create preview with source columns and new variable
            preview_columns = used_cols + [var_name]
            preview_df = temp_data[preview_columns].head()
            
            # Add statistics for the new variable
            stats_df = generate_stats(temp_data, var_name)
        
            # Display preview with source columns
            st.write("First rows with source columns:")
            styled_preview = (preview_df.style
                            .format(precision=2)
                            .highlight_null(props='color: red;')
                            .set_table_styles([
                                {'selector': 'th', 'props': [('background-color', '#f0f2f6')]},
                                {'selector': 'td', 'props': [('padding', '8px')]}
                            ]))
            st.dataframe(styled_preview, width='stretch', hide_index=True)
            
            # Display statistics
            st.write(f"Statistics for new variable :blue[{var_name}]:")
            st.dataframe(stats_df, width='stretch', hide_index=True)
            
            # Add Variable Button
            if st.button("Confirm & Add Variable to Dataset", type="primary"):
                main_data = temp_data.copy()
                
                # Trace the operation
                if 'transformation_manager' in st.session_state:
                    st.session_state.transformation_manager.add_step(
                        function_name="variable_computation",
                        params={
                            "variable_name": var_name,
                            "formula": preview_info['formula']
                        },
                        description=f"Computed new variable: {var_name}"
                    )
                
                st.success(f"✨ Variable '{var_name}' successfully added to the dataset!")
                st.session_state.preview_new_variable = None # Reset preview
                st.rerun()
    
    # Display the full dataset preview
    if st.checkbox("Show full dataset preview"):
        styled_df = (main_data.style
                    .format(precision=2)
                    .highlight_null(props='color: red;')
                    .set_table_styles([
                        {'selector': 'th', 'props': [('background-color', '#f0f2f6')]},
                        {'selector': 'td', 'props': [('padding', '8px')]}
                    ]))
        st.dataframe(styled_df, width='stretch')

    return main_data


# Safely merge transformed_data into main_data with replacement or ignoring duplicates
def merge_transformed_columns(main_data, transformed_data, handle_duplicates="replace"):
    """
    Merge transformed_data into main_data securely.

    Parameters:
    - main_data (pd.DataFrame): The original DataFrame.
    - transformed_data (pd.DataFrame): The DataFrame with new columns.
    - handle_duplicates (str): How to handle duplicate column names. 
                               Options: "replace" or "ignore".
    
    Returns:
    - pd.DataFrame: The merged DataFrame.
    """
    if handle_duplicates not in ["replace", "ignore"]:
        raise ValueError("handle_duplicates must be 'replace' or 'ignore'.")

    # Identify duplicate column names
    duplicate_columns = main_data.columns.intersection(transformed_data.columns)

    if not duplicate_columns.empty:
        if handle_duplicates == "replace":
            # Drop duplicate columns from main_data before merging
            main_data = main_data.drop(columns=duplicate_columns)
            st.info(f"Replaced duplicate columns: {list(duplicate_columns)}")
        elif handle_duplicates == "ignore":
            # Drop duplicate columns from transformed_data
            transformed_data = transformed_data.drop(columns=duplicate_columns)
            st.info(f"Ignored duplicate columns: {list(duplicate_columns)}")
    
    # Merge the DataFrames
    merged_data = pd.concat([main_data, transformed_data], axis=1)
    return merged_data



def compute_transformations(dataframe):
    st.subheader("Transformation and Aggregation")

    # Define available transformations
    TRANSFORMATIONS = {
        "Mean": lambda x: x.mean(),
        "Median": lambda x: x.median(),
        "Summation": lambda x: x.sum(),
        "Minimum Value": lambda x: x.min(),
        "Maximum Value": lambda x: x.max(),
        "Standard Deviation": lambda x: x.std(),
        "Natural Logarithm Transformation": lambda x: np.log(x + 1),
        "Min-Max Normalization": lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).flatten(),
        "Z-Score Standardization": lambda x: (x - x.mean()) / x.std(),
    }

    # Step 1: Transformation Type Selection
    transformation_type = st.selectbox(
        "Choose a transformation type:",
        options=["Statistical-Based", "Dimensional Reduction-Based", "Scaling-Based", "Cluster-Based", "Encoding"],
        help="Select the type of transformation."
    )

    # Step 2: Transformation Selection (based on transformation type)
    if transformation_type == "Statistical-Based":
        transformation = st.selectbox(
            "Choose a statistical transformation:",
            options=["Mean", "Median", "Summation", "Minimum Value", "Maximum Value", "Standard Deviation"],
            help="Select the statistical transformation to apply."
        )
    elif transformation_type == "Scaling-Based":
        transformation = st.selectbox(
            "Choose a scaling method:",
            options=["Min-Max Normalization", "Z-Score Standardization", "Natural Logarithm Transformation"],
            help="Select the scaling method."
        )
    elif transformation_type == "Dimensional Reduction-Based":
        # Define explanations for each dimensional reduction method
        transformation_explanations = {
            "PCA": ":grey_question: Principal Component Analysis (PCA) reduces dimensions by finding orthogonal directions (principal components) that explain the most variance in the dataset.",
            "t-SNE": ":grey_question: t-Distributed Stochastic Neighbor Embedding (t-SNE) is a non-linear technique that projects high-dimensional data into a lower-dimensional space, optimizing for local similarity.",
            "UMAP": ":grey_question: Uniform Manifold Approximation and Projection (UMAP) is a graph-based method that preserves both local and global structures of high-dimensional data.",
            "FAMD": ":grey_question: Factor Analysis of Mixed Data (FAMD) is designed for datasets containing both categorical and numerical variables, reducing dimensions while preserving mixed-data relationships."
        }
        
        transformation = st.selectbox(
            "Choose a dimensional reduction method:",
            options=["PCA", "t-SNE", "UMAP", "FAMD"],
            help="Select the dimensional reduction method."
        )
        
        # Display explanation for the selected transformation method
        st.info(transformation_explanations[transformation])

    elif transformation_type == "Cluster-Based":
        # Define explanations for each clustering method
        cluster_explanations = {
            "K-Means": ":grey_question: **K-Means** partitions data into K clusters by minimizing within-cluster distances. Creates a categorical variable indicating which cluster each sample belongs to.",
            "DBSCAN": ":grey_question: **DBSCAN** groups densely packed points, leaving sparse points as noise (-1). Good for finding clusters of arbitrary shapes and detecting outliers.",
            "Gaussian Mixture": ":grey_question: **Gaussian Mixture Model (GMM)** fits K Gaussian distributions to the data. Each component = one cluster. Provides soft (probabilistic) cluster assignments."
        }
        
        transformation = st.selectbox(
            "Choose a clustering algorithm:",
            options=["K-Means", "DBSCAN", "Gaussian Mixture"],
            help="Select the clustering algorithm to assign samples to groups."
        )
        # Display explanation for the selected method
        st.info(cluster_explanations[transformation])

    else:
        # Define explanations for each numerization method
        numerization_explanations = {
            "One-Hot Encoding": ":grey_question: One-Hot Encoding creates binary columns for each category in a categorical variable, ensuring no ordinal relationship is implied. Useful for non-ordinal categorical data.",
            "Label Encoding": ":grey_question: Label Encoding assigns a unique integer to each category. Suitable for ordinal or small categorical datasets but may introduce ordinal bias for non-ordinal data.",
            "Ordinal Encoding": ":grey_question: Ordinal Encoding assigns integers to categories based on a meaningful order. Use this for ordinal variables where the order between categories is significant."
        }
        transformation = st.selectbox(
            "Choose a Numerization method:",
            options=["One-Hot Encoding", "Label Encoding", "Ordinal Encoding"],
            help="Select the encoding method for categorical variables."
        )
        # Display explanation for the selected numerization method
        st.info(numerization_explanations[transformation])

    # Step 3: Column Selection based on transformation type
    if transformation_type == "Encoding":
        columns = st.multiselect(
            "Choose categorical columns to encode:",
            options=dataframe.select_dtypes(include=['object', 'category']).columns,
            help="Select one or more categorical columns from the dataset."
        )
        if transformation == "Ordinal Encoding":
            category_orders = {}

            for col in columns:
                st.write(f"Specify the order for column: {col}")
                category_orders[col] = st.multiselect(
                            f"Select the order for {col}:",
                            options=dataframe[col].unique(),
                            default=dataframe[col].unique(),
                            key=f"order_{col}"
                        )
    elif transformation_type == "Dimensional Reduction-Based" and transformation == "FAMD":
        columns = st.multiselect(
            "Choose columns to apply the selected transformation:",
            options=dataframe.select_dtypes(include=[ np.number, 'object', 'category']).columns,
            help="Select one or more columns from the dataset."
        )
    else:
        columns = st.multiselect(
            "Choose columns to apply the selected transformation:",
            options=dataframe.select_dtypes(include=np.number).columns,
            help="Select one or more columns from the dataset."
        )

    # Step 4: Naming Pattern Customization
    if transformation_type == "Dimensional Reduction-Based":
        naming_pattern_val = "Name_{method_applied}_Component"
    elif transformation_type == "Encoding":
        naming_pattern_val = "{initial_variable}_{category}"
    elif transformation_type == "Cluster-Based":
        naming_pattern_val = "Cluster_{method_applied}"
    else:
        naming_pattern_val = "{initial_variable}_{method_applied}"
    
    naming_pattern = st.text_input(
        "Enter a naming pattern for the new variables:",
        value=naming_pattern_val,
        help="Use placeholders: `{initial_variable}` for the original column name and `{method_applied}` for the selected transformation."
    )

    # Dimensional Reduction Settings
    if transformation_type == "Dimensional Reduction-Based" and len(columns) > 1:
        col1, col2, col3 = st.columns([1, 1, 3], vertical_alignment="center")

        with col1:
            st.markdown("### Reduction Settings")
            n_components = st.slider(
                "Select the number of components:",
                min_value=1,
                max_value=max(len(columns)-1, 2), 
                value=min(3, len(columns)-1),
                step=1,
                help="This determines how many dimensions the dataset will be reduced to. "
                    "The number of components should be less than or equal to the number of selected columns."
            )

            if transformation == "t-SNE":
                perplexity = st.slider(
                    "Perplexity:",
                    min_value=5,
                    max_value=50,
                    value=30,
                    help="Perplexity is a key parameter for t-SNE that defines the balance between local and global aspects of the data structure. "
                        "A higher perplexity considers more global patterns."
                )
                learning_rate = st.slider(
                    "Learning rate:",
                    min_value=10,
                    max_value=1000,
                    value=200,
                    help="The learning rate determines how fast the t-SNE optimization algorithm converges."
                )

            elif transformation == "UMAP":
                n_neighbors = st.slider(
                    "Number of neighbors:",
                    min_value=2,
                    max_value=100,
                    value=15,
                    help="Defines the size of the local neighborhood UMAP uses for manifold approximation. "
                        "Larger values emphasize global structure."
                )
                min_dist = st.slider(
                    "Minimum distance:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.1,
                    help="Controls the spacing of points in the low-dimensional space. "
                        "Smaller values create tighter clusters."
                )

        with col2:
            show_explained_variance = None
            if transformation in ["PCA", "FAMD"]:
                show_explained_variance = st.checkbox(
                    "Show Explained Variance",
                    value=False,
                    help="Display the amount of variance explained by each component. "
                        "Useful for understanding how much information is retained after reduction."
                )

        with col3:
            if show_explained_variance and transformation in ["PCA", "FAMD"]:
                
                if transformation == "PCA":
                    #scaler = StandardScaler()
                    #scaled_data = scaler.fit_transform(dataframe[columns])
                    #model = PCA()
                    #result = model.fit_transform(scaled_data)
                    #variance_ratio = result.explained_variance_ratio_

                    # Initialize the PCA model
                    model = prince.PCA(
                        n_components=len(columns),  # Number of principal components to retain
                        rescale_with_mean=True,     # Center the data by subtracting the mean
                        rescale_with_std=True,      # Scale the data by dividing by standard deviation
                        copy=True,                  # Do not modify the original data
                        check_input=True,           # Ensure input is valid
                        engine='sklearn'            # Select sklearn computation engine
                    )

                    # Fit the PCA model to the selected columns & transform the data into the lower-dimensional space
                    result = model.fit_transform(dataframe[columns])

                    variance_ratio = model.percentage_of_variance_

                else:  # FAMD
                    model = prince.FAMD(n_components=len(columns), random_state=42)
                    #famd = model.fit(dataframe[columns])
                    result = model.fit_transform(dataframe[columns])

                    # Get eigenvalues (variance explained by each dimension)
                    eigenvalues = model.eigenvalues_

                    # Compute proportion of variance explained
                    #total_variance = sum(eigenvalues)
                    variance_ratio = model.percentage_of_variance_#[eig / total_variance for eig in eigenvalues]

                cumulative_variance = np.cumsum(variance_ratio)

                # Plot Explained Variance
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o", linestyle="-", color="b")
                ax.set_ylim(0, max(cumulative_variance) * 1.1)
                ax.set_xlim(1, len(cumulative_variance))
                ax.vlines(n_components, 0, cumulative_variance[n_components - 1], color="red", linestyle="--", linewidth=2)
                ax.hlines(cumulative_variance[n_components - 1], 0, n_components, color="red", linestyle="--", linewidth=2)
                ax.text(n_components, cumulative_variance[n_components - 1], f"{cumulative_variance[n_components - 1]:.2f}", ha="center", va="bottom", fontsize=12, color="red")
                ax.set_title("Explained Variance by Components")
                ax.set_xlabel("Number of Components")
                ax.set_ylabel("Cumulative Explained Variance")
                ax.grid(True)
                st.pyplot(fig)

        if transformation in ["PCA", "FAMD"] and show_explained_variance:  
            st.subheader("Contributions to components")
            st.dataframe(styled_df, width='stretch')

    # Clustering Settings
    if transformation_type == "Cluster-Based":
        st.markdown("### Clustering Parameters")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if transformation == "K-Means":
                n_clusters = st.slider("Number of Clusters (K)", 2, 20, 3, key="clk_k")
            elif transformation == "DBSCAN":
                eps = st.slider("Epsilon (Neighborhood Size)", 0.1, 5.0, 0.5, 0.1, key="cldb_eps")
            elif transformation == "Gaussian Mixture":
                n_clusters = st.slider("Number of Components", 2, 20, 3, key="clgmm_k")
        with col_c2:
            if transformation == "DBSCAN":
                min_samples = st.slider("Min Samples", 2, 20, 5, key="cldb_min")

    # Apply Transformation Button
    if st.button("Apply Transformation", disabled=not (naming_pattern and columns)):
        if not columns:
            st.warning("Please select at least one column to proceed.")
            return

        new_columns = {}

        try:
            if transformation_type == "Dimensional Reduction-Based":
                # Standardize data
                if transformation in ["t-SNE", "UMAP"]:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(dataframe[columns])

                if transformation == "PCA":
                    #model = PCA(n_components=n_components)
                    model = prince.PCA(
                        n_components=n_components,  # Number of principal components to retain
                        rescale_with_mean=True,     # Center the data by subtracting the mean
                        rescale_with_std=True,      # Scale the data by dividing by standard deviation
                        copy=True,                  # Do not modify the original data
                        check_input=True,           # Ensure input is valid
                        engine='sklearn'            # Select sklearn computation engine
                    )
                    transformed_data = model.fit_transform(dataframe[columns])

                elif transformation == "t-SNE":
                    model = TSNE(n_components=n_components, perplexity=perplexity,
                                learning_rate=learning_rate, random_state=42)
                    transformed_data = model.fit_transform(scaled_data)
                elif transformation == "UMAP":
                    model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                                    min_dist=min_dist, random_state=42)
                    transformed_data = model.fit_transform(scaled_data)
                else:  # FAMD
                    model = prince.FAMD(n_components=n_components, random_state=42)
                    transformed_data = model.fit_transform(dataframe[columns])

                # Create column names and DataFrame
                component_names = [naming_pattern.format(method_applied=transformation.replace(" ", "_")) + f"_{i + 1}" for i in range(n_components)]

                if transformation in ["t-SNE", "UMAP"]:
                    result_df = pd.DataFrame(transformed_data, columns=component_names, index=dataframe.index)
                else:
                    # Ensure it's treated as a DataFrame (since FAMD already returns one)
                    result_df = transformed_data.copy()
                    result_df.columns = component_names

                
                # Convert to dictionary and update new_columns
                new_columns.update(result_df.to_dict(orient="list"))

            elif transformation_type == "Cluster-Based":
                from utils.clustering_utils import prepare_data_for_clustering, fit_kmeans, fit_dbscan, fit_gaussian_mixture
                
                # Prepare Data (handles standardization)
                numeric_cols = dataframe[columns].select_dtypes(include=np.number).columns.tolist()
                prep_data, valid_idx = prepare_data_for_clustering(dataframe[columns], numeric_cols)
                
                if len(prep_data) < 2:
                    st.error("❌ Not enough valid data points for clustering.")
                    return
                
                # Run Clustering
                if transformation == "K-Means":
                    labels, _ = fit_kmeans(prep_data, n_clusters)
                elif transformation == "DBSCAN":
                    labels, _ = fit_dbscan(prep_data, eps, min_samples)
                elif transformation == "Gaussian Mixture":
                    labels, _ = fit_gaussian_mixture(prep_data, n_clusters)
                
                # Map results back to original index
                full_labels = pd.Series(index=dataframe.index, data=np.nan)
                full_labels.loc[valid_idx] = labels
                
                col_name = naming_pattern.format(method_applied=transformation.replace(" ", "_"))
                new_columns[col_name] = full_labels.values


            elif transformation_type == "Encoding":
                if transformation == "One-Hot Encoding":
                    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                    for col in columns:
                        encoded_data = encoder.fit_transform(dataframe[[col]])
                        categories = encoder.categories_[0]
                        for i, category in enumerate(categories):
                            
                            new_col_name = naming_pattern.format(
                                initial_variable=col,
                                category=category
                            )
                            print(f"{new_col_name}:{encoded_data[:, i]}")
                            new_columns[new_col_name]  = encoded_data[:, i]
                
                elif transformation == "Label Encoding":
                    encoder = LabelEncoder()
                    for col in columns:
                        new_col_name = naming_pattern.format(
                            initial_variable=col,
                            category="label_encoded"
                        )
                        new_columns[new_col_name] = encoder.fit_transform(dataframe[col])
                
                elif transformation == "Ordinal Encoding":

                    for col in columns:
                        # Initialize the encoder with the specified category order for the current column
                        encoder = OrdinalEncoder(categories=[category_orders[col]])
                        new_col_name = naming_pattern.format(
                            initial_variable=col,
                            category="ordinal"
                        )
                        new_columns[new_col_name] = encoder.fit_transform(dataframe[[col]]).flatten()

            else:
                # Handle statistical and scaling transformations
                method_function = TRANSFORMATIONS.get(transformation, None)
                if method_function:
                    for col in columns:
                        new_col_name = naming_pattern.format(
                            initial_variable=col,
                            method_applied=transformation.replace(" ", "_")
                        )
                        result = method_function(dataframe[col])
                        # Ensure result is array-like
                        if np.isscalar(result):
                            result = [result] * len(dataframe)
                        new_columns[new_col_name] = result

            # Resolve duplicate column names
            final_new_columns = {}
            for col_name, data in new_columns.items():
                original_col_name = col_name
                counter = 1
                # Check against existing dataframe columns AND newly created columns
                while col_name in dataframe.columns or col_name in final_new_columns:
                    col_name = f"{original_col_name}_{counter}"
                    counter += 1
                final_new_columns[col_name] = data
            
            new_columns = final_new_columns

            # Create final result DataFrame
            if new_columns:
                result_df = pd.DataFrame(new_columns, index=dataframe.index)
                st.success("✨ Transformations applied successfully!")
                st.dataframe(result_df, width='stretch')

                # Update the session state
                new_df = pd.concat([dataframe, result_df], axis=1)
                st.session_state['enriched_df'] = new_df
                
                # Save snapshot
                snapshot_path = save_snapshot(new_df, "variable_transformation")

                # Log transformation
                params = {
                    "transformation_type": transformation_type,
                    "transformation": transformation,
                    "columns": columns,
                    "naming_pattern": naming_pattern,
                }
                
                # Add optional params if they exist in local scope
                if 'n_components' in locals(): params['n_components'] = n_components
                if 'perplexity' in locals(): params['perplexity'] = perplexity
                if 'learning_rate' in locals(): params['learning_rate'] = learning_rate
                if 'n_neighbors' in locals(): params['n_neighbors'] = n_neighbors
                if 'min_dist' in locals(): params['min_dist'] = min_dist
                if 'category_orders' in locals(): params['category_orders'] = category_orders
                if 'n_clusters' in locals(): params['n_clusters'] = n_clusters
                if 'eps' in locals(): params['eps'] = eps
                if 'min_samples' in locals(): params['min_samples'] = min_samples

                st.session_state.transformation_manager.add_step(
                    "variable_transformation", 
                    params, 
                    f"Applied {transformation}",
                    output_dataset_path=snapshot_path
                )

                return result_df
            else:
                st.warning("⚠️ No transformations were applied.")

        except Exception as e:
            st.error(f"Error applying transformation: {str(e)}")
            return None
        




# Helper function for controlled debug printing
def debug_print(*args, debug=False):
    if debug:
        print(*args)

# adapt mask size to transformation
def match_null_mask_to_encoded_columns(initial_null_mask, output_cols, original_cols):
    """
    Align initial null mask with output columns, including deducing one-hot encoded versions of categorical variables.
    Ensures the mask matches the order of output_cols.

    Parameters:
        initial_null_mask (pd.DataFrame): Original null mask for input data.
        output_cols (list): Ordered list of columns after imputation and encoding.
        original_cols (list): List of original columns from the input data.

    Returns:
        pd.DataFrame: Aligned null mask matching the column order in output_cols.
    """
    # Deduce categorical columns: they exist in original_cols but are split into new columns in output_cols
    cat_cols = [col for col in original_cols if col not in output_cols]

    # Initialize an empty mask with the same shape and order as output_cols
    aligned_mask = pd.DataFrame(False, index=initial_null_mask.index, columns=output_cols)
    
    for col in output_cols:
        # Check if the column corresponds to a one-hot encoded version of a categorical column
        matched_prefix = next((cat_col for cat_col in cat_cols if col.startswith(cat_col + "_")), None)
        
        if matched_prefix:
            # If matched, propagate the null mask of the original categorical column
            aligned_mask[col] = initial_null_mask[matched_prefix]
        elif col in initial_null_mask.columns:
            # If not encoded, directly use the original null mask
            aligned_mask[col] = initial_null_mask[col]
        else:
            # Default case: no match, keep as False (no nulls)
            aligned_mask[col] = False

    return aligned_mask


def apply_imputer(data, numerical_imputation_method='mean', categorical_imputation_method='most_frequent',
                  cat_encoder=False, num_scaler=False, remainder_columns=None, remainder_strategy='passthrough',
                  remainder_threshold=0.5, progress_placeholder=None, debug=False):
    """
    Applies an imputer to a pandas DataFrame, with options for numerical and categorical imputations, 
    and handling of remainder columns. Also tracks imputed values.
    
    Parameters:
        data (pd.DataFrame): The input data for imputation.
        numerical_imputation_method (str): Method for imputing numerical columns ('mean', 'median', 'knn', etc.).
        categorical_imputation_method (str): Method for imputing categorical columns ('most_frequent', 'missforest', etc.).
        cat_encoder (bool): Whether to apply one-hot encoding to categorical columns.
        num_scaler (bool): Whether to scale numerical columns.
        remainder_columns (list or str): Columns to pass through without transformation ('auto' for automatic detection).
        remainder_strategy (str): Strategy for handling remainder columns ('passthrough', 'drop', etc.).
        remainder_threshold (float): Threshold for automatic remainder column detection.
        progress_placeholder (object): Placeholder for showing progress in Streamlit.
        debug (bool): Flag to enable debug print statements.

    Returns:
        pd.DataFrame: The imputed DataFrame with updated column names.
        pd.DataFrame: A mask DataFrame indicating which values were imputed.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame with named columns.")

    debug_print("Starting imputation process...", debug=debug)

    # Record initial null values
    initial_null_mask = data.isna()
    if initial_null_mask.any().any():
        imputer, output_cols = edi.get_imputer(
            numerical_imputation_method=numerical_imputation_method.lower(),
            categorical_imputation_method=categorical_imputation_method.lower(),
            cat_encoder=cat_encoder,
            num_scaler=num_scaler,
            data=data,
            remainder_columns=remainder_columns,
            remainder_strategy=remainder_strategy,
            remainder_threshold=remainder_threshold,
            progress_placeholder=progress_placeholder,
            debug=debug
        )

        # Fit and transform the data
        debug_print("Fitting and transforming data with the imputer...", debug=debug)
        df_imputed = imputer.fit_transform(data)

        # Reconstruct the DataFrame with proper column names
        debug_print("Reconstructing the DataFrame with transformed columns...", debug=debug)
        df_imputed = pd.DataFrame(df_imputed, columns=output_cols, index=data.index)
        
        #adapt if a cat encoding output asked
        if cat_encoder:
            initial_null_mask = match_null_mask_to_encoded_columns(initial_null_mask, output_cols, data.columns.tolist())

        # Create an imputed values mask
        imputed_mask = initial_null_mask & df_imputed.notna()  # True for values that were imputed
        

        for col in df_imputed.columns:
            if df_imputed[col].dtype == 'object':
                original_na_count = df_imputed[col].isna().sum()
                debug_print(f"Processing column: '{col}'", debug=debug)
                debug_print(f"Original NA count: {original_na_count}", debug=debug)
                
                try:
                    # Attempt to convert to numeric, replacing ',' with '.' for floats
                    converted_col = pd.to_numeric(df_imputed[col], errors='coerce')
                    
                    # Check if conversion introduces additional NaNs
                    new_na_count = converted_col.isna().sum()
                    if new_na_count > original_na_count:
                        raise ValueError(f"Conversion introduced {new_na_count - original_na_count} additional NaN(s).")
                    
                    # Conversion successful
                    df_imputed[col] = converted_col
                    debug_print(f"Column '{col}' successfully converted to numeric.", debug=debug)
                
                except Exception as e:
                    # Revert to original type
                    df_imputed[col] = df_imputed[col].astype('object')
                    debug_print(f"Column '{col}' kept as object due to conversion issue: {e}", debug=debug)

        # Output debug information
        debug_print(f"Original DataFrame Columns: {data.columns.tolist()}", debug=debug)
        debug_print(f"Columns After Imputer Processing: {output_cols}", debug=debug)

        debug_print("Imputation process completed successfully.", debug=debug)
        
        return df_imputed, imputed_mask
    else:
        return data, initial_null_mask


def save_snapshot(df, step_name, base_dir="data/traces/snapshots"):
    """Saves a snapshot of the dataframe and returns the path."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_step = "".join([c for c in step_name if c.isalnum() or c=='_'])
    base_filename = f"{timestamp}_{safe_step}"
    
    df_path = os.path.join(base_dir, f"{base_filename}.parquet")
    try:
        # Ensure columns are strings for parquet compatibility
        df_to_save = df.copy()
        df_to_save.columns = df_to_save.columns.astype(str)
        df_to_save.to_parquet(df_path)
        return df_path
    except Exception as e:
        logging.error(f"Failed to save snapshot: {e}")
        return None


# TBD: Rework the cat of missingforest based on set(binary + cat)
def app():
    """Improved page for external data and variable definition."""
    st.title("Data Enrichment and Variable Definition")

    # Initialize DB Manager
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DBManager()

    # Initialize Transformation Manager
    if 'transformation_manager' not in st.session_state:
        st.session_state.transformation_manager = TransformationManager()

    # Dataset Management Sidebar (Automatic Versioning)
    with st.sidebar.expander("Dataset History", expanded=False):
        st.caption("Manage dataset versions")
        
        datasets = st.session_state.db_manager.get_available_datasets()
        if datasets:
            selected_dataset = st.selectbox("Select Dataset Version", datasets, index=0)
            if st.button("Load Selected Version"):
                df_loaded, msg = st.session_state.db_manager.load_dataset(selected_dataset)
                if df_loaded is not None:
                    st.session_state['data'] = df_loaded
                    st.session_state['working_df'] = df_loaded.copy()
                    st.session_state['enriched_df'] = df_loaded.copy()
                    st.session_state['current_dataset_name'] = selected_dataset
                    
                    # Initialize new session trace
                    st.session_state.transformation_manager.initialize_session(selected_dataset)
                    # Log initial load
                    st.session_state.transformation_manager.add_step(
                        "initial_load", 
                        {"dataset_name": selected_dataset}, 
                        f"Loaded dataset: {selected_dataset}"
                    )
                    
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        else:
            st.info("No saved datasets found.")

    # Reproduction Sidebar
    with st.sidebar.expander("Reproduce Analysis", expanded=False):
        st.caption("Reproduce analysis from a trace file")
        uploaded_trace = st.file_uploader("Upload Trace JSON", type=["json"])
        if uploaded_trace:
            from manage.reproduction_manager import reproduce_trace
            reproduce_trace(uploaded_trace)

    # Main navigation tabs
    main_tab, impute_tab, define_var_tab = st.tabs([
    "📥 Import External Data",
    "🛠️ Handle Missing Data",
    "⚙️ Create New Variables"
    ])

    if 'data' in st.session_state and st.session_state.data is not None:
        if 'enriched_df' not in st.session_state:
            st.session_state['enriched_df'] = st.session_state['data'].copy()
    # 1. Add External Data
    with main_tab:
        with st.expander("📥 Upload External Data", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Main Dataset")
                main_data = handle_main_data_upload()
                if main_data is not None:
                    st.dataframe(main_data.head(), width='stretch')
                    st.info(f"Rows: {len(main_data)} | Columns: {len(main_data.columns)}")

                    selected_columns_main = st.multiselect(
                        "Select columns to retain in Main Dataset",
                        options=main_data.columns.tolist(),
                        default=main_data.columns.tolist()
                    )
                    filtered_main_data = main_data[selected_columns_main]

            with col2:
                st.subheader("Enrichment Dataset")
                enrichment_data, input_file = handle_enrichment_data_upload()
                if enrichment_data is not None:
                    st.dataframe(enrichment_data.head(), width='stretch')
                    st.info(f"Rows: {len(enrichment_data)} | Columns: {len(enrichment_data.columns)}")

                    selected_columns_enrichment = st.multiselect(
                        "Select columns to retain in Enrichment Dataset",
                        options=enrichment_data.columns.tolist(),
                        default=enrichment_data.columns.tolist()
                    )
                    filtered_enrichment_data = enrichment_data[selected_columns_enrichment]

        with st.expander("🔄 Enrichment Configuration", expanded=False):
            if main_data is not None and enrichment_data is not None:
                enriched_df = configure_enrichment(
                    main_df=filtered_main_data,
                    enrichment_df=filtered_enrichment_data,
                    enrichment_file_path=input_file
                )
                if enriched_df is not None:
                    st.success("Enrichment successfully configured!")
                    st.session_state['enriched_df'] = enriched_df
            else:
                st.warning("Please upload both datasets first.")


        with st.expander("📊 Results", expanded=False):
            if 'enriched_df' in st.session_state and st.session_state['enriched_df'] is not None and 'filtered_main_data' in locals():
                display_external_data_results(st.session_state['enriched_df'], filtered_main_data)
            else:
                st.warning("No enrichment results available. Please complete the enrichment process first.")

    # 2. Handle Missing Values
    with impute_tab:
        with st.expander("Configure and apply imputation to your dataset", expanded=True):
            main_data = st.session_state['enriched_df']
            
            # --- Missing Data Overview ---
            missing_counts = main_data.isnull().sum()
            total_cells = np.prod(main_data.shape)
            total_missing = missing_counts.sum()
            missing_percent = (total_missing / total_cells) * 100
            
            col_metrics1, col_metrics2 = st.columns([1, 3])
            with col_metrics1:
                st.metric("Total Missing Values", f"{total_missing}", delta=f"{missing_percent:.2f}%")
            with col_metrics2:
                if missing_percent > 0:
                    st.info("💡 **Tip:** For complex datasets with correlated variables, **MICE** or **MissForest** often yield better results than simple Mean/Median imputation.")
            
            if missing_percent > 0:
                with st.expander("View Missing Values Details"):
                    st.dataframe(missing_counts[missing_counts > 0].rename("Missing Count"), width='stretch')
            else:
                st.success("✅ No missing values detected! You can skip this step unless you want to re-process.")

            # Parameter selection
            st.subheader("Select Imputation Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                numerical_imputation_method = st.selectbox(
                    "Numerical Imputation Method",
                    ["Mean", "Median", "KNN", "MICE", "MissForest"],
                    help="MICE (Multivariate Imputation by Chained Equations) models each feature with missing values as a function of other features."
                )
            with col2:
                categorical_imputation_method = st.selectbox(
                    "Categorical Imputation Method",
                    ["Most_Frequent", "MissForest"]
                )
            
            if categorical_imputation_method == numerical_imputation_method:
                st.info(f"ℹ️ {categorical_imputation_method} has been chosen for both numerical and categorical columns, a unified approach will be used.")
            
            col3, col4 = st.columns(2)
            with col3:
                cat_encoder = st.checkbox("Apply One-Hot Encoding to Categorical Columns?", value=False)
            with col4:
                num_scaler = st.checkbox("Scale Numerical Columns?", value=False)

            # --- Advanced Settings ---
            with st.expander("⚙️ Advanced Settings (Remainder Columns)"):
                remainder_option = st.radio(
                    "Handle Remainder Columns",
                    options=["Auto-detect", "Specify", "None"],
                    index=0
                )

                remainder_columns = None
                if remainder_option == "Specify":
                    remainder_columns = st.multiselect(
                        "Select Remainder Columns",
                        options=main_data.columns.tolist()
                    )
                elif remainder_option == "Auto-detect":
                    remainder_columns = "auto"

                remainder_strategy = st.selectbox(
                    "Remainder Strategy",
                    ["passthrough", "drop"]
                )
                remainder_threshold = st.slider(
                    "Remainder Threshold for Auto-detection",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.05,
                    help="Filter out non-numeric columns having a ratio distinct modalities / non empty rows higher than the fixed threshold"
                )

            # Apply imputation
            if st.button("Apply Imputer", type="primary"):
                try:
                    # Define a placeholder for the progress bar
                    progress_placeholder = st.empty()
                    # Call the imputer function
                    df_imputed, imputed_mask = apply_imputer(
                        data=main_data,
                        numerical_imputation_method=numerical_imputation_method,
                        categorical_imputation_method=categorical_imputation_method,
                        cat_encoder=cat_encoder,
                        num_scaler=num_scaler,
                        remainder_columns=remainder_columns,
                        remainder_strategy=remainder_strategy,
                        remainder_threshold=remainder_threshold,
                        progress_placeholder=progress_placeholder,
                        debug=True
                    )
                    if df_imputed is not None:
                        print(f"\n---------\n------\n --> Column dtypes after imputation:\n{df_imputed.dtypes}")

                        st.success("Imputation Applied Successfully!")
                        st.session_state['enriched_df'] = df_imputed
                        
                        # Save snapshot
                        snapshot_path = save_snapshot(df_imputed, "imputation")

                        # Log imputation step
                        params = {
                            "numerical_imputation_method": numerical_imputation_method,
                            "categorical_imputation_method": categorical_imputation_method,
                            "cat_encoder": cat_encoder,
                            "num_scaler": num_scaler,
                            "remainder_columns": remainder_columns,
                            "remainder_strategy": remainder_strategy,
                            "remainder_threshold": remainder_threshold
                        }
                        st.session_state.transformation_manager.add_step(
                            "imputation", 
                            params, 
                            "Applied data imputation",
                            output_dataset_path=snapshot_path
                        )

                        # Highlight imputed values
                        def highlight_imputed(data, mask):
                            """Apply highlighting to DataFrame based on imputed mask."""
                            styled = pd.DataFrame('', index=data.index, columns=data.columns)
                            styled[mask] = 'background-color: yellow'
                            return styled

                        styled_df = df_imputed.style.apply(highlight_imputed, mask=imputed_mask, axis=None)
                        st.write("Preview of Imputed Data:")
                        st.dataframe(styled_df)
                        #st.write(st.session_state['enriched_df'].info(verbose=True))
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        
        with st.expander("📊 Results", expanded=False):
            if 'enriched_df' in st.session_state and st.session_state['enriched_df'] is not None:
                display_imputation_results(st.session_state['enriched_df'], main_data=st.session_state['working_df'])

    # 3. Define New Variables
    with define_var_tab:
        with st.expander(":abacus: Apply Common Transformations to Multiple Selected Variables", expanded=True):
            
            transformed_data = compute_transformations(st.session_state['enriched_df'])
            # If transformations were successfully applied, merge the data
            if transformed_data is not None:
                if 'enriched_df' in st.session_state and st.session_state['enriched_df'] is not None:
                    # Add transformed columns to main_data
                    main_data = st.session_state['enriched_df']
                    st.success("Transformed data has been successfully added to the main dataset.")
                    st.subheader("Show Updated Dataset")
                    st.dataframe(main_data) #st.session_state['enriched_df'])
            else:
                st.warning("No transformation to apply.")
            

        with st.expander("⚙️ Variable Computation", expanded=True):
            if 'enriched_df' in st.session_state and st.session_state['enriched_df'] is not None:
                main_data = define_new_variables(main_data) #) st.session_state['enriched_df']
                #st.session_state['enriched_df'] = main_data
            else:
                st.warning("No enriched dataset")
            
        with st.expander("📊 Results", expanded=False):
            if 'enriched_df' in st.session_state and st.session_state['enriched_df'] is not None and main_data is not None:
                display_new_variables_results(main_data, main_data=st.session_state['working_df']) #st.session_state['enriched_df'] st.session_state['data']
                
                # st.dataframe(st.session_state['enriched_df'].head(), width='stretch')
                
                # col1, col2 = st.columns(2)
                
                # with col1:
                #     st.download_button(
                #         label="Download Dataset with New Variables",
                #         data=to_excel(st.session_state['enriched_df']),
                #         file_name="Dataset_with_New_Variables.xlsx",
                #         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                #         width='stretch'
                #     )
                # with col2:
                #     # Save to session state
                #     if st.button("Use This Dataset for Further Analysis ", width='stretch'):
                #         st.session_state.data = st.session_state['enriched_df']
                #         st.session_state.working_df = st.session_state['enriched_df']
                #         st.success("✅ Dataset saved and ready for further analysis!")
            else:
                st.warning("No dataset available. Please upload data first.")