import ast
import io
import json
import logging
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from rapidfuzz import fuzz

import enrich.external_data as eed
from app_pages.enrichment_mi import render_multiple_imputation
from core.expressions import safe_format
from core.population import catalogue
from enrich.formulas import (
    evaluate_formula_safely,
    extract_constant_value,
    far_outside,
    numeric_summary,
    validate_and_extract_columns_ast,
    validate_variable_name,
)
from enrich.imputation_service import impute_rows
from ingest.loaders import UPLOAD_TYPES
from manage.db_manager import DBManager
from ui.components import error_card, no_dataset
from ui.components.staging import render_draft_banner
from ui.state import get_state
from utils.data_analyzer import DataAnalyzer
from utils.data_paths import get_traces_dir
from utils.date_parser import smart_parse_dates
from utils.export_utils import to_excel
from utils.multipage import _load_excel_with_selection, get_file_hash, load_dataframe
from utils.optional import available_methods, method_available, missing_methods_message
from utils.session_scope import get_current_scope


def _workspace_path(path_text: str) -> str:
    """Only files inside the signed-in user's workspace may be referenced by path (audit E4)."""
    scope = get_current_scope()
    if scope is None:
        return path_text
    return str(scope.resolve_within(path_text))


def _artifact_dir(session_id: str) -> str:
    scope = get_current_scope()
    path = str(scope.artifacts_dir(session_id)) if scope else os.path.join(get_traces_dir(), "artifacts", session_id)
    os.makedirs(path, exist_ok=True)
    return path


logger = logging.getLogger(__name__)

# Note: DataTransformationEngine class was removed (dead code)
# See git history if needed

def _handle_multisheet_inline(uploaded_file, key_prefix: str) -> pd.DataFrame | None:
    """Handle multi-sheet Excel files inline with a selection UI.
    
    Persists the loaded DataFrame in session state so it survives Streamlit re-runs.
    Returns the loaded DataFrame, or None if the user hasn't made a selection yet.
    """
    # Check if we already loaded data for this prefix — return it immediately
    cache_key = f"{key_prefix}_loaded_df"
    cache_path_key = f"{key_prefix}_loaded_path"
    if cache_key in st.session_state and st.session_state[cache_key] is not None:
        st.success(f"Multi-sheet data loaded ({st.session_state[cache_key].shape[0]} rows × {st.session_state[cache_key].shape[1]} columns)")
        return st.session_state[cache_key]
    
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    
    xls = pd.ExcelFile(io.BytesIO(file_bytes), engine='openpyxl')
    sheet_names = xls.sheet_names
    
    st.markdown(f"**{len(sheet_names)} sheets detected:** {', '.join(sheet_names)}")
    
    load_mode = st.radio(
        "How would you like to load this file?",
        ["Load a single sheet", "Merge multiple sheets"],
        key=f"{key_prefix}_load_mode",
        horizontal=True
    )
    
    if load_mode == "Load a single sheet":
        selected_sheet = st.selectbox(
            "Select sheet to load",
            sheet_names,
            key=f"{key_prefix}_single_sheet"
        )
        if st.button("Load Sheet", key=f"{key_prefix}_btn_load_single"):
            df = _load_excel_with_selection(file_bytes, sheet_names, [selected_sheet])
            if df is not None:
                st.session_state[cache_key] = df
            return df
    else:
        selected_sheets = st.multiselect(
            "Select sheets to merge",
            sheet_names,
            default=sheet_names,
            key=f"{key_prefix}_multi_sheet"
        )
        if len(selected_sheets) >= 2:
            # Find common columns across selected sheets
            try:
                common_cols = None
                for s in selected_sheets:
                    cols = set(pd.read_excel(io.BytesIO(file_bytes), sheet_name=s, nrows=0, engine='openpyxl').columns)
                    common_cols = cols if common_cols is None else common_cols & cols
                common_cols = sorted(common_cols) if common_cols else []
            except Exception:
                common_cols = []
            
            if common_cols:
                merge_key = st.selectbox(
                    "Select merge key (row ID column)",
                    common_cols,
                    key=f"{key_prefix}_merge_key"
                )
            else:
                st.warning("No common columns found. Sheets will be concatenated vertically.")
                merge_key = None
            
            if st.button("Merge & Load Sheets", key=f"{key_prefix}_btn_merge"):
                df = _load_excel_with_selection(file_bytes, sheet_names, selected_sheets, merge_key)
                if df is not None:
                    st.session_state[cache_key] = df
                return df
        elif len(selected_sheets) == 1:
            st.info("Select at least 2 sheets to merge, or switch to single-sheet mode.")
        else:
            st.info("Please select at least one sheet.")
    return None


def handle_main_data_upload() -> pd.DataFrame | None:
    """The main dataset of the enrichment: the session dataset with the Enrich draft applied.

    Datasets are loaded from Data > Import or the sidebar (AppState.set_dataset), so every load starts a
    trace; this page no longer reads files of its own.
    """
    frame = get_state().working_frame()
    if frame is None:
        no_dataset()
        return None
    st.success("Main dataset: the current dataset, with the changes of the Enrich draft.")
    return frame

def handle_enrichment_data_upload() -> pd.DataFrame | str | None:
    """Handle the upload or path input for the enrichment dataset."""
    upload_method = st.radio("Choose upload method for Enrichment Dataset:", ["Upload File", "Enter File Path"], key='enrichment_data_method')
    
    if upload_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Drag and drop your enrichment dataset here",
            type=UPLOAD_TYPES,
            key='enrichment_data_uploader'
        )
        if uploaded_file is not None:
            try:
                st.write(f"file_name: {uploaded_file.name}")
                file_path = uploaded_file.name
                
                # Save to artifacts if session is active
                if 'transformation_manager' in st.session_state and st.session_state.transformation_manager.session_id:
                    session_id = st.session_state.transformation_manager.session_id
                    artifact_dir = _artifact_dir(session_id)
                    
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

                result = load_dataframe(uploaded_file)
                
                # Handle multi-sheet Excel
                if isinstance(result, dict) and result.get("__multi_sheet__"):
                    df = _handle_multisheet_inline(uploaded_file, "enrichment_data")
                    if df is None:
                        return None, None  # User hasn't selected yet
                else:
                    df = result
                
                if df is not None:
                    current_file_hash = get_file_hash(uploaded_file)
                    
                    if 'last_enrichment_file_hash' not in st.session_state or \
                       current_file_hash != st.session_state['last_enrichment_file_hash']:
                        st.session_state['last_enrichment_file_hash'] = current_file_hash
                        # Clear any stale multi-sheet cache from a previous file
                        st.session_state.pop('enrichment_data_loaded_df', None)
                        st.success("Enrichment dataset successfully loaded from upload.")
                    return df, file_path
            except Exception as e:
                st.error(f"Error loading file: {e!s}")
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
                df = load_dataframe(_workspace_path(file_path_input))
                st.success("Enrichment dataset successfully loaded from provided path.")
                return df, file_name
            except Exception as e:
                st.error(f"Error loading file: {e!s}")
                return None, None
    return None, None


def configure_enrichment(main_df: pd.DataFrame, enrichment_df: pd.DataFrame, enrichment_file_path: str) -> pd.DataFrame | None:
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
        
        # Preprocess enrichment data to match date format of the main dataset
        for col in enrichment_df.columns:
            if col in main_df.columns and pd.api.types.is_datetime64_any_dtype(main_df[col]):
                enrichment_df[col], _ = smart_parse_dates(enrichment_df[col])
        
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
                    params = {
                        "enrichment_file_path": enrichment_file_path,
                        "strategy": strategy,
                        "conflict_resolution": conflict_resolution,
                        "identifier": identifier,
                        "add_choice": add_choice,
                        "retained_main_columns": [str(c) for c in main_df.columns],
                        "retained_enrichment_columns": [str(c) for c in enrichment_df.columns],
                        "join_counts": enriched_df.attrs.get("join_counts"),
                    }
                    # Into the Enrich draft; traced when the draft is applied (AppState, audit E10)
                    get_state().stage(enriched_df, "enrichment", params, "Enriched with external data",
                                      output_dataset_path=snapshot_path)
                st.success("Enrichment completed successfully!")
            except Exception as e:
                st.error(f"Error performing enrichment: {e!s}")
    else:
        col1, col2 = st.columns(2)
        
        # Preprocess enrichment data to match date format of the main dataset
        for col in enrichment_df.columns:
            if col in main_df.columns and pd.api.types.is_datetime64_any_dtype(main_df[col]):
                enrichment_df[col], _ = smart_parse_dates(enrichment_df[col])

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
                if enriched_df is not None:
                    snapshot_path = save_snapshot(enriched_df, "enrichment")
                    params = {
                        "enrichment_file_path": enrichment_file_path,
                        "strategy": strategy,
                        "conflict_resolution": conflict_resolution,
                        "left_id_names": left_identifier,
                        "right_id_names": right_identifier,
                        "add_choice": add_choice,
                        "retained_main_columns": [str(c) for c in main_df.columns],
                        "retained_enrichment_columns": [str(c) for c in enrichment_df.columns],
                        "join_counts": enriched_df.attrs.get("join_counts"),
                    }
                    get_state().stage(enriched_df, "enrichment", params, "Enriched with external data",
                                      output_dataset_path=snapshot_path)
                st.success("Enrichment completed successfully!")
            except Exception as e:
                st.error(f"Error performing enrichment: {e!s}")
    
    return enriched_df


def display_category_box(title, columns):
    """
    Display a box containing column names for a specific category.

    Parameters:
    -----------
    title : str
        The title of the category box.
    columns : list
        List of column names to display.
    """
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
            state = get_state()
            applied, apply_msg = state.commit_staging()  # the Enrich draft becomes the dataset (traced)
            if not applied:
                st.warning(apply_msg)
                return
            st.success(apply_msg)

            # Automatic persistence
            if 'db_manager' in st.session_state:
                # Determine a base name based on context
                base_name = "cohort_data" 
                if "external" in key_base: base_name = "enriched_data"
                elif "imputation" in key_base: base_name = "imputed_data"
                elif "new_variables" in key_base: base_name = "calculated_data"
                
                success, msg, saved_name = state.save_version(base_name)
                if success:
                    
                    # Save a copy of the transformation trace next to the session traces (in the user's workspace)
                    if 'transformation_manager' in st.session_state:
                        saved, trace_msg = st.session_state.transformation_manager.save_trace(f"{saved_name}_trace.json")
                        if saved:
                            st.info(trace_msg)
                        else:
                            st.warning(f"The transformation trace could not be saved: {trace_msg}")

                    st.success(f"Dataset saved as {saved_name} and ready for analysis!")
                else:
                    st.error(f"Failed to save dataset: {msg}")
            else:
                 st.success("Dataset loaded to session (Persistence unavailable)!")


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


def tokenize_formula(formula: str) -> list[str]:
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
    logger.debug("formula tokens: %s", tokens)
    
    # Replace placeholders with the original quoted tokens
    for idx, token in enumerate(quoted_tokens):
        placeholder = f"__quoted_{idx}__"
        stripped_token = token[2:-2].strip()
        tokens = [stripped_token if t.strip() == placeholder else t.strip() for t in tokens]
    return tokens


def suggest_columns(computation_formula: str, column_names: list[str], threshold: int = 40, max_suggestions: int = 6) -> list[tuple[str, int]]:
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


def update_formula_callback(col, formula_key):
    """Callback to update formula when a suggestion is clicked."""
    if formula_key not in st.session_state:
        return

    current_formula = st.session_state[formula_key]
    
    # 1. format proper column string (quote if needed)
    if not re.match(r'^[a-zA-Z0-9_]+$', col):
        new_val = f'""{col}""'
    else:
        new_val = col

    # 2. Get the last token to replace
    tokens = tokenize_formula(current_formula)
    
    if tokens:
        last_token = tokens[-1]
        
        # 3. Replace the last occurrence of the token at the end of the string
        pattern = re.escape(last_token) + r"\s*$"
        
        if re.search(pattern, current_formula):
            new_formula = re.sub(pattern, new_val, current_formula)
            st.session_state[formula_key] = new_formula
        else:
            # Fallback
            st.session_state[formula_key] = current_formula + new_val
    else:
        # Empty formula case
        st.session_state[formula_key] = new_val

def display_column_suggestions(suggestions, formula_key="formula_input"):
    """
    Displays column suggestions as buttons. Clicking a button replaces the last token
    in the formula input with the selected column name.
    """
    # Dynamic CSS Generation for Score "Gradient"
    # We create a style rule for each range of scores (0-9, 10-19, ... 90-100)
    # Mapping 0->Red (Hue 0) to 100->Green (Hue 120) with pastel lightness
    
    gradient_css = """
    <style>
    /* Base Button Style */
    div.stButton > button[title*="Match Info]"] {
        text-align: left !important;
        display: block !important;
        width: 100% !important;
        height: auto !important;
        padding: 10px 15px !important;
        border: 1px solid rgba(0,0,0,0.1) !important;
        border-radius: 6px !important;
        box-shadow: 0px 1px 2px rgba(0,0,0,0.05) !important;
        color: #2c3e50 !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
        position: relative;
    }
    
    div.stButton > button[title*="Match Info]"]:hover {
        transform: translateY(-1px);
        box-shadow: 0px 3px 6px rgba(0,0,0,0.15) !important;
        filter: brightness(95%);
    }

    /* Target the internal text */
    div.stButton > button[title*="Match Info]"] p {
        font-size: 14px !important;
        line-height: 1.4 !important;
    }
    """
    
    # Generate CSS for each score tier (steps of 10)
    # [Score: 9] covers 90-99, [Score: 10] covers 100
    for i in range(11):
        # Calculate Hue: 0 (Red) -> 120 (Green)
        # i is 0..10. score roughly i*10. hue = i * 12
        hue = i * 12
        
        # Pastel background: Lightness ~90%, Saturation ~80%
        bg_color = f"hsl({hue}, 80%, 90%)"
        border_color = f"hsl({hue}, 60%, 80%)"
        
        # Selector for help text starting with [Score: i]
        # We use a specific tag like [Tier: i]
        gradient_css += f"""
        div.stButton > button[title^="[Tier: {i}]"] {{
            background-color: {bg_color} !important;
            border-color: {border_color} !important;
        }}
        """
        
    gradient_css += "</style>"
    st.markdown(gradient_css, unsafe_allow_html=True)

    if suggestions:
        cols = st.columns(min(3, len(suggestions)))
        
        for idx, (col, score) in enumerate(suggestions):
            # Calculate Tier (0 to 10)
            # 100 -> 10, 95 -> 9, 83 -> 8, ...
            tier = int(score / 10)
            
            # Icons just for the label
            with cols[idx % 3]:
                # Label
                label = f"{col}  \nScore: {score}%"
                
                # Help text starts with [Tier: X] to trigger the CSS
                help_text = f"[Tier: {tier}] [Match Info] Click to insert '{col}' into formula"
                
                # Button
                st.button(
                    label, 
                    key=f"suggest_btn_{idx}_{col}", 
                    help=help_text,
                    on_click=update_formula_callback,
                    args=(col, formula_key),
                    width="stretch"
                )
    else:
        st.info("No matching columns found. Try typing part of a column name.")


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
  

def render_new_variable_statistics(temp_data, var_name, used_cols):
    """
    Statistics of a new variable on the analysis population, robust (median, quartiles) and classical (mean, SD),
    and the values far outside the others with their source values: a few entry errors (a height in metres, a
    weight of 0) can move the mean and the SD far from what the median shows.
    """
    state = get_state()
    population = state.population
    rows, scope = temp_data.index, f"all {len(temp_data):,} rows"
    if population is not None and population.ok and population.criteria:
        rows = temp_data.index.intersection(population.included)
        scope = f"the analysis population ({population.label})"
    values = temp_data.loc[rows, var_name]
    st.write(f"Statistics for new variable :blue[{var_name}] on {scope}:")
    if not pd.api.types.is_numeric_dtype(values):
        st.dataframe(generate_stats(temp_data.loc[rows], var_name), width='stretch', hide_index=True)
        return
    st.dataframe(numeric_summary(values), width='stretch', hide_index=True)
    extreme, low, high = far_outside(values)
    if not len(extreme):
        return
    numbers = pd.to_numeric(values, errors="coerce")
    st.warning(
        f"{len(extreme):,} value(s) of {var_name} lie far outside the others (beyond 3 interquartile ranges from the "
        f"quartiles: below {low:.4g} or above {high:.4g}). They move the mean and the SD; the median and the quartiles "
        "show the typical values. Check their source values below: entry errors (for example a height in metres or "
        "a weight of 0) are best excluded with a clinical-anomaly criterion (Validate).",
        icon=":material/warning:",
    )
    distance = (numbers.loc[extreme] - numbers.median()).abs().sort_values(ascending=False)
    shown = temp_data.loc[distance.index[:10], [*used_cols, var_name]]
    st.dataframe(shown.style.format(precision=2), width='stretch')
    unticked = [c for c in catalogue(state.config) if c.id not in set(state.ticked)]
    if unticked and not (population is not None and population.criteria):
        st.caption(
            "No inclusion or exclusion criterion is ticked: these statistics use every row. Tick the clinical "
            "anomaly criteria in the sidebar to leave entry errors out of the analyses."
        )


def define_new_variables(main_data):
    """
    Enhanced version of define_new_variables with improved preview and validation.
    """
    if main_data is None:
        st.warning("Please load data first.")
        return main_data
    
    st.subheader("Define New Variable")

    # AI Suggestions Section
    with st.expander("AI Variable Suggestions (RAG)", expanded=False):
        if 'rag_manager' in st.session_state and st.session_state.rag_manager.initialized:
            
            use_specific = st.toggle("Specific Search", value=True)

            # -- ROW 1: Search & Strategy --
            col_input, col_mode = st.columns([1, 1], gap="medium")
            
            with col_input:
                if use_specific:
                    search_hint = st.text_input("Concept Search:", placeholder="e.g. 'BSA' or 'Diabetes Risk'", help="Enter a medical concept, variable name, or clinical score you want to find.")
                else:
                    search_hint = None
                    st.info("**Exploratory Mode**: The AI will scan your dataset to suggest generally useful medical scores (e.g. BMI, Charlson) based on available variables.")

            with col_mode:
                if use_specific:
                    suggestion_mode = st.selectbox(
                        "Suggestion Strategy",
                        ["Go To Target", "Go From Target", "Around Target"],
                        help="**Go To Target:** Find formulas to compute this concept (Target = Output).\n**Go From Target:** Find scores that use this concept (Target = Input).\n**Around Target:** Find proxies/surrogates."
                    )
                else:
                    suggestion_mode = "Comprehensive"
                    st.write("") # Spacer
                    st.caption("Standard Medical Indices")

            # -- ROW 2: Configuration --
            with st.expander("Advanced Configuration", expanded=True):
                col_conf_1, col_conf_2 = st.columns([1, 1])
                
                with col_conf_1:
                    num_suggestions = st.slider("Number of Suggestions", min_value=1, max_value=10, value=5)
                    restrict_cols = st.multiselect(
                        "Restrict to specific variables (Optional)", 
                        options=list(main_data.columns),
                        help="Force AI to only use these variables."
                    )

                with col_conf_2:
                    st.write("**Analysis Options:**")
                    allow_missing = st.checkbox("Include suggestions with missing variables", value=False)
                    use_taxonomy = st.checkbox("Use Variable Taxonomy (Better understanding)", value=True)
                    debug_mode = st.checkbox("Debug Mode (Show Raw Output)", value=False)

            # -- ROW 3: Action --
            st.write("")
            if st.button("Generate AI Suggestions", type="primary", width="stretch"):
                progress_bar = st.progress(0, text="Starting analysis...")
                
                def update_progress(percent, text):
                    progress_bar.progress(percent, text=text)

                try:
                    # Determine which columns to use
                    cols_to_use = list(main_data.columns)
                    filtering_msg = None
                    
                    if restrict_cols:
                        cols_to_use = restrict_cols
                    elif len(cols_to_use) > 50:
                        # SMART FILTERING (User asked for "lighter treatment")
                        # If dataset is large and no restriction is set, filter out likely IDs/spam
                        try:
                            analyzer = DataAnalyzer(main_data)
                            
                            # Candidates to keep: Numeric, Date, Binary, Low/Mid Cardinality Categorical
                            keep_cols = set(analyzer.numeric_cols + analyzer.date_cols + analyzer.binary_cols + analyzer.low_cardinality_numeric_cols)
                            
                            # Add decent categorical columns (not unique identifiers)
                            for col in analyzer.categorical_cols:
                                if col not in analyzer.high_cardinality_cat_cols:
                                    keep_cols.add(col)
                            
                            filtered_cols = [c for c in cols_to_use if c in keep_cols]
                            
                            # If we filtered out something, use the cleaner list
                            if len(filtered_cols) < len(cols_to_use) and len(filtered_cols) > 5:
                                removed_count = len(cols_to_use) - len(filtered_cols)
                                cols_to_use = filtered_cols
                                filtering_msg = f"Automatically filtered {removed_count} irrelevant columns (IDs, unique strings) to improve AI focus."
                        except Exception:
                            pass # Fallback to all columns if analysis fails

                    if filtering_msg:
                        st.info(filtering_msg)
                    
                    # Use validation method to ensure quality and filter useless suggestions
                    suggestions_text, error = st.session_state.rag_manager.suggest_computed_variables_with_validation(
                        cols_to_use,
                        search_hint=search_hint,
                        num_suggestions=num_suggestions,
                        suggestion_mode=suggestion_mode,
                        allow_missing_variables=allow_missing,
                        use_taxonomy=use_taxonomy,
                        progress_callback=update_progress
                    )
                    progress_bar.empty()
                    
                    if debug_mode:
                        with st.expander("Debug: Raw LLM Output", expanded=True):
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
            if st.session_state.get('ai_domain_analysis'):
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
                                    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                                        inputs.add(node.value)
                                
                                # Add explicitly missing variables from RAG suggestion
                                if 'missing_variables' in sugg:
                                    inputs.update(sugg['missing_variables'])

                                has_inputs = False
                                for inp in inputs:
                                    if inp in ['np', 'pd', 'log', 'exp', 'sqrt', 'abs', 'min', 'max', 'constant']: continue
                                    has_inputs = True
                                    if inp in main_data.columns:
                                        dot_code += f'  "{inp}" [shape=ellipse, style=filled, fillcolor="#f0f4c3"];\n'
                                    else:
                                        # Use a distinct style for missing inputs as requested
                                        dot_code += f'  "{inp}" [label="{inp}\\n(Missing)", shape=ellipse, style="dashed,filled", fillcolor="#ffcdd2", color="red", fontcolor="red"];\n'
                                    dot_code += f'  "{inp}" -> "{target_name}";\n'
                                
                                if not has_inputs:
                                     dot_code += '  "No Inputs" [shape=plaintext];\n'
                            except Exception:
                                dot_code += '  "Error" [shape=plaintext];\n'
                            
                            dot_code += '}'
                            st.graphviz_chart(dot_code, width='stretch')

                        with col_formula:
                            if sugg.get('markdown_formula'):
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
                                st.error(f"Missing required variables: {', '.join(missing_vars)}")
                                
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
                                    if st.button("Find Alternative Formula", key=f"alt_{i}_{sugg.get('name', 'unknown')}"):
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
                            st.markdown("**Source Information**")
                            
                            if source_type == "Document":
                                st.info(f"**Source:** {citation}", icon="")
                                if explanation:
                                    st.caption(f"_{explanation}_")
                                    
                            elif source_type == "Hybrid":
                                st.warning(f"**Hybrid Source:** {citation}", icon="")
                                st.markdown(f"**Logic:** {explanation}")
                                
                            elif source_type == "Model Knowledge":
                                st.info(f"**Model Knowledge:** {explanation}", icon="")
                                st.caption(
                                    "Recalled by the language model, not taken from your documents: compare its "
                                    "constants and exponents with the published reference before you use it."
                                )

                            else:
                                st.info(explanation or citation)

                        if st.button("Apply to Editor", key=f"apply_var_{i}_{sugg.get('name', 'unknown')}", width='stretch'):
                            st.session_state.variable_name_input = sugg.get('name', '')
                            st.session_state.formula_input = sugg.get('formula', '')
                            st.rerun()
                st.markdown("---")

        else:
            st.info("The AI engine is not initialised. Initialise it in Settings > AI engine.")
    
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
                succ, e, temp_main_data = evaluate_formula_safely(computation_formula, variable_name, main_data.copy())
                
                if succ:
                    st.success(f"Variable '{variable_name}' successfully computed! Review the preview below.")
                    
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
                st.error(f"Error computing variable: {e!s}")
                st.info("Make sure your formula uses valid column names and operators (+, -, *, /, etc.).")

    # Display Preview and Add Button
    if st.session_state.preview_new_variable:
        preview_info = st.session_state.preview_new_variable
        
        # Check if the current inputs match the preview (optional, but good UX)
        if variable_name != preview_info['name'] or computation_formula != preview_info['formula']:
            st.warning("Inputs have changed since last computation. Please click 'Compute & Preview' again.")
        else:
            st.write("### New Variable Preview")
            
            temp_data = preview_info['temp_data']
            var_name = preview_info['name']
            used_cols = preview_info['used_columns']
            
            # Create preview with source columns and new variable
            preview_columns = used_cols + [var_name]
            preview_df = temp_data[preview_columns].head()
            
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

            # Statistics on the rows the analyses use (the variable itself is computed for every row)
            render_new_variable_statistics(temp_data, var_name, used_cols)
            
            # Add Variable Button
            if st.button("Confirm & Add Variable to Dataset", type="primary"):
                # Into the Enrich draft; traced when the draft is applied (AppState)
                get_state().stage(
                    main_data.assign(**{var_name: temp_data[var_name]}),
                    "variable_computation",
                    {"variable_name": var_name, "formula": preview_info['formula']},
                    f"Computed new variable: {var_name}",
                )
                
                st.success(f"Variable '{var_name}' successfully added to the dataset!")
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
    # Local imports: heavy ML libraries are only needed in this section. UMAP and prince (FAMD) are in the optional
    # 'ml' extra: when missing, their methods are not offered (utils/optional.py).
    import importlib

    prince = importlib.import_module("prince") if method_available("FAMD") else None
    umap = importlib.import_module("umap") if method_available("UMAP") else None
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import (
        LabelEncoder,
        MinMaxScaler,
        OneHotEncoder,
        OrdinalEncoder,
        StandardScaler,
    )

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
        options=["Statistics", "Dimensionality Reduction", "Scaling", "Clustering", "Encoding"],
        help="Select the type of transformation."
    )

    # Step 2: Transformation Selection (based on transformation type)
    if transformation_type == "Statistics":
        transformation = st.selectbox(
            "Choose a statistical transformation:",
            options=["Mean", "Median", "Summation", "Minimum Value", "Maximum Value", "Standard Deviation"],
            help="Select the statistical transformation to apply."
        )
    elif transformation_type == "Scaling":
        transformation = st.selectbox(
            "Choose a scaling method:",
            options=["Min-Max Normalization", "Z-Score Standardization", "Natural Logarithm Transformation"],
            help="Select the scaling method."
        )
    elif transformation_type == "Dimensionality Reduction":
        # Define explanations for each dimensional reduction method
        transformation_explanations = {
            "PCA": ":grey_question: Principal Component Analysis (PCA) reduces dimensions by finding orthogonal directions (principal components) that explain the most variance in the dataset.",
            "t-SNE": ":grey_question: t-Distributed Stochastic Neighbor Embedding (t-SNE) is a non-linear technique that projects high-dimensional data into a lower-dimensional space, optimizing for local similarity.",
            "UMAP": ":grey_question: Uniform Manifold Approximation and Projection (UMAP) is a graph-based method that preserves both local and global structures of high-dimensional data.",
            "FAMD": ":grey_question: Factor Analysis of Mixed Data (FAMD) is designed for datasets containing both categorical and numerical variables, reducing dimensions while preserving mixed-data relationships."
        }
        
        reduction_methods, reduction_missing = available_methods(["PCA", "t-SNE", "UMAP", "FAMD"])
        transformation = st.selectbox(
            "Choose a dimensional reduction method:",
            options=reduction_methods,
            help="Select the dimensional reduction method."
        )
        if reduction_missing:
            st.caption(missing_methods_message(reduction_missing))
        
        # Display explanation for the selected transformation method
        st.info(transformation_explanations[transformation])

    elif transformation_type == "Clustering":
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
    elif transformation_type == "Dimensionality Reduction" and transformation == "FAMD":
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
    if transformation_type == "Dimensionality Reduction":
        naming_pattern_val = "Name_{method_applied}_Component"
    elif transformation_type == "Encoding":
        naming_pattern_val = "{initial_variable}_{category}"
    elif transformation_type == "Clustering":
        naming_pattern_val = "Cluster_{method_applied}"
    else:
        naming_pattern_val = "{initial_variable}_{method_applied}"
    
    naming_pattern = st.text_input(
        "Enter a naming pattern for the new variables:",
        value=naming_pattern_val,
        help="Use placeholders: `{initial_variable}` for the original column name and `{method_applied}` for the selected transformation."
    )

    # Dimensional Reduction Settings
    if transformation_type == "Dimensionality Reduction" and len(columns) > 1:
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
                    column_correlations = model.column_correlations

                else:  # FAMD
                    model = prince.FAMD(
                        n_components=len(columns),
                        n_iter=3,
                        copy=True,
                        check_input=True,
                        random_state=42,
                        engine="sklearn",
                        handle_unknown="error"
                    )
                    result = model.fit_transform(dataframe[columns])

                    # Get eigenvalues (variance explained by each dimension)
                    eigenvalues = model.eigenvalues_

                    # Compute proportion of variance explained
                    #total_variance = sum(eigenvalues)
                    variance_ratio = model.percentage_of_variance_#[eig / total_variance for eig in eigenvalues]
                    
                    # Use native column_contributions_ for FAMD
                    if hasattr(model, 'column_contributions_'):
                        column_correlations = model.column_contributions_
                        # Normalize to ensure they represent percentages of contribution (0-1 range)
                        column_correlations = column_correlations / column_correlations.sum(axis=0)
                    else:
                        try:
                            column_correlations = model.column_correlations(dataframe[columns])
                        except NotImplementedError:
                            column_correlations = None

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

                # Define styled_df for display
                styled_df = None
                if column_correlations is not None:
                    # Adapt styling for contributions (0-1) vs correlations (-1 to 1)
                    if transformation == "FAMD":
                         styled_df = (column_correlations.style
                                .background_gradient(cmap="Blues", vmin=0, vmax=1)
                                .format("{:.0%}"))
                    else:
                        styled_df = (column_correlations.style
                                .background_gradient(cmap="RdBu", vmin=-1, vmax=1)
                                .format(precision=2))

        if transformation in ["PCA", "FAMD"] and show_explained_variance:  
            st.subheader("Contributions to components")
            if styled_df is not None:
                st.dataframe(styled_df, width='stretch')
            else:
                st.info(f"Component contributions are not available for {transformation}.")

    # Clustering Settings
    if transformation_type == "Clustering":
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
        
        st.info("""
        **Methodological Note:**
        - **Standardization**: Input variables will be automatically standardized (Z-score) to ensure equal weight in distance calculations.
        - **Missing Values**: The selected columns must be complete: clustering stops when one of them has a missing value. Impute the missing values first, or choose complete columns.
        """)

    # Apply Transformation Button
    if st.button("Apply Transformation", disabled=not (naming_pattern and columns)):
        if not columns:
            st.warning("Please select at least one column to proceed.")
            return

        new_columns = {}

        # CHECK FOR MISSING VALUES (Elegant Error Handling)
        if transformation_type in ["Dimensionality Reduction", "Clustering"]:
             if dataframe[columns].isnull().any().any():
                 st.error("**Input columns contain missing values (NaNs).**\n\n"
                          "Dimensionality reduction (PCA, t-SNE, etc.) and Clustering require complete data to function correctly.\n\n"
                          "**Action Required:** Please go to the **Handle Missing Data** tab to impute these missing values before proceeding.")
                 return

        try:
            if transformation_type == "Dimensionality Reduction":
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
                component_names = [safe_format(naming_pattern, method_applied=transformation.replace(" ", "_")) + f"_{i + 1}" for i in range(n_components)]

                if transformation in ["t-SNE", "UMAP"]:
                    result_df = pd.DataFrame(transformed_data, columns=component_names, index=dataframe.index)
                else:
                    # Ensure it's treated as a DataFrame (since FAMD already returns one)
                    result_df = transformed_data.copy()
                    result_df.columns = component_names

                
                # Convert to dictionary and update new_columns
                new_columns.update(result_df.to_dict(orient="list"))

            elif transformation_type == "Clustering":
                from utils.clustering_utils import (
                    fit_dbscan,
                    fit_gaussian_mixture,
                    fit_kmeans,
                    prepare_data_for_clustering,
                )
                
                # Prepare Data (handles standardization)
                numeric_cols = dataframe[columns].select_dtypes(include=np.number).columns.tolist()
                prep_data, valid_idx = prepare_data_for_clustering(dataframe[columns], numeric_cols)
                
                if len(prep_data) < 2:
                    st.error("Not enough valid data points for clustering.")
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
                
                col_name = safe_format(naming_pattern, method_applied=transformation.replace(" ", "_"))
                new_columns[col_name] = full_labels.values


            elif transformation_type == "Encoding":
                if transformation == "One-Hot Encoding":
                    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                    for col in columns:
                        encoded_data = encoder.fit_transform(dataframe[[col]])
                        categories = encoder.categories_[0]
                        for i, category in enumerate(categories):
                            
                            new_col_name = safe_format(naming_pattern, 
                                initial_variable=col,
                                category=category
                            )
                            new_columns[new_col_name]  = encoded_data[:, i]
                
                elif transformation == "Label Encoding":
                    encoder = LabelEncoder()
                    for col in columns:
                        new_col_name = safe_format(naming_pattern, 
                            initial_variable=col,
                            category="label_encoded"
                        )
                        new_columns[new_col_name] = encoder.fit_transform(dataframe[col])
                
                elif transformation == "Ordinal Encoding":

                    for col in columns:
                        # Initialize the encoder with the specified category order for the current column
                        encoder = OrdinalEncoder(categories=[category_orders[col]])
                        new_col_name = safe_format(naming_pattern, 
                            initial_variable=col,
                            category="ordinal"
                        )
                        new_columns[new_col_name] = encoder.fit_transform(dataframe[[col]]).flatten()

            else:
                # Handle statistical and scaling transformations
                method_function = TRANSFORMATIONS.get(transformation, None)
                if method_function:
                    for col in columns:
                        new_col_name = safe_format(naming_pattern, 
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
                st.success("Transformations applied successfully!")
                st.dataframe(result_df, width='stretch')

                new_df = pd.concat([dataframe, result_df], axis=1)

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

                # Into the Enrich draft; traced when the draft is applied
                get_state().stage(new_df, "variable_transformation", params, f"Applied {transformation}",
                                  output_dataset_path=snapshot_path)

                return result_df
            else:
                st.warning("No transformations were applied.")

        except Exception as e:
            st.error(f"Error applying transformation: {e!s}")
            return None
        




def save_snapshot(df, step_name, base_dir=None):
    """Saves a snapshot of the dataframe in the user's workspace and returns the path."""
    import uuid

    if base_dir is None:
        scope = get_current_scope()
        base_dir = str(scope.snapshots_dir()) if scope else os.path.join(get_traces_dir(), "snapshots")
    os.makedirs(base_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_step = "".join([c for c in step_name if c.isalnum() or c=='_'])
    base_filename = f"{timestamp}_{safe_step}_{uuid.uuid4().hex[:8]}"
    
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


# Helper to extract statistics for context columns (Guided Mode)
def extract_column_stats(df, columns):
    """
    Extracts summary statistics (Min, Max, Median) and potential units for selected columns.
    Used to give context to the RAG model for unit conversion logic.
    """
    stats = {}
    for col in columns:
        if col not in df.columns:
            continue
            
        col_data = df[col].dropna()
        if col_data.empty:
            stats[col] = {"status": "empty"}
            continue

        if pd.api.types.is_numeric_dtype(col_data):
            stats[col] = {
                "type": "Numeric",
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "median": float(col_data.median()),
                "q1": float(col_data.quantile(0.25)),
                "q3": float(col_data.quantile(0.75))
            }
        else:
            # For categorical/object, show top values which might contain units like "mg/dL"
            top_vals = col_data.value_counts().head(5).to_dict()
            stats[col] = {
                "type": "Categorical/Object",
                "top_values": {str(k): v for k, v in top_vals.items()}
            }
    return stats


# TBD: Rework the cat of missingforest based on set(binary + cat)
def app():
    """
    Main application function of the Enrich page (Data > Enrich).
    
    Handles:
    1. External Data Upload and Enrichment
    2. Targeted Imputation (Manual/AI-assisted)
    3. Global Imputation (MICE, KNN, etc.)
    4. Dataset Versioning and Persistence
    """

    # Initialize DB Manager
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DBManager()

    # The transformation trace belongs to the session (AppState)
    get_state().transformation_manager

    # Initialize RAG Manager if not present
    if 'rag_manager' not in st.session_state:
        from manage.rag_manager import RAGManager
        st.session_state.rag_manager = RAGManager()

    # Dataset versions are managed from the shell sidebar (ui/shell.py).

    # Trace replay and reports live on the Reproduce page (single replay engine, audit E10).

    # The Enrich draft banner sits above the tabs. It is filled at the end of the run, so that it also lists the
    # changes staged by this run (rendered after the tabs, it used to appear at the bottom of the page).
    draft_slot = st.container()

    # Main navigation tabs
    main_tab, impute_tab, define_var_tab = st.tabs([
    "Import External Data",
    "Handle Missing Data",
    "Create New Variables"
    ])
    # 1. Add External Data
    with main_tab:
        with st.expander("Upload External Data", expanded=True):
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

        with st.expander("Enrichment Configuration", expanded=False):
            if main_data is not None and enrichment_data is not None:
                enriched_df = configure_enrichment(
                    main_df=filtered_main_data,
                    enrichment_df=filtered_enrichment_data,
                    enrichment_file_path=input_file
                )
                if enriched_df is not None:
                    st.success("Enrichment added to the Enrich draft: apply the draft (top of the page) to use it in the "
                               "analyses.")
            else:
                st.warning("Please upload both datasets first.")


        with st.expander("Results", expanded=False):
            if 'enriched_df' in st.session_state and st.session_state['enriched_df'] is not None and 'filtered_main_data' in locals():
                display_external_data_results(st.session_state['enriched_df'], filtered_main_data)
            else:
                st.warning("No enrichment results available. Please complete the enrichment process first.")

    # 2. Handle Missing Values
    # 2. Handle Missing Values
    with impute_tab:
        if st.session_state.get('enriched_df') is None:
            _NO_DATA = "No dataset loaded. Upload a dataset from the sidebar or load a saved version first."
            no_dataset()
            with define_var_tab:
                no_dataset()
            return

        tab_targeted, tab_global, tab_multiple = st.tabs([
            "Targeted Imputation (Custom Formula)",
            "Global Imputation (single imputation)",
            "Multiple imputation (for inference)",
        ])
        with tab_multiple:
            render_multiple_imputation()
        
        # --- TAB 1: Targeted Imputation ---
        with tab_targeted:
            st.markdown("### Targeted Imputation")
            st.info("Use this mode to manually fix specific columns using formulas or AI suggestions.")
            
            main_data = st.session_state['enriched_df']
            missing_counts = main_data.isnull().sum()
            cols_with_missing = missing_counts[missing_counts > 0].index.tolist()
            
            if not cols_with_missing:
                st.success("No missing values found in the current dataset.")
            else:
                col_target, col_rag = st.columns([1, 2])
                with col_target:
                    target_var = st.selectbox("Select Variable to Impute", cols_with_missing, key="target_impute_var")
                
                # AI Suggestions
                if 'rag_manager' in st.session_state and st.session_state.rag_manager.initialized:
                    with st.expander("AI Imputation Suggestions (RAG)", expanded=True):
                        
                        # Mode Selection
                        suggestion_mode = st.radio(
                            "Suggestion Mode:",
                            ["Free Thinking (Auto)", "Guided (with Hints)"],
                            horizontal=True,
                            help="Free Thinking scans all columns. Guided allows you to provide a specific clue and limit the context."
                        )
                        
                        search_hint = None
                        context_cols = None
                        
                        if "Guided" in suggestion_mode:
                            st.info("Provide a hint (e.g., 'Unit conversion') and select relevant columns.")
                            col_hint, col_ctx = st.columns([1, 1])
                            with col_hint:
                                search_hint = st.text_input("Clue / Hint", placeholder="e.g. Reverse BMI calculation")
                            with col_ctx:
                                context_cols = st.multiselect(
                                    "Context Columns", 
                                    options=[c for c in main_data.columns if c != target_var],
                                    help="Select columns that are likely related to the target variable."
                                )

                        if st.button("Generate Formula Suggestions", key="gen_impute_sugg"):
                            with st.spinner("Analyzing relationships..."):
                                available_cols = [c for c in main_data.columns if c != target_var]
                                
                                # Prepare context stats for Guided Mode to help with units/ranges
                                context_stats = None
                                if context_cols:
                                    context_stats = extract_column_stats(main_data, context_cols + [target_var])
                                
                                suggestions, error = st.session_state.rag_manager.suggest_imputation_formulas(
                                    target_variable=target_var, 
                                    available_columns=available_cols,
                                    search_hint=search_hint,
                                    context_columns=context_cols,
                                    context_stats=context_stats
                                )
                                
                                if error:
                                    st.error(f"RAG Error: {error}")
                                elif suggestions:
                                    st.session_state['imputation_suggestions'] = suggestions
                                else:
                                    st.warning("No suggestions found.")
                        
                        if 'imputation_suggestions' in st.session_state:
                            for i, sugg in enumerate(st.session_state['imputation_suggestions']):
                                st.markdown(f"**Option {i+1}: {sugg.get('name', 'Formula')}**")
                                st.code(sugg.get('formula', ''))
                                st.caption(sugg.get('reasoning', ''))
                                if st.button(f"Use Formula #{i+1}", key=f"use_imp_sugg_{i}"):
                                    st.session_state['imputation_formula_input'] = sugg.get('formula', '')
                                    st.rerun()

                # Formula Input
                st.markdown("#### Formula Editor")
                formula_input = st.text_area(
                    "Enter Imputation Formula", 
                    value=st.session_state.get('imputation_formula_input', ''),
                    key="imputation_formula_input",
                    help="Use standard operators (+, -, *, /) and column names. Example, a missing weight: "
                         "bmi * (height_cm / 100) ** 2"
                )
                exact = st.checkbox(
                    "Exact formula: the other variables determine the value",
                    key="imputation_exact",
                    help=(
                        "Tick it when the formula gives the true value, e.g. a weight from the BMI and the height, or "
                        "a total from its parts. The values are then imputed exactly: the analyses and the multiple "
                        "imputation keep them as observed values, without the single-imputation warning. Leave it "
                        "unticked for an approximation or a proxy: the values are then flagged like any single "
                        "imputation."
                    ),
                )

                if st.button("Apply Targeted Imputation", type="primary", disabled=not formula_input):
                    try:
                        # compute formula series
                        success, result_or_error, _ = evaluate_formula_safely(formula_input, target_var, main_data, return_series=True)
                        
                        if success:
                            computed_series = result_or_error
                            filled = main_data.copy()  # the session data changes through the draft only
                            original_missing = filled[target_var].isnull().sum()

                            # Apply fillna
                            filled[target_var] = filled[target_var].fillna(computed_series)

                            new_missing = filled[target_var].isnull().sum()
                            filled_count = original_missing - new_missing

                            if filled_count > 0:
                                st.success(f"Successfully filled {filled_count} missing values in '{target_var}'!")
                                snapshot_path = save_snapshot(filled, f"impute_{target_var}")
                                get_state().stage(
                                    filled,
                                    "targeted_imputation",
                                    {
                                        "target_variable": target_var,
                                        "formula": formula_input,
                                        "filled_count": int(filled_count),
                                        "exact": bool(exact),
                                    },
                                    f"Imputed {filled_count} values in {target_var} using "
                                    f"{'an exact formula' if exact else 'a formula'}",
                                    output_dataset_path=snapshot_path
                                )
                                st.rerun()
                            else:
                                st.warning("Formula computed successfully but didn't fill any missing values (check data overlap).")
                        else:
                            st.error(f"Formula Error: {result_or_error}")
                            
                    except Exception as e:
                        st.error(f"Execution Error: {e!s}")


        # --- TAB 2: Global Imputation ---
        with tab_global:
            st.markdown("### Global & Automated Imputation")
            st.info("Use this mode to fill all missing values at once with a single imputation (mean, median, KNN, chained equations or tree-based).")
            st.warning(
                "Single imputation fills each missing cell once. P-values and confidence intervals computed on the "
                "imputed dataset will be too narrow because the uncertainty of the imputation is ignored. Use it for "
                "exploration and visualisation; for inference prefer complete-case analysis with a sensitivity check "
                "or multiple imputation (next tab)."
            )

            main_data = st.session_state['enriched_df']

            # The imputation is fitted on, and fills, the rows of the analysis population only: excluded patients
            # neither inform nor receive imputed values (follow-up programme M1)
            state = get_state()
            population = state.population
            blocked = population is not None and not population.ok
            scoped = population is not None and bool(population.criteria) and not blocked
            if blocked:
                st.error("The analysis population cannot be evaluated: correct or untick the criterion in the "
                         "sidebar before imputing.")
            rows = main_data.index.intersection(population.included) if scoped else main_data.index
            if scoped:
                st.info(f"Imputation is fitted on, and fills, the {len(rows):,} rows of the analysis population "
                        f"({population.label}); the excluded rows keep their missing values.",
                        icon=":material/filter_alt:")

            # --- Missing Data Overview (rows that would be imputed) ---
            missing_counts = main_data.loc[rows].isnull().sum()
            total_cells = np.prod(main_data.loc[rows].shape)
            total_missing = missing_counts.sum()
            missing_percent = (total_missing / total_cells) * 100
            
            col_metrics1, col_metrics2 = st.columns([1, 3])
            with col_metrics1:
                st.metric("Total Missing Values", f"{total_missing}", delta=f"{missing_percent:.2f}%")
            with col_metrics2:
                if missing_percent > 0:
                    st.info("**Tip:** With correlated variables, chained-equation or tree-based imputation usually reproduces the data structure better than mean/median imputation.")
            
            if missing_percent > 0:
                with st.expander("View Missing Values Details"):
                    st.dataframe(missing_counts[missing_counts > 0].rename("Missing Count"), width='stretch')
            else:
                st.success("No missing values detected! You can skip this step unless you want to re-process.")

            # Parameter selection
            st.subheader("Select Imputation Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                numerical_imputation_method = st.selectbox(
                    "Numerical Imputation Method",
                    available_methods(["Mean", "Median", "KNN", "MICE", "MissForest"])[0],
                    format_func=lambda m: {
                        "KNN": "K-nearest neighbours",
                        "MICE": "Chained equations (IterativeImputer, single imputation)",
                        "MissForest": "Tree-based iterative (missForest algorithm with LightGBM)",
                    }.get(m, m),
                    help="Chained equations: each variable with missing values is predicted from the others "
                         "(scikit-learn IterativeImputer, one deterministic pass). This is single imputation, "
                         "not multiple imputation: no pooled uncertainty.",
                )
            with col2:
                categorical_imputation_method = st.selectbox(
                    "Categorical Imputation Method",
                    available_methods(["Most_Frequent", "MissForest"])[0],
                    format_func=lambda m: {
                        "Most_Frequent": "Most frequent value",
                        "MissForest": "Tree-based iterative (missForest algorithm with LightGBM)",
                    }.get(m, m),
                )
            
            if not method_available("MissForest"):
                st.caption(missing_methods_message(["MissForest"]))
            if categorical_imputation_method == numerical_imputation_method:
                st.info(f"{categorical_imputation_method} has been chosen for both numerical and categorical columns, a unified approach will be used.")
            
            col3, col4 = st.columns(2)
            column_changing_help = ("Changes the columns of the table, so it is only available when every row is "
                                    "imputed (no criterion ticked).") if scoped else None
            with col3:
                cat_encoder = st.checkbox("Apply One-Hot Encoding to Categorical Columns?", value=False,
                                          disabled=scoped, help=column_changing_help)
            with col4:
                num_scaler = st.checkbox("Scale Numerical Columns?", value=False, disabled=scoped,
                                         help=column_changing_help)

            # --- Advanced Settings ---
            with st.expander("Advanced Settings (Remainder Columns)"):
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
                    ["passthrough"] if scoped else ["passthrough", "drop"],
                    help="Dropping remainder columns changes the table: only when every row is imputed."
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
            if st.button("Apply Imputer", type="primary", disabled=blocked):
                try:
                    # Define a placeholder for the progress bar
                    progress_placeholder = st.empty()
                    # Fitted on the analysis population (every row when no criterion is ticked)
                    outcome = impute_rows(
                        main_data,
                        rows if scoped else None,
                        numerical=numerical_imputation_method,
                        categorical=categorical_imputation_method,
                        cat_encoder=cat_encoder,
                        num_scaler=num_scaler,
                        remainder_columns=remainder_columns,
                        remainder_strategy=remainder_strategy,
                        remainder_threshold=remainder_threshold,
                        progress_placeholder=progress_placeholder,
                    )
                    df_imputed = outcome.frame
                    imputed_mask = outcome.mask.reindex(index=df_imputed.index, columns=df_imputed.columns,
                                                        fill_value=False)
                    if df_imputed is not None:
                        st.success("Imputation added to the Enrich draft: apply the draft (top of the page) to use "
                                   "it in the analyses.")
                        
                        # Save snapshot
                        snapshot_path = save_snapshot(df_imputed, "imputation")
                        state.stage(df_imputed, "imputation", outcome.params,
                                    f"Applied data imputation ({numerical_imputation_method} / "
                                    f"{categorical_imputation_method}"
                                    + (f", fitted on the analysis population of {len(rows):,} rows)" if scoped else ")"),
                                    population=population if scoped else None,
                                    output_dataset_path=snapshot_path)

                        # Highlight imputed values
                        def highlight_imputed(data, mask):
                            """Apply highlighting to DataFrame based on imputed mask."""
                            styled = pd.DataFrame('', index=data.index, columns=data.columns)
                            styled[mask] = 'background-color: yellow'
                            return styled

                        styled_df = df_imputed.style.apply(highlight_imputed, mask=imputed_mask, axis=None)
                        st.write("Preview of Imputed Data:")
                        st.dataframe(styled_df, width="stretch")
                        #st.write(st.session_state['enriched_df'].info(verbose=True))
                except Exception as e:
                    error_card("Imputation failed. Check the columns listed under Advanced settings.", e)
        
        with st.expander("Results", expanded=False):
            if 'enriched_df' in st.session_state and st.session_state['enriched_df'] is not None:
                display_imputation_results(st.session_state['enriched_df'], main_data=st.session_state['working_df'])

    # 3. Define New Variables
    with define_var_tab:
        with st.expander("Apply Common Transformations to Multiple Selected Variables", expanded=True):
            
            transformed_data = compute_transformations(st.session_state['enriched_df'])
            # If transformations were successfully applied, merge the data
            if transformed_data is not None:
                if 'enriched_df' in st.session_state and st.session_state['enriched_df'] is not None:
                    # Add transformed columns to main_data
                    main_data = st.session_state['enriched_df']
                    st.success("Transformed data has been successfully added to the main dataset.")
                    st.subheader("Show Updated Dataset")
                    st.dataframe(main_data, width="stretch") #st.session_state['enriched_df'])
            else:
                st.warning("No transformation to apply.")
            

        with st.expander("Variable Computation", expanded=True):
            if 'enriched_df' in st.session_state and st.session_state['enriched_df'] is not None:
                main_data = define_new_variables(main_data) #) st.session_state['enriched_df']
                #st.session_state['enriched_df'] = main_data
            else:
                st.warning("No enriched dataset")
            
        with st.expander("Results", expanded=False):
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
                #         st.success("Dataset saved and ready for further analysis!")
            else:
                no_dataset()

    with draft_slot:
        render_draft_banner()
