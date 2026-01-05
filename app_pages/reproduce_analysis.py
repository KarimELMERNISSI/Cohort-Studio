import streamlit as st
import pandas as pd
import os
import json
from manage.db_manager import DBManager
# Import necessary modules for reproduction logic
import enrich.external_data as eed
from app_pages.transformation_logic import apply_variable_transformation
from app_pages.data_enrichment import apply_imputer, evaluate_formula_safely
from utils.multipage import load_dataframe

def app():
    st.title("🔄 Reproduce Analysis")
    
    trace_dir = "data/traces"
    if not os.path.exists(trace_dir):
        st.info("No analysis traces found.")
        return

    # List available traces
    traces = [f for f in os.listdir(trace_dir) if f.endswith(".json")]
    
    if not traces:
        st.info("No analysis traces found in data/traces/")
        return

    # Sort traces by modification time (newest first)
    traces.sort(key=lambda x: os.path.getmtime(os.path.join(trace_dir, x)), reverse=True)

    selected_trace_file = st.selectbox("Select Analysis Session", traces)
    
    if selected_trace_file:
        trace_path = os.path.join(trace_dir, selected_trace_file)
        with open(trace_path, 'r') as f:
            trace_data = json.load(f)
            
        source_dataset = trace_data.get('source_dataset')
        session_id = trace_data.get('session_id', 'Unknown Session')
        st.markdown(f"**Session:** `{session_id}`")
        st.markdown(f"**Source Dataset:** `{source_dataset}`")
        
        steps = trace_data.get('steps', [])
        
        if not steps:
            st.warning("This trace has no steps recorded.")
            return

        # Rerun Mode Selection
        rerun_mode = st.radio(
            "Replay Mode",
            ["Fast (Use Snapshots)", "Full Replay (From Source)"],
            help="Fast: Uses saved intermediate files if available. Full Replay: Re-executes all steps from the original source dataset."
        )

        st.subheader("Analysis Process Path")
        
        for i, step in enumerate(steps):
            # Create a visual step container
            with st.container():
                col1, col2, col3 = st.columns([0.5, 4, 1.5])
                
                # Step Number
                with col1:
                    st.markdown(f"<h3 style='text-align: center; color: #4CAF50;'>{i+1}</h3>", unsafe_allow_html=True)
                
                # Step Details
                with col2:
                    st.markdown(f"**{step.get('description', 'Operation')}**")
                    st.caption(f"Function: `{step['function']}` | Time: {step['timestamp']}")
                    
                    # Show available data links (Artifacts)
                    artifacts = []
                    if step.get('output_dataset_path') and os.path.exists(step.get('output_dataset_path')):
                        artifacts.append(f"✅ Snapshot: `{os.path.basename(step.get('output_dataset_path'))}`")
                    
                    if step.get('stats_cat_path') and os.path.exists(step.get('stats_cat_path')):
                        artifacts.append("📊 Categorical Stats")
                    
                    if step.get('stats_num_path') and os.path.exists(step.get('stats_num_path')):
                        artifacts.append("📈 Numerical Stats")
                        
                    if step.get('corr_matrix_path') and os.path.exists(step.get('corr_matrix_path')):
                        artifacts.append("📉 Correlation Matrix")
                    
                    if artifacts:
                        st.markdown(" | ".join(artifacts))

                # Action Button
                with col3:
                    if st.button(f"Reproduce to Step {i+1}", key=f"jump_{i}", width='stretch'):
                        jump_to_step(trace_data, i, rerun_mode)
                
                st.markdown("---")

def jump_to_step(trace_data, target_step_index, rerun_mode):
    """
    Reproduces the analysis up to the target step index.
    """
    steps = trace_data.get('steps', [])
    target_step = steps[target_step_index]
    
    # 1. Fast Mode: Try to load snapshot
    if rerun_mode == "Fast (Use Snapshots)" and target_step.get('output_dataset_path') and os.path.exists(target_step.get('output_dataset_path')):
        try:
            with st.spinner(f"Loading snapshot for Step {target_step_index+1}..."):
                # Determine file type
                path = target_step.get('output_dataset_path')
                if path.endswith('.parquet'):
                    df = pd.read_parquet(path)
                elif path.endswith('.csv'):
                    df = pd.read_csv(path)
                else:
                    st.error("Unknown file format for snapshot.")
                    return
                
                st.session_state['working_df'] = df
                st.session_state['data'] = df # Update main data reference too
                st.success(f"Successfully loaded state at Step {target_step_index+1} (Fast Mode)")
                return
        except Exception as e:
            st.warning(f"Failed to load snapshot: {e}. Falling back to replay.")
            # Fallback to replay
    
    # 2. Full Replay (or Fallback)
    with st.spinner(f"Replaying analysis up to Step {target_step_index+1}..."):
        source_dataset = trace_data.get('source_dataset')
        
        # Fallback: Check first step for source_path (for newer traces)
        if (not source_dataset or not os.path.exists(source_dataset)) and len(steps) > 0:
            first_step = steps[0]
            if first_step['function'] == 'initial_load' and 'source_path' in first_step['params']:
                candidate_path = first_step['params']['source_path']
                if os.path.exists(candidate_path):
                    source_dataset = candidate_path

        # Handle initial load
        if source_dataset and os.path.exists(source_dataset):
             # Determine file type
            if source_dataset.endswith('.csv'):
                df = pd.read_csv(source_dataset)
            elif source_dataset.endswith('.xlsx'):
                df = pd.read_excel(source_dataset)
            elif source_dataset.endswith('.parquet'):
                df = pd.read_parquet(source_dataset)
            else:
                st.error(f"Unknown file type for source dataset: {source_dataset}")
                return
        else:
            # Try DB Manager
            db_manager = DBManager()
            df, msg = db_manager.load_dataset(source_dataset)
        
        if df is None:
            st.error(f"Could not load source dataset '{source_dataset}'")
            return
            
        current_df = df.copy()
        
        # Replay loop
        for i in range(target_step_index + 1):
            step = steps[i]
            func_name = step['function']
            params = step['params']
            
            if func_name == "initial_load":
                continue # Already loaded

            try:
                if func_name == "enrichment":
                    # Logic to replay enrichment
                    enrichment_file_path = params.get("enrichment_file_path")
                    identifier = params.get("identifier")
                    strategy = params.get("strategy", "left")
                    conflict_resolution = params.get("conflict_resolution", "add")
                    
                    if enrichment_file_path and os.path.exists(enrichment_file_path):
                        enrichment_df = load_dataframe(enrichment_file_path)
                        if enrichment_df is not None:
                            # Ensure identifier is a list/set as expected by add_data
                            # In data_enrichment.py it handles str vs list, but let's be safe
                            # eed.add_data expects left_id_names and right_id_names
                            
                            # Calculate additional columns (all columns in enrichment_df except identifier)
                            # Note: In data_enrichment.py, it calculates this. We should do the same.
                            identifier_set = {identifier} if isinstance(identifier, str) else set(identifier)
                            additional_columns = list(set(enrichment_df.columns) - identifier_set)

                            current_df = eed.add_data(
                                df=current_df, 
                                additional_df=enrichment_df, 
                                left_id_names=identifier, 
                                right_id_names=identifier, 
                                additional_cols=additional_columns,
                                strategy=strategy, 
                                conflict_resolution=conflict_resolution
                            )
                        else:
                            st.error(f"Failed to load enrichment file: {enrichment_file_path}")
                            return
                    else:
                        st.warning(f"Enrichment file not found: {enrichment_file_path}")

                elif func_name == "variable_computation":
                     # Replay variable computation
                     formula = params.get("formula")
                     variable_name = params.get("variable_name")
                     
                     success, msg, current_df = evaluate_formula_safely(formula, variable_name, current_df)
                     if not success:
                         st.error(f"Failed to replay variable computation: {msg}")
                         return

                elif func_name == "imputation":
                    # Replay imputation
                    numerical_imputation_method = params.get("numerical_imputation_method", "Mean")
                    categorical_imputation_method = params.get("categorical_imputation_method", "Most_Frequent")
                    cat_encoder = params.get("cat_encoder", False)
                    num_scaler = params.get("num_scaler", False)
                    remainder_columns = params.get("remainder_columns", "auto")
                    remainder_strategy = params.get("remainder_strategy", "passthrough")
                    remainder_threshold = params.get("remainder_threshold", 0.3)
                    
                    current_df, _ = apply_imputer(
                        data=current_df,
                        numerical_imputation_method=numerical_imputation_method,
                        categorical_imputation_method=categorical_imputation_method,
                        cat_encoder=cat_encoder,
                        num_scaler=num_scaler,
                        remainder_columns=remainder_columns,
                        remainder_strategy=remainder_strategy,
                        remainder_threshold=remainder_threshold,
                        debug=False
                    )

                elif func_name == "variable_transformation":
                    # Replay variable transformation
                    result_df = apply_variable_transformation(current_df, params)
                    if result_df is not None:
                        current_df = pd.concat([current_df, result_df], axis=1)
                    else:
                        st.warning(f"Variable transformation returned no data at step {i+1}")

                elif func_name == "data_type_conversion":
                    # Replay data type conversion
                    columns = params.get("columns", [])
                    new_type = params.get("new_type")
                    
                    for col in columns:
                        if new_type == "datetime":
                            current_df[col] = pd.to_datetime(current_df[col])
                        else:
                            current_df[col] = current_df[col].astype(new_type)

                elif func_name == "missing_value_handling":
                    # Replay missing value handling
                    strategy = params.get("strategy")
                    columns = params.get("columns", [])
                    fill_value = params.get("fill_value")

                    if strategy == "Drop rows":
                        current_df.dropna(subset=columns, inplace=True)
                    elif strategy == "Fill with mean":
                        for col in columns:
                            current_df[col].fillna(current_df[col].mean(), inplace=True)
                    elif strategy == "Fill with median":
                        for col in columns:
                            current_df[col].fillna(current_df[col].median(), inplace=True)
                    elif strategy == "Fill with mode":
                        for col in columns:
                            # Safely handle mode if empty
                            mode_val = current_df[col].mode()
                            if not mode_val.empty:
                                current_df[col].fillna(mode_val[0], inplace=True)
                    elif strategy == "Fill with value":
                         for col in columns:
                            current_df[col].fillna(fill_value, inplace=True)

                # ... Add other replay handlers as needed ...
                    
            except Exception as e:
                st.error(f"Error replaying step {i+1} ({func_name}): {e}")
                return

        st.session_state['working_df'] = current_df
        st.session_state['data'] = current_df
        st.session_state['enriched_df'] = current_df
        st.success(f"Successfully reproduced state at Step {target_step_index+1} (Full Replay)")
