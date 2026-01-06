import streamlit as st
import pandas as pd
import json
import enrich.external_data as eed
from app_pages.transformation_logic import apply_variable_transformation
from manage.db_manager import DBManager
from utils.path_utils import resolve_path

def reproduce_trace(trace_file):
    try:
        trace = json.load(trace_file)
    except Exception as e:
        st.error(f"Invalid JSON file: {e}")
        return

    source_dataset = trace.get("source_dataset")
    steps = trace.get("steps", [])

    st.info(f"Source Dataset: {source_dataset}")
    st.info(f"Number of Steps: {len(steps)}")

    if st.button("Start Reproduction"):
        # Load source dataset
        db_manager = DBManager()
        if hasattr(db_manager, "load_dataset"):
            # Resolve path first
            resolved_source = resolve_path(source_dataset)
            if resolved_source:
                df, msg = db_manager.load_dataset(resolved_source)
            else:
                df, msg = None, f"Source file not found: {source_dataset}"
        else:
            # Fallback if db_manager doesn't have load_dataset directly (depending on impl)
            resolved_source = resolve_path(source_dataset)
            if resolved_source:
                try:
                    if resolved_source.endswith('.csv'):
                        df = pd.read_csv(resolved_source)
                        msg = "Success"
                    elif resolved_source.endswith('.xlsx'):
                        df = pd.read_excel(resolved_source)
                        msg = "Success"
                    else:
                        df, msg = None, "Unknown format"
                except Exception as e:
                    df, msg = None, str(e)
            else:
                df, msg = None, "File not found"
        
        if df is None:
            st.error(f"Could not load source dataset: {msg}")
            return

        st.success(f"Loaded source dataset: {source_dataset}")
        
        current_df = df.copy()
        
        progress_bar = st.progress(0)
        
        # Import here to avoid circular dependency
        from app_pages.data_enrichment import apply_imputer
        
        for i, step in enumerate(steps):
            func_name = step['function']
            params = step['params']
            description = step.get('description', func_name)
            
            st.write(f"Step {i+1}: {description}")
            
            try:
                if func_name == "enrichment":
                    # Need to load enrichment file
                    enrichment_file_path = params.get("enrichment_file_path")
                    resolved_enrich = resolve_path(enrichment_file_path)

                    if not resolved_enrich:
                        st.error(f"Could not load enrichment file: {enrichment_file_path}. Please ensure the file is accessible.")
                        return

                    # Try to load enrichment df
                    try:
                        if resolved_enrich.endswith('.xlsx'):
                            enrichment_df = pd.read_excel(resolved_enrich) 
                        else:
                            enrichment_df = pd.read_csv(resolved_enrich)
                    except Exception as e:
                        st.error(f"Could not load enrichment file: {resolved_enrich} (Original: {enrichment_file_path}). Error: {e}")
                        return
                    # identifier_set = {identifier} if isinstance(identifier, str) else set(identifier)
                    # additional_columns = list(set(enrichment_df.columns) - identifier_set)
                    
                    # Re-calculate additional columns as they are not stored in params (or maybe they should be?)
                    # In the original code: additional_columns = list(set(enrichment_df.columns) - identifier_set)
                    # We can re-compute it.
                    identifier_set = {identifier} if isinstance(identifier, str) else set(identifier)
                    additional_columns = list(set(enrichment_df.columns) - identifier_set)

                    current_df = eed.add_data(
                        df=current_df,
                        additional_df=enrichment_df,
                        left_id_names=identifier,
                        right_id_names=identifier,
                        additional_cols=additional_columns,
                        strategy=params.get("strategy"),
                        conflict_resolution=params.get("conflict_resolution")
                    )
                    
                elif func_name == "imputation":
                    current_df, _ = apply_imputer(
                        data=current_df,
                        numerical_imputation_method=params.get("numerical_imputation_method"),
                        categorical_imputation_method=params.get("categorical_imputation_method"),
                        cat_encoder=params.get("cat_encoder"),
                        num_scaler=params.get("num_scaler"),
                        remainder_columns=params.get("remainder_columns"),
                        remainder_strategy=params.get("remainder_strategy"),
                        remainder_threshold=params.get("remainder_threshold"),
                        progress_placeholder=st.empty()
                    )
                    
                elif func_name == "variable_transformation":
                    result_df = apply_variable_transformation(current_df, params)
                    if result_df is not None:
                        current_df = pd.concat([current_df, result_df], axis=1)
                
                else:
                    st.warning(f"Unknown function: {func_name}")
            
            except Exception as e:
                st.error(f"Error in step {i+1}: {e}")
                return

            progress_bar.progress((i + 1) / len(steps))

        st.success("Reproduction Complete!")
        st.dataframe(current_df.head())
        
        # Option to save
        if st.button("Save Reproduced Dataset"):
            db_manager.save_dataset(current_df, base_name=f"reproduced_{source_dataset}")
            st.success("Saved!")
