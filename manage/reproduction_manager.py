"""
Reproduction Manager Module.

Decouples data trace execution from Streamlit presentation logic, providing:
- execute_trace_pipeline: Pure data pipeline execution engine (headless, testable).
- reproduce_trace: Streamlit interactive runner for UI-based workflow replay.
"""

import json
import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

import enrich.external_data as eed
from app_pages.transformation_logic import apply_variable_transformation
from manage.db_manager import DBManager
from utils.path_utils import resolve_path

logger = logging.getLogger(__name__)


def execute_trace_pipeline(
    trace_data: dict[str, Any],
    progress_callback: Callable[[int, str], None] | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Execute a serialized transformation trace against the source dataset.
    
    Args:
        trace_data: Dictionary parsed from the trace JSON specification.
        progress_callback: Optional callback reporting (percent: int, message: str).
        
    Returns:
        Tuple of (result_dataframe, error_message). On failure, result_dataframe is None.
    """
    source_dataset = trace_data.get("source_dataset")
    steps = trace_data.get("steps", [])

    if not source_dataset:
        return None, "Trace missing 'source_dataset' field."

    # 1. Resolve and load the source dataset
    resolved_source = resolve_path(source_dataset)
    if not resolved_source:
        return None, f"Source file not found: {source_dataset}"

    db_manager = DBManager()
    if hasattr(db_manager, "load_dataset"):
        df, msg = db_manager.load_dataset(resolved_source)
    else:
        try:
            if resolved_source.endswith(".csv"):
                df, msg = pd.read_csv(resolved_source), "Success"
            elif resolved_source.endswith(".xlsx"):
                df, msg = pd.read_excel(resolved_source), "Success"
            else:
                df, msg = None, f"Unsupported file extension: {resolved_source}"
        except Exception as read_err:
            df, msg = None, str(read_err)

    if df is None:
        return None, f"Failed to load source dataset ({source_dataset}): {msg}"

    current_df = df.copy()
    total_steps = len(steps)

    # 2. Iterate through sequential trace steps
    for i, step in enumerate(steps):
        func_name = step.get("function")
        params = step.get("params", {})
        description = step.get("description", func_name)

        if progress_callback:
            progress_callback(int((i / max(total_steps, 1)) * 100), f"Step {i+1}: {description}")

        try:
            if func_name == "enrichment":
                enrichment_file_path = params.get("enrichment_file_path")
                resolved_enrich = resolve_path(enrichment_file_path)
                if not resolved_enrich:
                    return None, f"Could not find enrichment file: {enrichment_file_path}"

                if resolved_enrich.endswith(".xlsx"):
                    enrichment_df = pd.read_excel(resolved_enrich)
                else:
                    enrichment_df = pd.read_csv(resolved_enrich)

                identifier = params.get("identifier") or params.get("left_id_names")
                identifier_set = {identifier} if isinstance(identifier, str) else set(identifier if identifier else [])
                additional_columns = list(set(enrichment_df.columns) - identifier_set)

                current_df = eed.add_data(
                    df=current_df,
                    additional_df=enrichment_df,
                    left_id_names=identifier,
                    right_id_names=identifier,
                    additional_cols=additional_columns,
                    strategy=params.get("strategy"),
                    conflict_resolution=params.get("conflict_resolution"),
                )

            elif func_name == "imputation":
                from app_pages.data_enrichment import apply_imputer
                current_df, _ = apply_imputer(
                    data=current_df,
                    numerical_imputation_method=params.get("numerical_imputation_method"),
                    categorical_imputation_method=params.get("categorical_imputation_method"),
                    cat_encoder=params.get("cat_encoder"),
                    num_scaler=params.get("num_scaler"),
                    remainder_columns=params.get("remainder_columns"),
                    remainder_strategy=params.get("remainder_strategy"),
                    remainder_threshold=params.get("remainder_threshold"),
                )

            elif func_name == "variable_transformation":
                result_df = apply_variable_transformation(current_df, params)
                if result_df is not None:
                    current_df = pd.concat([current_df, result_df], axis=1)

            else:
                logger.warning(f"Skipping unknown pipeline step: {func_name}")

        except Exception as step_err:
            err_msg = f"Error during step {i+1} ({func_name}): {step_err}"
            logger.error(err_msg)
            return None, err_msg

    if progress_callback:
        progress_callback(100, "Pipeline Execution Complete")

    return current_df, None


def reproduce_trace(trace_file: Any) -> None:
    """
    Streamlit interactive wrapper for executing trace pipelines.
    
    Args:
        trace_file: File-like object containing trace JSON.
    """
    import streamlit as st

    try:
        trace = json.load(trace_file)
    except Exception as e:
        st.error(f"Invalid JSON file: {e}")
        return

    source_dataset = trace.get("source_dataset", "Unknown")
    steps = trace.get("steps", [])

    st.info(f"Source Dataset: {source_dataset}")
    st.info(f"Configured Pipeline Steps: {len(steps)}")

    if st.button("Start Reproduction", key="btn_start_reproduce"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def on_progress(pct: int, msg: str):
            progress_bar.progress(pct)
            status_text.text(msg)

        with st.spinner("Executing reproduction trace..."):
            result_df, err = execute_trace_pipeline(trace, progress_callback=on_progress)

        progress_bar.empty()
        status_text.empty()

        if err:
            st.error(err)
            return

        st.success("Reproduction Pipeline Complete")
        st.dataframe(result_df.head(), width="stretch")

        db_manager = DBManager()
        if st.button("Save Reproduced Dataset", key="btn_save_reproduce", width="stretch"):
            db_manager.save_dataset(result_df, base_name=f"reproduced_{source_dataset}")
            st.success("Dataset saved successfully.")
