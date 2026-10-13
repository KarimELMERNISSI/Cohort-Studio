"""
Reproduction engine: replays a transformation trace headlessly (no Streamlit).

One engine for every caller (audit E10): :func:`run_trace` executes the recorded steps against the
source dataset (or a pre-loaded frame) up to an optional step index, resolving file paths through
the user's workspace and the ``path_overrides`` chosen in the UI. :func:`execute_trace_pipeline`
keeps the historical ``(df, error)`` signature.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

import enrich.external_data as eed
from core.state import ParseOptions
from enrich.transformation_logic import apply_variable_transformation
from manage.db_manager import DBManager
from utils.path_utils import resolve_path

logger = logging.getLogger(__name__)

# Steps that only document state (no data mutation to replay)
SKIPPED_STEPS = {"initial_load", "parse_report", "save_version"}


@dataclass
class ReplayResult:
    df: pd.DataFrame | None
    error: str | None = None
    missing_file: tuple[int, str] | None = None  # (step index, recorded path) when a step needs a file that was not found
    warnings: list[str] = field(default_factory=list)
    steps_applied: int = 0


def _resolve(path: str | None, overrides: dict[str, str] | None) -> str | None:
    if not path:
        return None
    if overrides:
        candidate = overrides.get(path)
        if candidate and os.path.exists(candidate):
            return candidate
    return resolve_path(path)


def _read_any(path: str, parse_options: ParseOptions | None = None) -> pd.DataFrame | None:
    """Read csv/tsv/txt/xlsx/xls/parquet through the import engine; other names are DuckDB dataset names."""
    from ingest.loaders import file_kind, read_path

    try:
        file_kind(path)
    except ValueError:
        df, _ = DBManager().load_dataset(os.path.splitext(os.path.basename(path))[0])
        return df
    return read_path(path, parse_options).df


def load_source(trace_data: dict[str, Any], path_overrides: dict[str, str] | None = None) -> tuple[pd.DataFrame | None, str | None]:
    """The source dataset of a trace: the recorded artifact (newer traces) or the original path/name."""
    source = trace_data.get("source_dataset")
    steps = trace_data.get("steps", [])
    parse_options = None
    if steps and steps[0].get("function") == "initial_load":
        params = steps[0].get("params", {}) or {}
        if (not source or not os.path.exists(str(source))) and params.get("source_path"):
            source = params["source_path"]
        if params.get("parse_options"):
            parse_options = ParseOptions.from_dict(params["parse_options"])
    if not source:
        return None, "Trace missing 'source_dataset' field."
    resolved = _resolve(str(source), path_overrides)
    if resolved is None:
        if not os.path.splitext(str(source))[1]:  # a saved dataset name rather than a file
            df, _ = DBManager().load_dataset(str(source))
            if df is not None:
                return df, None
        return None, f"Source file not found: {source}"
    try:
        df = _read_any(resolved, parse_options)
    except Exception as exc:
        return None, f"Failed to load source dataset ({source}): {exc}"
    if df is None:
        return None, f"Failed to load source dataset ({source})"
    return df, None


def _apply_step(df: pd.DataFrame, step: dict[str, Any], index: int, overrides: dict[str, str] | None,
                warnings: list[str]) -> tuple[pd.DataFrame, str | None]:
    """Apply one recorded step; returns (df, missing_path)."""
    func_name = step.get("function")
    params = step.get("params", {}) or {}

    if func_name in SKIPPED_STEPS:
        return df, None

    if func_name == "enrichment":
        enrichment_file_path = params.get("enrichment_file_path")
        resolved = _resolve(enrichment_file_path, overrides)
        if not resolved:
            return df, str(enrichment_file_path)
        enrichment_df = _read_any(resolved)
        if enrichment_df is None:
            raise ValueError(f"could not read enrichment file {resolved}")
        identifier = params.get("identifier") or params.get("left_id_names")
        id_list = [identifier] if isinstance(identifier, str) else list(identifier or [])
        additional_columns = [c for c in enrichment_df.columns if c not in set(id_list)]
        return eed.add_data(
            df=df, additional_df=enrichment_df, left_id_names=id_list, right_id_names=id_list,
            additional_cols=additional_columns, strategy=params.get("strategy", "left"),
            conflict_resolution=params.get("conflict_resolution", "keep"),
        ), None

    if func_name == "imputation":
        from app_pages.data_enrichment import (
            apply_imputer,
        )

        new_df, _ = apply_imputer(
            data=df,
            numerical_imputation_method=params.get("numerical_imputation_method", "Mean"),
            categorical_imputation_method=params.get("categorical_imputation_method", "Most_Frequent"),
            cat_encoder=params.get("cat_encoder", False),
            num_scaler=params.get("num_scaler", False),
            remainder_columns=params.get("remainder_columns", "auto"),
            remainder_strategy=params.get("remainder_strategy", "passthrough"),
            remainder_threshold=params.get("remainder_threshold", 0.3),
        )
        return new_df, None

    if func_name == "variable_transformation":
        result_df = apply_variable_transformation(df, params)
        if result_df is None:
            warnings.append(f"step {index + 1}: variable transformation returned no data")
            return df, None
        return pd.concat([df, result_df], axis=1), None

    if func_name == "variable_computation":
        from app_pages.data_enrichment import (
            evaluate_formula_safely,
        )

        success, msg, new_df = evaluate_formula_safely(params.get("formula"), params.get("variable_name"), df)
        if not success:
            raise ValueError(f"variable computation failed: {msg}")
        return new_df, None

    if func_name == "data_type_conversion":
        new_df = df.copy()
        new_type = params.get("new_type")
        for col in params.get("columns", []):
            new_df[col] = pd.to_datetime(new_df[col]) if new_type == "datetime" else new_df[col].astype(new_type)
        return new_df, None

    if func_name == "missing_value_handling":
        new_df = df.copy()
        strategy = params.get("strategy")
        columns = params.get("columns", [])
        if strategy == "Drop rows":
            return new_df.dropna(subset=columns), None
        for col in columns:
            if strategy == "Fill with mean":
                new_df[col] = new_df[col].fillna(new_df[col].mean())
            elif strategy == "Fill with median":
                new_df[col] = new_df[col].fillna(new_df[col].median())
            elif strategy == "Fill with mode":
                mode = new_df[col].mode()
                if not mode.empty:
                    new_df[col] = new_df[col].fillna(mode[0])
            elif strategy == "Fill with value":
                new_df[col] = new_df[col].fillna(params.get("fill_value"))
        return new_df, None

    if func_name == "rename_columns":
        return df.rename(columns=params.get("mapping", {}) or {}), None

    if func_name == "apply_filter":
        kept = params.get("kept_index")
        if kept is None:
            warnings.append(f"step {index + 1}: filter recorded without the kept rows; it was not replayed")
            return df, None
        return df[df.index.isin(kept)], None

    warnings.append(f"step {index + 1}: unknown step '{func_name}' skipped")
    logger.warning("Skipping unknown pipeline step: %s", func_name)
    return df, None


def run_trace(
    trace_data: dict[str, Any],
    *,
    upto: int | None = None,
    source_df: pd.DataFrame | None = None,
    path_overrides: dict[str, str] | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> ReplayResult:
    """Replay ``trace_data`` (all steps, or up to and including ``upto``) and return the resulting frame."""
    steps = list(trace_data.get("steps", []))
    if upto is not None:
        steps = steps[: upto + 1]

    if source_df is None:
        source_df, error = load_source(trace_data, path_overrides)
        if error:
            return ReplayResult(df=None, error=error)
    current = source_df.copy()
    warnings: list[str] = []
    total = len(steps)

    for i, step in enumerate(steps):
        description = step.get("description", step.get("function"))
        if progress_callback:
            progress_callback(int(i / max(total, 1) * 100), f"Step {i + 1}: {description}")
        try:
            current, missing = _apply_step(current, step, i, path_overrides, warnings)
        except Exception as exc:
            message = f"Error during step {i + 1} ({step.get('function')}): {exc}"
            logger.error(message)
            return ReplayResult(df=None, error=message, warnings=warnings, steps_applied=i)
        if missing:
            return ReplayResult(df=None, error=f"Could not find enrichment file: {missing}", missing_file=(i, missing),
                                warnings=warnings, steps_applied=i)

    if progress_callback:
        progress_callback(100, "Pipeline Execution Complete")
    return ReplayResult(df=current, warnings=warnings, steps_applied=total)


def execute_trace_pipeline(
    trace_data: dict[str, Any],
    progress_callback: Callable[[int, str], None] | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    """Historical signature: ``(result_dataframe, error_message)``."""
    result = run_trace(trace_data, progress_callback=progress_callback)
    return result.df, result.error


__all__ = ["ReplayResult", "SKIPPED_STEPS", "execute_trace_pipeline", "load_source", "run_trace"]
