"""
Reproduction engine: replays a transformation trace headlessly (no Streamlit).

One engine for every caller (audit E10): :func:`run_trace` executes the recorded steps against the
source dataset (or a pre-loaded frame) up to an optional step index, resolving file paths through
the user's workspace and the ``path_overrides`` chosen in the UI. :func:`execute_trace_pipeline`
keeps the historical ``(df, error)`` signature.

Traces recorded since schema 2 carry, for every step, the fingerprint of the table it produced. The
replay checks each replayed step against it and reports the first *divergence*; steps that cannot be
recomputed (an adopted multiple imputation, a table replaced outside the app) are listed as not
replayable, and their recorded snapshot is used when there is one. A replay is *reproducible* when every
step was replayed and matched.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

import enrich.external_data as eed
from core.fingerprint import ensure_row_ids, frame_fingerprint, ranges_to_ids
from core.population import criteria_from_trace, evaluate_population
from core.session import restrict_cells, sort_filled
from core.state import ParseOptions
from core.steps import DOCUMENT, SNAPSHOT
from core.steps import ids as step_ids
from enrich.formulas import evaluate_formula_safely, read_formula
from enrich.imputation_service import (
    describe_formula_fill,
    describe_imputation,
    fill_with_formula,
    impute_rows,
    left_missing,
)
from enrich.transformation_logic import apply_variable_transformation
from manage.db_manager import DBManager
from utils.date_parser import smart_parse_dates
from utils.path_utils import resolve_path

logger = logging.getLogger(__name__)

# Steps that only document state (no data mutation to replay); core/steps.py declares each kind's replay
SKIPPED_STEPS = step_ids(replay=DOCUMENT)
# Steps the engine cannot recompute: their snapshot is used when one was recorded
NOT_REPLAYABLE = step_ids(replay=SNAPSHOT)


@dataclass(frozen=True)
class StepRecord:
    """What a recomputed step did on the replayed data, in the trace's terms (its settings and description): an
    imputation's counts and rows depend on the data, so a redone Enrich draft step is recorded with these."""

    params: dict[str, Any]
    description: str


@dataclass
class ReplayResult:
    df: pd.DataFrame | None
    error: str | None = None
    missing_file: tuple[int, str] | None = None  # (step index, recorded path) when a step needs a file that was not found
    warnings: list[str] = field(default_factory=list)
    steps_applied: int = 0
    divergence: tuple[int, str, str] | None = None  # (step index, recorded fingerprint, replayed fingerprint)
    not_replayable: list[int] = field(default_factory=list)  # step indexes that could not be recomputed
    verified_steps: int = 0  # steps whose result was checked against the recorded fingerprint
    imputations: dict[int, Any] = field(default_factory=dict)  # step index -> MIData re-computed
    mi_divergence: list[int] = field(default_factory=list)  # mi_impute steps whose cells differ
    single_imputed: dict[str, Any] = field(default_factory=dict)  # column -> row ids filled by single imputation
    exact_imputed: dict[str, Any] = field(default_factory=dict)  # column -> row ids imputed by an exact formula
    records: dict[int, StepRecord] = field(default_factory=dict)  # step index -> what a recomputed imputation did
    # during a replay only: the tables a later step's population was evaluated on, by fingerprint
    population_frames: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False, compare=False)

    @property
    def reproducible(self) -> bool:
        """Every step replayed, and every recorded fingerprint matched."""
        return (
            self.error is None
            and self.missing_file is None
            and self.divergence is None
            and not self.not_replayable
            and not self.mi_divergence
        )


class _NotReplayable(Exception):
    """A step the engine cannot recompute; carries the frame to continue from."""


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
        # a trace is user-supplied JSON: its paths are looked up in the workspace only (_resolve), never as typed
        if (not source or _resolve(str(source), path_overrides) is None) and params.get("source_path"):
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
                return ensure_row_ids(df), None
        return None, f"Source file not found: {source}"
    try:
        df = _read_any(resolved, parse_options)
    except Exception as exc:
        return None, f"Failed to load source dataset ({source}): {exc}"
    if df is None:
        return None, f"Failed to load source dataset ({source})"
    return ensure_row_ids(df), None  # the row ids a fresh load gives (AppState.set_dataset)


def _population_rows(
    df: pd.DataFrame,
    block: dict[str, Any],
    index: int,
    warnings: list[str],
    frames: Mapping[str, pd.DataFrame] | None = None,
) -> pd.Index:
    """
    Rows of ``df`` kept by the population recorded in a step (criteria re-evaluated as they were), on the table it was
    evaluated on when ``frames`` holds it (by its fingerprint), else on ``df``. Every step of an Enrich draft is scoped
    to the population of the data before the draft, which a fill of the same draft can change: a weight filled above
    an anomaly rule's bound must not take its row out of the next fill.
    """
    criteria = criteria_from_trace(block)
    if not criteria:
        return df.index
    frame = (frames or {}).get(str(block.get("data_fp") or ""), df)
    result = evaluate_population(frame, criteria)
    if not result.ok:
        raise ValueError(f"the population criteria cannot be evaluated: {', '.join(result.errors)}")
    if block.get("rows_fp") and result.rows_fingerprint != block["rows_fp"]:
        warnings.append(
            f"step {index + 1}: the analysis population re-evaluated on the replayed data keeps other rows "
            "than when the step was recorded"
        )
    return result.included if frame is df else df.index.intersection(result.included)


def _snapshot(step: dict[str, Any], overrides: dict[str, str] | None) -> pd.DataFrame | None:
    path = _resolve(step.get("output_dataset_path"), overrides)
    if path and path.endswith(".parquet") and os.path.exists(path):
        return ensure_row_ids(pd.read_parquet(path))
    return None


def _enrichment(df, step, index, overrides, warnings):
    params = step.get("params", {}) or {}
    enrichment_file_path = params.get("enrichment_file_path")
    resolved = _resolve(enrichment_file_path, overrides)
    if not resolved:
        return df, str(enrichment_file_path)
    # read as the page read it (the sheet, merge key, separator and decimal recorded with the step), then the date
    # keys parsed as the page parsed them
    recorded = params.get("parse_options")
    enrichment_df = _read_any(resolved, ParseOptions.from_dict(recorded) if recorded else None)
    if enrichment_df is None:
        raise ValueError(f"could not read enrichment file {resolved}")
    enrichment_df = eed.align_date_keys(df, enrichment_df)
    identifier = params.get("identifier") or params.get("left_id_names")
    left_ids = [identifier] if isinstance(identifier, str) else list(identifier or [])
    right = params.get("right_id_names") or left_ids
    right_ids = [right] if isinstance(right, str) else list(right)
    retained = params.get("retained_enrichment_columns")
    if retained:
        enrichment_df = enrichment_df[[c for c in enrichment_df.columns if str(c) in set(retained)]]
    main = df
    if params.get("retained_main_columns"):
        main = df[[c for c in df.columns if str(c) in set(params["retained_main_columns"])]]
    additional_columns = [c for c in enrichment_df.columns if c not in set(right_ids)]
    return eed.add_data(
        df=main, additional_df=enrichment_df, left_id_names=left_ids, right_id_names=right_ids,
        additional_cols=additional_columns, strategy=params.get("strategy", "left"),
        conflict_resolution=params.get("conflict_resolution", "keep"),
    ), None


def _with_inputs(step: dict[str, Any]) -> dict[str, Any] | None:
    """A targeted imputation's params with the variables of its formula other than its target ("inputs"), which
    earlier versions did not record: a value an exact formula computed from a single-imputed one, or from none, is
    single-imputed (core.session.sort_filled)."""
    params = step.get("params")
    if step.get("function") != "targeted_imputation" or not params or "inputs" in params:
        return params
    names = read_formula(str(params.get("formula") or "")).names
    return {**params, "inputs": [n for n in names if n != params.get("target_variable")]}


def _scoped_rows(
    df: pd.DataFrame,
    step: dict[str, Any],
    index: int,
    warnings: list[str],
    result: ReplayResult | None = None,
) -> pd.Index | None:
    """The rows an imputation is fitted on and fills: the population recorded with the step, as when it was applied;
    None (every row) when none was recorded, as by earlier versions, or when no criterion was ticked."""
    block = step.get("population")
    if not block or not block.get("criteria"):
        return None
    return _population_rows(df, block, index, warnings, result.population_frames if result else None)


def _imputation(df, step, index, overrides, warnings, result=None):
    params = step.get("params", {}) or {}
    rows = _scoped_rows(df, step, index, warnings, result)
    outcome = impute_rows(
        df,
        rows,
        numerical=params.get("numerical_imputation_method", "Mean"),
        categorical=params.get("categorical_imputation_method", "Most_Frequent"),
        cat_encoder=params.get("cat_encoder", False),
        num_scaler=params.get("num_scaler", False),
        remainder_columns=params.get("remainder_columns", "auto"),
        remainder_strategy=params.get("remainder_strategy", "passthrough"),
        remainder_threshold=params.get("remainder_threshold", 0.3),
    )
    if result is not None:
        methods = (outcome.params["numerical_imputation_method"], outcome.params["categorical_imputation_method"])
        description = describe_imputation(*methods, None if rows is None else len(rows))
        result.records[index] = StepRecord(outcome.params, description)
    return outcome.frame, None


def _targeted_imputation(df, step, index, overrides, warnings, result=None):
    params = step.get("params", {}) or {}
    rows = _scoped_rows(df, step, index, warnings, result)
    target, formula = params["target_variable"], params["formula"]
    # an exact formula fills the rows where every variable of it has a value (not recorded: every row, as before);
    # a declared derived variable's fill is rounded as the variable was recorded, and may be limited to the values a
    # text or code variable took when its definition was checked: decisions recorded, never made again here
    complete = bool(params.get("complete_inputs"))
    new_df, filled = fill_with_formula(
        df,
        target,
        formula,
        rows=rows,
        complete_inputs=complete,
        finite_only=bool(params.get("finite_only")),
        decimals=params.get("round_decimals"),
        levels=params.get("only_levels"),
    )
    recorded = params.get("filled_count")
    if recorded is not None and int(recorded) != filled:
        warnings.append(f"step {index + 1}: {filled} values filled on replay, {recorded} when recorded")
    if result is not None:
        n_rows = None if rows is None else len(rows)
        left = left_missing(df, target, formula, rows) if complete else (0, ())
        exact = bool(params.get("exact"))
        derived = params.get("derived")
        definition = derived.get("definition") if isinstance(derived, Mapping) else None
        of = definition.get("target") if isinstance(definition, Mapping) else None
        description = describe_formula_fill(
            target, filled, exact=exact, n_rows=n_rows, left=left, derived_of=of
        )
        result.records[index] = StepRecord({**params, "filled_count": filled}, description)
    return new_df, None


def _variable_transformation(df, step, index, overrides, warnings):
    result_df = apply_variable_transformation(df, step.get("params", {}) or {})
    if result_df is None:
        warnings.append(f"step {index + 1}: variable transformation returned no data")
        return df, None
    return pd.concat([df, result_df], axis=1), None


def _variable_computation(df, step, index, overrides, warnings):
    params = step.get("params", {}) or {}
    success, msg, new_df = evaluate_formula_safely(params.get("formula"), params.get("variable_name"), df)
    if not success:
        raise ValueError(f"variable computation failed: {msg}")
    return new_df, None


def _data_type_conversion(df, step, index, overrides, warnings):
    params = step.get("params", {}) or {}
    new_df = df.copy()
    new_type = params.get("new_type")
    for col in params.get("columns", []):
        if new_type == "datetime" and params.get("method") == "smart_parse_dates":
            new_df[col], _ = smart_parse_dates(new_df[col])
        elif new_type == "datetime":
            new_df[col] = pd.to_datetime(new_df[col])
        else:
            new_df[col] = new_df[col].astype(new_type)
    return new_df, None


def _missing_value_handling(df, step, index, overrides, warnings):
    params = step.get("params", {}) or {}
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


def _rename_columns(df, step, index, overrides, warnings):
    return df.rename(columns=(step.get("params", {}) or {}).get("mapping", {}) or {}), None


def _apply_filter(df, step, index, overrides, warnings):
    kept = (step.get("params", {}) or {}).get("kept_index")
    if kept is None:
        warnings.append(f"step {index + 1}: filter recorded without the kept rows; it was not replayed")
        return df, None
    return df[df.index.isin(kept)], None


def _population_definition(df, step, index, overrides, warnings):
    _population_rows(df, step.get("population") or {}, index, warnings)  # verification only
    return df, None


def _mi_impute(df, step, index, overrides, warnings, result=None):
    """Re-run a multiple imputation on its population and check the imputed cells against the recorded hash."""
    import numpy
    import sklearn

    from biostats.mi import ImputationSpec, impute

    params = step.get("params", {}) or {}
    block = step.get("population") or {}
    frames = result.population_frames if result is not None else None
    rows = _population_rows(df, block, index, warnings, frames) if block else df.index
    run = impute(df.loc[rows], ImputationSpec(**params["spec"]))
    if result is not None:
        result.imputations[index] = run.mi
    recorded = params.get("cells_fp")
    if recorded and run.mi.fingerprint() != recorded:
        if result is not None:
            result.mi_divergence.append(index)
        versions = params.get("versions") or {}
        now = {"numpy": numpy.__version__, "scikit-learn": sklearn.__version__}
        changed = [f"{k} {versions.get(k)} -> {v}" for k, v in now.items() if versions.get(k) != v]
        why = f" (changed since: {', '.join(changed)})" if changed else ""
        warnings.append(f"step {index + 1} (mi_impute): the re-computed imputation differs from the recorded one{why}")
    return df, None


def _freeze_population(df, step, index, overrides, warnings):
    block = step.get("population") or {}
    if block.get("criteria"):
        rows = _population_rows(df, block, index, warnings)
    elif block.get("included"):
        rows = pd.Index(ranges_to_ids(block["included"]))
    else:
        raise ValueError("the frozen population was not recorded")
    return df.loc[df.index.intersection(rows)], None


def _outlier_clipping(df, step, index, overrides, warnings):
    new_df = df.copy()
    for col, (low, high) in ((step.get("params", {}) or {}).get("bounds") or {}).items():
        new_df[col] = new_df[col].clip(lower=low, upper=high)
    return new_df, None


_HANDLED: dict[str, Callable[..., tuple[pd.DataFrame, str | None]]] = {
    "enrichment": _enrichment,
    "imputation": _imputation,
    "targeted_imputation": _targeted_imputation,
    "variable_transformation": _variable_transformation,
    "variable_computation": _variable_computation,
    "data_type_conversion": _data_type_conversion,
    "missing_value_handling": _missing_value_handling,
    "rename_columns": _rename_columns,
    "apply_filter": _apply_filter,
    "population_definition": _population_definition,
    "freeze_population": _freeze_population,
    "outlier_clipping": _outlier_clipping,
    "mi_impute": _mi_impute,
}
# The handlers that report on the replay result: the imputed datasets, and what each imputation did (StepRecord)
_REPORTING = frozenset({"imputation", "targeted_imputation", "mi_impute"})


def _apply_step(df: pd.DataFrame, step: dict[str, Any], index: int, overrides: dict[str, str] | None,
                warnings: list[str], result: ReplayResult | None = None) -> tuple[pd.DataFrame, str | None]:
    """Apply one recorded step; returns (df, missing_path). Raises _NotReplayable for steps it cannot redo."""
    func_name = str(step.get("function"))

    if func_name in SKIPPED_STEPS:
        return df, None

    if func_name in NOT_REPLAYABLE or step.get("replayable") is False:
        snapshot = _snapshot(step, overrides)
        if snapshot is not None:
            warnings.append(f"step {index + 1} ({func_name}) cannot be recomputed: its recorded snapshot was used")
            return snapshot, None
        handler = _HANDLED.get(func_name)
        if handler is not None:  # recorded in part only (e.g. a filter that also clipped values)
            warnings.append(f"step {index + 1} ({func_name}) was replayed only in part: it was not fully recorded")
            partial, _ = handler(df, step, index, overrides, warnings)
            raise _NotReplayable(partial)
        warnings.append(f"step {index + 1} ({func_name}) cannot be recomputed and has no snapshot: skipped")
        raise _NotReplayable(df)

    handler = _HANDLED.get(func_name)
    if handler is None:
        warnings.append(f"step {index + 1}: unknown step '{func_name}' skipped")
        logger.warning("Skipping unknown pipeline step: %s", func_name)
        raise _NotReplayable(df)
    if func_name in _REPORTING:
        return handler(df, step, index, overrides, warnings, result)
    return handler(df, step, index, overrides, warnings)


def apply_steps(
    df: pd.DataFrame,
    steps: list[dict[str, Any]],
    *,
    path_overrides: dict[str, str] | None = None,
    start: int = 0,
    progress_callback: Callable[[int, str], None] | None = None,
    frames: Mapping[str, pd.DataFrame] | None = None,
) -> ReplayResult:
    """Replay ``steps`` on ``df`` (step numbers in messages start at ``start``), checking fingerprints. ``frames``:
    tables a step's population may name by fingerprint although the replay does not pass through them (a draft
    redone one change at a time names the dataset it was built on)."""
    current = ensure_row_ids(df)
    result = ReplayResult(df=None)
    total = len(steps)
    # the tables the recorded populations were evaluated on, kept while a step names one (_population_rows)
    named = {str(fp) for step in steps if (fp := (step.get("population") or {}).get("data_fp"))}
    result.population_frames.update({fp: f for fp, f in (frames or {}).items() if fp in named})
    if named and (initial := frame_fingerprint(current)) in named:
        result.population_frames[initial] = current
    for offset, step in enumerate(steps):
        i = start + offset
        description = step.get("description", step.get("function"))
        if progress_callback:
            progress_callback(int(offset / max(total, 1) * 100), f"Step {i + 1}: {description}")
        replayed = True
        before = current
        try:
            current, missing = _apply_step(current, step, i, path_overrides, result.warnings, result)
        except _NotReplayable as skipped:
            current, missing, replayed = skipped.args[0], None, False
            result.not_replayable.append(i)
        except Exception as exc:
            message = f"Error during step {i + 1} ({step.get('function')}): {exc}"
            logger.error(message)
            result.error, result.steps_applied = message, offset
            result.population_frames.clear()
            return result
        if current is not None:
            result.single_imputed, result.exact_imputed = sort_filled(
                str(step.get("function")),
                _with_inputs(step),
                before,
                current,
                result.single_imputed,
                result.exact_imputed,
            )
        if missing:
            result.error = f"Could not find enrichment file: {missing}"
            result.missing_file, result.steps_applied = (i, missing), offset
            result.population_frames.clear()
            return result
        expected = (step.get("data") or {}).get("fp_out")
        verified = bool(expected and replayed)
        actual = frame_fingerprint(current) if verified or named else None
        if verified:
            result.verified_steps += 1
            if actual != expected and result.divergence is None:
                result.divergence = (i, str(expected), str(actual))
                result.warnings.append(
                    f"step {i + 1} ({step.get('function')}): the replayed table differs from the recorded one"
                )
        if actual in named:
            result.population_frames[str(actual)] = current
    result.population_frames.clear()
    result.df = current
    result.single_imputed = restrict_cells(result.single_imputed, current)
    result.exact_imputed = restrict_cells(result.exact_imputed, current)
    result.steps_applied = total
    if progress_callback:
        progress_callback(100, "Pipeline Execution Complete")
    return result


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
    return apply_steps(source_df, steps, path_overrides=path_overrides, progress_callback=progress_callback)


def execute_trace_pipeline(
    trace_data: dict[str, Any],
    progress_callback: Callable[[int, str], None] | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    """Historical signature: ``(result_dataframe, error_message)``."""
    result = run_trace(trace_data, progress_callback=progress_callback)
    return result.df, result.error


__all__ = [
    "NOT_REPLAYABLE",
    "SKIPPED_STEPS",
    "ReplayResult",
    "StepRecord",
    "apply_steps",
    "execute_trace_pipeline",
    "load_source",
    "run_trace",
]
