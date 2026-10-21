"""
The session's data, without Streamlit: the committed table, the Enrich draft and the analysis population.

One object owns every change of the data a researcher analyses (follow-up programme M1):

- the *base*: the committed table, with row ids in its index, a content fingerprint and a revision;
- the *draft*: changes prepared on the Enrich page (merges, derived variables, imputations) that reach the
  base only when applied, and only if neither the base nor the population changed underneath;
- the *population*: the ticked inclusion and exclusion criteria, evaluated on the base. Criteria are
  filters: they never delete rows (only :meth:`DataSession.freeze_population` does, on request);
- the *single-imputed cells*: the row ids, per column, of the values a single imputation filled
  (``SINGLE_IMPUTATION_STEPS``). Analyses treat them as observed, so their standard errors are too small;
  the pages say so (follow-up programme M2, C10);
- the *exactly imputed cells*: the values a targeted imputation filled with a formula the user marked exact, because
  the other variables determine them (a weight from the BMI and the height). Nothing was estimated, so they count
  as observed without a warning; they are kept apart so that the study dossier can say how they were imputed.

Every change of the base is written as one trace step through ``trace`` (the TransformationManager), with
what it did to the table: rows and columns in and out, and the fingerprints before and after.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import numpy as np
import pandas as pd

from core.fingerprint import ensure_row_ids, frame_fingerprint
from core.lineage import Change, describe_change
from core.population import PopulationResult, active_criteria, catalogue, evaluate_population
from core.steps import FITTED, FORMULA
from core.steps import ids as step_ids


class TraceSink(Protocol):
    def add_step(
        self,
        function_name: str,
        params: Any,
        description: str,
        output_dataset_path: str | None = None,
        *,
        data: Mapping[str, Any] | None = None,
        population: Mapping[str, Any] | None = None,
        replayable: bool | None = None,
        recorded_at: str | None = None,
    ) -> Any:
        """Record one step of the trace (implemented by manage.transformation_manager)."""


class StaleBaseError(RuntimeError):
    """The data changed after the change was prepared."""


class PopulationError(RuntimeError):
    """A ticked criterion cannot be evaluated: the analysis population is unknown."""


class StagingError(RuntimeError):
    """The Enrich draft cannot be applied (the base or the population changed underneath)."""


EMPTY, OK, BASE_CHANGED, POPULATION_CHANGED = "empty", "ok", "base_changed", "population_changed"

# steps that fill missing values with one value each (Enrich: global imputation, formula; the former MI panel)
SINGLE_IMPUTATION_STEPS = step_ids(fills=FITTED) | step_ids(fills=FORMULA)

Cells = dict[str, np.ndarray]  # column -> row ids


def filled_cells(before: pd.DataFrame, after: pd.DataFrame) -> Cells:
    """Row ids, per column, of the values missing in ``before`` and present in ``after`` (shared rows and columns)."""
    rows = before.index.intersection(after.index)
    cols = before.columns.intersection(after.columns)
    if not len(rows) or not len(cols):
        return {}
    filled = before.loc[rows, cols].isna().to_numpy() & after.loc[rows, cols].notna().to_numpy()
    ids = rows.to_numpy()
    return {
        str(col): np.asarray(ids[filled[:, j]], dtype=np.int64)
        for j, col in enumerate(cols)
        if filled[:, j].any()
    }


def merge_cells(a: Cells, b: Cells) -> Cells:
    out = dict(a)
    for col, ids in b.items():
        out[col] = np.union1d(out[col], ids) if col in out else np.asarray(ids, dtype=np.int64)
    return out


def restrict_cells(cells: Cells, frame: pd.DataFrame) -> Cells:
    """The cells still in ``frame`` (rows and columns present)."""
    out: Cells = {}
    for col, ids in cells.items():
        if col in frame.columns:
            kept = ids[np.isin(ids, frame.index.to_numpy())]
            if len(kept):
                out[col] = kept
    return out


def is_exact_imputation(function: str, params: Mapping[str, Any] | None) -> bool:
    """A targeted imputation whose formula the user marked exact: the other variables determine the values."""
    return function == "targeted_imputation" and bool((params or {}).get("exact"))


def sort_filled(
    function: str,
    params: Mapping[str, Any] | None,
    before: pd.DataFrame,
    after: pd.DataFrame,
    single: Cells,
    exact: Cells,
) -> tuple[Cells, Cells]:
    """``(single, exact)`` with the cells an imputation step filled added to the right ones (the session and the
    replay use this one rule)."""
    if function not in SINGLE_IMPUTATION_STEPS:
        return single, exact
    filled = filled_cells(before, after)
    if not is_exact_imputation(function, params):
        return merge_cells(single, filled), exact
    # exact only where every variable of the formula had a value, not a single-imputed one: a value computed from an
    # estimate is an estimate, and np.where answers a missing value with its "false" branch (the step records its
    # formula's variables, "inputs"; earlier versions did not, and every value they filled stays exact)
    target = str((params or {}).get("target_variable"))
    inputs = [str(c) for c in (params or {}).get("inputs") or () if str(c) != target]
    present = [c for c in inputs if c in before.columns]
    without = before.index[before[present].isna().any(axis=1)] if present else before.index[:0]
    imputed = [single[c] for c in inputs if c in single]
    estimated = np.unique(np.concatenate([without.to_numpy(dtype=np.int64), *imputed]))
    if not len(estimated):
        return single, merge_cells(exact, filled)
    of_estimates = {c: ids[np.isin(ids, estimated)] for c, ids in filled.items()}
    observed = {c: ids[~np.isin(ids, estimated)] for c, ids in filled.items()}
    return (
        merge_cells(single, {c: ids for c, ids in of_estimates.items() if len(ids)}),
        merge_cells(exact, {c: ids for c, ids in observed.items() if len(ids)}),
    )


def count_cells(
    cells: Cells, rows: pd.Index, columns: Iterable[str] | None = None
) -> dict[str, int]:
    """How many of ``cells`` each column has among ``rows`` (columns with none left out)."""
    wanted = None if columns is None else {str(c) for c in columns}
    ids = rows.to_numpy()
    out: dict[str, int] = {}
    for col, filled in cells.items():
        if wanted is None or col in wanted:
            n = int(np.isin(filled, ids).sum())
            if n:
                out[col] = n
    return out


@dataclass(frozen=True)
class PendingStep:
    """A change kept in the Enrich draft until it is applied."""

    function: str
    params: Mapping[str, Any]
    description: str
    change: Change
    population: Mapping[str, Any] | None = None
    replayable: bool = True
    output_dataset_path: str | None = None
    recorded_at: str = ""


@dataclass
class Draft:
    base_fp: str
    frame: pd.DataFrame
    steps: list[PendingStep] = field(default_factory=list)
    single_imputed: Cells = field(default_factory=dict)
    exact_imputed: Cells = field(default_factory=dict)

    @property
    def populations(self) -> set[str]:
        """The populations the population-scoped steps were computed on (each step's, not the last one's only)."""
        return {
            str(p.population["fingerprint"])
            for p in self.steps
            if p.population and p.population.get("fingerprint")
        }


@dataclass(frozen=True)
class StagingStatus:
    state: str  # EMPTY | OK | BASE_CHANGED | POPULATION_CHANGED
    n_steps: int = 0
    message: str = ""

    @property
    def can_apply(self) -> bool:
        return self.state == OK


@dataclass(frozen=True)
class DataIdentity:
    """What an analysis is computed on: results stamped with another identity are stale."""

    data_fp: str
    population_fp: str
    revision: int
    n_base: int
    n_population: int
    population_label: str


def _population_block(population: PopulationResult | Mapping[str, Any] | None) -> dict | None:
    if population is None:
        return None
    if isinstance(population, PopulationResult):
        return population.to_trace()
    return dict(population)


def _now() -> str:
    return datetime.now().isoformat()


class DataSession:
    """The data of one session (see the module docstring)."""

    def __init__(self, trace: TraceSink | None = None) -> None:
        self.trace = trace
        self.base: pd.DataFrame | None = None
        self.data_fp = ""
        self.revision = 0
        self.draft: Draft | None = None
        self.ticked: tuple[str, ...] = ()
        self._population: tuple[tuple[str, tuple[str, ...]], PopulationResult] | None = None
        self._analysis: tuple[str, pd.DataFrame] | None = None
        self._recorded_populations: set[str] = set()
        self.single_imputed: Cells = {}
        self.exact_imputed: Cells = {}
        self._memo: dict[str, tuple[tuple[str, Hashable], Any]] = {}

    # ------------------------------------------------------------------ state

    @property
    def loaded(self) -> bool:
        return self.base is not None

    def _require_base(self) -> pd.DataFrame:
        if self.base is None:
            raise RuntimeError("no dataset is loaded")
        return self.base

    def _record(
        self,
        function: str,
        params: Mapping[str, Any] | None,
        description: str,
        *,
        change: Change | None = None,
        population: Mapping[str, Any] | None = None,
        replayable: bool = True,
        output_dataset_path: str | None = None,
        recorded_at: str | None = None,
    ) -> None:
        if self.trace is None:
            return
        self.trace.add_step(
            function,
            dict(params or {}),
            description,
            output_dataset_path,
            data=change.to_dict() if change is not None else None,
            population=population,
            replayable=replayable,
            recorded_at=recorded_at,
        )

    def memo[T](self, name: str, key: Hashable, compute: Callable[[], T]) -> T:
        """
        A value computed from the base (a profile of its columns, a quality audit), kept until the base or ``key``
        (the caller's other inputs) changes. One slot per ``name``: a new key replaces the value kept.
        """
        slot = self._memo.get(name)
        full = (self.data_fp, key)
        if slot is not None and slot[0] == full:
            return slot[1]
        value = compute()
        self._memo[name] = (full, value)
        return value

    def _set_base(self, frame: pd.DataFrame, fp: str, *, revisions: int = 1) -> None:
        self.base = frame
        self.data_fp = fp
        self.revision += revisions
        self._population = None
        self._analysis = None
        self._memo.clear()

    # ------------------------------------------------------------------ loading

    def install(
        self,
        df: pd.DataFrame,
        *,
        function: str = "initial_load",
        params: Mapping[str, Any] | None = None,
        description: str = "Loaded dataset",
        record: bool = True,
    ) -> Change:
        """A new dataset: row ids, fingerprint, revision 0; the draft and the ticked criteria are reset."""
        base = ensure_row_ids(df)
        fp = frame_fingerprint(base)
        change = describe_change(None, base, fp_in="", fp_out=fp)
        self.base, self.data_fp, self.revision = base, fp, 0
        self.draft = None
        self.ticked = ()
        self.single_imputed = {}
        self.exact_imputed = {}
        self._population = None
        self._analysis = None
        self._memo.clear()
        self._recorded_populations.clear()
        if record:
            self._record(function, params, description, change=change)
        return change

    def clear(self) -> None:
        self.base, self.data_fp, self.revision = None, "", 0
        self.draft = None
        self.single_imputed = {}
        self.exact_imputed = {}
        self.ticked = ()
        self._population = None
        self._analysis = None
        self._memo.clear()
        self._recorded_populations.clear()

    # ------------------------------------------------------------------ base changes

    def commit(
        self,
        df: pd.DataFrame,
        function: str,
        params: Mapping[str, Any] | None,
        description: str,
        *,
        expected_fp: str | None = None,
        population: PopulationResult | Mapping[str, Any] | None = None,
        replayable: bool = True,
        output_dataset_path: str | None = None,
    ) -> Change | None:
        """Make ``df`` the base and trace it as one step; ``None`` when nothing changed."""
        self._require_base()
        if expected_fp is not None and expected_fp != self.data_fp:
            raise StaleBaseError("the dataset changed after this change was prepared")
        new = ensure_row_ids(df)
        fp = frame_fingerprint(new)
        if fp == self.data_fp:
            return None
        change = describe_change(self.base, new, fp_in=self.data_fp, fp_out=fp)
        assert self.base is not None
        single, exact = sort_filled(
            function, params, self.base, new, self.single_imputed, self.exact_imputed
        )
        self.single_imputed = restrict_cells(single, new)
        self.exact_imputed = restrict_cells(exact, new)
        self._set_base(new, fp)
        self._record(
            function,
            params,
            description,
            change=change,
            population=_population_block(population),
            replayable=replayable,
            output_dataset_path=output_dataset_path,
        )
        return change

    # ------------------------------------------------------------------ Enrich draft

    def working_frame(self) -> pd.DataFrame | None:
        """The table Enrich works on: the draft when there is one, otherwise the base."""
        return self.draft.frame if self.draft is not None else self.base

    def stage(
        self,
        df: pd.DataFrame,
        function: str,
        params: Mapping[str, Any] | None,
        description: str,
        *,
        population: PopulationResult | Mapping[str, Any] | None = None,
        replayable: bool = True,
        output_dataset_path: str | None = None,
    ) -> PendingStep | None:
        """Add a change to the draft (not traced until applied); ``None`` when it changes nothing."""
        base = self._require_base()
        if self.draft is None:
            self.draft = Draft(
                base_fp=self.data_fp,
                frame=base,
                single_imputed=dict(self.single_imputed),
                exact_imputed=dict(self.exact_imputed),
            )
        after = ensure_row_ids(df)
        before = self.draft.frame
        fp_in = self.draft.steps[-1].change.fp_out if self.draft.steps else self.data_fp
        change = describe_change(before, after, fp_in=fp_in)
        if not change.changed:
            return None
        block = _population_block(population)
        step = PendingStep(
            function,
            dict(params or {}),
            description,
            change,
            block,
            replayable,
            output_dataset_path,
            _now(),
        )
        single, exact = sort_filled(
            function, params, before, after, self.draft.single_imputed, self.draft.exact_imputed
        )
        self.draft.single_imputed = restrict_cells(single, after)
        self.draft.exact_imputed = restrict_cells(exact, after)
        self.draft.frame = after
        self.draft.steps.append(step)
        return step

    def staging_status(self, config: Mapping[str, Any] | None = None) -> StagingStatus:
        draft = self.draft
        if draft is None or not draft.steps:
            return StagingStatus(EMPTY)
        n = len(draft.steps)
        if draft.base_fp != self.data_fp:
            return StagingStatus(
                BASE_CHANGED, n, "The dataset changed after this draft was started."
            )
        populations = draft.populations
        # every imputation of the draft, not the last one only: an earlier one may be on another population
        if (
            populations
            and config is not None
            and populations != {self.population(config).fingerprint}
        ):
            return StagingStatus(
                POPULATION_CHANGED,
                n,
                "The analysis population changed after the imputation in this draft was fitted.",
            )
        return StagingStatus(OK, n)

    @property
    def working_fp(self) -> str:
        """The fingerprint of :meth:`working_frame`: the draft's last change, or the dataset's."""
        draft = self.draft
        return draft.steps[-1].change.fp_out if draft is not None and draft.steps else self.data_fp

    def commit_staging(self, config: Mapping[str, Any] | None = None) -> Change | None:
        """Trace every pending step and make the draft the base; refused when the draft is stale."""
        status = self.staging_status(config)
        if status.state == EMPTY:
            self.draft = None
            return None
        if not status.can_apply:
            raise StagingError(status.message)
        draft = self.draft
        assert draft is not None
        for step in draft.steps:
            self._record(
                step.function,
                step.params,
                step.description,
                change=step.change,
                population=step.population,
                replayable=step.replayable,
                output_dataset_path=step.output_dataset_path,
                recorded_at=step.recorded_at,
            )
        fp = draft.steps[-1].change.fp_out
        change = describe_change(self.base, draft.frame, fp_in=self.data_fp, fp_out=fp)
        self.single_imputed = restrict_cells(draft.single_imputed, draft.frame)
        self.exact_imputed = restrict_cells(draft.exact_imputed, draft.frame)
        self._set_base(draft.frame, fp, revisions=len(draft.steps))
        self.draft = None
        return change

    def discard_staging(self) -> None:
        self.draft = None

    # ------------------------------------------------------------------ imputed cells

    def _cells_in_base(self, cells: Mapping[str, Any]) -> Cells:
        base = self._require_base()
        return restrict_cells(
            {str(c): np.asarray(ids, dtype=np.int64) for c, ids in cells.items()}, base
        )

    def restore_single_imputed(self, cells: Mapping[str, Any]) -> None:
        """The single-imputed cells of a reloaded version or a replay (row ids per column)."""
        self.single_imputed = self._cells_in_base(cells)

    def restore_exact_imputed(self, cells: Mapping[str, Any]) -> None:
        """The cells imputed by an exact formula, of a reloaded version or a replay (row ids per column)."""
        self.exact_imputed = self._cells_in_base(cells)

    def single_imputed_counts(
        self, rows: pd.Index, columns: Iterable[str] | None = None
    ) -> dict[str, int]:
        """How many single-imputed values each column has among ``rows`` (columns with none left out)."""
        return count_cells(self.single_imputed, rows, columns)

    def exact_imputed_counts(
        self, rows: pd.Index, columns: Iterable[str] | None = None
    ) -> dict[str, int]:
        """How many values imputed by an exact formula each column has among ``rows``."""
        return count_cells(self.exact_imputed, rows, columns)

    # ------------------------------------------------------------------ population

    def set_ticked(self, ids: Iterable[str]) -> None:
        self.ticked = tuple(dict.fromkeys(str(i) for i in ids))
        self._analysis = None

    def tick(self, criterion_id: str, on: bool = True) -> None:
        ticked = [c for c in self.ticked if c != criterion_id]
        if on:
            ticked.append(criterion_id)
        self.set_ticked(ticked)

    def population(self, config: Mapping[str, Any] | None) -> PopulationResult:
        base = self._require_base()
        criteria = active_criteria(config, self.ticked)
        key = (self.data_fp, tuple(c.fingerprint for c in criteria))
        if self._population is not None and self._population[0] == key:
            return self._population[1]
        result = evaluate_population(base, criteria, data_fp=self.data_fp)
        self._population = (key, result)
        return result

    def analysis_frame(self, config: Mapping[str, Any] | None) -> pd.DataFrame:
        """The rows of the base kept by the ticked criteria (the base itself when none is ticked)."""
        base = self._require_base()
        population = self.population(config)
        if not population.ok:
            names = ", ".join(sorted(population.errors))
            raise PopulationError(f"these ticked criteria cannot be evaluated: {names}")
        if not population.criteria:
            return base
        if self._analysis is not None and self._analysis[0] == population.fingerprint:
            return self._analysis[1]
        frame = base.loc[population.included]
        self._analysis = (population.fingerprint, frame)
        return frame

    def identity(self, config: Mapping[str, Any] | None) -> DataIdentity:
        population = self.population(config)
        return DataIdentity(
            data_fp=self.data_fp,
            population_fp=population.fingerprint,
            revision=self.revision,
            n_base=population.n_base,
            n_population=population.n_included,
            population_label=population.label,
        )

    def record_population_use(self, config: Mapping[str, Any] | None) -> None:
        """Trace the population once per identity, when a result or a saved version first uses it."""
        population = self.population(config)
        if not population.criteria or population.fingerprint in self._recorded_populations:
            return
        self._record(
            "population_definition",
            {"criteria": [c.id for c in population.criteria], "n_included": population.n_included},
            f"Analysis population: {population.label}",
            population=population.to_trace(),
        )
        self._recorded_populations.add(population.fingerprint)

    def freeze_population(self, config: Mapping[str, Any] | None) -> Change | None:
        """Keep only the population's rows in the base (the one action that deletes rows)."""
        base = self._require_base()
        population = self.population(config)
        if not population.ok:
            raise PopulationError("the population cannot be evaluated")
        if not population.criteria or population.n_included == population.n_base:
            return None
        change = self.commit(
            base.loc[population.included],
            "freeze_population",
            {
                "criteria": [c.id for c in population.criteria],
                "n_before": population.n_base,
                "n_after": population.n_included,
            },
            f"Kept the analysis population as the dataset: {population.n_included} of "
            f"{population.n_base} rows",
            population=population,
        )
        self.set_ticked(())
        return change

    def criterion_counts(self, config: Mapping[str, Any] | None) -> dict[str, int | None]:
        """Rows of the whole table each criterion of the configuration would exclude (None: it errors),
        whether ticked or not (shown next to the checkboxes)."""
        base = self._require_base()
        criteria = catalogue(config)
        key = (self.data_fp, tuple(c.fingerprint for c in criteria))
        cached = getattr(self, "_counts", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        result = evaluate_population(base, criteria, data_fp=self.data_fp)
        counts: dict[str, int | None] = {c.id: None for c in criteria}
        for count in result.counts:
            counts[count.criterion.id] = count.marginal_base
        self._counts = (key, counts)
        return counts

    def families(self, config: Mapping[str, Any] | None) -> dict[str, list[str]]:
        """Criterion ids per family, in configuration order."""
        out: dict[str, list[str]] = {}
        for c in catalogue(config):
            out.setdefault(c.family, []).append(c.id)
        return out


__all__ = [
    "BASE_CHANGED",
    "EMPTY",
    "OK",
    "POPULATION_CHANGED",
    "DataIdentity",
    "DataSession",
    "Draft",
    "PendingStep",
    "PopulationError",
    "StaleBaseError",
    "StagingError",
    "StagingStatus",
    "TraceSink",
]
