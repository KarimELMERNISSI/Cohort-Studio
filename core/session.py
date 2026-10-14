"""
The session's data, without Streamlit: the committed table, the Enrich draft and the analysis population.

One object owns every change of the data a researcher analyses (follow-up programme M1):

- the *base*: the committed table, with row ids in its index, a content fingerprint and a revision;
- the *draft*: changes prepared on the Enrich page (merges, derived variables, imputations) that reach the
  base only when applied, and only if neither the base nor the population changed underneath;
- the *population*: the ticked inclusion and exclusion criteria, evaluated on the base. Criteria are
  filters: they never delete rows (only :meth:`DataSession.freeze_population` does, on request).

Every change of the base is written as one trace step through ``trace`` (the TransformationManager), with
what it did to the table: rows and columns in and out, and the fingerprints before and after.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import pandas as pd

from core.fingerprint import ensure_row_ids, frame_fingerprint
from core.lineage import Change, describe_change
from core.population import PopulationResult, active_criteria, catalogue, evaluate_population


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
    ) -> Any: ...


class StaleBaseError(RuntimeError):
    """The data changed after the change was prepared."""


class PopulationError(RuntimeError):
    """A ticked criterion cannot be evaluated: the analysis population is unknown."""


class StagingError(RuntimeError):
    """The Enrich draft cannot be applied (the base or the population changed underneath)."""


EMPTY, OK, BASE_CHANGED, POPULATION_CHANGED = "empty", "ok", "base_changed", "population_changed"


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
    population_fp: str | None = None  # population used by the population-scoped steps


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

    def _set_base(self, frame: pd.DataFrame, fp: str, *, revisions: int = 1) -> None:
        self.base = frame
        self.data_fp = fp
        self.revision += revisions
        self._population = None
        self._analysis = None

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
        self._population = None
        self._analysis = None
        self._recorded_populations.clear()
        if record:
            self._record(function, params, description, change=change)
        return change

    def clear(self) -> None:
        self.base, self.data_fp, self.revision = None, "", 0
        self.draft = None
        self.ticked = ()
        self._population = None
        self._analysis = None
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
            self.draft = Draft(base_fp=self.data_fp, frame=base)
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
        self.draft.frame = after
        self.draft.steps.append(step)
        if block and block.get("fingerprint"):
            self.draft.population_fp = str(block["fingerprint"])
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
        if draft.population_fp and config is not None:
            current = self.population(config)
            if current.fingerprint != draft.population_fp:
                return StagingStatus(
                    POPULATION_CHANGED,
                    n,
                    "The analysis population changed after the imputation in this draft was fitted.",
                )
        return StagingStatus(OK, n)

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
        self._set_base(draft.frame, fp, revisions=len(draft.steps))
        self.draft = None
        return change

    def discard_staging(self) -> None:
        self.draft = None

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
