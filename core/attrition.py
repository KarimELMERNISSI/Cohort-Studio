"""
The participant flow of a study (STROBE item 13, von Elm et al. 2007): how many records were read, loaded,
kept by each step that changed the rows, left after each family of criteria, and analysed by each reported
analysis, with the reasons for every exclusion.

The model is plain data, so the app (an interactive Plotly diagram) and the Word dossier (a static figure)
draw the same flow from the same numbers. It is built from:

- the import: records read in the file and malformed lines skipped (the ``parse_report`` trace step);
- the trace steps that removed or added rows (a merge, a population kept as the dataset);
- the analysis population (:mod:`core.population`): family by family, the rows excluded and, for each
  criterion, the rows it was the first reason for (these sum to the family's exclusions);
- the analyses: N analysed of the population, or the number of imputed datasets pooled.

:meth:`ParticipantFlow.inconsistencies` checks that every box adds up (rows in = rows out + exclusions).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from core.criteria import CriterionError, rule_kind
from core.population import EXCLUSION, PopulationResult
from core.steps import flow_name

SOURCE, LOADED, DATA, STAGE, POPULATION, ANALYSIS = (
    "source",
    "loaded",
    "data",
    "stage",
    "population",
    "analysis",
)


@dataclass(frozen=True)
class FlowBox:
    """A box of the main column (or an analysis under the population)."""

    id: str
    label: str
    n: int
    kind: str
    details: tuple[str, ...] = ()
    added: int = 0  # rows the step that leads to this box added (an outer merge)


@dataclass(frozen=True)
class FlowExit:
    """Rows leaving the flow after the box ``after``, with the reasons."""

    id: str
    after: str
    label: str
    n: int
    reasons: tuple[tuple[str, int], ...] = ()  # (criterion, rows it was the first reason for)
    details: tuple[str, ...] = ()  # the definitions of the criteria


@dataclass(frozen=True)
class SourceRows:
    """The imported file: records read and malformed lines skipped."""

    name: str
    n_read: int
    n_skipped: int = 0


@dataclass(frozen=True)
class AnalysisCount:
    """What one reported analysis used."""

    label: str
    n_population: int
    n_analysed: int | None = None
    not_used: str = "missing values"
    imputations: int | None = None  # m when the analysis pooled multiply imputed datasets
    note: str = ""  # e.g. the variables and options of the analysis


@dataclass(frozen=True)
class ParticipantFlow:
    main: tuple[FlowBox, ...]
    exits: tuple[FlowExit, ...] = ()
    analyses: tuple[FlowBox, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def item(self, item_id: str) -> FlowBox | FlowExit:
        for item in (*self.main, *self.exits, *self.analyses):
            if item.id == item_id:
                return item
        raise KeyError(item_id)

    def exits_after(self, box_id: str) -> tuple[FlowExit, ...]:
        return tuple(e for e in self.exits if e.after == box_id)

    def inconsistencies(self) -> list[str]:
        """Boxes that do not add up: rows of a box = rows of the previous box - exits + rows added."""
        problems = []
        for before, after in zip(self.main, self.main[1:], strict=False):
            expected = before.n - sum(e.n for e in self.exits_after(before.id)) + after.added
            if expected != after.n:
                problems.append(f"{after.label}: {after.n:,} rows, expected {expected:,}")
        for exit_ in self.exits:
            if exit_.reasons and sum(n for _, n in exit_.reasons) != exit_.n:
                problems.append(f"{exit_.label}: the reasons do not sum to {exit_.n:,}")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "main": [asdict(b) for b in self.main],
            "exits": [asdict(e) for e in self.exits],
            "analyses": [asdict(a) for a in self.analyses],
            "notes": list(self.notes),
        }


def _numeric_text(column: Any, low: Any, high: Any, exclude: bool) -> str:
    """Where a numeric rule fires, in words: inside the range, or outside it with the *exclude* strategy."""
    if low is not None and high is not None:
        return f"{column} {'outside' if exclude else 'within'} [{low}, {high}]"
    if low is not None:
        return f"{column} < {low}" if exclude else f"{column} >= {low}"
    return f"{column} > {high}" if exclude else f"{column} <= {high}"


def describe_definition(name: str, definition: Mapping[str, Any]) -> str:
    """A criterion's definition in words (what makes it fire), read with :func:`core.criteria.rule_kind`. Never an
    error, whatever the definition: a malformed one reads "cannot be evaluated" and why."""
    try:
        return _describe(name, definition)
    except Exception as exc:
        return f"cannot be evaluated: {exc}"


def _describe(name: str, definition: Mapping[str, Any]) -> str:
    try:
        kind = rule_kind(name, definition)
    except CriterionError as exc:
        return f"cannot be evaluated: {exc}"
    if kind == "threshold":
        members = ", ".join(str(m) for m in (definition.get("members") or {}))
        return f"at least {definition.get('threshold')} of: {members}"
    if kind == "row_ids":
        n = sum(int(b) - int(a) + 1 for a, b in definition.get("row_ids") or [])
        return f"{n:,} listed rows (a multivariate outlier method)"
    if kind == "bounds":
        parts = [
            f"{col} outside [{lo}, {hi}]"
            for col, (lo, hi) in (definition.get("bounds") or {}).items()
        ]
        return "; ".join(parts)
    if kind == "numeric":
        return _numeric_text(
            definition.get("column") or name,
            definition.get("lower_bound"),
            definition.get("upper_bound"),
            definition.get("strategy") == "exclude",
        )
    return str(definition.get("expression") or "").strip()


def _family_label(family: str, kind: str) -> str:
    words = family.replace("_", " ")
    return f"Excluded: {words}" if kind == EXCLUSION else f"Not meeting: {words}"


def participant_flow(
    population: PopulationResult,
    *,
    dataset: str = "dataset",
    source: SourceRows | None = None,
    n_loaded: int | None = None,
    steps: Sequence[Mapping[str, Any]] = (),
    analyses: Sequence[AnalysisCount] = (),
) -> ParticipantFlow:
    """
    The flow from the source file to the analyses. ``steps`` are trace steps after the load (only those that
    removed or added rows are drawn); ``n_loaded`` defaults to the records read minus the lines skipped.
    """
    main: list[FlowBox] = []
    exits: list[FlowExit] = []
    notes: list[str] = []

    if source is not None and source.n_skipped:
        main.append(FlowBox("source", f"Records read in {source.name}", source.n_read, SOURCE))
        exits.append(
            FlowExit(
                "skipped",
                "source",
                "Malformed lines skipped",
                source.n_skipped,
                details=("lines with a wrong number of fields (see the parse report)",),
            )
        )
    if n_loaded is None:
        n_loaded = (source.n_read - source.n_skipped) if source is not None else population.n_base
    label = (
        f"Records read in {source.name}"
        if source is not None and not source.n_skipped
        else f"Loaded: {dataset}"
    )
    main.append(
        FlowBox("loaded", label, int(n_loaded), SOURCE if label.startswith("Records") else LOADED)
    )

    for i, step in enumerate(steps):
        data = step.get("data") or {}
        removed, added = int(data.get("rows_removed") or 0), int(data.get("rows_added") or 0)
        if not removed and not added:
            continue
        current = main[-1]
        rows_in, rows_out = (
            int(data.get("rows_in", current.n)),
            int(data.get("rows_out", current.n)),
        )
        if rows_in != current.n:
            notes.append(
                f"Step {i + 1} starts from {rows_in:,} rows, not {current.n:,}: the trace does not record "
                "every change of the rows before it."
            )
        function = str(step.get("function") or "step")
        description = str(step.get("description") or function)
        reasons: tuple[tuple[str, int], ...] = ()
        details: tuple[str, ...] = (description,)
        block = step.get("population") or {}
        if function == "freeze_population" and block.get("counts"):
            names = {c["id"]: c["name"] for c in block.get("criteria") or []}
            reasons = tuple(
                (names.get(c["id"], c["id"]), int(c["sequential"]))
                for c in block["counts"]
                if int(c.get("sequential") or 0)
            )
            details = details + tuple(
                f"{c['name']}: {describe_definition(c['name'], c.get('definition') or {})}"
                for c in block.get("criteria") or []
            )
        if removed:
            exits.append(
                FlowExit(
                    f"step{i}:removed",
                    current.id,
                    f"Removed: {flow_name(function)}",
                    removed,
                    reasons=reasons if sum(n for _, n in reasons) == removed else (),
                    details=details,
                )
            )
        main.append(
            FlowBox(
                f"step{i}",
                f"After the {flow_name(function)}",
                rows_out,
                DATA,
                details=(description, *((f"{added:,} rows added",) if added else ())),
                added=added,
            )
        )

    if main[-1].n != population.n_base:
        notes.append(
            f"The dataset has {population.n_base:,} rows, not {main[-1].n:,}: the trace does not record every "
            "change of the rows."
        )
        main.append(FlowBox("dataset", f"Dataset: {dataset}", population.n_base, DATA))

    counts = {c.criterion.id: c for c in population.counts}
    stages = population.stages
    for k, stage in enumerate(stages):
        members = [c for c in population.criteria if c.family == stage.family]
        reasons = tuple(
            (c.name, counts[c.id].sequential)
            for c in members
            if c.id in counts and counts[c.id].sequential
        )
        exits.append(
            FlowExit(
                f"stage:{stage.family}",
                main[-1].id,
                _family_label(stage.family, stage.kind),
                stage.n_excluded,
                reasons=reasons,
                details=tuple(
                    f"{c.name}: {describe_definition(c.name, c.definition)}" for c in members
                ),
            )
        )
        last = k == len(stages) - 1
        main.append(
            FlowBox(
                "population" if last else f"after:{stage.family}",
                "Analysis population" if last else f"After: {stage.family.replace('_', ' ')}",
                stage.n_out,
                POPULATION if last else STAGE,
                details=(population.label,) if last else (),
            )
        )
    if not stages:
        main.append(
            FlowBox(
                "population",
                "Analysis population (no criterion ticked)",
                population.n_base,
                POPULATION,
            )
        )

    analysis_boxes = []
    for j, a in enumerate(analyses):
        if a.imputations:
            n, detail = (
                a.n_population,
                (f"pooled over {a.imputations} imputed datasets of the {a.n_population:,} rows",),
            )
        else:
            n = a.n_analysed if a.n_analysed is not None else a.n_population
            dropped = a.n_population - n
            detail = (
                (f"{dropped:,} not used: {a.not_used}",)
                if dropped > 0
                else ("every row of the population",)
            )
        if a.note:
            detail = (a.note, *detail)
        analysis_boxes.append(FlowBox(f"analysis:{j}", a.label, int(n), ANALYSIS, details=detail))

    return ParticipantFlow(tuple(main), tuple(exits), tuple(analysis_boxes), tuple(notes))


def flow_layout(flow: ParticipantFlow) -> dict[str, tuple[float, float]]:
    """
    Centre of every box, in flow units (a box is about 1 wide and 0.5 high): the main column at x = 0 going
    down, each exit to the right between its box and the next, the analyses in a row under the population.
    """
    pos: dict[str, tuple[float, float]] = {}
    for i, box in enumerate(flow.main):
        pos[box.id] = (0.0, -1.0 * i)
    for box in flow.main:
        x0, y0 = pos[box.id]
        for k, exit_ in enumerate(flow.exits_after(box.id)):
            pos[exit_.id] = (1.4 * (k + 1), y0 - 0.5)
    if flow.analyses:
        y = -1.0 * len(flow.main) - 0.2
        k = len(flow.analyses)
        for j, box in enumerate(flow.analyses):
            pos[box.id] = (1.15 * (j - (k - 1) / 2), y)
    return pos


def source_from_trace(trace: Mapping[str, Any]) -> tuple[SourceRows | None, int | None, list[dict]]:
    """
    From a trace: the imported file (records read, malformed lines skipped), the rows loaded, and the steps
    after the load.
    """
    steps = list(trace.get("steps") or [])
    load = next((i for i, s in enumerate(steps) if s.get("function") == "initial_load"), None)
    if load is None:
        return None, None, steps
    first = steps[load]
    params, data = first.get("params") or {}, first.get("data") or {}
    n_loaded = data.get("rows_out")
    name = str(params.get("filename") or params.get("dataset_name") or "the source file")
    report = next((s for s in steps[load + 1 :] if s.get("function") == "parse_report"), None)
    source = None
    if report is not None:
        p = report.get("params") or {}
        n_rows, bad = int(p.get("n_rows") or 0), int(p.get("bad_lines") or 0)
        source = SourceRows(name, n_rows + bad, bad)
    elif n_loaded is not None:
        source = SourceRows(name, int(n_loaded), 0)
    return source, (int(n_loaded) if n_loaded is not None else None), steps[load + 1 :]


__all__ = [
    "ANALYSIS",
    "DATA",
    "LOADED",
    "POPULATION",
    "SOURCE",
    "STAGE",
    "AnalysisCount",
    "FlowBox",
    "FlowExit",
    "ParticipantFlow",
    "SourceRows",
    "describe_definition",
    "flow_layout",
    "participant_flow",
    "source_from_trace",
]
