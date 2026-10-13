"""
Workflow progress: which steps of the research workflow are done for the data currently loaded.

Pure Python. Two kinds of evidence are combined:

- the session's transformation trace: every change of the data is a trace step (enrichment,
  computed variables, imputation, filtering...), so counting them gives the *revision* of the
  data; saving a version is a trace step as well;
- marks recorded by the pages when a step is performed on the current data (the quality check
  on Validate, statistics or a test on Explore and Analyze, the report downloaded on Reproduce).

A mark remembers the revision it was made at. When the data changes afterwards, the step is
*stale*: done, but on older data. That is how the incremental loop Validate <-> Enrich <->
Explore & analyse is followed: after an enrichment, validation and analysis are suggested again.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

# Trace steps that change the values, rows or columns of the session dataset.
DATA_CHANGES: frozenset[str] = frozenset(
    {
        "enrichment",
        "variable_computation",
        "variable_transformation",
        "targeted_imputation",
        "imputation",
        "multiple_imputation",
        "apply_filter",
    }
)
# The subset that belongs to the Enrich path (filtering belongs to validation).
ENRICH_CHANGES: frozenset[str] = DATA_CHANGES - {"apply_filter"}
SAVE_STEPS: frozenset[str] = frozenset({"save_version"})

STEPS: tuple[str, ...] = ("load", "validate", "enrich", "analyse", "report", "docs", "varmap")
CORE_PATH: tuple[str, ...] = ("load", "validate", "analyse", "report")
MARKABLE: frozenset[str] = frozenset({"validate", "analyse", "report"})

DONE, STALE, TODO = "done", "stale", "todo"


@dataclass(frozen=True)
class StepMark:
    """A step performed at a given revision of the data."""

    revision: int
    at: str = ""  # ISO timestamp
    detail: str = ""  # what was done, e.g. "quality score 87/100"


@dataclass(frozen=True)
class StepStatus:
    state: str  # DONE | STALE | TODO
    text: str  # one sentence for the map's hint line
    changes_since: int = 0  # data changes after the step (STALE only)


@dataclass(frozen=True)
class Progress:
    statuses: Mapping[str, StepStatus]
    next_step: str | None  # the suggested next step on the core path, None when all is up to date
    revision: int = 0
    marks: Mapping[str, StepMark] = field(default_factory=dict)

    def state(self, step: str) -> str:
        status = self.statuses.get(step)
        return status.state if status else TODO

    def text(self, step: str) -> str:
        status = self.statuses.get(step)
        return status.text if status else ""


def _function(step: Mapping[str, Any]) -> str:
    return str(step.get("function") or "")


def data_revision(steps: Iterable[Mapping[str, Any]]) -> int:
    """Number of changes made to the data since it was loaded."""
    return sum(1 for step in steps if _function(step) in DATA_CHANGES)


def trace_marks(steps: Iterable[Mapping[str, Any]]) -> dict[str, StepMark]:
    """Marks that the trace itself proves: the last enrichment and the last saved version."""
    revision = 0
    n_enrich = 0
    marks: dict[str, StepMark] = {}
    for step in steps:
        name = _function(step)
        at = str(step.get("timestamp") or "")
        description = str(step.get("description") or name.replace("_", " "))
        if name in DATA_CHANGES:
            revision += 1
        if name in ENRICH_CHANGES:
            n_enrich += 1
            plural = "s" if n_enrich > 1 else ""
            marks["enrich"] = StepMark(
                revision, at, f"{n_enrich} change{plural} applied, last: {description}"
            )
        if name in SAVE_STEPS:
            marks["report"] = StepMark(revision, at, description)
    return marks


def _latest(a: StepMark | None, b: StepMark | None) -> StepMark | None:
    if a is None or b is None:
        return a or b
    return max(a, b, key=lambda m: (m.revision, m.at))


def _changes(n: int) -> str:
    return f"{n} change{'s' if n != 1 else ''} to the data"


_DONE_TEXT = {
    "validate": "Quality checked on the current data ({detail}).",
    "analyse": "Analysed on the current data ({detail}).",
    "report": "Up to date: {detail}.",
}
_STALE_TEXT = {
    "validate": "Checked before the last {changes}: check the quality again.",
    "analyse": "Results predate the last {changes}: run them again.",
    "report": "Saved before the last {changes}: save a new version or export the report again.",
}
_TODO_TEXT = {
    "validate": "Not checked yet for this dataset.",
    "analyse": "No statistics or tests run yet for this dataset.",
    "report": "No version saved or report exported yet for this dataset.",
}


def evaluate(
    steps: Iterable[Mapping[str, Any]],
    marks: Mapping[str, StepMark] | None = None,
    *,
    has_data: bool,
    dataset_label: str = "",
    knowledge: Mapping[str, tuple[bool, str]] | None = None,
) -> Progress:
    """
    Status of every step and the suggested next one.

    ``steps`` is the current trace (list of step dicts), ``marks`` the marks recorded by the pages,
    ``knowledge`` the evidence for the two knowledge paths: ``{"docs": (done, text), "varmap": (done, text)}``.
    """
    steps = list(steps)
    revision = data_revision(steps)
    merged: dict[str, StepMark] = dict(trace_marks(steps))
    for step, mark in (marks or {}).items():
        merged[step] = _latest(merged.get(step), mark) or mark

    statuses: dict[str, StepStatus] = {}
    no_data = "Load a dataset first."
    if has_data:
        label = f": {dataset_label}" if dataset_label else ""
        statuses["load"] = StepStatus(DONE, f"Dataset loaded{label}.")
    else:
        statuses["load"] = StepStatus(TODO, "No dataset loaded yet.")

    for step in ("validate", "analyse", "report"):
        mark = merged.get(step)
        if not has_data:
            statuses[step] = StepStatus(TODO, no_data)
        elif mark is None:
            statuses[step] = StepStatus(TODO, _TODO_TEXT[step])
        elif mark.revision >= revision:
            statuses[step] = StepStatus(DONE, _DONE_TEXT[step].format(detail=mark.detail or "done"))
        else:
            n = revision - mark.revision
            statuses[step] = StepStatus(STALE, _STALE_TEXT[step].format(changes=_changes(n)), n)

    enrich = merged.get("enrich")
    if not has_data:
        statuses["enrich"] = StepStatus(TODO, no_data)
    elif enrich is None:
        statuses["enrich"] = StepStatus(TODO, "Optional: no change applied to this dataset yet.")
    else:
        statuses["enrich"] = StepStatus(DONE, f"{enrich.detail}.")

    for step in ("docs", "varmap"):
        done, text = (knowledge or {}).get(step, (False, ""))
        statuses[step] = StepStatus(DONE if done else TODO, text)

    return Progress(
        statuses=statuses,
        next_step=suggest_next(statuses, has_data),
        revision=revision,
        marks=merged,
    )


def suggest_next(statuses: Mapping[str, StepStatus], has_data: bool) -> str | None:
    """First core step that is not done for the current data (load, validate, analyse, report)."""
    if not has_data:
        return "load"
    for step in CORE_PATH[1:]:
        status = statuses.get(step)
        if status is None or status.state != DONE:
            return step
    return None


def initial_progress(has_data: bool) -> Progress:
    """Progress with no evidence beyond whether a dataset is loaded (fresh session, tests)."""
    return evaluate([], {}, has_data=has_data)


__all__ = [
    "CORE_PATH",
    "DATA_CHANGES",
    "DONE",
    "ENRICH_CHANGES",
    "MARKABLE",
    "SAVE_STEPS",
    "STALE",
    "STEPS",
    "TODO",
    "Progress",
    "StepMark",
    "StepStatus",
    "data_revision",
    "evaluate",
    "initial_progress",
    "suggest_next",
    "trace_marks",
]
