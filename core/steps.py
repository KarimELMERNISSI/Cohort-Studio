"""
The kinds of trace step: one catalogue (M5) instead of the name sets that the replay, the trace report, the workflow
progress, the participant flow, the study dossier and the imputation bookkeeping each kept.

Each kind declares:

- its title in the trace report;
- the part of the workflow whose page records it;
- how the replay treats it: recomputed from its parameters, replaced by the table recorded with it, or skipped
  because it only documents;
- whether it changes the values or the rows of the data: renaming or retyping columns does not. A step recorded
  since trace schema 2 also says it with its fingerprints, which win (they count a rename);
- for the participant flow, the noun of its box ("After the merge");
- whether it fills missing values with one value each, learnt from the rows or computed by a formula.

A step id is stored in every trace, so it is never renamed: a kind that no page records any more stays registered
as legacy, and old traces still read and replay. A new kind of step is one registration below, plus a handler in
``manage/reproduction_manager.py`` when the replay recomputes it; ``tests/core_layer/test_steps.py`` checks both.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.registry import Catalogue

# The part of the workflow whose page records the step (core.progress.STEPS), or the analysis population
LOAD, VALIDATE, ENRICH, POPULATION, ANALYSE, REPORT, OTHER = (
    "load",
    "validate",
    "enrich",
    "population",
    "analyse",
    "report",
    "other",
)
CATEGORIES = (LOAD, VALIDATE, ENRICH, POPULATION, ANALYSE, REPORT, OTHER)

# How the replay treats the step
RECOMPUTE = "recompute"  # redone from its parameters
SNAPSHOT = "snapshot"  # cannot be redone: the table recorded with it is used, when there is one
DOCUMENT = "document"  # records a state; there is nothing to redo
REPLAY_MODES = (RECOMPUTE, SNAPSHOT, DOCUMENT)

# How the step fills missing values with one value each
FITTED = "fitted"  # values learnt from the rows (a statistical imputation)
FORMULA = "formula"  # values computed by a formula: of each row's own variables, or an aggregate such as a mean


@dataclass(frozen=True)
class StepKind:
    id: str
    title: str
    category: str
    replay: str
    changes_data: bool
    flow: str = ""  # the noun of the participant flow; empty: the id in words
    fills: str = ""  # FITTED or FORMULA for a step that fills missing values with one value each

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(
                f"step {self.id!r}: category {self.category!r} is not one of {CATEGORIES}"
            )
        if self.replay not in REPLAY_MODES:
            raise ValueError(
                f"step {self.id!r}: replay {self.replay!r} is not one of {REPLAY_MODES}"
            )
        if self.fills not in ("", FITTED, FORMULA):
            raise ValueError(
                f"step {self.id!r}: fills {self.fills!r} is not {FITTED!r} or {FORMULA!r}"
            )

    @property
    def flow_name(self) -> str:
        return self.flow or self.id.replace("_", " ")


STEPS: Catalogue[StepKind] = Catalogue("trace step")


def _register(kind: StepKind, *, legacy: bool = False) -> None:
    STEPS.register(kind, legacy=legacy)


_register(StepKind("initial_load", "Dataset Loading", LOAD, DOCUMENT, changes_data=False))
_register(StepKind("parse_report", "Import Parse Report", LOAD, DOCUMENT, changes_data=False))
_register(
    StepKind(
        "data_type_conversion", "Data Type Conversion", VALIDATE, RECOMPUTE, changes_data=False
    )
)
_register(
    StepKind(
        "outlier_clipping",
        "Outliers Clipped to Their Bounds",
        VALIDATE,
        RECOMPUTE,
        changes_data=True,
    )
)
_register(
    StepKind("enrichment", "Data Enrichment", ENRICH, RECOMPUTE, changes_data=True, flow="merge")
)
_register(
    StepKind(
        "variable_computation", "New Variable Computation", ENRICH, RECOMPUTE, changes_data=True
    )
)
_register(
    StepKind(
        "variable_transformation", "Variable Transformation", ENRICH, RECOMPUTE, changes_data=True
    )
)
_register(
    StepKind(
        "imputation", "Missing Value Imputation", ENRICH, RECOMPUTE, changes_data=True, fills=FITTED
    )
)
_register(
    StepKind(
        "targeted_imputation",
        "Targeted Imputation (Formula)",
        ENRICH,
        RECOMPUTE,
        changes_data=True,
        fills=FORMULA,
    )
)
# the imputed datasets are kept beside the data, which the step does not change
_register(StepKind("mi_impute", "Multiple Imputation", ENRICH, RECOMPUTE, changes_data=False))
# recorded when a result first uses the population; the replay checks its rows
_register(
    StepKind(
        "population_definition", "Analysis Population", POPULATION, RECOMPUTE, changes_data=False
    )
)
_register(
    StepKind(
        "freeze_population",
        "Analysis Population Kept as the Dataset",
        POPULATION,
        RECOMPUTE,
        changes_data=True,
        flow="population kept as the dataset",
    )
)
_register(StepKind("rename_columns", "Columns Renamed", ANALYSE, RECOMPUTE, changes_data=False))
_register(StepKind("save_version", "Version Saved", REPORT, DOCUMENT, changes_data=False))
# a page replaced the data without going through AppState (strict mode refuses it)
_register(
    StepKind(
        "external_write",
        "Data Replaced by a Page (Not Replayable)",
        OTHER,
        SNAPSHOT,
        changes_data=True,
    )
)

# Legacy: no page records them any more; traces of earlier versions hold them
_register(
    StepKind(
        "apply_filter",
        "Filter Applied to the Dataset",
        VALIDATE,
        RECOMPUTE,
        changes_data=True,
        flow="filter",
    ),
    legacy=True,
)
_register(
    StepKind(
        "missing_value_handling",
        "Manual Missing Value Handling",
        ENRICH,
        RECOMPUTE,
        changes_data=True,
    ),
    legacy=True,
)
_register(
    StepKind(
        "multiple_imputation",
        "Multiple Imputation (Diagnostic, One Dataset Adopted)",
        ENRICH,
        SNAPSHOT,
        changes_data=True,
        fills=FITTED,
    ),
    legacy=True,
)
_register(
    StepKind(
        "outlier_handling", "Outlier Detection & Handling", VALIDATE, SNAPSHOT, changes_data=True
    ),
    legacy=True,
)


def step_title(step_id: str) -> str:
    """The heading of a step in the trace report; an unknown id in words."""
    kind = STEPS.find(step_id)
    return kind.title if kind is not None else step_id.replace("_", " ").title()


def flow_name(step_id: str) -> str:
    """The noun of the step's box in the participant flow ("After the merge")."""
    kind = STEPS.find(step_id)
    return kind.flow_name if kind is not None else step_id.replace("_", " ")


def ids(
    *,
    replay: str | None = None,
    category: str | None = None,
    changes_data: bool | None = None,
    fills: str | None = None,
) -> frozenset[str]:
    """The ids of the kinds that match every condition given (legacy kinds included: old traces hold them)."""
    return frozenset(
        k.id
        for k in STEPS
        if (replay is None or k.replay == replay)
        and (category is None or k.category == category)
        and (changes_data is None or k.changes_data == changes_data)
        and (fills is None or k.fills == fills)
    )


__all__ = [
    "ANALYSE",
    "CATEGORIES",
    "DOCUMENT",
    "ENRICH",
    "FITTED",
    "FORMULA",
    "LOAD",
    "OTHER",
    "POPULATION",
    "RECOMPUTE",
    "REPLAY_MODES",
    "REPORT",
    "SNAPSHOT",
    "STEPS",
    "VALIDATE",
    "StepKind",
    "flow_name",
    "ids",
    "step_title",
]
