"""The participant flow (STROBE item 13): from the source file to the analyses, every box adds up."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from core.attrition import (
    ANALYSIS,
    POPULATION,
    AnalysisCount,
    FlowBox,
    FlowExit,
    ParticipantFlow,
    SourceRows,
    describe_definition,
    flow_layout,
    participant_flow,
    source_from_trace,
)
from core.population import catalogue, criterion_id
from core.session import DataSession

ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = ROOT / "assets" / "tutorial"


class Sink:
    def __init__(self) -> None:
        self.steps: list[dict] = []

    def add_step(self, function_name, params, description, output_dataset_path=None, **extra):
        self.steps.append(
            {"function": function_name, "params": params, "description": description, **extra}
        )


@pytest.fixture(scope="module")
def tutorial() -> tuple[DataSession, dict]:
    from core.state import ParseOptions
    from ingest.loaders import read_table

    config = json.loads((TUTORIAL / "tutorial_config.json").read_text(encoding="utf-8"))
    raw = (TUTORIAL / "tutorial_cohort.csv").read_bytes()
    session = DataSession(trace=Sink())
    session.install(
        read_table(raw, "tutorial_cohort.csv", ParseOptions(dayfirst=True)).df,
        params={"filename": "tutorial_cohort.csv"},
    )
    session.set_ticked([c.id for c in catalogue(config)])
    return session, config


def test_the_tutorial_flow_matches_the_known_counts(tutorial) -> None:
    session, config = tutorial
    population = session.population(config)
    source, n_loaded, steps = source_from_trace({"steps": session.trace.steps})
    flow = participant_flow(
        population,
        dataset="tutorial_cohort",
        source=source,
        n_loaded=n_loaded,
        steps=steps,
        analyses=[
            AnalysisCount("Cox model", population.n_included, 718),
            AnalysisCount("Cox model, pooled", population.n_included, imputations=20),
        ],
    )
    assert [b.n for b in flow.main] == [1206, 1190, 928]
    assert flow.main[0].label == "Records read in tutorial_cohort.csv"
    assert flow.main[-1].kind == POPULATION and flow.main[-1].label == "Analysis population"
    assert [(e.label, e.n) for e in flow.exits] == [
        ("Excluded: clinical anomalies", 16),
        ("Not meeting: study inclusion", 262),
    ]
    assert all(sum(n for _, n in e.reasons) == e.n for e in flow.exits)
    assert all(e.details for e in flow.exits)  # the criteria definitions, shown on a click
    assert flow.inconsistencies() == [] and flow.notes == ()
    cox, pooled = flow.analyses
    assert (
        cox.kind == ANALYSIS and cox.n == 718 and cox.details == ("210 not used: missing values",)
    )
    assert pooled.n == 928 and "pooled over 20 imputed datasets" in pooled.details[0]


def test_malformed_lines_and_rows_removed_by_a_step_have_their_own_boxes() -> None:
    session = DataSession(trace=Sink())
    session.install(pd.DataFrame({"age": [12, 40, 150, 60, 30]}), params={"filename": "c.csv"})
    config = {
        "mask_families": {
            "clinical_anomalies": {
                "operator": "any",
                "Implausible age": {"expression": "df['age'] > 110"},
            },
            "study_inclusion": {"operator": "all", "Adult": {"expression": "df['age'] >= 18"}},
        }
    }
    session.set_ticked(
        [
            criterion_id("clinical_anomalies", "Implausible age"),
            criterion_id("study_inclusion", "Adult"),
        ]
    )
    session.freeze_population(config)
    trace = {"steps": session.trace.steps}
    trace["steps"].insert(1, {"function": "parse_report", "params": {"n_rows": 5, "bad_lines": 2}})
    source, n_loaded, steps = source_from_trace(trace)
    assert source == SourceRows("c.csv", 7, 2) and n_loaded == 5
    flow = participant_flow(
        session.population(config), source=source, n_loaded=n_loaded, steps=steps
    )
    assert [b.n for b in flow.main] == [7, 5, 3, 3]
    skipped, removed = flow.exits
    assert (skipped.label, skipped.n) == ("Malformed lines skipped", 2)
    assert removed.label == "Removed: population kept as the dataset" and removed.n == 2
    assert dict(removed.reasons) == {"Implausible age": 1, "Adult": 1}
    assert flow.main[-1].label == "Analysis population (no criterion ticked)"
    assert flow.inconsistencies() == []


def test_boxes_that_do_not_add_up_are_reported() -> None:
    flow = ParticipantFlow(
        main=(FlowBox("a", "A", 100, "loaded"), FlowBox("b", "B", 90, "stage")),
        exits=(FlowExit("x", "a", "Excluded", 5, reasons=(("r", 4),)),),
    )
    problems = flow.inconsistencies()
    assert "B: 90 rows, expected 95" in problems and any("do not sum" in p for p in problems)


def test_criteria_are_described_in_words() -> None:
    assert describe_definition("x", {"expression": "df['age'] > 110"}) == "df['age'] > 110"
    assert (
        describe_definition(
            "age", {"numeric": True, "lower_bound": 18, "upper_bound": 90, "strategy": "include"}
        )
        == "age within [18, 90]"
    )
    assert describe_definition("bmi", {"numeric": True, "upper_bound": 15}) == "bmi <= 15"
    assert describe_definition("o", {"bounds": {"ldl": [0.5, 9]}}) == "ldl outside [0.5, 9]"
    assert (
        describe_definition("o", {"row_ids": [[1, 3], [10, 10]]})
        == "4 listed rows (a multivariate outlier method)"
    )
    assert (
        describe_definition("t", {"threshold": 2, "members": {"a": {}, "b": {}, "c": {}}})
        == "at least 2 of: a, b, c"
    )


def test_the_layout_places_every_box(tutorial) -> None:
    session, config = tutorial
    flow = participant_flow(
        session.population(config),
        analyses=[AnalysisCount("A", 928, 900), AnalysisCount("B", 928, 928)],
    )
    pos = flow_layout(flow)
    assert set(pos) == {b.id for b in (*flow.main, *flow.exits, *flow.analyses)}
    assert all(pos[e.id][0] > 0 for e in flow.exits)  # exits to the right of the main column
    assert pos["analysis:0"][1] < pos["population"][1]  # analyses under the population
