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
    criterion_line,
    flow_layout,
    participant_flow,
    source_from_trace,
)
from core.criteria import describe_definition
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
        ("Excluded: implausible values", 16),
        ("Not meeting: study inclusion", 262),
    ]
    # the flow names the family in words; its stored id stays in the ids (config.json, criterion ids, traces)
    assert [e.id for e in flow.exits] == ["stage:clinical_anomalies", "stage:study_inclusion"]
    after = flow.main[1]
    assert after.id == "after:clinical_anomalies" and after.label == "After: implausible values"
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
    # both roles under one exit: each definition says what it does ("Adult: age >= 18" read as adults removed)
    assert removed.details[1:] == (
        "Implausible age: excluded when df['age'] > 110",
        "Adult: excluded unless df['age'] >= 18",
    )
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
        == "18 <= age <= 90"
    )
    assert describe_definition("bmi", {"numeric": True, "upper_bound": 15}) == "bmi <= 15"
    assert describe_definition("o", {"bounds": {"ldl": [0.5, 9]}}) == "ldl < 0.5 or ldl > 9"
    assert describe_definition("o", {"row_ids": [[1, 3], [10, 10]]}) == "one of the 4 listed rows"
    assert (
        describe_definition("o", {"row_ids": [[1, 3]], "method": "iqr"})
        == "one of the 3 listed rows (flagged by IQR)"
    )
    members = {
        "a": {"numeric": False, "expression": "df['x'] > 1"},
        "b": {"numeric": True, "upper_bound": 15},
        "c": {},
    }
    # a rule that the upload checks refuse reads as such, never as if it were sound
    assert describe_definition("t", {"threshold": 2, "members": members}) == (
        "cannot be evaluated: rule 't': criterion 'c' has neither an expression nor a numeric range"
    )
    assert describe_definition("f", {"bounds": {"ldl": [None, 9]}}) == "ldl > 9"


EXPRESSION_X = {"expression": "df['x'] > 1"}


@pytest.mark.parametrize(
    ("definition", "words"),
    [
        # listed rows: how the rows were flagged, with the numbers recorded, and on which columns
        (
            {
                "row_ids": [[2, 4]],
                "method": "zscore",
                "params": {"threshold": 3.0},
                "columns": ["sbp"],
            },
            "one of the 3 listed rows (flagged by |z| > 3 on sbp)",
        ),
        (
            {
                "row_ids": [[2, 4]],
                "method": "iqr",
                "params": {"multiplier": 1.5},
                "columns": ["ldl"],
            },
            "one of the 3 listed rows (flagged by more than 1.5 × IQR beyond the quartiles on ldl)",
        ),
        (
            {
                "row_ids": [[2, 4]],
                "method": "quantile",
                "params": {"lower": 0.01, "upper": 0.99},
                "columns": ["a", "b", "c", "d"],
            },
            "one of the 3 listed rows (flagged outside quantiles 0.01–0.99 on any of a, b, c, d)",
        ),
        ({"row_ids": [[7, 7]], "method": "DBSCAN"}, "the 1 listed row (flagged by DBSCAN)"),
        # a multivariate method flags a row on its columns together, a univariate one on any of them
        (
            {"row_ids": [[2, 4]], "method": "Isolation Forest", "columns": ["ldl", "hdl"]},
            "one of the 3 listed rows (flagged by Isolation Forest on ldl and hdl together)",
        ),
        # a reversed range lists no row, as the mask reads it
        ({"row_ids": [[2, 4], [9, 5]]}, "one of the 3 listed rows"),
        # none listed: never met ("excluded when no listed row" read as "excluded when not listed")
        ({"row_ids": [[5, 2]]}, "never met (no listed row)"),
        # of a method this version does not know, nothing is said but its columns
        (
            {"row_ids": [[2, 4]], "method": "Dr Smith's review", "columns": ["x", "y"]},
            "one of the 3 listed rows (flagged by an unrecognised method on x, y)",
        ),
        # only the names of columns: a loaded file can list anything there
        (
            {"row_ids": [[2, 4]], "method": "zscore", "columns": [73196451, "sbp"]},
            "one of the 3 listed rows (flagged by z-score on sbp)",
        ),
        # a method or parameter this version does not know is not written out: a file can hold any text there
        (
            {"row_ids": [[2, 4]], "method": "Dr Smith's review", "params": {"threshold": "high"}},
            "one of the 3 listed rows (flagged by an unrecognised method)",
        ),
        (
            {"row_ids": [[2, 4]], "method": "zscore", "params": {"threshold": "high"}},
            "one of the 3 listed rows (flagged by z-score)",
        ),
        # several columns of frozen bounds: a value outside any of them
        ({"bounds": {"ldl": [0.5, 9], "age": [None, 85]}}, "ldl < 0.5 or ldl > 9 or age > 85"),
        ({"bounds": {"ldl": [None, None]}}, "never met (no bound)"),
        # reversed bounds are said so: nothing lies within them
        (
            {"bounds": {"ldl": [9, 0.5]}},
            "ldl < 9 or ldl > 0.5 (reversed: met by every recorded value)",
        ),
        (
            {"numeric": True, "lower_bound": 85, "upper_bound": 18, "strategy": "include"},
            "85 <= age <= 18 (reversed: met by no value)",
        ),
        (
            {"numeric": True, "lower_bound": 85, "upper_bound": 18, "strategy": "exclude"},
            "age < 85 or age > 18 (reversed: met by every recorded value)",
        ),
        # a threshold is read as the mask reads it, and says when it is met by every row or by none
        (
            {"threshold": 1.5, "members": {"a": EXPRESSION_X, "b": EXPRESSION_X}},
            "at least 1 of: a (df['x'] > 1); b (df['x'] > 1)",
        ),
        (
            {"threshold": 0, "members": {"a": EXPRESSION_X}},
            "at least 0 of: a (df['x'] > 1) (met by every row)",
        ),
        (
            {"threshold": 3, "members": {"a": EXPRESSION_X, "b": EXPRESSION_X}},
            "at least 3 of: a (df['x'] > 1); b (df['x'] > 1) (met by no row)",
        ),
        # what the upload checks refuse reads "cannot be evaluated", never "age <= None"
        (
            {"numeric": True},
            "cannot be evaluated: rule 'age' has neither a lower nor an upper bound",
        ),
    ],
)
def test_the_words_say_what_a_rule_does_at_its_edges(definition: dict, words: str) -> None:
    assert describe_definition("age", definition) == words


def test_every_outlier_method_is_named() -> None:
    from biostats.outliers import MULTIVARIATE_METHODS, UNIVARIATE_METHODS
    from core.criteria import METHOD_WORDS

    assert set(METHOD_WORDS) == {*UNIVARIATE_METHODS, *MULTIVARIATE_METHODS}


def test_a_criterion_says_what_it_does() -> None:
    """An inclusion criterion is a requirement ("excluded unless"): "kept when age >= 18" read as if meeting it were
    enough to stay, whatever the other criteria."""
    from core.population import EXCLUSION, INCLUSION

    adult = {"numeric": True, "lower_bound": 18}
    assert criterion_line("age", INCLUSION, adult) == "age: excluded unless age >= 18"
    assert criterion_line("age", EXCLUSION, adult) == "age: excluded when age >= 18"
    assert criterion_line("age", "", adult) == "age: age >= 18"  # a role that was not recorded
    assert criterion_line("frozen", EXCLUSION, {"bounds": {}}) == "frozen: never met (no bound)"
    assert (
        criterion_line("listed", INCLUSION, {"row_ids": []}) == "listed: never met (no listed row)"
    )
    assert criterion_line("age", INCLUSION, {"numeric": True}).startswith(
        "age: cannot be evaluated"
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


def test_the_dossier_gives_the_rules_of_a_threshold_family() -> None:
    """The dossier gives the definition of every criterion: a family of rules that fire together listed their names
    only."""
    from app_pages.study_dossier import _criteria_table
    from core.fingerprint import ensure_row_ids
    from core.population import active_criteria, catalogue, evaluate_population

    frame = ensure_row_ids(pd.DataFrame({"age": [10.0, 50.0, 95.0], "sbp": [90.0, 200.0, 210.0]}))
    config = {
        "mask_families": {
            "warning_signs": {
                "operator": {"threshold": 2},
                "old": {"numeric": False, "expression": "df['age'] > 80"},
                "sbp": {"numeric": True, "upper_bound": 180, "strategy": "exclude"},
            }
        }
    }
    population = evaluate_population(
        frame, active_criteria(config, [c.id for c in catalogue(config)])
    )
    table = _criteria_table(population)
    assert table["Definition"].tolist() == ["at least 2 of: old (df['age'] > 80); sbp (sbp > 180)"]
