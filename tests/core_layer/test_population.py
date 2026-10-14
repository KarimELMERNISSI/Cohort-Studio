"""The analysis population: ticked criteria applied family by family, with the counts of a participant flow."""

import json
from pathlib import Path

import pandas as pd
import pytest

from core.fingerprint import frame_fingerprint
from core.population import (
    EXCLUSION,
    INCLUSION,
    active_criteria,
    catalogue,
    criterion_id,
    evaluate_population,
)

ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = ROOT / "assets" / "tutorial"


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    """The tutorial cohort read as the app reads it (import engine, day-first dates)."""
    from core.fingerprint import ensure_row_ids
    from core.state import ParseOptions
    from ingest.loaders import read_table

    data = (TUTORIAL / "tutorial_cohort.csv").read_bytes()
    return ensure_row_ids(read_table(data, "tutorial_cohort.csv", ParseOptions(dayfirst=True)).df)


@pytest.fixture(scope="module")
def config() -> dict:
    return json.loads((TUTORIAL / "tutorial_config.json").read_text(encoding="utf-8"))


def test_the_catalogue_lists_every_criterion_in_configuration_order(config: dict) -> None:
    criteria = catalogue(config)
    assert len(criteria) == 12
    assert [c.kind for c in criteria] == [EXCLUSION] * 8 + [INCLUSION] * 4
    assert criteria[0].id == criterion_id("clinical_anomalies", "Implausible age")
    assert criteria[8].name == "Written consent" and criteria[8].description
    assert criteria[4].columns == ("lvef_pct",)
    assert set(criteria[3].columns) == {"dbp_baseline", "sbp_baseline"}


def test_the_tutorial_flow(cohort: pd.DataFrame, config: dict) -> None:
    every = [c.id for c in catalogue(config)]
    pop = evaluate_population(cohort, active_criteria(config, every), data_fp="d")
    assert pop.ok
    assert [(s.family, s.n_in, s.n_excluded, s.n_out) for s in pop.stages] == [
        ("clinical_anomalies", 1206, 16, 1190),
        ("study_inclusion", 1190, 262, 928),
    ]
    in_stage = {
        c.criterion.name: c.marginal_in_stage for c in pop.counts if c.criterion.kind == INCLUSION
    }
    assert in_stage == {
        "Written consent": 14,
        "age": 71,
        "Symptomatic heart failure (NYHA II-IV)": 183,
        "No severe kidney disease": 8,
    }
    anomalies = {
        c.criterion.name: c.marginal_base for c in pop.counts if c.criterion.kind == EXCLUSION
    }
    assert anomalies["Implausible age"] == 2 and anomalies["Last contact before enrolment"] == 2
    sequential = [c.sequential for c in pop.counts if c.criterion.kind == INCLUSION]
    # each excluded row is counted once, by the first criterion it fails
    assert sum(sequential) == 262
    assert pop.n_included == 928 and pop.label == "928 of 1,206 rows (12 criteria)"


def test_inclusion_criteria_alone(cohort: pd.DataFrame, config: dict) -> None:
    inclusion = [c.id for c in catalogue(config) if c.kind == INCLUSION]
    pop = evaluate_population(cohort, active_criteria(config, inclusion))
    assert pop.n_included == 939
    assert [(s.n_in, s.n_excluded) for s in pop.stages] == [(1206, 267)]


def test_no_criterion_keeps_every_row(cohort: pd.DataFrame) -> None:
    pop = evaluate_population(cohort, ())
    assert pop.n_included == 1206 and pop.label == "all 1,206 rows" and not pop.stages


def test_reasons_name_the_criteria_that_exclude_a_row(cohort: pd.DataFrame, config: dict) -> None:
    pop = evaluate_population(cohort, active_criteria(config, [c.id for c in catalogue(config)]))
    excluded = cohort.index.difference(pop.included)
    assert all(pop.reasons(row) for row in excluded[:20])
    assert pop.reasons(pop.included[0]) == ()


def test_the_fingerprint_follows_the_rows_not_the_definitions(
    cohort: pd.DataFrame, config: dict
) -> None:
    data_fp = frame_fingerprint(cohort)
    consent = criterion_id("study_inclusion", "Written consent")
    base = evaluate_population(cohort, active_criteria(config, [consent]), data_fp=data_fp)
    same_rows = evaluate_population(cohort, active_criteria(config, [consent]), data_fp=data_fp)
    assert base.fingerprint == same_rows.fingerprint
    edited = json.loads(json.dumps(config))
    edited["mask_families"]["study_inclusion"]["age"]["upper_bound"] = 80
    age = criterion_id("study_inclusion", "age")
    narrower = evaluate_population(cohort, active_criteria(edited, [consent, age]), data_fp=data_fp)
    wider = evaluate_population(cohort, active_criteria(config, [consent, age]), data_fp=data_fp)
    assert narrower.spec_fingerprint != wider.spec_fingerprint
    assert narrower.fingerprint != wider.fingerprint  # the threshold edit changes the rows kept
    assert narrower.n_included < wider.n_included


def test_a_broken_criterion_fails_closed(cohort: pd.DataFrame) -> None:
    broken = {
        "mask_families": {
            "study_inclusion": {"operator": "all", "Typo": {"expression": "df['agee'] > 18"}}
        }
    }
    pop = evaluate_population(
        cohort, active_criteria(broken, [criterion_id("study_inclusion", "Typo")])
    )
    assert not pop.ok
    assert "agee" in pop.errors[criterion_id("study_inclusion", "Typo")]


def test_a_threshold_family_is_one_exclusion_rule() -> None:
    df = pd.DataFrame({"a": [1, 5, 5, 1], "b": [1, 5, 1, 1], "c": [5, 5, 1, 1]})
    config = {
        "mask_families": {
            "flags": {
                "operator": {"threshold": 2},
                "a": {"numeric": True, "lower_bound": 3},
                "b": {"numeric": True, "lower_bound": 3},
                "c": {"numeric": True, "lower_bound": 3},
            }
        }
    }
    (rule,) = catalogue(config)
    assert rule.kind == EXCLUSION and rule.name == "At least 2 rules of flags"
    pop = evaluate_population(df, (rule,))
    assert list(pop.included) == [0, 2, 3]  # row 1 breaks three rules; the others at most one


def test_the_trace_block_is_json_and_compact(cohort: pd.DataFrame, config: dict) -> None:
    pop = evaluate_population(
        cohort, active_criteria(config, [c.id for c in catalogue(config)]), data_fp="d"
    )
    block = pop.to_trace()
    json.dumps(block)
    assert block["n_included"] == 928 and len(block["criteria"]) == 12
    assert block["stages"][1]["n_out"] == 928
    assert sum(b - a + 1 for a, b in block["included"]) == 928
