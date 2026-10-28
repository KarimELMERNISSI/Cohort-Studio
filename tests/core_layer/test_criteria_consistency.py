"""
One reading of the criteria everywhere (M5): the Validate checks and the quality dashboard count what the analysis
population counts, the flow describes what a rule selects, and an ambiguous configuration is refused.

Before, the Validate checks and the dashboard read the rules with their own helpers: a rule holding both a range
and an expression counted 3 rows in Validate and 6 in the population, a threshold family 5 rows on the dashboard
and 1 in the population, and "All" made an inclusion family an exclusion family.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.criteria import (
    CriterionError,
    criterion_columns,
    criterion_mask,
    describe_definition,
    rule_kind,
)
from core.population import (
    EXCLUSION,
    INCLUSION,
    catalogue,
    configuration_problems,
    evaluate_population,
    family_masks,
)
from explore.data_quality_auditor import DataQualityAuditor

TUTORIAL = Path(__file__).resolve().parents[2] / "assets" / "tutorial"


@pytest.fixture(scope="module")
def tutorial() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(TUTORIAL / "tutorial_cohort.csv")
    config = json.loads((TUTORIAL / "tutorial_config.json").read_text(encoding="utf-8"))
    return df, config


def _excluded_by_family(df: pd.DataFrame, config: dict, family: str) -> pd.Index:
    criteria = [c for c in catalogue(config) if c.family == family]
    result = evaluate_population(df, criteria)
    assert result.ok, result.errors
    return df.index.difference(result.included)


def _flagged(df: pd.DataFrame, config: dict, family: str) -> pd.Index:
    masks = family_masks(df, config["mask_families"][family])
    assert masks.flag is not None, masks.errors
    # an exclusion family excludes the rows it flags; an inclusion family the rows that do not meet all criteria
    return df.index[masks.flag] if masks.operator != "all" else df.index[~masks.flag]


@pytest.mark.parametrize("family", ["clinical_anomalies", "study_inclusion"])
def test_validate_counts_what_the_population_counts(tutorial, family: str) -> None:
    df, config = tutorial
    assert _flagged(df, config, family).equals(_excluded_by_family(df, config, family))


def test_the_dashboard_counts_the_anomaly_family(tutorial) -> None:
    df, config = tutorial
    auditor = DataQualityAuditor(df, config)
    score = auditor.compute_clinical_validity()
    flagged = len(_flagged(df, config, "clinical_anomalies"))
    assert flagged == 16 and score == pytest.approx(round((1 - flagged / len(df)) * 100, 2))


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"age": [10, 17, 18, 40, 80, 86, 90, np.nan], "k": [1, 9, 1, 9, 1, 9, 9, 1]}
    )


def test_a_threshold_family_counts_the_same_on_every_screen() -> None:
    members = {
        "operator": {"threshold": 2},
        "old": {"numeric": False, "expression": "df['age'] > 70"},
        "high": {"numeric": False, "expression": "df['k'] > 5"},
        "age": {"numeric": True, "lower_bound": 15, "upper_bound": 85, "strategy": "exclude"},
    }
    config = {"mask_families": {"clinical_anomalies": members}}
    df = _frame()
    masks = family_masks(df, members)
    # rows 5 and 6 break all three rules; rows 0, 1, 3 and 4 break one each
    assert masks.operator == "threshold" and list(df.index[masks.flag]) == [5, 6]
    assert df.index[masks.flag].equals(_excluded_by_family(df, config, "clinical_anomalies"))
    auditor = DataQualityAuditor(df, config)
    assert auditor.compute_clinical_validity() == pytest.approx(round((1 - 2 / len(df)) * 100, 2))


def test_an_operator_is_read_without_regard_to_case() -> None:
    config = {
        "mask_families": {
            "study_inclusion": {
                "operator": "All",
                "adult": {"numeric": False, "expression": "df['age'] >= 18"},
            }
        }
    }
    assert {c.kind for c in catalogue(config)} == {INCLUSION}
    assert family_masks(_frame(), config["mask_families"]["study_inclusion"]).operator == "all"
    assert configuration_problems(config) == []
    unknown = {
        "mask_families": {
            "study_inclusion": {"operator": "every", "adult": {"expression": "df['age'] > 1"}}
        }
    }
    assert "unknown operator 'every'" in configuration_problems(unknown)[0]
    assert {c.kind for c in catalogue(unknown)} == {
        EXCLUSION
    }  # read as the default; the upload refuses it


def test_a_rule_with_a_range_and_an_expression_is_refused() -> None:
    both = {
        "numeric": True,
        "lower_bound": 18,
        "upper_bound": 85,
        "strategy": "include",
        "expression": "df['age'] > 80",
    }
    with pytest.raises(CriterionError, match="both a numeric range and an expression"):
        rule_kind("age", both)
    with pytest.raises(CriterionError, match="both a numeric range and an expression"):
        criterion_mask(_frame(), "age", both)
    config = {"mask_families": {"study_inclusion": {"operator": "all", "age": both}}}
    assert any(
        "both a numeric range and an expression" in p for p in configuration_problems(config)
    )
    masks = family_masks(_frame(), config["mask_families"]["study_inclusion"])
    assert masks.flag is None and "age" in masks.errors  # Validate shows the error, not a count
    population = evaluate_population(_frame(), catalogue(config))
    assert not population.ok  # and the population refuses it


def test_other_configuration_problems() -> None:
    config = {
        "mask_families": {
            "study_inclusion": {
                "operator": "all",
                "note": "free text",
                "age": {"numeric": True, "strategy": "keep"},
            },
            "outliers": {"operator": {"threshold": "two"}},
        }
    }
    problems = " | ".join(configuration_problems(config))
    assert "'note' is not a rule" in problems
    assert "rule 'age' has neither a lower nor an upper bound" in problems
    assert "strategy must be include or exclude" in problems
    assert "threshold must be a whole number" in problems


@pytest.mark.parametrize(
    ("definition", "text", "fires"),
    [
        ({"numeric": True, "lower_bound": 18}, "age >= 18", [2, 3, 4, 5, 6]),
        ({"numeric": True, "lower_bound": 18, "strategy": "exclude"}, "age < 18", [0, 1]),
        ({"numeric": True, "upper_bound": 85}, "age <= 85", [0, 1, 2, 3, 4]),
        ({"numeric": True, "upper_bound": 85, "strategy": "exclude"}, "age > 85", [5, 6]),
        (
            {"numeric": True, "lower_bound": 18, "upper_bound": 85, "strategy": "include"},
            "18 <= age <= 85",
            [2, 3, 4],
        ),
        (
            {"numeric": True, "lower_bound": 18, "upper_bound": 85, "strategy": "exclude"},
            "age < 18 or age > 85",
            [0, 1, 5, 6],
        ),
    ],
)
def test_the_description_says_where_a_numeric_rule_fires(
    definition: dict, text: str, fires: list[int]
) -> None:
    # the flow and the dossier said "age >= 18" for a one-sided exclude rule that fires on age < 18
    assert describe_definition("age", definition) == text
    assert list(_frame().index[criterion_mask(_frame(), "age", definition)]) == fires


def test_the_columns_of_every_kind() -> None:
    assert criterion_columns("o", {"row_ids": [[1, 3]], "columns": ["ldl", "hdl"]}) == (
        "ldl",
        "hdl",
    )
    assert criterion_columns("o", {"bounds": {"ldl": [0.5, 9]}}) == ("ldl",)
    assert criterion_columns("age", {"numeric": True, "lower_bound": 1}) == ("age",)
    assert criterion_columns("x", {"expression": "df['b'] > df['a']"}) == ("a", "b")
    threshold = {
        "threshold": 2,
        "members": {"a": {"expression": "df['x'] > 1"}, "y": {"numeric": True, "upper_bound": 2}},
    }
    assert criterion_columns("t", threshold) == ("x", "y")
