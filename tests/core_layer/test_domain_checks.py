"""
Domain Validity counts the checks a rule can make (11b): a check is one rule on one row that holds the values the rule
reads; score = 100 x (1 - fails / checks). Before, every row counted, so a row missing a rule's value passed it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.population import family_checks
from explore.data_quality_auditor import DataQualityAuditor
from ui.state import TUTORIAL_COHORT_PATH


def _score(df: pd.DataFrame, family: dict) -> float | None:
    return DataQualityAuditor(
        df, {"mask_families": {"clinical_anomalies": family}}
    ).compute_domain_validity()


def test_each_rule_is_checked_on_the_rows_holding_its_values() -> None:
    """A fails 1 of its 10 rows; B 1 of the 6 rows holding b: 2 of 16 checks, 87.50 (by rows, 2 of 10: 80.00)."""
    df = pd.DataFrame(
        {"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 99], "b": [1, 2, 3, 4, 5, 99] + [np.nan] * 4}
    )
    family = {"A": {"expression": "df['a'] > 50"}, "B": {"expression": "df['b'] > 50"}}
    found = family_checks(df, family)
    assert (found.checks, found.fails) == (16, 2)
    assert _score(df, family) == 87.5


def test_a_missing_value_never_raises_the_score() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 99.0, 4.0]})
    family = {"A": {"numeric": True, "column": "a", "upper_bound": 50, "strategy": "exclude"}}
    before = _score(df, family)
    masked = df.copy()
    masked.loc[1, "a"] = np.nan  # a value that passed, now missing
    assert before == 75.0 and _score(masked, family) == pytest.approx(round((1 - 1 / 3) * 100, 2))
    assert _score(masked, family) <= before


def test_a_row_where_a_rule_fires_is_checked_though_a_value_it_reads_is_missing() -> None:
    df = pd.DataFrame({"a": [np.nan, 1.0, 2.0], "b": [9.0, 1.0, np.nan]})
    found = family_checks(df, {"A or B": {"expression": "(df['a'] > 5) | (df['b'] > 5)"}})
    check = found.units["A or B"]
    assert check.checked.tolist() == [True, True, False]  # row 0 fires on b; row 2 holds no b
    assert (found.checks, found.fails) == (2, 1)


def test_a_rule_testing_missingness_checks_every_row() -> None:
    df = pd.DataFrame({"a": [np.nan, 1.0, 2.0, np.nan]})
    found = family_checks(df, {"No a": {"expression": "df['a'].isna()"}})
    assert (found.checks, found.fails) == (4, 2)


def test_bounds_make_one_check_per_column() -> None:
    df = pd.DataFrame({"x": [1.0, 50.0, np.nan], "y": [1.0, 2.0, 70.0]})
    found = family_checks(df, {"Frozen": {"bounds": {"x": [0, 10], "y": [0, 10]}}})
    assert set(found.units) == {"Frozen: x", "Frozen: y"}
    assert (found.checks, found.fails) == (5, 2)  # x on 2 rows, y on 3


def test_every_rule_family_counts_the_rows_its_rules_settle() -> None:
    df = pd.DataFrame({"a": [9.0, 9.0, 1.0, np.nan], "b": [9.0, 1.0, np.nan, np.nan]})
    family = {
        "operator": "all",
        "A": {"expression": "df['a'] > 5"},
        "B": {"expression": "df['b'] > 5"},
    }
    found = family_checks(df, family)
    # row 0: both fire; row 1: B does not; row 2: A does not; row 3: nothing known
    assert (found.checks, found.fails) == (3, 1)


@pytest.mark.parametrize(
    ("df", "family"),
    [
        (pd.DataFrame({"a": [1.0]}), {}),  # no rule
        (
            pd.DataFrame({"a": [np.nan, np.nan]}),
            {"A": {"expression": "df['a'] > 5"}},
        ),  # no row holds a
        (
            pd.DataFrame({"a": pd.Series([], dtype=float)}),
            {"A": {"expression": "df['a'] > 5"}},
        ),  # no row
    ],
)
def test_no_check_made_reads_n_a(df: pd.DataFrame, family: dict) -> None:
    assert _score(df, family) is None


def test_the_details_say_what_each_check_holds() -> None:
    df = pd.DataFrame({"a": [1.0, 99.0, 3.0], "b": [np.nan, np.nan, np.nan]})
    family = {"A": {"expression": "df['a'] > 50"}, "B": {"expression": "df['b'] > 50"}}
    auditor = DataQualityAuditor(df, {"mask_families": {"clinical_anomalies": family}})
    assert auditor.compute_domain_validity() == pytest.approx(round((1 - 1 / 3) * 100, 2))
    details = auditor.details["Domain Validity"].set_index("Where")
    assert details.loc["A", "Issue"] == "flagged, of 3 rows holding its values"
    assert details.loc["A", "Count"] == 1
    assert details.loc["B", "Issue"] == "not checked: no row holds its values"


def test_the_tutorial_reads_99_83_and_99_70_with_the_diabetes_rule() -> None:
    df = pd.read_csv(TUTORIAL_COHORT_PATH)
    config = json.loads(
        (Path(TUTORIAL_COHORT_PATH).parent / "tutorial_config.json").read_text(encoding="utf-8")
    )
    auditor = DataQualityAuditor(df, config)
    assert auditor.compute_domain_validity() == 99.83
    assert auditor.domain_checks == (16, 9593, 16)  # the rows flagged: still 16
    details = auditor.details["Domain Validity"].set_index("Where")
    assert (
        details.loc["Height not in centimetres", "Issue"]
        == "flagged, of 1,151 rows holding its values"
    )
    assert details.loc["Height not in centimetres", "Count"] == 3
    config["mask_families"]["clinical_anomalies"]["Possible undiagnosed diabetes"] = {
        "numeric": False,
        "expression": "(df['diabetes'] == 'No') & (df['hba1c_pct'] >= 6.5)",
    }
    auditor = DataQualityAuditor(df, config)
    assert auditor.compute_domain_validity() == 99.7
    assert auditor.domain_checks == (32, 10533, 32)
    details = auditor.details["Domain Validity"].set_index("Where")
    assert (
        details.loc["Possible undiagnosed diabetes", "Issue"]
        == "flagged, of 940 rows holding its values"
    )
    assert details.loc["Possible undiagnosed diabetes", "Count"] == 16
