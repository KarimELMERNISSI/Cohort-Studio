"""
Domain Validity counts the checks a rule can make (11b): a check is one rule on one row that holds the values the rule
reads; score = 100 x (1 - fails / checks), rounded down. Before, every row counted, so a row missing a rule's value
passed it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.criteria import CRITERION_KINDS, CriterionError, CriterionKind, rule_checks
from core.population import family_checks
from explore.data_quality_auditor import NO_RULE, DataQualityAuditor, failing_score
from ui.state import TUTORIAL_COHORT_PATH


def _score(df: pd.DataFrame, family: dict, **kwargs) -> float | None:
    return DataQualityAuditor(
        df, {"mask_families": {"clinical_anomalies": family}}, **kwargs
    ).compute_domain_validity()


def _units(df: pd.DataFrame, family: dict) -> dict:
    return {check.unit: check for check in family_checks(df, family).units}


def test_each_rule_is_checked_on_the_rows_holding_its_values() -> None:
    """A fails 1 of its 10 rows; B 1 of the 6 rows holding b: 2 of 16 checks, 87.50 (by rows, 2 of 10: 80.00)."""
    df = pd.DataFrame(
        {"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 99], "b": [1, 2, 3, 4, 5, 99] + [np.nan] * 4}
    )
    family = {"A": {"expression": "df['a'] > 50"}, "B": {"expression": "df['b'] > 50"}}
    found = family_checks(df, family)
    assert (found.checks, found.fails) == (16, 2)
    assert _score(df, family) == 87.5


def test_a_failing_check_never_reads_100() -> None:
    """One failing check in more than 20,000 rounded to 100.0, and the dashboard said "Perfect"."""
    assert failing_score(1, 20_001) == 99.99
    assert failing_score(0, 20_001) == 100.0
    assert failing_score(1, 3) == 66.67  # rounded as usual below
    df = pd.DataFrame({"a": [1.0] * 25_000 + [99.0]})
    auditor = DataQualityAuditor(
        df, {"mask_families": {"clinical_anomalies": {"A": {"expression": "df['a'] > 50"}}}}
    )
    metrics = auditor.run_audit()
    assert metrics["Domain Validity"] == 99.99
    assert any(a["category"] == "Domain Validity" for a in auditor.advice)


def test_a_missing_value_never_raises_the_score() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 99.0, 4.0]})
    family = {"A": {"numeric": True, "column": "a", "upper_bound": 50, "strategy": "exclude"}}
    before = _score(df, family)
    masked = df.copy()
    masked.loc[1, "a"] = np.nan  # a value that passed, now missing
    assert before == 75.0 and _score(masked, family) == failing_score(1, 3)
    assert (_score(masked, family) or 0) <= before


def test_a_row_where_a_rule_fires_is_checked_though_a_value_it_reads_is_missing() -> None:
    df = pd.DataFrame({"a": [np.nan, 1.0, 2.0], "b": [9.0, 1.0, np.nan]})
    family = {"A or B": {"expression": "(df['a'] > 5) | (df['b'] > 5)"}}
    check = _units(df, family)["A or B"]
    assert check.checked.tolist() == [True, True, False]  # row 0 fires on b; row 2 holds no b
    found = family_checks(df, family)
    assert (found.checks, found.fails) == (2, 1)


def test_a_rule_testing_missingness_checks_the_rows_it_fires_on_and_those_holding_its_values() -> (
    None
):
    df = pd.DataFrame({"a": [np.nan, 1.0, 2.0, np.nan]})
    found = family_checks(df, {"No a": {"expression": "df['a'].isna()"}})
    assert (found.checks, found.fails) == (4, 2)


@pytest.mark.parametrize(
    "expression",
    [
        "df['sbp'].notna() & (df['sbp'] > 250)",
        "df['sbp'].fillna(170) > 250",
        "(df['sbp'] > 250) & (df['note'] != 'notna')",  # the word in a text decides nothing
    ],
)
def test_how_an_expression_is_written_never_counts_a_missing_value_as_a_pass(
    expression: str,
) -> None:
    df = pd.DataFrame({"sbp": [120.0, 300.0, np.nan, 130.0], "note": ["x"] * 4})
    found = family_checks(df, {"SBP": {"expression": expression}})
    assert (found.checks, found.fails) == (3, 1)


def test_a_date_that_cannot_be_read_is_no_pass() -> None:
    """errors="coerce" turns an unreadable date into a missing value: not a value the rule checked."""
    df = pd.DataFrame(
        {
            "admission": ["2021-03-01", "2021-03-05", "unknown", "2021-04-01"],
            "discharge": ["2021-03-04", "2021-03-01", "2021-03-09", "not done"],
        }
    )
    expression = "pd.to_datetime(df['discharge'], errors='coerce') < pd.to_datetime(df['admission'], errors='coerce')"
    found = family_checks(df, {"Discharge first": {"expression": expression}})
    assert (found.checks, found.fails) == (2, 1)


def test_bounds_make_one_check_per_column() -> None:
    df = pd.DataFrame({"x": [1.0, 50.0, np.nan], "y": [1.0, 2.0, 70.0]})
    found = family_checks(df, {"Frozen": {"bounds": {"x": [0, 10], "y": [0, 10]}}})
    assert [c.unit for c in found.units] == ["Frozen: x", "Frozen: y"]
    assert (found.checks, found.fails) == (5, 2)  # x on 2 rows, y on 3


def test_two_checks_of_the_same_name_are_both_counted() -> None:
    """A rule named "Frozen: ldl" beside a bounds rule "Frozen" on ldl: before, one replaced the other."""
    df = pd.DataFrame({"ldl": [1.0, 50.0, 2.0]})
    family = {
        "Frozen: ldl": {"expression": "df['ldl'] > 10"},
        "Frozen": {"bounds": {"ldl": [0, 10]}},
    }
    found = family_checks(df, family)
    assert (found.checks, found.fails) == (6, 2)
    assert len(found.units) == 2


def test_listed_rows_count_the_rows_their_method_assessed() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0], "y": [1.0, np.nan, 3.0, 4.0]})
    lof = {"row_ids": [[3, 3]], "method": "Local Outlier Factor", "columns": ["x", "y"]}
    iqr = {"row_ids": [[3, 3]], "method": "iqr", "columns": ["x", "y"]}
    renamed = {"row_ids": [[3, 3]], "method": "iqr", "columns": ["x", "gone"]}
    assert (family_checks(df, {"LOF": lof}).checks, family_checks(df, {"LOF": lof}).fails) == (2, 1)
    assert family_checks(df, {"IQR": iqr}).checks == 4
    assert (
        family_checks(df, {"Renamed": renamed}).checks == 1
    )  # only the listed row: what was assessed is unknown


def test_an_imputed_value_is_no_check() -> None:
    df = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 99.0]}, index=pd.Index([10, 11, 12, 13], name="__row_id")
    )
    family = {"A": {"expression": "df['a'] > 50"}}
    assert family_checks(df, family).checks == 4
    imputed = {"a": np.array([11, 12])}
    assert family_checks(df, family, imputed=imputed).checks == 2
    assert _score(df, family, imputed=imputed) == 50.0


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


def test_a_kind_registered_without_its_own_checks_counts_the_rows_holding_its_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M6's date_order, registered without plausibility checks, counted every row: 2 of 4 rows hold no discharge."""
    kind = CriterionKind(
        "order_probe",
        "Order probe",
        mask=lambda df, name, d: (df[d["after"]] < df[d["before"]]).fillna(False).astype(bool),
        columns=lambda name, d: (d["before"], d["after"]),
        describe=lambda name, d: f"{d['after']} before {d['before']}",
    )
    kinds = CRITERION_KINDS.copy()
    kinds.register(kind)
    monkeypatch.setattr("core.criteria.CRITERION_KINDS", kinds)
    df = pd.DataFrame({"admission": [1.0, 5.0, 2.0, 3.0], "discharge": [4.0, 1.0, np.nan, np.nan]})
    family = {"Order": {"kind": "order_probe", "before": "admission", "after": "discharge"}}
    found = family_checks(df, family)
    assert (found.checks, found.fails) == (2, 1)
    assert _score(df, family) == 50.0


@pytest.mark.parametrize(
    "definition",
    [
        {"bounds": {"x": [0, 1, 2]}},
        {"bounds": {"x": ["abc", 1]}},
        {"threshold": "two", "members": {"A": {"expression": "df['x'] > 1"}}},
    ],
)
def test_a_rule_that_cannot_be_checked_is_listed_and_the_others_still_count(
    definition: dict,
) -> None:
    """Before, an error other than CriterionError blanked the whole dimension."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(CriterionError):
        rule_checks(df, "Broken", definition)
    family = {"Broken": definition, "Sound": {"expression": "df['x'] > 2"}}
    auditor = DataQualityAuditor(df, {"mask_families": {"clinical_anomalies": family}})
    assert auditor.compute_domain_validity() == failing_score(1, 3)
    details = auditor.details["Domain Validity"].set_index("Where")
    assert details.loc["Broken", "Issue"].startswith("cannot be evaluated")


def test_listed_rows_whose_columns_are_no_list_are_read_as_recording_none() -> None:
    """ "columns": 5 loaded, and the checks raised a TypeError."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    (check,) = rule_checks(df, "Listed", {"row_ids": [[1, 1]], "columns": 5})
    assert (int(check.checked.sum()), int(check.fired.sum())) == (3, 1)


@pytest.mark.parametrize(
    ("df", "family", "why"),
    [
        (pd.DataFrame({"a": [1.0]}), {}, NO_RULE),  # no rule
        (
            pd.DataFrame({"a": [np.nan, np.nan]}),
            {"A": {"expression": "df['a'] > 5"}},
            "no row holds the values the rules read",
        ),
        (
            pd.DataFrame({"a": pd.Series([], dtype=float)}),
            {"A": {"expression": "df['a'] > 5"}},
            "the table holds no row",
        ),
        (
            pd.DataFrame({"a": [1.0]}),
            {"A": {"expression": "df['missing_column'] > 5"}},
            "no rule can be evaluated (the details say why)",
        ),
    ],
)
def test_no_check_made_reads_n_a_and_says_why(df: pd.DataFrame, family: dict, why: str) -> None:
    auditor = DataQualityAuditor(df, {"mask_families": {"clinical_anomalies": family}})
    assert auditor.compute_domain_validity() is None
    assert auditor.domain_na == why


def test_the_details_say_what_each_check_holds() -> None:
    df = pd.DataFrame({"a": [1.0, 99.0, 3.0], "b": [np.nan, np.nan, np.nan]})
    family = {"A": {"expression": "df['a'] > 50"}, "B": {"expression": "df['b'] > 50"}}
    auditor = DataQualityAuditor(df, {"mask_families": {"clinical_anomalies": family}})
    assert auditor.compute_domain_validity() == failing_score(1, 3)
    details = auditor.details["Domain Validity"].set_index("Where")
    assert details.loc["A", "Issue"] == "flagged, of 3 rows checked"
    assert details.loc["A", "Count"] == 1
    assert details.loc["B", "Issue"] == "not checked: no row holds its values"


def test_in_an_all_family_the_family_s_line_counts() -> None:
    df = pd.DataFrame({"a": [9.0, 9.0, 1.0], "b": [9.0, 1.0, 9.0]})
    family = {
        "operator": "all",
        "A": {"expression": "df['a'] > 5"},
        "B": {"expression": "df['b'] > 5"},
    }
    auditor = DataQualityAuditor(df, {"mask_families": {"clinical_anomalies": family}})
    auditor.compute_domain_validity()
    details = auditor.details["Domain Validity"].set_index("Where")
    assert details.loc["A", "Issue"] == "fires, of 3 rows checked; the family's line counts"
    assert (
        details.loc["the family", "Issue"]
        == "rows where every rule fires, of 3 rows where the rules settle it"
    )


def test_one_rule_failing_often_makes_the_advice_high_whatever_the_pooled_score() -> None:
    """30 % of heights typed in metres, beside 9 rules that never fail: the pooled score stays above 90."""
    n = 1_000
    df = pd.DataFrame(
        {"height": [1.7] * 300 + [170.0] * 700, **{f"v{i}": [1.0] * n for i in range(9)}}
    )
    family = {"Height in metres": {"expression": "df['height'] < 3"}} | {
        f"V{i}": {"expression": f"df['v{i}'] > 5"} for i in range(9)
    }
    auditor = DataQualityAuditor(df, {"mask_families": {"clinical_anomalies": family}})
    assert auditor.run_audit()["Domain Validity"] == 97.0
    (advice,) = [a for a in auditor.advice if a["category"] == "Domain Validity"]
    assert advice["severity"] == "high"


def test_the_tutorial_reads_99_83_and_99_70_with_the_diabetes_rule() -> None:
    df = pd.read_csv(TUTORIAL_COHORT_PATH)
    config = json.loads(
        (Path(TUTORIAL_COHORT_PATH).parent / "tutorial_config.json").read_text(encoding="utf-8")
    )
    auditor = DataQualityAuditor(df, config)
    assert auditor.compute_domain_validity() == 99.83
    assert auditor.domain_checks == (16, 9593, 16)  # the rows flagged: still 16
    details = auditor.details["Domain Validity"].set_index("Where")
    assert details.loc["Height not in centimetres", "Issue"] == "flagged, of 1,151 rows checked"
    assert details.loc["Height not in centimetres", "Count"] == 3
    config["mask_families"]["clinical_anomalies"]["Possible undiagnosed diabetes"] = {
        "numeric": False,
        "expression": "(df['diabetes'] == 'No') & (df['hba1c_pct'] >= 6.5)",
    }
    auditor = DataQualityAuditor(df, config)
    assert auditor.compute_domain_validity() == 99.7
    assert auditor.domain_checks == (32, 10533, 32)
    details = auditor.details["Domain Validity"].set_index("Where")
    assert details.loc["Possible undiagnosed diabetes", "Issue"] == "flagged, of 940 rows checked"
    assert details.loc["Possible undiagnosed diabetes", "Count"] == 16
