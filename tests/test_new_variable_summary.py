"""A new variable's statistics resist a few entry errors, and the errors are found (Enrich > Create New Variables)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from enrich.formulas import evaluate_formula_safely, far_outside, numeric_summary


def _cohort() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"height_cm": rng.normal(172, 9, 200), "weight_kg": rng.normal(80, 12, 200)})
    df.loc[3, "height_cm"] = 1.68  # entered in metres
    df.loc[7, "weight_kg"] = 0.0  # missing coded as 0
    df.loc[9, "height_cm"] = 0.0  # division by zero
    return df


def test_entry_errors_are_found_and_do_not_hide_the_typical_values() -> None:
    ok, _, df = evaluate_formula_safely("weight_kg / (height_cm / 100) ** 2", "bmi", _cohort())
    assert ok
    summary = dict(numeric_summary(df["bmi"]).itertuples(index=False))
    assert summary["Infinite (division by zero)"] == "1" and summary["N (values)"] == "199"
    assert 24 < float(summary["Median"]) < 30  # the median shows a BMI
    assert float(summary["Max"]) > 10_000  # the height in metres
    extreme, low, high = far_outside(df["bmi"])
    # the three entry errors, and possibly a genuine extreme the user is asked to check (a BMI above 50)
    assert {3, 7, 9} <= set(extreme) and len(extreme) <= 4
    assert low is not None and high is not None and low < 24 < 30 < high


def test_a_clean_variable_has_no_far_out_values() -> None:
    values = pd.Series(np.random.default_rng(1).normal(27, 4, 500))
    extreme, _, _ = far_outside(values)
    assert len(extreme) == 0
    assert far_outside(pd.Series([1.0, 2.0]))[0].empty  # too few values to judge
