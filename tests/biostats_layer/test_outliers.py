"""The outlier detector shared by the Validate page, the quality dashboard and the health score (M5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biostats.outliers import DEFAULT_PARAMS, detect_outliers, is_continuous, univariate_bounds


def _frame(n: int = 200, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "sbp": rng.normal(130, 12, n),
            "ldl": rng.lognormal(1.0, 0.4, n),
            "visits": pd.array(rng.poisson(4, n), dtype="Int64"),  # a nullable integer column
            "diabetic": rng.integers(0, 2, n),  # 0/1: no outliers
            "arm": rng.choice(["control", "drug"], n),
            "patient_id": [f"P{i:04d}" for i in range(n)],  # an identifier
        }
    )
    df.loc[[3, 17], "sbp"] = [250.0, 20.0]
    df.loc[5, "ldl"] = np.nan
    return df


@pytest.mark.parametrize("method", ["zscore", "iqr", "quantile"])
def test_a_value_outside_the_bounds_is_flagged(method: str) -> None:
    df = _frame()
    detection = detect_outliers(df, ["sbp", "ldl", "visits", "diabetic"], method)
    assert detection.columns == ["sbp", "ldl", "visits"] and detection.skipped == ["diabetic"]
    for col, (low, high) in detection.bounds.items():
        values = df[col].astype(float)
        assert detection.flags[col].equals(((values < low) | (values > high)).fillna(False))
    assert {3, 17} <= set(detection.flagged)  # the two planted blood pressures


def test_the_bounds_of_each_method() -> None:
    values = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    low, high = univariate_bounds(values, "zscore", {"threshold": 2}) or (0, 0)
    assert (low, high) == pytest.approx(
        (values.mean() - 2 * values.std(), values.mean() + 2 * values.std())
    )
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    assert univariate_bounds(values, "iqr") == pytest.approx(
        (q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
    )
    assert univariate_bounds(values, "quantile") == pytest.approx(
        (values.quantile(0.01), values.quantile(0.99))
    )
    assert univariate_bounds(pd.Series([0, 1, 0, 0]), "iqr") is None  # binary
    assert univariate_bounds(pd.Series(["a", "b", "c"]), "iqr") is None and not is_continuous(
        pd.Series([True, False])
    )
    with pytest.raises(ValueError, match="not a univariate method"):
        univariate_bounds(values, "DBSCAN")
    with pytest.raises(ValueError, match="unknown outlier method"):
        detect_outliers(pd.DataFrame({"x": values}), ["x"], "grubbs")


@pytest.mark.parametrize("method", ["Isolation Forest", "Local Outlier Factor", "DBSCAN"])
def test_a_multivariate_method_assesses_the_complete_rows_only(method: str) -> None:
    df = _frame()
    detection = detect_outliers(df, ["sbp", "ldl", "visits", "arm", "patient_id"], method)
    assert detection.skipped == ["patient_id"] and detection.columns == [
        "sbp",
        "ldl",
        "visits",
        "arm",
    ]
    # the row with a missing ldl is not assessed
    assert detection.not_assessed == 1 and not detection.flags.loc[5, "overall"]
    assert np.isnan(detection.scores.loc[5]) and detection.scores.drop(index=5).notna().all()
    again = detect_outliers(df, ["sbp", "ldl", "visits", "arm", "patient_id"], method)
    assert detection.flags.equals(again.flags)  # seeded: the same rows every time
    assert detection.params == DEFAULT_PARAMS[method]


def test_isolation_forest_finds_the_planted_rows() -> None:
    detection = detect_outliers(
        _frame(), ["sbp", "ldl"], "Isolation Forest", {"contamination": 0.02}
    )
    assert {3, 17} <= set(detection.flagged)


def test_too_few_complete_rows_are_not_assessed() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [1.0, np.nan, 3.0]})
    detection = detect_outliers(df, ["a", "b"], "Local Outlier Factor")
    assert detection.flagged.empty and detection.not_assessed == 2
