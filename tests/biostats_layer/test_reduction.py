"""
Dimensionality reduction (biostats/reduction.py, M5): the columns a PCA or a FAMD can take, FAMD's typing of the
columns, and the share of the variance the components explain.

Before, Enrich offered every text column to FAMD (a date stored as text made one category per day), prince analysed
an integer column (the age in years) as one category per value, and the explained-variance plot showed percentages
on an axis that read as fractions, over the first components only: on the tutorial cohort, 3 components "explained"
0.79%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biostats.reduction import (
    FAMD,
    PCA,
    check_famd_input,
    famd_dimensions,
    famd_types,
    reduction_columns,
)


def _cohort(n: int = 120, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(40, 80, n)  # integers: a number, not 40 categories
    return pd.DataFrame(
        {
            "age": age,
            "bmi": (24 + (age - 60) * 0.05 + rng.normal(0, 3, n)).round(1),
            "arm": rng.choice(["control", "drug"], n),
            "nyha": rng.choice(["I", "II", "III", "IV"], n),
            "smoker": rng.choice([True, False], n),
        }
    )


def test_numbers_are_quantitative_and_other_columns_categories_as_in_factominer() -> None:
    assert famd_types(_cohort()) == (["age", "bmi"], ["arm", "nyha", "smoker"])
    # one per number; categories minus one per categorical column: 1 + 1 + 1 + 3 + 1
    assert famd_dimensions(_cohort()) == 7


def test_dates_identifiers_and_constants_are_not_offered() -> None:
    frame = _cohort().assign(
        visit=pd.date_range("2024-01-01", periods=120, freq="D"),
        enrol_date=[f"2024-01-{1 + i % 28:02d}" for i in range(120)],  # a date stored as text
        patient=[f"P{i:03d}" for i in range(120)],
        site="Luxembourg",
    )
    usable, excluded = reduction_columns(frame, FAMD, date_columns=["enrol_date"])
    assert usable == ["age", "bmi", "arm", "nyha", "smoker"]
    assert set(excluded) == {"visit", "enrol_date", "patient", "site"}
    assert "a date" in excluded["visit"] and "a date" in excluded["enrol_date"]
    assert "identifier" in excluded["patient"] and "single value" in excluded["site"]
    usable, excluded = reduction_columns(frame, PCA, date_columns=["enrol_date"])
    assert usable == ["age", "bmi"] and "not a number" in excluded["arm"]


@pytest.mark.parametrize(
    "frame, message",
    [
        (
            pd.DataFrame({"a": [1.0, 2.0], "d": pd.to_datetime(["2024-01-01", "2024-02-01"])}),
            "a date",
        ),
        (pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3, 1, 2]}), "use PCA"),
        (pd.DataFrame({"a": ["x", "y"], "b": ["u", "v"]}), "at least one numeric"),
    ],
)
def test_a_table_famd_cannot_analyse_is_refused_with_what_to_do(
    frame: pd.DataFrame, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        check_famd_input(frame)


def test_the_variance_profile_covers_every_dimension() -> None:
    pytest.importorskip("prince", reason="PCA and FAMD run with prince (the 'ml' extra)")
    from biostats.reduction import fit_famd, variance_profile

    frame = _cohort()
    profile = variance_profile(frame, FAMD)
    assert profile.n_dimensions == 7 and profile.complete
    assert profile.cumulative[-1] == pytest.approx(100.0, abs=1e-6)  # percentages, reaching 100%
    assert list(profile.contributions.index) == ["age", "bmi", "arm", "nyha", "smoker"]
    np.testing.assert_allclose(profile.contributions.sum(axis=0), 1.0)
    assert fit_famd(frame, 2).num_cols_ == ["age", "bmi"]  # the integer column is a number
    pca = variance_profile(frame[["age", "bmi"]], PCA)
    assert pca.n_dimensions == 2 and pca.cumulative[-1] == pytest.approx(100.0, abs=1e-6)


def test_missing_values_are_named_before_any_fit() -> None:
    from biostats.reduction import variance_profile

    frame = _cohort()
    frame.loc[:4, "bmi"] = np.nan
    with pytest.raises(ValueError, match=r"^Missing values in bmi \(5\): .* needs complete data"):
        variance_profile(frame, FAMD)
    with pytest.raises(ValueError, match=r"^Missing values in bmi \(5\)"):
        variance_profile(frame[["age", "bmi"]], PCA)


def test_a_famd_with_many_dimensions_says_how_many_it_shows() -> None:
    pytest.importorskip("prince", reason="FAMD runs with prince (the 'ml' extra)")
    from biostats.reduction import MAX_PROFILE_COMPONENTS, variance_profile

    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"x": rng.normal(size=200), "code": [f"C{i % 80}" for i in range(200)]})
    profile = variance_profile(frame, FAMD)
    assert profile.n_dimensions == 80 and len(profile.percent) == MAX_PROFILE_COMPONENTS
    assert not profile.complete and 0 < profile.cumulative[-1] < 100
