"""A statistic shown under one name is computed one way (M5): Cramer's V is bias-corrected everywhere, and Describe's
Shapiro-Wilk p-value is the analyses' (D'Agostino's K2 beyond 5,000 values)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biostats.effect_sizes import cramers_v
from biostats.missing import scan_missingness_dependencies
from biostats.normality import shapiro_p
from utils.statistics_utils import normality_test


def test_the_missingness_scan_reports_the_bias_corrected_v() -> None:
    # the scan reported the uncorrected V (0.500 on [[3, 1], [1, 3]] against 0.354 corrected) under the same name
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({"arm": rng.choice(["a", "b", "c"], n), "ldl": rng.normal(3, 1, n)})
    df.loc[df["arm"] == "a", "ldl"] = np.where(rng.random((df["arm"] == "a").sum()) < 0.6, np.nan, 3.0)
    scan = scan_missingness_dependencies(df, targets=["ldl"], predictors=["arm"])
    row = scan.table[scan.table["Predictor Variable"] == "arm"].iloc[0]
    table = pd.crosstab(df["arm"].astype(str), df["ldl"].isna())
    expected = cramers_v(table.to_numpy(dtype=float), n_boot=0).estimate
    assert row["Effect"] == f"Cramer's V (bias-corrected) = {expected:.2f}"


def test_describe_uses_the_analyses_normality_test() -> None:
    rng = np.random.default_rng(3)
    large = pd.Series(rng.normal(0, 1, 6000))
    assert normality_test(large, method="shapiro") == pytest.approx(round(shapiro_p(large), 4))
    small = pd.Series(rng.normal(0, 1, 50))
    assert normality_test(small, method="shapiro") == pytest.approx(round(shapiro_p(small), 4))
    assert np.isnan(normality_test(pd.Series([1.0, 1.0, 1.0, 1.0]), method="shapiro"))  # no variance
