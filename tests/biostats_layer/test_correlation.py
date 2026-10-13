"""Mixed-type association matrix with CIs and family-wise adjustment."""

import numpy as np
import pandas as pd
import pytest

from biostats.correlation import (
    association_matrix,
    correlation_ratio,
    pearson_ci,
    spearman_ci,
    variable_types,
)


def _frame(n: int = 120, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 2 * x + rng.normal(scale=0.5, size=n)
    b = np.where(x > 0, "Yes", "No")
    c = np.select([x < -0.5, x > 0.5], ["low", "high"], default="mid")
    noise = rng.normal(size=n)
    sparse = np.where(rng.random(n) < 0.95, np.nan, x)  # too few overlaps
    return pd.DataFrame({"x": x, "y": y, "b": b, "c": c, "noise": noise, "sparse": sparse})


def test_variable_types() -> None:
    types = variable_types(_frame())
    assert types == {
        "x": "numeric",
        "y": "numeric",
        "b": "binary",
        "c": "categorical",
        "noise": "numeric",
        "sparse": "numeric",
    }
    assert variable_types(pd.DataFrame({"flag": [0, 1, 0, 1]})) == {"flag": "binary"}


def test_fisher_and_bonett_intervals() -> None:
    ci = pearson_ci(0.5, 30)
    assert ci.lower == pytest.approx(0.1706, abs=1e-3) and ci.upper == pytest.approx(
        0.7288, abs=1e-3
    )
    wide = spearman_ci(0.5, 30)
    assert wide.lower < ci.lower and wide.upper > ci.upper  # Bonett-Wright is wider for rho
    assert np.isnan(pearson_ci(1.0, 30).lower)


def test_correlation_ratio() -> None:
    eta, p, n = correlation_ratio(pd.Series(["a", "a", "b", "b"]), pd.Series([1, 2, 1, 2]))
    assert eta == pytest.approx(0.0) and n == 4
    eta2, p2, _ = correlation_ratio(pd.Series(["a", "a", "b", "b"]), pd.Series([1, 1.1, 5, 5.1]))
    assert eta2 > 0.99 and p2 < 0.01


def test_association_matrix_measures_by_type() -> None:
    res = association_matrix(_frame(), method="pearson", min_periods=10)
    pairs = res.pairs.set_index(["var_1", "var_2"])
    assert (
        pairs.loc[("x", "y"), "measure"] == "Pearson r" and pairs.loc[("x", "y"), "estimate"] > 0.9
    )
    assert (
        pairs.loc[("x", "y"), "ci_low"]
        < pairs.loc[("x", "y"), "estimate"]
        < pairs.loc[("x", "y"), "ci_high"]
    )
    assert (
        pairs.loc[("x", "b"), "measure"] == "point-biserial r"
        and pairs.loc[("x", "b"), "estimate"] > 0.6
    )
    assert (
        pairs.loc[("x", "c"), "measure"] == "correlation ratio eta"
        and pairs.loc[("x", "c"), "estimate"] > 0.8
    )
    assert (
        pairs.loc[("b", "c"), "measure"].startswith("Cramer")
        and pairs.loc[("b", "c"), "estimate"] > 0.5
    )
    assert np.isnan(pairs.loc[("x", "sparse"), "estimate"]) and pairs.loc[("x", "sparse"), "n"] < 10
    assert (res.pairs["p_adj"].dropna() >= res.pairs["p_raw"].dropna()).all()
    assert res.matrix.shape == (6, 6) and np.allclose(np.diag(res.matrix), 1.0)
    assert res.matrix.loc["x", "y"] == res.matrix.loc["y", "x"]
    spearman = association_matrix(_frame(), columns=["x", "y"], method="spearman")
    assert spearman.pairs.loc[0, "measure"] == "Spearman rho"
    with pytest.raises(ValueError):
        association_matrix(_frame(), method="kendall")
