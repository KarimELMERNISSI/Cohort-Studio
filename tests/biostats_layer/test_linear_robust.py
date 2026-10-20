"""
Robust (HC3) variances in the linear model (biostats.linear, M5).

On the observed data, three groups or more are compared with Welch's ANOVA, which gives each group its own
variance. Under multiple imputation the pooled comparison used a linear model with equal variances, because
Welch's ANOVA has no pooling rule. The linear model with HC3 variances is poolable (D1) and, like Welch's ANOVA,
does not assume equal variances.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from biostats.design import design_matrix
from biostats.linear import ancova, linear_regression


def _groups(seed: int = 3) -> pd.DataFrame:
    """Three groups of different sizes and variances; the smallest group varies most."""
    rng = np.random.default_rng(seed)
    sizes, means, sds = (8, 15, 30), (5.0, 6.0, 5.5), (3.0, 1.5, 0.8)
    return pd.DataFrame(
        {
            "y": np.concatenate(
                [rng.normal(m, s, n) for n, m, s in zip(sizes, means, sds, strict=True)]
            ),
            "g": np.repeat(["a", "b", "c"], sizes),
            "x": rng.normal(0, 1, sum(sizes)),
        }
    )


def test_each_group_mean_keeps_its_own_variance() -> None:
    # one-way layout: the HC3 variance of a group mean is s^2 / (n - 1) for the group's own s and n, where the
    # equal-variance model uses the pooled residual variance over n
    df = _groups()
    robust = ancova(df, "y", "g", cov_type="HC3")
    classical = ancova(df, "y", "g")
    assert robust.adjusted_means is not None and classical.adjusted_means is not None
    robust_means = robust.adjusted_means.set_index("group")
    classical_means = classical.adjusted_means.set_index("group")
    for level, values in df.groupby("g")["y"]:
        n, sd = len(values), values.std(ddof=1)
        assert robust_means.loc[level, "mean"] == pytest.approx(values.mean(), rel=1e-12)
        assert robust_means.loc[level, "se"] == pytest.approx(sd / np.sqrt(n - 1), rel=1e-10)
        assert classical_means.loc[level, "se"] == pytest.approx(
            classical.sigma / np.sqrt(n), rel=1e-10
        )


def test_hc3_is_the_sandwich_with_leverage_weights() -> None:
    df = _groups()
    fit = linear_regression(df, "y", ["g", "x"], cov_type="HC3")
    x, _ = design_matrix(df[["y", "g", "x"]].dropna(), ["g", "x"])
    assert list(x.columns) == list(fit.params.index)
    xa = x.to_numpy(dtype=float)
    bread = np.linalg.inv(xa.T @ xa)
    leverage = np.einsum("ij,jk,ik->i", xa, bread, xa)
    weights = (fit.residuals / (1 - leverage)) ** 2
    expected = bread @ (xa.T @ (xa * weights[:, None])) @ bread
    np.testing.assert_allclose(fit.vcov.to_numpy(), expected, rtol=1e-10)
    se = np.sqrt(np.diag(expected))
    np.testing.assert_allclose(fit.table["se"], se, rtol=1e-10)
    np.testing.assert_allclose(
        fit.table["p"], 2 * stats.t.sf(np.abs(fit.params.to_numpy() / se), fit.df_resid), rtol=1e-9
    )
    assert fit.cov_type == "HC3"
    assert any("MacKinnon" in r for r in fit.references) and any(
        "Long JS" in r for r in fit.references
    )


def test_the_robust_test_of_a_factor_and_its_effect_size() -> None:
    df = _groups()
    robust = linear_regression(df, "y", ["g"], cov_type="HC3")
    classical = linear_regression(df, "y", ["g"])
    terms = robust.terms_of["g"]
    b = robust.params[terms].to_numpy()
    f_robust = float(b @ np.linalg.solve(robust.vcov.loc[terms, terms].to_numpy(), b)) / len(terms)
    row = robust.type2.set_index("source").loc["g"]
    same = classical.type2.set_index("source").loc["g"]
    assert row["F"] == pytest.approx(f_robust, rel=1e-12)
    assert row["p"] == pytest.approx(stats.f.sf(f_robust, len(terms), robust.df_resid), rel=1e-10)
    assert row["F"] != pytest.approx(same["F"], rel=1e-3)  # the variances differ: so do the tests
    # the sums of squares and partial eta squared describe the OLS fit, whatever the variance estimator
    assert row["sum_sq"] == pytest.approx(same["sum_sq"], rel=1e-12)
    assert row["partial_eta2"] == pytest.approx(same["partial_eta2"], rel=1e-12)


def test_a_group_of_one_value_has_no_robust_variance() -> None:
    df = pd.concat(
        [_groups(), pd.DataFrame({"y": [4.0], "g": ["d"], "x": [0.0]})], ignore_index=True
    )
    with pytest.raises(ValueError, match="leverage 1"):
        ancova(df, "y", "g", cov_type="HC3")
    assert ancova(df, "y", "g").adjusted_means is not None  # the equal-variance model still runs


def test_an_unknown_variance_estimator_is_refused() -> None:
    with pytest.raises(ValueError, match="cov_type is one of"):
        linear_regression(_groups(), "y", ["g"], cov_type="HC1")


@pytest.mark.slow
def test_the_robust_test_keeps_its_level_when_the_variances_differ() -> None:
    # no difference between the means; the smallest group varies most. The equal-variance F test rejected in 23%
    # of 300 runs at the 5% level, the HC3 test in 7%
    rng = np.random.default_rng(7)
    sizes, sds = (10, 20, 40), (4.0, 2.0, 1.0)
    rejected = {"nonrobust": 0, "HC3": 0}
    runs = 300
    for _ in range(runs):
        frame = pd.DataFrame(
            {
                "y": np.concatenate(
                    [rng.normal(0, sd, n) for n, sd in zip(sizes, sds, strict=True)]
                ),
                "g": np.repeat(["a", "b", "c"], sizes),
            }
        )
        for cov_type in rejected:
            fit = linear_regression(frame, "y", ["g"], cov_type=cov_type)
            rejected[cov_type] += bool(fit.type2.set_index("source").loc["g", "p"] < 0.05)
    assert rejected["nonrobust"] / runs > 0.15
    assert rejected["HC3"] / runs < 0.09
