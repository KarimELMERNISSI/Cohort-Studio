"""
Group means compared without assuming equal variances (biostats.linear, M5).

On the observed data, three groups or more are compared with Welch's ANOVA, which gives each group its own
variance. Under multiple imputation the pooled comparison used a linear model with equal variances, because
Welch's ANOVA has no pooling rule. ``welch_means`` is the poolable counterpart: HC2 variances, each group's own
s^2 / n, with Welch-type degrees of freedom. The biostatistician's review found that HC3 with the residual df, the
first choice, rejected too often when the small groups varied most.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.oneway import anova_oneway

from biostats.design import design_matrix
from biostats.linear import FEW_VALUES, ancova, linear_regression, welch_means


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


def test_on_complete_data_the_test_is_welchs_anova() -> None:
    df = _groups()
    result = welch_means(df, "y", "g")
    arrays = [df.loc[df["g"] == g, "y"].to_numpy() for g in "abc"]
    welch = anova_oneway(arrays, use_var="unequal", welch_correction=True)
    row = result.type2.set_index("source").loc["g"]
    assert row["F"] == pytest.approx(welch.statistic, rel=1e-10)
    assert result.factor_df["g"] == pytest.approx(welch.df_denom, rel=1e-10)
    assert row["p"] == pytest.approx(welch.pvalue, rel=1e-8)
    assert result.cov_type == "HC2" and any("Welch BL (1951)" in r for r in result.references)
    # what D1 pools is the Wald form of these variances: Welch's F without its second-order correction
    terms = result.terms_of["g"]
    b = result.params[terms].to_numpy()
    wald = float(b @ np.linalg.solve(result.vcov.loc[terms, terms].to_numpy(), b)) / len(terms)
    uncorrected = anova_oneway(arrays, use_var="unequal", welch_correction=False)
    assert wald == pytest.approx(uncorrected.statistic, rel=1e-10)


def test_for_two_groups_it_is_welchs_t_test() -> None:
    df = _groups()
    two = df[df["g"].isin(["a", "c"])]
    result = welch_means(two, "y", "g")
    welch = stats.ttest_ind(
        two.loc[two["g"] == "c", "y"], two.loc[two["g"] == "a", "y"], equal_var=False
    )
    term = result.table.set_index("term").loc["g_c"]
    assert term["p"] == pytest.approx(welch.pvalue, rel=1e-10)
    assert term["t"] == pytest.approx(welch.statistic, rel=1e-10)
    assert result.term_df["g_c"] == pytest.approx(welch.df, rel=1e-10)
    assert result.factor_df["g"] == pytest.approx(welch.df, rel=1e-10)  # Welch's df2 for k = 2


def test_each_group_mean_keeps_its_own_variance_and_df() -> None:
    df = _groups()
    means = welch_means(df, "y", "g").adjusted_means
    assert means is not None
    for level, values in df.groupby("g")["y"]:
        row = means.set_index("group").loc[level]
        n, sd = len(values), values.std(ddof=1)
        assert row["mean"] == pytest.approx(values.mean(), rel=1e-12)
        assert row["se"] == pytest.approx(sd / np.sqrt(n), rel=1e-10)  # HC2: Welch's s / sqrt(n)
        assert row["df"] == n - 1
        half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        assert row["ci_high"] - row["mean"] == pytest.approx(half, rel=1e-10)


def test_the_robust_variances_are_the_sandwich_with_leverage_weights() -> None:
    df = _groups()
    x, _ = design_matrix(df[["y", "g", "x"]].dropna(), ["g", "x"])
    xa = x.to_numpy(dtype=float)
    bread = np.linalg.inv(xa.T @ xa)
    leverage = np.einsum("ij,jk,ik->i", xa, bread, xa)
    for cov_type, power in (("HC2", 1), ("HC3", 2)):
        fit = linear_regression(df, "y", ["g", "x"], cov_type=cov_type)
        assert list(x.columns) == list(fit.params.index)
        weights = fit.residuals**2 / (1 - leverage) ** power
        expected = bread @ (xa.T @ (xa * weights[:, None])) @ bread
        np.testing.assert_allclose(fit.vcov.to_numpy(), expected, rtol=1e-10)
        se = np.sqrt(np.diag(expected))
        p = 2 * stats.t.sf(np.abs(fit.params.to_numpy() / se), fit.df_resid)
        np.testing.assert_allclose(fit.table["p"], p, rtol=1e-9)
        assert fit.cov_type == cov_type and any("MacKinnon" in r for r in fit.references)


def test_a_robust_table_shows_no_sum_of_squares_beside_its_f() -> None:
    # a robust F does not compare mean squares; partial eta squared still describes the OLS fit
    df = _groups()
    robust = linear_regression(df, "y", ["g"], cov_type="HC3").type2.set_index("source").loc["g"]
    same = linear_regression(df, "y", ["g"]).type2.set_index("source").loc["g"]
    assert np.isnan(robust["sum_sq"]) and not np.isnan(same["sum_sq"])
    assert robust["partial_eta2"] == pytest.approx(same["partial_eta2"], rel=1e-12)
    # the variances differ: so do the tests
    assert robust["F"] != pytest.approx(same["F"], rel=1e-3)


def test_small_or_constant_groups() -> None:
    one = pd.concat(
        [_groups(), pd.DataFrame({"y": [4.0], "g": ["d"], "x": [0.0]})], ignore_index=True
    )
    with pytest.raises(ValueError, match="leverage 1"):
        welch_means(one, "y", "g")
    assert ancova(one, "y", "g").adjusted_means is not None  # the equal-variance model still runs
    few = pd.concat(
        [_groups(), pd.DataFrame({"y": [4.0, 5.5, 7.0], "g": ["d"] * 3, "x": [0.0] * 3})],
        ignore_index=True,
    )
    warnings = welch_means(few, "y", "g").warnings
    # d has 3 values; a has 8, so it is not listed
    assert any(f"fewer than {FEW_VALUES} values: d (3)." in w for w in warnings)
    flat = pd.concat(
        [_groups(), pd.DataFrame({"y": [4.0, 4.0], "g": ["d", "d"], "x": [0.0, 0.0]})],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="variance of zero"):
        welch_means(flat, "y", "g")
    with pytest.raises(
        ValueError, match="absent from these rows"
    ):  # a selected group absent from the data
        welch_means(_groups(), "y", "g", levels={"g": ["a", "b", "c", "z"]})


def test_an_unknown_variance_estimator_is_refused() -> None:
    with pytest.raises(ValueError, match="cov_type is one of"):
        linear_regression(_groups(), "y", ["g"], cov_type="HC1")


@pytest.mark.slow
def test_welch_means_rejects_about_as_often_as_it_should_on_complete_data() -> None:
    # no difference between the means, and the smallest group varies most (complete data, 5% level): the
    # equal-variance F test rejects far too often, Welch-type means about as often as they should
    rng = np.random.default_rng(11)
    sizes, sds = (10, 20, 40), (4.0, 2.0, 1.0)
    rejected = {"equal": 0, "welch": 0}
    runs = 400
    for _ in range(runs):
        frame = pd.DataFrame(
            {
                "y": np.concatenate(
                    [rng.normal(0, sd, n) for n, sd in zip(sizes, sds, strict=True)]
                ),
                "g": np.repeat(["a", "b", "c"], sizes),
            }
        )
        equal = linear_regression(frame, "y", ["g"]).type2.set_index("source").loc["g", "p"]
        welch = welch_means(frame, "y", "g").type2.set_index("source").loc["g", "p"]
        rejected["equal"] += bool(equal < 0.05)
        rejected["welch"] += bool(welch < 0.05)
    assert rejected["equal"] / runs > 0.15
    assert 0.02 < rejected["welch"] / runs < 0.08
