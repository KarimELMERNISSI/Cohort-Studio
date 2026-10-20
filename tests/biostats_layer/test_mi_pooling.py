"""Pooling layer of multiple imputation (M2): Rubin's rules as in R mice, D1/D2, scales, Harel's R^2."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import stats

from biostats.imputation import rubin_pool
from biostats.mi import (
    NOT_POOLABLE,
    TermMismatchError,
    barnard_rubin_df,
    d1_wald,
    d2_chisq,
    not_poolable,
    pool_r_squared,
    pool_scalar,
    pool_terms,
    to_scale,
)

Q = [1.0, 1.2, 0.9, 1.1, 1.05]
U = [0.04, 0.05, 0.045, 0.04, 0.05]


# --- scalar pooling ---------------------------------------------------------------------------------


def test_scalar_pooling_without_complete_data_df_keeps_the_classic_values() -> None:
    pooled = pool_scalar(Q, U)
    assert pooled.estimate == pytest.approx(1.05)
    assert pooled.ubar == pytest.approx(0.045)
    assert pooled.b == pytest.approx(0.0125)
    assert pooled.t == pytest.approx(0.06)
    assert pooled.riv == pytest.approx(1 / 3)
    assert pooled.lam == pytest.approx(0.25)
    assert pooled.df == pytest.approx(64.0)  # (m - 1) / lambda^2
    assert pooled.fmi == pytest.approx((1 / 3 + 2 / 67) / (4 / 3))
    assert math.isinf(pooled.dfcom) and pooled.m == 5


def test_barnard_rubin_df_with_the_complete_data_df() -> None:
    """Hand-worked mice formulas (barnard.rubin, fmi, t statistic and interval) with nu_com = 50."""
    pooled = pool_scalar(Q, U, dfcom=50, term="age")
    assert pooled.term == "age"
    assert pooled.df == pytest.approx(23.07474785559431, rel=1e-12)
    assert pooled.fmi == pytest.approx(0.3075269225346766, rel=1e-12)
    assert pooled.statistic == pytest.approx(4.286607049870562, rel=1e-12)
    assert pooled.p_value == pytest.approx(0.0002739705689415813, rel=1e-10)
    assert pooled.ci.lower == pytest.approx(0.5433752521899015, rel=1e-12)
    assert pooled.ci.upper == pytest.approx(1.5566247478100985, rel=1e-12)
    assert pooled.se == pytest.approx(math.sqrt(0.06))
    assert pooled.df < pool_scalar(Q, U).df  # a finite sample never gives more df


def test_no_between_imputation_variance_gives_infinite_df_as_mice() -> None:
    pooled = pool_scalar([2.0, 2.0, 2.0], [0.25, 0.25, 0.25])
    assert pooled.b == 0 and pooled.lam == 0 and pooled.riv == 0 and pooled.fmi == 0
    assert math.isinf(pooled.df)  # mice 3.19 no longer floors lambda
    assert pooled.ci.lower == pytest.approx(2 - stats.norm.ppf(0.975) * 0.5)
    assert barnard_rubin_df(3, 0.0, 0.25, dfcom=10) == pytest.approx(
        11 / 13 * 10
    )  # the observed-data df


def test_scalar_pooling_rejects_unusable_inputs() -> None:
    with pytest.raises(ValueError, match="two imputations"):
        pool_scalar([1.0], [0.1])
    with pytest.raises(ValueError, match="same length"):
        pool_scalar([1.0, 2.0], [0.1])
    with pytest.raises(ValueError, match="missing or infinite"):
        pool_scalar([1.0, float("nan")], [0.1, 0.1])
    with pytest.raises(ValueError, match="negative"):
        pool_scalar([1.0, 2.0], [0.1, -0.1])
    with pytest.raises(ValueError, match="unknown pooling scale"):
        pool_scalar(Q, U, scale="sqrt")


def test_rubin_pool_wrapper_uses_n_minus_one_for_a_mean() -> None:
    old = rubin_pool(Q, U)
    assert old.df == pytest.approx(64.0) and old.total_variance == pytest.approx(0.06)
    with_n = rubin_pool(Q, U, n_complete=51)
    assert with_n.df == pytest.approx(pool_scalar(Q, U, dfcom=50).df)


# --- scales -----------------------------------------------------------------------------------------


def test_ratios_are_pooled_on_the_log_scale_and_reported_back() -> None:
    log_or = np.log([1.8, 2.1, 1.6, 2.4])
    pooled = pool_scalar(log_or, [0.04, 0.05, 0.04, 0.06], scale="log", term="exposure")
    assert pooled.natural_estimate == pytest.approx(math.exp(pooled.estimate))
    assert pooled.natural_ci is not None
    assert pooled.natural_ci.lower == pytest.approx(math.exp(pooled.ci.lower))
    assert pooled.natural_ci.upper == pytest.approx(math.exp(pooled.ci.upper))
    assert "log scale" in pooled.natural_ci.method


def test_proportions_are_transformed_with_the_delta_method() -> None:
    p, n = np.array([0.30, 0.34, 0.28, 0.32]), 200
    pooled = pool_scalar(p, p * (1 - p) / n, scale="logit", from_natural=True)
    assert pooled.natural_estimate is not None and 0.28 < pooled.natural_estimate < 0.34
    assert (
        pooled.natural_ci is not None and 0 < pooled.natural_ci.lower < pooled.natural_ci.upper < 1
    )
    est, var = to_scale(0.3, 0.3 * 0.7 / n, "logit")
    assert est == pytest.approx(math.log(0.3 / 0.7))
    assert var == pytest.approx((0.3 * 0.7 / n) / (0.3 * 0.7) ** 2)


def test_survival_probabilities_use_cloglog_and_keep_bounds_ordered() -> None:
    s = np.array([0.80, 0.78, 0.83, 0.79])
    pooled = pool_scalar(s, [0.0006, 0.0007, 0.0005, 0.0006], scale="cloglog", from_natural=True)
    assert (
        pooled.natural_ci is not None
        and pooled.natural_ci.lower < pooled.natural_estimate < pooled.natural_ci.upper
    )
    with pytest.raises(ValueError, match="outside the domain"):
        to_scale(1.0, 0.01, "logit")
    with pytest.raises(ValueError, match="outside the domain"):
        to_scale(0.0, 0.01, "log")


# --- tables of terms --------------------------------------------------------------------------------


def _fits(m: int = 5, seed: int = 3):
    """m analyses of one dataset whose 20 % missing outcomes are imputed differently each time."""
    rng = np.random.default_rng(seed)
    n = 150
    base = pd.DataFrame({"age": rng.normal(60, 8, n), "group": rng.choice(["a", "b", "c"], n)})
    base["y"] = 0.05 * base["age"] + (base["group"] == "b") * 0.8 + rng.normal(0, 1, n)
    imputed = rng.random(n) < 0.2
    fits = []
    for _ in range(m):
        df = base.copy()
        df.loc[imputed, "y"] = df.loc[imputed, "y"] + rng.normal(0, 0.7, int(imputed.sum()))
        fits.append(smf.ols("y ~ age + C(group)", df).fit())
    return fits


def test_pool_terms_pools_every_coefficient_like_mice() -> None:
    fits = _fits()
    table = pool_terms(
        [f.params for f in fits], covariances=[f.cov_params() for f in fits], dfcom=fits[0].df_resid
    )
    assert table.names == list(fits[0].params.index) and table.m == 5
    age = table.term("age")
    direct = pool_scalar(
        [f.params["age"] for f in fits], [f.bse["age"] ** 2 for f in fits], dfcom=fits[0].df_resid
    )
    assert age.estimate == pytest.approx(direct.estimate) and age.df == pytest.approx(direct.df)
    frame = table.to_frame()
    assert {
        "term",
        "estimate",
        "se",
        "df",
        "p_value",
        "ci_low",
        "ci_high",
        "fmi",
        "lambda",
        "riv",
    } <= set(frame)
    with pytest.raises(KeyError):
        table.term("bmi")
    by_variance = pool_terms([f.params for f in fits], variances=[f.bse**2 for f in fits])
    assert by_variance.term("age").estimate == pytest.approx(age.estimate)
    with pytest.raises(ValueError, match="covariance"):
        by_variance.wald(["age"])


def test_terms_must_be_the_same_in_every_imputation() -> None:
    fits = _fits()
    params = [f.params for f in fits]
    params[2] = params[2].drop("C(group)[T.c]")
    with pytest.raises(TermMismatchError, match="imputation 3"):
        pool_terms(params, variances=[f.bse**2 for f in fits])
    with pytest.raises(ValueError, match="covariance matrices or the variances"):
        pool_terms([f.params for f in fits])


def test_array_input_and_per_term_scales() -> None:
    q = np.log(np.array([[1.5, 2.0], [1.7, 2.2], [1.6, 1.9]]))
    table = pool_terms(
        q, variances=np.full((3, 2), 0.02), terms=["or_a", "or_b"], scales={"or_a": "log"}
    )
    assert (
        table.term("or_a").natural_estimate is not None
        and table.term("or_b").natural_estimate is None
    )
    with pytest.raises(ValueError, match="terms do not match"):
        pool_terms(q, variances=np.full((3, 2), 0.02), terms=["only_one"])


# --- multi-parameter tests --------------------------------------------------------------------------


def test_d1_for_one_parameter_is_the_squared_t_statistic() -> None:
    q = np.array(Q).reshape(-1, 1)
    u = np.array(U).reshape(-1, 1, 1)
    test = d1_wald(q, u)
    assert test.statistic == pytest.approx(pool_scalar(Q, U).statistic ** 2)
    assert test.df1 == 1 and test.method == "D1"


def test_d1_wald_test_of_a_factor_with_lrr_and_reiter_df() -> None:
    fits = _fits()
    table = pool_terms([f.params for f in fits], covariances=[f.cov_params() for f in fits])
    terms = ["C(group)[T.b]", "C(group)[T.c]"]
    large = table.wald(terms)
    q = np.array([[f.params[t] for t in terms] for f in fits])
    u = np.array([f.cov_params().loc[terms, terms].to_numpy() for f in fits])
    m, k = q.shape
    ubar, b = u.mean(axis=0), np.cov(q, rowvar=False, ddof=1)
    r = (1 + 1 / m) * np.trace(b @ np.linalg.inv(ubar)) / k
    stat = q.mean(0) @ np.linalg.inv(ubar) @ q.mean(0) / (k * (1 + r))
    t = k * (m - 1)
    assert large.statistic == pytest.approx(stat)
    assert large.df2 == pytest.approx(4 + (t - 4) * (1 + (1 - 2 / t) / r) ** 2)
    assert large.p_value == pytest.approx(stats.f.sf(stat, k, large.df2))
    assert large.p_value < 0.01  # the planted group effect
    small = d1_wald(q, u, dfcom=fits[0].df_resid, terms=terms)
    assert small.df2 < large.df2 and any("Reiter" in ref for ref in small.references)


def test_d1_edge_cases() -> None:
    q = np.ones((3, 2))
    u = np.array([np.eye(2) * 0.1] * 3)
    same = d1_wald(q, u)
    assert math.isinf(same.df2) and same.p_value == pytest.approx(
        stats.chi2.sf(same.statistic * 2, 2)
    )
    # no variance between the imputations with a finite complete-data df: that df (vstar), as mitml's .D1;
    # F(k, inf) was used, depending on whether rounding left the between variance at exactly zero
    finite = d1_wald(q, u, dfcom=40)
    assert finite.df2 == pytest.approx(41 / 43 * 40)
    assert finite.p_value == pytest.approx(stats.f.sf(finite.statistic, 2, 41 / 43 * 40))
    few = d1_wald(np.array([[1.0], [1.3]]), np.array([[[0.1]], [[0.1]]]), dfcom=30)
    assert "Too few imputations" in few.note
    with pytest.raises(ValueError, match="singular"):
        d1_wald(np.array([[1.0, 2.0], [1.1, 2.1]]), np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="m x k"):
        d1_wald(np.ones(3), np.ones((3, 1, 1)))


def test_d2_combines_chi_square_statistics() -> None:
    same = d2_chisq([6.0, 6.0, 6.0], df=2)
    assert same.statistic == pytest.approx(3.0) and math.isinf(same.df2)
    assert same.p_value == pytest.approx(stats.chi2.sf(6.0, 2))
    d = np.array([5.1, 7.3, 6.2, 4.8, 6.9])
    test = d2_chisq(d, df=2)
    r = (1 + 1 / 5) * np.var(np.sqrt(d), ddof=1)
    expected = (d.mean() / 2 - (6 / 4) * r) / (1 + r)
    assert test.statistic == pytest.approx(expected)
    assert test.df2 == pytest.approx(2 ** (-3 / 5) * 4 * (1 + 1 / r) ** 2)
    assert "approximate" in test.note
    low = d2_chisq([0.01, 3.0, 0.02], df=1)
    assert low.statistic <= 0 and low.p_value == 1.0
    with pytest.raises(ValueError):
        d2_chisq([1.0], df=1)
    with pytest.raises(ValueError):
        d2_chisq([1.0, -2.0], df=1)


def test_harel_pooled_r_squared() -> None:
    same = pool_r_squared([0.36, 0.36, 0.36], n=100)
    assert same.estimate == pytest.approx(0.36)
    assert same.ci is not None and same.ci.lower < 0.36 < same.ci.upper
    varied = pool_r_squared([0.30, 0.36, 0.42])
    assert 0.30 < varied.estimate < 0.42 and varied.ci is None
    with pytest.raises(ValueError):
        pool_r_squared([0.3, 1.2])


def test_not_poolable_registry() -> None:
    entry = not_poolable("mann_whitney")
    assert entry is not None and "Rubin" in entry.reason and entry.alternative
    assert not_poolable("cox") is None
    assert all(e.reason and e.alternative for e in NOT_POOLABLE.values())
