"""What pooling needs from each analysis (M2): frozen designs, coefficients, covariances, df and SEs."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.duration.survfunc import SurvfuncRight

from biostats import MIInfo, TestResult
from biostats.contingency import TwoByTwo, analyze_twobytwo
from biostats.design import design_matrix, factor_levels
from biostats.inference import compare_groups
from biostats.paired import one_sample_t, paired_t
from biostats.regression import count_regression, logistic_regression
from biostats.survival import cox_regression, kaplan_meier


def _cohort(n: int = 240, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.normal(60, 9, n),
            "site": rng.choice(["Metz", "Ettelbruck", "Luxembourg"], n),
            "smoker": rng.choice(["no", "yes"], n),
        }
    )
    lin = -4 + 0.05 * df["age"] + 0.7 * (df["smoker"] == "yes")
    df["event"] = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
    mu = np.exp(0.3 + 0.01 * df["age"])
    df["visits"] = rng.negative_binomial(2, 2 / (2 + mu))  # overdispersed counts
    df["time"] = rng.exponential(np.exp(3 - 0.02 * df["age"]))
    return df


# --- design -----------------------------------------------------------------------------------------


def test_design_matches_the_previous_get_dummies_layout() -> None:
    df = _cohort()
    x, refs = design_matrix(df, ["site", "age", "smoker"])
    legacy = pd.get_dummies(df[["site", "age", "smoker"]], drop_first=True, dtype=float)
    assert list(x.columns) == ["const", *legacy.columns]  # numeric first, then the dummies
    assert refs == {"site": "Ettelbruck", "smoker": "no"}
    np.testing.assert_allclose(x.drop(columns="const").to_numpy(), legacy.to_numpy())
    assert factor_levels(df, ["site", "age"]) == {"site": ["Ettelbruck", "Luxembourg", "Metz"]}


def test_frozen_levels_give_the_same_terms_or_say_which_level_is_absent() -> None:
    df = _cohort()
    levels = factor_levels(df, ["site", "smoker"])
    free, _ = design_matrix(df, ["site", "age"])
    frozen, _ = design_matrix(df, ["site", "age"], levels=levels)
    assert (
        list(frozen.columns)
        == list(free.columns)
        == ["const", "age", "site_Luxembourg", "site_Metz"]
    )
    without_metz = df[df["site"] != "Metz"]
    assert "site_Metz" not in design_matrix(without_metz, ["site", "age"])[0].columns
    with pytest.raises(ValueError, match="site: Metz"):
        design_matrix(without_metz, ["site", "age"], levels=levels)
    without_reference = df[df["site"] != "Ettelbruck"]
    with pytest.raises(ValueError, match="site: Ettelbruck"):
        design_matrix(without_reference, ["site"], levels=levels)
    odd = df.copy()
    odd.loc[odd.index[0], "site"] = "Arlon"
    with pytest.raises(ValueError, match="outside the levels"):
        design_matrix(odd, ["site"], levels=levels)


# --- regression and Cox -----------------------------------------------------------------------------


def test_logistic_regression_exposes_coefficients_covariance_and_df() -> None:
    df = _cohort()
    res = logistic_regression(df, "event", ["age", "smoker", "site"])
    assert res.params is not None and res.vcov is not None
    assert list(res.params.index) == ["const", "age", "smoker_yes", "site_Luxembourg", "site_Metz"]
    table = res.table.set_index("term")
    np.testing.assert_allclose(
        np.sqrt(np.diag(res.vcov.loc[table.index, table.index])), table["se"], rtol=1e-12
    )
    assert res.df_resid == res.n - 5
    levels = factor_levels(df, ["smoker", "site"])
    frozen = logistic_regression(df, "event", ["age", "smoker", "site"], levels=levels)
    assert frozen.params is not None and list(frozen.params.index) == list(res.params.index)
    with pytest.raises(ValueError, match="absent from these rows.*site: Metz"):
        logistic_regression(
            df[df["site"] != "Metz"].copy(), "event", ["age", "site"], levels=levels
        )


def test_count_regression_keeps_only_the_regression_coefficients() -> None:
    df = _cohort()
    nb = count_regression(df, "visits", ["age", "smoker"], family="negative_binomial")
    assert nb.params is not None and nb.vcov is not None
    assert list(nb.params.index) == ["const", "age", "smoker_yes"]  # the NB dispersion is left out
    assert nb.vcov.shape == (3, 3) and nb.df_resid == nb.n - 3
    poisson = count_regression(df, "visits", ["age"])
    assert poisson.params is not None and poisson.df_resid == poisson.n - 2


def test_cox_regression_exposes_log_hazard_ratios_and_df() -> None:
    df = _cohort()
    res = cox_regression(df, "time", "event", ["age", "smoker"])
    assert res.params is not None and res.vcov is not None
    assert list(res.params.index) == ["age", "smoker_yes"]
    np.testing.assert_allclose(res.params.to_numpy(), res.table["log_hr"].to_numpy())
    np.testing.assert_allclose(np.sqrt(np.diag(res.vcov.to_numpy())), res.table["se"].to_numpy())
    assert res.df_resid == res.events - 2  # events - parameters, as mice
    levels = factor_levels(df, ["site"])
    frozen = cox_regression(df, "time", "event", ["site"], levels=levels)
    assert frozen.params is not None and list(frozen.params.index) == [
        "site_Luxembourg",
        "site_Metz",
    ]
    with pytest.raises(ValueError, match="absent from these rows"):
        cox_regression(df[df["site"] != "Metz"].copy(), "time", "event", ["site"], levels=levels)


def test_kaplan_meier_keeps_the_greenwood_se() -> None:
    df = _cohort()
    curve = kaplan_meier(df["time"], df["event"])
    reference = SurvfuncRight(df["time"].to_numpy(), df["event"].to_numpy())
    assert curve.se is not None
    np.testing.assert_allclose(curve.se, np.asarray(reference.surv_prob_se))


# --- tests and 2x2 ----------------------------------------------------------------------------------


def test_t_tests_report_the_standard_error_of_their_estimate() -> None:
    rng = np.random.default_rng(9)
    a, b = rng.normal(5, 1, 40), rng.normal(4.5, 1.5, 55)
    for kind in ("welch", "student"):
        res = compare_groups({"a": a, "b": b}, kind=kind, reference="b")
        assert res.se is not None and res.estimate is not None and res.statistic is not None
        assert res.estimate / res.se == pytest.approx(res.statistic)
    before, after = rng.normal(10, 2, 30), rng.normal(10.6, 2, 30)
    paired = paired_t(after, before)
    assert paired.se == pytest.approx(np.std(after - before, ddof=1) / math.sqrt(30))
    one = one_sample_t(a, 5.0)
    assert one.se == pytest.approx(np.std(a, ddof=1) / math.sqrt(40))
    assert TestResult(name="x").mi is None


def test_two_by_two_reports_the_ses_behind_its_intervals() -> None:
    res = analyze_twobytwo(TwoByTwo(a=30, b=70, c=15, d=85))
    z = stats.norm.ppf(0.975)
    assert res.log_or_se is not None and res.log_rr_se is not None and res.rd_se is not None
    assert math.exp(math.log(res.odds_ratio) - z * res.log_or_se) == pytest.approx(
        res.odds_ratio_ci.lower
    )
    assert math.exp(math.log(res.relative_risk) + z * res.log_rr_se) == pytest.approx(
        res.relative_risk_ci.upper
    )
    assert res.rd_se == pytest.approx(math.sqrt(0.3 * 0.7 / 100 + 0.15 * 0.85 / 100))
    zero = analyze_twobytwo(TwoByTwo(a=0, b=50, c=5, d=45))
    assert zero.haldane_applied and zero.log_or_se == pytest.approx(
        math.sqrt(1 / 0.5 + 1 / 50.5 + 1 / 5.5 + 1 / 45.5)
    )


def test_mi_info_describes_a_pooled_result() -> None:
    info = MIInfo(m=20, fmi=0.12, riv=0.13, lam=0.11, df=310.2, dfcom=295.0)
    assert info.method.startswith("Rubin") and info.scale == "identity"
    pooled = TestResult(name="Welch t-test", estimate=0.4, se=0.1, mi=info)
    assert pooled.mi is not None and pooled.mi.m == 20
