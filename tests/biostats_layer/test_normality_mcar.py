"""Normality with the right null distribution; Little's MCAR test on EM estimates."""

import numpy as np
import pandas as pd
from scipy import stats

from biostats.mcar import em_mean_cov, littles_mcar_test
from biostats.normality import (
    anderson_p,
    lilliefors_p,
    normality_check,
    normality_summary,
    shapiro_p,
)


def test_lilliefors_is_calibrated_where_self_standardised_ks_under_rejects() -> None:
    x = np.random.default_rng(1).exponential(size=200)
    assert lilliefors_p(x) < 0.01
    # Null calibration (audit S12): with estimated mean/SD the plain KS table almost never rejects.
    rng = np.random.default_rng(123)
    lillie_rejections = plain_rejections = 0
    for _ in range(300):
        sample = rng.normal(size=50)
        lillie_rejections += lilliefors_p(sample) < 0.05
        standardized = (sample - sample.mean()) / sample.std()
        plain_rejections += stats.kstest(standardized, "norm").pvalue < 0.05
    assert 0.02 <= lillie_rejections / 300 <= 0.09
    assert plain_rejections < lillie_rejections / 2
    normal = np.random.default_rng(2).normal(size=200)
    assert lilliefors_p(normal) > 0.05
    assert anderson_p(normal) > 0.05 and anderson_p(x) < 0.01
    assert shapiro_p(normal) > 0.05


def test_normality_summary_flags_large_samples_and_handles_small_ones() -> None:
    big = normality_summary(np.random.default_rng(3).normal(size=5000))
    assert "large sample" in big["verdict"]
    assert normality_check(np.random.default_rng(3).normal(size=5000)).status in ("ok", "caution")
    assert np.isnan(normality_summary([1.0, 1.0, 1.0])["shapiro_p"])
    assert normality_check([1.0, 2.0]).status == "na"
    assert normality_check(np.random.default_rng(4).exponential(size=80)).status == "violated"


def _mvn(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cov = np.full((4, 4), 0.5) + 0.5 * np.eye(4)
    return rng.multivariate_normal([10, 20, 30, 40], cov, size=n)


def test_littles_test_does_not_reject_mcar_and_rejects_mar() -> None:
    rng = np.random.default_rng(7)
    x = _mvn(400, 7)
    mcar = x.copy()
    mcar[rng.random(400) < 0.10, 1] = np.nan
    mcar[rng.random(400) < 0.10, 2] = np.nan
    res = littles_mcar_test(pd.DataFrame(mcar, columns=list("abcd")))
    assert "error" not in res and res["df"] > 0
    assert res["p_value"] > 0.05
    assert res["result"].startswith("No evidence")

    mar = x.copy()
    high = mar[:, 0] > np.median(mar[:, 0])
    mar[high & (rng.random(400) < 0.6), 1] = np.nan
    res_mar = littles_mcar_test(pd.DataFrame(mar, columns=list("abcd")))
    assert res_mar["p_value"] < 1e-4
    assert "MAR or MNAR" in res_mar["interpretation"]


def test_em_estimates_are_less_biased_than_available_case_means_under_mar() -> None:
    rng = np.random.default_rng(11)
    x = _mvn(2000, 11)
    truth = x[:, 1].mean()
    mar = x.copy()
    high = mar[:, 0] > np.median(mar[:, 0])
    mar[high & (rng.random(2000) < 0.7), 1] = np.nan
    available_case = np.nanmean(mar[:, 1])
    mu, sigma, iterations = em_mean_cov(mar)
    assert abs(mu[1] - truth) < abs(available_case - truth) / 2
    assert sigma.shape == (4, 4) and np.allclose(sigma, sigma.T) and iterations >= 1


def test_mcar_error_paths() -> None:
    assert "error" in littles_mcar_test(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    assert "error" in littles_mcar_test(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
    single_pattern = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0], "b": [np.nan] * 4, "c": [2.0, 3.0, 4.0, 5.0]}
    )
    assert "error" in littles_mcar_test(single_pattern)
