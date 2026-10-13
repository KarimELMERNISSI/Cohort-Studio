"""Normality diagnostics with the right null distributions (audit S12)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors, normal_ad

from biostats._types import AssumptionCheck


def _clean(x: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(pd.to_numeric(pd.Series(list(x)), errors="coerce"), dtype=float)
    return arr[~np.isnan(arr)]


def lilliefors_p(x: Sequence[float] | np.ndarray | pd.Series) -> float:
    """Kolmogorov-Smirnov with estimated mean and SD, using Lilliefors' null distribution (not the plain KS table)."""
    arr = _clean(x)
    if len(arr) < 4 or np.std(arr) == 0:
        return float("nan")
    return float(lilliefors(arr, dist="norm", pvalmethod="table")[1])


def anderson_p(x: Sequence[float] | np.ndarray | pd.Series) -> float:
    """Anderson-Darling p-value from statsmodels' normal_ad (the critical-value table approach is not a p-value)."""
    arr = _clean(x)
    if len(arr) < 8 or np.std(arr) == 0:
        return float("nan")
    return float(normal_ad(arr)[1])


def shapiro_p(x: Sequence[float] | np.ndarray | pd.Series) -> float:
    arr = _clean(x)
    if len(arr) < 3 or np.std(arr) == 0:
        return float("nan")
    if len(arr) > 5000:
        return float(stats.normaltest(arr)[1])  # D'Agostino K2 beyond Shapiro's range
    return float(stats.shapiro(arr)[1])


def normality_summary(x: Sequence[float] | np.ndarray | pd.Series, *, alpha: float = 0.05) -> dict:
    """Shape statistics plus p-values; the verdict text carries the sample-size caveat."""
    arr = _clean(x)
    n = len(arr)
    out = {
        "n": n,
        "skewness": float(stats.skew(arr)) if n > 2 else float("nan"),
        "excess_kurtosis": float(stats.kurtosis(arr)) if n > 3 else float("nan"),
        "shapiro_p": shapiro_p(arr),
        "lilliefors_p": lilliefors_p(arr),
        "anderson_p": anderson_p(arr),
    }
    p = out["shapiro_p"]
    if np.isnan(p):
        verdict = "not assessable (too few values or no variance)"
    elif n > 300:
        verdict = (
            "large sample: tests flag tiny deviations; judge the Q-Q plot, skewness and kurtosis instead "
            f"(Shapiro/D'Agostino p = {p:.3g})"
        )
    elif p < alpha:
        verdict = f"evidence against normality (p = {p:.3g})"
    else:
        verdict = f"no evidence against normality (p = {p:.3g})"
    out["verdict"] = verdict
    return out


def normality_check(
    x: Sequence[float] | np.ndarray | pd.Series, label: str = "normality", *, alpha: float = 0.05
) -> AssumptionCheck:
    summary = normality_summary(x, alpha=alpha)
    p = summary["shapiro_p"]
    if np.isnan(p):
        status = "na"
    elif summary["n"] > 300:
        status = "caution" if p < alpha else "ok"
    else:
        status = "violated" if p < alpha else "ok"
    return AssumptionCheck(
        name=label, status=status, detail=summary["verdict"], p_value=None if np.isnan(p) else p
    )


__all__ = ["anderson_p", "lilliefors_p", "normality_check", "normality_summary", "shapiro_p"]
