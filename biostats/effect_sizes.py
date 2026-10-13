"""
Effect sizes with confidence intervals.

Conventions (audit S8): every effect is *comparison minus reference*, so the sign of Hedges' g, the
Vargha-Delaney A and the rank-biserial r agree. CIs: noncentral-t inversion for d/g (Steiger 2004),
noncentral-F inversion for eta squared, bootstrap percentile intervals for rank-based measures and
the bias-corrected Cramer's V.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from scipy.special import gammaln

from biostats._types import CI, RANDOM_STATE, EffectSize
from biostats.references import cite

ArrayLike = Sequence[float] | np.ndarray | pd.Series


def _clean(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(pd.to_numeric(pd.Series(list(x)), errors="coerce"), dtype=float)
    return arr[~np.isnan(arr)]


def hedges_j(df: float) -> float:
    """Exact small-sample correction J(df) = Gamma(df/2) / (sqrt(df/2) Gamma((df-1)/2))."""
    if df <= 1:
        return 1.0
    return float(math.exp(gammaln(df / 2.0) - 0.5 * math.log(df / 2.0) - gammaln((df - 1.0) / 2.0)))


def _invert_noncentral(
    cdf: Callable[[float], float], target: float, lo: float, hi: float
) -> float | None:
    """Solve cdf(ncp) = target for a cdf that decreases monotonically in the noncentrality parameter."""
    f = lambda nc: cdf(nc) - target
    for _ in range(12):
        try:
            flo, fhi = f(lo), f(hi)
        except Exception:
            return None
        if np.isnan(flo) or np.isnan(fhi):
            return None
        if flo > 0 > fhi:
            try:
                return float(brentq(f, lo, hi, xtol=1e-6, maxiter=200))
            except Exception:
                return None
        lo, hi = lo - (hi - lo), hi + (hi - lo)
    return None


def cohens_d(
    comparison: ArrayLike, reference: ArrayLike, *, alpha: float = 0.05, hedges: bool = True
) -> EffectSize:
    """
    Standardised mean difference (comparison - reference) / pooled SD, with the exact Hedges correction
    when ``hedges`` is True, and a noncentral-t confidence interval (Steiger 2004).
    """
    x, y = _clean(comparison), _clean(reference)
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return EffectSize(
            name="Hedges' g" if hedges else "Cohen's d",
            kind="g" if hedges else "d",
            estimate=float("nan"),
            n=n1 + n2,
            note="fewer than 2 observations in a group",
        )
    df = n1 + n2 - 2
    pooled = math.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / df)
    if pooled <= 1e-12:
        return EffectSize(
            name="Hedges' g" if hedges else "Cohen's d",
            kind="g" if hedges else "d",
            estimate=0.0,
            n=n1 + n2,
            note="zero within-group variance",
        )
    d = float((np.mean(x) - np.mean(y)) / pooled)
    scale = math.sqrt(n1 * n2 / (n1 + n2))
    t = d * scale
    j = hedges_j(df) if hedges else 1.0
    lower = _invert_noncentral(
        lambda nc: stats.nct.cdf(t, df, nc), 1 - alpha / 2, t - 5 - abs(t), t + 5 + abs(t)
    )
    upper = _invert_noncentral(
        lambda nc: stats.nct.cdf(t, df, nc), alpha / 2, t - 5 - abs(t), t + 5 + abs(t)
    )
    if lower is None or upper is None:  # large-sample fallback (Hedges & Olkin 1985)
        se = math.sqrt((n1 + n2) / (n1 * n2) + d * d / (2 * (n1 + n2)))
        z = stats.norm.ppf(1 - alpha / 2)
        ci = CI(
            lower=j * (d - z * se),
            upper=j * (d + z * se),
            level=1 - alpha,
            method="Hedges-Olkin large-sample",
        )
    else:
        ci = CI(
            lower=j * lower / scale,
            upper=j * upper / scale,
            level=1 - alpha,
            method="noncentral t (Steiger 2004)",
        )
    return EffectSize(
        name="Hedges' g" if hedges else "Cohen's d",
        kind="g" if hedges else "d",
        estimate=j * d,
        ci=ci,
        n=n1 + n2,
        reference="; ".join(cite("hedges1981", "steiger2004")),
    )


def vargha_delaney_a(comparison: ArrayLike, reference: ArrayLike) -> float:
    """A = P(comparison > reference) + 0.5 P(tie) (Vargha & Delaney 2000); 0.5 means no stochastic difference."""
    x, y = _clean(comparison), _clean(reference)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    u1 = stats.mannwhitneyu(x, y, alternative="two-sided").statistic  # U for x
    return float(u1 / (len(x) * len(y)))


def rank_biserial(
    comparison: ArrayLike,
    reference: ArrayLike,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = RANDOM_STATE,
) -> EffectSize:
    """r = 2A - 1 (Kerby 2014): positive when the comparison group tends to be larger. Bootstrap percentile CI."""
    x, y = _clean(comparison), _clean(reference)
    if len(x) == 0 or len(y) == 0:
        return EffectSize(
            name="rank-biserial r",
            kind="r",
            estimate=float("nan"),
            n=len(x) + len(y),
            note="empty group",
        )
    estimate = 2 * vargha_delaney_a(x, y) - 1
    ci = None
    if len(x) >= 3 and len(y) >= 3 and n_boot > 0:
        rng = np.random.default_rng(seed)
        boots = np.empty(n_boot)
        for b in range(n_boot):
            bx = rng.choice(x, size=len(x), replace=True)
            by = rng.choice(y, size=len(y), replace=True)
            boots[b] = 2 * vargha_delaney_a(bx, by) - 1
        lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        ci = CI(
            lower=float(lo),
            upper=float(hi),
            level=1 - alpha,
            method=f"bootstrap percentile ({n_boot})",
        )
    return EffectSize(
        name="rank-biserial r",
        kind="r",
        estimate=float(estimate),
        ci=ci,
        n=len(x) + len(y),
        reference="; ".join(cite("vargha2000", "kerby2014")),
    )


def eta_squared(
    groups: Sequence[ArrayLike], *, alpha: float = 0.05
) -> tuple[EffectSize, EffectSize]:
    """(eta squared with a noncentral-F CI, omega squared) for a one-way design."""
    arrays = [_clean(g) for g in groups]
    arrays = [a for a in arrays if len(a) > 0]
    k = len(arrays)
    n = int(sum(len(a) for a in arrays))
    ref = "; ".join(cite("steiger2004"))
    if k < 2 or n <= k:
        nan = float("nan")
        return (
            EffectSize(
                name="eta squared", kind="eta2", estimate=nan, n=n, note="insufficient groups"
            ),
            EffectSize(
                name="omega squared", kind="omega2", estimate=nan, n=n, note="insufficient groups"
            ),
        )
    all_values = np.concatenate(arrays)
    grand = all_values.mean()
    ss_between = float(sum(len(a) * (a.mean() - grand) ** 2 for a in arrays))
    ss_total = float(((all_values - grand) ** 2).sum())
    ss_within = ss_total - ss_between
    df1, df2 = k - 1, n - k
    ms_within = ss_within / df2 if df2 > 0 else 0.0
    eta = ss_between / ss_total if ss_total > 0 else 0.0
    omega = (
        max(0.0, (ss_between - df1 * ms_within) / (ss_total + ms_within))
        if (ss_total + ms_within) > 0
        else 0.0
    )
    ci = None
    if ss_within > 0 and df2 > 0:
        f_stat = (ss_between / df1) / ms_within
        lo = _invert_noncentral(
            lambda nc: stats.ncf.cdf(f_stat, df1, df2, nc),
            1 - alpha / 2,
            0.0,
            max(10.0, 4 * f_stat * df1),
        )
        hi = _invert_noncentral(
            lambda nc: stats.ncf.cdf(f_stat, df1, df2, nc),
            alpha / 2,
            0.0,
            max(10.0, 4 * f_stat * df1),
        )
        lam_lo = 0.0 if lo is None else max(0.0, lo)
        lam_hi = hi
        if lam_hi is not None:
            ci = CI(
                lower=lam_lo / (lam_lo + n),
                upper=lam_hi / (lam_hi + n),
                level=1 - alpha,
                method="noncentral F (Steiger 2004)",
            )
    return (
        EffectSize(name="eta squared", kind="eta2", estimate=float(eta), ci=ci, n=n, reference=ref),
        EffectSize(name="omega squared", kind="omega2", estimate=float(omega), n=n, reference=ref),
    )


def eta_squared_h(h_statistic: float, k: int, n: int) -> EffectSize:
    """Kruskal-Wallis effect size eta^2_H = (H - k + 1) / (n - k) (Tomczak & Tomczak 2014)."""
    value = (h_statistic - k + 1) / (n - k) if n > k else float("nan")
    return EffectSize(
        name="eta squared (H)",
        kind="eta2_h",
        estimate=float(max(0.0, value)) if not np.isnan(value) else value,
        n=n,
        reference="; ".join(cite("tomczak2014")),
    )


def _cramers_v_corrected(table: np.ndarray) -> float:
    n = table.sum()
    if n == 0 or min(table.shape) < 2:
        return float("nan")
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    r, k = table.shape
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(math.sqrt(phi2_corr / denom)) if denom > 0 else float("nan")


def cramers_v(
    table: pd.DataFrame | np.ndarray,
    *,
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int = RANDOM_STATE,
) -> EffectSize:
    """Bias-corrected Cramer's V (Bergsma 2013) from the Pearson chi-square (no Yates correction), bootstrap CI."""
    arr = np.asarray(table, dtype=float)
    n = int(arr.sum())
    estimate = _cramers_v_corrected(arr)
    ci = None
    if n >= 20 and n_boot > 0 and not np.isnan(estimate):
        rng = np.random.default_rng(seed)
        probs = arr.ravel() / n
        boots = np.empty(n_boot)
        for b in range(n_boot):
            sample = rng.multinomial(n, probs).reshape(arr.shape)
            boots[b] = _cramers_v_corrected(sample)
        boots = boots[~np.isnan(boots)]
        if len(boots) > 50:
            lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            ci = CI(
                lower=float(lo),
                upper=float(hi),
                level=1 - alpha,
                method=f"bootstrap percentile ({n_boot})",
            )
    return EffectSize(
        name="Cramer's V (bias-corrected)",
        kind="v",
        estimate=estimate,
        ci=ci,
        n=n,
        reference="; ".join(cite("bergsma2013")),
    )


__all__ = [
    "cohens_d",
    "cramers_v",
    "eta_squared",
    "eta_squared_h",
    "hedges_j",
    "rank_biserial",
    "vargha_delaney_a",
]
