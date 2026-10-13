"""Method-comparison statistics: Bland-Altman limits of agreement with confidence intervals (audit S10)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from biostats._types import CI
from biostats.references import cite


@dataclass
class BlandAltman:
    n: int
    bias: float
    bias_ci: CI
    sd: float
    loa_lower: float
    loa_upper: float
    loa_lower_ci: CI
    loa_upper_ci: CI
    within_loa_pct: float
    proportional_bias_slope: float
    proportional_bias_p: float
    means: np.ndarray
    differences: np.ndarray
    references: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def bland_altman(
    method_a: Sequence[float], method_b: Sequence[float], *, alpha: float = 0.05
) -> BlandAltman:
    """Differences a - b against their means; SD with ddof = 1; CIs for the bias and each limit (Bland & Altman 1986)."""
    a = np.asarray(pd.to_numeric(pd.Series(list(method_a)), errors="coerce"), dtype=float)
    b = np.asarray(pd.to_numeric(pd.Series(list(method_b)), errors="coerce"), dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 3:
        raise ValueError("at least three paired observations are required")
    diff = a - b
    mean = (a + b) / 2
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    z = stats.norm.ppf(1 - alpha / 2)
    t = stats.t.ppf(1 - alpha / 2, n - 1)
    se_bias = sd / math.sqrt(n)
    se_loa = sd * math.sqrt(3.0 / n)
    loa_lo, loa_hi = bias - z * sd, bias + z * sd
    slope, _, _, p_slope, _ = (
        stats.linregress(mean, diff) if np.ptp(mean) > 0 else (0.0, 0.0, 0.0, 1.0, 0.0)
    )
    warnings = []
    if p_slope < 0.05:
        warnings.append(
            "proportional bias: the difference changes with the magnitude (slope p < 0.05); consider ratios or log scale"
        )
    return BlandAltman(
        n=n,
        bias=bias,
        bias_ci=CI(lower=bias - t * se_bias, upper=bias + t * se_bias, level=1 - alpha, method="t"),
        sd=sd,
        loa_lower=float(loa_lo),
        loa_upper=float(loa_hi),
        loa_lower_ci=CI(
            lower=float(loa_lo - t * se_loa),
            upper=float(loa_lo + t * se_loa),
            level=1 - alpha,
            method="Bland-Altman",
        ),
        loa_upper_ci=CI(
            lower=float(loa_hi - t * se_loa),
            upper=float(loa_hi + t * se_loa),
            level=1 - alpha,
            method="Bland-Altman",
        ),
        within_loa_pct=float(np.mean((diff >= loa_lo) & (diff <= loa_hi)) * 100),
        proportional_bias_slope=float(slope),
        proportional_bias_p=float(p_slope),
        means=mean,
        differences=diff,
        references=cite("bland1986", "bland1999"),
        warnings=warnings,
    )


__all__ = ["BlandAltman", "bland_altman"]
