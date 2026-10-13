"""
Power and sample-size calculations with explicit total / per-group semantics.

Audit finding S2: ``statsmodels.stats.power.FTestAnovaPower`` takes and returns the **total** sample
size (sum over groups), but the page treated it as the per-group size, overstating required samples
by a factor k and understating achieved power. Every function here states which quantity it uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from statsmodels.stats.power import FTestAnovaPower, GofChisquarePower

ARE_NORMAL = 0.955  # asymptotic relative efficiency of rank tests vs. F/t under normality (3/pi)
ARE_WORST_CASE = 0.864  # Hodges-Lehmann lower bound (any continuous distribution)


@dataclass(frozen=True)
class PowerResult:
    test: str
    mode: str  # "sample_size" | "power"
    effect_size: float
    effect_name: str
    alpha: float
    power: float
    n_total: float
    n_per_group: int | None
    k_groups: int | None
    degrees_of_freedom: int | None
    method: str
    note: str = ""


def anova_sample_size(
    f: float, k_groups: int, alpha: float = 0.05, power: float = 0.8
) -> PowerResult:
    """Total and per-group sample size for a one-way ANOVA (noncentral F; Cohen's f)."""
    n_total = float(
        FTestAnovaPower().solve_power(
            effect_size=f, nobs=None, alpha=alpha, power=power, k_groups=k_groups
        )
    )
    per_group = math.ceil(n_total / k_groups)
    return PowerResult(
        test="One-way ANOVA",
        mode="sample_size",
        effect_size=f,
        effect_name="Cohen's f",
        alpha=alpha,
        power=power,
        n_total=n_total,
        n_per_group=per_group,
        k_groups=k_groups,
        degrees_of_freedom=k_groups - 1,
        method="statsmodels FTestAnovaPower (noncentral F); nobs is the TOTAL sample size",
        note=f"Balanced design: {per_group} per group, {per_group * k_groups} in total.",
    )


def anova_power(f: float, n_per_group: float, k_groups: int, alpha: float = 0.05) -> PowerResult:
    """Achieved power of a one-way ANOVA given the per-group sample size."""
    n_total = float(n_per_group) * k_groups
    achieved = float(
        FTestAnovaPower().solve_power(
            effect_size=f, nobs=n_total, alpha=alpha, power=None, k_groups=k_groups
        )
    )
    return PowerResult(
        test="One-way ANOVA",
        mode="power",
        effect_size=f,
        effect_name="Cohen's f",
        alpha=alpha,
        power=achieved,
        n_total=n_total,
        n_per_group=int(round(n_per_group)),
        k_groups=k_groups,
        degrees_of_freedom=k_groups - 1,
        method="statsmodels FTestAnovaPower (noncentral F); nobs is the TOTAL sample size",
    )


def kruskal_sample_size(
    f: float, k_groups: int, alpha: float = 0.05, power: float = 0.8, are: float = ARE_NORMAL
) -> PowerResult:
    """Kruskal-Wallis sample size: ANOVA requirement divided by the asymptotic relative efficiency."""
    base = anova_sample_size(f, k_groups, alpha, power)
    n_total = base.n_total / are
    per_group = math.ceil(n_total / k_groups)
    label = "normal data (ARE 0.955)" if abs(are - ARE_NORMAL) < 1e-9 else f"ARE {are:.3f}"
    return PowerResult(
        test="Kruskal-Wallis",
        mode="sample_size",
        effect_size=f,
        effect_name="Cohen's f (equivalent)",
        alpha=alpha,
        power=power,
        n_total=n_total,
        n_per_group=per_group,
        k_groups=k_groups,
        degrees_of_freedom=k_groups - 1,
        method=f"ANOVA sample size / ARE, {label}",
        note=f"Balanced design: {per_group} per group, {per_group * k_groups} in total. The 0.864 figure is the "
        "worst case over all continuous distributions; heavy-tailed data favour the rank test.",
    )


def kruskal_power(
    f: float, n_per_group: float, k_groups: int, alpha: float = 0.05, are: float = ARE_NORMAL
) -> PowerResult:
    """Approximate Kruskal-Wallis power: ANOVA power at the ARE-deflated total sample size."""
    n_total = float(n_per_group) * k_groups
    achieved = float(
        FTestAnovaPower().solve_power(
            effect_size=f, nobs=n_total * are, alpha=alpha, power=None, k_groups=k_groups
        )
    )
    return PowerResult(
        test="Kruskal-Wallis",
        mode="power",
        effect_size=f,
        effect_name="Cohen's f (equivalent)",
        alpha=alpha,
        power=achieved,
        n_total=n_total,
        n_per_group=int(round(n_per_group)),
        k_groups=k_groups,
        degrees_of_freedom=k_groups - 1,
        method=f"ANOVA power at nobs x ARE ({are:.3f})",
    )


def chi2_degrees_of_freedom(n_rows: int, n_cols: int) -> int:
    return max(1, (int(n_rows) - 1) * (int(n_cols) - 1))


def chi2_independence_sample_size(
    w: float, df: int, alpha: float = 0.05, power: float = 0.8
) -> PowerResult:
    """Total sample size for a chi-square test of independence with ``df`` degrees of freedom (Cohen's w)."""
    n_total = float(
        GofChisquarePower().solve_power(
            effect_size=w, nobs=None, alpha=alpha, power=power, n_bins=int(df) + 1
        )
    )
    return PowerResult(
        test="Chi-square (independence)",
        mode="sample_size",
        effect_size=w,
        effect_name="Cohen's w",
        alpha=alpha,
        power=power,
        n_total=n_total,
        n_per_group=None,
        k_groups=None,
        degrees_of_freedom=int(df),
        method="statsmodels GofChisquarePower (noncentral chi-square) with n_bins = df + 1",
    )


def chi2_independence_power(w: float, n_total: float, df: int, alpha: float = 0.05) -> PowerResult:
    """Achieved power of a chi-square test of independence for a total sample size."""
    achieved = float(
        GofChisquarePower().solve_power(
            effect_size=w, nobs=float(n_total), alpha=alpha, power=None, n_bins=int(df) + 1
        )
    )
    return PowerResult(
        test="Chi-square (independence)",
        mode="power",
        effect_size=w,
        effect_name="Cohen's w",
        alpha=alpha,
        power=achieved,
        n_total=float(n_total),
        n_per_group=None,
        k_groups=None,
        degrees_of_freedom=int(df),
        method="statsmodels GofChisquarePower (noncentral chi-square) with n_bins = df + 1",
    )


__all__ = [
    "ARE_NORMAL",
    "ARE_WORST_CASE",
    "PowerResult",
    "anova_power",
    "anova_sample_size",
    "chi2_degrees_of_freedom",
    "chi2_independence_power",
    "chi2_independence_sample_size",
    "kruskal_power",
    "kruskal_sample_size",
]
