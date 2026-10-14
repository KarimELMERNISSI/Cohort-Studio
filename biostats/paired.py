"""Paired and one-sample tests (audit S11): paired t, Wilcoxon signed-rank, McNemar, one-sample t, sign/binomial."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar as _mcnemar
from statsmodels.stats.proportion import proportion_confint

from biostats._types import CI, RANDOM_STATE, AssumptionCheck, EffectSize, TestResult
from biostats.normality import normality_check
from biostats.references import REFERENCES, cite

REFERENCES.setdefault(
    "wilcoxon1945",
    "Wilcoxon F (1945). Individual comparisons by ranking methods. Biometrics Bull 1:80-83.",
)
REFERENCES.setdefault(
    "mcnemar1947",
    "McNemar Q (1947). Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika 12:153-157.",
)
REFERENCES.setdefault(
    "student1908", "Student (1908). The probable error of a mean. Biometrika 6:1-25."
)


def _pairs(a: Sequence, b: Sequence) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(pd.to_numeric(pd.Series(list(a)), errors="coerce"), dtype=float)
    y = np.asarray(pd.to_numeric(pd.Series(list(b)), errors="coerce"), dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    return x[mask], y[mask]


def paired_t(after: Sequence, before: Sequence, *, alpha: float = 0.05) -> TestResult:
    """Paired t-test on after - before with the mean difference, its CI and Cohen's d_z (noncentral-t CI)."""
    x, y = _pairs(after, before)
    d = x - y
    n = len(d)
    if n < 2:
        return TestResult(
            name="Paired t-test",
            p_value=None,
            n_per_group=[n],
            warnings=["fewer than two complete pairs"],
        )
    res = stats.ttest_rel(x, y)
    ci = res.confidence_interval(1 - alpha)
    sd = d.std(ddof=1)
    dz = float(d.mean() / sd) if sd > 0 else 0.0
    from biostats.effect_sizes import _invert_noncentral

    t = dz * np.sqrt(n)
    lo = _invert_noncentral(
        lambda nc: stats.nct.cdf(t, n - 1, nc), 1 - alpha / 2, t - 5 - abs(t), t + 5 + abs(t)
    )
    hi = _invert_noncentral(
        lambda nc: stats.nct.cdf(t, n - 1, nc), alpha / 2, t - 5 - abs(t), t + 5 + abs(t)
    )
    dz_ci = (
        CI(lower=lo / np.sqrt(n), upper=hi / np.sqrt(n), level=1 - alpha, method="noncentral t")
        if lo is not None and hi is not None
        else None
    )
    return TestResult(
        name="Paired t-test",
        family="parametric",
        statistic=float(res.statistic),
        statistic_label="t",
        df=[float(n - 1)],
        p_value=float(res.pvalue),
        estimate=float(d.mean()),
        estimate_label="mean within-pair difference (after - before)",
        estimate_ci=CI(lower=float(ci.low), upper=float(ci.high), level=1 - alpha, method="t"),
        se=float(sd / np.sqrt(n)),
        effect_sizes=[
            EffectSize(
                name="Cohen's d_z",
                kind="d",
                estimate=dz,
                ci=dz_ci,
                n=n,
                reference=cite("student1908")[0],
            )
        ],
        assumptions=[normality_check(d, "normality of the differences", alpha=alpha)],
        references=cite("student1908"),
        groups=["pairs"],
        n_per_group=[n],
        alpha=alpha,
    )


def wilcoxon_signed_rank(
    after: Sequence,
    before: Sequence,
    *,
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int = RANDOM_STATE,
) -> TestResult:
    """Wilcoxon signed-rank test with the Hodges-Lehmann pseudo-median of the differences (bootstrap CI)."""
    x, y = _pairs(after, before)
    d = x - y
    d = d[d != 0]
    n = len(d)
    if n < 2:
        return TestResult(
            name="Wilcoxon signed-rank",
            p_value=None,
            n_per_group=[n],
            warnings=["fewer than two non-zero differences"],
        )
    res = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
    walsh = (d[:, None] + d[None, :]) / 2
    hl = float(np.median(walsh[np.triu_indices(n)]))
    ci = None
    if n >= 5 and n_boot > 0:
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            s = rng.choice(d, n, replace=True)
            w = (s[:, None] + s[None, :]) / 2
            boots.append(np.median(w[np.triu_indices(n)]))
        lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        ci = CI(
            lower=float(lo),
            upper=float(hi),
            level=1 - alpha,
            method=f"bootstrap percentile ({n_boot})",
        )
    r = float(1 - 2 * res.statistic / (n * (n + 1) / 2))  # matched-pairs rank-biserial (Kerby 2014)
    return TestResult(
        name="Wilcoxon signed-rank",
        family="nonparametric",
        statistic=float(res.statistic),
        statistic_label="W",
        p_value=float(res.pvalue),
        estimate=hl,
        estimate_label="pseudo-median difference (after - before)",
        estimate_ci=ci,
        effect_sizes=[
            EffectSize(
                name="matched-pairs rank-biserial r",
                kind="r",
                estimate=r,
                n=n,
                reference=cite("kerby2014")[0],
            )
        ],
        references=cite("wilcoxon1945"),
        groups=["pairs"],
        n_per_group=[n],
        alpha=alpha,
    )


def one_sample_t(values: Sequence, reference_value: float, *, alpha: float = 0.05) -> TestResult:
    x = np.asarray(pd.to_numeric(pd.Series(list(values)), errors="coerce"), dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return TestResult(
            name="One-sample t-test",
            p_value=None,
            n_per_group=[n],
            warnings=["fewer than two values"],
        )
    res = stats.ttest_1samp(x, reference_value)
    ci = res.confidence_interval(1 - alpha)
    sd = x.std(ddof=1)
    return TestResult(
        name="One-sample t-test",
        family="parametric",
        statistic=float(res.statistic),
        statistic_label="t",
        df=[float(n - 1)],
        p_value=float(res.pvalue),
        estimate=float(x.mean() - reference_value),
        estimate_label=f"mean minus reference ({reference_value:g})",
        estimate_ci=CI(
            lower=float(ci.low - reference_value),
            upper=float(ci.high - reference_value),
            level=1 - alpha,
            method="t",
        ),
        se=float(sd / np.sqrt(n)),
        effect_sizes=[
            EffectSize(
                name="Cohen's d",
                kind="d",
                estimate=float((x.mean() - reference_value) / sd) if sd > 0 else 0.0,
                n=n,
            )
        ],
        assumptions=[normality_check(x, "normality", alpha=alpha)],
        references=cite("student1908"),
        groups=["sample"],
        n_per_group=[n],
        alpha=alpha,
    )


def one_sample_wilcoxon(
    values: Sequence, reference_value: float, *, alpha: float = 0.05
) -> TestResult:
    x = np.asarray(pd.to_numeric(pd.Series(list(values)), errors="coerce"), dtype=float)
    x = x[~np.isnan(x)] - reference_value
    return wilcoxon_signed_rank(x, np.zeros_like(x), alpha=alpha)


def mcnemar_test(
    table: Sequence[Sequence[int]] | pd.DataFrame, *, exact: bool = True, alpha: float = 0.05
) -> TestResult:
    """McNemar's test on a 2x2 table of paired binary outcomes (rows = before, columns = after); exact by default."""
    arr = np.asarray(table, dtype=float)
    if arr.shape != (2, 2):
        raise ValueError("a 2x2 table is required")
    res = _mcnemar(arr, exact=exact, correction=True)
    b, c = arr[0, 1], arr[1, 0]
    n = arr.sum()
    diff = (b - c) / n if n else float("nan")
    # Wald CI for the difference in correlated proportions (Agresti-Min style adjustment with +0.5 per discordant cell)
    bb, cc = b + 0.5, c + 0.5
    se = np.sqrt((bb + cc) - (bb - cc) ** 2 / (n + 1)) / (n + 1)
    z = stats.norm.ppf(1 - alpha / 2)
    discordant = int(b + c)
    warnings = (
        []
        if discordant >= 10
        else ["fewer than 10 discordant pairs: the exact test is used and power is low"]
    )
    return TestResult(
        name="McNemar's test (exact)" if exact else "McNemar's test",
        family="exact" if exact else "parametric",
        statistic=float(res.statistic),
        statistic_label="discordant min" if exact else "chi2",
        p_value=float(res.pvalue),
        estimate=float(diff),
        estimate_label="difference in proportions (after - before)",
        estimate_ci=CI(
            lower=float(diff - z * se),
            upper=float(diff + z * se),
            level=1 - alpha,
            method="Wald (+0.5)",
        ),
        references=cite("mcnemar1947"),
        groups=["pairs"],
        n_per_group=[int(n)],
        warnings=warnings,
        alpha=alpha,
    )


def one_sample_proportion(
    successes: int, n: int, reference_proportion: float, *, alpha: float = 0.05
) -> TestResult:
    """Exact binomial test with a Wilson interval for the observed proportion."""
    res = stats.binomtest(int(successes), int(n), reference_proportion)
    lo, hi = proportion_confint(successes, n, alpha=alpha, method="wilson")
    return TestResult(
        name="Exact binomial test",
        family="exact",
        statistic=float(successes),
        statistic_label="successes",
        p_value=float(res.pvalue),
        estimate=float(successes / n),
        estimate_label=f"observed proportion (reference {reference_proportion:g})",
        estimate_ci=CI(lower=float(lo), upper=float(hi), level=1 - alpha, method="Wilson"),
        references=cite("wilson1927"),
        groups=["sample"],
        n_per_group=[int(n)],
        alpha=alpha,
        assumptions=[
            AssumptionCheck(
                name="sample size", status="ok" if n >= 20 else "caution", detail=f"n = {n}"
            )
        ],
    )


__all__ = [
    "mcnemar_test",
    "one_sample_proportion",
    "one_sample_t",
    "one_sample_wilcoxon",
    "paired_t",
    "wilcoxon_signed_rank",
]
