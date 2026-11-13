"""Paired and one-sample tests (audit S11): paired t, Wilcoxon signed-rank, McNemar, one-sample t, sign/binomial."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar as _mcnemar
from statsmodels.stats.proportion import proportion_confint

from biostats._types import CI, AssumptionCheck, EffectSize, TestResult
from biostats.normality import normality_check
from biostats.rank_intervals import disagreement, pseudo_median
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
    if lo is not None and hi is not None:
        dz_ci = CI(
            lower=lo / np.sqrt(n), upper=hi / np.sqrt(n), level=1 - alpha, method="noncentral t"
        )
    else:  # large-sample (Hedges & Olkin 1985), as cohens_d falls back
        se = float(np.sqrt(1 / n + dz * dz / (2 * n)))
        z = float(stats.norm.ppf(1 - alpha / 2))
        dz_ci = CI(
            lower=dz - z * se,
            upper=dz + z * se,
            level=1 - alpha,
            method="Hedges-Olkin large-sample",
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


def _signed_rank(
    differences: np.ndarray,
    *,
    alpha: float,
    label: str,
    without: tuple[str, str],
    kept: str,
) -> TestResult:
    """
    The signed-rank test of ``differences`` against 0, with their pseudo-median and its interval from the ordered
    Walsh averages (biostats.rank_intervals): nothing is drawn at random. The differences of 0 are left out of all
    three, by the test's rule, and the result says how many: ``without`` names one and several of them ("pair
    without a difference"), ``kept`` the others ("pairs that differ").
    """
    d = np.asarray(differences, dtype=float)
    d = d[~np.isnan(d)]  # an infinite value on both sides: no difference to rank
    zeros = int((d == 0).sum())
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
    hl, ci = pseudo_median(d, alpha=alpha, zeros=zeros > 0)
    notes: list[str] = []
    if zeros:
        notes.append(
            f"{zeros} {without[zeros > 1]} {'were' if zeros > 1 else 'was'} left out, by the signed-rank test's "
            f"rule: the test, the pseudo-median and its interval are those of the {n} {kept}."
        )
    infinite = int(np.sum(~np.isfinite(d)))
    if infinite:
        notes.append(
            f"{infinite} infinite difference{'s' if infinite > 1 else ''}: the test ranks "
            f"{'them' if infinite > 1 else 'it'} as the largest, and the pseudo-median and its interval are those "
            "of the other differences."
        )
    parted = disagreement(float(res.pvalue), ci, alpha)
    if parted:
        notes.append(parted)
    r = float(1 - 2 * res.statistic / (n * (n + 1) / 2))  # matched-pairs rank-biserial (Kerby 2014)
    return TestResult(
        name="Wilcoxon signed-rank",
        family="nonparametric",
        statistic=float(res.statistic),
        statistic_label="W",
        p_value=float(res.pvalue),
        estimate=hl,
        estimate_label=label,
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
        warnings=notes,
        references=cite("wilcoxon1945", "hollander2014"),
        groups=["pairs"],
        n_per_group=[n],
        alpha=alpha,
    )


def wilcoxon_signed_rank(
    after: Sequence,
    before: Sequence,
    *,
    alpha: float = 0.05,
) -> TestResult:
    """Wilcoxon signed-rank test with the Hodges-Lehmann pseudo-median of the differences and its interval."""
    x, y = _pairs(after, before)
    return _signed_rank(
        x - y,
        alpha=alpha,
        label="pseudo-median difference (after - before)",
        without=("pair without a difference", "pairs without a difference"),
        kept="pairs that differ",
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
    return _signed_rank(
        x,
        alpha=alpha,
        label=f"pseudo-median minus reference ({reference_value:g})",
        without=("value equal to the reference", "values equal to the reference"),
        kept="values that differ from it",
    )


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


def paired_table(
    before: pd.Series, after: pd.Series, *, counted: object | None = None
) -> pd.DataFrame:
    """
    The 2x2 table of a yes/no characteristic measured twice on the same participants: rows the first measurement,
    columns the second, over the rows where both are recorded.

    Both columns must hold the same two values (ValueError otherwise): two columns of different values, "Negative /
    Positive" and "Detected / Not detected", were paired by the alphabetical order of their labels, and a test came
    out of the wrong pairs. ``counted`` is the value whose share is compared: it comes second, so that the
    difference :func:`mcnemar_test` gives is the change in the share with that value. Without it, the second value in
    sorted order.
    """
    from biostats.reliability import category, category_order

    both = before.notna().to_numpy() & after.notna().to_numpy()
    first = [category(v) for v in before[both]]
    second = [category(v) for v in after[both]]
    if set(first) != set(second) or len(set(first)) != 2:
        raise ValueError(
            "both columns must hold the same two values in the rows where both are recorded"
        )
    levels = category_order(list(set(first)))
    if counted is not None:
        wanted = category(counted)
        if wanted not in levels:
            raise ValueError("the value counted is not one of the two values")
        levels = [name for name in levels if name != wanted] + [wanted]
    table = pd.crosstab(
        pd.Categorical(first, categories=levels),
        pd.Categorical(second, categories=levels),
        dropna=False,
    )
    table.index.name, table.columns.name = str(before.name), str(after.name)
    return table


def one_sample_proportion(
    successes: int, n: int, reference_proportion: float, *, alpha: float = 0.05
) -> TestResult:
    """Exact binomial test with a Wilson interval for the observed proportion."""
    res = stats.binomtest(int(successes), int(n), reference_proportion)
    lo, hi = proportion_confint(successes, n, alpha=alpha, method="wilson")
    # the exact test and the Wilson interval are two methods: next to the reference one can reject it while the
    # other holds it (3 of 10 against 0.10: p = 0.070 beside an interval from 0.108 to 0.603)
    notes: list[str] = []
    rejects, inside = float(res.pvalue) < alpha, lo <= reference_proportion <= hi
    if rejects == inside:
        said = (
            "the test rejects the reference and the interval holds it"
            if rejects
            else "the test does not reject the reference and the interval leaves it out"
        )
        notes.append(
            f"The p-value and the interval part at the border: {said}. They are two methods (an exact binomial "
            "test, a Wilson interval): report both."
        )
    return TestResult(
        name="Exact binomial test",
        family="exact",
        statistic=float(successes),
        statistic_label="number counted",
        p_value=float(res.pvalue),
        estimate=float(successes / n),
        estimate_label=f"observed proportion (reference {reference_proportion:g})",
        estimate_ci=CI(lower=float(lo), upper=float(hi), level=1 - alpha, method="Wilson"),
        warnings=notes,
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
    "paired_table",
    "wilcoxon_signed_rank",
]
