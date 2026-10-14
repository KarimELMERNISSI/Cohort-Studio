"""
Group comparisons with a Welch-first policy (audit S7): assumptions are reported as diagnostics,
never used to gate the test; every result carries an estimate with a CI before its p-value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.oneway import anova_oneway

from biostats._types import CI, RANDOM_STATE, AssumptionCheck, EffectSize, TestResult
from biostats.effect_sizes import cohens_d, eta_squared, eta_squared_h, rank_biserial
from biostats.normality import normality_check
from biostats.references import cite

ArrayLike = Sequence[float] | np.ndarray | pd.Series

KINDS = ("auto", "welch", "student", "mannwhitney", "welch_anova", "anova", "kruskal")
LEGACY_NAMES = {
    "Welch's t-test": "welch",
    "Student's t-test": "student",
    "Mann-Whitney U": "mannwhitney",
    "Welch ANOVA": "welch_anova",
    "ANOVA": "anova",
    "Kruskal-Wallis": "kruskal",
    "Auto-Detect": "auto",
}
DISPLAY_NAMES = {
    "welch": "Welch's t-test",
    "student": "Student's t-test",
    "mannwhitney": "Mann-Whitney U",
    "welch_anova": "Welch ANOVA",
    "anova": "One-way ANOVA",
    "kruskal": "Kruskal-Wallis",
}


def _clean(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(pd.to_numeric(pd.Series(list(x)), errors="coerce"), dtype=float)
    return arr[~np.isnan(arr)]


def groups_from_frame(
    df: pd.DataFrame, group_col: str, value_col: str, *, levels: Sequence | None = None
) -> dict[str, np.ndarray]:
    """Ordered mapping level -> values (missing values dropped), levels in order of first appearance unless given."""
    clean = df[[group_col, value_col]].dropna()
    order = list(levels) if levels is not None else list(pd.unique(clean[group_col]))
    return {str(level): _clean(clean.loc[clean[group_col] == level, value_col]) for level in order}


def _hodges_lehmann(x: np.ndarray, y: np.ndarray) -> float:
    diffs = np.subtract.outer(x, y)
    return float(np.median(diffs))


def _bootstrap_ci(
    stat_fn, x: np.ndarray, y: np.ndarray, *, alpha: float, n_boot: int, seed: int
) -> CI | None:
    if len(x) < 3 or len(y) < 3 or n_boot <= 0:
        return None
    rng = np.random.default_rng(seed)
    boots = np.array(
        [
            stat_fn(rng.choice(x, len(x), replace=True), rng.choice(y, len(y), replace=True))
            for _ in range(n_boot)
        ]
    )
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return CI(
        lower=float(lo), upper=float(hi), level=1 - alpha, method=f"bootstrap percentile ({n_boot})"
    )


def _assumptions(groups: Mapping[str, np.ndarray], *, alpha: float) -> list[AssumptionCheck]:
    checks: list[AssumptionCheck] = []
    for label, values in groups.items():
        if len(values) >= 3:
            checks.append(normality_check(values, f"normality ({label})", alpha=alpha))
        else:
            checks.append(
                AssumptionCheck(
                    name=f"normality ({label})",
                    status="na",
                    detail=f"n = {len(values)}: not assessable",
                )
            )
    arrays = [v for v in groups.values() if len(v) >= 2]
    if len(arrays) >= 2:
        try:
            p = float(stats.levene(*arrays, center="median")[1])
            ratio = max(np.var(a, ddof=1) for a in arrays) / max(
                min(np.var(a, ddof=1) for a in arrays), 1e-12
            )
            status = "ok" if p >= alpha else ("caution" if ratio < 4 else "violated")
            checks.append(
                AssumptionCheck(
                    name="equal variances (Levene, median-centred)",
                    status=status,
                    detail=f"p = {p:.3g}; variance ratio {ratio:.2f}",
                    p_value=p,
                )
            )
        except Exception as exc:
            checks.append(AssumptionCheck(name="equal variances", status="na", detail=str(exc)))
    sizes = [len(v) for v in groups.values()]
    if min(sizes) < 15:
        checks.append(
            AssumptionCheck(
                name="sample size",
                status="caution",
                detail=f"smallest group n = {min(sizes)}: asymptotic approximations are rough; consider the rank-based route",
            )
        )
    else:
        checks.append(
            AssumptionCheck(
                name="sample size", status="ok", detail=f"smallest group n = {min(sizes)}"
            )
        )
    return checks


def recommend(
    groups: Mapping[str, np.ndarray], *, nonparametric: bool = False
) -> tuple[str, list[str]]:
    """Welch by default (Delacre 2017 / Welch 1951); rank tests when asked for or when groups are tiny and skewed."""
    k = len(groups)
    reasons: list[str] = []
    sizes = [len(v) for v in groups.values()]
    if nonparametric:
        reasons.append("rank-based route requested")
        return ("mannwhitney" if k == 2 else "kruskal"), reasons
    tiny = min(sizes) < 15
    skewed = any(len(v) >= 3 and abs(stats.skew(v)) > 1 for v in groups.values())
    if tiny and skewed:
        reasons.append("very small and strongly skewed groups: rank-based test recommended")
        return ("mannwhitney" if k == 2 else "kruskal"), reasons
    reasons.append(
        "Welch procedures do not assume equal variances and lose little power when they are equal (Delacre 2017)"
    )
    return ("welch" if k == 2 else "welch_anova"), reasons


def compare_groups(
    groups: Mapping[str, ArrayLike],
    *,
    kind: str = "auto",
    reference: str | None = None,
    alpha: float = 0.05,
    nonparametric: bool = False,
    n_boot: int = 1000,
    seed: int = RANDOM_STATE,
) -> TestResult:
    """
    Compare 2 or more independent groups. ``groups`` maps level -> values; ``reference`` names the
    reference level for two-group estimates (default: the first level). ``kind`` is one of KINDS.
    """
    kind = LEGACY_NAMES.get(kind, kind)
    if kind not in KINDS:
        raise ValueError(f"Unknown test kind {kind!r}; expected one of {KINDS}")
    data = {str(k): _clean(v) for k, v in groups.items()}
    data = {k: v for k, v in data.items() if len(v) > 0}
    labels = list(data)
    sizes = [len(v) for v in data.values()]
    if len(data) < 2 or min(sizes) < 2:
        return TestResult(
            name="not computed",
            p_value=None,
            groups=labels,
            n_per_group=sizes,
            warnings=["at least two groups with two or more observations are required"],
        )

    if kind == "auto":
        kind, reasons = recommend(data, nonparametric=nonparametric)
    else:
        reasons = [f"{DISPLAY_NAMES[kind]} selected manually"]
    if len(data) == 2 and kind in ("welch_anova", "anova", "kruskal"):
        kind = {"welch_anova": "welch", "anova": "student", "kruskal": "mannwhitney"}[kind]
    if len(data) > 2 and kind in ("welch", "student", "mannwhitney"):
        kind = {"welch": "welch_anova", "student": "anova", "mannwhitney": "kruskal"}[kind]

    checks = _assumptions(data, alpha=alpha)
    warnings = list(reasons)
    effects: list[EffectSize] = []
    references: list[str] = []

    if len(data) == 2:
        ref_label = reference if reference in data else labels[0]
        comp_label = next(label for label in labels if label != ref_label)
        x, y = data[comp_label], data[ref_label]
        if kind in ("welch", "student"):
            res = stats.ttest_ind(x, y, equal_var=(kind == "student"))
            ci_obj = res.confidence_interval(1 - alpha)
            df = float(res.df) if hasattr(res, "df") else float(len(x) + len(y) - 2)
            nx, ny, vx, vy = len(x), len(y), float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
            if kind == "welch":
                se = float(np.sqrt(vx / nx + vy / ny))
            else:
                pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
                se = float(np.sqrt(pooled * (1 / nx + 1 / ny)))
            effects = [
                cohens_d(x, y, alpha=alpha, hedges=True),
                rank_biserial(x, y, alpha=alpha, n_boot=min(n_boot, 1000), seed=seed),
            ]
            references = cite("welch1947", "delacre2017") if kind == "welch" else cite("hedges1981")
            return TestResult(
                name=DISPLAY_NAMES[kind],
                family="parametric",
                statistic=float(res.statistic),
                statistic_label="t",
                df=[df],
                p_value=float(res.pvalue),
                estimate=float(np.mean(x) - np.mean(y)),
                estimate_label=f"mean difference ({comp_label} - {ref_label})",
                estimate_ci=CI(
                    lower=float(ci_obj.low),
                    upper=float(ci_obj.high),
                    level=1 - alpha,
                    method="Welch t" if kind == "welch" else "Student t",
                ),
                se=se,
                effect_sizes=effects,
                assumptions=checks,
                warnings=warnings,
                references=references,
                groups=labels,
                n_per_group=sizes,
                reference_group=ref_label,
                comparison_group=comp_label,
                alpha=alpha,
            )
        res = stats.mannwhitneyu(x, y, alternative="two-sided")
        hl = _hodges_lehmann(x, y)
        hl_ci = _bootstrap_ci(_hodges_lehmann, x, y, alpha=alpha, n_boot=n_boot, seed=seed)
        effects = [
            rank_biserial(x, y, alpha=alpha, n_boot=min(n_boot, 1000), seed=seed),
            cohens_d(x, y, alpha=alpha, hedges=True),
        ]
        return TestResult(
            name=DISPLAY_NAMES["mannwhitney"],
            family="nonparametric",
            statistic=float(res.statistic),
            statistic_label="U",
            p_value=float(res.pvalue),
            estimate=hl,
            estimate_label=f"Hodges-Lehmann shift ({comp_label} - {ref_label})",
            estimate_ci=hl_ci,
            effect_sizes=effects,
            assumptions=checks,
            warnings=warnings,
            references=cite("mann1947", "hodges1963"),
            groups=labels,
            n_per_group=sizes,
            reference_group=ref_label,
            comparison_group=comp_label,
            alpha=alpha,
        )

    arrays = list(data.values())
    if kind == "welch_anova":
        res = anova_oneway(arrays, use_var="unequal", welch_correction=True)
        eta, omega = eta_squared(arrays, alpha=alpha)
        return TestResult(
            name=DISPLAY_NAMES[kind],
            family="parametric",
            statistic=float(res.statistic),
            statistic_label="F",
            df=[float(res.df_num), float(res.df_denom)],
            p_value=float(res.pvalue),
            effect_sizes=[eta, omega],
            assumptions=checks,
            warnings=warnings,
            references=cite("welch1951"),
            groups=labels,
            n_per_group=sizes,
            alpha=alpha,
        )
    if kind == "anova":
        res = stats.f_oneway(*arrays)
        eta, omega = eta_squared(arrays, alpha=alpha)
        k, n = len(arrays), sum(sizes)
        return TestResult(
            name=DISPLAY_NAMES[kind],
            family="parametric",
            statistic=float(res.statistic),
            statistic_label="F",
            df=[float(k - 1), float(n - k)],
            p_value=float(res.pvalue),
            effect_sizes=[eta, omega],
            assumptions=checks,
            warnings=warnings,
            references=cite("steiger2004"),
            groups=labels,
            n_per_group=sizes,
            alpha=alpha,
        )
    res = stats.kruskal(*arrays)
    k, n = len(arrays), sum(sizes)
    return TestResult(
        name=DISPLAY_NAMES["kruskal"],
        family="nonparametric",
        statistic=float(res.statistic),
        statistic_label="H",
        df=[float(k - 1)],
        p_value=float(res.pvalue),
        effect_sizes=[eta_squared_h(float(res.statistic), k, n)],
        assumptions=checks,
        warnings=warnings,
        references=cite("kruskal1952", "tomczak2014"),
        groups=labels,
        n_per_group=sizes,
        alpha=alpha,
    )


__all__ = [
    "DISPLAY_NAMES",
    "KINDS",
    "LEGACY_NAMES",
    "compare_groups",
    "groups_from_frame",
    "recommend",
]
