"""
Pairwise comparisons that report every pair with raw and adjusted p-values (audit S7).

Methods: Games-Howell (default after Welch ANOVA), Tukey HSD, Dunn (tie-corrected, after
Kruskal-Wallis), pairwise Welch/Student t and pairwise Mann-Whitney with Holm adjustment.
Estimates are always ``group_2 - group_1`` (and effects use group_2 as the comparison group).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from biostats._types import RANDOM_STATE
from biostats.effect_sizes import cohens_d, rank_biserial
from biostats.multiplicity import adjust, normalize_method
from biostats.references import cite

ArrayLike = Sequence[float] | np.ndarray | pd.Series
METHODS = ("games-howell", "tukey", "dunn", "welch", "student", "mannwhitney")
COLUMNS = ["group_1", "group_2", "n_1", "n_2", "estimate", "estimate_label", "ci_low", "ci_high", "statistic",
           "p_raw", "p_adj", "effect", "effect_name", "method", "adjustment"]


def _clean(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(pd.to_numeric(pd.Series(list(x)), errors="coerce"), dtype=float)
    return arr[~np.isnan(arr)]


def default_method(test_name: str) -> str:
    """The post-hoc family that matches the omnibus test."""
    name = test_name.lower()
    if "kruskal" in name:
        return "dunn"
    if "welch" in name:
        return "games-howell"
    if "anova" in name:
        return "tukey"
    if "mann" in name:
        return "mannwhitney"
    return "games-howell"


def games_howell(data: Mapping[str, np.ndarray], *, alpha: float) -> list[dict]:
    k = len(data)
    rows = []
    for (a, x), (b, y) in itertools.combinations(data.items(), 2):
        n1, n2 = len(x), len(y)
        v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)
        se = np.sqrt(v1 / n1 + v2 / n2)
        df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
        diff = float(np.mean(y) - np.mean(x))
        t = diff / se if se > 0 else np.nan
        q = abs(t) * np.sqrt(2)
        p = float(stats.studentized_range.sf(q, k, df)) if np.isfinite(q) else np.nan
        q_crit = stats.studentized_range.ppf(1 - alpha, k, df)
        half = q_crit / np.sqrt(2) * se
        rows.append({"group_1": a, "group_2": b, "n_1": n1, "n_2": n2, "estimate": diff, "estimate_label": "mean difference (group_2 - group_1)",
                     "ci_low": diff - half, "ci_high": diff + half, "statistic": float(t), "p_raw": p, "p_adj": p,
                     "effect": cohens_d(y, x, alpha=alpha).estimate, "effect_name": "Hedges' g",
                     "method": "Games-Howell", "adjustment": "intrinsic (studentized range)"})
    return rows


def tukey(data: Mapping[str, np.ndarray], *, alpha: float) -> list[dict]:
    values = np.concatenate(list(data.values()))
    labels = np.concatenate([[k] * len(v) for k, v in data.items()])
    res = pairwise_tukeyhsd(values, labels, alpha=alpha)
    rows = []
    for (a, b), diff, p, lo, hi in zip(itertools.combinations(res.groupsunique, 2), res.meandiffs, res.pvalues,
                                        res.confint[:, 0], res.confint[:, 1], strict=True):
        a, b = str(a), str(b)
        rows.append({"group_1": a, "group_2": b, "n_1": len(data[a]), "n_2": len(data[b]), "estimate": float(diff),
                     "estimate_label": "mean difference (group_2 - group_1)", "ci_low": float(lo), "ci_high": float(hi),
                     "statistic": np.nan, "p_raw": float(p), "p_adj": float(p), "effect": cohens_d(data[b], data[a], alpha=alpha).estimate,
                     "effect_name": "Hedges' g", "method": "Tukey HSD", "adjustment": "intrinsic (studentized range)"})
    return rows


def dunn(data: Mapping[str, np.ndarray], *, alpha: float, adjustment: str) -> list[dict]:
    values = np.concatenate(list(data.values()))
    ranks = stats.rankdata(values)
    n_total = len(values)
    offsets = np.cumsum([0, *[len(v) for v in data.values()]])
    mean_ranks = {k: float(ranks[offsets[i]:offsets[i + 1]].mean()) for i, k in enumerate(data)}
    _, counts = np.unique(values, return_counts=True)
    tie_term = float(np.sum(counts ** 3 - counts)) / (12 * (n_total - 1)) if n_total > 1 else 0.0
    base = n_total * (n_total + 1) / 12 - tie_term
    rows = []
    for (a, x), (b, y) in itertools.combinations(data.items(), 2):
        se = np.sqrt(base * (1 / len(x) + 1 / len(y)))
        z = (mean_ranks[b] - mean_ranks[a]) / se if se > 0 else np.nan
        p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
        rows.append({"group_1": a, "group_2": b, "n_1": len(x), "n_2": len(y), "estimate": mean_ranks[b] - mean_ranks[a],
                     "estimate_label": "mean rank difference (group_2 - group_1)", "ci_low": np.nan, "ci_high": np.nan,
                     "statistic": float(z), "p_raw": p, "p_adj": np.nan, "effect": rank_biserial(y, x, alpha=alpha, n_boot=0).estimate,
                     "effect_name": "rank-biserial r", "method": "Dunn (tie-corrected)", "adjustment": adjustment})
    return rows


def pairwise_t(data: Mapping[str, np.ndarray], *, alpha: float, adjustment: str, welch: bool) -> list[dict]:
    rows = []
    for (a, x), (b, y) in itertools.combinations(data.items(), 2):
        res = stats.ttest_ind(y, x, equal_var=not welch)
        ci = res.confidence_interval(1 - alpha)
        rows.append({"group_1": a, "group_2": b, "n_1": len(x), "n_2": len(y), "estimate": float(np.mean(y) - np.mean(x)),
                     "estimate_label": "mean difference (group_2 - group_1)", "ci_low": float(ci.low), "ci_high": float(ci.high),
                     "statistic": float(res.statistic), "p_raw": float(res.pvalue), "p_adj": np.nan,
                     "effect": cohens_d(y, x, alpha=alpha).estimate, "effect_name": "Hedges' g",
                     "method": "Welch t" if welch else "Student t", "adjustment": adjustment})
    return rows


def pairwise_mannwhitney(data: Mapping[str, np.ndarray], *, alpha: float, adjustment: str, seed: int) -> list[dict]:
    rows = []
    for (a, x), (b, y) in itertools.combinations(data.items(), 2):
        res = stats.mannwhitneyu(y, x, alternative="two-sided")
        hl = float(np.median(np.subtract.outer(y, x)))
        rows.append({"group_1": a, "group_2": b, "n_1": len(x), "n_2": len(y), "estimate": hl,
                     "estimate_label": "Hodges-Lehmann shift (group_2 - group_1)", "ci_low": np.nan, "ci_high": np.nan,
                     "statistic": float(res.statistic), "p_raw": float(res.pvalue), "p_adj": np.nan,
                     "effect": rank_biserial(y, x, alpha=alpha, n_boot=0, seed=seed).estimate, "effect_name": "rank-biserial r",
                     "method": "Mann-Whitney U", "adjustment": adjustment})
    return rows


def pairwise_comparisons(groups: Mapping[str, ArrayLike], *, method: str = "games-howell", adjust_method: str = "holm",
                         alpha: float = 0.05, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """All pairs, raw and adjusted p-values, estimates with CIs where the method provides them."""
    if method not in METHODS:
        raise ValueError(f"Unknown post-hoc method {method!r}; expected one of {METHODS}")
    data = {str(k): _clean(v) for k, v in groups.items()}
    data = {k: v for k, v in data.items() if len(v) >= 2}
    if len(data) < 2:
        return pd.DataFrame(columns=COLUMNS)
    adjustment = normalize_method(adjust_method)
    label = {"holm": "Holm", "bonferroni": "Bonferroni", "fdr_bh": "Benjamini-Hochberg", "none": "none"}[adjustment]
    if method == "games-howell":
        rows = games_howell(data, alpha=alpha)
    elif method == "tukey":
        rows = tukey(data, alpha=alpha)
    elif method == "dunn":
        rows = dunn(data, alpha=alpha, adjustment=label)
    elif method in ("welch", "student"):
        rows = pairwise_t(data, alpha=alpha, adjustment=label, welch=(method == "welch"))
    else:
        rows = pairwise_mannwhitney(data, alpha=alpha, adjustment=label, seed=seed)
    table = pd.DataFrame(rows, columns=COLUMNS)
    if method in ("dunn", "welch", "student", "mannwhitney"):
        table["p_adj"] = adjust(table["p_raw"].tolist(), adjustment, alpha=alpha)
    keys = {"games-howell": "games_howell1976", "tukey": "tukey1949", "dunn": "dunn1964", "welch": "welch1947",
            "student": "hedges1981", "mannwhitney": "mann1947"}
    table.attrs["references"] = cite(keys[method], *(["holm1979"] if adjustment == "holm" else []))
    table.attrs["alpha"] = alpha
    return table


__all__ = ["COLUMNS", "METHODS", "default_method", "pairwise_comparisons"]
