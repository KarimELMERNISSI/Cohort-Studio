"""
Association measures across variable types (audit S13): Pearson/Spearman with Fisher-z / Bonett-Wright
CIs, point-biserial for binary x numeric, the correlation ratio for categorical x numeric, and the
bias-corrected Cramer's V for categorical pairs; p-values adjusted over the whole matrix (BH).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from biostats._types import CI
from biostats.effect_sizes import cramers_v
from biostats.multiplicity import adjust
from biostats.references import cite

PAIR_COLUMNS = ["var_1", "var_2", "type_1", "type_2", "measure", "estimate", "ci_low", "ci_high", "p_raw", "p_adj", "n"]


def pearson_ci(r: float, n: int, alpha: float = 0.05) -> CI:
    """Fisher z interval."""
    if n < 4 or abs(r) >= 1:
        return CI(lower=float("nan"), upper=float("nan"), level=1 - alpha, method="Fisher z")
    z = np.arctanh(r)
    se = 1 / math.sqrt(n - 3)
    q = stats.norm.ppf(1 - alpha / 2)
    return CI(lower=float(np.tanh(z - q * se)), upper=float(np.tanh(z + q * se)), level=1 - alpha, method="Fisher z")


def spearman_ci(r: float, n: int, alpha: float = 0.05) -> CI:
    """Bonett & Wright (2000) interval: Fisher z with SE sqrt((1 + r^2/2)/(n - 3))."""
    if n < 4 or abs(r) >= 1:
        return CI(lower=float("nan"), upper=float("nan"), level=1 - alpha, method="Bonett-Wright")
    z = np.arctanh(r)
    se = math.sqrt((1 + r * r / 2) / (n - 3))
    q = stats.norm.ppf(1 - alpha / 2)
    return CI(lower=float(np.tanh(z - q * se)), upper=float(np.tanh(z + q * se)), level=1 - alpha, method="Bonett-Wright")


def correlation_ratio(categories: pd.Series, values: pd.Series) -> tuple[float, float, int]:
    """eta = sqrt(SS_between / SS_total) and the one-way ANOVA p-value; (eta, p, n)."""
    frame = pd.DataFrame({"c": categories, "v": pd.to_numeric(values, errors="coerce")}).dropna()
    n = len(frame)
    groups = [g["v"].to_numpy() for _, g in frame.groupby("c", observed=True) if len(g) > 0]
    if n < 3 or len(groups) < 2:
        return float("nan"), float("nan"), n
    grand = frame["v"].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = float(((frame["v"] - grand) ** 2).sum())
    eta = math.sqrt(ss_between / ss_total) if ss_total > 0 else float("nan")
    p = float(stats.f_oneway(*groups)[1]) if all(len(g) >= 2 for g in groups) else float("nan")
    return eta, p, n


def variable_types(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, str]:
    """numeric | binary | categorical (binary = exactly two distinct non-missing values, whatever the dtype)."""
    out: dict[str, str] = {}
    for col in columns or df.columns:
        s = df[col]
        nunique = s.nunique(dropna=True)
        if nunique == 2:
            out[col] = "binary"
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            out[col] = "numeric"
        elif nunique <= 1:
            continue
        else:
            out[col] = "categorical"
    return out


@dataclass
class AssociationResult:
    matrix: pd.DataFrame          # square, estimates (NaN when not computable)
    pairs: pd.DataFrame           # long table with CIs and p-values
    types: dict[str, str]
    method: str
    min_periods: int
    references: list[str]


def _binary_codes(s: pd.Series) -> pd.Series:
    levels = sorted(s.dropna().unique(), key=str)
    return s.map({levels[0]: 0.0, levels[1]: 1.0})


def association_matrix(df: pd.DataFrame, columns: list[str] | None = None, *, method: str = "pearson", min_periods: int = 10,
                       alpha: float = 0.05, adjust_method: str = "fdr_bh") -> AssociationResult:
    """Pairwise associations with a measure chosen by the variable types; pairs with n < min_periods are left empty."""
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")
    types = variable_types(df, columns)
    names = list(types)
    rows: list[dict] = []
    for a, b in itertools.combinations(names, 2):
        ta, tb = types[a], types[b]
        pair = df[[a, b]].dropna()
        n = len(pair)
        row = {"var_1": a, "var_2": b, "type_1": ta, "type_2": tb, "n": n, "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
               "p_raw": np.nan, "measure": ""}
        if n >= min_periods:
            kinds = {ta, tb}
            if kinds <= {"numeric", "binary"} and "numeric" in kinds or kinds == {"numeric"}:
                x = _binary_codes(pair[a]) if ta == "binary" else pd.to_numeric(pair[a], errors="coerce")
                y = _binary_codes(pair[b]) if tb == "binary" else pd.to_numeric(pair[b], errors="coerce")
                ok = ~(x.isna() | y.isna())
                x, y = x[ok], y[ok]
                if len(x) >= min_periods and x.std() > 0 and y.std() > 0:
                    if "binary" in kinds:
                        r, p = stats.pearsonr(x, y)
                        row.update(measure="point-biserial r", estimate=float(r), p_raw=float(p))
                        ci = pearson_ci(float(r), len(x), alpha)
                    elif method == "pearson":
                        r, p = stats.pearsonr(x, y)
                        row.update(measure="Pearson r", estimate=float(r), p_raw=float(p))
                        ci = pearson_ci(float(r), len(x), alpha)
                    else:
                        r, p = stats.spearmanr(x, y)
                        row.update(measure="Spearman rho", estimate=float(r), p_raw=float(p))
                        ci = spearman_ci(float(r), len(x), alpha)
                    row.update(ci_low=ci.lower, ci_high=ci.upper)
            elif "numeric" in kinds:  # numeric x categorical
                cat, num = (a, b) if ta == "categorical" else (b, a)
                eta, p, _ = correlation_ratio(pair[cat], pair[num])
                row.update(measure="correlation ratio eta", estimate=eta, p_raw=p)
            else:  # binary/categorical pairs
                table = pd.crosstab(pair[a], pair[b])
                if table.shape[0] >= 2 and table.shape[1] >= 2:
                    v = cramers_v(table, alpha=alpha, n_boot=0)
                    p = float(stats.chi2_contingency(table, correction=False)[1])
                    row.update(measure="Cramer's V (bias-corrected)", estimate=v.estimate, p_raw=p)
        rows.append(row)
    pairs = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    pairs["p_adj"] = adjust(pairs["p_raw"].tolist(), adjust_method, alpha=alpha) if len(pairs) else []
    matrix = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for name in names:
        matrix.loc[name, name] = 1.0
    for _, r in pairs.iterrows():
        matrix.loc[r["var_1"], r["var_2"]] = r["estimate"]
        matrix.loc[r["var_2"], r["var_1"]] = r["estimate"]
    return AssociationResult(matrix=matrix, pairs=pairs, types=types, method=method, min_periods=min_periods,
                             references=cite("fisher1915", "bonett2000", "bergsma2013", "benjamini1995", "schober2018"))


__all__ = ["PAIR_COLUMNS", "AssociationResult", "association_matrix", "correlation_ratio", "pearson_ci", "spearman_ci", "variable_types"]
