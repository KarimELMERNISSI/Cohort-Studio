"""
Missing-data diagnostics.

Audit finding S3: the global missingness dependency scan kept only tests with p < 0.05 and the
dashboard then "corrected" that already-filtered subset, so the multiplicity correction controlled
nothing and the reported family size was the number of hits. :func:`scan_missingness_dependencies`
returns **every** test it performed with raw and family-adjusted p-values.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

Correction = Literal["none", "bonferroni", "holm", "fdr_bh"]
TestPreference = Literal["auto", "parametric", "nonparametric"]

COLUMNS = [
    "Target Variable",
    "Predictor Variable",
    "Dependency Type",
    "Test Used",
    "P-Value",
    "P-Value (Adj)",
    "Significant",
    "Effect",
    "Direction",
    "n_missing",
    "n_observed",
    "Conclusion",
]

_PREFERENCE_ALIASES = {
    "auto": "auto",
    "auto-detect": "auto",
    "parametric": "parametric",
    "force parametric": "parametric",
    "nonparametric": "nonparametric",
    "non-parametric": "nonparametric",
    "force non-parametric": "nonparametric",
}
_CORRECTION_ALIASES = {
    "none": "none",
    "": "none",
    "bonferroni": "bonferroni",
    "holm": "holm",
    "fdr_bh": "fdr_bh",
    "benjamini-hochberg (fdr)": "fdr_bh",
    "benjamini-hochberg": "fdr_bh",
    "fdr": "fdr_bh",
    "bh": "fdr_bh",
}


@dataclass
class ScanResult:
    table: pd.DataFrame
    n_tests: int
    n_skipped: int
    n_significant: int
    correction: str
    alpha: float

    @property
    def significant(self) -> pd.DataFrame:
        return self.table[self.table["Significant"]]


def _cohens_d(x: pd.Series, y: pd.Series) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    return float((x.mean() - y.mean()) / np.sqrt(pooled)) if pooled > 0 else float("nan")


def _cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if n > 0 and k > 0 else float("nan")


def _numeric_test(
    g_missing: pd.Series, g_observed: pd.Series, preference: str
) -> tuple[str, float]:
    use_parametric = preference == "parametric"
    if preference == "auto":
        normal = True
        for g in (g_missing, g_observed):
            if len(g) < 3 or len(g) > 5000:
                normal = len(g) > 5000
                if not normal:
                    break
                continue
            try:
                normal = normal and stats.shapiro(g)[1] >= 0.05
            except Exception:
                normal = False
        use_parametric = normal
    if use_parametric:
        stat, p = stats.ttest_ind(g_missing, g_observed, equal_var=False, nan_policy="omit")
        return "Welch's t-test", float(p)
    stat, p = stats.mannwhitneyu(g_missing, g_observed)
    return "Mann-Whitney U", float(p)


def scan_missingness_dependencies(
    df: pd.DataFrame,
    targets: list[str] | None = None,
    predictors: list[str] | None = None,
    *,
    test_preference: str = "auto",
    correction: str = "fdr_bh",
    alpha: float = 0.05,
    min_group: int = 5,
    min_observed: int = 10,
    progress_callback: Callable[[float], None] | None = None,
) -> ScanResult:
    """Test whether missingness of each target is associated with the values / missingness of predictors.

    Every performed test is returned; ``P-Value (Adj)`` is corrected over the full family with the
    requested method (Bonferroni, Holm or Benjamini-Hochberg). Associations are not causal claims.
    """
    preference = _PREFERENCE_ALIASES.get(str(test_preference).strip().lower(), "auto")
    method = _CORRECTION_ALIASES.get(str(correction).strip().lower(), "fdr_bh")
    if targets is None:
        targets = [c for c in df.columns if bool(df[c].isnull().any())]
    if predictors is None:
        predictors = df.columns.tolist()

    missing = {c: df[c].isnull() for c in df.columns}
    numeric = set(df.select_dtypes(include=[np.number]).columns)
    rows: list[dict] = []
    skipped = 0
    total = max(1, len(targets) * len(predictors))
    done = 0

    for target in targets:
        t_missing = missing[target]
        if t_missing.sum() == 0 or t_missing.sum() == len(df):
            done += len(predictors)
            skipped += len(predictors)
            continue
        for pred in predictors:
            done += 1
            if progress_callback and done % 10 == 0:
                progress_callback(done / total)
            if pred == target:
                continue
            observed = ~missing[pred]
            if observed.sum() >= min_observed:
                values = df.loc[observed, pred]
                groups = t_missing.loc[observed]
                n_missing, n_observed = int(groups.sum()), int((~groups).sum())
                if n_missing >= min_group and n_observed >= min_group:
                    try:
                        if pred in numeric:
                            g_missing = pd.to_numeric(values[groups], errors="coerce").dropna()
                            g_observed = pd.to_numeric(values[~groups], errors="coerce").dropna()
                            test_name, p = _numeric_test(g_missing, g_observed, preference)
                            effect = _cohens_d(g_missing, g_observed)
                            direction = (
                                "higher" if g_missing.mean() > g_observed.mean() else "lower"
                            ) + f" {pred} when {target} is missing"
                            effect_text = f"d = {effect:.2f}" if np.isfinite(effect) else ""
                        else:
                            table = pd.crosstab(values.astype(str), groups)
                            if table.shape[0] < 2 or table.shape[1] < 2:
                                raise ValueError("degenerate table")
                            _, p, _, expected = stats.chi2_contingency(table, correction=False)
                            if (expected < 5).any() and table.shape == (2, 2):
                                _, p = stats.fisher_exact(table)
                                test_name = "Fisher's exact test"
                            else:
                                test_name = "Chi-square test"
                            effect = _cramers_v(table)
                            direction = f"distribution of {pred} differs when {target} is missing"
                            effect_text = (
                                f"Cramer's V = {effect:.2f}" if np.isfinite(effect) else ""
                            )
                        rows.append(
                            {
                                "Target Variable": target,
                                "Predictor Variable": pred,
                                "Dependency Type": "Value association",
                                "Test Used": test_name,
                                "P-Value": float(p),
                                "Effect": effect_text,
                                "Direction": direction,
                                "n_missing": n_missing,
                                "n_observed": n_observed,
                                "Conclusion": f"Values of {pred} differ between rows where {target} is missing and rows where it is observed (association, not causation).",
                            }
                        )
                    except Exception:
                        skipped += 1
                else:
                    skipped += 1
            else:
                skipped += 1

            p_missing = missing[pred]
            if 0 < p_missing.sum() < len(df):
                try:
                    table = pd.crosstab(t_missing, p_missing)
                    if table.shape == (2, 2):
                        _, p, _, expected = stats.chi2_contingency(table, correction=False)
                        if (expected < 5).any():
                            _, p = stats.fisher_exact(table)
                            test_name = "Fisher's exact test"
                        else:
                            test_name = "Chi-square test"
                        both = int((t_missing & p_missing).sum())
                        rows.append(
                            {
                                "Target Variable": target,
                                "Predictor Variable": pred,
                                "Dependency Type": "Missingness co-occurrence",
                                "Test Used": test_name,
                                "P-Value": float(p),
                                "Effect": f"phi = {_cramers_v(table):.2f}",
                                "Direction": f"{both} rows missing both",
                                "n_missing": int(p_missing.sum()),
                                "n_observed": int((~p_missing).sum()),
                                "Conclusion": f"Missingness of {pred} is associated with missingness of {target}.",
                            }
                        )
                except Exception:
                    skipped += 1
    if progress_callback:
        progress_callback(1.0)

    table_out = pd.DataFrame(
        rows, columns=[c for c in COLUMNS if c not in ("P-Value (Adj)", "Significant")]
    )
    if table_out.empty:
        table_out["P-Value (Adj)"] = pd.Series(dtype=float)
        table_out["Significant"] = pd.Series(dtype=bool)
        return ScanResult(table_out[COLUMNS], 0, skipped, 0, method, alpha)

    p_raw = table_out["P-Value"].fillna(1.0).to_numpy()
    if method == "none":
        p_adj, reject = p_raw.copy(), p_raw < alpha
    else:
        reject, p_adj, _, _ = multipletests(p_raw, alpha=alpha, method=method)
    table_out["P-Value (Adj)"] = p_adj
    table_out["Significant"] = reject.astype(bool)
    table_out = table_out[COLUMNS].sort_values(["P-Value (Adj)", "P-Value"]).reset_index(drop=True)
    return ScanResult(
        table_out, int(len(table_out)), skipped, int(table_out["Significant"].sum()), method, alpha
    )


__all__ = ["COLUMNS", "Correction", "ScanResult", "TestPreference", "scan_missingness_dependencies"]
