"""
Multiple imputation with Rubin's rules (audit S4, stage 2).

``multiple_impute`` draws m completed datasets with chained equations (posterior sampling); the
analysis is run on each; ``rubin_pool`` combines the estimates and their variances.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from biostats._types import CI, RANDOM_STATE
from biostats.references import REFERENCES

REFERENCES.setdefault(
    "rubin1987", "Rubin DB (1987). Multiple Imputation for Nonresponse in Surveys. Wiley."
)
REFERENCES.setdefault(
    "vanbuuren2011",
    "van Buuren S, Groothuis-Oudshoorn K (2011). mice: multivariate imputation by chained equations in R. J Stat Softw 45(3).",
)
REFERENCES.setdefault(
    "barnard1999",
    "Barnard J, Rubin DB (1999). Small-sample degrees of freedom with multiple imputation. Biometrika 86:948-955.",
)


@dataclass
class PooledEstimate:
    estimate: float  # Q-bar
    within_variance: float  # U-bar
    between_variance: float  # B
    total_variance: float  # T
    df: float
    fmi: float  # fraction of missing information
    ci: CI
    m: int
    references: list[str] = field(default_factory=list)

    @property
    def se(self) -> float:
        return float(np.sqrt(self.total_variance))


def rubin_pool(
    estimates: Sequence[float],
    variances: Sequence[float],
    *,
    alpha: float = 0.05,
    n_complete: int | None = None,
) -> PooledEstimate:
    """Rubin's rules: Q-bar, T = U-bar + (1 + 1/m) B, df (old formula, Barnard-Rubin when n_complete is given), FMI."""
    q = np.asarray(estimates, dtype=float)
    u = np.asarray(variances, dtype=float)
    m = len(q)
    if m < 2 or len(u) != m:
        raise ValueError("at least two imputations with matching variances are required")
    q_bar = float(q.mean())
    u_bar = float(u.mean())
    b = float(q.var(ddof=1))
    t = u_bar + (1 + 1 / m) * b
    if b == 0:
        df = float("inf")
        r = 0.0
    else:
        r = (1 + 1 / m) * b / u_bar if u_bar > 0 else float("inf")
        df = (m - 1) * (1 + 1 / r) ** 2 if np.isfinite(r) and r > 0 else float("inf")
        if n_complete is not None and np.isfinite(df):
            lam = (1 + 1 / m) * b / t
            df_obs = (n_complete - 1 + 1) / (n_complete - 1 + 3) * (n_complete - 1) * (1 - lam)
            df = 1 / (1 / df + 1 / df_obs)
    fmi = (r + 2 / (df + 3)) / (r + 1) if np.isfinite(df) and np.isfinite(r) else 0.0
    crit = stats.t.ppf(1 - alpha / 2, df) if np.isfinite(df) else stats.norm.ppf(1 - alpha / 2)
    se = float(np.sqrt(t))
    return PooledEstimate(
        estimate=q_bar,
        within_variance=u_bar,
        between_variance=b,
        total_variance=float(t),
        df=float(df),
        fmi=float(fmi),
        ci=CI(lower=q_bar - crit * se, upper=q_bar + crit * se, level=1 - alpha, method="Rubin"),
        m=m,
        references=[REFERENCES["rubin1987"], REFERENCES["barnard1999"]],
    )


def multiple_impute(
    df: pd.DataFrame,
    *,
    m: int = 5,
    max_iter: int = 10,
    seed: int = RANDOM_STATE,
    exclude: Sequence[str] = (),
) -> list[pd.DataFrame]:
    """m completed copies of the numeric columns (chained equations with posterior sampling); other columns pass through."""
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]
    if not numeric:
        raise ValueError("no numeric column to impute")
    completed: list[pd.DataFrame] = []
    for i in range(m):
        imputer = IterativeImputer(sample_posterior=True, max_iter=max_iter, random_state=seed + i)
        values = imputer.fit_transform(df[numeric].to_numpy(dtype=float))
        copy = df.copy()
        copy[numeric] = values
        completed.append(copy)
    return completed


def pooled_statistics(
    completed: Sequence[pd.DataFrame],
    columns: Sequence[str],
    statistic: Callable[[pd.Series], tuple[float, float]] | None = None,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Pool a per-column statistic across completed datasets (default: the mean and its squared standard error)."""

    def _mean_and_var(s: pd.Series) -> tuple[float, float]:
        x = pd.to_numeric(s, errors="coerce").dropna()
        return float(x.mean()), float(x.var(ddof=1) / len(x)) if len(x) > 1 else float("nan")

    fn = statistic or _mean_and_var
    rows = []
    for col in columns:
        qs, us = zip(*[fn(frame[col]) for frame in completed], strict=True)
        pooled = rubin_pool(qs, us, alpha=alpha)
        rows.append(
            {
                "column": col,
                "estimate": pooled.estimate,
                "se": pooled.se,
                "ci_low": pooled.ci.lower,
                "ci_high": pooled.ci.upper,
                "df": pooled.df,
                "fmi": pooled.fmi,
                "between_variance": pooled.between_variance,
                "m": pooled.m,
            }
        )
    return pd.DataFrame(rows)


__all__ = ["PooledEstimate", "multiple_impute", "pooled_statistics", "rubin_pool"]
