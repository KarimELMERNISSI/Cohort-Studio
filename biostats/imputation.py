"""
Multiple imputation with Rubin's rules (audit S4, stage 2).

``multiple_impute`` draws m completed datasets with chained equations (posterior sampling); the
analysis is run on each; ``rubin_pool`` combines the estimates and their variances. The pooling itself
lives in :mod:`biostats.mi.pooling` (formulas of R mice); ``rubin_pool`` is kept for its callers.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from biostats._types import CI, RANDOM_STATE
from biostats.mi.pooling import pool_scalar
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
    """
    Rubin's rules for one estimate (wrapper over :func:`biostats.mi.pooling.pool_scalar`).

    ``n_complete`` is the sample size of a mean (complete-data df n - 1, Barnard-Rubin df); without it the
    large-sample df is used. For a regression coefficient call ``pool_scalar`` with the residual df.
    """
    dfcom = math.inf if n_complete is None else max(n_complete - 1, 1)
    pooled = pool_scalar(estimates, variances, dfcom=dfcom, alpha=alpha)
    return PooledEstimate(
        estimate=pooled.estimate,
        within_variance=pooled.ubar,
        between_variance=pooled.b,
        total_variance=pooled.t,
        df=pooled.df,
        fmi=pooled.fmi,
        ci=CI(lower=pooled.ci.lower, upper=pooled.ci.upper, level=1 - alpha, method="Rubin"),
        m=pooled.m,
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
        # the default statistic is a mean: complete-data df n - 1 (Barnard-Rubin); unknown for a custom one
        n = int(pd.to_numeric(completed[0][col], errors="coerce").notna().sum())
        pooled = rubin_pool(qs, us, alpha=alpha, n_complete=n if statistic is None else None)
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
