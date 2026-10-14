"""
Linear regression and ANCOVA (ordinary least squares).

``linear_regression`` fits an outcome on covariates (factors one-hot coded, see :mod:`biostats.design`)
and returns the coefficient table with what pooling across imputed datasets needs: the coefficients,
their covariance and the residual df. ``ancova`` adds, for a grouping factor, the Type II table and the
adjusted means.

Type II tests: in a model without interactions, the Type II F test of a covariate (all the columns of a
factor at once) is the extra-sum-of-squares test of adding it last, which equals the Wald F test of its
coefficients; partial eta squared = F q / (F q + residual df). Adjusted (least-squares) means are the
model predictions for each group at the mean of every other covariate (Searle, Speed & Milliken 1980).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from biostats.design import design_matrix
from biostats.references import REFERENCES

REFERENCES.setdefault(
    "searle1980",
    "Searle SR, Speed FM, Milliken GA (1980). Population marginal means in the linear model: an alternative to "
    "least squares means. Am Stat 34:216-221.",
)
REFERENCES.setdefault(
    "cohen1973",
    "Cohen J (1973). Eta-squared and partial eta-squared in fixed factor ANOVA designs. Educ Psychol Meas "
    "33:107-112.",
)


@dataclass
class LinearModelResult:
    table: pd.DataFrame  # term, coef, se, ci_low, ci_high, t, p (intercept included)
    params: pd.Series
    vcov: pd.DataFrame
    df_resid: float
    n: int
    r2: float
    r2_adj: float
    sigma: float  # residual standard deviation
    reference_levels: dict[str, str]
    terms_of: dict[str, list[str]]  # covariate -> its design columns (several for a factor)
    type2: pd.DataFrame  # source, df, sum_sq, F, p, partial_eta2; last row "Residual"
    residuals: np.ndarray
    fitted: np.ndarray
    references: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class AncovaResult(LinearModelResult):
    group: str = ""
    group_terms: list[str] = field(default_factory=list)
    adjusted_means: pd.DataFrame | None = None  # group, mean, se, ci_low, ci_high


def _terms_of(
    covariates: Sequence[str], columns: Sequence[str], refs: Mapping[str, str]
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for col in covariates:
        cols = (
            [c for c in columns if c.startswith(f"{col}_")]
            if col in refs
            else [col]
            if col in columns
            else []
        )
        if cols:  # a factor with one level or a constant column has no term
            out[col] = cols
    return out


def _wald_block(params: np.ndarray, vcov: np.ndarray, idx: list[int]) -> float:
    b = params[idx]
    return float(b @ np.linalg.solve(vcov[np.ix_(idx, idx)], b)) / len(idx)


def linear_regression(
    df: pd.DataFrame,
    outcome: str,
    covariates: Sequence[str],
    *,
    levels: Mapping[str, Sequence[str]] | None = None,
    alpha: float = 0.05,
) -> LinearModelResult:
    """OLS of ``outcome`` on ``covariates`` (complete cases); ``levels`` fixes the factor levels."""
    frame = df[[outcome, *covariates]].dropna()
    y = pd.to_numeric(frame[outcome], errors="coerce")
    if y.isna().any():
        raise ValueError(f"{outcome} must be numeric")
    x, refs = design_matrix(frame, covariates, levels=levels)
    n, p = x.shape
    if n <= p:
        raise ValueError(f"{n} complete rows for {p} parameters: the model cannot be estimated")
    res = sm.OLS(y.to_numpy(dtype=float), x).fit()
    names = list(x.columns)
    params = np.asarray(res.params, dtype=float)
    vcov = np.asarray(res.cov_params(), dtype=float)
    se = np.sqrt(np.diag(vcov))
    df_resid = float(res.df_resid)
    crit = stats.t.ppf(1 - alpha / 2, df_resid)
    table = pd.DataFrame(
        {
            "term": names,
            "coef": params,
            "se": se,
            "ci_low": params - crit * se,
            "ci_high": params + crit * se,
            "t": params / se,
            "p": 2 * stats.t.sf(np.abs(params / se), df_resid),
        }
    )
    terms_of = _terms_of(covariates, names, refs)
    rss = float(res.ssr)
    rows = []
    for source, cols in terms_of.items():
        idx = [names.index(c) for c in cols]
        f_stat = _wald_block(params, vcov, idx)
        q = len(idx)
        rows.append(
            {
                "source": source,
                "df": float(q),
                "sum_sq": f_stat * q * rss / df_resid,
                "F": f_stat,
                "p": float(stats.f.sf(f_stat, q, df_resid)),
                "partial_eta2": f_stat * q / (f_stat * q + df_resid),
            }
        )
    rows.append(
        {
            "source": "Residual",
            "df": df_resid,
            "sum_sq": rss,
            "F": np.nan,
            "p": np.nan,
            "partial_eta2": np.nan,
        }
    )
    warnings: list[str] = []
    if n < 10 * (p - 1):
        warnings.append("fewer than 10 observations per parameter: estimates may be unstable")
    return LinearModelResult(
        table=table,
        params=pd.Series(params, index=names),
        vcov=pd.DataFrame(vcov, index=names, columns=names),
        df_resid=df_resid,
        n=int(n),
        r2=float(res.rsquared),
        r2_adj=float(res.rsquared_adj),
        sigma=float(np.sqrt(res.scale)),
        reference_levels=refs,
        terms_of=terms_of,
        type2=pd.DataFrame(rows),
        residuals=np.asarray(res.resid, dtype=float),
        fitted=np.asarray(res.fittedvalues, dtype=float),
        references=[REFERENCES["cohen1973"]],
        warnings=warnings,
    )


def adjusted_means(
    result: LinearModelResult,
    data: pd.DataFrame,
    group: str,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Least-squares means of ``group``: predictions for each level with numeric covariates at their mean and
    the levels of any other factor weighted equally (the emmeans / SAS LSMEANS convention).
    """
    all_levels = _levels_from(result)
    levels = all_levels[group]
    x, _ = design_matrix(
        data[list(result.terms_of)].dropna(), list(result.terms_of), levels=all_levels
    )
    base = x.mean(axis=0)
    for factor, factor_levels in all_levels.items():
        for term in result.terms_of.get(factor, []):
            base[term] = 1.0 / len(factor_levels)
    crit = stats.t.ppf(1 - alpha / 2, result.df_resid)
    names = list(result.params.index)
    rows = []
    for level in levels:
        c = base.copy()
        for term in result.terms_of[group]:
            c[term] = 1.0 if term == f"{group}_{level}" else 0.0
        vec = c.reindex(names).to_numpy(dtype=float)
        mean = float(vec @ result.params.to_numpy())
        se = float(np.sqrt(vec @ result.vcov.to_numpy() @ vec))
        rows.append(
            {
                "group": level,
                "mean": mean,
                "se": se,
                "ci_low": mean - crit * se,
                "ci_high": mean + crit * se,
            }
        )
    return pd.DataFrame(rows)


def _levels_from(result: LinearModelResult) -> dict[str, list[str]]:
    """The factor levels a fitted model used (reference first)."""
    out: dict[str, list[str]] = {}
    for col, ref in result.reference_levels.items():
        out[col] = [ref, *[c.split(f"{col}_", 1)[1] for c in result.terms_of.get(col, [])]]
    return out


def ancova(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    covariates: Sequence[str] = (),
    *,
    levels: Mapping[str, Sequence[str]] | None = None,
    alpha: float = 0.05,
) -> AncovaResult:
    """ANCOVA: group differences in ``outcome`` adjusted for ``covariates`` (Type II F test, adjusted means)."""
    covs = [c for c in covariates if c not in (outcome, group)]
    frame = df[[outcome, group, *covs]].dropna()
    frame = frame.assign(**{group: frame[group].astype(str)})
    fit = linear_regression(frame, outcome, [group, *covs], levels=levels, alpha=alpha)
    if not fit.terms_of.get(group):
        raise ValueError(f"{group} needs at least two levels")
    means = adjusted_means(fit, frame, group, alpha=alpha)
    return AncovaResult(
        **{k: getattr(fit, k) for k in fit.__dataclass_fields__},
        group=group,
        group_terms=list(fit.terms_of[group]),
        adjusted_means=means,
    )


__all__ = ["AncovaResult", "LinearModelResult", "adjusted_means", "ancova", "linear_regression"]
