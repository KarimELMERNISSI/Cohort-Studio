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

Robust variances (``cov_type="HC2"`` or ``"HC3"``, MacKinnon & White 1985; Long & Ervin 2000): the coefficients,
the Wald tests and the adjusted means use a covariance that does not assume equal variances, with the t and F
reference distributions of R's ``lmtest::coeftest`` and ``car::Anova(white.adjust=)`` (the residual df). The
Type II table then leaves the sums of squares out, since its F no longer compares mean squares; partial eta
squared still describes the OLS fit.

``welch_means`` compares the group means of a one-way layout as Welch's ANOVA does. Its HC2 variances are each
group's own s^2 / n, and its degrees of freedom are Welch-type: n - 1 for a group mean, Satterthwaite's for a
difference between two groups, Welch's (1951) denominator df for the test of the group effect. On complete data
that test is Welch's ANOVA itself (for two groups, Welch's t-test). Pooled over imputed datasets, D1 combines its
Wald form (Welch's ANOVA without the second-order correction, which has no pooling rule) with Welch's df. The
residual df would be too large when the smaller groups vary more. Each term keeps its df (``term_df``,
``factor_df``), and the pooling over imputed datasets uses them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

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
REFERENCES.setdefault(
    "mackinnon1985",
    "MacKinnon JG, White H (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved "
    "finite sample properties. J Econom 29:305-325.",
)
REFERENCES.setdefault(
    "long2000",
    "Long JS, Ervin LH (2000). Using heteroscedasticity consistent standard errors in the linear regression model. "
    "Am Stat 54:217-224.",
)

COV_TYPES = ("nonrobust", "HC2", "HC3")
FEW_VALUES = 5  # a group's own variance is estimated poorly from fewer values


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
    # "HC2" or "HC3": robust standard errors and tests (unequal variances allowed)
    cov_type: str = "nonrobust"
    # complete-data df of a term or of a factor's test when it is not df_resid (welch_means)
    term_df: dict[str, float] = field(default_factory=dict)
    factor_df: dict[str, float] = field(default_factory=dict)


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
    cov_type: str = "nonrobust",
) -> LinearModelResult:
    """
    OLS of ``outcome`` on ``covariates`` (complete cases); ``levels`` fixes the factor levels. ``cov_type`` "HC2"
    or "HC3" gives robust standard errors and tests, which do not assume equal variances.
    """
    if cov_type not in COV_TYPES:
        raise ValueError(f"cov_type is one of {list(COV_TYPES)}, not {cov_type!r}")
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
    classical = np.asarray(res.cov_params(), dtype=float)
    vcov = classical
    robust = cov_type != "nonrobust"
    if robust:
        xa = x.to_numpy(dtype=float)
        leverage = np.einsum("ij,jk,ik->i", xa, np.linalg.pinv(xa.T @ xa), xa)
        if np.any(leverage > 1 - 1e-8):
            raise ValueError(
                f"a row has leverage 1 (a group with a single value, for example): its robust ({cov_type}) "
                "variance cannot be estimated"
            )
        vcov = np.asarray(res.cov_HC2 if cov_type == "HC2" else res.cov_HC3, dtype=float)
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
        f_ols = _wald_block(params, classical, idx)  # the sums of squares describe the OLS fit
        q = len(idx)
        rows.append(
            {
                "source": source,
                "df": float(q),
                # a robust F does not compare mean squares: no sum of squares beside it
                "sum_sq": np.nan if robust else f_ols * q * rss / df_resid,
                "F": f_stat,
                "p": float(stats.f.sf(f_stat, q, df_resid)),
                "partial_eta2": f_ols * q / (f_ols * q + df_resid),
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
        references=[REFERENCES["cohen1973"]]
        + ([REFERENCES["mackinnon1985"], REFERENCES["long2000"]] if robust else []),
        warnings=warnings,
        cov_type=cov_type,
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
    cov_type: str = "nonrobust",
) -> AncovaResult:
    """
    ANCOVA: group differences in ``outcome`` adjusted for ``covariates`` (Type II F test, adjusted means);
    ``cov_type`` "HC2" or "HC3" for robust tests and means that do not assume equal variances.
    """
    covs = [c for c in covariates if c not in (outcome, group)]
    frame = df[[outcome, group, *covs]].dropna()
    frame = frame.assign(**{group: frame[group].astype(str)})
    fit = linear_regression(
        frame, outcome, [group, *covs], levels=levels, alpha=alpha, cov_type=cov_type
    )
    if not fit.terms_of.get(group):
        raise ValueError(f"{group} needs at least two levels")
    means = adjusted_means(fit, frame, group, alpha=alpha)
    return AncovaResult(
        **{k: getattr(fit, k) for k in fit.__dataclass_fields__},
        group=group,
        group_terms=list(fit.terms_of[group]),
        adjusted_means=means,
    )


def welch_means(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    *,
    levels: Mapping[str, Sequence[str]] | None = None,
    alpha: float = 0.05,
) -> AncovaResult:
    """
    The group means of ``outcome`` compared without assuming equal variances, as Welch's ANOVA: the linear model on
    the groups with HC2 variances (each group's own s^2 / n) and Welch-type degrees of freedom, n - 1 for a group
    mean, Satterthwaite's for a difference from the reference group, Welch's (1951) df for the group effect, whose
    Type II row is Welch's ANOVA (for two groups, Welch's t-test). Each group needs two values or more, and a
    variance above zero.
    """
    if levels and group in levels:
        df = df[df[group].astype(str).isin([str(v) for v in levels[group]])]
    fit = ancova(df, outcome, group, levels=levels, alpha=alpha, cov_type="HC2")
    order = _levels_from(fit)[group]
    frame = df[[outcome, group]].dropna()
    values = pd.to_numeric(frame[outcome], errors="coerce").groupby(frame[group].astype(str))
    n = values.count().reindex(order).fillna(0).astype(float)
    empty = [level for level in order if n[level] == 0]
    if empty:
        raise ValueError(f"{', '.join(empty)}: no value in the data")
    variance = values.var(ddof=1).reindex(order)
    constant = [level for level in order if not variance[level] > 0]
    if constant:
        raise ValueError(
            f"{', '.join(constant)}: the same value throughout, so a variance of zero, which a comparison that "
            "does not assume equal variances cannot weigh"
        )
    v = variance / n  # the variance of each group mean
    reference = order[0]

    def satterthwaite(level: str) -> float:
        return float(
            (v[level] + v[reference]) ** 2
            / (v[level] ** 2 / (n[level] - 1) + v[reference] ** 2 / (n[reference] - 1))
        )

    term_df = {"const": float(n[reference] - 1)} | {
        f"{group}_{level}": satterthwaite(level) for level in order[1:]
    }
    share = (1 / v) / (1 / v).sum()
    k = len(order)
    spread = float(((1 - share) ** 2 / (n - 1)).sum() / (k**2 - 1))
    df2 = 1 / (3 * spread)
    # the Type II row is Welch's ANOVA itself: the Wald F of these variances with Welch's second-order correction
    # (1 for two groups), as on the observed data; the pooled D1 uses the Wald form, which has a pooling rule
    correction = 1 + 2 * (k - 2) * spread

    table = fit.table.copy()
    dfs = table["term"].map(term_df).astype(float).to_numpy()
    crit = stats.t.ppf(1 - alpha / 2, dfs)
    table["ci_low"] = table["coef"] - crit * table["se"]
    table["ci_high"] = table["coef"] + crit * table["se"]
    table["p"] = 2 * stats.t.sf(np.abs(table["t"]), dfs)
    type2 = fit.type2.copy()
    row = type2["source"] == group
    welch_f = type2.loc[row, "F"].to_numpy(dtype=float) / correction
    type2.loc[row, "F"] = welch_f
    type2.loc[row, "p"] = stats.f.sf(welch_f, k - 1, df2)
    means = fit.adjusted_means.copy() if fit.adjusted_means is not None else pd.DataFrame()
    means["df"] = means["group"].map(n - 1).astype(float)
    crit_means = stats.t.ppf(1 - alpha / 2, means["df"].to_numpy())
    means["ci_low"] = means["mean"] - crit_means * means["se"]
    means["ci_high"] = means["mean"] + crit_means * means["se"]
    few = [f"{level} ({int(n[level])})" for level in order if n[level] < FEW_VALUES]
    warnings = list(fit.warnings) + (
        [
            f"groups with fewer than {FEW_VALUES} values: {', '.join(few)}. A group's own variance is estimated "
            "poorly from so few values, and the test can then reject too often."
        ]
        if few
        else []
    )
    return replace(
        fit,
        table=table,
        type2=type2,
        adjusted_means=means,
        term_df=term_df,
        factor_df={group: df2},
        references=[REFERENCES["mackinnon1985"], REFERENCES["welch1951"]],
        warnings=warnings,
    )


__all__ = [
    "COV_TYPES",
    "FEW_VALUES",
    "AncovaResult",
    "LinearModelResult",
    "adjusted_means",
    "ancova",
    "linear_regression",
    "welch_means",
]
