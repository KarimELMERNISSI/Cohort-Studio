"""Logistic and count regression with effect tables (audit S11)."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from biostats.design import design_matrix
from biostats.references import REFERENCES

logger = logging.getLogger(__name__)

REFERENCES.setdefault(
    "hosmer2013",
    "Hosmer DW, Lemeshow S, Sturdivant RX (2013). Applied Logistic Regression, 3rd ed. Wiley.",
)
REFERENCES.setdefault(
    "cameron2013",
    "Cameron AC, Trivedi PK (2013). Regression Analysis of Count Data, 2nd ed. Cambridge UP.",
)

# The AUC of a logistic model is computed on the data the model was fitted to: it overstates the discrimination to
# expect in new patients, so it is always shown with this label.
APPARENT_AUC = "apparent AUC (in-sample)"
APPARENT_AUC_NOTE = (
    "The apparent AUC is computed on the data the model was fitted to, so it overstates how well the model would "
    "discriminate in new patients; that needs internal validation (bootstrap or cross-validation) or an external "
    "dataset."
)


@dataclass
class RegressionResult:
    kind: str  # logistic | poisson | negative_binomial
    table: pd.DataFrame  # term, coef, se, ratio (OR/IRR), ci_low, ci_high, z, p
    n: int
    events: int | None
    llf: float
    aic: float
    pseudo_r2: float | None
    auc: float | None  # logistic only: the apparent (in-sample) AUC, see APPARENT_AUC
    auc_ci: tuple[float, float] | None
    dispersion: float | None
    reference_levels: dict[str, str]
    references: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # every coefficient (``const`` included) on the log scale, its covariance and the residual df n - p:
    # what pooling across imputed datasets needs
    params: pd.Series | None = None
    vcov: pd.DataFrame | None = None
    df_resid: float | None = None


def _design(
    df: pd.DataFrame, covariates: Sequence[str], levels: Mapping[str, Sequence[str]] | None = None
) -> tuple[pd.DataFrame, dict[str, str]]:
    return design_matrix(df, covariates, levels=levels)


EXTREME_LOG = 10.0  # a log ratio beyond this (a ratio below 5e-5 or above 22,000) with...
EXTREME_SE = 100.0  # ...a standard error above this: the likelihood has no maximum there


def extreme_estimates(log_estimates, se) -> str | None:
    """
    A warning when terms have a log ratio beyond EXTREME_LOG with a standard error above EXTREME_SE: a level without
    events (or where every row has the event) sends its ratio to 0 or to infinity (monotone likelihood, separation),
    and the fit stops at a finite but meaningless value with an interval without bound. The terms are counted, not
    named: a level held by a few rows is a participant's value, and the warning reaches the study dossier.
    """
    log = np.abs(np.asarray(log_estimates, dtype=float))
    spread = np.asarray(se, dtype=float)
    k = int(((log > EXTREME_LOG) & (spread > EXTREME_SE)).sum())
    if not k:
        return None
    return (
        f"{k} term{'s' if k > 1 else ''} {'have' if k > 1 else 'has'} an extreme estimate (a ratio near 0 or "
        "infinite, an interval without bound): a level without events, or where every row has the event (monotone "
        "likelihood, separation); such a ratio cannot be read. Clean or merge such levels on subject-matter grounds, "
        "or use Firth's penalised likelihood (not available here)."
    )


def _coefficients(res, x: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame | None]:
    """Regression coefficients and their covariance, restricted to the design columns."""
    k = x.shape[1]
    names = list(x.columns)
    params = pd.Series(np.asarray(res.params, dtype=float)[:k], index=names)
    try:
        cov = np.asarray(res.cov_params(), dtype=float)[:k, :k]
    except ValueError:  # no Hessian (a parameter at the boundary): no covariance to pool
        return params, None
    return params, pd.DataFrame(cov, index=names, columns=names)


def _table(res, x: pd.DataFrame, alpha: float, ratio_label: str) -> pd.DataFrame:
    z = stats.norm.ppf(1 - alpha / 2)
    params = np.asarray(res.params, dtype=float)
    bse = np.asarray(res.bse, dtype=float)
    with np.errstate(
        over="ignore"
    ):  # an extreme estimate's interval runs to infinity (extreme_estimates says so)
        table = pd.DataFrame(
            {
                "term": list(x.columns),
                "coef": params,
                "se": bse,
                ratio_label: np.exp(params),
                "ci_low": np.exp(params - z * bse),
                "ci_high": np.exp(params + z * bse),
                "z": params / bse,
                "p": 2 * stats.norm.sf(np.abs(params / bse)),
            }
        )
    return table[table["term"] != "const"].reset_index(drop=True)


def logistic_regression(
    df: pd.DataFrame,
    outcome_col: str,
    covariates: Sequence[str],
    *,
    event_level=None,
    alpha: float = 0.05,
    levels: Mapping[str, Sequence[str]] | None = None,
) -> RegressionResult:
    """
    Binary logistic regression; odds ratios with Wald CIs, DeLong AUC of the fitted probabilities (apparent: computed
    on the data the model was fitted to).

    ``levels`` fixes the factor levels (the same terms in every imputed dataset).
    """
    frame = df[[outcome_col, *covariates]].dropna()
    y_raw = frame[outcome_col]
    outcome_levels = sorted(y_raw.astype(str).unique())
    if len(outcome_levels) != 2:
        raise ValueError("the outcome must have exactly two levels")
    event_level = str(event_level) if event_level is not None else outcome_levels[1]
    y = (y_raw.astype(str) == event_level).astype(float).to_numpy()
    x, refs = _design(frame, covariates, levels)
    warnings: list[str] = []
    try:
        res = sm.Logit(y, x).fit(disp=0, maxiter=200)
    except Exception as exc:
        raise ValueError(f"the model did not converge: {exc}") from exc
    if not res.mle_retvals.get("converged", True):
        warnings.append(
            "the optimiser reported non-convergence; check for separation (a covariate perfectly predicting the outcome)"
        )
    auc = auc_ci = None
    try:
        from biostats.diagnostic import delong_auc

        d = delong_auc(
            ["1" if v == 1 else "0" for v in y],
            np.asarray(res.predict(x)),
            pos_label="1",
            alpha=alpha,
        )
        auc, auc_ci = d["auc"], (d["ci"].lower, d["ci"].upper)
    except Exception as exc:  # the AUC is reported as missing, the model itself stands
        logger.warning("logistic regression: AUC of the fitted probabilities unavailable: %s", exc)
    events = int(y.sum())
    params, vcov = _coefficients(res, x)
    if events < 10 * (x.shape[1] - 1) or (len(y) - events) < 10 * (x.shape[1] - 1):
        warnings.append(
            "fewer than 10 events per parameter: odds ratios may be unstable (Peduzzi rule of thumb)"
        )
    table = _table(res, x, alpha, "odds_ratio")
    if (extreme := extreme_estimates(table["coef"], table["se"])) is not None:
        warnings.append(extreme)
    return RegressionResult(
        kind="logistic",
        table=table,
        n=int(len(y)),
        events=events,
        llf=float(res.llf),
        aic=float(res.aic),
        pseudo_r2=float(res.prsquared),
        auc=auc,
        auc_ci=auc_ci,
        dispersion=None,
        reference_levels={**refs, outcome_col: f"event = {event_level}"},
        references=[REFERENCES["hosmer2013"]],
        warnings=warnings,
        params=params,
        vcov=vcov,
        df_resid=float(len(y) - x.shape[1]),
    )


def count_regression(
    df: pd.DataFrame,
    count_col: str,
    covariates: Sequence[str],
    *,
    exposure_col: str | None = None,
    family: str = "poisson",
    alpha: float = 0.05,
    levels: Mapping[str, Sequence[str]] | None = None,
) -> RegressionResult:
    """
    Poisson (or negative binomial) regression for counts, with an optional exposure offset; incidence-rate ratios.

    ``levels`` fixes the factor levels (the same terms in every imputed dataset).
    """
    cols = [count_col, *covariates] + ([exposure_col] if exposure_col else [])
    frame = df[cols].dropna()
    y = pd.to_numeric(frame[count_col], errors="coerce").to_numpy(dtype=float)
    x, refs = _design(frame, covariates, levels)
    offset = None
    if exposure_col:
        exposure = pd.to_numeric(frame[exposure_col], errors="coerce").to_numpy(dtype=float)
        if (exposure <= 0).any():
            raise ValueError("exposure must be positive")
        offset = np.log(exposure)
    warnings: list[str] = []
    if family == "poisson":
        res = sm.GLM(y, x, family=sm.families.Poisson(), offset=offset).fit()
        dispersion = float(res.pearson_chi2 / res.df_resid) if res.df_resid > 0 else None
        if dispersion is not None and dispersion > 1.5:
            warnings.append(
                f"over-dispersion (Pearson chi2/df = {dispersion:.2f}); consider the negative binomial family"
            )
    elif family == "negative_binomial":
        res = sm.NegativeBinomial(y, x, offset=offset).fit(disp=0, maxiter=200)
        dispersion = float(
            np.asarray(res.params, dtype=float)[-1]
        )  # NB2 alpha: variance = mu + alpha mu^2
    else:
        raise ValueError("family must be 'poisson' or 'negative_binomial'")
    if family == "negative_binomial":
        z = stats.norm.ppf(1 - alpha / 2)
        params = np.asarray(res.params, dtype=float)[: x.shape[1]]
        bse = np.asarray(res.bse, dtype=float)[: x.shape[1]]
        with np.errstate(over="ignore"):  # an extreme estimate's interval runs to infinity
            table = pd.DataFrame(
                {
                    "term": list(x.columns),
                    "coef": params,
                    "se": bse,
                    "rate_ratio": np.exp(params),
                    "ci_low": np.exp(params - z * bse),
                    "ci_high": np.exp(params + z * bse),
                    "z": params / bse,
                    "p": 2 * stats.norm.sf(np.abs(params / bse)),
                }
            )
        table = table[table["term"] != "const"].reset_index(drop=True)
    else:
        table = _table(res, x, alpha, "rate_ratio")
    params, vcov = _coefficients(res, x)
    if (extreme := extreme_estimates(table["coef"], table["se"])) is not None:
        warnings.append(extreme)
    return RegressionResult(
        kind=family,
        table=table,
        n=int(len(y)),
        events=int(y.sum()),
        llf=float(res.llf),
        aic=float(res.aic),
        pseudo_r2=None,
        auc=None,
        auc_ci=None,
        dispersion=dispersion,
        reference_levels=refs,
        references=[REFERENCES["cameron2013"]],
        warnings=warnings,
        params=params,
        vcov=vcov,
        df_resid=float(len(y) - x.shape[1]),
    )


__all__ = [
    "APPARENT_AUC",
    "APPARENT_AUC_NOTE",
    "RegressionResult",
    "extreme_estimates",
    "count_regression",
    "logistic_regression",
]
