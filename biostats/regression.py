"""Logistic and count regression with effect tables (audit S11)."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from biostats.design import design_matrix, dominant_counts, indicator_counts, level_counts
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
    # factor -> level -> (rows, events, rows with an event) among the rows the model read
    # (biostats.design.level_counts)
    level_counts: dict[str, dict[str, tuple[int, int, int]]] = field(default_factory=dict)
    # the same for a numeric or True/False covariate that holds two values (biostats.design.indicator_counts)
    indicator_counts: dict[str, dict[str, tuple[int, int, int]]] = field(default_factory=dict)
    # on and off the commonest value of a numeric covariate of more than two values, when at least half of the rows
    # hold it (biostats.design.dominant_counts)
    dominant_counts: dict[str, dict[str, tuple[int, int, int]]] = field(default_factory=dict)
    # a logistic model's two outcome levels as read, sorted: the event is one of them
    outcome_levels: tuple[str, ...] = ()
    converged: bool = (
        True  # False: the estimates may not be the maximum of the likelihood (a warning says so)
    )
    dispersion_se: float | None = None  # of a negative binomial model's dispersion
    # a count model's rows that hold at least one event: its events are a sum of counts, not people
    rows_with_events: int | None = None


def _design(
    df: pd.DataFrame,
    covariates: Sequence[str],
    levels: Mapping[str, Sequence[str]] | None = None,
    drop_constant: bool | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    return design_matrix(df, covariates, levels=levels, drop_constant=drop_constant)


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
    drop_constant: bool | None = None,
) -> RegressionResult:
    """
    Binary logistic regression; odds ratios with Wald CIs, DeLong AUC of the fitted probabilities (apparent: computed
    on the data the model was fitted to).

    ``levels`` fixes the factor levels (the same terms in every imputed dataset); ``drop_constant``: as
    :func:`biostats.design.design_matrix`.
    """
    frame = df[[outcome_col, *covariates]].dropna()
    y_raw = frame[outcome_col]
    outcome_levels = sorted(y_raw.astype(str).unique())
    if len(outcome_levels) != 2:
        raise ValueError("the outcome must have exactly two levels")
    event_level = str(event_level) if event_level is not None else outcome_levels[1]
    y = (y_raw.astype(str) == event_level).astype(float).to_numpy()
    x, refs = _design(frame, covariates, levels, drop_constant)
    warnings: list[str] = []
    try:
        res = sm.Logit(y, x).fit(disp=0, maxiter=200)
    except Exception as exc:
        raise ValueError(f"the model did not converge: {exc}") from exc
    converged = bool(res.mle_retvals.get("converged", True))
    if not converged:
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
        level_counts=level_counts(frame, covariates, y, levels=levels),
        indicator_counts=indicator_counts(frame, covariates, y, levels=levels),
        dominant_counts=dominant_counts(frame, covariates, y, levels=levels),
        outcome_levels=tuple(outcome_levels),
        converged=converged,
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
    drop_constant: bool | None = None,
) -> RegressionResult:
    """
    Poisson (or negative binomial) regression for counts, with an optional exposure offset; incidence-rate ratios.

    ``levels`` fixes the factor levels (the same terms in every imputed dataset); ``drop_constant``: as
    :func:`biostats.design.design_matrix`.
    """
    cols = [count_col, *covariates] + ([exposure_col] if exposure_col else [])
    frame = df[cols].dropna()
    y = pd.to_numeric(frame[count_col], errors="coerce").to_numpy(dtype=float)
    x, refs = _design(frame, covariates, levels, drop_constant)
    offset = None
    if exposure_col:
        exposure = pd.to_numeric(frame[exposure_col], errors="coerce").to_numpy(dtype=float)
        if (exposure <= 0).any():
            raise ValueError("exposure must be positive")
        offset = np.log(exposure)
    warnings: list[str] = []
    converged, dispersion_se = True, None
    if family == "poisson":
        # statsmodels stops on the deviance and takes the covariance from the weights of the step before the last:
        # the standard errors then sit some 1e-7 (relative) from those at the optimum, at a point that moves with
        # the start. One more fit from where it stopped, a single step, ends within 1e-13 of them from any start
        model = sm.GLM(y, x, family=sm.families.Poisson(), offset=offset)
        res = model.fit(tol=1e-10)
        res = model.fit(start_params=np.asarray(res.params, dtype=float), tol=1e-10)
        converged = bool(getattr(res, "converged", True))
        dispersion = float(res.pearson_chi2 / res.df_resid) if res.df_resid > 0 else None
        if dispersion is not None and dispersion > 1.5:
            warnings.append(
                f"over-dispersion (Pearson chi2/df = {dispersion:.2f}); consider the negative binomial family"
            )
    elif family == "negative_binomial":
        model = sm.NegativeBinomial(y, x, offset=offset)
        res = model.fit(
            disp=0, maxiter=200
        )  # BFGS: it stops at a gradient of 1e-5, some 1e-3 standard error away
        try:
            # Newton steps from there end at the optimum (1e-11 standard error whatever the path taken)
            with catch_warnings():
                simplefilter("ignore")
                finished = model.fit(
                    start_params=np.asarray(res.params, dtype=float),
                    method="newton",
                    disp=0,
                    maxiter=100,
                )
            if (
                finished.mle_retvals.get("converged", False)
                and np.isfinite(np.asarray(finished.params, dtype=float)).all()
                and np.isfinite(np.asarray(finished.bse, dtype=float)).all()
            ):
                res = finished
        except (
            Exception
        ) as exc:  # a dispersion at its boundary has no Hessian to step with: the BFGS fit stands
            logger.debug("negative binomial: Newton steps not taken: %s", type(exc).__name__)
        converged = bool(res.mle_retvals.get("converged", True))
        if not converged:
            warnings.append(
                "the optimiser reported non-convergence: the estimates may not be the maximum of the likelihood"
            )
        dispersion = float(
            np.asarray(res.params, dtype=float)[-1]
        )  # NB2 alpha: variance = mu + alpha mu^2
        spread = float(np.asarray(res.bse, dtype=float)[-1])
        dispersion_se = spread if np.isfinite(spread) else None
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
        rows_with_events=int((np.asarray(y) > 0).sum()),
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
        level_counts=level_counts(frame, covariates, y, levels=levels),
        indicator_counts=indicator_counts(frame, covariates, y, levels=levels),
        dominant_counts=dominant_counts(frame, covariates, y, levels=levels),
        converged=converged,
        dispersion_se=dispersion_se,
    )


__all__ = [
    "APPARENT_AUC",
    "APPARENT_AUC_NOTE",
    "RegressionResult",
    "extreme_estimates",
    "count_regression",
    "logistic_regression",
]
