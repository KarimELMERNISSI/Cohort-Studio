"""Logistic and count regression with effect tables (audit S11)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from biostats.references import REFERENCES

REFERENCES.setdefault(
    "hosmer2013",
    "Hosmer DW, Lemeshow S, Sturdivant RX (2013). Applied Logistic Regression, 3rd ed. Wiley.",
)
REFERENCES.setdefault(
    "cameron2013",
    "Cameron AC, Trivedi PK (2013). Regression Analysis of Count Data, 2nd ed. Cambridge UP.",
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
    auc: float | None
    auc_ci: tuple[float, float] | None
    dispersion: float | None
    reference_levels: dict[str, str]
    references: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _design(df: pd.DataFrame, covariates: Sequence[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    raw = df[list(covariates)]
    refs: dict[str, str] = {}
    for col in covariates:
        if (
            not pd.api.types.is_numeric_dtype(raw[col])
            or raw[col].nunique(dropna=True) <= 2
            and raw[col].dtype == object
        ):
            levels = sorted(raw[col].dropna().astype(str).unique())
            if levels:
                refs[col] = levels[0]
    x = pd.get_dummies(raw.astype({c: str for c in refs}), drop_first=True, dtype=float)
    x = x.loc[:, x.std(ddof=0) > 0]
    return sm.add_constant(x, has_constant="add"), refs


def _table(res, x: pd.DataFrame, alpha: float, ratio_label: str) -> pd.DataFrame:
    z = stats.norm.ppf(1 - alpha / 2)
    params = np.asarray(res.params, dtype=float)
    bse = np.asarray(res.bse, dtype=float)
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
) -> RegressionResult:
    """Binary logistic regression; odds ratios with Wald CIs, DeLong AUC of the fitted probabilities."""
    frame = df[[outcome_col, *covariates]].dropna()
    y_raw = frame[outcome_col]
    levels = sorted(y_raw.astype(str).unique())
    if len(levels) != 2:
        raise ValueError("the outcome must have exactly two levels")
    event_level = str(event_level) if event_level is not None else levels[1]
    y = (y_raw.astype(str) == event_level).astype(float).to_numpy()
    x, refs = _design(frame, covariates)
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
    except Exception:
        pass
    events = int(y.sum())
    if events < 10 * (x.shape[1] - 1) or (len(y) - events) < 10 * (x.shape[1] - 1):
        warnings.append(
            "fewer than 10 events per parameter: odds ratios may be unstable (Peduzzi rule of thumb)"
        )
    return RegressionResult(
        kind="logistic",
        table=_table(res, x, alpha, "odds_ratio"),
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
    )


def count_regression(
    df: pd.DataFrame,
    count_col: str,
    covariates: Sequence[str],
    *,
    exposure_col: str | None = None,
    family: str = "poisson",
    alpha: float = 0.05,
) -> RegressionResult:
    """Poisson (or negative binomial) regression for counts, with an optional exposure offset; incidence-rate ratios."""
    cols = [count_col, *covariates] + ([exposure_col] if exposure_col else [])
    frame = df[cols].dropna()
    y = pd.to_numeric(frame[count_col], errors="coerce").to_numpy(dtype=float)
    x, refs = _design(frame, covariates)
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
    )


__all__ = ["RegressionResult", "count_regression", "logistic_regression"]
