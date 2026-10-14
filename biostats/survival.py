"""
Time-to-event analysis (audit S11): Kaplan-Meier curves, log-rank test and Cox proportional hazards.

statsmodels and scipy provide the estimators; ``lifelines`` (optional ``survival`` extra) adds the
proportional-hazards diagnostic when installed. Follow-up times must be non-negative and the event
indicator 1 = event, 0 = censored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight

from biostats._types import CI
from biostats.design import design_matrix
from biostats.references import REFERENCES

REFERENCES.setdefault(
    "kaplan1958",
    "Kaplan EL, Meier P (1958). Nonparametric estimation from incomplete observations. JASA 53:457-481.",
)
REFERENCES.setdefault(
    "mantel1966",
    "Mantel N (1966). Evaluation of survival data and two new rank order statistics arising in its consideration. Cancer Chemother Rep 50:163-170.",
)
REFERENCES.setdefault(
    "cox1972", "Cox DR (1972). Regression models and life-tables. JRSS B 34:187-220."
)
REFERENCES.setdefault(
    "efron1977",
    "Efron B (1977). The efficiency of Cox's likelihood function for censored data. JASA 72:557-565.",
)
REFERENCES.setdefault(
    "grambsch1994",
    "Grambsch PM, Therneau TM (1994). Proportional hazards tests and diagnostics based on weighted residuals. Biometrika 81:515-526.",
)


def _clean(times: Sequence, events: Sequence) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(pd.to_numeric(pd.Series(list(times)), errors="coerce"), dtype=float)
    e = pd.Series(list(events))
    if e.dtype == object or e.dtype == bool or str(e.dtype).startswith("str"):
        e = e.map(
            lambda v: (
                1.0
                if str(v).strip().lower()
                in ("1", "1.0", "true", "yes", "y", "event", "died", "dead")
                else 0.0
            )
        )
    e = np.asarray(pd.to_numeric(e, errors="coerce"), dtype=float)
    mask = ~(np.isnan(t) | np.isnan(e)) & (t >= 0)
    return t[mask], e[mask].astype(int)


@dataclass
class KMCurve:
    label: str
    times: np.ndarray
    survival: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    n: int
    events: int
    median: float | None
    at_risk_times: np.ndarray
    at_risk: np.ndarray
    se: np.ndarray | None = (
        None  # Greenwood standard error of the survival probability at ``times``
    )


@dataclass
class SurvivalComparison:
    curves: list[KMCurve]
    logrank_statistic: float | None
    logrank_df: int
    logrank_p: float | None
    references: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def kaplan_meier(
    times: Sequence, events: Sequence, label: str = "all", *, alpha: float = 0.05
) -> KMCurve:
    """Kaplan-Meier estimate with Greenwood (log-log) confidence band and the median survival time."""
    t, e = _clean(times, events)
    if len(t) == 0:
        raise ValueError("no usable follow-up times")
    sf = SurvfuncRight(t, e)
    surv = np.asarray(sf.surv_prob, dtype=float)
    se = np.asarray(sf.surv_prob_se, dtype=float)
    # Pointwise Greenwood interval on the log(-log S) scale (stays inside [0, 1])
    z = stats.norm.ppf(1 - alpha / 2)
    lower, upper = surv.copy(), surv.copy()
    inside = (surv > 0) & (surv < 1) & np.isfinite(se)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.log(-np.log(surv[inside]))
        se_c = se[inside] / (surv[inside] * np.abs(np.log(surv[inside])))
        lower[inside] = np.exp(-np.exp(c + z * se_c))
        upper[inside] = np.exp(-np.exp(c - z * se_c))
    median = sf.quantile(0.5)
    if median is not None and not np.isfinite(median):
        median = None
    unique_times = np.asarray(sf.surv_times, dtype=float)
    at_risk = np.array([(t >= u).sum() for u in unique_times])
    return KMCurve(
        label=str(label),
        times=unique_times,
        survival=np.asarray(sf.surv_prob, dtype=float),
        lower=np.clip(lower, 0, 1),
        upper=np.clip(upper, 0, 1),
        n=int(len(t)),
        events=int(e.sum()),
        median=None if median is None else float(median),
        at_risk_times=unique_times,
        at_risk=at_risk,
        se=se,
    )


def compare_survival(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    group_col: str | None = None,
    *,
    alpha: float = 0.05,
) -> SurvivalComparison:
    """Kaplan-Meier per group and the log-rank test (Mantel 1966) when there are two or more groups."""
    frame = df[[time_col, event_col] + ([group_col] if group_col else [])].dropna()
    curves: list[KMCurve] = []
    warnings: list[str] = []
    if not group_col:
        curves.append(kaplan_meier(frame[time_col], frame[event_col], "all", alpha=alpha))
        return SurvivalComparison(
            curves=curves,
            logrank_statistic=None,
            logrank_df=0,
            logrank_p=None,
            references=[REFERENCES["kaplan1958"]],
            warnings=warnings,
        )
    groups = [
        (str(level), part) for level, part in frame.groupby(group_col, observed=True, sort=True)
    ]
    for label, part in groups:
        if len(part) >= 2:
            curves.append(kaplan_meier(part[time_col], part[event_col], label, alpha=alpha))
        else:
            warnings.append(f"group {label} has fewer than 2 observations and was skipped")
    if len(curves) < 2:
        return SurvivalComparison(
            curves=curves,
            logrank_statistic=None,
            logrank_df=0,
            logrank_p=None,
            references=[REFERENCES["kaplan1958"]],
            warnings=warnings + ["log-rank needs at least two groups"],
        )
    t, e = _clean(frame[time_col], frame[event_col])
    g = frame[group_col].astype(str).to_numpy()
    keep = ~(
        pd.to_numeric(frame[time_col], errors="coerce").isna() | pd.isna(frame[event_col])
    ).to_numpy()
    g = g[keep]
    from statsmodels.duration.survfunc import survdiff

    stat, p = survdiff(t, e, g)
    return SurvivalComparison(
        curves=curves,
        logrank_statistic=float(stat),
        logrank_df=len(curves) - 1,
        logrank_p=float(p),
        references=[REFERENCES["kaplan1958"], REFERENCES["mantel1966"]],
        warnings=warnings,
    )


@dataclass
class CoxResult:
    table: pd.DataFrame  # term, log_hr, se, hr, ci_low, ci_high, z, p
    n: int
    events: int
    ties: str
    concordance: float | None
    ph_test: (
        pd.DataFrame | None
    )  # per term: statistic, p (Grambsch-Therneau) when lifelines is installed
    references: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # log hazard ratios, their covariance and the complete-data df n - p used when pooling imputations
    params: pd.Series | None = None
    vcov: pd.DataFrame | None = None
    df_resid: float | None = None


def cox_regression(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    covariates: Sequence[str],
    *,
    alpha: float = 0.05,
    ties: str = "efron",
    levels: Mapping[str, Sequence[str]] | None = None,
) -> CoxResult:
    """
    Cox proportional-hazards model (Efron ties by default); categorical covariates are one-hot coded (first level reference).

    ``levels`` fixes the factor levels (the same terms in every imputed dataset).
    """
    frame = df[[time_col, event_col, *covariates]].dropna()
    t, e = _clean(frame[time_col], frame[event_col])
    x, _ = design_matrix(frame, covariates, levels=levels, intercept=False)
    if x.shape[1] == 0:
        raise ValueError("no covariate with variation")
    model = PHReg(t, x.to_numpy(), status=e, ties=ties)
    res = model.fit()
    z = stats.norm.ppf(1 - alpha / 2)
    params = np.asarray(res.params, dtype=float)
    bse = np.asarray(res.bse, dtype=float)
    table = pd.DataFrame(
        {
            "term": list(x.columns),
            "log_hr": params,
            "se": bse,
            "hr": np.exp(params),
            "ci_low": np.exp(params - z * bse),
            "ci_high": np.exp(params + z * bse),
            "z": params / bse,
            "p": 2 * stats.norm.sf(np.abs(params / bse)),
        }
    )
    warnings: list[str] = []
    concordance = None
    ph_table = None
    try:
        from lifelines import CoxPHFitter
        from lifelines.statistics import proportional_hazard_test

        data = x.copy()
        data["_time"], data["_event"] = t, e
        cph = CoxPHFitter().fit(data, duration_col="_time", event_col="_event")
        concordance = float(cph.concordance_index_)
        ph = proportional_hazard_test(cph, data, time_transform="rank")
        summary = ph.summary
        ph_table = pd.DataFrame(
            {
                "term": list(summary.index),
                "statistic": summary["test_statistic"].to_numpy(),
                "p": summary["p"].to_numpy(),
            }
        )
        if (ph_table["p"] < 0.05).any():
            warnings.append(
                "the proportional-hazards assumption is questionable for: "
                + ", ".join(ph_table.loc[ph_table["p"] < 0.05, "term"])
                + " (Grambsch-Therneau p < 0.05)"
            )
    except ImportError:
        warnings.append(
            "install the 'survival' extra (lifelines) for the proportional-hazards test and the concordance index"
        )
    except Exception as exc:  # diagnostics are optional
        warnings.append(f"proportional-hazards diagnostic unavailable: {exc}")
    return CoxResult(
        table=table,
        n=int(len(t)),
        events=int(e.sum()),
        ties=ties,
        concordance=concordance,
        ph_test=ph_table,
        references=[REFERENCES["cox1972"], REFERENCES["efron1977"], REFERENCES["grambsch1994"]],
        warnings=warnings,
        params=pd.Series(params, index=list(x.columns)),
        vcov=pd.DataFrame(
            np.asarray(res.cov_params(), dtype=float),
            index=list(x.columns),
            columns=list(x.columns),
        ),
        df_resid=float(len(t) - x.shape[1]),
    )


__all__ = [
    "CI",
    "CoxResult",
    "KMCurve",
    "SurvivalComparison",
    "compare_survival",
    "cox_regression",
    "kaplan_meier",
]
