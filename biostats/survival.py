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
    "brookmeyer1982",
    "Brookmeyer R, Crowley J (1982). A confidence interval for the median survival time. Biometrics 38:29-41.",
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


EVENT_WORDS = ("1", "1.0", "true", "yes", "y", "event", "died", "dead", "death", "deceased")
CENSORED_WORDS = ("0", "0.0", "false", "no", "n", "censored", "alive", "survived")


def _code(value: object) -> str:
    return str(int(value)) if isinstance(value, float) and value.is_integer() else repr(value)


def _refuse(unknown: pd.Series, codes: set[float]) -> ValueError:
    """The error for event codes other than 0 and 1, with the advice that fits the coding found."""
    counts = unknown.value_counts()
    listed = ", ".join(
        f"{_code(value)} ({n} row{'s' if n > 1 else ''})" for value, n in counts.head(5).items()
    )
    advice = (
        "If 1 means censored and 2 the event (as in R's survival package), recode 1 to 0 and 2 to 1."
        if codes == {1.0, 2.0}
        else "A competing event or another outcome needs its own analysis: recode it to 0 or 1 first."
    )
    return ValueError(
        f"the event indicator must be 1 (event) or 0 (censored), but it also holds {listed}. {advice}"
    )


def event_indicator(events: Sequence) -> np.ndarray:
    """
    1 = event, 0 = censored; missing values stay NaN. Text, categories and booleans are read with EVENT_WORDS and
    CENSORED_WORDS. Any other value is refused rather than guessed: a code of 2 would count an event twice in the
    Kaplan-Meier estimate, and an unknown word (a competing event, a misspelling) would silently become censored.
    """
    e = pd.Series(list(events))
    numeric = pd.api.types.is_numeric_dtype(e.dtype) and not pd.api.types.is_bool_dtype(e.dtype)
    if not numeric:
        words = e.map(lambda v: np.nan if pd.isna(v) else str(v).strip().lower())
        known = words.isna() | words.isin(EVENT_WORDS + CENSORED_WORDS)
        if not known.all():
            codes = set(pd.to_numeric(words.dropna(), errors="coerce").dropna().tolist())
            raise _refuse(e[~known], codes)
        return np.asarray(
            words.map(lambda w: np.nan if pd.isna(w) else float(w in EVENT_WORDS)), dtype=float
        )
    values = np.asarray(pd.to_numeric(e, errors="coerce"), dtype=float)
    unknown = ~np.isnan(values) & ~np.isin(values, (0.0, 1.0))
    if unknown.any():
        raise _refuse(
            pd.Series(values[unknown]), set(np.unique(values[~np.isnan(values)]).tolist())
        )
    return values


def _clean(times: Sequence, events: Sequence) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(pd.to_numeric(pd.Series(list(times)), errors="coerce"), dtype=float)
    e = event_indicator(events)
    mask = ~(np.isnan(t) | np.isnan(e)) & (t >= 0)
    return t[mask], e[mask].astype(int)


def _usable(frame: pd.DataFrame, time_col: str, event_col: str) -> tuple[pd.DataFrame, list[str]]:
    """
    The rows ``_clean`` keeps (known time >= 0, known event), so that groups and covariates stay aligned with the
    times; a warning says how many rows with a negative follow-up time were left out.
    """
    t = np.asarray(pd.to_numeric(frame[time_col], errors="coerce"), dtype=float)
    e = event_indicator(frame[event_col])
    negative = int((t < 0).sum())
    keep = ~(np.isnan(t) | np.isnan(e)) & (t >= 0)
    warnings = (
        [
            f"{negative} row{'s' if negative > 1 else ''} with a negative follow-up time "
            f"{'were' if negative > 1 else 'was'} left out"
        ]
        if negative
        else []
    )
    return frame.loc[keep], warnings


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
    # the median's interval: the same rule on the band's lower and upper curves (None: not reached)
    median_lower: float | None = None
    median_upper: float | None = None
    level: float = 0.95  # of the band and of the median's interval


@dataclass
class SurvivalComparison:
    curves: list[KMCurve]
    logrank_statistic: float | None
    logrank_df: int
    logrank_p: float | None
    references: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


QUANTILE_TOLERANCE = 1e-8  # a survival equal to 1 - p within this is on a flat stretch at 1 - p


def km_quantile(
    times: Sequence, survival: Sequence, p: float = 0.5, *, last: float | None = None
) -> float | None:
    """
    The p-quantile of a survival curve, read as R's quantile.survfit reads it (Brookmeyer & Crowley 1982): the
    first time at which S(t) <= 1 - p. Where the curve stays at exactly 1 - p, from the time it reaches it to the
    next time it falls (or to ``last``, the last follow-up time, when it never does), the midpoint of that stretch:
    times 1 to 10, every one an event, give a median of 5.5, where the first time at or below 0.5 gives 5. None
    when the curve never reaches 1 - p.
    """
    t = np.asarray(times, dtype=float)
    s = np.asarray(survival, dtype=float)
    target = 1.0 - p
    reached = np.flatnonzero(s <= target + QUANTILE_TOLERANCE)
    if reached.size == 0:
        return None
    first = int(reached[0])
    if abs(s[first] - target) >= QUANTILE_TOLERANCE:
        return float(t[first])
    falls = np.flatnonzero(s[first:] < target - QUANTILE_TOLERANCE)
    end = t[first + int(falls[0])] if falls.size else (t[-1] if last is None else max(last, t[-1]))
    return float((t[first] + end) / 2)


def kaplan_meier(
    times: Sequence, events: Sequence, label: str = "all", *, alpha: float = 0.05
) -> KMCurve:
    """
    Kaplan-Meier estimate with Greenwood (log-log) confidence band, and the median survival time with its interval
    (km_quantile on the curve and on the band's curves).
    """
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
    unique_times = np.asarray(sf.surv_times, dtype=float)
    lower, upper = np.clip(lower, 0, 1), np.clip(upper, 0, 1)
    last = float(
        t.max()
    )  # the curve's times are its event times; a stretch at 0.5 may run past the last one
    median = km_quantile(unique_times, surv, last=last)
    at_risk = np.array([(t >= u).sum() for u in unique_times])
    return KMCurve(
        label=str(label),
        times=unique_times,
        survival=np.asarray(sf.surv_prob, dtype=float),
        lower=lower,
        upper=upper,
        n=int(len(t)),
        events=int(e.sum()),
        median=median,
        at_risk_times=unique_times,
        at_risk=at_risk,
        se=se,
        median_lower=km_quantile(unique_times, lower, last=last),
        median_upper=km_quantile(unique_times, upper, last=last),
        level=1 - alpha,
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
    frame, warnings = _usable(
        df[[time_col, event_col] + ([group_col] if group_col else [])].dropna(), time_col, event_col
    )
    curves: list[KMCurve] = []
    if not group_col:
        curves.append(kaplan_meier(frame[time_col], frame[event_col], "all", alpha=alpha))
        return SurvivalComparison(
            curves=curves,
            logrank_statistic=None,
            logrank_df=0,
            logrank_p=None,
            references=[REFERENCES["kaplan1958"], REFERENCES["brookmeyer1982"]],
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
            references=[REFERENCES["kaplan1958"], REFERENCES["brookmeyer1982"]],
            warnings=warnings + ["log-rank needs at least two groups"],
        )
    t, e = _clean(frame[time_col], frame[event_col])
    g = frame[group_col].astype(str).to_numpy()
    from statsmodels.duration.survfunc import survdiff

    stat, p = survdiff(t, e, g)
    return SurvivalComparison(
        curves=curves,
        logrank_statistic=float(stat),
        logrank_df=len(curves) - 1,
        logrank_p=float(p),
        references=[
            REFERENCES["kaplan1958"],
            REFERENCES["brookmeyer1982"],
            REFERENCES["mantel1966"],
        ],
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
    # log hazard ratios, their covariance and the complete-data df (events - p, as mice) used when pooling
    params: pd.Series | None = None
    vcov: pd.DataFrame | None = None
    df_resid: float | None = None
    reference_levels: dict[str, str] = field(default_factory=dict)  # factor -> its reference level


class _Skipped(Exception):
    """The optional diagnostics were not requested."""


def cox_regression(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    covariates: Sequence[str],
    *,
    alpha: float = 0.05,
    ties: str = "efron",
    levels: Mapping[str, Sequence[str]] | None = None,
    diagnostics: bool = True,
) -> CoxResult:
    """
    Cox proportional-hazards model (Efron ties by default); categorical covariates are one-hot coded (first level reference).

    ``levels`` fixes the factor levels (the same terms in every imputed dataset). ``diagnostics=False`` skips the
    proportional-hazards test and the concordance (lifelines), which pooled analyses run on a few imputations only.
    """
    frame, warnings = _usable(df[[time_col, event_col, *covariates]].dropna(), time_col, event_col)
    t, e = _clean(frame[time_col], frame[event_col])
    x, refs = design_matrix(frame, covariates, levels=levels, intercept=False)
    if x.shape[1] == 0:
        raise ValueError("no covariate with variation")
    model = PHReg(t, x.to_numpy(), status=e, ties=ties)
    res = model.fit()
    z = stats.norm.ppf(1 - alpha / 2)
    params = np.asarray(res.params, dtype=float)
    bse = np.asarray(res.bse, dtype=float)
    if not (np.isfinite(params).all() and np.isfinite(bse).all()):
        # a level without events sends its log hazard ratio to minus infinity (monotone likelihood): the fit then
        # returns no finite estimate, for any term, and a table of missing values would read as a result
        raise ValueError(
            "the Cox model has no finite estimate: a level of a categorical covariate without events or held by very "
            "few rows, or a covariate that orders the event times perfectly, makes a hazard ratio infinite (monotone "
            "likelihood). Merge or clean such levels, then fit again."
        )
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
    concordance = None
    ph_table = None
    try:
        if not diagnostics:
            raise _Skipped
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
        if bool((ph_table["p"] < 0.05).any()):
            warnings.append(
                "the proportional-hazards assumption is questionable for: "
                + ", ".join(ph_table.loc[ph_table["p"] < 0.05, "term"])
                + " (Grambsch-Therneau p < 0.05)"
            )
    except _Skipped:
        pass  # diagnostics=False: no proportional-hazards test or concordance, as asked
    except ImportError:
        warnings.append(
            "install the 'survival' extra (lifelines) for the proportional-hazards test and the concordance index"
        )
    except (
        Exception
    ) as exc:  # diagnostics are optional; lifelines' message goes on with advice and a link
        reason = str(exc).split("https://")[0].split(". ")[0].strip().rstrip(".")[:160]
        warnings.append(f"proportional-hazards diagnostic unavailable: {reason}")
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
        df_resid=float(e.sum() - x.shape[1]),  # events - parameters, as mice for coxph
        reference_levels=refs,
    )


__all__ = [
    "CI",
    "CoxResult",
    "KMCurve",
    "SurvivalComparison",
    "compare_survival",
    "cox_regression",
    "kaplan_meier",
    "km_quantile",
]
