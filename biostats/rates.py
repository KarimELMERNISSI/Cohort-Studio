"""Incidence rates, rate ratios, Mantel-Haenszel stratified measures and standardised ratios (audit S11)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.rates import confint_poisson, confint_poisson_2indep, test_poisson_2indep

from biostats._types import CI
from biostats.references import REFERENCES

REFERENCES.setdefault(
    "mantel1959",
    "Mantel N, Haenszel W (1959). Statistical aspects of the analysis of data from retrospective studies of disease. JNCI 22:719-748.",
)
REFERENCES.setdefault(
    "breslow1980",
    "Breslow NE, Day NE (1980). Statistical Methods in Cancer Research, Vol. I. IARC.",
)
REFERENCES.setdefault(
    "greenland1985",
    "Greenland S, Robins JM (1985). Estimation of a common effect parameter from sparse follow-up data. Biometrics 41:55-68.",
)
REFERENCES.setdefault(
    "ulm1990",
    "Ulm K (1990). A simple method to calculate the confidence interval of a standardized mortality ratio. Am J Epidemiol 131:373-375.",
)
REFERENCES.setdefault(
    "garwood1936",
    "Garwood F (1936). Fiducial limits for the Poisson distribution. Biometrika 28:437-442.",
)
REFERENCES.setdefault(
    "clopper1934",
    "Clopper CJ, Pearson ES (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. Biometrika 26:404-413.",
)


@dataclass
class IncidenceRate:
    events: int
    person_time: float
    rate: float  # per unit of person-time
    ci: CI
    per: float  # multiplier used for display (e.g. 1000)
    references: list[str] = field(default_factory=list)


def incidence_rate(
    events: int, person_time: float, *, per: float = 1000.0, alpha: float = 0.05
) -> IncidenceRate:
    """Rate = events / person-time with the exact (Garwood) Poisson interval."""
    if person_time <= 0:
        raise ValueError("person-time must be positive")
    lo, hi = confint_poisson(int(events), float(person_time), method="exact-c", alpha=alpha)
    return IncidenceRate(
        events=int(events),
        person_time=float(person_time),
        rate=events / person_time,
        ci=CI(lower=float(lo), upper=float(hi), level=1 - alpha, method="exact Poisson (Garwood)"),
        per=per,
        references=[REFERENCES["garwood1936"]],
    )


@dataclass
class RateRatio:
    exposed: IncidenceRate
    reference: IncidenceRate
    ratio: float
    ci: CI
    p_value: float
    rate_difference: float
    test: str = "exact conditional"  # the test of IRR = 1 behind p_value, consistent with ``ci``
    references: list[str] = field(default_factory=list)


def _exact_conditional(
    events_exposed: int,
    time_exposed: float,
    events_reference: int,
    time_reference: float,
    alpha: float,
) -> tuple[float, float, float]:
    """
    The exact conditional interval of the rate ratio and its two-sided p-value for IRR = 1.

    Given the total number of events, the exposed events are binomial with probability IRR * t1 / (IRR * t1 + t0).
    The Clopper-Pearson interval of that probability, mapped to the ratio, is the interval of R's ``poisson.test``.
    The p-value is twice the smaller binomial tail (capped at 1), which excludes 1 exactly when the interval does.
    R's ``poisson.test`` reports the same interval with a minimum-likelihood p-value instead, which can fall below
    0.05 while its interval still contains 1.
    """
    total = events_exposed + events_reference
    ci = stats.binomtest(events_exposed, total).proportion_ci(
        confidence_level=1 - alpha, method="exact"
    )
    scale = time_reference / time_exposed
    upper = ci.high / (1 - ci.high) * scale if ci.high < 1 else float("inf")
    null = time_exposed / (time_exposed + time_reference)
    tails = (
        stats.binom.cdf(events_exposed, total, null),
        stats.binom.sf(events_exposed - 1, total, null),
    )
    return ci.low / (1 - ci.low) * scale, upper, float(min(1.0, 2 * min(tails)))


def rate_ratio(
    events_exposed: int,
    time_exposed: float,
    events_reference: int,
    time_reference: float,
    *,
    per: float = 1000.0,
    alpha: float = 0.05,
    ci_method: Literal["exact", "score"] = "exact",
) -> RateRatio:
    """
    Incidence rate ratio with an interval and a test that agree: by default the exact conditional interval with
    the exact conditional test (see ``_exact_conditional``); ``ci_method="score"`` gives the score interval with
    the score test, which need events in both groups.
    """
    if events_exposed + events_reference == 0:
        raise ValueError("there are no events in either group, so the rate ratio is undefined")
    exp_ = incidence_rate(events_exposed, time_exposed, per=per, alpha=alpha)
    ref = incidence_rate(events_reference, time_reference, per=per, alpha=alpha)
    if ci_method == "exact":
        lo, hi, p_value = _exact_conditional(
            int(events_exposed),
            float(time_exposed),
            int(events_reference),
            float(time_reference),
            alpha,
        )
        method, test, cited = (
            "exact conditional (Clopper-Pearson)",
            "exact conditional",
            ["garwood1936", "clopper1934"],
        )
    elif ci_method == "score":
        if min(events_exposed, events_reference) == 0:
            raise ValueError(
                "the score interval needs events in both groups; use the exact interval"
            )
        counts = (events_exposed, time_exposed, events_reference, time_reference)
        lo, hi = confint_poisson_2indep(*counts, method="score", compare="ratio", alpha=alpha)
        p_value = float(test_poisson_2indep(*counts, method="score", compare="ratio").pvalue)
        method, test, cited = "score", "score", ["garwood1936"]
    else:
        raise ValueError(f"unknown interval method {ci_method!r}: use 'exact' or 'score'")
    return RateRatio(
        exposed=exp_,
        reference=ref,
        ratio=exp_.rate / ref.rate if ref.rate > 0 else float("inf"),
        ci=CI(lower=float(lo), upper=float(hi), level=1 - alpha, method=method),
        p_value=float(p_value),
        rate_difference=exp_.rate - ref.rate,
        test=test,
        references=[REFERENCES[key] for key in cited],
    )


@dataclass
class StratifiedResult:
    n_strata: int
    or_mh: float
    or_mh_ci: CI
    rr_mh: float
    rr_mh_ci: CI
    crude_or: float
    breslow_day_statistic: float
    breslow_day_p: float
    mh_test_statistic: float
    mh_test_p: float
    per_stratum: pd.DataFrame
    crude_rr: float = float("nan")  # unadjusted risk ratio from the pooled table
    references: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _differ(crude: float, adjusted: float) -> bool:
    """The crude and adjusted ratios are known and differ by more than 10%."""
    known = np.isfinite(crude) and np.isfinite(adjusted) and crude > 0 and adjusted > 0
    return bool(known and abs(np.log(crude) - np.log(adjusted)) > np.log(1.1))


def _change_in_estimate(crude_or: float, or_mh: float, crude_rr: float, rr_mh: float) -> list[str]:
    """
    The 10% change-in-estimate hint, read on the risk ratio first. Odds ratios are non-collapsible: the crude and
    adjusted odds ratios can differ when the stratum variable is not a confounder at all, while the risk ratios
    then agree.
    """
    rr_known = bool(np.isfinite(crude_rr) and np.isfinite(rr_mh))
    if rr_known and _differ(crude_rr, rr_mh):
        return [
            "the crude and adjusted risk ratios differ by more than 10%: the stratum variable may be a confounder"
        ]
    if not _differ(crude_or, or_mh):
        return []
    if rr_known:
        return [
            "the crude and adjusted odds ratios differ by more than 10% but the risk ratios do not: odds ratios "
            "are non-collapsible, so this difference need not mean confounding"
        ]
    return [
        "the crude and adjusted odds ratios differ by more than 10%: the stratum variable may be a confounder, "
        "although odds ratios can also differ without confounding (they are non-collapsible)"
    ]


def _rr_mh(tables: Sequence[np.ndarray], alpha: float) -> tuple[float, CI]:
    """Mantel-Haenszel risk ratio with the Greenland-Robins variance."""
    num = den = 0.0
    var_num = 0.0
    for tbl in tables:
        a, b, c, d = tbl[0, 0], tbl[0, 1], tbl[1, 0], tbl[1, 1]
        n1, n0, n = a + b, c + d, a + b + c + d
        if n == 0:
            continue
        num += a * n0 / n
        den += c * n1 / n
        var_num += (n1 * n0 * (a + c) - a * c * n) / n**2
    rr = num / den if den > 0 else float("nan")
    se = np.sqrt(var_num / (num * den)) if num > 0 and den > 0 else float("nan")
    z = stats.norm.ppf(1 - alpha / 2)
    return float(rr), CI(
        lower=float(np.exp(np.log(rr) - z * se)),
        upper=float(np.exp(np.log(rr) + z * se)),
        level=1 - alpha,
        method="Greenland-Robins",
    )


def mantel_haenszel(
    df: pd.DataFrame,
    exposure_col: str,
    outcome_col: str,
    stratum_col: str,
    *,
    exposure_level=None,
    event_level=None,
    alpha: float = 0.05,
) -> StratifiedResult:
    """Pooled odds and risk ratios across strata (Mantel-Haenszel), Breslow-Day homogeneity test, MH test of no association."""
    frame = df[[exposure_col, outcome_col, stratum_col]].dropna().astype(str)
    exp_levels = sorted(frame[exposure_col].unique())
    out_levels = sorted(frame[outcome_col].unique())
    if len(exp_levels) != 2 or len(out_levels) != 2:
        raise ValueError("exposure and outcome must both be binary")
    exposure_level = str(exposure_level) if exposure_level is not None else exp_levels[1]
    event_level = str(event_level) if event_level is not None else out_levels[1]
    tables, rows, warnings = [], [], []
    for stratum, part in frame.groupby(stratum_col, sort=True):
        e = part[exposure_col] == exposure_level
        y = part[outcome_col] == event_level
        tbl = np.array(
            [
                [int((e & y).sum()), int((e & ~y).sum())],
                [int((~e & y).sum()), int((~e & ~y).sum())],
            ],
            dtype=float,
        )
        tables.append(tbl)
        a, b, c, d = tbl.ravel()
        rows.append(
            {
                "stratum": stratum,
                "exposed_events": int(a),
                "exposed_total": int(a + b),
                "reference_events": int(c),
                "reference_total": int(c + d),
                "odds_ratio": (a * d) / (b * c) if b * c > 0 else float("nan"),
                "risk_ratio": (a / (a + b)) / (c / (c + d))
                if (a + b) > 0 and (c + d) > 0 and c > 0
                else float("nan"),
            }
        )
        if min(a + b, c + d) < 5:
            warnings.append(f"stratum {stratum}: fewer than 5 subjects in a group")
    if len(tables) < 2:
        raise ValueError("at least two strata are required")
    st = StratifiedTable(tables)
    or_ci = st.oddsratio_pooled_confint(alpha=alpha)
    rr, rr_ci = _rr_mh(tables, alpha)
    bd = st.test_equal_odds(adjust=False)
    mh = st.test_null_odds(correction=True)
    pooled = np.sum(tables, axis=0)
    crude = (
        (pooled[0, 0] * pooled[1, 1]) / (pooled[0, 1] * pooled[1, 0])
        if pooled[0, 1] * pooled[1, 0] > 0
        else float("nan")
    )
    a, b, c, d = pooled.ravel()
    crude_rr = (a / (a + b)) / (c / (c + d)) if a + b > 0 and c + d > 0 and c > 0 else float("nan")
    warnings.extend(_change_in_estimate(crude, float(st.oddsratio_pooled), crude_rr, rr))
    return StratifiedResult(
        n_strata=len(tables),
        or_mh=float(st.oddsratio_pooled),
        or_mh_ci=CI(
            lower=float(or_ci[0]),
            upper=float(or_ci[1]),
            level=1 - alpha,
            method="Mantel-Haenszel (Robins-Breslow-Greenland)",
        ),
        rr_mh=rr,
        rr_mh_ci=rr_ci,
        crude_or=float(crude),
        breslow_day_statistic=float(bd.statistic),
        breslow_day_p=float(bd.pvalue),
        mh_test_statistic=float(mh.statistic),
        mh_test_p=float(mh.pvalue),
        per_stratum=pd.DataFrame(rows),
        crude_rr=float(crude_rr),
        references=[
            REFERENCES["mantel1959"],
            REFERENCES["breslow1980"],
            REFERENCES["greenland1985"],
        ],
        warnings=warnings,
    )


@dataclass
class StandardisedRatio:
    observed: int
    expected: float
    ratio: float
    ci: CI
    p_value: float
    references: list[str] = field(default_factory=list)


def standardised_ratio(observed: int, expected: float, *, alpha: float = 0.05) -> StandardisedRatio:
    """SMR / SIR = observed / expected with the exact Poisson interval on the observed count."""
    if expected <= 0:
        raise ValueError("expected count must be positive")
    lo, hi = confint_poisson(int(observed), 1.0, method="exact-c", alpha=alpha)
    p = float(
        min(
            1.0,
            2
            * min(stats.poisson.cdf(observed, expected), stats.poisson.sf(observed - 1, expected)),
        )
    )
    return StandardisedRatio(
        observed=int(observed),
        expected=float(expected),
        ratio=observed / expected,
        ci=CI(
            lower=float(lo / expected),
            upper=float(hi / expected),
            level=1 - alpha,
            method="exact Poisson",
        ),
        p_value=p,
        references=[REFERENCES["ulm1990"]],
    )


__all__ = [
    "IncidenceRate",
    "RateRatio",
    "StandardisedRatio",
    "StratifiedResult",
    "incidence_rate",
    "mantel_haenszel",
    "rate_ratio",
    "standardised_ratio",
]
