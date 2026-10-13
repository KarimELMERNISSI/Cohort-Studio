"""Incidence rates, rate ratios, Mantel-Haenszel stratified measures and standardised ratios (audit S11)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

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
    references: list[str] = field(default_factory=list)


def rate_ratio(
    events_exposed: int,
    time_exposed: float,
    events_reference: int,
    time_reference: float,
    *,
    per: float = 1000.0,
    alpha: float = 0.05,
) -> RateRatio:
    """Incidence rate ratio with a Poisson-based CI and exact conditional test."""
    exp_ = incidence_rate(events_exposed, time_exposed, per=per, alpha=alpha)
    ref = incidence_rate(events_reference, time_reference, per=per, alpha=alpha)
    lo, hi = confint_poisson_2indep(
        events_exposed,
        time_exposed,
        events_reference,
        time_reference,
        method="score",
        compare="ratio",
        alpha=alpha,
    )
    test = test_poisson_2indep(
        events_exposed,
        time_exposed,
        events_reference,
        time_reference,
        method="exact-cond",
        compare="ratio",
    )
    return RateRatio(
        exposed=exp_,
        reference=ref,
        ratio=exp_.rate / ref.rate if ref.rate > 0 else float("inf"),
        ci=CI(lower=float(lo), upper=float(hi), level=1 - alpha, method="score"),
        p_value=float(test.pvalue),
        rate_difference=exp_.rate - ref.rate,
        references=[REFERENCES["garwood1936"]],
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
    references: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


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
    if (
        np.isfinite(crude)
        and np.isfinite(st.oddsratio_pooled)
        and abs(np.log(crude) - np.log(st.oddsratio_pooled)) > np.log(1.1)
    ):
        warnings.append(
            "the crude and adjusted odds ratios differ by more than 10%: the stratum variable is a confounder"
        )
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
