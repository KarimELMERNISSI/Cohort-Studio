"""Incidence rates, rate ratios, Mantel-Haenszel stratified measures and standardised ratios (audit S11)."""

from __future__ import annotations

import math
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
from biostats.study_designs import BEFORE, DIFFERENCE, RATIO, StudyDesign, design_named

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
    "robins1986",
    "Robins J, Breslow N, Greenland S (1986). Estimators of the Mantel-Haenszel variance consistent in both sparse data and large-strata limiting models. Biometrics 42:311-323.",
)
REFERENCES.setdefault(
    "sato1989",
    "Sato T (1989). On the variance estimator for the Mantel-Haenszel risk difference. Biometrics 45:1323-1324.",
)
REFERENCES.setdefault(
    "rothman2008",
    "Rothman KJ, Greenland S, Lash TL (2008). Modern Epidemiology, 3rd edition. Lippincott Williams & Wilkins.",
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
    # the strata the two tests read: those that hold both groups and both outcomes (None: a result kept before
    # this was counted)
    n_tested: int | None = None
    n: int | None = None  # the rows of the tables: those with the three values
    compared: str = ""  # the group whose odds and risk stand above the line
    reference: str = ""  # the group they are compared with
    event: str = ""  # the level of the outcome counted as the event
    # how the participants were sampled (biostats.study_designs), and the measures it supports: "" and () for a
    # result kept before the design was asked, read as it was then (every measure)
    design: str = ""
    measures: tuple[str, ...] = ()
    rd_mh: float = float("nan")  # the risk (or prevalence) difference, with Sato's variance
    rd_mh_ci: CI | None = None
    # the test that the strata's risk (or prevalence) ratios are alike (Woolf), and the strata it reads
    woolf_statistic: float = float("nan")
    woolf_p: float = float("nan")
    n_woolf: int | None = None


def _informative(table: np.ndarray) -> bool:
    """A stratum that carries information on the odds ratio: it holds both groups and both outcomes. Any other adds
    nothing to the Mantel-Haenszel odds ratio (its two products are 0), has no variance in the test of the
    association, and no odds ratio of its own to compare with the others."""
    return bool(table.sum(axis=1).min() > 0 and table.sum(axis=0).min() > 0)


def read_by_the_tests(informative: int) -> str:
    """What the two tests read of the strata that hold both groups and both outcomes, by their number: the test
    that the strata are alike needs two of them ("the two tests read the other 1" was said where it read
    none)."""
    if informative >= 2:
        return f"the two tests read the other {informative}"
    if informative == 1:
        return (
            "the test of the association reads the other one; whether the strata differ needs two"
        )
    return "no stratum is left for the two tests"


def _breslow_day(tables: Sequence[np.ndarray], odds_ratio: float) -> tuple[float, float]:
    """
    The Breslow-Day test that the odds ratios of the strata are alike (Breslow & Day 1980; no Tarone correction):
    in each stratum, the events among those compared that the combined odds ratio expects with the stratum's
    margins fixed, against those seen. Statistic and p-value; without a value for a combined ratio of 0 or
    without one.

    Computed here: the library's own divides by 1 - odds ratio. At a combined ratio of exactly 1 (opposite
    strata, a balanced table) it had no value, where the expected count is the one of chance alone and the
    statistic can be 40; one rounding away from 1 it was wrong (28.1 for 5.57, and a negative one).
    """
    nan = float("nan")
    if not math.isfinite(odds_ratio) or odds_ratio <= 0:
        return nan, nan
    total = 0.0
    for table in tables:
        a, b, c, d = (float(cell) for cell in np.asarray(table, dtype=float).ravel())
        n1, n0, m1 = a + b, c + d, a + c
        # the expected count A solves A (n0 - m1 + A) = odds ratio x (n1 - A)(m1 - A)
        qa = 1 - odds_ratio
        qb = (n0 - m1) + odds_ratio * (n1 + m1)
        qc = -odds_ratio * n1 * m1
        low, high = max(0.0, m1 - n0), min(n1, m1)
        if abs(qa) < 1e-9:  # a ratio of 1: the count expected by chance alone, n1 m1 / n
            expected = -qc / qb
        else:
            root = math.sqrt(max(qb * qb - 4 * qa * qc, 0.0))
            q = -0.5 * (
                qb + math.copysign(root, qb)
            )  # the stable form: no two near-equal numbers subtracted
            found = [
                x for x in ((q / qa), (qc / q if q else nan)) if low - 1e-7 <= x <= high + 1e-7
            ]
            if not found:
                return nan, nan
            expected = found[0]
        cells = (expected, n1 - expected, m1 - expected, n0 - m1 + expected)
        if min(cells) <= 0:
            return nan, nan
        total += (a - expected) ** 2 * sum(1 / cell for cell in cells)
    return total, float(stats.chi2.sf(total, len(tables) - 1))


def _differ(crude: float, adjusted: float) -> bool:
    """The crude and adjusted ratios are known and differ by more than 10%."""
    known = np.isfinite(crude) and np.isfinite(adjusted) and crude > 0 and adjusted > 0
    return bool(known and abs(np.log(crude) - np.log(adjusted)) > np.log(1.1))


def _woolf(tables: Sequence[np.ndarray]) -> tuple[float, float, int]:
    """
    The Woolf test that the strata's risk (or prevalence) ratios are alike (Rothman, Greenland & Lash 2008, ch. 15):
    the inverse-variance weighted sum of squares of the strata's log ratios about their weighted mean, the variance
    of each Katz's, b/(a n1) + d/(c n0). It reads the strata with an event in both groups and a variance above 0
    (not everyone with the event in both); statistic, p-value on the strata read - 1 degrees of freedom, and the
    strata read. Breslow-Day reads the odds ratios: with a common outcome they differ between strata whose risk
    ratios do not, so each test stands beside its own measure.
    """
    nan = float("nan")
    logs, weights = [], []
    for table in tables:
        a, b, c, d = (float(cell) for cell in np.asarray(table, dtype=float).ravel())
        if a <= 0 or c <= 0:
            continue
        variance = b / (a * (a + b)) + d / (c * (c + d))
        if variance <= 0:
            continue
        logs.append(math.log((a / (a + b)) / (c / (c + d))))
        weights.append(1.0 / variance)
    if len(logs) < 2:
        return nan, nan, len(logs)
    log_ratios, weight = np.asarray(logs), np.asarray(weights)
    mean = float(np.sum(weight * log_ratios) / np.sum(weight))
    statistic = float(np.sum(weight * (log_ratios - mean) ** 2))
    if statistic < 1e-12:  # strata of one ratio: 0, not the rounding's 1.2e-29
        statistic = 0.0
    return statistic, float(stats.chi2.sf(statistic, len(logs) - 1)), len(logs)


def _rd_mh(tables: Sequence[np.ndarray], alpha: float) -> tuple[float, CI | None]:
    """The Mantel-Haenszel risk (or prevalence) difference, with Sato's variance (Sato 1989): consistent in sparse
    strata as in large ones, where the Greenland-Robins variance was half the variance seen in 200 strata of
    2 + 2. None for its interval when no stratum holds both groups, and when the variance is 0: every stratum then
    lies at the edge, a difference of -100 or +100 points (the interval read "-1 to 1", from a standard error
    without a value)."""
    num = den = p_sum = q_sum = 0.0
    for table in tables:
        a, b, c, d = (float(cell) for cell in np.asarray(table, dtype=float).ravel())
        n1, n0 = a + b, c + d
        n = n1 + n0
        if n1 <= 0 or n0 <= 0:
            continue
        num += (a * n0 - c * n1) / n
        den += n1 * n0 / n
        p_sum += (n1**2 * c - n0**2 * a + n1 * n0 * (n0 - n1) / 2) / n**2
        q_sum += (a * (n0 - c) + c * (n1 - a)) / (2 * n)
    if den <= 0:
        return float("nan"), None
    rd = num / den
    variance = (rd * p_sum + q_sum) / den**2
    if not variance > 1e-15:  # 0 up to rounding: never below it
        return float(rd), None
    se = math.sqrt(variance)
    z = stats.norm.ppf(1 - alpha / 2)
    return float(rd), CI(
        lower=float(max(-1.0, rd - z * se)),
        upper=float(min(1.0, rd + z * se)),
        level=1 - alpha,
        method="Sato",
    )


def stratum_differences(per_stratum: pd.DataFrame) -> np.ndarray:
    """Each stratum's own risk (or prevalence) difference, in the strata that hold both groups: those the
    Mantel-Haenszel difference combines. Read beside it, since one combined difference sums the strata up only
    when theirs are alike, and with a common ratio they change with each stratum's risk."""
    a, n1, c, n0 = (
        pd.to_numeric(per_stratum[name], errors="coerce").to_numpy(dtype=float)
        for name in ("exposed_events", "exposed_total", "reference_events", "reference_total")
    )
    both = (n1 > 0) & (n0 > 0)
    return a[both] / n1[both] - c[both] / n0[both]


def reference_risks(per_stratum: pd.DataFrame) -> tuple[float, float, float]:
    """
    Whether the reference group's risk differs between the strata the two tests of homogeneity read (those that hold
    both groups and both outcomes): the p-value of a chi-square test of its events over those strata (Pearson,
    without continuity correction), and its lowest and highest risk. The tests part by scale only where it differs:
    at one baseline risk a common risk ratio is a common odds ratio. p without a value for fewer than two strata, or
    when no reference participant, or every one, had the event.
    """
    a, n1, c, n0 = (
        pd.to_numeric(per_stratum[name], errors="coerce").to_numpy(dtype=float)
        for name in ("exposed_events", "exposed_total", "reference_events", "reference_total")
    )
    tested = (n1 > 0) & (n0 > 0) & (a + c > 0) & ((n1 - a) + (n0 - c) > 0)
    events, others = c[tested], (n0 - c)[tested]
    nan = float("nan")
    if not tested.any():
        return nan, nan, nan
    risks = events / n0[tested]
    if tested.sum() < 2 or events.sum() == 0 or others.sum() == 0:
        return nan, float(risks.min()), float(risks.max())
    p = float(stats.chi2_contingency(np.column_stack([events, others]), correction=False)[1])
    return p, float(risks.min()), float(risks.max())


def _change_by_design(
    design: StudyDesign, crude_or: float, or_mh: float, crude_rr: float, rr_mh: float
) -> list[str]:
    """The 10% change-in-estimate hint as the design reads it: on its ratio first (a cohort, a cross-sectional
    sample), as chance in a randomised trial, on the odds ratio alone otherwise."""
    if design.id == BEFORE:
        return _change_in_estimate(crude_or, or_mh, crude_rr, rr_mh)
    if design.supports(RATIO):
        noun = "prevalence ratios" if design.shares == "prevalence" else "risk ratios"
        if design.randomised and _differ(crude_rr, rr_mh):
            return [
                "the adjusted and unadjusted risk ratios differ by more than 10%: in a randomised trial the "
                "groups differ in the stratum variable by chance only, so this is no confounding; report the "
                "analysis the protocol named (adjusting for the factors the randomisation was stratified by is "
                "the usual one)"
            ]
        return [
            line.replace("risk ratios", noun)
            for line in _change_in_estimate(crude_or, or_mh, crude_rr, rr_mh)
        ]
    if not _differ(crude_or, or_mh):
        return []
    if design.id == "case-control":
        return [
            "the crude and adjusted odds ratios differ by more than 10%: the stratum variable may be a "
            "confounder; if the controls were matched on it, the crude odds ratio is biased by the matching: "
            "report the adjusted one"
        ]
    return [
        "the crude and adjusted odds ratios differ by more than 10%: the stratum variable may be a confounder, "
        "although odds ratios can also differ without confounding (they are non-collapsible)"
    ]


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
    design: str = BEFORE,
) -> StratifiedResult:
    """
    Odds and risk ratios combined over strata (Mantel-Haenszel), the Cochran-Mantel-Haenszel test of the association
    and the Breslow-Day test that the strata's odds ratios are alike.

    ``design`` is how the participants were sampled (:mod:`biostats.study_designs`): only the measures it supports
    are computed and returned (``measures``), so that no reader can show the risk ratio of a case-control sample.
    The odds ratio and the test always; the risk (prevalence) ratio, its Woolf test of homogeneity and the risk
    (prevalence) difference with Sato's variance for a cohort, a trial and a cross-sectional sample. Without a
    design (a caller of before), every measure, as then.

    The two tests read the strata that hold both groups and both outcomes. With every stratum, one in which every
    participant had the event gave the Breslow-Day test a degree of freedom too many (p = 0.48 for 0.33), and one of
    a single row left both tests without a value. The risk ratio reads every stratum: one in which everyone had
    the event says that the two risks are equal there.
    """
    sampling = design_named(design)
    with_ratio = sampling.supports(RATIO)
    frame = df[[exposure_col, outcome_col, stratum_col]].dropna().astype(str)
    exp_levels = sorted(frame[exposure_col].unique())
    out_levels = sorted(frame[outcome_col].unique())
    if len(exp_levels) != 2 or len(out_levels) != 2:
        raise ValueError("exposure and outcome must both be binary")
    exposure_level = str(exposure_level) if exposure_level is not None else exp_levels[1]
    event_level = str(event_level) if event_level is not None else out_levels[1]
    tables, rows, warnings = [], [], []
    small = 0  # strata with fewer than 5 participants in a group
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
                # without a value when both products are 0; without an upper bound when only the lower one is
                "odds_ratio": (a * d) / (b * c)
                if b * c > 0
                else (float("inf") if a * d > 0 else float("nan")),
                # as the odds ratio: without an upper bound when only the reference group has no event ("not
                # estimable" was said of a stratum of 10 events in 20 against 0 in 20)
                "risk_ratio": (
                    (a / (a + b)) / (c / (c + d))
                    if c > 0
                    else (float("inf") if a > 0 else float("nan"))
                )
                if (a + b) > 0 and (c + d) > 0
                else float("nan"),
            }
        )
        if (
            not with_ratio
        ):  # not computed where the design does not support it: never in a table or a record
            rows[-1].pop("risk_ratio")
        # among the strata the tests read: one that holds a single group is counted with those that carry no
        # information, not twice (the plan said "1 of 7" and the result "3 of 7")
        small += bool(_informative(tbl) and min(a + b, c + d) < 5)
    if small:  # counted, never named: a stratum's value is the data's (a site, a clinic), copied into the dossier
        warnings.append(
            f"{small} of {len(tables)} strata {'has' if small == 1 else 'have'} fewer than 5 participants in a group"
        )
    if len(tables) < 2:
        raise ValueError("at least two strata are required")
    tested = [table for table in tables if _informative(table)]
    if len(tested) < len(tables):  # counted, never named, as the small strata
        left = len(tables) - len(tested)
        warnings.append(
            f"{left} of {len(tables)} strata hold{'s' if left == 1 else ''} one group only or one outcome only: "
            f"{'it adds' if left == 1 else 'they add'} nothing to the odds ratio, and "
            + read_by_the_tests(len(tested))
        )
    st = StratifiedTable(tested or tables)
    or_ci = st.oddsratio_pooled_confint(alpha=alpha)
    nan_ci = CI(lower=float("nan"), upper=float("nan"), level=1 - alpha, method="")
    rr, rr_ci = _rr_mh(tables, alpha) if with_ratio else (float("nan"), nan_ci)
    rd, rd_ci = _rd_mh(tables, alpha) if sampling.supports(DIFFERENCE) else (float("nan"), None)
    woolf_statistic, woolf_p, n_woolf = (
        _woolf(tables) if with_ratio else (float("nan"), float("nan"), None)
    )
    nan = float("nan")
    mh_statistic, mh_p = nan, nan
    if tested:
        mh = st.test_null_odds(correction=True)
        mh_statistic, mh_p = float(mh.statistic), float(mh.pvalue)
    bd_statistic, bd_p = nan, nan
    if len(tested) >= 2:  # one stratum has no other to differ from
        bd_statistic, bd_p = _breslow_day(tested, float(st.oddsratio_pooled))
    pooled = np.sum(tables, axis=0)
    crude = (
        (pooled[0, 0] * pooled[1, 1]) / (pooled[0, 1] * pooled[1, 0])
        if pooled[0, 1] * pooled[1, 0] > 0
        else float("nan")
    )
    a, b, c, d = pooled.ravel()
    crude_rr = (
        (a / (a + b)) / (c / (c + d))
        if with_ratio and a + b > 0 and c + d > 0 and c > 0
        else float("nan")
    )
    warnings.extend(_change_by_design(sampling, crude, float(st.oddsratio_pooled), crude_rr, rr))
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
        breslow_day_statistic=bd_statistic,
        breslow_day_p=bd_p,
        mh_test_statistic=mh_statistic,
        mh_test_p=mh_p,
        per_stratum=pd.DataFrame(rows),
        crude_rr=float(crude_rr),
        references=[
            REFERENCES["mantel1959"],
            REFERENCES["robins1986"],
            REFERENCES["breslow1980"],
            *([REFERENCES["greenland1985"], REFERENCES["rothman2008"]] if with_ratio else []),
            *([REFERENCES["sato1989"]] if sampling.supports(DIFFERENCE) else []),
        ],
        warnings=warnings,
        n_tested=len(tested),
        n=int(np.sum(tables)),
        compared=exposure_level,
        reference=next((level for level in exp_levels if level != exposure_level), ""),
        event=event_level,
        design=sampling.id,
        measures=sampling.measures if sampling.id != BEFORE else (),
        rd_mh=rd,
        rd_mh_ci=rd_ci,
        woolf_statistic=woolf_statistic,
        woolf_p=woolf_p,
        n_woolf=n_woolf,
    )


@dataclass
class RatesResult:
    """
    One rate, or the rates of two groups and their ratio: ``rates`` in order (the group compared, then the reference
    group); ``ratio`` None for one rate, and for two groups without an event in either (two rates of 0 have no
    ratio: ``warnings`` says so). ``unit`` is the unit of the person-time (a stored id): a rate without it is no
    number a reader can use, so the result carries it, as each rate carries the multiplier it is shown with.
    """

    rates: tuple[IncidenceRate, ...]
    ratio: RateRatio | None = None
    warnings: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    unit: str = "person-years"


def has_ratio(events: int, reference_events: int | None) -> bool:
    """Whether two groups have a ratio of their rates: a reference group, and an event in one of the two (two rates
    of 0 have none). One rule, read where the rates are computed and where a calculation is titled."""
    return reference_events is not None and events + reference_events > 0


def rates(
    events: int,
    person_time: float,
    reference_events: int | None = None,
    reference_person_time: float | None = None,
    *,
    per: float = 1000.0,
    alpha: float = 0.05,
    unit: str = "person-years",
) -> RatesResult:
    """One rate with its exact interval, or the rates of two groups and their ratio (:func:`rate_ratio`). The
    reference group is given whole (its events and its person-time) or not at all."""
    if (reference_events is None) != (reference_person_time is None):
        raise ValueError("the reference group needs its events and its person-time")
    first = incidence_rate(events, person_time, per=per, alpha=alpha)
    if reference_events is None or reference_person_time is None:
        return RatesResult((first,), references=list(first.references), unit=unit)
    second = incidence_rate(reference_events, reference_person_time, per=per, alpha=alpha)
    if not has_ratio(first.events, second.events):
        return RatesResult(
            (first, second),
            warnings=["No event in either group: two rates of 0 have no ratio."],
            references=list(first.references),
            unit=unit,
        )
    ratio = rate_ratio(
        first.events, first.person_time, second.events, second.person_time, per=per, alpha=alpha
    )
    return RatesResult((first, second), ratio, references=list(ratio.references), unit=unit)


@dataclass
class StandardisedRatio:
    observed: int
    expected: float
    ratio: float
    ci: CI
    p_value: float
    references: list[str] = field(default_factory=list)
    # what the events are (a stored id, "deaths" or "cases"): it names the ratio, mortality or incidence
    events_are: str = "deaths"
    warnings: list[str] = field(default_factory=list)


def standardised_ratio(
    observed: int, expected: float, *, alpha: float = 0.05, events_are: str = "deaths"
) -> StandardisedRatio:
    """
    SMR / SIR = observed / expected with the exact Poisson interval on the observed count (Garwood) and the test
    that agrees with it: twice the smaller Poisson tail, which excludes 1 exactly when the interval does. R's
    ``poisson.test(observed, T = expected)`` gives the same interval and a minimum-likelihood p-value, which can
    differ (25 against 18.4: 0.165 here, 0.128 there).
    """
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
        references=[REFERENCES["ulm1990"], REFERENCES["garwood1936"]],
        events_are=events_are,
        warnings=(
            ["Fewer than 1 event expected: the ratio rests on almost nothing."]
            if expected < 1
            else []
        ),
    )


__all__ = [
    "IncidenceRate",
    "RateRatio",
    "RatesResult",
    "StandardisedRatio",
    "StratifiedResult",
    "has_ratio",
    "incidence_rate",
    "mantel_haenszel",
    "rate_ratio",
    "rates",
    "standardised_ratio",
]
