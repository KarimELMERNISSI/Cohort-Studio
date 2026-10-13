"""
2x2 contingency analysis with an explicit table orientation.

Audit finding S1: the previous engine took a sorted ``pd.crosstab`` positionally, so for outcomes
coded No/Yes, 0/1 or False/True the "event" column was actually the non-event and every risk
measure described the wrong comparison. :class:`TwoByTwo` therefore always carries the level names
it was built from, and :meth:`TwoByTwo.from_crosstab` counts cells with explicit masks.

Measures (all relative to the reference group): risk ratio (Katz log CI), odds ratio (Woolf log CI;
Haldane-Anscombe +0.5 when a cell is zero, flagged), risk difference (Newcombe hybrid score CI by
default, Wald optional), NNT with the interval derived from the RD interval (Altman 1998),
attributable fraction in the exposed, Pearson / Yates / Fisher tests with the primary test chosen by
the expected-count rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats
from scipy.stats import norm

from biostats._types import CI

Design = Literal["unknown", "cohort", "rct", "case-control", "cross-sectional"]

EVENT_TOKENS = {
    "1",
    "1.0",
    "true",
    "yes",
    "y",
    "positive",
    "pos",
    "present",
    "case",
    "cases",
    "event",
    "dead",
    "died",
    "death",
    "deceased",
    "abnormal",
    "disease",
}
REFERENCE_TOKENS = {
    "0",
    "0.0",
    "false",
    "no",
    "n",
    "control",
    "controls",
    "placebo",
    "unexposed",
    "absent",
    "reference",
    "ref",
    "none",
    "negative",
    "neg",
    "healthy",
    "normal",
    "standard",
}


def _token(value: Any) -> str:
    return str(value).strip().lower()


def detect_event_level(levels: list[Any]) -> Any | None:
    """The level that looks like the event (1/True/Yes/positive/case...), else ``None``."""
    matches = [lvl for lvl in levels if _token(lvl) in EVENT_TOKENS]
    return matches[0] if len(matches) == 1 else None


def detect_reference_level(levels: list[Any]) -> Any | None:
    """The level that looks like the reference/control (0/False/No/control/placebo...), else ``None``."""
    matches = [lvl for lvl in levels if _token(lvl) in REFERENCE_TOKENS]
    return matches[0] if len(matches) == 1 else None


def _sorted_levels(series: pd.Series) -> list[Any]:
    return sorted(series.dropna().unique().tolist(), key=lambda v: (str(type(v)), str(v)))


def _match(series: pd.Series, level: Any) -> pd.Series:
    mask = series == level
    if not mask.any():  # tolerate type coercion between widgets and data (e.g. "1" vs 1)
        mask = series.astype(str).str.strip() == str(level).strip()
    return mask


@dataclass(frozen=True)
class TwoByTwo:
    """Counts with explicit orientation: rows exposure (comparison vs reference), columns outcome."""

    a: int  # exposed & event
    b: int  # exposed & no event
    c: int  # reference & event
    d: int  # reference & no event
    exposure_level: str = "Exposed"
    reference_level: str = "Unexposed"
    event_level: str = "Event"
    nonevent_level: str = "No event"
    exposure_name: str = "Exposure"
    outcome_name: str = "Outcome"

    @classmethod
    def from_crosstab(
        cls,
        df: pd.DataFrame,
        exposure_col: str,
        outcome_col: str,
        *,
        exposure_level: Any | None = None,
        exposure_ref: Any | None = None,
        event_level: Any | None = None,
    ) -> TwoByTwo:
        """Build the table from two binary columns with explicit (or auto-detected) orientation.

        Auto-detection: the event is the level that looks like 1/True/Yes/positive; the reference
        group is the level that looks like 0/False/No/control/placebo; otherwise the first sorted
        level is the reference and the last sorted level the event. Both can be overridden.
        """
        data = df[[exposure_col, outcome_col]].dropna()
        exp_levels = _sorted_levels(data[exposure_col])
        out_levels = _sorted_levels(data[outcome_col])
        if len(exp_levels) != 2 or len(out_levels) != 2:
            raise ValueError(
                f"A 2x2 table needs exactly two levels; '{exposure_col}' has {len(exp_levels)} and "
                f"'{outcome_col}' has {len(out_levels)}."
            )

        def pick(levels: list[Any], wanted: Any | None, default: Any) -> Any:
            if wanted is None:
                return default
            for lvl in levels:
                if lvl == wanted or str(lvl) == str(wanted):
                    return lvl
            raise ValueError(f"Level {wanted!r} not found among {levels!r}")

        event = pick(out_levels, event_level, detect_event_level(out_levels) or out_levels[-1])
        nonevent = next(lvl for lvl in out_levels if lvl != event)
        if exposure_ref is None and exposure_level is not None:
            exposed = pick(exp_levels, exposure_level, None)
            reference = next(lvl for lvl in exp_levels if lvl != exposed)
        else:
            reference = pick(
                exp_levels, exposure_ref, detect_reference_level(exp_levels) or exp_levels[0]
            )
            exposed = next(lvl for lvl in exp_levels if lvl != reference)
            if exposure_level is not None and str(exposure_level) != str(exposed):
                raise ValueError("exposure_level and exposure_ref must be the two different levels")

        is_exposed = _match(data[exposure_col], exposed)
        is_event = _match(data[outcome_col], event)
        return cls(
            a=int((is_exposed & is_event).sum()),
            b=int((is_exposed & ~is_event).sum()),
            c=int((~is_exposed & is_event).sum()),
            d=int((~is_exposed & ~is_event).sum()),
            exposure_level=str(exposed),
            reference_level=str(reference),
            event_level=str(event),
            nonevent_level=str(nonevent),
            exposure_name=str(exposure_col),
            outcome_name=str(outcome_col),
        )

    @property
    def n_total(self) -> int:
        return self.a + self.b + self.c + self.d

    def as_frame(self) -> pd.DataFrame:
        """Labelled table with margins."""
        return pd.DataFrame(
            [
                [self.a, self.b, self.a + self.b],
                [self.c, self.d, self.c + self.d],
                [self.a + self.c, self.b + self.d, self.n_total],
            ],
            index=[
                f"{self.exposure_level} (comparison)",
                f"{self.reference_level} (reference)",
                "Total",
            ],
            columns=[f"{self.event_level} (event)", f"{self.nonevent_level}", "Total"],
        )


class TwoByTwoResult(BaseModel):
    """Measures of association for a 2x2 table (comparison group relative to the reference group)."""

    a: int
    b: int
    c: int
    d: int
    n_total: int
    exposure_name: str = "Exposure"
    outcome_name: str = "Outcome"
    exposure_level: str = "Exposed"
    reference_level: str = "Unexposed"
    event_level: str = "Event"
    nonevent_level: str = "No event"
    design: str = "unknown"
    haldane_applied: bool = False

    risk_exposed: float
    risk_reference: float
    prevalence_event: float

    odds_ratio: float
    odds_ratio_ci: CI
    relative_risk: float
    relative_risk_ci: CI
    risk_difference: float
    risk_difference_ci: CI
    nnt: float | None = None
    nnt_ci: CI | None = None
    nnt_label: str = ""
    attributable_fraction_exposed: float | None = None
    attributable_fraction_ci: CI | None = None

    chi2_statistic: float
    chi2_pvalue: float
    chi2_yates_statistic: float
    chi2_yates_pvalue: float
    fisher_exact_pvalue: float
    expected_min: float
    primary_test: str
    primary_pvalue: float

    guidance: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)

    def legacy_dict(self) -> dict[str, Any]:
        """Dictionary in the shape the pages consumed before the audit (plus the new keys)."""
        payload = self.model_dump()
        for key in (
            "odds_ratio_ci",
            "relative_risk_ci",
            "risk_difference_ci",
            "nnt_ci",
            "attributable_fraction_ci",
        ):
            if payload.get(key) is not None:
                payload[key] = {
                    "lower": payload[key]["lower"],
                    "upper": payload[key]["upper"],
                    "level": payload[key]["level"],
                    "method": payload[key]["method"],
                }
        return payload


def _newcombe_rd_ci(a: int, n1: int, c: int, n0: int, alpha: float) -> tuple[float, float, str]:
    """Newcombe (1998) hybrid score interval for the difference of two proportions (method 10)."""
    try:
        from statsmodels.stats.proportion import confint_proportions_2indep

        low, upp = confint_proportions_2indep(
            a, n1, c, n0, method="newcomb", compare="diff", alpha=alpha
        )
        return float(low), float(upp), "Newcombe hybrid score (1998)"
    except Exception:
        return _wald_rd_ci(a, n1, c, n0, alpha)


def _wald_rd_ci(a: int, n1: int, c: int, n0: int, alpha: float) -> tuple[float, float, str]:
    z = norm.ppf(1.0 - alpha / 2.0)
    p1 = a / n1 if n1 else 0.0
    p0 = c / n0 if n0 else 0.0
    se = math.sqrt((p1 * (1 - p1) / n1 if n1 else 0.0) + (p0 * (1 - p0) / n0 if n0 else 0.0))
    diff = p1 - p0
    return max(-1.0, diff - z * se), min(1.0, diff + z * se), "Wald"


def analyze_twobytwo(
    table: TwoByTwo,
    *,
    alpha: float = 0.05,
    design: Design = "unknown",
    rd_ci: Literal["newcombe", "wald"] = "newcombe",
) -> TwoByTwoResult:
    """Compute all measures of association for an explicitly oriented 2x2 table."""
    a, b, c, d = table.a, table.b, table.c, table.d
    n1, n0 = a + b, c + d
    n = table.n_total
    if n == 0 or n1 == 0 or n0 == 0:
        raise ValueError("Both exposure groups need at least one observation.")
    z = norm.ppf(1.0 - alpha / 2.0)
    level = 1.0 - alpha
    guidance: list[str] = []

    haldane = any(cell == 0 for cell in (a, b, c, d))
    a_, b_, c_, d_ = (x + 0.5 if haldane else float(x) for x in (a, b, c, d))
    if haldane:
        guidance.append(
            "Haldane-Anscombe correction (+0.5 to each cell) was applied to the odds ratio and risk ratio "
            "because at least one cell is zero; interpret those estimates with caution."
        )

    # Risks (raw) and risk difference
    p1 = a / n1
    p0 = c / n0
    rd = p1 - p0
    if rd_ci == "wald":
        rd_low, rd_upp, rd_method = _wald_rd_ci(a, n1, c, n0, alpha)
    else:
        rd_low, rd_upp, rd_method = _newcombe_rd_ci(a, n1, c, n0, alpha)

    # Odds ratio (Woolf) and risk ratio (Katz)
    odds_ratio = (a_ * d_) / (b_ * c_)
    se_log_or = math.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    or_low, or_upp = (
        math.exp(math.log(odds_ratio) - z * se_log_or),
        math.exp(math.log(odds_ratio) + z * se_log_or),
    )
    p1_, p0_ = a_ / (a_ + b_), c_ / (c_ + d_)
    relative_risk = p1_ / p0_
    se_log_rr = math.sqrt(b_ / (a_ * (a_ + b_)) + d_ / (c_ * (c_ + d_)))
    rr_low, rr_upp = (
        math.exp(math.log(relative_risk) - z * se_log_rr),
        math.exp(math.log(relative_risk) + z * se_log_rr),
    )

    # NNT / NNH from the RD interval (Altman 1998)
    nnt: float | None = None
    nnt_ci: CI | None = None
    nnt_label = ""
    if abs(rd) > 1e-12:
        nnt = 1.0 / abs(rd)
        nnt_label = (
            "NNH (more events in the comparison group)"
            if rd > 0
            else "NNT (fewer events in the comparison group)"
        )
        if rd_low > 0 or rd_upp < 0:
            bounds = sorted((1.0 / abs(rd_low), 1.0 / abs(rd_upp)))
            nnt_ci = CI(
                lower=bounds[0],
                upper=bounds[1],
                level=level,
                method=f"from the {rd_method} RD interval",
            )
        else:
            guidance.append(
                f"The risk-difference interval includes 0, so the number needed to treat is not bounded: it ranges "
                f"from NNTB {1.0 / abs(rd_low):.1f} through infinity to NNTH {1.0 / abs(rd_upp):.1f} "
                f"(Altman 1998)."
                if rd_low != 0 and rd_upp != 0
                else "The risk-difference interval includes 0, so the number needed to treat is not bounded (Altman 1998)."
            )

    # Attributable fraction in the exposed (and its interval from the RR interval)
    afe: float | None = None
    afe_ci: CI | None = None
    if relative_risk > 1.0:
        afe = (relative_risk - 1.0) / relative_risk
        if rr_low > 1.0:
            afe_ci = CI(
                lower=(rr_low - 1.0) / rr_low,
                upper=(rr_upp - 1.0) / rr_upp,
                level=level,
                method="from the Katz RR interval",
            )

    # Tests
    obs = np.array([[a, b], [c, d]])
    chi2_stat, chi2_p, _, expected = stats.chi2_contingency(obs, correction=False)
    yates_stat, yates_p, _, _ = stats.chi2_contingency(obs, correction=True)
    _, fisher_p = stats.fisher_exact(obs)
    expected_min = float(expected.min())
    if expected_min < 5.0:
        primary_test, primary_p = "Fisher's exact test (two-sided)", float(fisher_p)
        guidance.append(
            f"At least one expected count is below 5 (minimum {expected_min:.1f}); the p-value reported comes from "
            "Fisher's exact test rather than the chi-square test."
        )
    else:
        primary_test, primary_p = "Pearson chi-square test", float(chi2_p)

    prevalence = (a + c) / n
    if design == "case-control":
        guidance.append(
            "Case-control design: only the odds ratio is interpretable as a measure of association; risks, the risk "
            "ratio, the risk difference and NNT are not estimable because the outcome frequency was fixed by sampling."
        )
    else:
        if prevalence < 0.10:
            guidance.append(
                f"The outcome is uncommon in this sample ({prevalence * 100:.1f}% overall), so the odds ratio "
                f"({odds_ratio:.2f}) approximates the risk ratio ({relative_risk:.2f})."
            )
        else:
            guidance.append(
                f"The outcome is common ({prevalence * 100:.1f}% overall): the odds ratio ({odds_ratio:.2f}) is further from 1 "
                f"than the risk ratio ({relative_risk:.2f}). Report the risk ratio or risk difference for cohort, trial "
                "and cross-sectional designs."
            )
    if nnt is not None:
        ci_text = (
            f" (95% CI {nnt_ci.lower:.1f} to {nnt_ci.upper:.1f})" if nnt_ci is not None else ""
        )
        guidance.append(
            f"{nnt_label.split(' (')[0]} = {nnt:.1f}{ci_text}: about {nnt:.0f} people in the "
            f"'{table.exposure_level}' group per one additional event compared with '{table.reference_level}'."
        )

    return TwoByTwoResult(
        a=a,
        b=b,
        c=c,
        d=d,
        n_total=n,
        exposure_name=table.exposure_name,
        outcome_name=table.outcome_name,
        exposure_level=table.exposure_level,
        reference_level=table.reference_level,
        event_level=table.event_level,
        nonevent_level=table.nonevent_level,
        design=design,
        haldane_applied=haldane,
        risk_exposed=p1,
        risk_reference=p0,
        prevalence_event=prevalence,
        odds_ratio=odds_ratio,
        odds_ratio_ci=CI(lower=or_low, upper=or_upp, level=level, method="Woolf (log) interval"),
        relative_risk=relative_risk,
        relative_risk_ci=CI(lower=rr_low, upper=rr_upp, level=level, method="Katz (log) interval"),
        risk_difference=rd,
        risk_difference_ci=CI(lower=rd_low, upper=rd_upp, level=level, method=rd_method),
        nnt=nnt,
        nnt_ci=nnt_ci,
        nnt_label=nnt_label,
        attributable_fraction_exposed=afe,
        attributable_fraction_ci=afe_ci,
        chi2_statistic=float(chi2_stat),
        chi2_pvalue=float(chi2_p),
        chi2_yates_statistic=float(yates_stat),
        chi2_yates_pvalue=float(yates_p),
        fisher_exact_pvalue=float(fisher_p),
        expected_min=expected_min,
        primary_test=primary_test,
        primary_pvalue=primary_p,
        guidance=guidance,
        references=[
            "Katz D et al. (1978) Biometrics 34:469-474 (risk ratio interval)",
            "Woolf B (1955) Ann Hum Genet 19:251-253 (odds ratio interval)",
            "Newcombe RG (1998) Stat Med 17:873-890 (risk difference interval, method 10)",
            "Altman DG (1998) BMJ 317:1309-1312 (confidence intervals for NNT)",
        ],
    )


__all__ = [
    "Design",
    "TwoByTwo",
    "TwoByTwoResult",
    "analyze_twobytwo",
    "detect_event_level",
    "detect_reference_level",
]
