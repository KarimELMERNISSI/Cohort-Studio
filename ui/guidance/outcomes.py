"""
Guidance for Analyze > Survival & regression, a page organised by the outcome: its outcome types, help texts, the
plan shown before a model runs, the terms of the comparison a result is read by, and the diagnostics of a fitted
model.

No Streamlit call: the page renders these, the tests check them. The plan reads the rows as the models do
(biostats.survival.event_indicator for the event, biostats.design.design_matrix for the terms), so it says what Run
will do and on how many rows, and what would make a model's terms unstable or impossible to estimate: a variable left
with one value, one value written several ways, levels held by few rows or without events, terms that are exact
combinations of others, strong collinearity. It advises, and never decides: what runs is what the user chose.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg

from biostats.design import design_matrix
from biostats.regression import APPARENT_AUC, RegressionResult
from biostats.survival import CoxResult, event_indicator
from core.registry import as_code
from ui.guidance.analysis import Advice

NONE = "(none)"  # no variable chosen
TIME_TO_EVENT, BINARY, COUNT = "time_to_event", "binary", "count"
FAMILIES = {"poisson": "Poisson", "negative_binomial": "Negative binomial"}
EPV_MIN = 10  # events per estimated term below which ratios may be unstable (Peduzzi rule of thumb)
LEVEL_MIN = 10  # rows below which a level's ratio is unstable
# variance inflation above which terms are too collinear to be read apart (rule of thumb); a categorical variable's
# generalised VIF is put on the same scale, GVIF^(1/df) (Fox & Monette 1992)
VIF_MAX = 10.0


@dataclass(frozen=True)
class OutcomeType:
    id: str  # kept in the page's state
    label: str
    gives: str  # what the page computes for it
    terms: tuple[str, ...]  # its words in the glossary (ui.guidance.glossary)
    simpler: str = ""  # where an unadjusted comparison is made instead, if anywhere


OUTCOME_TYPES: tuple[OutcomeType, ...] = (
    OutcomeType(
        TIME_TO_EVENT,
        "Time to event",
        "Kaplan-Meier curves with their medians and the log-rank test; hazard ratios from a Cox model.",
        (
            "Kaplan-Meier curve",
            "Censoring",
            "Log-rank test",
            "Hazard ratio (HR)",
            "Proportional hazards",
        ),
    ),
    OutcomeType(
        BINARY,
        "Binary (yes/no)",
        "Odds ratios from a logistic model, adjusted for the variables you choose.",
        ("Odds ratio (OR)", "Confounder", "Events per parameter (EPV)"),
        "Risk ratio, risk difference and number needed to treat, without adjustment",
    ),
    OutcomeType(
        COUNT,
        "Count or rate",
        "Incidence-rate ratios from a Poisson or negative binomial model, with person-time as offset.",
        ("Incidence rate", "Incidence rate ratio (IRR)", "Over-dispersion"),
        "Crude rates and the exact rate ratio, without adjustment",
    ),
)
OUTCOME = {o.id: o for o in OUTCOME_TYPES}

HELP: dict[str, str] = {
    "outcome": "What the analysis explains: the time until an event (some participants followed without it), "
    "whether an event happened, or how many events happened.",
    "guidance": "Shows the plan before each run, checked on your data. Turn it off once you know the page.",
    "time": "Time from the start of follow-up (time zero) to the event or to censoring, in one unit (days, months "
    "or years). Rows with a negative time are left out.",
    "event": "1 / Yes / True = the event happened; 0 / No / False = censored (followed without the event). Any "
    "other code is refused rather than guessed.",
    "analyses": "Kaplan-Meier: the survival curves and their medians, and the log-rank test when there are groups. "
    "Cox model: hazard ratios, adjusted for the variables you choose.",
    "compare_time": "The exposure or group you compare (treatment arm, risk factor). A grouping variable gives a "
    "curve per level and the log-rank test, and in the Cox model a hazard ratio per level against the first; a "
    "numeric variable gives a hazard ratio per unit.",
    "compare": "The exposure or group you compare (treatment arm, risk factor): its ratio is the result shown "
    "first, per level against the first level, or per unit of a numeric variable. Leave it empty to read every "
    "term of the model alike.",
    "adjust": "Variables measured at or before the start of follow-up that may confound the comparison (age, sex, "
    "severity), never its consequences (mediators). Their own ratios are adjustment terms, not effects to "
    "interpret. Numeric variables enter linearly; categorical ones by level, the first being the reference.",
    "binary_outcome": "A variable with two levels (yes/no, dead/alive).",
    "event_level": "The level counted as the event: the odds ratios compare the odds of this level.",
    "count": "Events per participant or unit: whole numbers, 0 or more.",
    "person_time": "Each row's time at risk. It enters as log(person-time), so that the ratios compare event rates; "
    "without it they compare mean counts.",
    "family": "Poisson assumes that the counts vary as much as their mean. When they vary more (over-dispersion, "
    "shown in Checks), its intervals are too narrow: the negative binomial model allows the extra variance.",
    "read_km": "Each step down is an event; vertical ticks would mark censoring. The band is the pointwise 95% "
    "interval. The log-rank test weights every time equally and loses power when curves cross; report medians and "
    "the hazard ratio from the Cox model with it. With imputed data the curves are pooled on the log(-log S) scale "
    "at every time any curve steps, and the median is read from the pooled curve.",
    "read_cox": "A hazard ratio compares the event rate at any time between two levels (or per unit), the other "
    "variables of the model held fixed: below 1, fewer events. It assumes the ratio stays the same over follow-up "
    "(proportional hazards, tested in Checks). The ratios of the variables you adjust for are not effects to "
    "interpret: each is adjusted for the others, not for its own confounders (the 'Table 2 fallacy').",
    "read_logistic": "An odds ratio compares the odds of the event between two levels (or per unit), the other "
    "variables of the model held fixed. It is close to the risk ratio only when the event is rare (below about "
    "10%); otherwise it lies further from 1. The ratios of the variables you adjust for are not effects to "
    "interpret (the 'Table 2 fallacy').",
    "read_count": "An incidence-rate ratio compares event rates (events per person-time) between two levels (or per "
    "unit), the other variables of the model held fixed; without person-time it compares mean counts. The ratios "
    "of the variables you adjust for are not effects to interpret (the 'Table 2 fallacy').",
}


# ----------------------------------------------------------------------------- the plan


@dataclass(frozen=True)
class PlanStep:
    text: str  # what Run will do, in words (names as code)
    used: int  # rows it will use
    events: int | None = None
    terms: int | None = (
        None  # terms the model estimates (a factor counts its levels but the reference)
    )


@dataclass(frozen=True)
class ModelPlan:
    """What Run will do, checked against the data before anything is computed."""

    rows: int  # rows of the analysis population
    steps: tuple[PlanStep, ...]
    advice: tuple[Advice, ...] = ()

    @property
    def status(self) -> str:
        levels = {a.level for a in self.advice}
        return "violated" if "violated" in levels else "caution" if "caution" in levels else "ok"


def chosen(name: str | None) -> str | None:
    """A variable picked in a list that starts with NONE, or None."""
    return None if not name or name == NONE else name


def covariates_of(compare: str | None, adjust: Sequence[str]) -> list[str]:
    """The model's covariates: the comparison first, then the adjustment (a variable named twice, once)."""
    first = [compare] if chosen(compare) else []
    return first + [c for c in adjust if c not in first]


def listing(names: Sequence[Any]) -> str:
    """Names as code, joined in a sentence."""
    shown = [as_code(n) for n in names]
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]


def _model_words(compare: str | None, adjust: Sequence[str]) -> str:
    """'comparing `arm`, adjusted for `age`' / 'comparing `arm`, unadjusted' / 'with `age` and `sex`'."""
    compare = chosen(compare)
    others = [c for c in adjust if c != compare]
    if compare is None:
        return f"with {listing(others)}"
    return f"comparing {as_code(compare)}, " + (
        f"adjusted for {listing(others)}" if others else "unadjusted"
    )


def _terms(frame: pd.DataFrame, covariates: Sequence[str]) -> tuple[int | None, str | None]:
    """The number of terms the model estimates on ``frame`` (as the model builds them), or why it cannot."""
    try:
        x, _ = design_matrix(frame, list(covariates), intercept=False)
    except ValueError as exc:
        return None, str(exc)
    return int(x.shape[1]), None


def _too_few(events: int, terms: int | None, what: str, ratios: str) -> Advice | None:
    if not terms or events >= EPV_MIN * terms:
        return None
    return Advice(
        "caution",
        f"{events:,} {what} for {terms} term{'s' if terms > 1 else ''} ({events / terms:.1f} per term): {ratios} "
        f"may be unstable below about {EPV_MIN} per term (Peduzzi rule of thumb). Fewer adjustment variables, or "
        "fewer levels, help.",
    )


def _said(text: object) -> str:
    """A message that can quote the data (a model's error, an event code) as text: no Markdown drawn from it."""
    from ui.components.text import plain

    return plain(text, limit=400)


def _some(levels: Sequence[Any], limit: int = 4) -> str:
    """
    At most ``limit`` levels as code, then how many more; a hidden space shows as the lists show it ("F␣", which
    code would read "F", as the level "F").
    """
    from ui.components.levels import show_level  # the lists' own reading of a level

    shown = ", ".join(as_code(show_level(level)) for level in levels[:limit])
    return shown + (f" and {len(levels) - limit} more" if len(levels) > limit else "")


def _spellings(col: str, values: pd.Series) -> Advice | None:
    """One value written several ways (spaces, capitals): each is a level of its own, as the lists show them."""
    from ui.components.levels import alike_levels, cleaning_formula

    alike = alike_levels(sorted(values.dropna().astype(str).unique()))
    if not alike:
        return None
    shown = "; ".join(_some(found, 3) for found in alike[:3])
    return Advice(
        "caution",
        f"{as_code(col)} holds one value written several ways ({shown}): each is a group of its own. Clean it "
        f"first, as a new variable {as_code(cleaning_formula(col, alike))} (Enrich > Create new variables).",
    )


def _collinearity(frame: pd.DataFrame, covariates: Sequence[str], ratios: str) -> list[Advice]:
    """
    Terms that are an exact combination of the others (a pivoted QR of the standardised design), else the variables
    whose variance inflation exceeds VIF_MAX, each with the variable it is most correlated with.
    """
    owner: dict[str, str] = {}  # design column -> its variable
    parts = []
    for col in covariates:
        x, _ = design_matrix(frame, [col], intercept=False)
        owner.update(dict.fromkeys(x.columns, col))
        parts.append(x)
    x = pd.concat(parts, axis=1)
    values = x.to_numpy(dtype=float)
    spread = values.std(axis=0)
    names = [n for n, s in zip(x.columns, spread, strict=True) if s > 0]
    if len({owner[n] for n in names}) < 2 or len(frame) <= len(names):
        return []
    z = (values[:, spread > 0] - values[:, spread > 0].mean(axis=0)) / spread[spread > 0]
    _, r, pivot = linalg.qr(z, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    rank = int((diag > diag.max() * max(z.shape) * np.finfo(float).eps * 100).sum())
    if rank < len(names):
        exact = list(dict.fromkeys(owner[names[i]] for i in pivot[rank:]))
        return [
            Advice(
                "violated",
                f"{listing(exact)} {'is an exact combination' if len(exact) == 1 else 'are exact combinations'} of "
                "the other variables: the model cannot separate them. Leave one out.",
            )
        ]
    corr = np.corrcoef(z, rowvar=False)
    whole = np.linalg.det(corr)
    advice: list[Advice] = []
    seen: set[frozenset[str]] = set()
    for var in dict.fromkeys(owner[n] for n in names):
        own = [i for i, n in enumerate(names) if owner[n] == var]
        rest = [i for i in range(len(names)) if i not in own]
        gvif = (
            np.linalg.det(corr[np.ix_(own, own)]) * np.linalg.det(corr[np.ix_(rest, rest)]) / whole
        )
        vif = float(
            gvif ** (1 / len(own))
        )  # GVIF^(1/(2 df)), squared: the scale of a single term's VIF
        partner = owner[names[max(rest, key=lambda j: max(abs(corr[i, j]) for i in own))]]
        pair = frozenset((var, partner))
        if vif > VIF_MAX and pair not in seen:
            seen.add(pair)
            advice.append(
                Advice(
                    "caution",
                    f"{as_code(var)} and {as_code(partner)} are strongly collinear (variance inflation {vif:.0f}): "
                    f"their {ratios} are imprecise and cannot be read apart. Keep the one that answers your "
                    "question, or combine them.",
                )
            )
    return advice


def _design_advice(
    frame: pd.DataFrame, covariates: Sequence[str], *, ratios: str, events: pd.Series | None = None
) -> list[Advice]:
    """
    What would make the terms of a model fitted on ``frame`` (the rows it uses) unstable or impossible to estimate:
    a variable left with one value (the model drops it), one value written several ways, levels held by fewer than
    LEVEL_MIN rows or, with ``events`` (per row), without an event, and collinearity.
    """
    advice: list[Advice] = []
    varying: list[str] = []
    for col in covariates:
        values = pd.Series(frame[col])
        if values.nunique(dropna=True) <= 1:
            advice.append(
                Advice(
                    "info",
                    f"{as_code(col)} takes one value in the rows used: the model leaves it out.",
                )
            )
            continue
        varying.append(col)
        if pd.api.types.is_numeric_dtype(values):
            continue
        spelt = _spellings(col, values)
        if spelt is not None:
            advice.append(spelt)
        text = values.astype(str)
        sizes = text.value_counts()
        few = [str(level) for level, n in sizes.items() if n < LEVEL_MIN]
        if not few:
            continue
        per_level = events.groupby(text).sum() if events is not None else None
        empty = [lvl for lvl in few if per_level is not None and per_level.get(lvl, 0) == 0]
        advice.append(
            Advice(
                "caution",
                f"{as_code(col)}: {len(few)} level{'s' if len(few) > 1 else ''} held by fewer than {LEVEL_MIN} rows "
                f"({_some(few)})"
                + (f", {len(empty)} of them without an event" if empty else "")
                + f": their {ratios} are unstable"
                + (" or cannot be estimated, and the model may not converge" if empty else "")
                + ". Merge them with another level, or clean the variable first.",
            )
        )
    if len(varying) >= 2:  # a design that cannot be built is said where the terms are counted
        with contextlib.suppress(ValueError, np.linalg.LinAlgError):
            advice.extend(_collinearity(frame, varying, ratios))
    return advice


def _left_out(
    df: pd.DataFrame, rows: pd.Series, covariates: Sequence[str], model: str
) -> Advice | None:
    """Rows of ``rows`` the model leaves out for a missing covariate, and which covariates miss values."""
    missing = [c for c in covariates if df.loc[rows, c].isna().any()]
    if not missing:
        return None
    lost = int((rows & df[list(covariates)].isna().any(axis=1)).sum())
    return Advice(
        "info",
        f"The {model} leaves out the {lost:,} row{'s' if lost > 1 else ''} missing {listing(missing)} (complete "
        "cases). With many, consider multiple imputation (Enrich > Handle missing data).",
    )


def plan_time_to_event(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    compare: str | None,
    adjust: Sequence[str],
    *,
    group: bool,
    curves: bool = True,
    model: bool = True,
) -> ModelPlan:
    """
    Kaplan-Meier curves (by ``compare`` when ``group``: a grouping variable) and a Cox model of the comparison and
    the adjustment, read on the rows each one keeps: a known follow-up time of 0 or more, a known event, and the
    variables each reads.
    """
    rows = len(df)
    try:
        event = pd.Series(event_indicator(df[event_col]), index=df.index)
    except ValueError as exc:
        return ModelPlan(rows, (), (Advice("violated", f"{as_code(event_col)}: {_said(exc)}"),))
    time = pd.to_numeric(df[time_col], errors="coerce")
    advice: list[Advice] = []
    negative = int((time < 0).sum())
    if negative:
        advice.append(
            Advice(
                "caution",
                f"{negative:,} row{'s' if negative > 1 else ''} with a negative {as_code(time_col)} "
                f"{'are' if negative > 1 else 'is'} left out: check how the follow-up time was computed.",
            )
        )
    usable = time.notna() & event.notna() & (time >= 0)
    compare = chosen(compare)
    steps: list[PlanStep] = []
    if curves:
        by = compare if group else None
        kept = usable & df[by].notna() if by else usable
        if by:
            sizes = df.loc[kept, by].astype(str).value_counts()
            small = [level for level, n in sizes.items() if n < 2]
            text = f"Kaplan-Meier curves by {as_code(by)} ({len(sizes)} groups), their medians and the log-rank test"
            if small:
                advice.append(
                    Advice(
                        "info",
                        f"Fewer than 2 rows, so no curve: {_some(small)} of {as_code(by)}.",
                    )
                )
            spelt = _spellings(by, df.loc[kept, by])
            if spelt is not None:
                advice.append(spelt)
            if len(sizes) - len(small) < 2:
                advice.append(
                    Advice(
                        "caution",
                        "The log-rank test needs two groups of at least 2 rows: none is computed.",
                    )
                )
        else:
            text = "A Kaplan-Meier curve of all the rows and its median"
            if compare:
                advice.append(
                    Advice(
                        "info",
                        f"{as_code(compare)} is numeric: no curves by it. The Cox model gives its hazard ratio "
                        "per unit; to draw curves, group it first (Enrich > Create new variables).",
                    )
                )
        steps.append(PlanStep(text, int(kept.sum()), int(event[kept].sum())))
    covariates = covariates_of(compare, adjust)
    if model and not covariates:
        advice.append(
            Advice(
                "info",
                "The Cox model needs a variable to compare by or to adjust for: Run fits none.",
            )
        )
    elif model:
        kept = usable & df[covariates].notna().all(axis=1)
        events = int(event[kept].sum())
        terms, problem = _terms(df.loc[kept], covariates)
        steps.append(
            PlanStep(
                f"A Cox model {_model_words(compare, adjust)} (Efron ties)",
                int(kept.sum()),
                events,
                terms,
            )
        )
        if problem:
            advice.append(Advice("violated", f"The Cox model cannot be built: {_said(problem)}."))
        else:
            advice += _design_advice(
                df.loc[kept], covariates, ratios="hazard ratios", events=event[kept]
            )
        advice.extend(
            a
            for a in (
                _too_few(events, terms, "events", "hazard ratios"),
                _left_out(df, usable, covariates, "Cox model"),
            )
            if a is not None
        )
    return ModelPlan(
        rows, tuple(steps), tuple(dict.fromkeys(advice))
    )  # a variable both steps read: said once


def plan_binary(
    df: pd.DataFrame, outcome: str, event_level: Any, compare: str | None, adjust: Sequence[str]
) -> ModelPlan:
    """A logistic model of ``outcome`` = ``event_level``, on the rows where every variable it reads is recorded."""
    rows = len(df)
    covariates = covariates_of(compare, adjust)
    if not covariates:
        return ModelPlan(
            rows, (), (Advice("info", "Choose a variable to compare by or to adjust for."),)
        )
    kept = df[[outcome, *covariates]].notna().all(axis=1)
    y = df.loc[kept, outcome].astype(str)
    levels = sorted(y.unique())
    if len(levels) != 2:
        return ModelPlan(
            rows,
            (),
            (
                Advice(
                    "violated",
                    f"{as_code(outcome)} has {len(levels)} level{'s' if len(levels) != 1 else ''} in the rows "
                    "where every variable is recorded: a logistic model needs two.",
                ),
            ),
        )
    events = int((y == str(event_level)).sum())
    others = len(y) - events
    terms, problem = _terms(df.loc[kept], covariates)
    advice: list[Advice] = []
    if problem:
        advice.append(Advice("violated", f"The logistic model cannot be built: {_said(problem)}."))
    else:
        advice += _design_advice(df.loc[kept], covariates, ratios="odds ratios")
    fewer, what = (events, "events") if events <= others else (others, "rows without the event")
    advice.extend(
        a
        for a in (
            _too_few(fewer, terms, what, "odds ratios"),
            _left_out(df, df[outcome].notna(), covariates, "logistic model"),
        )
        if a is not None
    )
    for col in covariates:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        share = (y == str(event_level)).groupby(df.loc[kept, col].astype(str)).mean()
        pure = [str(level) for level, s in share.items() if s in (0.0, 1.0)]
        if pure:
            advice.append(
                Advice(
                    "caution",
                    f"Every row or none has the event at {_some(pure, 3)} of {as_code(col)}: the odds ratio of "
                    "such a level cannot be estimated (separation). Merge it with another level, or leave it out.",
                )
            )
    text = f"A logistic model of {as_code(outcome)} = {as_code(event_level)}, {_model_words(compare, adjust)}"
    return ModelPlan(rows, (PlanStep(text, int(kept.sum()), events, terms),), tuple(advice))


def plan_count(
    df: pd.DataFrame,
    count_col: str,
    person_time: str | None,
    compare: str | None,
    adjust: Sequence[str],
    family: str,
) -> ModelPlan:
    """A Poisson or negative binomial model of ``count_col``, per ``person_time`` when one is chosen."""
    rows = len(df)
    covariates = covariates_of(compare, adjust)
    if not covariates:
        return ModelPlan(
            rows, (), (Advice("info", "Choose a variable to compare by or to adjust for."),)
        )
    offset = chosen(person_time)
    read = [count_col, *covariates, *([offset] if offset else [])]
    kept = df[read].notna().all(axis=1)
    y = pd.to_numeric(df.loc[kept, count_col], errors="coerce")
    advice: list[Advice] = []
    negative = int((y < 0).sum())
    fractional = int(((y >= 0) & (y % 1 != 0)).sum())
    if negative:
        advice.append(
            Advice(
                "violated",
                f"{negative:,} value(s) of {as_code(count_col)} below 0: a count cannot be negative.",
            )
        )
    if fractional:
        advice.append(
            Advice(
                "caution",
                f"{fractional:,} value(s) of {as_code(count_col)} are not whole numbers: a count model expects "
                "numbers of events.",
            )
        )
    if offset:
        not_positive = int((pd.to_numeric(df.loc[kept, offset], errors="coerce") <= 0).sum())
        if not_positive:
            advice.append(
                Advice(
                    "violated",
                    f"{not_positive:,} row(s) with a {as_code(offset)} of 0 or less: person-time must be positive.",
                )
            )
    else:
        advice.append(
            Advice(
                "info",
                "Without person-time, the ratios compare mean counts, not rates: rows followed longer have had "
                "more time to count events.",
            )
        )
    terms, problem = _terms(df.loc[kept], covariates)
    if problem:
        advice.append(Advice("violated", f"The count model cannot be built: {_said(problem)}."))
    else:
        advice += _design_advice(df.loc[kept], covariates, ratios="rate ratios", events=y)
    left_out = _left_out(
        df, df[count_col].notna(), [*covariates, *([offset] if offset else [])], "count model"
    )
    if left_out is not None:
        advice.append(left_out)
    per = f" per {as_code(offset)} of person-time" if offset else ""
    text = f"A {FAMILIES.get(family, family)} model of {as_code(count_col)}{per}, {_model_words(compare, adjust)}"
    events = int(y[y >= 0].sum()) if len(y) else 0
    return ModelPlan(rows, (PlanStep(text, int(kept.sum()), events, terms),), tuple(advice))


# ----------------------------------------------------------------------------- reading a result


@dataclass(frozen=True)
class ComparisonTerm:
    term: str  # the model's term
    label: str  # how it reads: "Treatment vs Placebo", "per unit of age"


def comparison_terms(
    terms: Sequence[str], compare: str | None, values: pd.Series, reference: str | None = None
) -> list[ComparisonTerm]:
    """
    The terms of the comparison among a model's ``terms``, and how each reads. A factor's terms are named
    "<variable>_<level>" (biostats.design.design_matrix), each against the reference level (the first, sorted); a
    numeric variable is one term, read per unit, or as "b vs a" when it takes two values one unit apart (a 0/1 code, a
    boolean): a dose of 0 or 50 still reads per unit, the ratio being the model's per unit.
    ``values`` are the comparison's values in the data the model read.
    """
    compare = chosen(compare)
    if compare is None:
        return []
    present = set(terms)
    if pd.api.types.is_numeric_dtype(values):
        if compare not in present:
            return []
        distinct = sorted(pd.Series(values).dropna().astype(float).unique())
        # the model's ratio is per unit: "b vs a" only when b is a + 1 (0/1, a boolean); 0/50 is still per unit
        label = (
            f"{distinct[1]:g} vs {distinct[0]:g}"
            if len(distinct) == 2 and distinct[1] - distinct[0] == 1
            else f"per unit of {compare}"
        )
        return [ComparisonTerm(compare, label)]
    levels = sorted(pd.Series(values).dropna().astype(str).unique())
    base = reference if reference is not None else (levels[0] if levels else "")
    return [
        ComparisonTerm(f"{compare}_{level}", f"{level} vs {base}")
        for level in levels
        if level != base and f"{compare}_{level}" in present
    ]


def diagnostics(result: Any) -> list[tuple[str, str, str]]:
    """(check, value, how to read it) for a fitted Cox, logistic or count model; nothing for other results."""
    rows: list[tuple[str, str, str]] = []
    if isinstance(result, CoxResult):
        terms = len(result.table)
        rows.append(
            (
                "Events per term",
                f"{result.events / terms:.1f}" if terms else "n/a",
                f"{result.events:,} events for {terms} term{'s' if terms != 1 else ''}; below about {EPV_MIN}, "
                "hazard ratios may be unstable.",
            )
        )
        if result.concordance is not None:
            rows.append(
                (
                    "Concordance (apparent)",
                    f"{result.concordance:.3f}",
                    "How well the model orders the event times, on the data it was fitted to (optimistic): "
                    "0.5 = chance, 1 = perfect.",
                )
            )
        rows.append(("Ties", result.ties.capitalize(), "How tied event times are handled."))
    elif isinstance(result, RegressionResult):
        terms = len(result.table)
        if result.kind == "logistic" and result.events is not None:
            fewer = min(result.events, result.n - result.events)
            rows.append(
                (
                    "Events per term",
                    f"{fewer / terms:.1f}" if terms else "n/a",
                    f"The fewer of events and non-events ({fewer:,}) for {terms} term{'s' if terms != 1 else ''}; "
                    f"below about {EPV_MIN}, odds ratios may be unstable.",
                )
            )
            if result.auc is not None:
                ci = (
                    f" (95% CI {result.auc_ci[0]:.3f} to {result.auc_ci[1]:.3f})"
                    if result.auc_ci
                    else ""
                )
                rows.append(
                    (
                        APPARENT_AUC[0].upper() + APPARENT_AUC[1:],
                        f"{result.auc:.3f}{ci}",
                        "Discrimination on the data the model was fitted to: it overstates the discrimination in "
                        "new patients.",
                    )
                )
            if result.pseudo_r2 is not None:
                rows.append(
                    (
                        "Pseudo-R² (McFadden)",
                        f"{result.pseudo_r2:.3f}",
                        "Improvement of the log-likelihood over a model without covariates; not a share of "
                        "variance explained.",
                    )
                )
        elif result.kind == "poisson" and result.dispersion is not None:
            rows.append(
                (
                    "Dispersion (Pearson chi²/df)",
                    f"{result.dispersion:.2f}",
                    "About 1 when the Poisson variance fits; well above 1 (over-dispersion), the intervals are too "
                    "narrow: fit the negative binomial model.",
                )
            )
        elif result.kind == "negative_binomial" and result.dispersion is not None:
            rows.append(
                (
                    "Alpha (negative binomial)",
                    f"{result.dispersion:.3f}",
                    "The extra variance: variance = mean + alpha × mean². Near 0, the Poisson model fits as well.",
                )
            )
        rows.append(
            (
                "AIC",
                f"{result.aic:.1f}",
                "For comparing models fitted to the same rows: lower is better.",
            )
        )
    return rows


__all__ = [
    "BINARY",
    "COUNT",
    "EPV_MIN",
    "FAMILIES",
    "HELP",
    "NONE",
    "OUTCOME",
    "OUTCOME_TYPES",
    "TIME_TO_EVENT",
    "ComparisonTerm",
    "ModelPlan",
    "OutcomeType",
    "PlanStep",
    "chosen",
    "comparison_terms",
    "covariates_of",
    "diagnostics",
    "listing",
    "plan_binary",
    "plan_count",
    "plan_time_to_event",
]
