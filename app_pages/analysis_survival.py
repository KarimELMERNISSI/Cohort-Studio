"""
Survival & regression page body, organised by the outcome (audit S11): the time to an event (Kaplan-Meier curves,
the log-rank test and the Cox model), a binary outcome (logistic regression) or a count of events (Poisson or
negative binomial regression). For each, the user chooses what is compared and what it is adjusted for; the plan
checked on the data says what Run will do (ui.guidance.outcomes); one click runs it; the result reads effect first,
then the model and its checks (app_pages.analysis_survival_results). Nothing is decided for the user: every default
is a visible choice.

With multiply imputed data (``mi``), every analysis is run on the m imputed datasets and pooled
(biostats.mi.analyze_mi): Kaplan-Meier curves on the cloglog scale with the median read from the pooled
curve, the log-rank test by D2, hazard, odds and rate ratios with Rubin's rules and a D1 test per factor.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import streamlit as st

from app_pages.analysis_survival_results import (
    PH_TEST_IMPUTATIONS,
    Comparison,
    show_count,
    show_logistic,
    show_time_to_event,
)
from biostats.regression import count_regression, logistic_regression
from biostats.survival import EVENT_WORDS, compare_survival, cox_regression
from ui.components import error_card, glossary_popover, page_link, status_chip
from ui.components.mi_mode import mi_token
from ui.components.text import plain
from ui.guidance.outcomes import (
    BINARY,
    COUNT,
    FAMILIES,
    HELP,
    NONE,
    OUTCOME,
    OUTCOME_TYPES,
    TIME_TO_EVENT,
    ModelPlan,
    chosen,
    covariates_of,
    plan_binary,
    plan_count,
    plan_time_to_event,
)
from ui.progress import mark_step
from ui.results import recall, remember
from ui.state import get_state

OUTCOME_KEY = "sr_outcome"
GUIDANCE_KEY = "sr_guidance"
KM, COX = "km", "cox"
ANALYSES = {KM: "Kaplan-Meier & log-rank", COX: "Cox model"}
TIME_HINTS = ("time", "days", "months", "years", "duration", "follow", "fu_")
EVENT_HINTS = ("died", "death", "dead", "event", "status", "relapse", "fail")
_ICON = {
    "ok": ":material/check_circle:",
    "info": ":material/info:",
    "caution": ":material/warning:",
    "violated": ":material/error:",
}
_STATUS = {
    "idle": ("nothing to run", "neutral"),
    "ok": ("ready", "ok"),
    "caution": ("check the warnings", "caution"),
    "violated": ("fix before running", "violated"),
}


def _group_candidates(df: pd.DataFrame, analyzer, max_levels: int = 8) -> list[str]:
    numeric = set(analyzer.numeric_cols)
    out = []
    for col in df.columns:
        n = int(df[col].nunique(dropna=True))
        if 2 <= n <= max_levels and (col not in numeric or n <= 3):
            out.append(col)
    return out


def _variables(df: pd.DataFrame, analyzer, mi) -> tuple[list[str], list[str], list[str]]:
    """
    The numeric, binary and grouping variables the panels offer; with imputed data, only those it holds: a variable
    created after the imputation is not in it, and reading it stopped the page.
    """
    numeric = list(analyzer.numeric_cols)
    binary = list(analyzer.binary_cols)
    groups = _group_candidates(df, analyzer)
    if mi is None:
        return numeric, binary, groups
    held = set(mi.base.columns)
    return (
        [c for c in numeric if c in held],
        [c for c in binary if c in held],
        [c for c in groups if c in held],
    )


def _event_index(levels: list[str]) -> int:
    """
    The level a binary outcome's event most likely is: the first that reads as an event (Yes, 1, Died, Event: the
    words the event indicator reads), else the last sorted. The last sorted alone took "Survived" for Died/Survived.
    """
    for i, level in enumerate(levels):
        if str(level).strip().lower() in EVENT_WORDS:
            return i
    return len(levels) - 1


def _default_index(options: list[str], hints: tuple[str, ...]) -> int:
    """Index of the first option whose name contains a hint (e.g. 'time', 'died'), else 0."""
    for i, name in enumerate(options):
        if any(h in str(name).lower() for h in hints):
            return i
    return 0


def _candidates(
    df: pd.DataFrame, numeric: Sequence[str], groups: Sequence[str], exclude: Sequence[str]
) -> list[str]:
    """What a model can compare by or adjust for: numeric variables, and categorical ones of 2 to 8 levels."""
    return [c for c in df.columns if (c in numeric or c in groups) and c not in exclude]


def _compare_and_adjust(
    candidates: list[str], *, keys: tuple[str, str], help_compare: str
) -> tuple[str | None, list[str]]:
    """The second row of every outcome: what is compared, and what it is adjusted for."""
    left, right = st.columns([1, 2])
    compare = chosen(
        left.selectbox(
            "Compare by", [NONE, *candidates], key=keys[0], persist_state="page", help=help_compare
        )
    )
    adjust = right.multiselect(
        "Adjust for",
        candidates,  # the comparison too: covariates_of reads it once, and a variable picked twice is kept
        key=keys[1],
        persist_state="page",
        help=HELP["adjust"],
        placeholder="No adjustment",
    )
    return compare, adjust


def _render_plan(plan: ModelPlan, *, note: str = "") -> None:
    """What Run will do, checked on the data (while Guidance is on): a status, each step's rows, the advice."""
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown("**Before you run: the plan**")
            status_chip(*_STATUS[plan.status])
        for step in plan.steps:
            counts = f"{step.used:,} of {plan.rows:,} rows"
            if step.events is not None:
                counts += f", {step.events:,} events"
            if step.terms:
                counts += f", {step.terms} term{'s' if step.terms > 1 else ''}"
            st.markdown(f"- {step.text} ({counts})")
        for item in plan.advice:
            st.markdown(f"{_ICON.get(item.level, '')} {item.text}")
        if note:
            st.caption(note)


def _failed(what: str, exc: Exception) -> None:
    """An error card whose first line says why: the models' messages are written for the reader (as text)."""
    error_card(f"{what}: {plain(exc, limit=400)}", exc)


def _planned(mi, frame: pd.DataFrame, columns: Sequence[str]) -> tuple[pd.DataFrame, str]:
    """The rows the plan reads: the observed data, or the first imputed dataset when the analysis is pooled."""
    if mi is None:
        return frame, ""
    return mi.complete(
        1, [c for c in dict.fromkeys(columns) if c]
    ), "Counted on the first imputed dataset."


def run_survival(df: pd.DataFrame, analyzer, *, mi=None) -> None:
    numeric, binary, groups = _variables(df, analyzer, mi)
    token = mi_token(get_state()) if mi is not None else ""

    with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
        outcome = st.segmented_control(
            "Outcome",
            [o.id for o in OUTCOME_TYPES],
            format_func=lambda o: OUTCOME[o].label,
            default=TIME_TO_EVENT,
            required=True,
            persist_state="page",
            key=OUTCOME_KEY,
            help=HELP["outcome"],
        )
        guided = st.toggle(
            "Guidance", value=True, persist_state="page", key=GUIDANCE_KEY, help=HELP["guidance"]
        )
    kind = OUTCOME[outcome]
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.caption(kind.gives)
        glossary_popover(*kind.terms, label="Terms")
    if guided:
        with st.container(horizontal=True, gap="medium"):
            page_link("which_test", "Not sure which analysis fits?", icon=":material/help_center:")
            if kind.simpler:
                page_link(
                    "analyze",
                    f"{kind.simpler}: Hypothesis tests & epidemiology",
                    icon=":material/science:",
                )
    panel = {TIME_TO_EVENT: _time_to_event, BINARY: _binary, COUNT: _count}[outcome]
    panel(df, numeric, binary, groups, mi=mi, token=token, guided=guided)


def _comparison(
    mi, df: pd.DataFrame, compare: str | None, adjust: Sequence[str], result
) -> Comparison:
    """How a result reads: the comparison's values in the data the model read, and its reference level."""
    if compare is None:
        return Comparison(None, tuple(adjust))
    if mi is not None:
        levels = mi.levels([compare]).get(compare)
        return Comparison(compare, tuple(adjust), mi.base[compare], levels[0] if levels else None)
    reference = getattr(result, "reference_levels", {}).get(compare) if result is not None else None
    return Comparison(compare, tuple(adjust), df[compare], reference)


# ----------------------------------------------------------------------------- time to event


def _fit_curves(df, mi, key, config, time_col, event_col, group) -> None:
    try:
        if mi is not None:
            from biostats.mi import analyze_mi

            columns = [time_col, event_col, *([group] if group else [])]
            res = analyze_mi(
                lambda d: compare_survival(d, time_col, event_col, group), mi, columns=columns
            )
            remember(key, res, config=config, n_analysed=len(mi.base), columns=columns)
        else:
            res = compare_survival(df, time_col, event_col, group)
            remember(
                key,
                res,
                config=config,
                n_analysed=sum(c.n for c in res.curves),
                columns=[time_col, event_col, group or ""],
            )
        mark_step("analyse", "survival analysis")
    except Exception as exc:
        _failed("The survival curves could not be computed", exc)


def _fit_cox(df, mi, key, config, time_col, event_col, covariates) -> None:
    try:
        if mi is not None:
            from biostats.mi import analyze_mi

            levels = mi.levels(covariates)
            fitted = analyze_mi(
                lambda d, i: cox_regression(
                    d,
                    time_col,
                    event_col,
                    covariates,
                    levels=levels,
                    diagnostics=i <= PH_TEST_IMPUTATIONS,
                ),
                mi,
                columns=[time_col, event_col, *covariates],
            )
            remember(
                key,
                fitted,
                config=config,
                n_analysed=len(mi.base),
                columns=[time_col, event_col, *covariates],
            )
        else:
            fitted = cox_regression(df, time_col, event_col, covariates)
            remember(
                key,
                fitted,
                config=config,
                n_analysed=fitted.n,
                columns=[time_col, event_col, *covariates],
            )
        mark_step("analyse", "Cox model")
    except Exception as exc:
        _failed("The Cox model could not be fitted", exc)


def _time_to_event(df, numeric, binary, groups, *, mi, token, guided) -> None:
    if not numeric or not binary:
        st.info("A numeric follow-up time and a binary event indicator are needed.")
        return
    c1, c2, c3 = st.columns(3)
    time_col = c1.selectbox(
        "Follow-up time",
        numeric,
        index=_default_index(numeric, TIME_HINTS),
        key="surv_time",
        persist_state="page",
        help=HELP["time"],
    )
    event_col = c2.selectbox(
        "Event",
        binary,
        index=_default_index(binary, EVENT_HINTS),
        key="surv_event",
        persist_state="page",
        help=HELP["event"],
    )
    analyses = c3.pills(
        "Analyses",
        list(ANALYSES),
        format_func=ANALYSES.get,
        selection_mode="multi",
        default=list(ANALYSES),
        key="surv_analyses",
        persist_state="page",
        help=HELP["analyses"],
    )
    compare, adjust = _compare_and_adjust(
        _candidates(df, numeric, groups, (time_col, event_col)),
        keys=("surv_group", "surv_adjust"),
        help_compare=HELP["compare_time"],
    )
    group = (
        compare if compare in groups else None
    )  # curves need groups: a numeric comparison draws one curve
    covariates = covariates_of(compare, adjust)
    curves, model = KM in analyses, COX in analyses and bool(covariates)
    if guided:
        frame, note = _planned(mi, df, [time_col, event_col, *covariates])
        plan = plan_time_to_event(
            frame,
            time_col,
            event_col,
            compare,
            adjust,
            group=group is not None,
            curves=KM in analyses,
            model=COX in analyses,
        )
        _render_plan(plan, note=note)
    pooled = mi is not None
    km_key, km_config = (
        ("survival_mi", (time_col, event_col, group or NONE, token))
        if pooled
        else ("survival", (time_col, event_col, group or NONE))
    )
    cox_key, cox_config = (
        ("cox_mi", (time_col, event_col, tuple(covariates), token))
        if pooled
        else ("cox", (time_col, event_col, tuple(covariates)))
    )
    if st.button(
        "Run analysis",
        type="primary",
        key="surv_run",
        disabled=not (curves or model),
        help="Runs the analyses chosen above on the analysis population.",
    ):
        if curves:
            _fit_curves(df, mi, km_key, km_config, time_col, event_col, group)
        if model:
            _fit_cox(df, mi, cox_key, cox_config, time_col, event_col, covariates)
    km = recall(km_key, config=km_config) if curves else None
    cox = recall(cox_key, config=cox_config) if model else None
    show_time_to_event(
        km,
        cox,
        keys=(km_key, cox_key),
        comparison=_comparison(mi, df, compare, adjust, cox),
        pooled=pooled,
        time_col=time_col,
        frame=None if pooled else df,
        event_col=event_col,
        group=group,
        cox_chosen=COX in analyses,
        cox_fits=bool(covariates),
    )


# ----------------------------------------------------------------------------- binary outcome


def _binary(df, numeric, binary, groups, *, mi, token, guided) -> None:
    if not binary:
        st.info("A binary outcome (two levels, such as yes/no) is needed.")
        return
    c1, c2, _ = st.columns(3)
    outcome = c1.selectbox(
        "Outcome", binary, key="logit_outcome", persist_state="page", help=HELP["binary_outcome"]
    )
    levels_of_outcome = sorted(df[outcome].dropna().astype(str).unique())
    event_level = c2.selectbox(
        "Event",
        levels_of_outcome,
        index=_event_index(levels_of_outcome),
        key="logit_event",
        persist_state="page",
        help=HELP["event_level"],
    )
    compare, adjust = _compare_and_adjust(
        _candidates(df, numeric, groups, (outcome,)),
        keys=("logit_compare", "logit_adjust"),
        help_compare=HELP["compare"],
    )
    covs = covariates_of(compare, adjust)
    if guided:
        frame, note = _planned(mi, df, [outcome, *covs])
        _render_plan(plan_binary(frame, outcome, event_level, compare, adjust), note=note)
    pooled = mi is not None
    key, config = (
        ("logistic_mi", (outcome, event_level, tuple(covs), token))
        if pooled
        else ("logistic", (outcome, event_level, tuple(covs)))
    )
    if st.button(
        "Run analysis",
        key="logit_run",
        type="primary",
        disabled=not covs,
        help="Fits the logistic model on the analysis population; it needs a variable to compare by or to "
        "adjust for.",
    ):
        try:
            if pooled:
                from biostats.mi import analyze_mi

                levels = mi.levels(covs)
                fitted = analyze_mi(
                    lambda d: logistic_regression(
                        d, outcome, covs, event_level=event_level, levels=levels
                    ),
                    mi,
                    columns=[outcome, *covs],
                )
                remember(
                    key, fitted, config=config, n_analysed=len(mi.base), columns=[outcome, *covs]
                )
            else:
                fitted = logistic_regression(df, outcome, covs, event_level=event_level)
                remember(key, fitted, config=config, n_analysed=fitted.n, columns=[outcome, *covs])
            mark_step("analyse", "logistic regression")
        except Exception as exc:
            _failed("The logistic model could not be fitted", exc)
    res = recall(key, config=config)
    show_logistic(
        res,
        key=key,
        comparison=_comparison(mi, df, compare, adjust, res),
        pooled=pooled,
        outcome=outcome,
        event=event_level,
    )


# ----------------------------------------------------------------------------- count outcome


def _count(df, numeric, binary, groups, *, mi, token, guided) -> None:
    if not numeric:
        st.info("A numeric count of events is needed.")
        return
    c1, c2, c3 = st.columns(3)
    count_col = c1.selectbox(
        "Count", numeric, key="count_outcome", persist_state="page", help=HELP["count"]
    )
    exposure = c2.selectbox(
        "Person-time (optional)",
        [NONE, *[c for c in numeric if c != count_col]],
        key="count_time",
        persist_state="page",
        help=HELP["person_time"],
    )
    family = c3.segmented_control(
        "Model",
        list(FAMILIES),
        format_func=FAMILIES.get,
        default="poisson",
        required=True,
        key="count_family",
        persist_state="page",
        help=HELP["family"],
    )
    exposure_col = chosen(exposure)
    compare, adjust = _compare_and_adjust(
        _candidates(df, numeric, groups, (count_col, exposure_col or "")),
        keys=("count_compare", "count_adjust"),
        help_compare=HELP["compare"],
    )
    covs = covariates_of(compare, adjust)
    if guided:
        frame, note = _planned(mi, df, [count_col, exposure_col or "", *covs])
        _render_plan(plan_count(frame, count_col, exposure, compare, adjust, family), note=note)
    pooled = mi is not None
    key, config = (
        ("count_mi", (count_col, exposure, family, tuple(covs), token))
        if pooled
        else ("count", (count_col, exposure, family, tuple(covs)))
    )
    if st.button(
        "Run analysis",
        key="count_run",
        type="primary",
        disabled=not covs,
        help="Fits the count model on the analysis population; it needs a variable to compare by or to adjust "
        "for.",
    ):
        try:
            if pooled:
                from biostats.mi import analyze_mi

                levels = mi.levels(covs)
                fitted = analyze_mi(
                    lambda d: count_regression(
                        d, count_col, covs, exposure_col=exposure_col, family=family, levels=levels
                    ),
                    mi,
                    columns=[count_col, *covs, *([exposure_col] if exposure_col else [])],
                )
                remember(
                    key,
                    fitted,
                    config=config,
                    n_analysed=len(mi.base),
                    columns=[count_col, *covs, exposure_col or ""],
                )
            else:
                fitted = count_regression(
                    df, count_col, covs, exposure_col=exposure_col, family=family
                )
                remember(
                    key,
                    fitted,
                    config=config,
                    n_analysed=fitted.n,
                    columns=[count_col, *covs, exposure_col or ""],
                )
            mark_step("analyse", "count regression")
        except Exception as exc:
            _failed("The count model could not be fitted", exc)
    res = recall(key, config=config)
    show_count(
        res,
        key=key,
        comparison=_comparison(mi, df, compare, adjust, res),
        pooled=pooled,
        rates=exposure_col is not None,
        outcome=count_col,
    )


__all__ = ["run_survival"]
