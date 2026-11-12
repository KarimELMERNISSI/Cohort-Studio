"""
Paired, rates & stratified, a tab of Analyze > Hypothesis tests & epidemiology: four questions, one open at a time
(ui.guidance.measures): two measurements of the same participants (a number: paired t-test or Wilcoxon signed-rank; yes
or no: McNemar), one group against a reference (a mean, or a share), an association within strata
(Mantel-Haenszel), and rates from counts (a calculator).

The analyses of the dataset follow the page pattern of Survival & regression (app_pages.analysis_layout): nothing is
preselected, the plan says what Run will do, one click runs it, the result reads the estimate first and is kept with
what it was computed on. The calculators read the numbers typed: their fields start empty, they have no button, and
their result says where its numbers came from.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from app_pages import analysis_layout as layout
from app_pages.analysis_guidance_ui import guidance_on
from app_pages.analysis_layout import Metric
from biostats._types import TestResult
from biostats.paired import (
    mcnemar_test,
    one_sample_proportion,
    one_sample_t,
    one_sample_wilcoxon,
    paired_t,
    paired_table,
    wilcoxon_signed_rank,
)
from biostats.rates import incidence_rate, mantel_haenszel, rate_ratio, standardised_ratio
from biostats.reliability import category, category_order
from core.registry import as_code
from ui.components import empty_state, error_card, help_popover
from ui.components.mi_mode import mi_token, show_mi_notes
from ui.guidance import measures as m
from ui.guidance.measures import HELP
from ui.progress import mark_step
from ui.results import provenance, remember
from ui.state import get_state

QUESTION_KEY = "paired_question"
OBSERVED = "Imputed data are selected; this analysis reads the observed data: "
POOLED = "Pooled over the {m} imputed datasets (Rubin's rules). "
NOT_TYPED = "Uses the numbers you type: the imputation does not apply."


def _not_pooled(analysis: str) -> None:
    """The reason a test has no pooling rule, and what to run instead (biostats.mi.not_poolable)."""
    from biostats.mi import not_poolable

    entry = not_poolable(analysis)
    if entry is not None:
        st.info(
            f"Not pooled with imputed data. {entry.reason} {entry.alternative} Choose the mean to pool it, or set "
            "Analysis data to Observed in the sidebar to run this test.",
            icon=":material/block:",
        )


def _held(columns: list[str], mi: Any) -> list[str]:
    """With imputed data, a pooled analysis offers the columns of the imputed datasets only: one created after the
    imputation is not in them."""
    return columns if mi is None else [c for c in columns if c in mi.base.columns]


def _rows_read(result: Any, mi: Any) -> int:
    """The rows a pooled test read: those of its result (every row of the population was stamped as analysed, where
    a column left out of the imputation lacks 150 values)."""
    pooled = result.extra.get("result") if hasattr(result, "extra") else None
    rows = getattr(pooled, "n_total", None)
    return int(rows) if rows else len(mi.base)


def _pooled(stored: Any) -> tuple[TestResult | None, Any]:
    """A stored test as it is shown: the pooled test with its notes, or the observed one."""
    if stored is None:
        return None, None
    if hasattr(stored, "extra") and isinstance(getattr(stored, "extra", None), dict):
        return stored.extra["result"], stored
    return stored, None


# ----------------------------------------------------------------------------- paired measurements, a number


def _paired_number(df: pd.DataFrame, *, mi: Any, guided: bool) -> None:
    offered = _held(m.measurements(df), mi)
    if len(offered) < 2:
        empty_state(
            "Fewer than two numeric columns",
            "A change between two measurements needs two numeric columns measured on the same participants.",
            links=(("enrich", "Create new variables"),),
        )
        return
    c1, c2, c3 = st.columns(3)
    before = c1.selectbox(
        "First measurement (before)",
        offered,
        index=None,
        placeholder="Choose a numeric column",
        key="pair_before",
        persist_state="page",
        help=HELP["first"],
    )
    after = c2.selectbox(
        "Second measurement (after)",
        m.look_alikes(before, offered),
        index=None,
        placeholder="Choose a numeric column",
        key="pair_after",
        persist_state="page",
        help=HELP["second"],
    )
    kind = c3.radio(
        "Summary of the change",
        list(m.CHANGES),
        format_func=lambda value: m.CHANGES[value],
        key="pair_kind",
        persist_state="page",
        help=HELP["change"],
    )
    if before and after:
        st.caption(f"Change = {as_code(after)} - {as_code(before)}.")
    plan = layout.kept(
        "pair_plan", (before, after, kind), lambda: m.plan_paired(df, before, after, kind)
    )
    if guided:
        layout.plan_box(plan)
    else:
        layout.checks(plan)
    pooled = mi is not None
    if pooled and kind == m.SIGNED_RANK:
        _not_pooled("wilcoxon_signed_rank")
        return
    if pooled:
        layout.imputed_line(
            POOLED.format(m=mi.m) + "The complete pairs are shown below the result for comparison."
        )
    key, config = (
        ("paired_mi", (before, after, kind, mi_token(get_state())))
        if pooled
        else ("paired", (before, after, kind))
    )
    if layout.run_button("pair_run", ready=bool(plan.steps), what="the paired test"):
        try:
            if pooled:
                from biostats.mi import analyze_mi

                found = analyze_mi(
                    lambda d: paired_t(d[after], d[before]), mi, columns=[before, after]
                )
                remember(
                    key,
                    found,
                    config=config,
                    n_analysed=_rows_read(found, mi),
                    columns=[before, after],
                )
            else:
                found = (
                    paired_t(df[after], df[before])
                    if kind == m.T_TEST
                    else wilcoxon_signed_rank(df[after], df[before])
                )
                remember(key, found, config=config, columns=[before, after])
            mark_step("analyse", "paired test")
        except Exception as exc:
            error_card("The paired test could not be computed.", exc)
    result, whole = _pooled(layout.result_place(key, config, guided=guided))
    if result is not None:
        _show_change(result, df, before, after, kind, pooled=pooled)  # type: ignore[arg-type]
        if whole is not None:
            show_mi_notes(
                whole,
                complete_case=lambda cc: _show_change(cc, df, before, after, kind, pooled=False),  # type: ignore[arg-type]
            )
        layout.details(result)
        provenance(key)


def _show_change(
    result: TestResult, df: pd.DataFrame, before: str, after: str, kind: str, *, pooled: bool
) -> None:
    mean = kind == m.T_TEST
    estimate = result.estimate
    metrics = [
        Metric(
            "Mean change" if mean else "Typical change (pseudo-median)",
            layout.num(estimate),
            f"{layout.interval(result.estimate_ci)} · p {layout.p_text(result.p_value)}",
        )
    ]
    pairs = df[[before, after]].apply(pd.to_numeric, errors="coerce").dropna()
    if not pooled and len(pairs):
        metrics.append(
            Metric(
                "Before and after",
                f"{layout.num(float(pairs[before].mean()), 4)} to {layout.num(float(pairs[after].mean()), 4)}",
                f"means of {as_code(before)} and {as_code(after)}, on the pairs",
            )
        )
    lacking = len(df) - len(pairs)
    metrics.append(
        Metric(
            "Pairs" if mean else "Pairs that differ",
            f"{result.n_total:,}",
            f"of {len(df):,} rows"
            + (f"; {lacking:,} lack a value" if lacking and not pooled else ""),
        )
    )
    words = ""
    if estimate is not None and np.isfinite(estimate):
        words = (
            f"{as_code(after)} is {layout.num(abs(estimate))} {'lower' if estimate < 0 else 'higher'} than "
            f"{as_code(before)} {'on average' if mean else 'typically'} ({result.name})."
        )
    layout.headline(metrics, words, len(result.warnings))
    layout.warn(result.warnings)


# ----------------------------------------------------------------------------- paired measurements, yes or no


def _paired_yes_no(df: pd.DataFrame, *, mi: Any, guided: bool) -> None:
    offered = m.two_valued(df)
    if len(offered) < 2:
        empty_state(
            "Fewer than two yes/no columns",
            "A change in a yes/no characteristic needs two columns that hold the same two values. A column whose "
            "values are written several ways can be cleaned in Data > Clean.",
            links=(("clean", "Clean categories"), ("enrich", "Create new variables")),
        )
        return
    c1, c2, c3 = st.columns(3)
    before = c1.selectbox(
        "First measurement (before)",
        offered,
        index=None,
        placeholder="Choose a yes/no column",
        key="mc_before",
        persist_state="page",
        help=HELP["first"],
    )
    after = c2.selectbox(
        "Second measurement (after)",
        m.look_alikes(before, offered),
        index=None,
        placeholder="Choose a yes/no column",
        key="mc_after",
        persist_state="page",
        help=HELP["second"],
    )
    levels = m.levels_of(df[before]) if before else []
    guess = m.present_value(levels)
    layout.depends("mc_counted", (before, tuple(map(str, levels))), guess)
    counted = c3.selectbox(
        "Share counted: participants with",
        levels,
        index=None,
        format_func=str,
        placeholder="Choose the value" if before else "Choose the first measurement",
        disabled=not before,
        key="mc_counted",
        persist_state="page",
        help=HELP["counted"],
    )
    choices = (before, after, None if counted is None else str(counted))
    plan = layout.kept("mc_plan", choices, lambda: m.plan_paired_yes_no(df, before, after, counted))
    if guided:
        layout.plan_box(plan)
    else:
        layout.checks(plan)
    if mi is not None:
        layout.imputed_line(
            OBSERVED + "the rows where both measurements are recorded. Not pooled in this version."
        )
    if layout.run_button("mc_run", ready=bool(plan.steps), what="McNemar's test"):
        try:
            table = paired_table(df[before], df[after], counted=counted)
            remember(
                "mcnemar",
                {"res": mcnemar_test(table.to_numpy()), "table": table},
                config=choices,
                n_analysed=int(table.to_numpy().sum()),
                columns=[before, after],  # type: ignore[list-item]
            )
            mark_step("analyse", "McNemar test")
        except Exception as exc:
            error_card("McNemar's test could not be computed.", exc)
    stored = layout.result_place("mcnemar", choices, guided=guided)
    if stored is not None:
        _show_yes_no(stored["res"], stored["table"], before, after)  # type: ignore[arg-type]
        layout.details(stored["res"])
        provenance("mcnemar")


def _show_yes_no(result: TestResult, table: pd.DataFrame, before: str, after: str) -> None:
    counts = table.to_numpy()
    other, counted = (str(name) for name in table.index)
    n = int(counts.sum())
    had, has = int(counts[1].sum()), int(counts[:, 1].sum())
    gained, lost = int(counts[0, 1]), int(counts[1, 0])
    ci = result.estimate_ci
    layout.headline(
        [
            Metric(
                f"Change in the share with {counted}",
                f"{100 * result.estimate:+.1f} points",
                f"95% CI {100 * ci.lower:+.1f} to {100 * ci.upper:+.1f} · p {layout.p_text(result.p_value)}",
            ),
            Metric(
                "Before and after",
                f"{layout.percent(had / n)} to {layout.percent(has / n)}",
                f"{had:,} and {has:,} of {n:,}",
            ),
            Metric("Pairs", f"{n:,}", f"{gained + lost:,} changed"),
        ],
        f"{lost:,} participant{'s' if lost != 1 else ''} changed from {as_code(counted)} to {as_code(other)}, "
        f"{gained:,} from {as_code(other)} to {as_code(counted)} (McNemar's exact test on the {gained + lost:,} that "
        "changed).",
        len(result.warnings),
    )
    shown = table.copy()
    shown.index = [f"{before} = {name}" for name in table.index]
    shown.columns = [f"{after} = {name}" for name in table.columns]
    st.caption(
        f"Rows: {as_code(before)} (first measurement). Columns: {as_code(after)} (second). The two cells off the "
        "diagonal hold the participants who changed."
    )
    st.dataframe(shown, width="stretch")
    layout.warn(result.warnings)


# ----------------------------------------------------------------------------- one group against a reference


def _one_mean(df: pd.DataFrame, *, mi: Any, guided: bool) -> None:
    offered = _held(m.measurements(df), mi)
    if not offered:
        empty_state(
            "No numeric column",
            "A mean against a reference needs one numeric column.",
            links=(("enrich", "Create new variables"),),
        )
        return
    c1, c2, c3 = st.columns(3)
    column = c1.selectbox(
        "Measurement",
        offered,
        index=None,
        placeholder="Choose a numeric column",
        key="one_col",
        persist_state="page",
        help=HELP["measurement"],
    )
    layout.depends("one_ref", column, None)  # a reference is in the unit of its measurement
    reference = c2.number_input(
        "Reference value (fixed beforehand)",
        value=None,
        placeholder="a published mean, a target, a limit",
        format="%g",
        key="one_ref",
        persist_state="page",
        help=HELP["reference_value"],
    )
    if column:
        held = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(held):
            c2.caption(
                f"{as_code(column)} here: mean {layout.num(float(held.mean()), 4)}, from "
                f"{layout.num(float(held.min()), 4)} to {layout.num(float(held.max()), 4)}."
            )
    kind = c3.radio(
        "Summary",
        list(m.SUMMARIES),
        format_func=lambda value: m.SUMMARIES[value],
        key="one_kind",
        persist_state="page",
        help=HELP["summary"],
    )
    plan = layout.kept(
        "one_plan", (column, reference, kind), lambda: m.plan_one_mean(df, column, reference, kind)
    )
    if guided:
        layout.plan_box(plan)
    else:
        layout.checks(plan)
    pooled = mi is not None
    if pooled and kind == m.SIGNED_RANK:
        _not_pooled("wilcoxon_signed_rank")
        return
    if pooled:
        layout.imputed_line(
            POOLED.format(m=mi.m) + "The observed values are shown below the result for comparison."
        )
    key, config = (
        ("one_sample_mi", (column, reference, kind, mi_token(get_state())))
        if pooled
        else ("one_sample", (column, reference, kind))
    )
    if layout.run_button("one_run", ready=bool(plan.steps), what="the one-sample test"):
        try:
            if pooled:
                from biostats.mi import analyze_mi

                found = analyze_mi(
                    lambda d: one_sample_t(d[column], reference), mi, columns=[column]
                )
                remember(
                    key, found, config=config, n_analysed=_rows_read(found, mi), columns=[column]
                )
            else:
                found = (
                    one_sample_t(df[column], reference)
                    if kind == m.ONE_T
                    else one_sample_wilcoxon(df[column], reference)
                )
                remember(key, found, config=config, columns=[column])
            mark_step("analyse", "one-sample test")
        except Exception as exc:
            error_card("The one-sample test could not be computed.", exc)
    result, whole = _pooled(layout.result_place(key, config, guided=guided))
    if result is not None and reference is not None:
        _show_mean(result, column, float(reference), kind, len(df))  # type: ignore[arg-type]
        if whole is not None:
            show_mi_notes(
                whole,
                complete_case=lambda cc: _show_mean(cc, column, float(reference), kind, len(df)),  # type: ignore[arg-type]
            )
        layout.details(result)
        provenance(key)


def _show_mean(result: TestResult, column: str, reference: float, kind: str, rows: int) -> None:
    mean = kind == m.ONE_T
    estimate, ci = result.estimate, result.estimate_ci
    metrics = []
    if estimate is not None and ci is not None:
        metrics.append(
            Metric(
                "Mean" if mean else "Typical value (pseudo-median)",
                layout.num(estimate + reference, 4),
                f"95% CI {layout.span(ci.lower + reference, ci.upper + reference, 4)}",
            )
        )
    metrics += [
        Metric(
            f"Difference from {reference:g}",
            f"{estimate:+.3g}" if estimate is not None else "n/a",
            f"{layout.interval(ci)} · p {layout.p_text(result.p_value)}",
        ),
        Metric(
            "Values" if mean else "Values that differ", f"{result.n_total:,}", f"of {rows:,} rows"
        ),
    ]
    words = ""
    if estimate is not None and np.isfinite(estimate):
        words = (
            f"The {'mean' if mean else 'typical value'} of {as_code(column)} is {layout.num(abs(estimate))} "
            f"{'below' if estimate < 0 else 'above'} the reference of {reference:g} ({result.name}). Reference: "
            "typed."
        )
    layout.headline(metrics, words, len(result.warnings))
    layout.warn(result.warnings)


def _share_metrics(
    result: TestResult, with_it: int, counted: int, reference: float, what: str, where: str
) -> None:
    ci = result.estimate_ci
    share = float("nan") if result.estimate is None else float(result.estimate)
    layout.headline(
        [
            Metric(
                f"Share with {what}",
                layout.percent(share),
                f"{with_it:,} of {counted:,} · {layout.interval(ci, scale=100, unit='%')} (Wilson)",
            ),
            Metric(
                "Against a reference of",
                layout.percent(reference),
                f"p {layout.p_text(result.p_value)} (exact binomial test)",
            ),
            Metric("Participants", f"{counted:,}", where),
        ],
        f"The share is {100 * abs(share - reference):.1f} points "
        f"{'below' if share < reference else 'above'} the reference. Reference: typed.",
    )


def _one_share(df: pd.DataFrame, *, mi: Any, guided: bool) -> None:
    source = layout.choose(
        "Counts from", m.SOURCES, key="prop_source", help=HELP["source"], default=m.DATASET
    )
    if source == m.TYPED:
        c1, c2, c3 = st.columns(3)
        with_it = c1.number_input(
            "Number with the characteristic",
            min_value=0,
            value=None,
            step=1,
            placeholder="for example 30",
            key="prop_k",
            persist_state="page",
            help=HELP["with_characteristic"],
        )
        counted = c2.number_input(
            "Number of participants",
            min_value=1,
            value=None,
            step=1,
            placeholder="for example 100",
            key="prop_n",
            persist_state="page",
            help=HELP["participants"],
        )
        reference = c3.number_input(
            "Reference share (fixed beforehand)",
            min_value=0.0,
            max_value=1.0,
            value=None,
            step=0.01,
            placeholder="0.20 for 20%",
            key="prop_ref",
            persist_state="page",
            help=HELP["reference_share"],
        )
        if mi is not None:
            layout.imputed_line(NOT_TYPED)
        ready, advice = m.check_share(with_it, counted, reference)
        layout.advice_lines(advice)
        if not ready:
            if not advice:
                st.caption(
                    ":material/info: Type the number with the characteristic, the number of participants and the "
                    "reference share: the result appears here."
                )
            return
        result = one_sample_proportion(int(with_it), int(counted), float(reference))  # type: ignore[arg-type]
        _share_metrics(
            result, int(with_it), int(counted), float(reference), "the characteristic", "typed"
        )  # type: ignore[arg-type]
        layout.typed_line(f"From the numbers typed: {int(with_it):,} of {int(counted):,}.")  # type: ignore[arg-type]
        st.caption("Method: " + " | ".join(result.references))
        return
    offered = m.two_valued(df)
    if not offered:
        empty_state(
            "No yes/no column",
            "A share read from the dataset needs a column with two values. Choose Numbers I type to enter counts.",
            links=(("clean", "Clean categories"),),
        )
        return
    c1, c2, c3 = st.columns(3)
    column = c1.selectbox(
        "Variable (yes or no)",
        offered,
        index=None,
        placeholder="Choose a yes/no column",
        key="prop_col",
        persist_state="page",
        help=HELP["share_variable"],
    )
    levels = m.levels_of(df[column]) if column else []
    guess = m.present_value(levels)
    layout.depends("prop_level", (column, tuple(map(str, levels))), guess)
    level = c2.selectbox(
        "Share counted: participants with",
        levels,
        index=None,
        format_func=str,
        placeholder="Choose the value" if column else "Choose the variable first",
        disabled=not column,
        key="prop_level",
        persist_state="page",
        help=HELP["share_level"],
    )
    reference = c3.number_input(
        "Reference share (fixed beforehand)",
        min_value=0.0,
        max_value=1.0,
        value=None,
        step=0.01,
        placeholder="0.20 for 20%",
        key="prop_ref",
        persist_state="page",
        help=HELP["reference_share"],
    )
    config = (column, None if level is None else str(level), reference)
    plan = layout.kept("prop_plan", config, lambda: m.plan_share(df, column, level, reference))
    if guided:
        layout.plan_box(plan)
    else:
        layout.checks(plan)
    if mi is not None:
        layout.imputed_line(OBSERVED + "the rows with a value. Not pooled in this version.")
    if layout.run_button("prop_run", ready=bool(plan.steps), what="the exact binomial test"):
        try:
            with_it, counted = m.share_counts(df, column, level)  # type: ignore[arg-type]
            remember(
                "proportion",
                one_sample_proportion(with_it, counted, float(reference)),  # type: ignore[arg-type]
                config=config,
                n_analysed=counted,
                columns=[column],  # type: ignore[list-item]
            )
            mark_step("analyse", "one-sample proportion")
        except Exception as exc:
            error_card("The share could not be compared with its reference.", exc)
    result = layout.result_place("proportion", config, guided=guided)
    if result is not None and reference is not None:
        with_it, counted = m.share_counts(df, column, level)  # type: ignore[arg-type]
        _share_metrics(
            result,
            with_it,
            counted,
            float(reference),
            f"{column} = {category(level)}",
            as_code(column),
        )
        st.caption("Method: " + " | ".join(result.references))
        provenance("proportion")


# ----------------------------------------------------------------------------- an association within strata


def _levels_as_text(df: pd.DataFrame, column: str | None) -> list[str]:
    return sorted(df[column].dropna().astype(str).unique()) if column else []


def _strata(df: pd.DataFrame, *, mi: Any, guided: bool) -> None:
    offered = m.two_valued(df)
    groups = m.strata(df)
    if len(offered) < 2 or not groups:
        empty_state(
            "Not enough columns for a stratified analysis",
            "It needs a yes/no outcome, an exposure in two values and a variable of 2 to 8 groups to adjust for. A "
            "column whose values are written several ways can be cleaned in Data > Clean.",
            links=(("clean", "Clean categories"), ("enrich", "Create new variables")),
        )
        return
    c1, c2 = st.columns(2)
    outcome = c1.selectbox(
        "Outcome (yes or no)",
        offered,
        index=None,
        placeholder="Choose a yes/no column",
        key="mh_outcome",
        persist_state="page",
        help=HELP["outcome"],
    )
    events = _levels_as_text(df, outcome)
    guess = m.present_value(events)
    layout.depends("mh_ev_level", (outcome, tuple(events)), guess)
    event = c2.selectbox(
        "Event",
        events,
        index=None,
        placeholder="Choose the value" if outcome else "Choose the outcome first",
        disabled=not outcome,
        key="mh_ev_level",
        persist_state="page",
        help=HELP["event"],
    )
    c3, c4 = st.columns(2)
    compare = c3.selectbox(
        "Compare by",
        [c for c in offered if c != outcome],
        index=None,
        placeholder="Choose a column with two values",
        key="mh_exposure",
        persist_state="page",
        help=HELP["compare"],
    )
    compared = _levels_as_text(df, compare)
    layout.depends("mh_reference", (compare, tuple(compared)), compared[0] if compared else None)
    reference = c4.selectbox(
        "Reference group",
        compared,
        index=None,
        placeholder="Choose what is compared first",
        disabled=not compare,
        key="mh_reference",
        persist_state="page",
        help=HELP["reference_group"],
    )
    candidates = [c for c in groups if c not in (outcome, compare)]
    stratum = st.selectbox(
        "Adjust for (one variable, 2 to 8 groups)",
        candidates,
        index=None,
        placeholder="Choose a variable with 2 to 8 groups",
        key="mh_stratum",
        persist_state="page",
        help=HELP["adjust"],
    )
    exposed = next((level for level in compared if level != reference), None)
    config = (compare, outcome, stratum, exposed, event)
    plan = layout.kept(
        "mh_plan",
        (outcome, event, compare, reference, stratum),
        lambda: m.plan_strata(df, outcome, event, compare, reference, stratum),
    )
    # a stratum whose values are written several ways is cleaned from the plan, and its copy takes its place
    actions = {"group_key": "mh_stratum", "groupable": candidates, "use": "Adjust for"}
    clean = {stratum: actions} if stratum else {}
    if guided:
        layout.plan_box(plan, clean=clean)
    else:
        layout.checks(plan)
    if mi is not None:
        layout.imputed_line(
            OBSERVED
            + "the rows with the three values. Not pooled in this version. For a pooled, adjusted odds "
            "ratio use a logistic model (Survival & regression, Binary)."
        )
    if layout.run_button("mh_run", ready=bool(plan.steps), what="the stratified analysis"):
        try:
            remember(
                "mantel_haenszel",
                mantel_haenszel(
                    df,
                    compare,  # type: ignore[arg-type]
                    outcome,  # type: ignore[arg-type]
                    stratum,  # type: ignore[arg-type]
                    exposure_level=exposed,
                    event_level=event,
                ),
                config=config,
                columns=[compare, outcome, stratum],  # type: ignore[list-item]
            )
            mark_step("analyse", "Mantel-Haenszel analysis")
        except Exception as exc:
            error_card("The stratified analysis could not be computed.", exc)
    result = layout.result_place("mantel_haenszel", config, guided=guided)
    if result is not None:
        _show_strata(result, outcome, event, compare, exposed, reference, stratum)  # type: ignore[arg-type]
        provenance("mantel_haenszel")


def _ratio(value: Any) -> str:
    """A ratio to three significant figures: at two decimals, a lower bound of 0.996 read "1.00" beside p = 0.051."""
    return layout.num(value)


def _ratio_ci(ci: Any) -> str:
    return layout.interval(ci)


def _show_strata(
    res: Any, outcome: str, event: str, compare: str, exposed: str, reference: str, stratum: str
) -> None:
    layout.headline(
        [
            Metric(
                f"Odds ratio, adjusted for {stratum}",
                _ratio(res.or_mh),
                f"{_ratio_ci(res.or_mh_ci)} · p {layout.p_text(res.mh_test_p)}",
            ),
            Metric("Risk ratio, adjusted", _ratio(res.rr_mh), _ratio_ci(res.rr_mh_ci)),
            Metric(
                "Unadjusted",
                f"{_ratio(res.crude_or)} and {_ratio(res.crude_rr)}",
                "odds ratio and risk ratio",
            ),
        ],
        f"{as_code(outcome)} = {as_code(event)}, {as_code(exposed)} against {as_code(reference)} "
        f"({as_code(compare)}), within {res.n_strata} strata of {as_code(stratum)}.",
        len(res.warnings),
    )
    ratios = pd.to_numeric(res.per_stratum["odds_ratio"], errors="coerce").dropna()
    ratios = ratios[np.isfinite(ratios)]
    span = (
        f"The odds ratios of the strata run from {ratios.min():.2f} to {ratios.max():.2f}: "
        if len(ratios)
        else ""
    )
    p = res.breslow_day_p
    if p is None or not np.isfinite(float(p)):
        st.markdown(
            ":material/help: Whether the strata differ was not tested: a stratum holds one group only, or one "
            "outcome only. Join or leave out such strata to test it."
        )
    elif p >= 0.05:
        st.markdown(
            f":material/check_circle: {span}no evidence that they differ (Breslow-Day p {layout.p_text(p)}). "
            "That is no proof that they are alike: the test has little power in small strata."
        )
    else:
        st.markdown(
            f":material/warning: {span}they differ (Breslow-Day p {layout.p_text(p)}). Report the ratio of each "
            "stratum, not one combined ratio."
        )
    shown = res.per_stratum.rename(
        columns={
            "stratum": "Stratum",
            "exposed_events": f"{exposed}: events",
            "exposed_total": f"{exposed}: participants",
            "reference_events": f"{reference}: events",
            "reference_total": f"{reference}: participants",
            "odds_ratio": "Odds ratio",
            "risk_ratio": "Risk ratio",
        }
    )
    st.dataframe(
        shown.style.format(
            {"Odds ratio": "{:.3f}", "Risk ratio": "{:.3f}"}, na_rep="not estimable"
        ),
        hide_index=True,
        width="stretch",
    )
    layout.warn(res.warnings)
    with st.expander("Details: the tests and the method"):
        st.caption(
            f"Test of the association over the strata (Cochran-Mantel-Haenszel, continuity-corrected): chi2 = "
            f"{layout.num(res.mh_test_statistic)}, p {layout.p_text(res.mh_test_p)}. Test that the odds ratios "
            f"of the strata are alike (Breslow-Day): chi2 = {layout.num(res.breslow_day_statistic)}, p "
            f"{layout.p_text(res.breslow_day_p)}. Intervals: Robins-Breslow-Greenland for the odds ratio, "
            "Greenland-Robins for the risk ratio."
        )
        st.caption("Method: " + " | ".join(res.references))
    help_popover(
        "The adjusted ratios combine the ratios found within each stratum; their distance from the unadjusted ones "
        "says how much the variable adjusted for mattered. One combined ratio sums the strata up only when their "
        "ratios are alike. To adjust for several variables at once, use a logistic model (Survival & regression, "
        "Binary)."
    )


# ----------------------------------------------------------------------------- rates, from counts


def _count(label: str, key: str, column: Any, example: str) -> int | None:
    return column.number_input(
        label,
        min_value=0,
        value=None,
        step=1,
        placeholder=example,
        key=key,
        persist_state="page",
        help=HELP["events"],
    )


def _time(label: str, key: str, column: Any, example: str) -> float | None:
    return column.number_input(
        label,
        min_value=0.0,
        value=None,
        format="%g",
        placeholder=example,
        key=key,
        persist_state="page",
        help=HELP["person_time"],
    )


def _rate_metric(label: str, rate: Any, per: int, unit: str) -> Metric:
    return Metric(
        label,
        f"{rate.rate * per:.2f}",
        f"per {per:,} {unit} · {rate.events:,} events / {rate.person_time:,.6g} · 95% CI "
        f"{rate.ci.lower * per:.2f} to {rate.ci.upper * per:.2f}",
    )


def _more_or_fewer(ratio: float) -> str:
    if ratio < 1:
        return f"{1 - ratio:.0%} fewer"
    if ratio >= 2:
        return f"{ratio:.1f} times as many"
    return f"{ratio - 1:.0%} more"


def _rates(df: pd.DataFrame, *, mi: Any, guided: bool) -> None:
    calculation = layout.choose(
        "Calculate",
        m.CALCULATIONS,
        key="rate_calc",
        help=HELP["calculate"],
        default=m.TWO_RATES,
        depth=2,
    )
    if mi is not None:
        layout.imputed_line(NOT_TYPED)
    if calculation == m.EXPECTED:
        _expected()
        return
    two = calculation == m.TWO_RATES
    label, e_col, t_col = st.columns([1.2, 2, 2], vertical_alignment="bottom")
    label.markdown("**Group compared**" if two else "**The group**")
    events = _count("Events", "rate_e1", e_col, "for example 15")
    person_time = _time("Person-time at risk", "rate_t1", t_col, "for example 1200")
    reference_events = reference_time = None
    if two:
        label, e_col, t_col = st.columns([1.2, 2, 2], vertical_alignment="bottom")
        label.markdown("**Reference group**")
        reference_events = _count("Events (reference group)", "rate_e0", e_col, "for example 4")
        reference_time = _time(
            "Person-time at risk (reference group)", "rate_t0", t_col, "for example 1000"
        )
    u1, u2 = st.columns(2)
    unit = u1.selectbox(
        "Unit of person-time",
        list(m.PERSON_TIME),
        key="rate_unit",
        persist_state="page",
        help=HELP["unit"],
    )
    per = u2.selectbox(
        "Show rates per",
        [100, 1000, 10000, 100000],
        index=1,
        format_func=lambda value: f"{value:,}",
        key="rate_per",
        persist_state="page",
        help=HELP["per"],
    )
    ready, advice = m.check_rates(
        calculation, events, person_time, reference_events, reference_time
    )
    layout.advice_lines(advice)
    if not ready:
        if not advice:
            st.caption(
                ":material/info: Type the events and the person-time at risk"
                + (" of both groups" if two else "")
                + ": the result appears here."
            )
        return
    try:
        first = incidence_rate(int(events), float(person_time), per=per)  # type: ignore[arg-type]
        typed = (
            f"From the numbers typed: {first.events:,} events over {first.person_time:,.6g} {unit}"
        )
        if not two:
            layout.headline([_rate_metric("Rate", first, per, unit)])
            layout.typed_line(typed + ".")
            st.caption("Method: " + " | ".join(first.references))
            return
        second = incidence_rate(int(reference_events), float(reference_time), per=per)  # type: ignore[arg-type]
        typed += f" against {second.events:,} over {second.person_time:,.6g}."
        metrics = [
            _rate_metric("Group compared", first, per, unit),
            _rate_metric("Reference group", second, per, unit),
        ]
        if not first.events and not second.events:
            layout.headline(metrics, "No event in either group: no ratio.")
            layout.typed_line(typed)
            st.caption("Method: " + " | ".join(first.references))
            return
        ratio = rate_ratio(
            int(events),  # type: ignore[arg-type]
            float(person_time),  # type: ignore[arg-type]
            int(reference_events),  # type: ignore[arg-type]
            float(reference_time),  # type: ignore[arg-type]
            per=per,
        )
        bounded = np.isfinite(ratio.ratio) and np.isfinite(ratio.ci.upper)
        head = Metric(
            "Rate ratio",
            _ratio(ratio.ratio) if bounded else "no upper bound",
            (
                f"{_ratio_ci(ratio.ci)}"
                if bounded
                else f"at least {_ratio(ratio.ci.lower)} (95% lower bound)"
            )
            + f" · p {layout.p_text(ratio.p_value)}",
        )
        layout.headline(
            [head, *metrics],
            f"Group compared against the reference group ({ratio.test} test and interval for the ratio; exact "
            "Poisson intervals for the rates).",
        )
        layout.typed_line(typed)
        with st.expander("Details: the rate difference and the method"):
            st.caption(
                f"Rate difference: {ratio.rate_difference * per:.2f} per {per:,} {unit} (no interval is given for "
                "it)."
            )
            st.caption("Method: " + " | ".join(ratio.references))
    except Exception as exc:
        error_card("The rates could not be computed.", exc)
        return
    help_popover(
        "A rate is events over the time at risk, not over a number of participants. The ratio compares two rates; "
        "its interval and its test are exact and agree: the interval leaves 1 out exactly when p is below 0.05."
    )


def _expected() -> None:
    c1, c2, c3 = st.columns(3)
    observed = c1.number_input(
        "Observed events",
        min_value=0,
        value=None,
        step=1,
        placeholder="for example 25",
        key="smr_obs",
        persist_state="page",
        help=HELP["observed"],
    )
    expected = c2.number_input(
        "Expected events",
        min_value=0.0,
        value=None,
        placeholder="for example 18.4",
        format="%g",
        key="smr_exp",
        persist_state="page",
        help=HELP["expected"],
    )
    kind = c3.radio(
        "The events are",
        list(m.EVENT_KINDS),
        format_func=lambda value: m.EVENT_KINDS[value],
        horizontal=True,
        key="smr_kind",
        persist_state="page",
        help=HELP["event_kind"],
    )
    ready, advice = m.check_expected(observed, expected)
    layout.advice_lines(advice)
    if not ready:
        if not advice:
            st.caption(
                ":material/info: Type the observed and the expected events: the result appears here."
            )
        return
    try:
        found = standardised_ratio(int(observed), float(expected))  # type: ignore[arg-type]
    except Exception as exc:
        error_card("The ratio could not be computed.", exc)
        return
    layout.headline(
        [
            Metric(
                m.RATIO_NAMES[kind],
                _ratio(found.ratio),
                f"{_ratio_ci(found.ci)} · p {layout.p_text(found.p_value)} (exact Poisson)",
            ),
            Metric("Observed", f"{found.observed:,}"),
            Metric("Expected", f"{found.expected:,.6g}"),
        ],
        f"{_more_or_fewer(found.ratio).capitalize()} events than expected; the interval runs from "
        f"{_more_or_fewer(found.ci.lower)} to {_more_or_fewer(found.ci.upper)}.",
    )
    layout.typed_line(
        f"From the numbers typed: {found.observed:,} observed against {found.expected:,.6g} expected."
    )
    st.caption("Method: " + " | ".join(found.references))
    help_popover(
        "An SMR or SIR compares the events observed in the cohort with those expected if the reference "
        "population's rates applied. It speaks for that reference population only, and takes the expected number "
        "as known."
    )


# ----------------------------------------------------------------------------- the tab


def _paired(df: pd.DataFrame, *, mi: Any, guided: bool) -> None:
    measure = layout.choose(
        "The measurement is", m.MEASURES, key="pair_measure", help=HELP["measure"], depth=2
    )
    panel = _paired_number if measure == m.NUMBER else _paired_yes_no
    panel(df, mi=mi, guided=guided)


def _one_group(df: pd.DataFrame, *, mi: Any, guided: bool) -> None:
    compared = layout.choose(
        "Compare", m.COMPARED, key="one_compare", help=HELP["compared"], depth=2
    )
    panel = _one_mean if compared == m.MEAN else _one_share
    panel(df, mi=mi, guided=guided)


def run_paired_and_rates(df: pd.DataFrame, analyzer: Any, *, mi: Any = None) -> None:
    """The tab: its Question selector, then the analysis it names. With imputed data (``mi``), the paired and the
    one-sample t-tests are pooled over the imputed datasets; every other analysis says which rows it reads."""
    guided = guidance_on()
    entry = layout.choose_entry(m.PAIRED_ENTRIES, key=QUESTION_KEY, guided=guided)
    panel = {m.PAIRED: _paired, m.ONE_GROUP: _one_group, m.STRATA: _strata, m.RATES: _rates}[
        entry.id
    ]
    panel(df, mi=mi, guided=guided)


__all__ = ["QUESTION_KEY", "category_order", "run_paired_and_rates"]
