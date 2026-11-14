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

from types import SimpleNamespace
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


def _not_pooled(analysis: str, instead: str | None = None) -> None:
    """The reason a test has no pooling rule, and what to run instead (biostats.mi.not_poolable); ``instead``
    where the catalogue's alternative is another analysis's (it names the paired test for one group too)."""
    from biostats.mi import not_poolable

    entry = not_poolable(analysis)
    if entry is not None:
        st.info(
            f"Not pooled with imputed data. {entry.reason} {instead or entry.alternative} To run this test, set "
            "Analysis data to Observed in the sidebar.",
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
            links=(("enrich", "Create new variables", "variables"),),
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
        **layout.start_at("pair_kind", m.CHANGES, 3),
    )
    if before and after:
        st.caption(f"Change = {as_code(after)} - {as_code(before)}.")
    pooled = mi is not None
    if pooled and kind == m.SIGNED_RANK:
        # before any plan: a plan read "ready" over an analysis that has no Run
        _not_pooled("wilcoxon_signed_rank")
        return
    # the plan of what runs: pooled, the test reads the rows of the imputed datasets, not the complete pairs
    read = m.pooled_rows(mi, [before, after]) if pooled and before and after else None
    plan = layout.kept(
        "pair_plan",
        (before, after, kind, mi_token(get_state()) if pooled else None),
        lambda: m.plan_paired(df, before, after, kind, read),
    )
    if guided:
        layout.plan_box(plan)
    else:
        layout.checks(plan)
    if pooled:
        layout.imputed_line(
            POOLED.format(m=mi.m) + "The complete pairs are shown below the result for comparison."
        )
    key, config = (
        ("paired_mi", (before, after, kind, mi_token(get_state())))
        if pooled
        else ("paired", (before, after, kind))
    )
    if layout.run_button("pair_run", ready=bool(plan.steps), what="the paired test", plan=plan):
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
    result, whole = _pooled(layout.result_place(key, config, guided=guided, words=m.CHANGES))
    if result is not None:
        _show_change(result, df, before, after, kind, pooled=pooled)  # type: ignore[arg-type]
        if whole is not None:
            show_mi_notes(
                whole,
                complete_case=lambda cc: _show_change(cc, df, before, after, kind, pooled=False),  # type: ignore[arg-type]
            )
        layout.details(result, more=m.WITHIN_ONE_GROUP)
        provenance(key)


def _named(name: str) -> str:
    """A test's name inside brackets: "Paired t-test (pooled over 3 imputations)" reads "paired t-test, pooled over 3
    imputations", without brackets within brackets, and in lower case unless it starts with a person's name
    ("Wilcoxon signed-rank")."""
    said = name.replace(" (", ", ").replace(")", "")
    common = said.replace("-", " ").split()[:1] in (["Paired"], ["One"], ["Exact"])
    return said[0].lower() + said[1:] if common else said


def _show_change(
    result: TestResult, df: pd.DataFrame, before: str, after: str, kind: str, *, pooled: bool
) -> None:
    mean = kind == m.T_TEST
    estimate = result.estimate
    value, bounds = layout.with_interval(estimate, result.estimate_ci)
    metrics = [
        Metric(
            "Mean change" if mean else "Typical change (pseudo-median)",
            value,
            f"{bounds} · p {layout.p_text(result.p_value)}",
        )
    ]
    pairs = df[[before, after]].apply(pd.to_numeric, errors="coerce").dropna()
    if not pooled and len(pairs):
        metrics.append(
            Metric(
                "Before and after",
                layout.span(float(pairs[before].mean()), float(pairs[after].mean()), 4),
                f"means of {as_code(before)} and {as_code(after)}, on the pairs",
            )
        )
    # pooled, the pairs are those of an imputed dataset: fewer than the rows when a column was not imputed
    lacking = len(df) - (result.n_total if pooled else len(pairs))
    metrics.append(
        Metric(
            "Pairs" if mean else "Pairs that differ",
            f"{result.n_total:,}",
            f"of {len(df):,} rows"
            + (
                f"; {lacking:,} lack a value" + (" that was not imputed" if pooled else "")
                if lacking > 0
                else ""
            ),
        )
    )
    words = ""
    if estimate is not None and np.isfinite(estimate):
        # the sentence repeats the headline's own figure ("5.5 lower" stood under "-5.50")
        how = "on average" if mean else "typically"
        words = (
            f"{as_code(after)} is the same as {as_code(before)} {how}"
            if float(estimate) == 0
            else f"{as_code(after)} is {value.lstrip('-')} {'lower' if estimate < 0 else 'higher'} than "
            f"{as_code(before)} {how}"
        ) + f" ({_named(result.name)})."
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
            links=(("clean", "Clean categories"), ("enrich", "Create new variables", "variables")),
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
        fields = ((before, "mc_before", "first"), (after, "mc_after", "second"))
        clean = {
            column: {
                "group_key": key,
                "groupable": offered,
                "use": f"Use as the {which} measurement:",
            }
            for column, key, which in fields
            if column
        }
        layout.plan_box(plan, clean=clean)
    else:
        layout.checks(plan)
    if mi is not None:
        layout.imputed_line(
            OBSERVED + "the rows where both measurements are recorded. Not pooled in this version."
        )
    if layout.run_button("mc_run", ready=bool(plan.steps), what="McNemar's test", plan=plan):
        try:
            table = paired_table(df[before], df[after], counted=counted)
            # the change is that of one value's share: without its name, the dossier's "difference in proportions
            # (after - before) = -0.0625" read 0.0625 with the other value counted, and nothing said which
            found = mcnemar_test(table.to_numpy()).model_copy(
                update={
                    "estimate_label": f"change in the share with {before} = {category(counted)} "
                    f"(second - first, {after} against {before})"
                }
            )
            remember(
                "mcnemar",
                {"res": found, "table": table},
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


def _signed(value: float, places: int = 1) -> str:
    """A change with its sign, by the one rule for a number: "+3.5", "-12.5", and "0.0" without a sign."""
    text = layout.fixed(value, places)
    return f"+{text}" if layout.known(value) and float(value) > 0 and text.strip("0.,") else text


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
                f"{_signed(100 * result.estimate)} points",
                f"95% CI {_signed(100 * ci.lower)} to {_signed(100 * ci.upper)} · p "
                f"{layout.p_text(result.p_value)}",
            ),
            Metric(
                "Before and after",
                f"{layout.percent(had / n)} to {layout.percent(has / n)}",
                f"{had:,} and {has:,} of {n:,}",
            ),
            Metric("Pairs", f"{n:,}", f"{gained + lost:,} changed"),
        ],
        f"From {as_code(before)} to {as_code(after)}: {lost:,} participant{'s' if lost != 1 else ''} changed from "
        f"{as_code(counted)} to {as_code(other)}, "
        f"{gained:,} from {as_code(other)} to {as_code(counted)} (McNemar's exact test).",
        len(result.warnings),
    )
    # the headers are the two values, the caption names the axes; with its totals, as the other tables
    shown = table.copy()
    shown.index = [str(name) for name in table.index]
    shown.columns = [str(name) for name in table.columns]
    shown["Total"] = shown.sum(axis=1)
    shown.loc["Total"] = shown.sum(axis=0)
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
            links=(("enrich", "Create new variables", "variables"),),
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
        **layout.start_at("one_kind", m.SUMMARIES, 3),
    )
    pooled = mi is not None
    if pooled and kind == m.SIGNED_RANK:
        _not_pooled("wilcoxon_signed_rank", "Pool the mean (one-sample t-test) instead.")
        return
    read = m.pooled_rows(mi, [column]) if pooled and column else None
    plan = layout.kept(
        "one_plan",
        (column, reference, kind, mi_token(get_state()) if pooled else None),
        lambda: m.plan_one_mean(df, column, reference, kind, read),
    )
    if guided:
        layout.plan_box(plan)
    else:
        layout.checks(plan)
    if pooled:
        layout.imputed_line(
            POOLED.format(m=mi.m) + "The observed values are shown below the result for comparison."
        )
    key, config = (
        ("one_sample_mi", (column, reference, kind, mi_token(get_state())))
        if pooled
        else ("one_sample", (column, reference, kind))
    )
    if layout.run_button("one_run", ready=bool(plan.steps), what="the one-sample test", plan=plan):
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
    result, whole = _pooled(layout.result_place(key, config, guided=guided, words=m.SUMMARIES))
    if result is not None and reference is not None:
        # the rows without a value: those of the data, or (pooled) those a column left out of the imputation lacks
        lacking = (
            len(df) - result.n_total
            if pooled
            else int(pd.to_numeric(df[column], errors="coerce").isna().sum())  # type: ignore[index]
        )
        _show_mean(result, column, float(reference), kind, len(df), lacking, pooled=pooled)  # type: ignore[arg-type]
        if whole is not None:
            show_mi_notes(
                whole,
                complete_case=lambda cc: _show_mean(  # type: ignore[arg-type]
                    cc,
                    column,
                    float(reference),
                    kind,
                    len(df),
                    int(pd.to_numeric(df[column], errors="coerce").isna().sum()),  # type: ignore[index]
                ),
            )
        layout.details(result)
        provenance(key)


def _show_mean(
    result: TestResult,
    column: str,
    reference: float,
    kind: str,
    rows: int,
    lacking: int,
    *,
    pooled: bool = False,
) -> None:
    mean = kind == m.ONE_T
    estimate, ci = result.estimate, result.estimate_ci
    metrics = []
    if estimate is not None and ci is not None:
        # the value and its interval with the same decimals ("6.45, 95% CI 6.350 to 6.550")
        around = SimpleNamespace(lower=ci.lower + reference, upper=ci.upper + reference)
        metrics.append(
            Metric(
                "Mean" if mean else "Typical value (pseudo-median)",
                *layout.with_interval(estimate + reference, around, 4),
            )
        )
    difference, bounds = layout.with_interval(estimate, ci)
    metrics += [
        Metric(
            f"Difference from {layout.num(reference, 6)}",
            ("+" if estimate is not None and estimate > 0 else "") + difference,
            f"{bounds} · p {layout.p_text(result.p_value)}",
        ),
        Metric(
            "Participants" if mean else "Participants who differ",
            f"{result.n_total:,}",
            f"of {rows:,} rows"
            + (
                f"; {lacking:,} lack a value" + (" that was not imputed" if pooled else "")
                if lacking > 0
                else ""
            ),
        ),
    ]
    words = ""
    if estimate is not None and np.isfinite(estimate):
        # the sentence repeats the headline's own figure
        what = f"The {'mean' if mean else 'typical value'} of {as_code(column)}"
        against = f"the reference of {layout.num(reference, 6)}"
        words = (
            f"{what} equals {against}"
            if float(estimate) == 0
            else f"{what} is {difference.lstrip('-')} {'below' if estimate < 0 else 'above'} {against}"
        ) + f" ({_named(result.name)})."
    layout.headline(metrics, words, len(result.warnings))
    layout.warn(result.warnings)


def _share_metrics(
    result: TestResult, with_it: int, counted: int, reference: float, what: str, where: str = ""
) -> None:
    ci = result.estimate_ci
    share = float("nan") if result.estimate is None else float(result.estimate)
    metrics = [
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
    ]
    if where:  # counted in the dataset: of a number typed, the field and "3 of 10" have said it
        metrics.append(Metric("Participants", f"{counted:,}", where))
    gap = 100 * (share - reference)
    layout.headline(
        metrics,
        "The share equals the reference."
        if abs(gap) < 0.05
        else f"The share is {layout.fixed(abs(gap), 1)} points {'below' if gap < 0 else 'above'} the "
        "reference.",
        len(result.warnings),
    )
    layout.warn(result.warnings)
    with st.expander("Details: the method"):
        st.caption("Method: " + " | ".join(result.references))
    help_popover(m.SHARE_READING)


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
            # its own key: the reference typed for a variable of the dataset became the calculator's
            key="prop_ref_typed",
            persist_state="page",
            help=HELP["reference_share"],
        )
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
        _share_metrics(result, int(with_it), int(counted), float(reference), "the characteristic")  # type: ignore[arg-type]
        layout.typed_line(f"From the numbers typed: {int(with_it):,} of {int(counted):,}.")  # type: ignore[arg-type]
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
    layout.depends("prop_ref", column, None)  # a reference share is that of one characteristic
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
    if layout.run_button(
        "prop_run", ready=bool(plan.steps), what="the exact binomial test", plan=plan
    ):
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
            # as the mean and the pairs say it: of how many rows, and how many lack a value
            f"of {len(df):,} rows"
            + (f"; {len(df) - counted:,} lack a value" if len(df) > counted else ""),
        )
        provenance("proportion")


# ----------------------------------------------------------------------------- an association within strata


def _levels_as_text(df: pd.DataFrame, column: str | None) -> list[str]:
    return sorted(df[column].dropna().astype(str).unique()) if column else []


def strata_choices(
    df: pd.DataFrame,
    *,
    outcome: str,
    event: Any,
    compare: str,
    reference: Any,
    stratum: str | None = None,
) -> dict[str, Any]:
    """
    The session values that open the stratified analysis on choices made elsewhere on the page: the 2x2 panel of
    the Analysis tab holds the outcome, its event, the two groups and the reference group, and its button opened
    this analysis with every field empty. They are the researcher's own choices, not a guess: the plan shows
    them before Run, and the variable to adjust for is left to choose.

    Each is carried only when this analysis offers it on ``df`` (the panel can compare two groups of a variable
    that holds more: such a variable is not offered here); what is not carried is left empty. ``stratum``: the
    variable to adjust for kept from an earlier use, let go when it is now the outcome or the groups.
    """
    offered = m.two_valued(df)
    chosen: dict[str, Any] = {"mh_outcome": None, "mh_exposure": None}
    if outcome in offered:
        chosen["mh_outcome"] = outcome
        events = _levels_as_text(df, outcome)
        if str(event) in events:
            chosen |= layout.given("mh_ev_level", (outcome, tuple(events)), str(event))
    if compare in offered and compare != outcome:
        chosen["mh_exposure"] = compare
        compared = _levels_as_text(df, compare)
        if str(reference) in compared:
            chosen |= layout.given("mh_reference", (compare, tuple(compared)), str(reference))
    if stratum is not None and stratum in (outcome, compare):
        chosen["mh_stratum"] = None
    return chosen


def _strata(df: pd.DataFrame, *, mi: Any, guided: bool) -> None:
    offered = m.two_valued(df)
    groups = m.strata(df)
    if len(offered) < 2 or not groups:
        empty_state(
            "Not enough columns for a stratified analysis",
            "It needs a yes/no outcome, an exposure in two values and a variable of 2 to 8 groups to adjust for. A "
            "column whose values are written several ways can be cleaned in Data > Clean.",
            links=(("clean", "Clean categories"), ("enrich", "Create new variables", "variables")),
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
    # in the left half, under what is compared: it spanned the width under two half rows
    c5, _ = st.columns(2)
    stratum = c5.selectbox(
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
    if layout.run_button(
        "mh_run", ready=bool(plan.steps), what="the stratified analysis", plan=plan
    ):
        try:
            found = mantel_haenszel(
                df,
                compare,  # type: ignore[arg-type]
                outcome,  # type: ignore[arg-type]
                stratum,  # type: ignore[arg-type]
                exposure_level=exposed,
                event_level=event,
            )
            remember(
                "mantel_haenszel",
                found,
                config=config,
                n_analysed=found.n,  # the rows of the tables: those with the three values
                columns=[compare, outcome, stratum],  # type: ignore[list-item]
            )
            mark_step("analyse", "Mantel-Haenszel analysis")
        except Exception as exc:
            error_card("The stratified analysis could not be computed.", exc)
    result = layout.result_place("mantel_haenszel", config, guided=guided)
    if result is not None:
        _show_strata(result, outcome, event, compare, exposed, reference, stratum, len(df))  # type: ignore[arg-type]
        provenance("mantel_haenszel")


def _as(value: Any, like: str) -> str:
    """``value`` with the decimals of ``like``, a number already written: an unadjusted ratio beside the adjusted
    one it is read against ("Unadjusted 0.7 and 0.804")."""
    return layout.num(value) if "n/a" in like else layout.fixed(value, len(like.partition(".")[2]))


def _ratio(value: Any) -> str:
    """A ratio to three significant figures: at two decimals, a lower bound of 0.996 read "1.00" beside p = 0.051."""
    return layout.num(value)


def _show_strata(
    res: Any,
    outcome: str,
    event: str,
    compare: str,
    exposed: str,
    reference: str,
    stratum: str,
    total: int,
) -> None:
    p = float("nan") if res.breslow_day_p is None else float(res.breslow_day_p)
    tested = bool(np.isfinite(p))
    informative = getattr(res, "n_tested", None)
    # the word that qualifies the combined ratio stands with it: the headline gave one ratio, and the line under it
    # said not to report it
    chip = (
        ("the strata differ: read each stratum", "caution")
        if tested and p < 0.05
        else None
        if tested
        else ("strata not tested", "neutral")
    )
    read = getattr(res, "n", None)
    rows = ""
    if read is not None:
        rows = f"; {read:,} of {total:,} rows" + (
            f", {total - read:,} lack a value" if total > read else ""
        )
    # each ratio with the decimals of its interval ("1.12, 95% CI 0.845 to 1.489"), and the unadjusted ones with
    # those of the adjusted ratios they are read against
    odds, odds_bounds = layout.with_interval(res.or_mh, res.or_mh_ci)
    risk, risk_bounds = layout.with_interval(res.rr_mh, res.rr_mh_ci)
    layout.headline(
        [
            Metric(
                f"Odds ratio, adjusted for {stratum}",
                odds,
                " · ".join(
                    part for part in (odds_bounds, f"p {layout.p_text(res.mh_test_p)}") if part
                ),
            ),
            Metric("Risk ratio, adjusted", risk, risk_bounds),
            Metric(
                "Unadjusted",
                f"{_as(res.crude_or, odds)} and {_as(res.crude_rr, risk)}",
                "odds ratio and risk ratio",
            ),
        ],
        f"{as_code(outcome)} = {as_code(event)}, {as_code(exposed)} against {as_code(reference)} "
        f"({as_code(compare)}), within {res.n_strata} strata of {as_code(stratum)}{rows}.",
        len(res.warnings),
        chip=chip,
    )
    # as the table writes them; a stratum whose ratio has no upper bound is one the test read: named, not left
    # out ("run from 0.458 to 0.861: they differ" stood over a fifth stratum, the one that differed)
    every = pd.to_numeric(res.per_stratum["odds_ratio"], errors="coerce").dropna()
    ratios, unbounded = every[np.isfinite(every)], int(np.isinf(every).sum())
    span = ""
    if len(ratios):
        span = (
            f"The odds ratios of the strata run from {layout.fixed(ratios.min(), 3)} to "
            f"{layout.fixed(ratios.max(), 3)}"
        )
        if unbounded:
            span += (
                f" in {len(ratios)} of the {len(ratios) + unbounded} strata tested; in the other"
                f"{'s' if unbounded > 1 else ''} the ratio has no upper bound (a cell of the table is empty)"
            )
        span += ": "
    if not tested:
        st.markdown(
            ":material/help: Whether the strata differ was not tested: "
            + (
                "fewer than two strata hold both groups and both outcomes."
                if informative is not None and informative < 2
                else "the test has no value on these strata (a combined odds ratio of 0, or without a value)."
            )
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
    # a stratum written with a hidden space reads "Metz␣", as in the lists
    from ui.components.levels import show_level

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
    shown["Stratum"] = [show_level(name) for name in shown["Stratum"]]
    for ratio in ("Odds ratio", "Risk ratio"):  # in words: a missing ratio was drawn "None"
        shown[ratio] = [
            layout.fixed(value, 3)
            if pd.notna(value) and np.isfinite(value)
            else ("no upper bound" if pd.notna(value) and value > 0 else "not estimable")
            for value in shown[ratio]
        ]
    # written as text for "not estimable", the ratios stood flush left beside counts aligned right
    right = {
        name: st.column_config.TextColumn(alignment="right")
        for name in ("Odds ratio", "Risk ratio")
    }
    st.dataframe(shown, hide_index=True, width="stretch", column_config=right)
    layout.warn(res.warnings)
    with st.expander("Details: the tests and the method"):
        st.caption(
            f"Test of the association over the strata (Cochran-Mantel-Haenszel, continuity-corrected): chi2 = "
            f"{layout.num(res.mh_test_statistic)}, p {layout.p_text(res.mh_test_p)}. "
            + (
                "Test that the odds ratios of the strata are alike (Breslow-Day): chi2 = "
                f"{layout.num(res.breslow_day_statistic)}, p {layout.p_text(res.breslow_day_p)}. "
                if tested
                else "The test that the odds ratios of the strata are alike (Breslow-Day) was not run. "
            )
            + "Intervals: Robins-Breslow-Greenland for the odds ratio, Greenland-Robins for the risk ratio."
        )
        st.caption("Method: " + " | ".join(res.references))
    help_popover(
        "The adjusted ratios combine the ratios found within each stratum. How much the variable adjusted for "
        "mattered is read on the risk ratio, adjusted against unadjusted: an odds ratio can move without any "
        "distortion (it is not collapsible). One combined ratio sums the strata up only when their ratios are "
        "alike. To adjust for several variables at once, use a logistic model (Survival & regression, Binary)."
    )


# ----------------------------------------------------------------------------- rates, from counts


def _count(label: str, key: str, column: Any, example: str, *, shown: bool = True) -> int | None:
    """``shown``: the label is drawn (the second row of a table of fields keeps it for screen readers only: its row
    names the group and the first row's labels head the columns)."""
    return column.number_input(
        label,
        label_visibility="visible" if shown else "collapsed",
        min_value=0,
        value=None,
        step=1,
        placeholder=example,
        key=key,
        persist_state="page",
        help=HELP["events"],
    )


def _time(label: str, key: str, column: Any, example: str, *, shown: bool = True) -> float | None:
    return column.number_input(
        label,
        label_visibility="visible" if shown else "collapsed",
        min_value=0.0,
        value=None,
        format="%g",
        placeholder=example,
        key=key,
        persist_state="page",
        help=HELP["person_time"],
    )


def _rate_metric(label: str, rate: Any, per: int, unit: str) -> Metric:
    """A rate with its exact interval; the numbers typed stand in the fields and in the line under the result."""
    # by the rule of every estimate with its interval: at two decimals, 3 events over 2,500,000 read "0.00, 95%
    # CI 0.00 to 0.00" per 1,000, and a rate of 12,500 "12500.00"
    scaled = SimpleNamespace(lower=rate.ci.lower * per, upper=rate.ci.upper * per)
    value, bounds = layout.with_interval(rate.rate * per, scaled)
    return Metric(label, value, f"per {per:,} {unit} · {bounds}")


def _more_or_fewer(ratio: float, times: str | None = None) -> str:
    """A ratio of counts in words; ``times``: the ratio as its headline writes it, repeated from 2 upwards ("7.5
    times as many" stood under "7.50")."""
    if ratio < 1:
        return f"{1 - ratio:.0%} fewer"
    if ratio >= 2:
        return f"{times or layout.num(ratio)} times as many"
    return f"{ratio - 1:.0%} more"


def _against_expected(ratio: float, times: str | None = None) -> str:
    """ "36% more events than expected", "7.50 times as many events as expected" (it read "as many events
    than")."""
    said = _more_or_fewer(ratio, times)
    return f"{said[:1].upper()}{said[1:]} events {'as' if ratio >= 2 else 'than'} expected"


def _rates(df: pd.DataFrame, *, mi: Any, guided: bool) -> None:
    calculation = layout.choose(
        "Calculate",
        m.CALCULATIONS,
        key="rate_calc",
        help=HELP["calculate"],
        default=m.TWO_RATES,
        depth=2,
    )
    if calculation == m.EXPECTED:
        _expected()
        return
    two = calculation == m.TWO_RATES
    if two:
        label, e_col, t_col = st.columns([1.2, 2, 2], vertical_alignment="bottom")
        label.markdown("**Group compared**")
    else:  # one group: no row to name, its two fields in the two halves
        e_col, t_col = st.columns(2)
    events = _count("Events", "rate_e1", e_col, "for example 15")
    person_time = _time("Person-time at risk", "rate_t1", t_col, "for example 1200")
    reference_events = reference_time = None
    if two:
        label, e_col, t_col = st.columns([1.2, 2, 2], vertical_alignment="bottom")
        label.markdown("**Reference group**")
        reference_events = _count(
            "Events (reference group)", "rate_e0", e_col, "for example 4", shown=False
        )
        reference_time = _time(
            "Person-time at risk (reference group)",
            "rate_t0",
            t_col,
            "for example 1000",
            shown=False,
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
        second = (
            incidence_rate(int(reference_events), float(reference_time), per=per)  # type: ignore[arg-type]
            if two
            else None
        )
        # no event in either group: two rates of 0, and no ratio of them
        ratio = (
            rate_ratio(first.events, first.person_time, second.events, second.person_time, per=per)
            if second is not None and (first.events or second.events)
            else None
        )
    except Exception as exc:
        error_card("The rates could not be computed.", exc)
        return
    typed = (
        f"From the numbers typed: {first.events:,} event{'s' if first.events != 1 else ''} over "
        f"{layout.num(first.person_time, 6)} {unit}"
    )
    if second is None:
        layout.headline([_rate_metric("Rate", first, per, unit)])
    else:
        typed += f" against {second.events:,} over {layout.num(second.person_time, 6)}"
        rates = [
            _rate_metric("Rate, group compared", first, per, unit),
            _rate_metric("Rate, reference group", second, per, unit),
        ]
        if ratio is None:
            layout.headline(rates)  # the line above the result has said why there is no ratio
        else:
            head = _ratio_metric(ratio)
            layout.headline([head, *rates], _ratio_words(ratio, head.value))
    # the methods are folded, as a test's; the reading and where the numbers came from close the result
    with st.expander(
        "Details: the method" if ratio is None else "Details: the rate difference and the method"
    ):
        if ratio is not None:
            st.caption(
                f"Rate difference: {layout.num(ratio.rate_difference * per)} per {per:,} {unit} (no interval is given for "
                f"it). The ratio's test and interval: {ratio.test}; the rates' intervals: exact Poisson."
            )
        st.caption("Method: " + " | ".join((ratio or first).references))
    help_popover(m.RATE_READING if ratio is None else m.RATIO_READING)
    layout.typed_line(typed + ".")


def _ratio_metric(ratio: Any) -> Metric:
    """The rate ratio with its exact interval and test; without an event in the reference group it has no upper
    bound, and its lower end is what the data say."""
    bounded = bool(np.isfinite(ratio.ratio) and np.isfinite(ratio.ci.upper))
    value, bounds = layout.with_interval(ratio.ratio, ratio.ci)
    return Metric(
        "Rate ratio",
        value if bounded else "no upper bound",
        (
            bounds
            if bounded
            else f"at least {_ratio(ratio.ci.lower)} (the lower end of the 95% interval)"
        )
        + f" · p {layout.p_text(ratio.p_value)}",
    )


def _ratio_words(ratio: Any, written: str) -> str:
    """The rate ratio in a sentence, as the standardised ratio's: which group has the higher rate, and by how
    much."""
    if not np.isfinite(ratio.ratio):
        return ""  # no event in the reference group: the line above the result has said it
    if ratio.ratio == 0:
        return "No event in the group compared: the ratio is 0, and its interval says how large it may be."
    # the headline's own figure: "3.1 times the rate" stood under "3.125"
    return f"The group compared has {written} times the rate of the reference group."


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
    value, bounds = layout.with_interval(found.ratio, found.ci)
    layout.headline(
        [
            Metric(
                m.RATIO_NAMES[kind],
                value,
                f"{bounds} · p {layout.p_text(found.p_value)} (exact Poisson)",
            )
        ],
        f"{_against_expected(found.ratio, value)}; the interval runs from {_more_or_fewer(found.ci.lower)} to "
        f"{_more_or_fewer(found.ci.upper)}.",
    )
    with st.expander("Details: the method"):
        st.caption("Method: " + " | ".join(found.references))
    help_popover(m.EXPECTED_READING)
    layout.typed_line(
        f"From the numbers typed: {found.observed:,} observed against {layout.num(found.expected, 6)} expected."
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


__all__ = ["QUESTION_KEY", "category_order", "run_paired_and_rates", "strata_choices"]
