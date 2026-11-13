"""
What the analysis tabs share on the screen (Hypothesis tests & epidemiology: Diagnostic accuracy & agreement; Paired,
rates & stratified), in the pattern of Survival & regression: the selector of one analysis among a tab's entries, the
plan shown before a run, one Run, the headline of a result (the estimate with its interval first, then the test, then
the counts), the place of a result that is not there yet, the line of an analysis under imputed data and the line of
a calculator's result.

One analysis is open at a time; each keeps its choices (``persist_state="page"``) and its result (ui.results) while
another is open. A link can open an entry: the page declares its tabs and entries (``ui.nav.PageDef.tabs``), and the
selectors read the link's labels as their default.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import streamlit as st

from app_pages.clean_categories import take_done
from app_pages.clean_dialog import clean_actions
from biostats._types import TestResult
from biostats.assumptions import read_assumptions
from ui import nav
from ui.components import glossary_popover, help_popover, status_chip
from ui.components.text import plain
from ui.guidance.interpretation import how_to_read
from ui.guidance.measures import HELP, Entry, number
from ui.guidance.measures import decimals as _decimals
from ui.guidance.outcomes import ModelPlan
from ui.results import MISSING, STALE_CONFIG, recall, status, stored_settings
from ui.state import get_state

PAGE = "analyze"  # the page of the two tabs, as the navigation names it
TYPED = "Not from the dataset: the analysis population and the imputation do not apply."
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


@dataclass(frozen=True)
class Metric:
    label: str
    value: str
    caption: str = ""
    help: str | None = None


def p_text(p: Any) -> str:
    """A p-value as a sentence writes it after "p": "< 0.001", "= 0.017"; "n/a" when it was not computed. Next to
    0.05 a fourth decimal tells 0.0499 from 0.0501, which both read "= 0.050"."""
    if p is None or not math.isfinite(float(p)):
        return "n/a"
    if p < 0.001:
        return "< 0.001"
    return f"= {p:.4f}" if 0.0495 <= p < 0.0505 else f"= {p:.3f}"


def num(value: Any, digits: int = 3) -> str:
    """A number of a result, by the rule the plans write theirs with (ui.guidance.measures.number): ``digits``
    significant figures for its decimals, thousands grouped, the whole part kept, no exponent; "n/a" without a
    value."""
    return number(value, digits)


def span(lower: Any, upper: Any, digits: int = 3, *, unit: str = "") -> str:
    """ "0.902 to 0.953": two bounds written with the same decimals, those the bound nearer 0 needs for ``digits``
    significant figures (each rounded on its own, an interval read "-9 to -7.38"); a percentage with one decimal."""
    if any(v is None or not math.isfinite(float(v)) for v in (lower, upper)):
        return f"{num(lower, digits)}{unit} to {num(upper, digits)}{unit}"
    low, high = float(lower), float(upper)
    decimals = 1 if unit == "%" else min(6, max(_decimals(low, digits), _decimals(high, digits)))
    return f"{low:,.{decimals}f}{unit} to {high:,.{decimals}f}{unit}"


def interval(ci: Any, digits: int = 3, *, scale: float = 1.0, unit: str = "") -> str:
    """ "95% CI 0.902 to 0.953" from an interval object; "" without one, or without bounds ("95% CI n/a% to n/a%"
    stood beside a predictive value that has no value)."""
    if ci is None or not all(
        bound is not None and math.isfinite(float(bound)) for bound in (ci.lower, ci.upper)
    ):
        return ""
    return f"95% CI {span(ci.lower * scale, ci.upper * scale, digits, unit=unit)}"


def with_interval(estimate: Any, ci: Any, digits: int = 3) -> tuple[str, str]:
    """An estimate and its interval written with the same decimals ("3.125" and "95% CI 0.996 to 12.935"): rounded
    each on its own, a typical change read "-5.5, 95% CI -6.00 to -4.50" and a rate ratio "3.12, 0.996 to 12.935"."""
    values = [estimate, getattr(ci, "lower", None), getattr(ci, "upper", None)]
    if any(v is None or not math.isfinite(float(v)) for v in values):
        return num(estimate, digits), interval(ci, digits)
    decimals = min(6, max(_decimals(float(v), digits) for v in values))
    return (
        f"{float(estimate):,.{decimals}f}",
        f"95% CI {float(ci.lower):,.{decimals}f} to {float(ci.upper):,.{decimals}f}",
    )


def alike(*values: Any, digits: int = 3) -> list[str]:
    """Numbers that are read together, written with the same decimals: those the one nearest 0 needs for ``digits``
    significant figures (two limits of agreement read "-8.19 to 20.17" over "95% CI 19.4 to 21.0", and "20.2" in the
    figure's legend). A number without a value reads "n/a"."""
    known = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    decimals = min(6, max((_decimals(v, digits) for v in known), default=0))
    return [
        f"{float(v):,.{decimals}f}" if v is not None and math.isfinite(float(v)) else num(v, digits)
        for v in values
    ]


def percent(value: Any, digits: int = 1) -> str:
    return (
        "n/a" if value is None or not math.isfinite(float(value)) else f"{100 * value:.{digits}f}%"
    )


def headline(
    metrics: Sequence[Metric],
    words: str = "",
    warnings: int = 0,
    *,
    chip: tuple[str, str] | None = None,
) -> None:
    """The result first: the estimate with its interval, the test and the counts, then one sentence saying what was
    compared and in which direction."""
    with st.container(border=True):
        for start in range(0, len(metrics), 4):
            row = metrics[start : start + 4]
            for column, metric in zip(st.columns(len(row)), row, strict=True):
                column.metric(metric.label, metric.value, help=metric.help)
                if metric.caption:
                    column.caption(metric.caption)
        if words or warnings or chip:
            with st.container(horizontal=True, vertical_alignment="center"):
                if words:
                    st.caption(words)
                # what qualifies the number above it (label, status): read with it, not after it
                if chip:
                    status_chip(*chip)
                if warnings:
                    status_chip(f"{warnings} warning{'s' if warnings > 1 else ''} below", "caution")


# ----------------------------------------------------------------------------- one analysis at a time


def linked(depth: int, labels: Mapping[str, str]) -> str | None:
    """The id a link to this page names at ``depth`` of its path (the tab is 0, its entry 1, the entry's second choice
    2), among ``labels`` (id to label); None when the page was not opened by such a link."""
    opened = nav.opened_tabs(PAGE, st.query_params.get(nav.TAB_PARAM))
    if len(opened) <= depth:
        return None
    return next((key for key, label in labels.items() if label == opened[depth]), None)


def _start(key: str, value: str) -> dict[str, str]:
    """A selector's ``default``, given only while it holds no value: a button of the page may have set its key (the
    orientation, a suggestion), and a widget given both a default and a value set in the session is reported."""
    return {} if key in st.session_state else {"default": value}


def choose_entry(entries: Sequence[Entry], *, key: str, guided: bool) -> Entry:
    """The Question selector of a tab: one entry is open at a time. Under it, what the entry answers and its terms.
    (The way to the test chooser is in the map above the tabs: said again under every entry, 250 px lower, it was a
    row of its own that named nothing new.)"""
    labels = {entry.id: entry.label for entry in entries}
    chosen = st.segmented_control(
        "Question",
        list(labels),
        format_func=lambda entry: labels[entry],
        required=True,
        key=key,
        persist_state="page",
        wrap=True,
        help=HELP["question"],
        **_start(key, linked(1, labels) or entries[0].id),
    )
    entry = next((e for e in entries if e.id == chosen), entries[0])
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.caption(entry.gives, width="stretch")  # wraps beside Terms
        glossary_popover(*entry.terms, label="Terms")
    return entry


def choose(
    label: str,
    options: Mapping[str, str],
    *,
    key: str,
    help: str,
    default: str | None = None,
    depth: int | None = None,
) -> str:
    """A second choice of an entry (the kind of measurement, what a calculator computes): ids kept in the state, their
    labels shown; a link names it at ``depth`` of its path."""
    start = (
        (linked(depth, options) if depth is not None else None) or default or next(iter(options))
    )
    chosen = st.segmented_control(
        label,
        list(options),
        format_func=lambda option: options[option],
        required=True,
        key=key,
        persist_state="page",
        help=help,
        **_start(key, start),
    )
    return chosen if chosen in options else start


def depends(key: str, on: Any, value: Any) -> None:
    """
    Give the widget ``key`` the value offered for what it depends on (``on``: the column chosen above it, with its
    levels), once each time that changes: the level that says "present" for a reference standard, no cut-off for
    another test, no reference value for another measurement. Call it before the widget is drawn, and give the
    widget no default of its own.

    The value goes through the widget's state because that is what reaches both sides. A default (``index=``,
    ``value=``) is read by the script, not by a widget the browser already shows: the field went on reading "Choose
    the value" over a result computed for Yes. A value cleared by removing its key stayed on the screen the same
    way. And a value kept from another column was reported ("PPV at prevalence 0.502" for a condition 10.9% of
    participants have).
    """
    marker = _mark(key)
    if marker not in st.session_state or st.session_state[marker] != on:
        st.session_state[marker] = on
        st.session_state[key] = value


def _mark(key: str) -> str:
    """Where :func:`depends` keeps what the value of the widget ``key`` was given for."""
    return f"{key}::for"


def given(key: str, on: Any, value: Any) -> dict[str, Any]:
    """
    The session values that give the widget ``key`` a value chosen elsewhere (a button of another tab) for what it
    depends on (``on``, as :func:`depends` is called with on the page): the value, and the mark ``depends`` keeps,
    so that the field shows the choice carried and not its own offer. To be set before the widget is drawn (the
    ``on_click`` of the button).
    """
    return {key: value, _mark(key): on}


def start_at(key: str, options: Mapping[str, str], depth: int) -> dict[str, int]:
    """
    A radio's ``index``, given only while it holds no value: the option a link to this page names at ``depth`` of
    its path (the summary of a signed-rank test, which "Which test?" recommends: the page opened on the mean, and
    Run gave a test that was not the one recommended). A choice made afterwards is kept.
    """
    if key in st.session_state:
        return {}
    named = linked(depth, options)
    return {} if named is None else {"index": list(options).index(named)}


def kept(name: str, key: tuple, compute: Callable[[], Any]) -> Any:
    """A plan kept until the data, the analysis population or ``key`` (the panel's choices) changes: computed at every
    rerun, it would be read again at each click."""
    state = get_state()
    identity = state.identity()
    if identity is None:
        return compute()
    return state.session.memo(name, (identity.population_fp, key), compute)


def plan_box(plan: ModelPlan, *, clean: Mapping[str, Mapping[str, Any]] | None = None) -> None:
    """
    What Run will do, checked on the data: a status, each step with its rows, the advice. ``clean`` gives, for a
    variable whose values are written several ways, the widget it was chosen in (the arguments of
    app_pages.clean_dialog.clean_actions): it is then cleaned here, and replaced by its copy.
    """
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown("**Before you run: the plan**")
            status_chip(*_STATUS[plan.status])
        # the rows once, on the first step, when every step reads the same: after each they doubled the brackets
        # ("(101 with the condition, 827 without) (928 of 928 rows)")
        alike = len({step.used for step in plan.steps}) == 1
        for position, step in enumerate(plan.steps):
            rows = "" if alike and position else f": {step.used:,} of {plan.rows:,} rows"
            st.markdown(f"- {step.text}{rows}")
        acted: set[str] = set()
        for item in plan.advice:
            st.markdown(f"{_ICON.get(item.level, '')} {item.text}")
            column = getattr(item, "column", None)
            if column and clean and column in clean and column not in acted:
                acted.add(column)
                clean_actions(column, **clean[column])
        if clean:
            for done in take_done():  # what Apply did in the dialog, said once
                st.success(done, icon=":material/check:")


def checks(plan: ModelPlan) -> None:
    """With Guidance off the plan is not drawn: what would make the result wrong, or impossible, is still said."""
    for item in plan.advice:
        if item.level == "violated":
            st.warning(item.text, icon=":material/error:")


def run_button(key: str, *, ready: bool, what: str, plan: ModelPlan | None = None) -> bool:
    """The one Run of an analysis of the dataset. Disabled only while nothing can be computed: advice never blocks.
    Its help shows while it is disabled, and says what the plan says is missing (with Guidance off the plan is not
    drawn, and nothing on the screen said why Run was greyed). On a button just clicked a help stays open as long
    as the button keeps the focus, and lay over the last lines of the plan: an enabled Run has none."""
    missing = (
        next((item.text for item in plan.advice), "") if plan is not None and not ready else ""
    )
    return st.button(
        "Run analysis",
        type="primary",
        key=key,
        disabled=not ready,
        help=None
        if ready
        else missing or f"Nothing to run yet: complete the choices above to run {what}.",
    )


def result_place(
    key: str,
    config: Any,
    *,
    guided: bool,
    silent: bool = False,
    words: Mapping[str, str] | None = None,
) -> Any | None:
    """
    The stored result of these choices, or None with a word on why none is shown: nothing was run yet (Guidance on),
    or the last result belongs to other choices (it did not vanish). A change of the data or of the population is
    announced by ``recall``. ``silent``: the second result of one Run, whose first has said it. ``words``: the
    page's labels for the option values among the settings ("names" is "Names without an order" on the screen).
    """
    value = recall(key, config=config)
    if value is not None or silent:
        return value
    state = status(key, config=config)
    if state == STALE_CONFIG:
        found = stored_settings(key)
        if found is not None:
            said = ", ".join((words or {}).get(part, part) for part in found[0].split(", "))
            st.caption(
                f":material/history: Last result here: {plain(said, limit=200)} ({found[1]}). Run analysis to "
                "compute it for the choices above."
            )
    elif state == MISSING and guided:
        st.caption(
            ":material/info: The result appears here and stays while these choices are unchanged."
        )
    return None


def imputed_line(text: str) -> None:
    """What an analysis reads while imputed data are selected in the sidebar: pooled, or which observed rows."""
    st.caption(f":material/stacks: {text}")


def typed_line(numbers: str) -> None:
    """Under a calculator's result, in place of the line "computed on": where its numbers came from."""
    st.caption(f":material/edit_note: {numbers} {TYPED}")


def advice_lines(advice: Sequence[Any]) -> None:
    """The checks of the numbers typed in a calculator."""
    for item in advice:
        st.markdown(f"{_ICON.get(item.level, '')} {item.text}")


# ----------------------------------------------------------------------------- a test's result


def warn(warnings: Sequence[str]) -> None:
    """A result's own warnings, each once (the result card never showed them: 34 pairs left out by a rank test, fewer
    than 10 pairs that changed)."""
    for text in dict.fromkeys(warnings):
        st.warning(plain(text))


def details(
    result: TestResult,
    *,
    label: str = "Details: the test, its checks and the method",
    more: str = "",
) -> None:
    """Folded under a test's headline: the statistic, the effect sizes, the assumption checks (diagnostics, not
    gates) and the method. ``more``: what is to know of this analysis whatever the data, said once where the result
    is read (in the plan it was said again at every run)."""
    with st.expander(label):
        statistic = f"{result.statistic_label or 'statistic'} = {num(result.statistic)}"
        if result.df:
            statistic += ", df = " + ", ".join(num(d, 4) for d in result.df)
        st.caption(f"{plain(result.name)}: {statistic}, p {p_text(result.p_value)}.")
        if result.estimate_ci is not None and result.estimate_ci.method:
            st.caption(f"Interval of the estimate: {result.estimate_ci.method}.")
        if result.effect_sizes:
            parts = []
            for effect in result.effect_sizes:
                text = f"{effect.name} = {num(effect.estimate)}"
                if effect.ci is not None:
                    text += f" (95% CI {effect.ci.as_text(2)})"
                parts.append(text)
            st.caption("Effect sizes: " + "; ".join(parts) + ".")
        if result.mi is not None:
            info = result.mi
            st.caption(
                f":material/stacks: Pooled over {info.m} imputed datasets with Rubin's rules: df = {num(info.df, 4)}, "
                f"FMI = {info.fmi:.3f}, RIV = {info.riv:.3f}."
            )
        if result.assumptions:
            st.caption("**Assumption checks**: diagnostics, not gates; the test ran regardless.")
            for reading in read_assumptions(result):
                chip, text = st.columns([2, 5], vertical_alignment="center")
                with chip:
                    status_chip(
                        reading.check, "neutral" if reading.status == "na" else reading.status
                    )
                text.caption(reading.sentence)
        if result.references:
            st.caption("Method: " + " | ".join(result.references))
    help_popover(how_to_read(result) + (f"\n\n{more}" if more else ""))


__all__ = [
    "PAGE",
    "TYPED",
    "Metric",
    "advice_lines",
    "alike",
    "checks",
    "choose",
    "choose_entry",
    "depends",
    "details",
    "given",
    "headline",
    "imputed_line",
    "interval",
    "kept",
    "linked",
    "num",
    "p_text",
    "percent",
    "plan_box",
    "result_place",
    "run_button",
    "span",
    "start_at",
    "typed_line",
    "warn",
    "with_interval",
]
