"""
Diagnostic accuracy & agreement, a tab of Analyze > Hypothesis tests & epidemiology: three questions, one open at a
time (ui.guidance.measures): the accuracy of a test against a reference standard (sensitivity, specificity, the ROC
curve and its AUC), the agreement of two measurements (Bland-Altman limits and the ICC, on one set of columns), and the
agreement of two raters on categories (Cohen's kappa).

Each follows the page pattern of Survival & regression (app_pages.analysis_layout): nothing is preselected, the plan
says what Run will do, one click runs it, and the result reads the estimate first. What each analysis computes is a
function of its own here (``accuracy_of``, ``kappa_of``), so that the numbers are checked without the page.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_pages import analysis_layout as layout
from app_pages.analysis_guidance_ui import guidance_on
from app_pages.analysis_layout import Metric
from biostats._types import CI, YOUDEN_BOOTSTRAP
from biostats.agreement import SLOPE_READING, SLOPE_WARNING, bland_altman
from biostats.diagnostic import (
    YOUDEN_MIN_CLASS,
    confusion_at,
    delong_auc,
    diagnostic_accuracy,
    youden_cutoff,
)
from biostats.reliability import (
    NEAR_MISSES,
    cohens_kappa,
    icc,
    koo_li,
    landis_koch,
    rater_table,
)
from core.registry import as_code
from ui.components import empty_state, error_card, help_popover
from ui.guidance import measures as m
from ui.guidance.measures import HELP
from ui.progress import mark_step
from ui.results import provenance, remember
from ui.theme import OKABE_ITO

QUESTION_KEY = "diag_question"
MUTED = "#6B7C86"  # the line of chance: readable on the light and on the dark surface alike
BAND_ALPHA = 0.25  # at 0.12 the intervals of the limits were lost on the dark surface
OBSERVED = "Imputed data are selected; this analysis reads the observed data: "


def _rgba(colour: str, alpha: float) -> str:
    r, g, b = (int(colour[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


# ----------------------------------------------------------------------------- accuracy of a test


def _as_measured(found: dict) -> dict:
    """A cut-off found on the reversed marker (lower values pointing to the condition), on the marker's own scale."""
    shown = dict(found)
    shown["cutoff"] = -found["cutoff"]
    ci = found.get("ci")
    if ci is not None:
        shown["ci"] = CI(lower=-ci.upper, upper=-ci.lower, level=ci.level, method=ci.method)
    return shown


def accuracy_of(
    df: pd.DataFrame,
    outcome: str,
    present: Any,
    score: str,
    direction: str,
    mode: str,
    cutoff: float | None,
    prevalence: float | None,
) -> dict[str, Any]:
    """
    The accuracy of ``score`` against ``outcome`` = ``present`` on the rows where both are recorded: the AUC always;
    with a cut-off (fixed beforehand, or found on these data by Youden's J), the 2x2 table and its measures.

    ``direction`` says which way the test points. For "lower", the same functions read the reversed marker, and the
    cut-off is given back on the marker's own scale (positive when the test is at most the cut-off): a marker that
    falls with the condition had no result (the cut-off found called everyone positive).
    """
    clean = df[[outcome, score]].dropna()
    clean = clean[clean[score].apply(lambda v: isinstance(v, int | float | np.number))]
    if int(clean[outcome].nunique()) != 2:
        raise ValueError(
            "the reference standard must have exactly two levels after dropping missing values"
        )
    lower = direction == m.LOWER
    truth = clean[outcome].tolist()
    marker = (-clean[score]).tolist() if lower else clean[score].tolist()
    auc = delong_auc(truth, marker, pos_label=present)
    youden: dict | None = None
    at: float | None = None  # on the scale the functions read: the reversed marker's for "lower"
    if mode == m.YOUDEN:
        found = youden_cutoff(
            truth,
            marker,
            pos_label=present,
            **(
                YOUDEN_BOOTSTRAP.arguments()
                if range_drawn(clean, outcome, present)
                else {"n_boot": 0}
            ),
        )
        at = float(found["cutoff"])
        youden = _as_measured(found) if lower else found
    elif mode == m.FIXED:
        if cutoff is None:
            raise ValueError("a cut-off fixed beforehand is a number to type")
        at = -float(cutoff) if lower else float(cutoff)
    acc = None
    if at is not None:
        tp, fp, fn, tn = confusion_at(truth, marker, at, pos_label=present)
        acc = diagnostic_accuracy(
            tp, fp, fn, tn, prevalence=None if prevalence is None else float(prevalence)
        )
    shown = None if at is None else (-at if lower else at)
    point = None if acc is None else (1 - acc.specificity, acc.sensitivity)
    with_it = np.asarray([v == present for v in truth])
    values = np.asarray(marker, dtype=float)
    cautions: list[str] = []
    if auc["ci"].upper < 0.5:
        # in the words of the direction asked above: the functions read the reversed marker for "lower", and their
        # own caution then said the opposite ("higher values go with the other level") and advised to take the
        # other level as positive, which gives the accuracy for the absence of the condition
        other = m.HIGHER if lower else m.LOWER
        cautions.append(
            f"These data contradict the direction chosen: read with {direction} values pointing to the condition, "
            f"the area under the curve is {auc['auc']:.3f}, below 0.5. Choose {other} above, or check the value "
            "that means the condition is present."
        )
    return {
        "auc": auc,
        "pairs": _pairs(with_it, values),
        "cautions": cautions,
        "youden": youden,
        "cutoff": shown,
        "acc": acc,
        "outcome": outcome,
        "score": score,
        "pos_label": present,
        "mode": mode,
        "n": len(clean),
        "direction": direction,
        "prevalence": None if prevalence is None else float(prevalence),
        "fig": _roc_figure(
            with_it,
            values,
            auc,
            point,
            None if shown is None else f"cut-off {_bound(lower)} {layout.num(shown)}",
        ),
    }


def _pairs(with_it: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    """Of every pair of one participant with the condition and one without: the shares in which the one with it
    has the value pointing further to the condition, the same value, the other. The AUC is the first plus half
    the second: with ties, "in 66 of 100 pairs the higher value" was said of 51."""
    ill, well = np.sort(values[with_it]), np.sort(values[~with_it])
    total = len(ill) * len(well)
    if not total:
        return float("nan"), float("nan"), float("nan")
    below = np.searchsorted(
        well, ill, side="left"
    )  # those without it strictly below each one with it
    upto = np.searchsorted(well, ill, side="right")
    further, same = float(below.sum()) / total, float((upto - below).sum()) / total
    return further, same, 1.0 - further - same


def _in_100(share: float) -> str:
    """A share as "51 of 100 pairs" is said: never "100" for less than all, never "0" for some."""
    count = round(100 * share)
    if count == 100 and share < 1:
        return "more than 99"
    if count == 0 and share > 0:
        return "fewer than 1"
    return str(count)


def range_drawn(clean: pd.DataFrame, outcome: str, present: Any) -> bool:
    """Whether the range of a cut-off found on the data is drawn at random: only when each group holds enough rows.
    The draws are then part of the result's settings: a result drawn with another seed is another result."""
    positives = int((clean[outcome] == present).sum())
    return min(positives, len(clean) - positives) >= YOUDEN_MIN_CLASS


def _bound(lower: bool) -> str:
    return "at most" if lower else "at least"


def _roc_figure(
    truth: np.ndarray,
    marker: np.ndarray,
    auc: dict,
    point: tuple[float, float] | None,
    label: str | None,
) -> go.Figure:
    """The ROC curve, square, with the point of the 2x2 table when a cut-off is set (the table's own sensitivity and
    specificity: the nearest threshold of the curve was drawn, which is another point for a typed cut-off)."""
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(truth.astype(int), marker)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC curve (AUC {auc['auc']:.3f})",
            line={"color": OKABE_ITO[0], "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="no better than chance",
            line={"dash": "dash", "color": MUTED},
        )
    )
    if point is not None:
        fig.add_trace(
            go.Scatter(
                x=[point[0]],
                y=[point[1]],
                mode="markers",
                marker={"size": 12, "color": OKABE_ITO[1]},
                name=label or "cut-off",
            )
        )
    fig.update_layout(
        xaxis={
            "title": "1 - specificity (positives among those without the condition)",
            "range": [0, 1],
            "constrain": "domain",
        },
        yaxis={"title": "sensitivity", "range": [0, 1.02], "scaleanchor": "x", "scaleratio": 1},
        height=460,
        legend={"orientation": "h", "yanchor": "top", "y": -0.22, "x": 0},
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
    )
    return fig


def _accuracy(df: pd.DataFrame, *, mi: Any, guided: bool) -> None:
    standards, scores = m.two_valued(df), m.measurements(df)
    if not standards or not scores:
        lacking = "column with two values" if not standards else "numeric column"
        empty_state(
            f"No {lacking}",
            "The accuracy of a test needs a reference standard in two values (the condition present or absent) and "
            "a numeric score or marker. A column whose two values are written several ways can be cleaned in Data "
            "> Clean.",
            links=(("clean", "Clean categories"), ("enrich", "Create new variables", "variables")),
        )
        return
    c1, c2, c3 = st.columns(3)
    outcome = c1.selectbox(
        "Reference standard",
        standards,
        index=None,
        placeholder="Choose a column with two values",
        key="diag_outcome",
        persist_state="page",
        help=HELP["standard"],
    )
    levels = m.levels_of(df[outcome]) if outcome else []
    guess = m.present_value(levels)
    layout.depends("diag_pos", (outcome, tuple(map(str, levels))), guess)
    present = c2.selectbox(
        "Condition present when it is",
        levels,
        index=None,
        format_func=str,
        placeholder="Choose the value" if outcome else "Choose the reference standard first",
        disabled=not outcome,
        key="diag_pos",
        persist_state="page",
        help=HELP["present"],
    )
    score = c3.selectbox(
        "Test or score assessed (index test)",
        [c for c in scores if c != outcome],
        index=None,
        placeholder="Choose a numeric column",
        key="diag_score",
        persist_state="page",
        help=HELP["index"],
    )
    d1, d2 = st.columns([1, 2])
    direction = d1.radio(
        "The condition is more likely when the test is",
        [m.HIGHER, m.LOWER],
        index=None,
        horizontal=True,
        key="diag_direction",
        persist_state="page",
        help=HELP["direction"],
    )
    medians = (
        m.group_medians(df, outcome, present, score)
        if outcome and score and present is not None
        else None
    )
    if medians is not None:
        d1.caption(
            f"In these data: median {layout.num(medians[0], 4)} with the condition, {layout.num(medians[1], 4)} without."
        )
    mode = d2.radio(
        "Cut-off for a positive test",
        list(m.CUTOFFS),
        format_func=lambda value: m.CUTOFFS[value],
        horizontal=True,
        key="diag_mode",
        persist_state="page",
        help=HELP["cutoff"],
    )
    lower = direction == m.LOWER
    layout.depends("diag_cut", score, None)  # a cut-off is in the unit of its test
    cutoff = None
    if mode == m.FIXED:
        cutoff = d2.number_input(
            f"Positive when {score or 'the test'} is {_bound(lower)}"
            if direction
            else f"Cut-off fixed beforehand, in the unit of {score or 'the test'}",
            value=None,
            placeholder="a value fixed beforehand",
            format="%g",
            key="diag_cut",
            persist_state="page",
            help=HELP["cutoff_value"],
        )
        if score:
            held = pd.to_numeric(df[score], errors="coerce").dropna()
            if len(held):
                d2.caption(
                    f"{as_code(score)} runs from {layout.num(float(held.min()), 4)} to "
                    f"{layout.num(float(held.max()), 4)} (median {layout.num(float(held.median()), 4)})."
                )
    share = None if medians is None else medians[2] / (medians[2] + medians[3])
    layout.depends("diag_prev", outcome, None)  # a prevalence is that of one condition
    with st.expander("More options"):
        prevalence = st.number_input(
            "Prevalence where the test will be used (optional)",
            min_value=0.0,
            max_value=1.0,
            value=None,
            step=0.01,
            placeholder="as in these data" + (f": {share:.1%}" if share is not None else ""),
            key="diag_prev",
            persist_state="page",
            help=HELP["prevalence"],
        )
    choices = (
        outcome,
        None if present is None else str(present),
        score,
        direction,
        mode,
        cutoff,
        prevalence,
    )
    plan = layout.kept(
        "diag_plan",
        choices,
        lambda: m.plan_accuracy(df, outcome, present, score, direction, mode, cutoff, prevalence),
    )
    if guided:
        layout.plan_box(plan)
    else:
        layout.checks(plan)
    if mi is not None:
        layout.imputed_line(
            OBSERVED
            + "the rows where the reference standard and the test are both recorded. Not pooled in this version."
        )
    ready = bool(plan.steps)
    drawn = bool(
        ready and mode == m.YOUDEN and range_drawn(df[[outcome, score]].dropna(), outcome, present)  # type: ignore[index]
    )
    # the settings in use only: a cut-off typed for another mode is no part of this result
    config = (
        outcome,
        score,
        None if present is None else str(present),
        direction,
        mode,
        cutoff if mode == m.FIXED else None,
        prevalence,
        *((YOUDEN_BOOTSTRAP.setting,) if drawn else ()),
    )
    if layout.run_button("diag_run", ready=ready, what="the accuracy of the test", plan=plan):
        try:
            result = accuracy_of(df, outcome, present, score, direction, mode, cutoff, prevalence)  # type: ignore[arg-type]
            remember(
                "diagnostic",
                result,
                config=config,
                n_analysed=result["n"],
                columns=[outcome, score],  # type: ignore[list-item]
            )
            mark_step("analyse", "diagnostic accuracy")
        except Exception as exc:
            error_card("The accuracy of the test could not be computed.", exc)
    result = layout.result_place("diagnostic", config, guided=guided, words=m.CUTOFFS)
    if result is not None:
        _show_accuracy(result)
        provenance("diagnostic")


def _share(value: Any, ci: Any) -> str:
    """A share with its interval, as a sentence writes it: "40.8% (95% CI 34.5% to 47.6%)", or the share alone when
    it has no interval ("n/a (95% CI n/a% to n/a%)" stood for a predictive value without a value)."""
    bounds = layout.interval(ci, scale=100, unit="%")
    return layout.percent(value) + (f" ({bounds})" if bounds else "")


def _measure(name: str, value: Any, ci: Any, *, share: bool) -> tuple[str, str, str]:
    """A row of the table of every measure: a share as a percentage, as the headline writes it (the headline read
    86.1% and the table 0.861), a ratio to three significant figures; its interval with the same decimals."""
    known = value is not None and bool(np.isfinite(float(value)))
    bounded = ci is not None and all(np.isfinite(float(b)) for b in (ci.lower, ci.upper))
    if share:
        return (
            name,
            layout.percent(value) if known else "n/a",
            layout.span(100 * ci.lower, 100 * ci.upper, unit="%") if bounded else "",
        )
    if known and bounded:
        estimate, bounds = layout.with_interval(value, ci)
        return name, estimate, bounds.removeprefix("95% CI ")
    return name, layout.num(value) if known else "n/a", ""


def _show_accuracy(res: dict[str, Any]) -> None:
    auc, acc, youden = res["auc"], res["acc"], res.get("youden")
    lower = res.get("direction") == m.LOWER
    score, outcome, present = (
        as_code(res["score"]),
        as_code(res["outcome"]),
        as_code(res["pos_label"]),
    )
    metrics = [
        Metric(
            "Area under the ROC curve (AUC)",
            f"{auc['auc']:.3f}",
            f"{layout.interval(auc['ci'])} (DeLong)",
        )
    ]
    words = f"{score} against {outcome} = {present}, {'lower' if lower else 'higher'} values pointing to the condition."
    cautions: list[str] = list(res.get("cautions") or [])
    if acc is None:
        metrics += [
            Metric(
                "With the condition", f"{auc['n_positive']:,}", f"of {res['n']:,} rows analysed"
            ),
            Metric("Without", f"{auc['n_negative']:,}"),
        ]
        further, same, _ = res.get("pairs") or (auc["auc"], 0.0, 1 - auc["auc"])
        words += (
            f" In {_in_100(further)} of 100 pairs of one participant with the condition and one without, the one "
            f"with it has the {'lower' if lower else 'higher'} value"
        )
        # the AUC counts a tie for a half: said apart as soon as ties are a pair in 100
        words += f"; in {_in_100(same)} the two have the same value." if same >= 0.01 else "."
        words += (
            " No cut-off: choose one above to read sensitivity, specificity and predictive values."
        )
    else:
        found = res["mode"] == m.YOUDEN
        origin = "found on these data: optimistic" if found else "fixed beforehand"
        metrics += [
            Metric(
                "Sensitivity",
                layout.percent(acc.sensitivity),
                f"{acc.tp:,} of {acc.tp + acc.fn:,} · "
                f"{layout.interval(acc.sensitivity_ci, scale=100, unit='%')}",
            ),
            Metric(
                "Specificity",
                layout.percent(acc.specificity),
                f"{acc.tn:,} of {acc.tn + acc.fp:,} · "
                f"{layout.interval(acc.specificity_ci, scale=100, unit='%')}",
            ),
            Metric(
                "Cut-off",
                f"{_bound(lower)} {layout.num(res['cutoff'])}",
                origin,
                help="The test is called positive on this side of the cut-off.",
            ),
        ]
        cautions += list(acc.warnings)
    layout.headline(metrics, words, len(cautions))
    left, right = st.columns(2)
    left.plotly_chart(res["fig"], width="stretch")
    with right:
        if acc is not None:
            positive = f"Test positive ({_bound(lower)} {layout.num(res['cutoff'])})"
            table = pd.DataFrame(
                {
                    f"Condition present ({res['pos_label']})": [acc.tp, acc.fn, acc.tp + acc.fn],
                    "Condition absent": [acc.fp, acc.tn, acc.fp + acc.tn],
                    "Total": [acc.tp + acc.fp, acc.fn + acc.tn, acc.tp + acc.fp + acc.fn + acc.tn],
                },
                index=[positive, "Test negative", "Total"],
            )
            st.markdown("**At the cut-off**")
            st.dataframe(table, width="stretch")
            share = layout.percent(acc.prevalence_observed)
            st.caption(
                f"Predictive values at the share with the condition in these data ({share}): positive "
                f"{_share(acc.ppv, acc.ppv_ci)}, negative {_share(acc.npv, acc.npv_ci)}."
            )
    layout.warn(cautions)
    if acc is not None:
        with st.expander("Details: every measure"):
            share = layout.percent(acc.prevalence_observed)
            rows = [
                ("Sensitivity", acc.sensitivity, acc.sensitivity_ci),
                ("Specificity", acc.specificity, acc.specificity_ci),
                (
                    f"Positive predictive value, at the share in these data ({share})",
                    acc.ppv,
                    acc.ppv_ci,
                ),
                (
                    f"Negative predictive value, at the share in these data ({share})",
                    acc.npv,
                    acc.npv_ci,
                ),
            ]
            if res.get("prevalence") is not None:
                typed = layout.percent(res["prevalence"])
                rows += [
                    (
                        f"Positive predictive value, at a prevalence of {typed}",
                        acc.ppv_at_prevalence,
                        None,
                    ),
                    (
                        f"Negative predictive value, at a prevalence of {typed}",
                        acc.npv_at_prevalence,
                        None,
                    ),
                ]
            rows += [
                ("Positive likelihood ratio (LR+)", acc.lr_positive, acc.lr_positive_ci),
                ("Negative likelihood ratio (LR-)", acc.lr_negative, acc.lr_negative_ci),
                ("Diagnostic odds ratio", acc.dor, acc.dor_ci),
                ("Youden's J", acc.youden, None),
                ("Share classified correctly", acc.accuracy, None),
            ]
            shares = {"Sensitivity", "Specificity", "Share classified correctly"}
            table = pd.DataFrame(
                [
                    _measure(name, value, ci, share=name in shares or "predictive value" in name)
                    for name, value, ci in rows
                ],
                columns=["Measure", "Estimate", "95% CI"],
            )
            st.dataframe(table, hide_index=True, width="stretch")
            if res["mode"] == m.YOUDEN and youden:
                ci = youden.get("ci")
                if ci is not None:
                    st.caption(
                        f"In {youden.get('resamples') or 0:,} resamples of these data, 95% of the cut-offs found "
                        f"lie between {layout.num(ci.lower)} and {layout.num(ci.upper)} (a range, not a confidence "
                        "interval)."
                    )
                st.caption(youden["note"])
            st.caption("Method: " + " | ".join([*acc.references, *auc["references"]]))
    else:
        with st.expander("Details: the method"):
            st.caption("Method: " + " | ".join(auc["references"]))
    help_popover(
        "Sensitivity and specificity describe the test; predictive values describe what a result means for the "
        "person tested, and change with how common the condition is. Likelihood ratios turn the odds before the test "
        "into the odds after it and do not depend on the prevalence. The AUC sums up how well the test separates "
        "the two groups over every cut-off; its interval shows how precisely."
    )


# ----------------------------------------------------------------------------- agreement of two measurements


def _swap_measurements() -> None:
    st.session_state["agree_cols"] = list(reversed(st.session_state.get("agree_cols") or []))


def _limits(ba: Any) -> list[str]:
    """The two limits of agreement, then the bounds of the lower limit's interval and of the upper limit's, written
    with the same decimals: the headline, its caption, its sentence and the figure's legend give a number one way
    (the upper limit read "20.17" over "95% CI 19.4 to 21.0", and "20.2" in the legend)."""
    return layout.alike(
        ba.loa_lower,
        ba.loa_upper,
        ba.loa_lower_ci.lower,
        ba.loa_lower_ci.upper,
        ba.loa_upper_ci.lower,
        ba.loa_upper_ci.upper,
    )


def _bland_altman_figure(ba: Any, first: str, second: str) -> go.Figure:
    fig = go.Figure()
    x0, x1 = float(np.min(ba.means)), float(np.max(ba.means))
    limits = _limits(ba)  # as the headline: "20.17", not "20.2"
    for ci in (ba.bias_ci, ba.loa_upper_ci, ba.loa_lower_ci):  # under the points
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, x1, x0],
                y=[ci.lower, ci.lower, ci.upper, ci.upper],
                fill="toself",
                mode="none",
                fillcolor=_rgba(OKABE_ITO[0], BAND_ALPHA),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=ba.means,
            y=ba.differences,
            mode="markers",
            name="differences",
            marker={"color": OKABE_ITO[0], "size": 6, "opacity": 0.7},
        )
    )
    for y, name, dash, colour in (
        (
            ba.bias,
            f"mean difference {layout.with_interval(ba.bias, ba.bias_ci)[0]}",
            "dash",
            OKABE_ITO[1],
        ),
        (ba.loa_upper, f"upper limit {limits[1]}", "dot", OKABE_ITO[2]),
        (ba.loa_lower, f"lower limit {limits[0]}", "dot", OKABE_ITO[2]),
    ):
        fig.add_trace(
            go.Scatter(
                x=[x0, x1], y=[y, y], mode="lines", name=name, line={"dash": dash, "color": colour}
            )
        )
    fig.update_layout(
        xaxis_title=f"mean of {first} and {second}",
        yaxis_title=f"{first} - {second}",
        height=420,
        legend={"orientation": "h", "yanchor": "top", "y": -0.2, "x": 0},
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
    )
    return fig


def _measurements(df: pd.DataFrame, *, mi: Any, guided: bool) -> None:
    offered = m.measurements(df)
    if len(offered) < 2:
        empty_state(
            "Fewer than two numeric columns",
            "The agreement of two measurements needs two numeric columns that measure the same quantity on the same "
            "participants.",
            links=(("enrich", "Create new variables", "variables"),),
        )
        return
    kept = [c for c in st.session_state.get("agree_cols") or [] if c in offered]
    if kept != (st.session_state.get("agree_cols") or []):
        st.session_state["agree_cols"] = kept  # a column the data no longer hold
    columns = st.multiselect(
        "Measurements of the same quantity (one column each)",
        offered,
        default=[],
        placeholder="Choose two or more columns",
        key="agree_cols",
        persist_state="page",
        help=HELP["measurements"],
    )
    if len(columns) == 2:
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.caption(
                f"Differences are {as_code(columns[0])} - {as_code(columns[1])}.", width="content"
            )
            st.button(
                "Swap",
                key="agree_swap",
                type="tertiary",
                icon=":material/swap_horiz:",
                on_click=_swap_measurements,
                help="Takes the differences the other way round.",
            )
    a1, a2 = st.columns([1, 2])
    analyses = a1.pills(
        "Analyses",
        list(m.AGREEMENT),
        format_func=lambda value: m.AGREEMENT[value],
        selection_mode="multi",
        default=list(m.AGREEMENT),
        key="agree_analyses",
        persist_state="page",
        help=HELP["analyses"],
    )
    # each option is drawn with the analysis that reads it (both stayed on the screen with their analysis unticked)
    model = st.session_state.get("icc_model") or m.ABSOLUTE
    if m.ICC in analyses:
        model = a2.radio(
            "What counts as agreement (ICC)",
            list(m.ICC_FORMS),
            format_func=lambda value: m.ICC_FORMS[value],
            key="icc_model",
            persist_state="page",
            help=HELP["icc_form"],
        )
    layout.depends("agree_delta", tuple(columns), None)  # in the unit of these measurements
    delta = None
    if m.BLAND_ALTMAN in analyses and len(columns) == 2:
        with st.expander("More options"):
            delta = st.number_input(
                "Largest difference that would not matter (optional)",
                min_value=0.0,
                value=None,
                placeholder="in the unit of the measurements",
                format="%g",
                key="agree_delta",
                persist_state="page",
                help=HELP["delta"],
            )
    plan = layout.kept(
        "agree_plan",
        (tuple(columns), tuple(analyses), model),
        lambda: m.plan_measurements(df, columns, analyses, model),
    )
    if guided:
        layout.plan_box(plan)
    else:
        layout.checks(plan)
    if mi is not None:
        layout.imputed_line(
            OBSERVED
            + "the rows with every measurement. Limits of agreement have no pooling rule, and none is "
            "established for the ICC: report both on the complete rows."
        )
    pair = len(columns) == 2
    with_ba = pair and m.BLAND_ALTMAN in analyses and bool(plan.steps)
    with_icc = m.ICC in analyses and len(columns) >= 2 and bool(plan.steps)
    ba_config = tuple(columns) if pair else ()
    icc_config = (tuple(columns), model)
    if layout.run_button(
        "agree_run", ready=bool(plan.steps), what="the agreement of the measurements", plan=plan
    ):
        try:
            if with_ba:
                found = bland_altman(df[columns[0]], df[columns[1]])
                remember(
                    "bland_altman",
                    found,
                    config=ba_config,
                    n_analysed=found.n,
                    columns=list(columns),
                )
            if with_icc:
                reliability = icc(df[list(columns)], model=model)
                remember(
                    "icc",
                    reliability,
                    config=icc_config,
                    n_analysed=reliability.n_subjects,
                    columns=list(columns),
                )
            mark_step("analyse", "agreement of two measurements")
        except Exception as exc:
            error_card("The agreement could not be computed.", exc)
    # one "How to read this" for the two results of one Run, after the last
    readings: list[str] = []
    ba = layout.result_place("bland_altman", ba_config, guided=guided) if with_ba else None
    if ba is not None:
        readings.append(_show_bland_altman(ba, columns[0], columns[1], delta))
        provenance("bland_altman")
    reliability = (
        # after the limits of agreement, their place has said why no result is shown
        layout.result_place(
            "icc", icc_config, guided=guided, silent=with_ba and ba is None, words=m.ICC_FORMS
        )
        if with_icc
        else None
    )
    if reliability is not None:
        readings.append(_show_icc(reliability, len(df)))
        provenance("icc")
    if readings:
        help_popover("\n\n".join(readings))


def _show_bland_altman(ba: Any, first: str, second: str, delta: float | None) -> str:
    """The limits of agreement; gives back how to read them (said once, after the last result of the Run)."""
    a, b = as_code(first), as_code(second)
    reads = "higher" if ba.bias >= 0 else "lower"
    bias, bounds = layout.with_interval(ba.bias, ba.bias_ci)
    low, high, low_from, low_to, high_from, high_to = _limits(ba)
    layout.headline(
        [
            Metric("Bland-Altman: mean difference", bias, bounds),
            Metric(
                "Limits of agreement",
                f"{low} to {high}",
                f"95% CI {low_from} to {low_to} and {high_from} to {high_to}",
            ),
            Metric("Pairs", f"{ba.n:,}"),
        ],
        f"{a} reads {bias.removeprefix('-')} {reads} than {b} on average. For about 95 of 100 participants the "
        f"two differ by between {low} and {high}.",
        len(ba.warnings),
    )
    st.plotly_chart(_bland_altman_figure(ba, first, second), width="stretch")
    if ba.proportional_bias_p < 0.05:
        st.markdown(
            f":material/warning: The difference changes with the size of the measurement "
            f"({layout.num(ba.proportional_bias_slope, 2)} per unit, p {layout.p_text(ba.proportional_bias_p)}). "
            f"{SLOPE_READING.capitalize()}: look at the plot."
        )
    if delta is not None and delta > 0:
        within = float(np.mean(np.abs(ba.differences) <= delta))
        inside = ba.loa_lower >= -delta and ba.loa_upper <= delta
        st.markdown(
            f":material/straighten: With {layout.num(delta)} as the largest difference that would not matter: "
            f"{layout.percent(within)} of the differences lie within it, and the limits of agreement "
            f"{'lie within it too' if inside else 'reach beyond it'}."
        )
    # the line above says the result's own word on a difference that changes with the measurement: not said twice
    layout.warn([text for text in ba.warnings if text != SLOPE_WARNING])
    with st.expander("Details: the differences and the method"):
        st.caption(
            f"SD of the differences {layout.num(ba.sd)} (ddof = 1); {layout.percent(ba.within_loa_pct / 100)} of "
            "the differences lie within the limits (close to 95% by construction). Slope of the difference on the "
            f"mean: {layout.num(ba.proportional_bias_slope, 2)} per unit, p "
            f"{layout.p_text(ba.proportional_bias_p)}: it shows that the two spreads differ, not why."
        )
        st.caption("Method: " + " | ".join(ba.references))
    return (
        "**Limits of agreement.** The range within which 95% of the differences between the two measurements are "
        "expected. Whether that range is acceptable is a judgement for the field of use, not a statistical one: "
        "decide the largest difference that would not matter before looking. The intervals around the limits show "
        "how precisely they are known."
    )


def _show_icc(res: Any, rows: int) -> str:
    """The intraclass correlation; gives back how to read it (said once, after the last result of the Run)."""
    absolute = res.model.startswith(m.ABSOLUTE)
    left_out = rows - res.n_subjects
    known = bool(np.isfinite(res.icc))
    # at 1, absolute agreement is the same value; consistency is the same ranking and spacing (two columns 10 apart
    # everywhere give 1)
    top = "they agree fully" if absolute else "the same ranking and spacing"
    counted = (
        "the same value (absolute agreement)"
        if absolute
        else "the same ranking and spacing, a constant shift between columns not counting (consistency)"
    )
    layout.headline(
        [
            Metric(
                m.ABSOLUTE
                if absolute
                else m.CONSISTENCY,  # the result's name: two results follow one Run
                f"{res.icc:.3f}" if known else "n/a",
                f"{layout.interval(res.ci)} (Shrout-Fleiss)"
                if known
                else "no value on these columns",
            ),
            Metric(
                "Participants",
                f"{res.n_subjects:,}",
                f"of {rows:,} rows, on {res.n_raters} columns"
                + (f"; {left_out:,} lack a value" if left_out else ""),
            ),
        ],
        f"Counted as agreement: {counted}, one measurement each. 0 = the measurements do not tell participants "
        f"apart, 1 = {top}.",
    )
    with st.expander("Details: the test, the benchmark and the method"):
        low, high = koo_li(res.ci.lower), koo_li(res.ci.upper)
        text = f"F = {layout.num(res.f_statistic)}, df = {res.df1:.0f}, {res.df2:.0f}, p {layout.p_text(res.p_value)}."
        if known:  # an ICC without a value has no word
            text += " Benchmark (Koo & Li), read on the interval: " + (
                f"{low}." if low == high else f"from {low} to {high}."
            )
        st.caption(text)
        st.caption("Method: " + " | ".join(res.references))
    return (
        "**Intraclass correlation.** The share of the differences between measurements that comes from the "
        "participants themselves rather than from the raters or occasions. It depends on how much the participants "
        "differ: the same raters score lower in a group that is alike. Read its interval, not the estimate alone."
    )


# ----------------------------------------------------------------------------- agreement of two raters


def kappa_of(
    df: pd.DataFrame, first: str, second: str, *, order: Sequence[str] | None, weights: str | None
) -> dict[str, Any]:
    """
    Cohen's kappa of two raters' categories on the participants rated by both, with the cross-table it is computed
    on. With ``weights`` (ordered grades), ``order`` gives the grades from the lowest to the highest: the weights
    read distances in that order.
    """
    table = rater_table(df[first], df[second], order=order if weights else None)
    return {
        "kappa": cohens_kappa(table.to_numpy(), weights=weights),
        "table": table,
        "order": [str(name) for name in table.index] if weights else [],
        "weights": weights,
    }


def _raters(df: pd.DataFrame, *, mi: Any, guided: bool) -> None:
    offered = m.rated(df)
    if len(offered) < 2:
        empty_state(
            "Fewer than two columns of categories",
            "The agreement of two raters needs two columns that hold the same 2 to 10 categories. A column whose "
            "categories are written several ways can be cleaned in Data > Clean.",
            links=(("clean", "Clean categories"),),
        )
        return
    c1, c2 = st.columns(2)
    first = c1.selectbox(
        "First rater",
        offered,
        index=None,
        placeholder="Choose a column of categories",
        key="kappa_a",
        persist_state="page",
        help=HELP["rater_a"],
    )
    second = c2.selectbox(
        "Second rater (same categories)",
        m.look_alikes(first, offered),
        index=None,
        placeholder="Choose a column of categories",
        key="kappa_b",
        persist_state="page",
        help=HELP["rater_b"],
    )
    categories = m.rater_categories(df, first, second)[0] if first and second else []
    scale, order, weights = m.NAMES, None, m.LINEAR
    if len(categories) > 2:
        s1, s2 = st.columns([1, 2])
        scale = s1.radio(
            "The categories are",
            list(m.SCALES),
            format_func=lambda value: m.SCALES[value],
            key="kappa_scale",
            persist_state="page",
            help=HELP["scale"],
        )
        if scale == m.GRADES:
            layout.depends(
                "kappa_order", (first, second, tuple(categories)), m.grade_order(categories)
            )
            order = s2.multiselect(
                "Order of the grades, lowest first",
                categories,
                key="kappa_order",
                persist_state="page",
                help=HELP["order"],
            )
            weights = s2.radio(
                "A near miss counts",
                [m.LINEAR, m.QUADRATIC],
                format_func=lambda value: m.NEAR_MISSES[value],
                horizontal=True,
                key="kappa_w",
                persist_state="page",
                help=HELP["near_miss"],
            )
    graded = scale == m.GRADES and len(categories) > 2
    used = (tuple(order or ()), weights) if graded else ()
    plan = layout.kept(
        "kappa_plan",
        (first, second, scale, *used),
        lambda: m.plan_raters(df, first, second, scale, order, weights),
    )
    if guided:
        # a column whose categories are written several ways is cleaned from the plan, and its copy takes its place:
        # sent to Data > Clean, the researcher lost every choice made on this page
        fields = ((first, "kappa_a", "first"), (second, "kappa_b", "second"))
        clean = {
            column: {"group_key": key, "groupable": offered, "use": f"Use as the {which} rater:"}
            for column, key, which in fields
            if column
        }
        layout.plan_box(plan, clean=clean)
    else:
        layout.checks(plan)
    if mi is not None:
        layout.imputed_line(OBSERVED + "the rows rated by both. Not pooled in this version.")
    config = (first, second, scale if graded else m.NAMES, *used)
    if layout.run_button(
        "kappa_run", ready=bool(plan.steps), what="the agreement of the two raters", plan=plan
    ):
        try:
            result = kappa_of(
                df,
                first,  # type: ignore[arg-type]
                second,  # type: ignore[arg-type]
                order=order if graded else None,
                weights=weights if graded else None,
            )
            remember(
                "kappa",
                result,
                config=config,
                n_analysed=result["kappa"].n,
                columns=[first, second],  # type: ignore[list-item]
            )
            mark_step("analyse", "agreement (kappa)")
        except Exception as exc:
            error_card("Kappa could not be computed.", exc)
    result = layout.result_place(
        "kappa", config, guided=guided, words={**m.SCALES, **m.NEAR_MISSES}
    )
    if result is not None:
        _show_kappa(result, first, second)  # type: ignore[arg-type]
        provenance("kappa")


def _show_kappa(result: dict[str, Any], first: str, second: str) -> None:
    res, table, weights = result["kappa"], result["table"], result.get("weights")
    same = int(np.trace(table.to_numpy()))
    name = f"Weighted kappa ({weights})" if weights else "Cohen's kappa"
    scale = (
        "grades in the order " + " < ".join(result["order"])
        if weights
        else "categories without an order"
    )
    words = (
        "0 = no better than chance, 1 = complete agreement. By chance alone "
        f"{layout.percent(res.expected_agreement)} would fall in the same category."
    )
    if weights:
        words += f" {NEAR_MISSES}"
    counts = table.to_numpy()
    # one rater puts everyone in one category: kappa is 0 whatever the other says, and its interval has no width
    fixed = int((counts.sum(axis=1) > 0).sum()) == 1 or int((counts.sum(axis=0) > 0).sum()) == 1
    layout.headline(
        [
            Metric(
                "Same category",
                layout.percent(res.observed_agreement),
                f"{same:,} of {res.n:,} rated by both",
            ),
            Metric(
                name,
                f"{res.kappa:.3f}",
                "0 by construction: one rater uses one category only"
                if fixed
                else f"{layout.interval(res.ci)} · {scale}",
            ),
        ],
        words,
    )
    shown = table.copy()
    shown["Total"] = shown.sum(axis=1)
    shown.loc["Total"] = shown.sum(axis=0)
    # the headers are the categories: the caption names the two axes ("nyha_class_central = " stood in every one)
    shown.index = [str(name) for name in shown.index]
    shown.columns = [str(name) for name in shown.columns]
    st.caption(
        f"Rows: {as_code(first)} (first rater). Columns: {as_code(second)} (second rater). The diagonal holds the "
        "participants both put in the same category."
    )
    st.dataframe(shown, width="stretch")
    label = "Details: the standard error and the method"
    with st.expander(label if weights else label.replace(" and", ", the benchmark and")):
        text = f"Standard error {res.se:.3f} (normal approximation for the interval)."
        if not weights and not fixed:
            low, high = landis_koch(res.ci.lower), landis_koch(res.ci.upper)
            text += " Benchmark (Landis & Koch), read on the interval: " + (
                f"{low} agreement." if low == high else f"from {low} to {high} agreement."
            )
        st.caption(text)
        st.caption("Method: " + " | ".join(res.references))
    help_popover(
        "Kappa compares the share of participants the two raters put in the same category with the share expected "
        "by chance from how often each rater uses each category. It can be low although the raters agree on most "
        "participants, when nearly everyone falls in one category: read the share of same categories beside it."
    )


# ----------------------------------------------------------------------------- the tab


def run_diagnostic_accuracy(df: pd.DataFrame, analyzer: Any, *, mi: Any = None) -> None:
    """The tab: its Question selector, then the analysis it names. ``analyzer`` is the page's column profile (the
    entries read the columns themselves: a whole-number score is a measurement here)."""
    guided = guidance_on()
    entry = layout.choose_entry(m.DIAGNOSTIC_ENTRIES, key=QUESTION_KEY, guided=guided)
    panel = {m.ACCURACY: _accuracy, m.MEASUREMENTS: _measurements, m.RATERS: _raters}[entry.id]
    panel(df, mi=mi, guided=guided)


__all__ = ["QUESTION_KEY", "accuracy_of", "kappa_of", "range_drawn", "run_diagnostic_accuracy"]
