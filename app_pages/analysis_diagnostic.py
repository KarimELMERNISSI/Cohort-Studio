"""
Diagnostic accuracy & agreement, a tab of Analyze > Hypothesis tests & epidemiology: three questions, one open at a
time (ui.guidance.measures): the accuracy of a test against a reference standard (a score, with the ROC curve and its
AUC, or a yes/no test: sensitivity, specificity, predictive values by the sampling), the agreement of two measurements (Bland-Altman limits and the ICC, on one set of columns), and the
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
    KOOPMAN,
    SAMPLING_NOT_STATED,
    SAMPLINGS,
    SCORE_CARRIED,
    SERIES,
    TWO_GATE,
    YOUDEN_MIN_CLASS,
    confusion_at,
    delong_auc,
    diagnostic_accuracy,
    verification,
    yes_no_table,
    youden_cutoff,
)
from biostats.plan import run as run_spec
from biostats.plan import specify
from biostats.references import cite
from biostats.reliability import (
    NEAR_MISSES,
    cohens_kappa,
    icc,
    koo_li,
    landis_koch,
    rater_table,
)
from core.registry import as_code
from manage.report_tables import (
    auc_no_width,
    auc_row,
    dossier_min_n,
    measure_rows,
    table_cautions,
)
from ui.components import empty_state, error_card, help_popover
from ui.guidance import measures as m
from ui.guidance.measures import HELP
from ui.guidance.numbers import area_parts, share_decimals, share_parts
from ui.progress import mark_step
from ui.results import CURRENT, provenance, remember, status
from ui.theme import OKABE_ITO

QUESTION_KEY = "diag_question"
MUTED = "#6B7C86"  # the line of chance: readable on the light and on the dark surface alike
BAND_ALPHA = 0.25  # at 0.12 the intervals of the limits were lost on the dark surface
OBSERVED = "Imputed data are selected; this analysis reads the observed data: "


def _rgba(colour: str, alpha: float) -> str:
    r, g, b = (int(colour[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


# ----------------------------------------------------------------------------- accuracy of a test


# the form of the index test among a result's settings, in words (the dossier's line, "Last result here")
FORM_SETTING = {m.NUMBER: "a score", m.YES_NO: "a yes/no test"}


def accuracy_settings(
    form: str,
    present: Any,
    positive: Any = None,
    direction: str | None = None,
    mode: str | None = None,
    cutoff: float | None = None,
    prevalence: float | None = None,
    sampling: str = SAMPLING_NOT_STATED,
) -> dict[str, Any]:
    """
    The options in use of one accuracy analysis, by stored id: those ``biostats.plan.KINDS["diagnostic"]`` declares,
    so that the identity of a result on the page (its settings, built from these: :func:`_config`) is the identity
    of its request.

    The form and the value that means the condition, always; the test's positive value for a yes/no test; the
    direction and the cut-off's mode for a score, the cut-off for one typed beforehand only; the sampling and the
    prevalence wherever a 2x2 table is read (a yes/no test, a score at a cut-off). An option left out takes its
    declared default: a score without a cut-off is one analysis whatever the sampling question holds.
    """
    settings: dict[str, Any] = {
        "index_form": form,
        "present": None if present is None else str(present),
    }
    if form == m.YES_NO:
        settings["positive"] = None if positive is None else str(positive)
    else:
        settings["direction"] = direction
        settings["mode"] = mode
        if mode == m.FIXED:
            settings["cutoff"] = None if cutoff is None else float(cutoff)
    if form == m.YES_NO or mode != m.NO_CUTOFF:
        settings["sampling"] = sampling
        settings["prevalence"] = None if prevalence is None else float(prevalence)
    return settings


def _config(columns: Sequence[str | None], settings: dict[str, Any], *more: str) -> tuple:
    """A result's settings as the page keeps them and the dossier's line writes them: the two columns, then each
    option in use (:func:`accuracy_settings`), in the words of its id's catalogue."""
    words: list[Any] = [settings["present"]]
    if "positive" in settings:
        positive = settings["positive"]
        words.append(None if positive is None else f"positive when {positive}")
    words.append(FORM_SETTING[settings["index_form"]])
    if "mode" in settings:
        words += [settings["direction"], m.CUTOFFS[settings["mode"]]]
    if "cutoff" in settings:
        words.append(settings["cutoff"])
    if "sampling" in settings:
        words += [
            _prevalence_setting(settings["prevalence"]),
            m.SAMPLING_SETTING[settings["sampling"]],
        ]
    return (*columns, *words, *more)


def _as_measured(found: dict) -> dict:
    """A cut-off found on the reversed marker (lower values pointing to the condition), on the marker's own scale."""
    shown = dict(found)
    shown["cutoff"] = -found["cutoff"]
    ci = found.get("ci")
    if ci is not None:
        shown["ci"] = CI(lower=-ci.upper, upper=-ci.lower, level=ci.level, method=ci.method)
    return shown


def _sides_of(ci: Any, scores: pd.Series, lower: bool) -> tuple[tuple[int, int], ...] | None:
    """
    The participants on each side of each end of a cut-off's range (below it, then above it, on the marker's own
    scale; an end itself counted on the side it would call positive). An end is a score found in a resample, a
    participant's own when few lie beyond it: the dossier writes an end only when enough lie on each side.
    """
    if ci is None:
        return None
    values = scores.to_numpy(dtype=float)
    sides = []
    for end in (float(ci.lower), float(ci.upper)):
        # positive at or above the end for a rising marker, at or below it for a falling one
        below = int((values <= end).sum()) if lower else int((values < end).sum())
        sides.append((below, int(len(values) - below)))
    return tuple(sides)


def accuracy_of(
    df: pd.DataFrame,
    outcome: str,
    present: Any,
    score: str,
    direction: str,
    mode: str,
    cutoff: float | None,
    prevalence: float | None,
    sampling: str = SAMPLING_NOT_STATED,
) -> dict[str, Any]:
    """
    The accuracy of ``score`` against ``outcome`` = ``present`` on the rows where both are recorded: the AUC always;
    with a cut-off (fixed beforehand, or found on these data by Youden's J), the 2x2 table and its measures, by the
    ``sampling`` (predictive values at the share in the data for a series only).

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
        youden = {**youden, "ci_sides": _sides_of(youden.get("ci"), clean[score], lower)}
    elif mode == m.FIXED:
        if cutoff is None:
            raise ValueError("a cut-off fixed beforehand is a number to type")
        at = -float(cutoff) if lower else float(cutoff)
    acc = None
    if at is not None:
        tp, fp, fn, tn = confusion_at(truth, marker, at, pos_label=present)
        acc = diagnostic_accuracy(
            tp,
            fp,
            fn,
            tn,
            prevalence=None if prevalence is None else float(prevalence),
            sampling=sampling,
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
            f"the area under the curve is {area_parts(auc['auc'])[0]}, below 0.5. Choose {other} above, or check "
            "the value that means the condition is present."
        )
    if auc_no_width(auc):
        cautions.append(_no_width(auc, lower))
    verified = None
    if at is not None:
        # the reference standard done unequally on the two sides of the cut-off, among the rows that hold the test
        recorded = pd.to_numeric(df[score], errors="coerce")
        called = (-recorded >= at if lower else recorded >= at).where(recorded.notna())
        verified = verification(called, df[outcome])
    return {
        "index_form": m.NUMBER,
        "absent": next((v for v in m.levels_of(clean[outcome]) if v != present), None),
        "verified": verified,
        "verification": ""
        if verified is None or acc is None
        else m.verification_words(verified, acc.prevalence_observed, acc.tp + acc.fn),
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
            name=f"ROC curve (AUC {area_parts(auc['auc'])[0]})",
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


def accuracy_yes_no_of(
    df: pd.DataFrame,
    outcome: str,
    present: Any,
    test: str,
    positive: Any,
    prevalence: float | None,
    sampling: str = SAMPLING_NOT_STATED,
) -> dict[str, Any]:
    """
    The accuracy of the yes/no ``test`` (positive when it is ``positive``) against ``outcome`` = ``present``, on the
    rows where both are recorded: the 2x2 table and its measures, by the sampling. A yes/no test has one point on
    the ROC plane: no curve, no area, no cut-off. The result has the keys of a score's (``score`` names the index
    test's column, whatever its form), with ``index_form``, the test's two values, and no ``auc``.
    """
    tp, fp, fn, tn = yes_no_table(df, outcome, test, present=present, positive=positive)
    acc = diagnostic_accuracy(
        tp,
        fp,
        fn,
        tn,
        prevalence=None if prevalence is None else float(prevalence),
        sampling=sampling,
    )
    frame = df[[outcome, test]].dropna()
    absent = next(v for v in m.levels_of(frame[outcome]) if v != present)
    negative = next(v for v in m.levels_of(frame[test]) if v != positive)
    cautions: list[str] = []
    if acc.sensitivity < 1 - acc.specificity:  # kept with the result: Guidance off still says it
        cautions.append(
            m.other_way_words(f"{test} = {positive}", acc.sensitivity, 1 - acc.specificity)
        )
    verified = verification((df[test] == positive).where(df[test].notna()), df[outcome])
    return {
        "verified": verified,
        "verification": m.verification_words(verified, acc.prevalence_observed, acc.tp + acc.fn),
        "auc": None,
        "pairs": None,
        "cautions": cautions,
        "youden": None,
        "cutoff": None,
        "acc": acc,
        "outcome": outcome,
        "score": test,
        "pos_label": present,
        "absent": absent,
        "positive": positive,
        "negative": negative,
        "index_form": m.YES_NO,
        "mode": None,
        "n": len(frame),
        "direction": None,
        "prevalence": None if prevalence is None else float(prevalence),
        "fig": None,
    }


def _reference_standard(df: pd.DataFrame, standards: list[str], first: Any, second: Any) -> tuple:
    """The reference standard and the value that means the condition is present: asked once for a score and for a
    yes/no test of the same study (their keys are shared)."""
    outcome = first.selectbox(
        "Reference standard",
        standards,
        index=None,
        placeholder="Choose a column with two values",
        key="diag_outcome",
        persist_state="page",
        help=HELP["standard"],
    )
    levels = m.levels_of(df[outcome]) if outcome else []
    layout.depends("diag_pos", (outcome, tuple(map(str, levels))), m.present_value(levels))
    present = second.selectbox(
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
    return outcome, present


def _share_text(share: Any) -> str:
    """The share with the condition as a percentage, with the decimals that tell it from 0% and from 100%
    (``share_decimals``): 24 of 50,000 read "0.0%" beside a negative predictive value of 99.992%."""
    return (
        layout.percent(share, share_decimals(share))
        if layout.known(share)
        else layout.percent(share)
    )


def _sampling_question(
    sampling_key: str,
    prevalence_key: str,
    share: float | None,
    *,
    where: str,
    helps: tuple[str, str],
    disabled: bool = False,
) -> tuple[str, float | None]:
    """
    How the participants entered the study, and under it the prevalence where the test will be used: two of the
    three answers read predictive values only at that prevalence, so it stands in view (it was the one field of a
    folded "More options"). Under the question, in half the width: beside it, in a third of the row, its label
    wrapped onto two lines in a 1,280-pixel window and its field sat below the question's first option. ``share``:
    the share with the condition in ``where`` ("these data", "this table"), said for a series only: beside another
    answer it is the share that answer has just set aside, and typing it back would read the predictive values
    withheld.
    """
    sampling = (
        st.radio(
            m.DESIGN_QUESTION,
            SAMPLINGS,
            format_func=m.SAMPLING_LABELS.__getitem__,
            disabled=disabled,
            key=sampling_key,
            persist_state="page",
            help=helps[0],
        )
        or SAMPLING_NOT_STATED
    )
    half, _ = st.columns(2)
    prevalence = half.number_input(
        "Prevalence where the test will be used (optional)",
        min_value=0.0,
        max_value=1.0,
        value=None,
        step=0.01,
        format="%g",  # as typed: two decimals showed 1 in 2,500 as "0.00"
        placeholder="for example 0.02",
        key=prevalence_key,
        persist_state="page",
        help=helps[1],
    )
    if share is not None and sampling == SERIES:
        half.caption(f"In {where}: {_share_text(share)} (a series).")
    return sampling, prevalence


def _sampling_and_prevalence(outcome: str | None, share: float | None) -> tuple[str, float | None]:
    """
    The question and the prevalence of a test of the dataset (:func:`_sampling_question`). The question is a
    statement about who was taken into the study, asked once the reference standard is chosen and again for
    another; it starts at "Not stated", which reads no predictive value at the share in the data.
    """
    layout.depends("diag_sampling", outcome, SAMPLING_NOT_STATED)
    layout.depends("diag_prev", outcome, None)  # a prevalence is that of one condition
    return _sampling_question(
        "diag_sampling",
        "diag_prev",
        share,
        where="these data",
        helps=(HELP["sampling"], HELP["prevalence"]),
        disabled=not outcome,
    )


def _prevalence_setting(prevalence: float | None) -> str | None:
    """A prevalence given, among a result's settings: in words (a bare "0.05" stood in the dossier's line), with
    twelve significant figures (two prevalences a person can type are two results; six kept 0.1234561 and 0.1234564
    as one)."""
    return None if prevalence is None else f"at a prevalence of {prevalence:.12g}"


def _accuracy(df: pd.DataFrame, *, mi: Any, guided: bool) -> None:
    form = layout.choose(
        "The test's result is",
        m.TEST_RESULTS,
        key="diag_result",
        help=HELP["test_result"],
        default=m.NUMBER,
        depth=2,
    )
    panel = _accuracy_score if form == m.NUMBER else _accuracy_yes_no
    panel(df, mi=mi, guided=guided)


def _table_cell(
    label: str, key: str, column: Any, example: str, help_key: str, *, shown: bool = True
) -> int | None:
    """One count of the 2x2 table typed. ``shown``: the label is drawn (the second row keeps it for screen readers
    only: its row names the test's result and the first row's labels head the columns)."""
    return column.number_input(
        label,
        label_visibility="visible" if shown else "collapsed",
        min_value=0,
        value=None,
        step=1,
        placeholder=example,
        key=key,
        persist_state="page",
        help=HELP[help_key],
    )


def _accuracy_typed() -> None:
    """
    The accuracy of a test from the four counts of a 2x2 table typed (a paper's, another study's): a calculator,
    computed as the numbers are typed and through the plan, as the dossier computes it again from the same numbers.
    It reads no row: no plan, no Run, no line of imputed data. The fields are laid out as the table, so no table is
    echoed under the result; its margins are, to compare with the source's.
    """
    st.caption(
        "Rows: the test's result. Columns: the condition, by the reference standard. A paper's table may be the "
        "other way round."
    )
    label, present, absent = st.columns([1.2, 2, 2], vertical_alignment="bottom")
    label.markdown("**Test positive**")
    tp = _table_cell("Condition present", "diag_tp", present, "for example 90", "table_tp")
    fp = _table_cell("Condition absent", "diag_fp", absent, "for example 30", "table_fp")
    label, present, absent = st.columns([1.2, 2, 2], vertical_alignment="bottom")
    label.markdown("**Test negative**")
    fn = _table_cell(
        "Condition present, test negative",
        "diag_fn",
        present,
        "for example 10",
        "table_fn",
        shown=False,
    )
    tn = _table_cell(
        "Condition absent, test negative",
        "diag_tn",
        absent,
        "for example 270",
        "table_tn",
        shown=False,
    )
    whole = tp is not None and fp is not None and fn is not None and tn is not None
    if whole:
        # the one check of a table typed the other way round (the condition in its rows), which no measure shows
        st.caption(
            m.table_totals(int(tp), int(fp), int(fn), int(tn))  # type: ignore[arg-type]
            + " Compare these totals with the paper's: they differ when the table is typed the other way round."
        )
    counted = int(tp) + int(fp) + int(fn) + int(tn) if whole else 0  # type: ignore[arg-type]
    # the sampling and the prevalence of this table, apart from those of a test of the dataset (their own keys: a
    # prevalence typed for a column of the dataset became the table's)
    sampling, prevalence = _sampling_question(
        "diag_sampling_typed",
        "diag_prev_typed",
        (int(tp) + int(fn)) / counted if counted else None,  # type: ignore[arg-type]
        where="this table",
        helps=(HELP["table_sampling"], HELP["table_prevalence"]),
    )
    source = layout.numbers_source("diag_from")
    # a side under the dossier's threshold: said here, before the table is added and found held back there
    ready, advice = m.check_table(tp, fp, fn, tn, prevalence, dossier_min=dossier_min_n())
    layout.advice_lines(advice)
    if not ready:
        if not advice:
            st.caption(
                ":material/info: Type the four counts of the table: the result appears here."
            )
        return
    try:
        spec = specify(
            "diagnostic",
            typed={"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            options={
                "sampling": sampling,
                "prevalence": None if prevalence is None else float(prevalence),
            },
        )
        acc = run_spec(spec).result
    except Exception as exc:
        error_card("The accuracy of the test could not be computed.", exc)
        return
    _show_accuracy_typed(acc)
    layout.typed_close(spec, source, "diag_from")


def _show_accuracy_typed(acc: Any) -> None:
    """The result of a 2x2 table typed: the headline of a yes/no test, the predictive values its sampling allows,
    every measure folded. What to know of the table (a test that points the other way) is said with the measures'
    own warnings, as the dossier writes them."""
    cautions = [*table_cautions(acc), *acc.warnings]
    with_it, without = acc.tp + acc.fn, acc.fp + acc.tn
    layout.headline(
        [
            _share_metric("Sensitivity", acc.sensitivity, acc.sensitivity_ci, acc.tp, with_it),
            _share_metric("Specificity", acc.specificity, acc.specificity_ci, acc.tn, without),
            Metric(
                "Participants",
                f"{with_it + without:,}",
                f"{with_it:,} with the condition, {without:,} without",
            ),
        ],
        m.SAMPLING_SENTENCE.get(acc.sampling, ""),
        len(cautions),
    )
    said = _predictive_caption(acc, acc.prevalence_assumed, typed=True)
    if said:
        st.caption(said)
    layout.warn(cautions)
    with st.expander("Details: every measure"):
        st.dataframe(
            _measure_rows(acc, acc.prevalence_assumed, typed=True),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            _intervals(acc, acc.prevalence_assumed)
            + " A 2x2 table has one point on the ROC plane: its area would be (sensitivity + specificity) / 2, "
            "nothing beyond the two shares above."
        )
        st.caption("Method: " + " | ".join(acc.references))
    help_popover(m.ACCURACY_READING)


def _accuracy_yes_no(df: pd.DataFrame, *, mi: Any, guided: bool) -> None:
    source = layout.choose(
        "Counts from",
        m.SOURCES,
        key="diag_source",
        help=HELP["table_source"],
        default=m.DATASET,
        depth=3,
    )
    if source == m.TYPED:
        _accuracy_typed()
        return
    columns = m.two_valued(df)
    if len(columns) < 2:
        empty_state(
            "No yes/no test in this dataset",
            "A yes/no test is assessed against a reference standard: two columns with two values each. A test "
            "recorded as a number can be assessed as a score (A number), or made yes/no on Data > Enrich. For the "
            "2x2 table of a paper, choose Numbers I type.",
            links=(("enrich", "Create new variables", "variables"), ("clean", "Clean categories")),
        )
        return
    c1, c2 = st.columns(2)
    outcome, present = _reference_standard(df, columns, c1, c2)
    c3, c4 = st.columns(2)
    candidates = [c for c in columns if c != outcome]
    test = c3.selectbox(
        "Test assessed (index test)",
        candidates,
        index=None,
        placeholder="Choose a column with two values",
        key="diag_test",
        persist_state="page",
        help=HELP["yes_no_test"],
    )
    results = m.levels_of(df[test]) if test else []
    # the value that reads as positive (Positive, Yes, 1 beside 0, Detected...), else to choose: offered again for
    # another test
    layout.depends("diag_test_pos", (test, tuple(map(str, results))), m.present_value(results))
    positive = c4.selectbox(
        "Test positive when it is",
        results,
        index=None,
        format_func=str,
        placeholder="Choose the value" if test else "Choose the test first",
        disabled=not test,
        key="diag_test_pos",
        persist_state="page",
        help=HELP["test_positive"],
    )
    share = None
    if outcome and test and present is not None:
        held = df[[outcome, test]].dropna()[outcome]
        share = float((held == present).mean()) if len(held) else None
    sampling, prevalence = _sampling_and_prevalence(outcome, share)
    choices = (
        outcome,
        None if present is None else str(present),
        test,
        None if positive is None else str(positive),
        prevalence,
        sampling,
    )
    plan = layout.kept(
        "diag_plan_yes_no",
        choices,
        lambda: m.plan_accuracy_yes_no(df, outcome, present, test, positive, prevalence, sampling),
    )
    if guided:
        # a column whose two values are written several ways is cleaned from the plan, and its copy takes its place
        fields = (
            (outcome, "diag_outcome", "Use as the reference standard:"),
            (test, "diag_test", "Use as the test:"),
        )
        clean = {
            column: {"group_key": key, "groupable": columns, "use": use}
            for column, key, use in fields
            if column
        }
        layout.plan_box(plan, clean=clean)
    else:
        layout.checks(plan)
    if mi is not None:
        layout.imputed_line(
            OBSERVED
            + "the rows where the reference standard and the test are both recorded. Not pooled in this version."
        )
    # the form and the sampling in words among the settings (the dossier's line, "Last result here")
    config = _config(
        (outcome, test),
        accuracy_settings(
            m.YES_NO, present, positive=positive, prevalence=prevalence, sampling=sampling
        ),
    )
    if layout.run_button(
        "diag_run_yes_no", ready=bool(plan.steps), what="the accuracy of the test", plan=plan
    ):
        try:
            result = accuracy_yes_no_of(df, outcome, present, test, positive, prevalence, sampling)  # type: ignore[arg-type]
            remember(
                "diagnostic",
                result,
                config=config,
                n_analysed=result["n"],
                columns=[outcome, test],  # type: ignore[list-item]
            )
            mark_step("analyse", "diagnostic accuracy")
        except Exception as exc:
            error_card("The accuracy of the test could not be computed.", exc)
    result = layout.result_place("diagnostic", config, guided=guided)
    if result is not None and result.get("index_form") == m.YES_NO:
        _show_accuracy_yes_no(result)
        provenance("diagnostic")


def _no_width(auc: dict[str, Any], lower: bool) -> str:
    """Why the area is written without an interval: DeLong's variance is 0, and "1.000 (95% CI 1.000 to 1.000)" read
    as an area known exactly. ``lower``: lower values point to the condition (the area is that of the reversed
    marker)."""
    if auc["auc"] == 1:
        return (
            "The two groups do not overlap: every participant with the condition has a "
            f"{'lower' if lower else 'higher'} value than every participant without it. The area under the curve is "
            "1 in these data, and the DeLong interval, which is computed from the overlap, has no width: none is "
            f"written. With {auc['n_positive']:,} with the condition and {auc['n_negative']:,} without, an area of 1 "
            "does not show a test without error."
        )
    return (
        "The DeLong interval of the area has no width in these data: the values of the two groups leave it nothing "
        "to measure (no overlap, the same value for all, or one participant in a group). No interval is written."
    )


def _counts_table(table: pd.DataFrame) -> None:
    """A 2x2 table with its totals, the counts written as the headline writes them ("1,072": the table read "1072"
    under "667 of 1,072")."""
    right = {name: st.column_config.TextColumn(alignment="right") for name in table.columns}
    st.dataframe(table.map(lambda count: f"{int(count):,}"), width="stretch", column_config=right)


def _predictive_pair(positive: tuple[Any, Any], negative: tuple[Any, Any]) -> str:
    """The two predictive values of one line, each with its interval; one that has no value is said with its reason
    ("negative n/a" stood for a test that called everyone positive)."""
    said = []
    for name, nobody, (value, ci) in (
        ("positive", "no one tested positive", positive),
        ("negative", "no one tested negative", negative),
    ):
        if layout.known(value):
            said.append(f"{name} {_share(value, ci)}")
        else:
            said.append(f"no {name} predictive value: {nobody}")
    return ", ".join(said)


def _predictive_caption(
    acc: Any, prevalence: float | None, unequal: bool = False, *, typed: bool = False
) -> str:
    """The predictive values under the 2x2 table, by the sampling: at the share in the data for a series, at the
    prevalence given whatever the sampling (its interval carried from the likelihood ratio's, the prevalence taken
    as known). The numbers first; where none is read, why, with the share named and what to do. ``unequal``: the
    reference standard was done unequally by test result, and the share among those who had it is then no
    prevalence, even in a series. ``typed``: a table typed into the calculator, which has no Run to run again and
    whose share is the table's."""
    share = _share_text(acc.prevalence_observed)
    again = "" if typed else ", then run again"
    at = ""
    if prevalence is not None and acc.ppv_at_prevalence is not None:
        at = (
            f"At a prevalence of {m.prevalence_text(prevalence)}: "
            + _predictive_pair(
                (acc.ppv_at_prevalence, acc.ppv_at_prevalence_ci),
                (acc.npv_at_prevalence, acc.npv_at_prevalence_ci),
            )
            + ", the prevalence taken as known."
        )
    if acc.sampling == TWO_GATE:
        if at:
            return (
                f"{at} Not at the share with the condition here ({share}): those with and without it were sampled "
                "apart."
            )
        return (
            "No predictive values: those with and without the condition were sampled apart, so the share with it "
            f"here ({share}) was set by the sampling. Give the prevalence where the test will be used{again}."
        )
    if acc.sampling != SERIES:
        return at or (
            "No predictive values: if the participants were taken as they came (a series), say so to read them at "
            f"the share in {'this table' if typed else 'these data'} ({share}); otherwise give the prevalence where "
            "the test will be used." + ("" if typed else " Then run again.")
        )
    where = (
        f"Predictive values among those who had the reference standard ({share} with the condition)"
        if unequal
        else "Predictive values at the share with the condition in "
        f"{'this table' if typed else 'these participants'} ({share}, a series)"
    )
    text = f"{where}: {_predictive_pair((acc.ppv, acc.ppv_ci), (acc.npv, acc.npv_ci))}."
    return f"{text} {at}".strip()


def _measure_rows(
    acc: Any, prevalence: float | None, unequal: bool = False, *, typed: bool = False
) -> pd.DataFrame:
    """Every measure of a 2x2 table, with its interval: those its sampling supports (predictive values at the share
    in the data and the share classified correctly for a series only). The rows are those the dossier's block
    writes (``manage.report_tables.measure_rows``), with Youden's J and the share classified correctly: a measure
    has one name and one number on the page and in the document. ``unequal``: the reference standard was done
    unequally by test result, and the rows name that share as the caption does. ``typed``: a table typed into the
    calculator, whose share is the table's."""
    return pd.DataFrame(
        measure_rows(acc, prevalence, unequal=unequal, typed=typed, every=True),
        columns=["Measure", "Estimate", "95% CI"],
    )


def _intervals(acc: Any, prevalence: float | None) -> str:
    """The methods of the intervals, in one line under the table of every measure."""
    score = any(
        ci is not None and ci.method == KOOPMAN for ci in (acc.lr_positive_ci, acc.lr_negative_ci)
    )
    text = "Intervals: Wilson score for the shares; Simel's log interval for the likelihood ratios"
    if score:
        text += ", and the one bound of Koopman's score interval at a zero cell"
    text += "; Woolf's log interval for the diagnostic odds ratio"
    if prevalence is not None and acc.ppv_at_prevalence is not None:
        carried = any(
            ci is not None and ci.method == SCORE_CARRIED
            for ci in (acc.ppv_at_prevalence_ci, acc.npv_at_prevalence_ci)
        )
        text += (
            "; a predictive value at the prevalence given carries its likelihood ratio's interval (Mercaldo's logit "
            "interval"
            + (
                "; from the score bound where a likelihood ratio is 0 or has no upper bound"
                if carried
                else ""
            )
            + "), the prevalence taken as known"
        )
    return text + "."


def _unequal(res: dict[str, Any]) -> bool:
    """Whether the reference standard was done unequally by test result in this result."""
    verified = res.get("verified")
    return bool(verified is not None and verified.partial)


def _methods(res: dict[str, Any], *more: str) -> str:
    """The references of a result's measures; with the caution of partial verification, its own."""
    cited = [*res["acc"].references, *more]
    if res.get("verification"):
        cited += [text for text in cite("begg1983") if text not in cited]
    return "Method: " + " | ".join(cited)


def _show_accuracy_yes_no(res: dict[str, Any]) -> None:
    acc = res["acc"]
    test, outcome = as_code(res["score"]), as_code(res["outcome"])
    positive, present = as_code(res["positive"]), as_code(res["pos_label"])
    cautions = [
        *(res.get("cautions") or []),
        *([res["verification"]] if res.get("verification") else []),
        *acc.warnings,
    ]
    with_it, without = acc.tp + acc.fn, acc.fp + acc.tn
    layout.headline(
        [
            _share_metric("Sensitivity", acc.sensitivity, acc.sensitivity_ci, acc.tp, with_it),
            _share_metric("Specificity", acc.specificity, acc.specificity_ci, acc.tn, without),
            Metric(
                "Participants",
                f"{res['n']:,}",
                f"{with_it:,} with the condition, {without:,} without",
            ),
        ],
        f"{test} = {positive} counted as a positive test, against {outcome} = {present}. "
        + m.SAMPLING_SENTENCE.get(acc.sampling, ""),
        len(cautions),
    )
    st.caption(f"Rows: {test} (the test). Columns: {outcome} (the reference standard).")
    table = pd.DataFrame(
        {
            f"Condition present ({res['pos_label']})": [acc.tp, acc.fn, with_it],
            f"Condition absent ({res['absent']})": [acc.fp, acc.tn, without],
            "Total": [acc.tp + acc.fp, acc.fn + acc.tn, res["n"]],
        },
        index=[
            f"Test positive ({res['positive']})",
            f"Test negative ({res['negative']})",
            "Total",
        ],
    )
    _counts_table(table)
    said = _predictive_caption(acc, res.get("prevalence"), _unequal(res))
    if said:
        st.caption(said)
    layout.warn(cautions)
    with st.expander("Details: every measure"):
        st.dataframe(
            _measure_rows(acc, res.get("prevalence"), _unequal(res)),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            _intervals(acc, res.get("prevalence"))
            + " A yes/no test has one point on the ROC plane: its area would be (sensitivity + specificity) / 2, "
            "nothing beyond the two shares above."
        )
        st.caption(_methods(res))
    help_popover(m.ACCURACY_READING)


def _accuracy_score(df: pd.DataFrame, *, mi: Any, guided: bool) -> None:
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
    outcome, present = _reference_standard(df, standards, c1, c2)
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
    # each option is drawn with the analysis that reads it: without a cut-off there is no predictive value to read
    sampling, prevalence = (
        (SAMPLING_NOT_STATED, None)
        if mode == m.NO_CUTOFF
        else _sampling_and_prevalence(outcome, share)
    )
    choices = (
        outcome,
        None if present is None else str(present),
        score,
        direction,
        mode,
        cutoff,
        prevalence,
        sampling,
    )
    plan = layout.kept(
        "diag_plan",
        choices,
        lambda: m.plan_accuracy(
            df, outcome, present, score, direction, mode, cutoff, prevalence, sampling
        ),
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
    # the settings in use only: a cut-off typed for another mode is no part of this result, nor the sampling of
    # a curve read without a cut-off
    config = _config(
        (outcome, score),
        accuracy_settings(
            m.NUMBER,
            present,
            direction=direction,
            mode=mode,
            cutoff=cutoff,
            prevalence=prevalence,
            sampling=sampling,
        ),
        *((YOUDEN_BOOTSTRAP.setting,) if drawn else ()),
    )
    if layout.run_button("diag_run", ready=ready, what="the accuracy of the test", plan=plan):
        try:
            result = accuracy_of(
                df,
                outcome,  # type: ignore[arg-type]
                present,
                score,  # type: ignore[arg-type]
                direction,  # type: ignore[arg-type]
                mode,
                cutoff,
                prevalence,
                sampling,
            )
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
    result = layout.result_place("diagnostic", config, guided=guided)
    if result is not None and result.get("index_form", m.NUMBER) == m.NUMBER:
        _show_accuracy(result)
        provenance("diagnostic")


def _share_metric(label: str, value: Any, ci: Any, count: int, of: int) -> Metric:
    """A share of the headline, with its count and its interval, written as the table of every measure writes it
    (2,499 of 2,500 read "100.0%" above and "more than 99.9%" in the table): ``share_parts``, the rule of both and of
    the dossier's block."""
    estimate, bounds = share_parts(value, ci)
    return Metric(
        label, estimate, f"{count:,} of {of:,}" + (f" · 95% CI {bounds}" if bounds else "")
    )


def _share(value: Any, ci: Any) -> str:
    """A share with its interval, as a sentence writes it: "40.8% (95% CI 34.5% to 47.6%)", or the share alone when
    it has no interval ("n/a (95% CI n/a% to n/a%)" stood for a predictive value without a value)."""
    estimate, bounds = share_parts(value, ci)
    return estimate + (f" (95% CI {bounds})" if bounds else "")


def _show_accuracy(res: dict[str, Any]) -> None:
    auc, acc, youden = res["auc"], res["acc"], res.get("youden")
    lower = res.get("direction") == m.LOWER
    score, outcome, present = (
        as_code(res["score"]),
        as_code(res["outcome"]),
        as_code(res["pos_label"]),
    )
    # the area and its interval as the dossier's block writes them (``auc_row``): the decimals that tell them from 0
    # and from 1, and no interval where DeLong's has no width (the caution says why)
    name, area, around = auc_row(auc)
    metrics = [Metric(name, area, f"95% CI {around}" if around else "")]
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
            _share_metric(
                "Sensitivity", acc.sensitivity, acc.sensitivity_ci, acc.tp, acc.tp + acc.fn
            ),
            _share_metric(
                "Specificity", acc.specificity, acc.specificity_ci, acc.tn, acc.tn + acc.fp
            ),
            Metric(
                "Cut-off",
                f"{_bound(lower)} {layout.num(res['cutoff'])}",
                origin,
                help="The test is called positive on this side of the cut-off.",
            ),
        ]
        if res.get("verification"):
            cautions.append(res["verification"])
        cautions += list(acc.warnings)
        words += " " + m.SAMPLING_SENTENCE.get(acc.sampling, "")
    layout.headline(metrics, words, len(cautions))
    left, right = st.columns(2)
    left.plotly_chart(res["fig"], width="stretch")
    with right:
        if acc is not None:
            positive = f"Test positive ({_bound(lower)} {layout.num(res['cutoff'])})"
            table = pd.DataFrame(
                {
                    f"Condition present ({res['pos_label']})": [acc.tp, acc.fn, acc.tp + acc.fn],
                    "Condition absent"
                    + (f" ({res['absent']})" if res.get("absent") is not None else ""): [
                        acc.fp,
                        acc.tn,
                        acc.fp + acc.tn,
                    ],
                    "Total": [acc.tp + acc.fp, acc.fn + acc.tn, acc.tp + acc.fp + acc.fn + acc.tn],
                },
                index=[positive, "Test negative", "Total"],
            )
            st.markdown("**At the cut-off**")
            _counts_table(table)
            said = _predictive_caption(acc, res.get("prevalence"), _unequal(res))
            if said:
                st.caption(said)
    layout.warn(cautions)
    if acc is not None:
        with st.expander("Details: every measure"):
            st.dataframe(
                _measure_rows(acc, res.get("prevalence"), _unequal(res)),
                hide_index=True,
                width="stretch",
            )
            st.caption(_intervals(acc, res.get("prevalence")))
            if res["mode"] == m.YOUDEN and youden:
                ci = youden.get("ci")
                if ci is not None:
                    st.caption(
                        f"In {youden.get('resamples') or 0:,} resamples of these data, 95% of the cut-offs found "
                        f"lie between {layout.num(ci.lower)} and {layout.num(ci.upper)} (a range, not a confidence "
                        "interval)."
                    )
                st.caption(youden["note"])
            st.caption(_methods(res, *auc["references"]))
    else:
        with st.expander("Details: the method"):
            st.caption("Method: " + " | ".join(auc["references"]))
    help_popover(
        "Sensitivity and specificity describe the test; predictive values describe what a result means for the "
        "person tested, and change with how common the condition is: read them at the share in the data only for a "
        "series of participants, otherwise at the prevalence of the setting. Likelihood ratios turn the odds before "
        "the test into the odds after it and do not depend on the prevalence. The AUC sums up how well the test separates "
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
    # the ICC does not read the order of its columns: Swap turns the differences round, and keeps its result
    icc_config = (tuple(sorted(columns)), model)
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
    # one "How to read this" for the two results of one Run, in its usual place: after the Details of the last
    # result and before its line "Computed on" (after the second "Add to the study dossier" it read as a footer)
    icc_follows = with_icc and status("icc", config=icc_config) == CURRENT
    readings: list[str] = []
    ba = layout.result_place("bland_altman", ba_config, guided=guided) if with_ba else None
    if ba is not None:
        readings.append(_show_bland_altman(ba, columns[0], columns[1], delta))
        if not icc_follows:
            help_popover("\n\n".join(readings))
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
        help_popover("\n\n".join(readings))
        provenance("icc")
    elif icc_follows and readings:  # the ICC's result was let go meanwhile: the reading still shows
        help_popover("\n\n".join(readings))


def _agreement_words(a: str, b: str, bias: str, reads: str, low: str, high: str) -> str:
    """The sentence under the limits of agreement, from the figures of the headline. Without a difference on
    average, or with the same difference in every pair, it says so ("reads 0 higher than ... the two differ by
    between 0 and 0" stood for two columns that hold the same values)."""
    none = not float(bias.replace(",", ""))
    size = bias.removeprefix("-")
    if low == high:  # the differences do not vary
        return (
            f"{a} and {b} hold the same value in every pair."
            if none
            else f"{a} reads {size} {reads} than {b} in every pair."
        )
    average = (
        f"{a} and {b} read the same on average"
        if none
        else f"{a} reads {size} {reads} than {b} on average"
    )
    return (
        f"{average}. For about 95 of 100 participants the two differ by between {low} and {high}."
    )


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
        _agreement_words(a, b, bias, reads, low, high),
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
        if low == high or not np.isfinite(ba.proportional_bias_p):
            # the differences do not vary: "100.0% within the limits (close to 95% by construction)" and "slope 0
            # per unit, p n/a" stood for two columns that hold the same values
            st.caption(
                f"SD of the differences {layout.num(ba.sd)} (ddof = 1): every pair differs by the same amount, "
                "which leaves no slope of the difference on the mean to read."
            )
        else:
            st.caption(
                f"SD of the differences {layout.num(ba.sd)} (ddof = 1); "
                f"{layout.percent(ba.within_loa_pct / 100)} of the differences lie within the limits (close to 95% "
                "by construction). Slope of the difference on the mean: "
                f"{layout.num(ba.proportional_bias_slope, 2)} per unit, p {layout.p_text(ba.proportional_bias_p)}"
                # said of a slope only: "it shows that the two spreads differ" stood for none
                + (
                    ": it shows that the two spreads differ, not why."
                    if ba.proportional_bias_p < 0.05
                    else "."
                )
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
    # three decimals for the ICC and for its interval; at its edges the caption says what is missing (two columns
    # that hold the same values read "1.000" over "(Shrout-Fleiss)" alone, a constant column "-0.000")
    value, bounds = layout.with_interval(res.icc, res.ci, places=3)
    if not known:
        around = "no value on these columns"
    elif bounds:
        around = f"{bounds} (Shrout-Fleiss)"
    else:
        around = (
            (
                "no interval: the columns hold the same values in these rows"
                if absolute
                # consistency: two columns 10 apart in every row give 1 as well
                else "no interval: apart from a constant shift, the columns hold the same values in these rows"
            )
            if res.icc >= 1
            else "no interval on these columns"
        )
    layout.headline(
        [
            Metric(
                m.ABSOLUTE
                if absolute
                else m.CONSISTENCY,  # the result's name: two results follow one Run
                value,
                around,
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
        text = (
            f"F = {layout.num(res.f_statistic)}, df = {res.df1:.0f}, {res.df2:.0f}, p "
            f"{layout.p_text(res.p_value)}."
            if np.isfinite(res.f_statistic)
            else "No F test: apart from a constant shift between them, the columns hold the same values in "
            "these rows, which leaves nothing to test against."
        )
        if known and bounds:  # read on the interval: without one, no word
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
    _, own, other, _ = m.rater_categories(df, first, second)
    return {
        "kappa": cohens_kappa(table.to_numpy(), weights=weights),
        "table": table,
        "order": [str(name) for name in table.index] if weights else [],
        "weights": weights,
        # how the ratings were read as categories, kept with the result: the plan says it, and is not drawn
        # with Guidance off (texts read as numbers beside a column of numbers; codes kept apart)
        "notes": [item.text for item in m.writings(df, first, second, own, other)],
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
    # three decimals for kappa and for its interval ("0.199, 95% CI 0.0237 to 0.3745")
    value, bounds = layout.with_interval(res.kappa, res.ci, places=3)
    layout.headline(
        [
            Metric(
                "Same category",
                layout.percent(res.observed_agreement),
                f"{same:,} of {res.n:,} rated by both",
            ),
            Metric(
                name,
                value,
                "0 by construction: one rater uses one category only"
                if fixed
                else " · ".join(part for part in (bounds, scale) if part),
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
    for note in result.get("notes") or ():  # which writings were read as one number, or kept apart
        st.caption(f":material/info: {note}")
    label = "Details: the standard error and the method"
    with st.expander(label if weights else label.replace(" and", ", the benchmark and")):
        text = (
            f"Standard error {layout.fixed(res.se, 3)} (normal approximation for the interval)."
            if np.isfinite(res.se)
            else "No standard error, and no interval"
            + (": one rater uses one category only." if fixed else " on this table.")
        )
        if not weights and not fixed and bounds:
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


__all__ = [
    "FORM_SETTING",
    "QUESTION_KEY",
    "accuracy_of",
    "accuracy_settings",
    "accuracy_yes_no_of",
    "kappa_of",
    "range_drawn",
    "run_diagnostic_accuracy",
]
