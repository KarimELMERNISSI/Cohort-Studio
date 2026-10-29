"""
How the results of the Survival & regression page read: the effect first (the comparison's ratios with their
interval and p-value), then the figure (Kaplan-Meier curves with the numbers at risk, or the model's ratios on a log
scale), the warnings of the fit, then every term of the model and its checks (ui.guidance.outcomes.diagnostics), and
under each stored result what it was computed on. Observed and pooled (multiply imputed) results read the same way.
"""

from __future__ import annotations

import html
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from biostats.regression import APPARENT_AUC_NOTE
from biostats.survival import event_indicator
from core.registry import as_code
from ui.components import help_popover, status_chip
from ui.components.mi_mode import pooled_frame, show_mi_notes, show_pooled_terms, show_tests
from ui.components.text import plain
from ui.guidance.outcomes import HELP, ComparisonTerm, comparison_terms, diagnostics, listing
from ui.results import provenance
from ui.theme import OKABE_ITO

PH_TEST_IMPUTATIONS = (
    5  # the proportional-hazards test (lifelines) runs on the first imputed datasets only
)
MUTED = "#6B7C86"  # adjustment terms: about 4.2:1 on the light and on the dark surface alike
BAND_ALPHA = 0.12
PLOT_HEIGHT = 380  # of the plotting area, in pixels
LABEL_GAP = (
    30  # pixels between an "At risk" label and the time axis: the count at time 0 is centred on it
)

COX_FORMATS = {"log_hr": "{:.3f}", "se": "{:.3f}", "hr": "{:.3f}", "ci_low": "{:.3f}", "ci_high": "{:.3f}", "z": "{:.2f}", "p": "{:.4f}"}  # fmt: skip
LOGIT_FORMATS = {"coef": "{:.3f}", "se": "{:.3f}", "odds_ratio": "{:.3f}", "ci_low": "{:.3f}", "ci_high": "{:.3f}", "z": "{:.2f}", "p": "{:.4f}"}  # fmt: skip
COUNT_FORMATS = {"coef": "{:.3f}", "se": "{:.3f}", "rate_ratio": "{:.3f}", "ci_low": "{:.3f}", "ci_high": "{:.3f}", "z": "{:.2f}", "p": "{:.4f}"}  # fmt: skip


@dataclass(frozen=True)
class Comparison:
    """What a model's result is read by: the comparison, its values in the data the model read, the adjustment."""

    compare: str | None
    adjust: tuple[str, ...] = ()
    values: pd.Series | None = None
    reference: str | None = None  # the comparison's reference level, when it is a factor

    def terms(self, terms: Sequence[str]) -> list[ComparisonTerm]:
        if self.values is None:
            return []
        return comparison_terms(terms, self.compare, self.values, self.reference)

    @property
    def words(self) -> str:
        if not self.compare:
            return "No comparison chosen: every term of the model reads alike, in Model details."
        others = [a for a in self.adjust if a != self.compare]
        return f"Adjusted for {listing(others)}." if others else "Unadjusted."


@dataclass(frozen=True)
class Metric:
    label: str
    value: str
    caption: str = ""
    help: str | None = None


def _p(p: Any) -> str:
    if p is None or not math.isfinite(float(p)):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def _ratio(value: Any) -> str:
    return f"{value:.2f}" if value is not None and math.isfinite(float(value)) else "n/a"


def _median(value: float | None) -> str:
    """
    The median survival time, or 'not reached' when the curve stays above 0.5: in the time's unit, without an
    exponent (1,119 days, not 1.12e+03), to two decimals at most (a midpoint ends in .5; 2.37 years).
    """
    return "not reached" if value is None else f"{value:,.2f}".rstrip("0").rstrip(".")


def _table(df: pd.DataFrame, formats: dict) -> None:
    st.dataframe(df.style.format(formats, na_rep=""), hide_index=True, width="stretch")


def _effects(table: pd.DataFrame, ratio: str, comparison: Comparison, name: str) -> list[Metric]:
    """The comparison's ratios, one metric per term (the first level is the reference)."""
    rows = table.set_index("term")
    return [
        Metric(
            f"{name} · {plain(term.label)}",
            _ratio(rows.at[term.term, ratio]),
            f"95% CI {_ratio(rows.at[term.term, 'ci_low'])} to {_ratio(rows.at[term.term, 'ci_high'])} · "
            f"p {_p(rows.at[term.term, 'p'])}",
        )
        for term in comparison.terms(list(table["term"]))
    ]


def _headline(metrics: list[Metric], words: str, warnings: int) -> None:
    """The result first: the comparison's effect, the test and the counts, then what the model is adjusted for."""
    with st.container(border=True):
        for start in range(0, len(metrics), 4):
            row = metrics[start : start + 4]
            for column, metric in zip(st.columns(len(row)), row, strict=True):
                column.metric(metric.label, metric.value, help=metric.help)
                if metric.caption:
                    column.caption(metric.caption)
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption(words)
            if warnings:
                status_chip(f"{warnings} warning{'s' if warnings > 1 else ''} below", "caution")


# ----------------------------------------------------------------------------- figures


def _rgba(colour: str, alpha: float) -> str:
    r, g, b = (int(colour[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _ticks(top: float, count: int = 5) -> np.ndarray:
    """
    Round ticks from 0 to ``top``, about ``count`` of them: 1, 2, 2.5 or 5 times a power of ten apart (1,450 days:
    every 250, not every 500).
    """
    if not math.isfinite(top) or top <= 0:
        return np.array([0.0])
    magnitude = 10 ** math.floor(math.log10(top / count))
    step = min((m * magnitude for m in (1, 2, 2.5, 5, 10)), key=lambda s: abs(top / s - count))
    return np.arange(0, top + step * 1e-9, step)


@dataclass(frozen=True)
class Curve:
    label: str
    name: str  # in the legend
    times: np.ndarray
    survival: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    end: float | None = (
        None  # the last follow-up time: the curve runs on to it, flat, after its last event
    )
    censored: np.ndarray | None = (
        None  # the times at which rows were censored: a tick on the curve each
    )


def curve_rows(
    frame: pd.DataFrame, time_col: str, event_col: str, group: str | None
) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """Per group, the times and events of the rows the curves read (a known time >= 0, a known event)."""
    columns = [time_col, event_col, *([group] if group else [])]
    data = frame[columns].dropna()
    try:
        events = event_indicator(data[event_col])
    except ValueError:
        return None
    times = pd.to_numeric(data[time_col], errors="coerce").to_numpy(dtype=float)
    keep = ~np.isnan(times) & ~np.isnan(events) & (times >= 0)
    times, events = times[keep], events[keep]
    if not group:
        return {"all": (times, events)}
    labels = data[group].astype(str).to_numpy()[keep]
    return {
        level: (times[labels == level], events[labels == level]) for level in sorted(set(labels))
    }


def _at_risk(
    follow_up: dict[str, tuple[np.ndarray, np.ndarray]], ticks: np.ndarray
) -> dict[str, list[int]]:
    """Rows still followed at each tick, per group."""
    return {
        label: [int((times >= tick).sum()) for tick in ticks]
        for label, (times, _) in follow_up.items()
    }


def km_figure(
    curves: Sequence[Curve], *, time_label: str, at_risk: dict[str, list[int]] | None = None
) -> go.Figure:
    """
    Kaplan-Meier curves, each with its pointwise band in its own colour (Okabe-Ito, in the order of the sorted
    levels), run on to its last follow-up after its last event, with a tick where rows were censored, and the numbers
    at risk under the axis, at its ticks.
    """
    order = sorted(c.label for c in curves)
    colour = {label: OKABE_ITO[i % len(OKABE_ITO)] for i, label in enumerate(order)}
    fig = go.Figure()
    top = 0.0
    for curve in curves:
        times = np.concatenate([[0.0], np.asarray(curve.times, dtype=float)])
        survival = np.concatenate([[1.0], np.asarray(curve.survival, dtype=float)])
        band = (
            np.concatenate([[1.0], np.asarray(curve.lower, dtype=float)]),
            np.concatenate([[1.0], np.asarray(curve.upper, dtype=float)]),
        )
        if (
            curve.end is not None and curve.end > times[-1]
        ):  # flat to the last follow-up, as the estimate is
            times = np.append(times, curve.end)
            survival = np.append(survival, survival[-1])
            band = (np.append(band[0], band[0][-1]), np.append(band[1], band[1][-1]))
        known = ~(np.isnan(band[0]) | np.isnan(band[1]))
        top = max(top, float(np.nanmax(times)))
        fig.add_trace(
            go.Scatter(
                x=times,
                y=survival,
                mode="lines",
                line={"shape": "hv", "width": 2, "color": colour[curve.label]},
                name=html.escape(curve.name),
                meta=html.escape(
                    curve.label
                ),  # not in the template itself, where %{...} would be read
                hovertemplate="%{x:.4g}: %{y:.3f}<extra>%{meta}</extra>",
            )
        )
        if curve.censored is not None and len(curve.censored):
            at = (
                np.searchsorted(times, curve.censored, side="right") - 1
            )  # the curve's value at each censoring
            fig.add_trace(
                go.Scatter(
                    x=curve.censored,
                    y=survival[np.clip(at, 0, len(survival) - 1)],
                    mode="markers",
                    # an open symbol is stroked in marker.color: unset, it took the palette's next colour
                    marker={
                        "symbol": "line-ns-open",
                        "size": 6,
                        "color": colour[curve.label],
                        "line": {"width": 1, "color": colour[curve.label]},
                    },
                    meta=html.escape(curve.label),
                    hovertemplate="censored at %{x:.4g}<extra>%{meta}</extra>",
                    showlegend=False,
                )
            )
        t = times[known]
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([t, t[::-1]]),
                y=np.concatenate([band[1][known], band[0][known][::-1]]),
                fill="toself",
                line={"width": 0, "shape": "hv"},
                fillcolor=_rgba(colour[curve.label], BAND_ALPHA),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    ticks = _ticks(top)
    rows = [(label, at_risk[label]) for label in order if at_risk and label in at_risk]
    below = 64 + 20 * (len(rows) + 1) if rows else 50
    if rows:  # the labels left of the axis, clear of the counts at time 0 (centred on it)
        fig.add_annotation(
            x=0, xref="paper", xanchor="right", xshift=-LABEL_GAP, y=-62 / PLOT_HEIGHT, yref="paper",
            yanchor="top", text="<b>At risk</b>", showarrow=False, font={"size": 12},
        )  # fmt: skip
        for i, (label, counts) in enumerate(rows):
            y = -(62 + 20 * (i + 1)) / PLOT_HEIGHT
            fig.add_annotation(
                x=0, xref="paper", xanchor="right", xshift=-LABEL_GAP, y=y, yref="paper", yanchor="top",
                showarrow=False, font={"size": 12},
                text=f"<span style='color:{colour[label]}'>━</span> {html.escape(label[:24])}",
            )  # fmt: skip
            for tick, n in zip(ticks, counts, strict=True):
                fig.add_annotation(
                    x=tick, xref="x", xanchor="center", y=y, yref="paper", yanchor="top", showarrow=False,
                    text=f"{n:,}", font={"size": 12},
                )  # fmt: skip
    longest = max((len(label[:24]) for label, _ in rows), default=0)
    fig.update_xaxes(title_text=html.escape(time_label), tickvals=ticks, range=[0, top * 1.02 or 1])
    fig.update_yaxes(title_text="survival probability", range=[0, 1.02])
    fig.update_layout(
        height=PLOT_HEIGHT + 30 + below,
        margin={"t": 30, "b": below, "l": max(70, LABEL_GAP + 24 + 7 * longest), "r": 30},
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0},
    )
    return fig


def forest_figure(
    table: pd.DataFrame, *, ratio: str, highlight: set[str], axis_title: str
) -> go.Figure:
    """
    The model's ratios on a log scale, with their 95% intervals: the comparison's terms filled and in colour, the
    adjustment terms hollow and grey; a hairline at 1. Terms without a finite interval are left to the table.
    """
    data = table[table["term"] != "const"]
    values = data[[ratio, "ci_low", "ci_high"]].astype(float)
    data = data[np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)]
    first = data["term"].isin(highlight)
    data = pd.concat([data[first], data[~first]]).iloc[::-1]  # the comparison at the top
    fig = go.Figure()
    fig.add_vline(x=1, line_width=1, line_color=MUTED)
    labels = [html.escape(str(t)) for t in data["term"]]
    for emphasised in (True, False):
        part = data[data["term"].isin(highlight) == emphasised]
        if part.empty:
            continue
        strong = emphasised or not highlight
        colour = OKABE_ITO[0] if strong else MUTED
        fig.add_trace(
            go.Scatter(
                x=part[ratio],
                y=[html.escape(str(t)) for t in part["term"]],
                mode="markers",
                marker={
                    "size": 10 if strong else 8,
                    "color": colour,
                    "symbol": "circle" if strong else "circle-open",
                    "line": {"width": 2, "color": colour},
                },
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": part["ci_high"] - part[ratio],
                    "arrayminus": part[ratio] - part["ci_low"],
                    "color": colour,
                    "thickness": 2 if strong else 1.5,
                    "width": 0,
                },
                customdata=np.column_stack([part["ci_low"], part["ci_high"], part["p"]]),
                hovertemplate="%{y}: %{x:.2f} (95% CI %{customdata[0]:.2f} to %{customdata[1]:.2f}), "
                "p = %{customdata[2]:.3g}<extra></extra>",
                name="Comparison" if emphasised else ("Adjustment terms" if highlight else "Terms"),
                showlegend=bool(highlight) and len(part) < len(data),
            )
        )
    lo = min(float(data["ci_low"].min()), 1.0) if len(data) else 0.5
    hi = max(float(data["ci_high"].max()), 1.0) if len(data) else 2.0
    ticks = _ratio_ticks(lo, hi)
    fig.update_xaxes(
        type="log",
        title_text=axis_title,
        range=[math.log10(lo / 1.15), math.log10(hi * 1.15)],
        tickvals=ticks,
        ticktext=[f"{t:g}" for t in ticks],
    )
    fig.update_yaxes(categoryorder="array", categoryarray=labels, automargin=True)
    longest = max((len(str(t)) for t in data["term"]), default=0)
    fig.update_layout(
        height=max(200, 90 + 34 * len(data)),
        margin={"t": 30, "b": 50, "r": 30, "l": min(280, 24 + 7 * longest)},
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0},
    )
    return fig


def _decades(steps: Sequence[float]) -> list[float]:
    return [m * 10.0**k for k in range(-4, 5) for m in steps]


def _ratio_ticks(lo: float, hi: float) -> list[float]:
    """
    Round ratios from ``lo`` to ``hi`` for a log axis, 1 among them: 0.1, 0.2, 0.5, 1, 2, 5 on a wide range, halves
    and doubles on a middle one (0.25, 0.5, 1, 2, 4), finer steps on a narrow one (0.8, 0.9, 1), never more than 8.
    Plotly's own log ticks label 0.5 to 0.9 as "5 ... 9".
    """
    sequences = (
        _decades((1, 2, 5)),
        [2.0**k for k in range(-10, 11)],
        _decades((1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8)),
        _decades((1, 1.1, 1.2, 1.3, 1.5, 1.7, 2, 2.5, 3, 4, 5, 6, 7, 8, 9)),
    )
    inside: list[float] = [1.0]
    for sequence in sequences:
        inside = sorted({round(t, 6) for t in sequence if lo <= t <= hi})
        if len(inside) >= 4:
            break
    while len(inside) > 8:
        centre = inside.index(1.0) if 1.0 in inside else 0
        inside = [t for i, t in enumerate(inside) if (i - centre) % 2 == 0]
    return inside


# ----------------------------------------------------------------------------- time to event


def _curves(km: Any, pooled: bool, follow_up: dict | None = None) -> list[Curve]:
    """The curves to draw; with the rows they read (``follow_up``), observed curves get their censoring ticks."""
    if pooled:
        return [
            Curve(
                label,
                f"{label} (pooled)",
                c["time"].to_numpy(),
                c["survival"].to_numpy(),
                c["lower"].to_numpy(dtype=float),
                c["upper"].to_numpy(dtype=float),
            )  # fmt: skip
            for label, c in km.extra["curves"].items()
        ]
    return [
        Curve(
            c.label,
            f"{c.label} (n={c.n}, events={c.events})",
            c.times,
            c.survival,
            c.lower,
            c.upper,
            end=getattr(
                c, "last_time", None
            ),  # a curve stored before kept none: it stops at its last event
            censored=np.unique(follow_up[c.label][0][follow_up[c.label][1] == 0])
            if follow_up and c.label in follow_up
            else None,
        )
        for c in km.curves
    ]


def _median_ci(lower: float | None, upper: float | None) -> str:
    """The median's interval: 'not reached' alone when neither end is reached (the lower one is reached first)."""
    if lower is None and upper is None:
        return "not reached"
    return f"{_median(lower)} to {_median(upper)}"


def _group_table(km: Any, pooled: bool) -> None:
    if pooled:
        intervals = km.extra.get("median_ci") or {}  # a result pooled before: none
        rows = [
            {
                "group": label,
                "median survival (pooled curve)": _median(km.extra["medians"].get(label)),
                "median 95% CI (pooled band)": _median_ci(*intervals[label])
                if label in intervals
                else "n/a",
                "survival at last time": float(curve["survival"].iloc[-1]),
            }
            for label, curve in km.extra["curves"].items()
        ]
    else:
        rows = [
            {
                "group": c.label,
                "n": c.n,
                "events": c.events,
                "median survival": _median(c.median),
                f"median {c.level:.0%} CI": _median_ci(c.median_lower, c.median_upper),
                "survival at last time": float(c.survival[-1]),
            }
            for c in km.curves
        ]
    _table(pd.DataFrame(rows), {"survival at last time": "{:.3f}"})


def show_km_observed(km: Any) -> None:
    """A Kaplan-Meier result of the observed data, as the complete-case comparison under a pooled one."""
    st.plotly_chart(km_figure(_curves(km, False), time_label="time"), width="stretch")
    _group_table(km, False)
    for warning in km.warnings:
        st.warning(plain(warning))


def _logrank(km: Any, pooled: bool) -> Metric | None:
    if pooled:
        test = km.tests.get("log_rank")
        if test is None:
            return None
        return Metric(
            "Log-rank test, pooled",
            f"p {_p(test.p_value)}",
            f"D2: F = {test.statistic:.3g}, df = {test.df1:.0f}, {test.df2:.1f}",
            "Equal survival in every group; the test statistics of the imputed datasets combined (D2).",
        )
    if km.logrank_statistic is None:
        return None
    return Metric(
        "Log-rank test",
        f"p {_p(km.logrank_p)}",
        f"chi² = {km.logrank_statistic:.2f}, df = {km.logrank_df}",
        "Equal survival in every group, over the whole follow-up.",
    )


def show_time_to_event(
    km: Any,
    cox: Any,
    *,
    keys: tuple[str, str],
    comparison: Comparison,
    pooled: bool,
    time_col: str,
    event_col: str = "",
    follow_up: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    cox_chosen: bool = False,
    cox_fits: bool = False,
) -> None:
    """
    The curves and the Cox model of one time-to-event analysis: hazard ratios and the log-rank test first.
    ``follow_up`` (curve_rows, observed data): the rows the curves read, for the numbers at risk and the censoring
    ticks. ``cox_chosen``: the Cox model is among the analyses; ``cox_fits``: it has a variable to read, so a missing
    one has not been fitted with these variables (or could not be).
    """
    if km is None and cox is None:
        return
    metrics: list[Metric] = []
    table = None
    if cox is not None:
        table = pooled_frame(cox, ratio_label="hr") if pooled else cox.table
        metrics += _effects(table, "hr", comparison, "Hazard ratio")
    adjusted = cox is not None and any(a != comparison.compare for a in comparison.adjust)
    if km is not None and (test := _logrank(km, pooled)) is not None:
        if adjusted:  # beside an adjusted ratio, the curves and their test are not
            test = Metric(f"{test.label} (unadjusted)", test.value, test.caption, test.help)
        metrics.append(test)
    if (
        not pooled
    ):  # the rows of the ratio beside it: the model's, which can be fewer than the curves'
        curves = (
            (sum(c.n for c in km.curves), sum(c.events for c in km.curves))
            if km is not None
            else None
        )
        if cox is not None:
            also = (
                f"; the curves: {curves[1]:,} of {curves[0]:,}"
                if curves and curves[0] != cox.n
                else ""
            )
            metrics.append(
                Metric("Events", f"{cox.events:,}", f"of {cox.n:,} rows in the Cox model{also}")
            )
        elif curves is not None:
            metrics.append(Metric("Events", f"{curves[1]:,}", f"of {curves[0]:,} rows analysed"))
    # the curves and the model read the same rows: a warning both give (a negative time left out) is shown once
    notes = list(
        dict.fromkeys(
            [*(getattr(km, "warnings", None) or []), *(getattr(cox, "warnings", None) or [])]
        )
    )
    if cox is not None:
        words = (
            f"Hazard of {as_code(event_col)}. {comparison.words}" if event_col else comparison.words
        )
    elif cox_chosen and not cox_fits:
        words = "The Cox model needs a variable to compare by or to adjust for."
    elif cox_chosen:
        words = "No Cox model with these variables yet: Run analysis fits it (a message above says why if it failed)."
    else:
        words = "Kaplan-Meier curves only: choose the Cox model above for a hazard ratio."
    _headline(metrics, words, len(notes))
    if km is not None:
        at_risk = None
        if not pooled and follow_up:
            top = max(
                (float(times.max()) for times, _ in follow_up.values() if len(times)), default=0.0
            )
            at_risk = _at_risk(follow_up, _ticks(top))
        st.plotly_chart(
            km_figure(_curves(km, pooled, follow_up), time_label=time_col, at_risk=at_risk),
            width="stretch",
        )
        _group_table(km, pooled)
    if pooled:  # one pooling line and the warnings once, then each complete-case comparison
        if km is not None:
            show_mi_notes(
                km,
                complete_case=show_km_observed,
                label="Complete-case analysis, for comparison: curves",
                warnings=False,
            )
        if cox is not None:
            show_mi_notes(
                cox,
                complete_case=lambda cc: _table(cc.table, COX_FORMATS),
                label="Complete-case analysis, for comparison: Cox model",
                caption=km is None,
                warnings=False,
            )
    for warning in notes:
        st.warning(plain(warning))
    if cox is not None:
        with st.expander("Model details: every term of the Cox model"):
            if pooled:
                show_pooled_terms(cox, ratio_label="hazard_ratio", alt="Pooled hazard ratios")
                show_tests(cox, title="Tests of each factor (all its levels at once)")
                st.caption(
                    f"The proportional-hazards test ran on the first {PH_TEST_IMPUTATIONS} imputed datasets; its "
                    "warnings say on how many it flagged a term."
                )
            else:
                _table(cox.table, COX_FORMATS)
                if cox.reference_levels:
                    st.caption(
                        "Reference levels: "
                        + "; ".join(
                            f"{plain(k)}: {plain(v)}" for k, v in cox.reference_levels.items()
                        )
                    )
                st.caption("Method: " + " | ".join(cox.references))
        if not pooled:
            with st.expander("Checks"):
                _diagnostics(cox)
                if cox.ph_test is not None:
                    st.markdown("**Proportional hazards (Grambsch-Therneau), per term**")
                    _table(cox.ph_test, {"statistic": "{:.3f}", "p": "{:.4f}"})
                    st.caption(
                        "A small p-value questions a constant hazard ratio for that term over follow-up."
                    )
    if km is not None and not pooled:
        st.caption("Method: " + " | ".join(km.references))
    with st.container(horizontal=True):
        if km is not None:
            help_popover(HELP["read_km"], label="How to read the curves")
        if cox is not None:
            help_popover(HELP["read_cox"], label="How to read the hazard ratios")
    for key, result, name in zip(
        keys, (km, cox), ("Kaplan-Meier curves", "Cox model"), strict=True
    ):
        if (
            result is not None
        ):  # each stored result, said which: each has its own button for the dossier
            st.caption(f"**{name}**")
            provenance(key)


# ----------------------------------------------------------------------------- binary and count outcomes


def _diagnostics(result: Any) -> None:
    rows = diagnostics(result)
    if rows:
        st.dataframe(
            pd.DataFrame(rows, columns=["Check", "Value", "How to read it"]),
            hide_index=True,
            width="stretch",
            column_config={"How to read it": st.column_config.TextColumn(width="large")},
        )


def _show_model(
    res: Any,
    *,
    key: str,
    comparison: Comparison,
    pooled: bool,
    ratio: str,
    name: str,
    formats: dict,
    summary: Callable[[Any], str],
    read: str,
    notes: tuple[str, ...] = (),
    subject: str = "",
) -> None:
    """A logistic or count model: the comparison's ratios first, the forest of every ratio, then details and checks."""
    if res is None:
        return
    table = pooled_frame(res, ratio_label=ratio) if pooled else res.table
    metrics = _effects(table, ratio, comparison, name)
    if not pooled:
        metrics.append(Metric("Events", f"{res.events:,}", f"of {res.n:,} rows analysed"))
    _headline(metrics, f"{subject} {comparison.words}".strip(), len(res.warnings))
    highlight = {t.term for t in comparison.terms(list(table["term"]))}
    st.plotly_chart(
        forest_figure(table, ratio=ratio, highlight=highlight, axis_title=f"{name} (log scale)"),
        width="stretch",
    )
    if pooled:
        show_mi_notes(res, complete_case=lambda cc: _table(cc.table, formats))
    else:
        for warning in res.warnings:
            st.warning(plain(warning))
    with st.expander("Model details: every term"):
        if pooled:
            show_pooled_terms(res, ratio_label=ratio, alt=f"Pooled {name.lower()}s")
            show_tests(res, title="Tests of each factor (all its levels at once)")
        else:
            st.caption(summary(res))
            _table(res.table, formats)
            st.caption("Method: " + " | ".join(res.references))
    if not pooled:
        with st.expander("Checks"):
            _diagnostics(res)
            for note in notes:
                st.caption(note)
    help_popover(read, label=f"How to read the {name.lower()}s")
    provenance(key)


def _reference_words(res: Any) -> str:
    return "; ".join(f"{plain(k)}: {plain(v)}" for k, v in res.reference_levels.items())


def show_logistic(
    res: Any,
    *,
    key: str,
    comparison: Comparison,
    pooled: bool,
    outcome: str = "",
    event: Any = None,
) -> None:
    """``outcome`` and ``event``: what the odds are of, said with the result."""
    _show_model(
        res,
        key=key,
        comparison=comparison,
        pooled=pooled,
        ratio="odds_ratio",
        name="Odds ratio",
        formats=LOGIT_FORMATS,
        summary=lambda r: (
            f"n = {r.n:,}, events = {r.events:,}; reference levels: {_reference_words(r)}"
        ),
        read=HELP["read_logistic"],
        notes=(APPARENT_AUC_NOTE,),
        subject=f"Odds of {as_code(outcome)} = {as_code(event)}." if outcome else "",
    )


def show_count(
    res: Any,
    *,
    key: str,
    comparison: Comparison,
    pooled: bool,
    rates: bool = True,
    outcome: str = "",
) -> None:
    """``rates``: person-time was given, so the ratios compare rates; without it they compare mean counts."""
    _show_model(
        res,
        key=key,
        comparison=comparison,
        pooled=pooled,
        ratio="rate_ratio",
        name="Incidence-rate ratio" if rates else "Ratio of mean counts",
        formats=COUNT_FORMATS,
        summary=lambda r: (
            f"n = {r.n:,}, events = {r.events:,}"
            + (f"; reference levels: {_reference_words(r)}" if r.reference_levels else "")
        ),
        read=HELP["read_count"],
        subject=f"{'Rate' if rates else 'Mean count'} of {as_code(outcome)}." if outcome else "",
    )


__all__ = [
    "PH_TEST_IMPUTATIONS",
    "Comparison",
    "Curve",
    "curve_rows",
    "forest_figure",
    "km_figure",
    "show_count",
    "show_km_observed",
    "show_logistic",
    "show_time_to_event",
]
