"""
Charts of the global data-quality score (Validate > Data Quality Dashboard).

- :func:`ring_figure`: the score as a ring built from the audit dimensions. Each dimension is a
  coloured segment sized by its weight; the grey gap after it is the points it loses. Hovering a
  segment explains it; the centre shows the score and its change since the previous check.
- :func:`loss_figure`: where the score loses points, one clickable bar per dimension.
- :func:`history_figure`: the score at each check of the dataset (one per revision of the data).

Colours of a dimension are the same in every chart. Figures leave backgrounds and text colours
to Streamlit's chart theme, so they follow light and dark mode.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence

import plotly.graph_objects as go

from explore.quality_score import (
    QualityScore,
    band,
    lost_text,
    ordered,
    points_text,
    score_text,
    weighted_score,
)
from ui.theme import BRAND, OKABE_ITO, TOKENS

DIMENSION_COLORS: dict[str, str] = {
    "Completeness": OKABE_ITO[0],  # blue
    "Statistical Validity": OKABE_ITO[2],  # green
    "Domain Validity": OKABE_ITO[4],  # reddish purple
    "Consistency": OKABE_ITO[5],  # sky blue
    "Uniformity": OKABE_ITO[1],  # orange
    "Uniqueness": BRAND["teal"],
}
LOST_COLOR = "rgba(128, 140, 145, 0.30)"
BAND_COLORS = {
    "excellent": TOKENS["success"],
    "good": TOKENS["success"],
    "fair": TOKENS["warning"],
    "poor": TOKENS["danger"],
}
_CHART_CONFIG = {"displayModeBar": False}


def dimension_color(name: str) -> str:
    return DIMENSION_COLORS.get(name, TOKENS["muted"])


def chart_config() -> dict:
    return dict(_CHART_CONFIG)


def ring_figure(
    quality: QualityScore,
    previous: float | None = None,
    *,
    height: int = 270,
    notes: Mapping[str, str] | None = None,
) -> go.Figure:
    """The global score as a ring of its dimensions; ``notes``: a line added to a dimension's hover."""
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    hovers: list[str] = []
    for d in quality.dimensions:
        if d.share <= 0:
            continue
        labels.append(d.name)
        values.append(d.earned)
        colors.append(dimension_color(d.name))
        hovers.append(
            f"<b>{d.name}</b><br>score {score_text(d.score)}%<br>weight {d.weight:g}: {d.share:.0%} of the global score"
            f"<br>adds {d.earned:.2f} points, loses {points_text(d.lost)}"
            + (f"<br>{html.escape(notes[d.name])}" if notes and d.name in notes else "")
        )
        if d.lost > 0:
            labels.append(f"{d.name}: points lost")
            values.append(d.lost)
            colors.append(LOST_COLOR)
            hovers.append(
                f"<b>{d.name}: points lost</b><br>{points_text(d.lost)} of 100<br>{lost_text(100 - d.score)}% of its checks fail"
            )
    if not values:  # every weight at zero
        labels, values, colors, hovers = (
            ["no weight"],
            [1.0],
            [LOST_COLOR],
            ["Every weight is zero."],
        )

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.74,
            sort=False,
            direction="clockwise",
            marker={"colors": colors, "line": {"width": 0}},
            textinfo="none",
            customdata=hovers,
            hovertemplate="%{customdata}<extra></extra>",
        )
    )
    level = band(quality.total)
    annotations = [
        {"text": f"<b>{score_text(quality.total)}</b>", "font": {"size": 40}, "y": 0.58},
        {"text": f"/ 100 · {level}", "font": {"size": 13, "color": BAND_COLORS[level]}, "y": 0.40},
    ]
    if previous is not None and abs(quality.total - previous) >= 0.05:
        delta = quality.total - previous
        annotations.append(
            {
                "text": f"{'▲' if delta > 0 else '▼'} {abs(delta):.1f} since the last check",
                "font": {"size": 12, "color": TOKENS["success"] if delta > 0 else TOKENS["danger"]},
                "y": 0.28,
            }
        )
    for a in annotations:
        a.update({"x": 0.5, "xref": "paper", "yref": "paper", "showarrow": False})
    fig.update_layout(
        annotations=annotations,
        showlegend=False,
        height=height,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
    )
    return fig


def loss_figure(quality: QualityScore) -> go.Figure | None:
    """Points lost per dimension, the most costly on top (a loss that rounds to nothing included: a failing check
    is never hidden); ``None`` when no dimension loses points."""
    dims = [d for d in quality.by_loss() if d.lost > 0]
    if not dims:
        return None
    fig = go.Figure(
        go.Bar(
            x=[d.lost for d in dims],
            y=[d.name for d in dims],
            orientation="h",
            marker={"color": [dimension_color(d.name) for d in dims]},
            text=[f"−{points_text(d.lost)}" for d in dims],
            textposition="outside",
            cliponaxis=False,
            customdata=[[d.name, score_text(d.score), d.share, points_text(d.lost)] for d in dims],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>loses %{customdata[3]} points of 100"
                "<br>score %{customdata[1]}% · %{customdata[2]:.0%} of the global score"
                "<br>Click to see where<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=70 + 40 * len(dims),
        margin={"l": 8, "r": 48, "t": 8, "b": 36},
        xaxis={
            "title": {"text": "points lost (of 100)"},
            "rangemode": "tozero",
            "fixedrange": True,
        },
        yaxis={"autorange": "reversed", "fixedrange": True},
        bargap=0.35,
    )
    return fig


def _short(label: str, width: int = 26) -> str:
    return label if len(label) <= width else label[: width - 1] + "…"


def history_figure(
    checks: Sequence[Mapping], weights: Mapping[str, float], *, height: int = 280
) -> go.Figure:
    """Global score at each check (current weights), with each dimension available from the legend."""
    xs = list(range(len(checks)))
    labels = [str(c.get("label", "")) for c in checks]
    ticks = [f"{i}. {_short(label)}" for i, label in enumerate(labels)]
    scores = [weighted_score(c["metrics"], weights) for c in checks]
    totals = [s.total for s in scores]
    hover = [
        f"<b>{t:.1f}</b> / 100<br>{'data ' + label if i == 0 else 'after: ' + label}<br>checked {str(c.get('at', ''))[11:16]}"
        for i, (t, label, c) in enumerate(zip(totals, labels, checks, strict=True))
    ]
    fig = go.Figure(
        go.Scatter(
            x=xs,
            y=totals,
            mode="lines+markers",
            name="Global score",
            line={"color": BRAND["teal"], "width": 3},
            marker={"size": 9},
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        )
    )
    names = ordered({n for c in checks for n in c["metrics"]})
    for name in names:
        values = [c["metrics"].get(name) for c in checks]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=values,
                mode="lines+markers",
                name=name,
                line={"color": dimension_color(name), "width": 1.5, "dash": "dot"},
                marker={"size": 6},
                visible="legendonly",
                hovertemplate=f"{name}: %{{y:.1f}}%<extra></extra>",
            )
        )
    low = min([*totals, *[v for c in checks for v in c["metrics"].values()]], default=0.0)
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 30, "b": 8},
        xaxis={"tickmode": "array", "tickvals": xs, "ticktext": ticks, "fixedrange": True},
        yaxis={
            "title": {"text": "score"},
            "range": [max(0.0, low - 5.0), 100.5],
            "fixedrange": True,
        },
        legend={"orientation": "h", "y": 1.08, "x": 0, "font": {"size": 11}},
        hovermode="closest",
    )
    return fig


__all__ = [
    "DIMENSION_COLORS",
    "chart_config",
    "dimension_color",
    "history_figure",
    "loss_figure",
    "ring_figure",
]
