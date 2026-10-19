"""
The figures of a group comparison (Analyze > Hypothesis tests, the detailed view), built to be read side by side:
- the distribution of each group (box plot, with the significance brackets and n);
- the mean of each group with its 95% confidence interval;
- the histograms, as the percentage of each group, so that groups of different sizes compare;
- a normal Q-Q plot of each group, standardised within the group, which matches the normality check of each group
  in the result card.

One colour per group in every figure (the application's colour-blind-safe palette), the application's Plotly
template, and one height, so that a two-column grid stays readable on any screen. No Streamlit here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

import utils.visualization_utils as vu
from ui.theme import OKABE_ITO, PLOTLY_TEMPLATE_NAME, plotly_template

HEIGHT = 380
MARGIN = {"l": 55, "r": 20, "t": 60, "b": 50}


def group_order(df: pd.DataFrame, group: str) -> list[str]:
    """The groups in the order of the box plot (first appearance), as text."""
    return [str(g) for g in df[group].dropna().unique()]


def group_colors(levels: list[str]) -> dict[str, str]:
    return {level: OKABE_ITO[i % len(OKABE_ITO)] for i, level in enumerate(levels)}


def _values(df: pd.DataFrame, group: str, target: str) -> dict[str, np.ndarray]:
    key = df[group].astype(str)
    return {
        level: pd.to_numeric(df.loc[key == level, target], errors="coerce").dropna().to_numpy()
        for level in group_order(df, group)
    }


def _style(fig: go.Figure, title: str) -> go.Figure:
    plotly_template()
    fig.update_layout(
        template=PLOTLY_TEMPLATE_NAME,
        title={"text": title, "font": {"size": 15}, "x": 0.0, "xanchor": "left"},
        # the group variable is in the title: the legend needs no title, and sits on the right, clear of it
        legend={"title": {"text": ""}, "orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1.0},
        height=HEIGHT,
        width=None,
        margin=MARGIN,
    )
    return fig


def box_figure(
    df: pd.DataFrame, group: str, target: str, *, significant_pairs=None, reference: float | None = None
) -> go.Figure | None:
    """Each group's distribution, its mean as a dashed line, the significance brackets and n under each box."""
    plotly_template()  # registered before the helper below names it
    levels = group_order(df, group)
    colors = group_colors(levels)
    fig = vu.create_boxplot_with_significance_streamlit(
        df=df,
        group_col=group,
        value_col=target,
        significant_pairs=significant_pairs or [],
        title="",
        y_label=target,
        palette=[colors[level] for level in levels],
        height=HEIGHT,
        width=None,
        font_family="Source Sans 3, Source Sans Pro, Segoe UI, sans-serif",
        font_size=13,
        template=PLOTLY_TEMPLATE_NAME,
    )
    if fig is None:
        return None
    if reference is not None:
        fig.add_hline(y=reference, line_dash="dash", line_color="#B42318")
    fig.update_xaxes(title=group)
    return _style(fig, f"{target} by {group}")


def means_figure(df: pd.DataFrame, group: str, target: str, *, reference: float | None = None) -> go.Figure:
    """The mean of each group with its 95% confidence interval (t distribution)."""
    rows = []
    for level, values in _values(df, group, target).items():
        n = len(values)
        mean = float(np.mean(values)) if n else float("nan")
        half = float(stats.t.ppf(0.975, n - 1) * np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        rows.append((level, n, mean, half))
    colors = group_colors([r[0] for r in rows])
    fig = go.Figure(
        go.Scatter(
            x=[f"{level}<br>n = {n}" for level, n, _, _ in rows],
            y=[mean for _, _, mean, _ in rows],
            error_y={"type": "data", "array": [half for *_, half in rows], "thickness": 2, "width": 8},
            mode="markers",
            marker={"size": 12, "color": [colors[level] for level, *_ in rows]},
            customdata=[[mean - half, mean + half] for _, _, mean, half in rows],
            hovertemplate="%{x}<br>mean %{y:.3g}<br>95% CI %{customdata[0]:.3g} to %{customdata[1]:.3g}<extra></extra>",
            showlegend=False,
        )
    )
    if reference is not None:
        fig.add_hline(y=reference, line_dash="dash", line_color="#B42318")
    fig.update_xaxes(title=group, range=[-0.6, len(rows) - 0.4])  # the groups inside the panel, not on its edges
    fig.update_yaxes(title=f"Mean {target}")
    return _style(fig, f"Mean {target} with 95% CI")


def histogram_figure(df: pd.DataFrame, group: str, target: str, *, reference: float | None = None) -> go.Figure:
    """Overlaid histograms as the percentage of each group, semi-transparent with outlines."""
    levels = group_order(df, group)
    frame = df[[group, target]].dropna().assign(**{group: lambda d: d[group].astype(str)})
    fig = px.histogram(
        frame,
        x=target,
        color=group,
        barmode="overlay",
        histnorm="percent",
        opacity=0.55,
        color_discrete_map=group_colors(levels),
        category_orders={group: levels},
    )
    fig.update_traces(marker_line_width=0.6, marker_line_color="white")
    if reference is not None:
        fig.add_vline(x=reference, line_dash="dash", line_color="#B42318")
    fig.update_yaxes(title="% of the group")
    return _style(fig, f"Histogram of {target} (% of each group)")


def qq_figure(df: pd.DataFrame, group: str, target: str) -> go.Figure:
    """A normal Q-Q plot of each group, standardised within the group: on the line when the group is normal."""
    fig = go.Figure()
    values = _values(df, group, target)
    colors = group_colors(list(values))
    low = high = 0.0
    for level, x in values.items():
        if len(x) < 3 or np.std(x, ddof=1) == 0:
            continue
        z = (x - np.mean(x)) / np.std(x, ddof=1)
        theoretical, ordered = stats.probplot(z, dist="norm", fit=False)
        low, high = min(low, float(theoretical.min())), max(high, float(theoretical.max()))
        fig.add_trace(
            go.Scatter(
                x=theoretical,
                y=ordered,
                mode="markers",
                name=f"{level} (n = {len(x)})",
                marker={"size": 5, "color": colors[level], "opacity": 0.7},
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="normal",
            line={"color": "#5B7480", "dash": "dash"},
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(title="Normal quantiles")
    fig.update_yaxes(title="Standardised values")
    return _style(fig, f"Normal Q-Q plot of {target}, by {group}")


__all__ = [
    "HEIGHT",
    "box_figure",
    "group_colors",
    "group_order",
    "histogram_figure",
    "means_figure",
    "qq_figure",
]
