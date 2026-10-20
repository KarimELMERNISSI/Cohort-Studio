"""
The participant flow diagram in the app (STROBE item 13): the boxes of core.attrition drawn with Plotly. A click
on a box shows its details below the diagram: the definitions of the criteria and how many rows each excluded
first (a menu offers the same details without the mouse).
"""

from __future__ import annotations

from collections.abc import Sequence
from textwrap import wrap

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.attrition import (
    ANALYSIS,
    POPULATION,
    AnalysisCount,
    FlowExit,
    ParticipantFlow,
    flow_layout,
    participant_flow,
    source_from_trace,
)

BOX_W, BOX_H = 1.0, 0.56
FILL = {"exit": "#fbf1e4", POPULATION: "#d9ecf5", ANALYSIS: "#e6f3ea"}
LINE = {"exit": "#b36b00", POPULATION: "#1f5f8b", ANALYSIS: "#2e7d4f"}
DEFAULT_FILL, DEFAULT_LINE = "#f2f5f8", "#5b7083"


def current_flow(state, analyses: Sequence[AnalysisCount] = ()) -> ParticipantFlow | None:
    """The flow of the session's dataset: its trace, the analysis population and the given analyses."""
    population = state.population
    if population is None or not population.ok:
        return None
    try:
        trace = state.transformation_manager.get_trace() or {}
    except Exception:
        trace = {}
    source, n_loaded, steps = source_from_trace(trace)
    ref = state.dataset
    return participant_flow(
        population,
        dataset=ref.label if ref else "dataset",
        source=source,
        n_loaded=n_loaded,
        steps=steps,
        analyses=analyses,
    )


def _text(label: str, n: int) -> str:
    return "<br>".join(wrap(label, 26)) + f"<br><b>n = {n:,}</b>"


def flow_figure(flow: ParticipantFlow) -> go.Figure:
    pos = flow_layout(flow)
    fig = go.Figure()
    items = [*flow.main, *flow.exits, *flow.analyses]
    for item in items:
        x, y = pos[item.id]
        kind = "exit" if isinstance(item, FlowExit) else item.kind
        width = BOX_W * (1.1 if isinstance(item, FlowExit) else 1.0)
        fig.add_shape(
            type="rect",
            x0=x - width / 2,
            x1=x + width / 2,
            y0=y - BOX_H / 2,
            y1=y + BOX_H / 2,
            line={"color": LINE.get(kind, DEFAULT_LINE), "width": 1.5},
            fillcolor=FILL.get(kind, DEFAULT_FILL),
            layer="below",
        )
        fig.add_annotation(
            x=x,
            y=y,
            text=_text(item.label, item.n),
            showarrow=False,
            font={"size": 11, "color": "#1b2733"},
        )

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        fig.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=1.2,
            arrowcolor="#5b7083",
        )

    for before, after in zip(flow.main, flow.main[1:], strict=False):
        (x0, y0), (x1, y1) = pos[before.id], pos[after.id]
        arrow(x0, y0 - BOX_H / 2, x1, y1 + BOX_H / 2)
    for exit_ in flow.exits:
        x, y = pos[exit_.id]
        arrow(0.0, y, x - BOX_W * 0.55, y)
    population = next((b for b in flow.main if b.kind == POPULATION), flow.main[-1])
    px, py = pos[population.id]
    for box in flow.analyses:
        x, y = pos[box.id]
        arrow(px, py - BOX_H / 2, x, y + BOX_H / 2)

    ids = [item.id for item in items]
    fig.add_trace(
        go.Scatter(
            x=[pos[i][0] for i in ids],
            y=[pos[i][1] for i in ids],
            mode="markers",
            marker={"size": 46, "opacity": 0, "symbol": "square"},
            customdata=ids,
            hovertemplate="%{text}<extra></extra>",
            text=[f"{item.label}: {item.n:,}" for item in items],
            showlegend=False,
        )
    )
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    fig.update_xaxes(visible=False, range=[min(xs) - 0.8, max(xs) + 0.8])
    fig.update_yaxes(
        visible=False, range=[min(ys) - 0.5, max(ys) + 0.5], scaleanchor="x", scaleratio=1
    )
    fig.update_layout(
        height=max(320, int(150 * (max(ys) - min(ys) + 1))),
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        dragmode=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _select(key: str, ids: list[str]) -> None:
    event = st.session_state.get(key)
    points = (
        (getattr(event, "selection", None) or {}).get("points", []) if event is not None else []
    )
    if points:
        index = points[0].get("point_index")
        if index is not None and 0 <= int(index) < len(ids):
            st.session_state[f"{key}::item"] = ids[int(index)]


def render_flow(flow: ParticipantFlow | None, *, key: str) -> None:
    """The diagram, then the details of the box the user clicked (or chose in the menu)."""
    if flow is None:
        st.info("The participant flow needs an analysis population that can be evaluated.")
        return
    items = [*flow.main, *flow.exits, *flow.analyses]
    ids = [item.id for item in items]
    st.plotly_chart(
        flow_figure(flow),
        key=key,
        on_select=lambda: _select(key, ids),
        selection_mode="points",
        config={"displayModeBar": False},
        width="stretch",
    )
    for note in flow.notes:
        st.caption(note)
    for problem in flow.inconsistencies():
        st.warning(f"The flow does not add up: {problem}")
    labels = {item.id: f"{item.label} (n = {item.n:,})" for item in items}
    if st.session_state.get(f"{key}::item") not in labels:  # the flow changed since the last click
        st.session_state.pop(f"{key}::item", None)
    chosen = st.selectbox(
        "Details of a box (or click it in the diagram)",
        ids,
        format_func=labels.__getitem__,
        key=f"{key}::item",
        persist_state="page",  # kept while the tab that shows the flow is closed (M5b)
    )
    item = flow.item(chosen)
    for line in item.details:
        st.caption(line)
    if isinstance(item, FlowExit) and item.reasons:
        st.dataframe(
            pd.DataFrame(item.reasons, columns=["Criterion", "Rows it excluded first"]),
            hide_index=True,
            alt="Criteria behind this exclusion",
        )
    st.caption(
        "Participant flow (STROBE item 13): each exclusion counts the rows that criterion excluded first, in the "
        "order of the study configuration, so the counts add up to the next box."
    )


__all__ = ["current_flow", "flow_figure", "render_flow"]
