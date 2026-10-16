"""
Reproduce > Study dossier (follow-up programme M3): the participant flow of the analysis population with the
analyses added to the report, the list of those results (current, or computed on other data or another
population), and the Word study dossier that documents them.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from core.attrition import AnalysisCount
from ui.components import no_dataset, section, status_chip
from ui.components.flow import current_flow, render_flow
from ui.results import (
    CURRENT,
    STALE_DATA,
    ReportItem,
    remove_from_report,
    report_items,
    report_status,
)
from ui.state import get_state


def imputations_of(value: Any) -> int | None:
    """m when the result pools multiply imputed datasets (biostats.mi.analyze.MIResult), else None."""
    if hasattr(value, "m_total") and isinstance(getattr(value, "m", None), int):
        return int(value.m)
    if isinstance(value, dict):  # the pooled group comparison keeps one pooled result per variable
        for pooled in (value.get("results") or {}).values():
            m = imputations_of(getattr(pooled, "result", None))
            if m is not None:
                return m
    return None


def analysis_counts(items: list[ReportItem]) -> list[AnalysisCount]:
    """What each result of the report analysed, for the boxes under the population."""
    return [
        AnalysisCount(
            item.title,
            item.stamp.n_population,
            item.stamp.n_analysed,
            not_used=item.stamp.not_used,
            imputations=imputations_of(item.value),
            note=item.settings,
        )
        for item in items
    ]


def _stale_text(status: str) -> str:
    if status == STALE_DATA:
        return "computed on data that have changed since"
    return "computed on another analysis population"


def app() -> None:
    state = get_state()
    if not state.has_data:
        no_dataset()
        return
    items = report_items()
    current = [item for item in items if report_status(item) == CURRENT]

    section(
        "Participant flow",
        "From the source file to the analysis population and each analysis of the dossier (STROBE item 13).",
    )
    render_flow(current_flow(state, analysis_counts(current)), key="dossier_flow")

    section(
        "Results in the dossier",
        "Add a result with the button under it on the analysis pages; it keeps what it was computed on.",
    )
    if not items:
        st.info(
            "No result yet. Run an analysis (Analyze) and click **Add to the study dossier** under the result.",
            icon=":material/note_add:",
        )
    for item in items:
        status = report_status(item)
        with st.container(border=True):
            head, chip, remove = st.columns([6, 1.4, 1], vertical_alignment="center")
            head.markdown(f"**{item.title}**" + (f" · {item.settings}" if item.settings else ""))
            with chip:
                status_chip("current" if status == CURRENT else "stale", "ok" if status == CURRENT else "caution")
            remove.button(
                "Remove",
                key=f"dossier_remove::{item.id}",
                icon=":material/delete:",
                on_click=remove_from_report,
                args=(item.id,),
            )
            st.caption(f":material/fingerprint: {item.stamp.caption}")
            if status != CURRENT:
                st.caption(
                    f":material/warning: {_stale_text(status).capitalize()}: the dossier leaves it out unless you "
                    "include stale results, marked as such. Run the analysis again to update it."
                )


__all__ = ["analysis_counts", "app", "imputations_of"]
