"""
Rendering of the guidance of Analyze > Hypothesis tests & epidemiology (rules in ``ui.guidance.analysis``).

Everything here is shown only while the page's Guidance switch is on, so experienced users can keep
the page compact.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pandas as pd
import streamlit as st

from ui.components import page_link, status_chip
from ui.guidance.analysis import HELP, TAB_GUIDE, TEST_NOTES, AnalysisPlan, AncovaCheck

GUIDANCE_KEY = "epi_guidance"
_ICON = {"ok": ":material/check_circle:", "info": ":material/info:", "caution": ":material/warning:",
         "violated": ":material/error:"}


def guidance_toggle() -> bool:
    return st.toggle("Guidance", value=True, persist_state="page", key=GUIDANCE_KEY, help=HELP["guidance"])


def guidance_on() -> bool:
    return bool(st.session_state.get(GUIDANCE_KEY, True))


def render_orientation(
    open_analysis: Callable[[str, dict[str, str]], None],
    selectors: Mapping[str, tuple[str, Mapping[str, str]]],
) -> None:
    """
    Which tab answers which question, as a map that opens what it names: each button opens its tab, and the analysis
    on it where the tab holds several (``selectors``: a tab's label, the key of its Question selector and the
    entries, id to label). ``open_analysis`` is the page's callback: a tab or a selector is set before it is drawn.
    """
    with st.expander("Which tab do I need?", icon=":material/explore:"):
        for guide in TAB_GUIDE:
            question, buttons = st.columns([3, 4], vertical_alignment="center")
            question.markdown(f"**{guide.question}**")
            question.caption(guide.examples)
            key, entries = selectors.get(guide.tab, ("", {}))
            with buttons.container(horizontal=True, gap="small"):
                for entry, label in entries.items() or {"": guide.tab}.items():
                    st.button(
                        label,
                        type="tertiary",
                        icon=":material/arrow_forward:",
                        key=f"epi_open::{guide.tab}::{entry}",
                        on_click=open_analysis,
                        args=(guide.tab, {key: entry} if entry else {}),
                        help=f"Opens the tab {guide.tab}" + (", on this analysis." if entry else "."),
                    )
        page_link("which_test", "Not sure which test fits? Answer four questions", icon=":material/help_center:")


def _advice(level: str, text: str) -> None:
    st.markdown(f"{_ICON.get(level, '')} {text}")


def render_plan(plan: AnalysisPlan, *, correction: str, correction_key: str) -> None:
    """What Run will do, checked against the data, before anything is computed."""
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown("**Before you run: the plan**")
            levels = {a.level for a in plan.advice}
            if "violated" in levels:
                status_chip("fix before running", "violated")
            elif "caution" in levels:
                status_chip("check the warnings", "caution")
            else:
                status_chip("ready", "ok")
        sizes = " · ".join(f"{g}: n = {n}" for g, n in plan.group_sizes.items())
        st.caption(f"Groups: {sizes}")
        rows = [
            {
                "Variable": p.variable,
                "Test that will run": p.test,
                "Why": p.reason,
                "n": p.n,
                "Missing": p.missing_share,
            }
            for p in plan.tests
        ]
        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                width="stretch",
                column_config={
                    "Why": st.column_config.TextColumn(width="large"),
                    "n": st.column_config.NumberColumn(format="%d", help="Non-missing values in the selected groups."),
                    "Missing": st.column_config.NumberColumn(format="percent", help="Share of missing values."),
                },
            )
        for note in plan.notes:
            st.caption(note)
        for item in plan.advice:
            _advice(item.level, item.text)
        rec = plan.correction
        if rec is not None:
            if rec.option == correction:
                _advice("ok", f"Multiple-testing correction: {correction}. {rec.reason}")
            else:
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.markdown(f":material/lightbulb: Suggested correction: **{rec.option}**. {rec.reason}")
                    st.button(
                        f"Use {rec.option.split(' (')[0]}",
                        key="epi_use_recommended_correction",
                        on_click=lambda: st.session_state.update({correction_key: rec.option}),
                        help="Sets the multiple-testing correction above to the suggested method.",
                    )


def render_ancova_check(check: AncovaCheck) -> None:
    with st.container(border=True):
        st.markdown("**Model check**")
        st.caption(
            f"{check.n_complete} complete observations · {check.parameters} model parameters "
            "(intercept, group contrasts, covariates)"
        )
        for item in check.advice:
            _advice(item.level, item.text)
        st.caption(
            "Choose covariates before looking at the results: measured before the group was assigned, related to the "
            "outcome, and not a consequence of the group. The adjusted means compare groups at the covariates' means."
        )


def render_test_note(test: str) -> None:
    note = TEST_NOTES.get(test)
    if note:
        st.caption(f":material/info: {note}")


__all__ = [
    "GUIDANCE_KEY",
    "guidance_on",
    "guidance_toggle",
    "render_ancova_check",
    "render_orientation",
    "render_plan",
    "render_test_note",
]
