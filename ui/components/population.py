"""
The analysis population in the UI: the sidebar selector, the banner of the analysis pages and the panel of
the Validate page.

Inclusion and exclusion criteria are filters: ticking or unticking them changes which rows the analyses
use, never the dataset. The ticked criteria live in the session (``AppState``); the checkboxes mirror them,
because they also change from Validate (activate a whole family) and when a saved version is loaded.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from core.population import EXCLUSION, catalogue, family_words
from ui.components.text import plain

_FREEZE_CONFIRM = "pop_freeze_confirm"


def _widget_key(criterion_id: str) -> str:
    return "pop::" + hashlib.sha1(criterion_id.encode("utf-8")).hexdigest()[:12]


def _family_key(family: str) -> str:
    return "popfam::" + hashlib.sha1(family.encode("utf-8")).hexdigest()[:12]


def _on_toggle(criterion_id: str) -> None:
    from ui.state import get_state

    get_state().tick(criterion_id, bool(st.session_state.get(_widget_key(criterion_id))))


def _on_family(family: str) -> None:
    from ui.state import get_state

    get_state().activate_family(family, bool(st.session_state.get(_family_key(family))))


def activate_family(family: str, on: bool = True) -> None:
    """Callback for buttons outside the sidebar (Validate): tick or untick a whole family."""
    from ui.state import get_state

    get_state().activate_family(family, on)


def family_title(family: str, kind: str) -> str:
    verb = "Exclude" if kind == EXCLUSION else "Require"
    return f"{verb}: {family_words(family)}"


def render_population_selector(state) -> None:
    """Sidebar: one checkbox per criterion, grouped by family, with what each excludes."""
    criteria = catalogue(state.config)
    population = state.population
    if population is None:
        return
    if not criteria:
        st.caption(
            ":material/filter_alt: Analysis population: every row. Define inclusion or exclusion criteria "
            "on Data > Validate to filter it."
        )
        return
    ticked = set(state.ticked)
    counts = state.session.criterion_counts(state.config)
    icon = ":material/filter_alt:" if population.ok else ":material/error:"
    with st.expander(f"Population: {population.label}", icon=icon, expanded=not population.ok):
        if not population.ok:
            st.error(
                "A ticked criterion cannot be evaluated, so the analyses are stopped: "
                + "; ".join(plain(e, limit=300) for e in population.errors.values())
            )
        families: dict[str, list] = {}
        for criterion in criteria:
            families.setdefault(criterion.family, []).append(criterion)
        ss = st.session_state
        # the boxes mirror the session's ticked criteria, set on every run: never kept on their own (M5b)
        for family, members in families.items():
            family_box = _family_key(family)
            ss[family_box] = all(c.id in ticked for c in members)
            st.checkbox(
                f"**{family_title(family, members[0].kind)}**",
                key=family_box,
                on_change=_on_family,
                args=(family,),
                help="Tick or untick every criterion of this family.",
            )
            for criterion in members:
                criterion_box = _widget_key(criterion.id)
                ss[criterion_box] = criterion.id in ticked
                excludes = counts.get(criterion.id)
                suffix = (
                    " (cannot be evaluated)" if excludes is None else f" ({excludes:,} excluded)"
                )
                st.checkbox(
                    f"{plain(criterion.name, limit=200)}{suffix}",
                    key=criterion_box,
                    on_change=_on_toggle,
                    args=(criterion.id,),
                    help=plain(criterion.description, limit=1000) or None,
                )
        st.caption(
            "Criteria are filters: no row is deleted, and every analysis says which population it used."
        )


def population_banner(*, purpose: str = "analyses") -> None:
    """Top of an analysis page: which rows are analysed; stops the page when the population is broken."""
    from ui.state import get_state

    state = get_state()
    if not state.has_data:
        return
    population = state.population
    if population is None:
        return
    if not population.ok:
        st.error(
            "The analysis population cannot be evaluated: "
            + "; ".join(plain(e, limit=300) for e in population.errors.values())
            + ". Untick the criterion in the sidebar or correct it on Data > Validate.",
            icon=":material/error:",
        )
        st.stop()
    if population.criteria:
        st.info(
            f"Analysis population: **{population.label}**. The {purpose} use these rows; "
            "change the ticked criteria in the sidebar.",
            icon=":material/filter_alt:",
        )
    mismatched = state.imputations_on_other_rows()
    if mismatched:
        st.warning(
            "The data contain imputed values computed on other rows than the current population: "
            + "; ".join(mismatched)
            + ". A statistical imputation, or a formula of other rows such as a mean, depends on the rows it was "
            "computed on, and the rows the population gained since were not imputed. Tick the same criteria again, "
            "or go back to the data before the imputation (Reproduce > Versions & traces, Reproduce to Step) and "
            "impute on the current population.",
            icon=":material/warning:",
        )


def _freeze(save_as: str) -> None:
    from ui.state import get_state

    state = get_state()
    if not st.session_state.get(_FREEZE_CONFIRM):
        return
    state.freeze_population()
    st.session_state[_FREEZE_CONFIRM] = False
    if save_as.strip():
        state.save_version(save_as.strip())


def render_population_panel(state) -> None:
    """Validate > Analysis population: the participant flow of the ticked criteria and the freeze action."""
    population = state.population
    if population is None:
        return
    criteria = catalogue(state.config)
    if not criteria:
        st.info(
            "No inclusion or exclusion criterion is defined yet: add them in the implausible values and inclusion tabs."
        )
        return
    if not population.criteria:
        st.info(
            "No criterion is ticked: the analyses use every row. Tick criteria in the sidebar, or use the "
            "buttons of the implausible values and inclusion tabs."
        )
        return
    if not population.ok:
        st.error("; ".join(plain(e, limit=300) for e in population.errors.values()))
        return
    st.markdown(f"**{population.label}**")
    from ui.components.flow import current_flow, render_flow

    render_flow(current_flow(state), key="validate_flow")
    stages = pd.DataFrame(
        [
            {
                "Step": family_title(s.family, s.kind),
                "Rows in": s.n_in,
                "Excluded": s.n_excluded,
                "Rows out": s.n_out,
            }
            for s in population.stages
        ]
    )
    st.dataframe(stages, hide_index=True, width="stretch")
    counts = pd.DataFrame(
        [
            {
                "Criterion": c.criterion.name,
                "Family": c.criterion.family,
                "Excludes (whole table)": c.marginal_base,
                "Excludes at its step": c.marginal_in_stage,
                "First reason for": c.sequential,
            }
            for c in population.counts
        ]
    )
    st.dataframe(counts, hide_index=True, width="stretch")
    st.caption(
        "Excludes at its step: rows the criterion excludes among the rows entering its step. First reason "
        "for: rows it excludes that no earlier criterion of the step excluded (these sum to the step's "
        "exclusions, as in a STROBE participant-flow diagram)."
    )
    excluded = population_excluded_rows(state, limit=500)
    if not excluded.empty:
        with st.expander(
            f"Excluded rows and their reasons ({population.n_base - population.n_included:,})"
        ):
            st.dataframe(excluded, hide_index=True, width="stretch")
    with st.container(border=True):
        st.markdown("**Keep only the analysis population in the dataset**")
        st.caption(
            "Use this to export or hand over the selected rows. It is one step of the trace; the excluded rows "
            "are removed from the current data (saved versions are not changed), and the criteria are unticked."
        )
        save_as = st.text_input(
            "Save the result as a version (optional)",
            key="pop_freeze_name",
            persist_state="page",  # kept while the tab is closed (M5b)
            placeholder="e.g. analysis_set",
        )
        st.checkbox(  # a confirmation: asked again when the tab has been left, so it is not kept
            "I understand that the excluded rows are removed from the current data",
            key=_FREEZE_CONFIRM,
        )
        st.button(
            "Keep only these rows",
            key="pop_freeze",
            icon=":material/content_cut:",
            disabled=not st.session_state.get(_FREEZE_CONFIRM),
            on_click=_freeze,
            args=(save_as,),
        )


def population_excluded_rows(state, limit: int = 500) -> pd.DataFrame:
    population = state.population
    base = state.base
    if population is None or base is None:
        return pd.DataFrame()
    excluded = base.index.difference(population.included)[:limit]
    return pd.DataFrame(
        {"Row": excluded, "Excluded by": [", ".join(population.reasons(r)) for r in excluded]}
    )


__all__ = [
    "activate_family",
    "family_title",
    "population_banner",
    "population_excluded_rows",
    "render_population_panel",
    "render_population_selector",
]
