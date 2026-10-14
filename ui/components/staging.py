"""
The Enrich draft banner: what is waiting to be applied, and whether it can still be applied.

Changes made on the Enrich page (merges, derived variables, transformations, imputations) go into a draft
(``AppState.stage``). They reach the dataset, and the analyses, only when the draft is applied, and the
draft can only be applied over the data it was built on: when the dataset or the analysis population changed
underneath, applying would overwrite those changes, so the banner offers to discard it instead.
"""

from __future__ import annotations

import streamlit as st

from core.session import EMPTY, OK, POPULATION_CHANGED

_NOTICE_KEY = "_draft_notice"


def _apply() -> None:
    from ui.state import get_state

    ok, message = get_state().commit_staging()
    st.session_state[_NOTICE_KEY] = ("success" if ok else "warning", message)


def _discard() -> None:
    from ui.state import get_state

    get_state().discard_staging()
    st.session_state[_NOTICE_KEY] = (
        "info",
        "Draft discarded: Enrich starts again from the current dataset.",
    )


def render_draft_banner() -> None:
    """Status of the Enrich draft with Apply / Discard (callback buttons, so they work in tests too)."""
    from ui.state import get_state

    notice = st.session_state.pop(_NOTICE_KEY, None)
    if notice:
        getattr(st, notice[0])(notice[1])
    state = get_state()
    if not state.has_data:
        return
    status = state.staging_status()
    if status.state == EMPTY:
        st.caption(
            ":material/edit_note: Changes made on this page go into a draft; they reach the analyses when "
            "you apply the draft."
        )
        return
    steps = state.pending_steps()
    listing = "\n".join(f"- {step.description}" for step in steps)
    n = status.n_steps
    with st.container(border=True, key="cmp-enrich-draft"):
        if status.state == OK:
            st.markdown(
                f"**:material/edit_note: Enrich draft: {n} change{'s' if n != 1 else ''} not applied yet**"
            )
            st.markdown(listing)
            with st.container(horizontal=True):
                st.button(
                    "Apply to the dataset",
                    key="draft_apply",
                    type="primary",
                    icon=":material/check:",
                    on_click=_apply,
                    help="The changes become one traced step each and the analyses use them.",
                )
                st.button(
                    "Discard the draft",
                    key="draft_discard",
                    icon=":material/delete:",
                    on_click=_discard,
                    help="Forget these changes; the dataset is not modified.",
                )
            return
        what = (
            "the analysis population changed after the imputation in it was fitted"
            if status.state == POPULATION_CHANGED
            else "the dataset changed after it was started (for example a filter or a rename)"
        )
        st.warning(
            f"**This draft cannot be applied:** {what}. Applying it would overwrite those changes. "
            "Discard it and redo the changes on the current data."
        )
        st.markdown(listing)
        st.button(
            "Discard the draft and restart from the current data",
            key="draft_discard_stale",
            type="primary",
            icon=":material/restart_alt:",
            on_click=_discard,
        )


__all__ = ["render_draft_banner"]
