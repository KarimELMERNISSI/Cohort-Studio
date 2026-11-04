"""
A variable cleaned where it is analysed (Data > Clean, from the analysis warnings). Groups that differ only by spaces or
capitals are each analysed apart (ui.components.levels.level_warning); the warning offers to clean the variable in a
dialog that holds Data > Clean's own editor, since a page switch would drop the analysis page's choices (its widgets
keep their values while the user stays on the page), and, once a copy cleans it, to group by the copy.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import Any

import streamlit as st

from app_pages.clean_categories import DONE, category_editor, clean_candidates
from core.registry import as_code
from enrich.recode import MAX_VALUES, recoded_columns
from ui.components.levels import CLEAN_WHERE, level_warning
from ui.state import get_state


@st.dialog("Clean a variable's values", width="large")
def _dialog(column: str) -> None:
    """Data > Clean's editor of ``column``: Apply writes its copy and closes the dialog (the editor's own rerun)."""
    state = get_state()
    if state.base is None or column not in state.base.columns:
        st.caption(f"{as_code(column)} is no longer a column of the data.")
        return
    st.caption(
        f"As on {CLEAN_WHERE}: the clean values of {as_code(column)} go into a copy beside it, the original kept. "
        "Apply writes the copy and closes this window; the analysis can then group by the copy."
    )
    category_editor(state, state.base, column, host="dialog")


def _use(group_key: str, copy: str) -> None:
    st.session_state[group_key] = copy


def group_warning(
    column: Any,
    values: Iterable[Any],
    *,
    group_key: str | None = None,
    groupable: Collection[str] | None = None,
    imputed: bool = False,
) -> None:
    """The warning under a list of groups (ui.components.levels.level_warning), then what can be done here
    (:func:`clean_actions`)."""
    if level_warning(column, values):
        clean_actions(column, group_key=group_key, groupable=groupable, imputed=imputed)


def clean_actions(
    column: Any,
    *,
    group_key: str | None = None,
    groupable: Collection[str] | None = None,
    imputed: bool = False,
) -> None:
    """
    What can be done here about a variable whose values are written several ways: clean it in a dialog (Data > Clean's
    editor, the source opened when the analysis reads a copy), or, once a copy cleans it, group by the copy
    (``group_key``: the page's grouping widget, set to the copy). ``groupable``: the variables the page offers to group
    by, when not every column; ``imputed``: they are the imputed datasets' (the pooled comparison), which a copy
    written after the imputation is not in.
    """
    state = get_state()
    base = state.base
    if base is None:
        return
    name = str(column)
    recoded = recoded_columns(state.transformation_manager.history)  # source -> its copy
    source = next(
        (s for s, copy in recoded.items() if copy == name), name
    )  # grouped by a copy: its source
    copy = recoded.get(source, "")
    if done := st.session_state.pop(DONE, None):  # what Apply did in the dialog, said once
        st.success(done, icon=":material/check:")
    if copy and copy != name and copy in base.columns:
        if groupable is not None and copy not in groupable:
            st.caption(
                f"{as_code(copy)} cleans {as_code(source)}, but was written after the imputation: run the imputation "
                "again to compare the groups of the copy."
                if imputed
                else f"{as_code(copy)} cleans {as_code(source)}, but this analysis does not offer it as a grouping."
            )
        elif group_key:
            st.button(
                f"Group by {as_code(copy)}, its clean copy",
                key=f"clean_use::{group_key}",
                icon=":material/swap_horiz:",
                on_click=_use,
                args=(group_key, copy),
            )
        return
    if source not in {c.column for c in clean_candidates(state)}:
        st.caption(
            f"{CLEAN_WHERE} lists text variables of 2 to {MAX_VALUES} values: {as_code(source)} is not one, so it "
            "cannot be cleaned there."
        )
        return
    if st.button(
        f"Clean {as_code(source)} here",
        key=f"clean_open::{group_key or source}",
        icon=":material/cleaning_services:",
    ):
        _dialog(source)


__all__ = ["clean_actions", "group_warning"]
