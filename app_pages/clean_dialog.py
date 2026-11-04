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

from app_pages.clean_categories import category_editor, clean_candidates, take_done
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


def _use_in_list(list_key: str, name: str, copy: str) -> None:
    chosen = list(st.session_state.get(list_key) or [])
    st.session_state[list_key] = [
        copy if c == name else c for c in chosen if c != copy or name not in chosen
    ]


def _copy_of(state: Any, name: str) -> tuple[str, str, bool]:
    """The variable ``name`` stands for (itself, or the variable whose clean copy it is), that variable's copy ("" when
    none is a column) and whether the copy follows it as it is (Clean's list: one computation, shared)."""
    base = state.base
    if base is None:
        return name, "", True
    recoded = recoded_columns(state.transformation_manager.history)  # source -> its last copy
    source = next((s for s, copy in recoded.items() if copy == name), name)
    copy = recoded.get(source, "")
    if not copy or copy not in base.columns or source not in base.columns:
        return name, "", True
    listed = next((c for c in clean_candidates(state) if c.column == source), None)
    return source, copy, listed.follows if listed is not None else True


def group_warning(
    column: Any,
    values: Iterable[Any],
    *,
    group_key: str | None = None,
    groupable: Collection[str] | None = None,
    imputed: bool = False,
) -> None:
    """The warning under a list of groups (ui.components.levels.level_warning), which names the variable's clean copy
    once one is written, then what can be done here (:func:`clean_actions`)."""
    source, copy, follows = _copy_of(get_state(), str(column))
    if level_warning(column, values, source=source, copy=copy, follows=follows):
        clean_actions(column, group_key=group_key, groupable=groupable, imputed=imputed, told=True)


def clean_actions(
    column: Any,
    *,
    group_key: str | None = None,
    list_key: str | None = None,
    groupable: Collection[str] | None = None,
    imputed: bool = False,
    told: bool = False,
) -> None:
    """
    What can be done here about a variable whose values are written several ways: clean it in a dialog (Data > Clean's
    editor, the source opened when the analysis reads a copy, or when its copy misses a later change of it), or, once a
    copy follows it, group by the copy (``group_key``: the page's grouping widget, set to the copy; ``list_key``: a
    list of variables, such as a model's covariates, where the copy takes the variable's place). ``groupable``: the
    variables the page offers, when not every column; ``imputed``: they are the imputed datasets' (the pooled
    comparison), which a copy written after the imputation is not in; ``told``: the warning above already says whether
    a copy cleans the variable (the groups' warning), else a copy that misses a later change of it is said here.
    """
    state = get_state()
    base = state.base
    if base is None:
        return
    name = str(column)
    source, copy, follows = _copy_of(state, name)
    for done in take_done(source):  # what Apply did in the dialog, said once, under this variable
        st.success(done, icon=":material/check:")
    if copy and copy != name and follows:
        if groupable is not None and copy not in groupable:
            st.caption(
                f"{as_code(copy)} cleans {as_code(source)}, but was written after the imputation: run the imputation "
                f"again to {'compare the groups of the copy' if group_key else 'use the copy'}."
                if imputed
                else f"{as_code(copy)} cleans {as_code(source)}, but this analysis does not offer it"
                + (" as a grouping." if group_key else ".")
            )
        elif group_key:
            st.button(
                f"Group by {as_code(copy)}, its clean copy",
                key=f"clean_use::{group_key}",
                icon=":material/swap_horiz:",
                on_click=_use,
                args=(group_key, copy),
            )
        elif list_key:
            st.button(
                f"Use {as_code(copy)}, its clean copy, instead of {as_code(name)}",
                key=f"clean_use::{list_key}::{name}",
                icon=":material/swap_horiz:",
                on_click=_use_in_list,
                args=(list_key, name, copy),
            )
        return
    if copy and copy != name and not told:
        # behind its variable: the rows a later step changed are not in it, and grouping by it would drop them
        st.caption(
            f"{as_code(copy)} was written before a later step changed {as_code(source)}: clean {as_code(source)} "
            "again, then use the copy."
        )
    if source not in {c.column for c in clean_candidates(state)}:
        st.caption(
            f"{CLEAN_WHERE} lists text variables of 2 to {MAX_VALUES} values: {as_code(source)} is not one, so it "
            "cannot be cleaned there."
        )
        return
    if st.button(
        f"Clean {as_code(source)} {'again ' if copy else ''}here",
        key=f"clean_open::{group_key or list_key or 'here'}::{source}",
        icon=":material/cleaning_services:",
    ):
        _dialog(source)


__all__ = ["clean_actions", "group_warning"]
