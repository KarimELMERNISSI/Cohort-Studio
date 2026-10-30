"""
Data > Clean: how values are written, fixed. Categories recodes a variable's values into a new column, the original
kept; Dates reads the text columns that look like dates (Validate's Date checks tab until 2026-10; its link id
``date_checks`` kept). Each fix is one traced step, and no row is removed here: Validate checks the data and chooses
the rows, Clean fixes how values are written, Enrich adds information.
"""

from __future__ import annotations

from app_pages.clean_categories import render_categories
from app_pages.validate_dates import render_date_checks
from ui.components import no_dataset
from ui.components.tabs import PageTab, lazy_tabs
from ui.state import get_state

CLEAN_TABS = (
    PageTab("categories", "Categories", lambda opened: render_categories()),
    PageTab("date_checks", "Dates", lambda opened: render_date_checks()),
)
TABS = tuple(tab.label for tab in CLEAN_TABS)
TAB_KEY = "clean_tab"


def app() -> None:
    if get_state().base is None:
        no_dataset()
        return
    lazy_tabs("clean", CLEAN_TABS, key=TAB_KEY)


__all__ = ["CLEAN_TABS", "TABS", "TAB_KEY", "app"]
