"""
A page's tabs as a registry: each tab its label and what draws it, and only the open one runs (M5b's rule, written
once). A tab that must always run says so, and is told whether it is open: a file uploader cannot keep its file while
its tab is closed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ui.components import linked_tabs


@dataclass(frozen=True)
class PageTab:
    id: str  # the tab's link id (ui.nav.PageDef.tabs)
    label: str  # part of the tab's identity in Streamlit 1.66: fixed, without counts
    render: Callable[[bool], None]  # told whether the tab is open
    always_runs: bool = False


def lazy_tabs(page: str, tabs: Sequence[PageTab], *, key: str) -> None:
    """The row of ``tabs`` on ``page``, opened on the tab a link names (ui.components.linked_tabs); only the open tab
    runs, and every tab that always runs."""
    containers = linked_tabs(page, [tab.label for tab in tabs], key=key, on_change="rerun")
    for tab, container in zip(tabs, containers, strict=True):
        with container:
            if container.open or tab.always_runs:
                tab.render(bool(container.open))


__all__ = ["PageTab", "lazy_tabs"]
