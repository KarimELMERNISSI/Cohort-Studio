"""
A page's tabs as a registry (ui.components.tabs.lazy_tabs): only the open tab runs, a tab that must always run does
and is told whether it is open, and a link opens its tab.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from ui import nav


def _script() -> None:
    import streamlit as st

    from ui.components.tabs import PageTab, lazy_tabs

    ran = st.session_state.setdefault("ran", [])
    lazy_tabs(
        "page_tabs_test",
        [
            PageTab("first", "First", lambda opened: ran.append(("first", opened))),
            PageTab("second", "Second", lambda opened: ran.append(("second", opened))),
            PageTab(
                "always", "Always", lambda opened: ran.append(("always", opened)), always_runs=True
            ),
        ],
        key="page_tabs_test_tab",
    )


def test_only_the_open_tab_runs_and_one_that_always_runs_is_told_it_is_closed() -> None:
    at = AppTest.from_function(_script)
    at.run()
    assert not at.exception
    assert at.session_state["ran"] == [("first", True), ("always", False)]
    at.session_state["ran"] = []
    at.session_state["page_tabs_test_tab"] = "Second"
    at.run()
    assert at.session_state["ran"] == [("second", True), ("always", False)]


def test_a_link_opens_its_tab(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        nav,
        "opened_tabs",
        lambda page, value: tuple(["Second"] if page == "page_tabs_test" else []),
    )
    at = AppTest.from_function(_script)
    at.run()
    assert not at.exception
    assert at.session_state["ran"] == [("second", True), ("always", False)]
