"""Power analysis inputs prefilled from the results ("Load Parameters into Calculator") keep the prefilled value
without Streamlit's warning about a widget with both a default value and a session-state value (AppTest)."""

from __future__ import annotations

from unittest import mock

import pytest
from streamlit.elements.lib import policies
from streamlit.testing.v1 import AppTest


def _script(helper: bool) -> None:
    import streamlit as st

    from app_pages.epidemiology import _prefilled_number_input

    if "seeded" not in st.session_state:
        st.session_state["seeded"] = True
        # what the results above write before the calculator renders
        st.session_state["ratio_ttest"] = 1.5
    if helper:
        prefilled = _prefilled_number_input("Ratio", 1.0, step=0.1, key="ratio_ttest", help="n2/n1")
    else:  # the former call: a default value as well
        prefilled = st.number_input("Ratio", value=1.0, step=0.1, key="ratio_ttest", help="n2/n1")
    st.session_state["prefilled"] = prefilled
    st.session_state["default"] = _prefilled_number_input(
        "Group 1", 30, step=1, key="n1_ttest", help="n1"
    )


def _run(helper: bool, monkeypatch: pytest.MonkeyPatch) -> tuple[AppTest, mock.MagicMock]:
    # Streamlit shows this warning once per process: reset its flag so that every run can show it
    monkeypatch.setattr(policies, "_shown_default_value_warning", False)
    with mock.patch.object(policies, "_LOGGER") as logger:
        at = AppTest.from_function(_script, kwargs={"helper": helper}, default_timeout=60).run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    return at, logger


def test_a_prefilled_input_keeps_its_value_without_the_duplicate_value_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    at, logger = _run(True, monkeypatch)
    assert at.session_state["prefilled"] == 1.5  # the prefilled value, not the default 1.0
    assert at.session_state["default"] == 30  # an unset key takes the default
    assert not logger.warning.called, logger.warning.call_args


def test_the_former_call_shows_the_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """The check above can see the warning: a default value with a prefilled key triggers it."""
    at, logger = _run(False, monkeypatch)
    assert at.session_state["prefilled"] == 1.5
    assert logger.warning.called
