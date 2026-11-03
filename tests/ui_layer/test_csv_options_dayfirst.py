"""
A delimited file read outside Data > Import (an enrichment file, a configuration's data, Compare) reads its dates in
the user's order: day first until the user chooses, as an import does; the step that reads it records the order.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _app() -> None:
    import streamlit as st

    from ui.components.csv_options import csv_options

    options = csv_options(b"visit;value\n01/02/2024;1\n01/03/2024;2\n", "visits.csv", key="t")
    st.session_state["read_dayfirst"] = None if options is None else options.dayfirst


def test_a_file_read_outside_import_reads_its_dates_in_the_users_order() -> None:
    at = AppTest.from_function(_app, default_timeout=60)
    at.run()
    assert not at.exception
    # day first by default: 01/02/2024 is 1 February
    assert at.session_state["read_dayfirst"] is True
    at.session_state["import_default_dayfirst"] = False  # the user chose month first in Settings
    at.run()
    assert at.session_state["read_dayfirst"] is False
