"""
A merge on identifiers says how many rows of the dataset the file would repeat, and asks (M6): a file that holds an
identifier on several rows repeated the dataset's rows silently, and the analyses counted each repeat as a participant
(the merge counted them in the trace, which nothing showed; the participant flow called them added rows).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core.attrition import participant_flow
from core.fingerprint import ensure_row_ids
from core.population import evaluate_population
from enrich.external_data import KeyCheck, add_data, key_check

MAIN = ensure_row_ids(pd.DataFrame({"pid": [1.0, 2.0, 3.0, np.nan], "age": [50, 60, 70, 80]}))
# pid 2 on two rows, a missing pid on two rows (pandas matches missing keys), pid 5 for no one
LABS = pd.DataFrame(
    {"pid": [2.0, 2.0, 3.0, np.nan, np.nan, 5.0], "hb": [11.0, 12.0, 13.0, 9.0, 9.5, 14.0]}
)


def test_the_check_counts_the_rows_a_merge_would_repeat() -> None:
    assert key_check(MAIN, LABS, ["pid"], ["pid"]) == KeyCheck(
        file_keys_repeated=2, rows_repeated=2, rows_added=2
    )
    assert not key_check(MAIN, LABS.drop_duplicates("pid"), "pid", "pid").repeats  # type: ignore[union-attr]
    two_keys = key_check(
        pd.DataFrame({"site": ["a", "a", "b"], "pid": [1, 2, 1]}),
        pd.DataFrame({"centre": ["a", "a", "a", "b"], "id": [1, 1, 1, 1]}),
        ["site", "pid"],
        ["centre", "id"],
    )
    assert two_keys == KeyCheck(file_keys_repeated=1, rows_repeated=1, rows_added=2)


@pytest.mark.parametrize(
    ("left", "right", "file"),
    [
        (["pid"], ["pid"], LABS.assign(pid=LABS["pid"].astype(str))),  # keys of another type
        (["pid"], ["missing"], LABS),
        (["pid", "age"], ["pid"], LABS),
        ([], [], LABS),
    ],
)
def test_keys_that_cannot_be_compared_give_no_check(left, right, file) -> None:
    assert key_check(MAIN, file, left, right) is None


@pytest.mark.parametrize(
    ("file_rows", "hb", "left_out"),
    [("first", [np.nan, 11.0, 13.0, 9.0], 2), ("last", [np.nan, 12.0, 13.0, 9.5], 2)],
)
def test_one_row_of_the_file_per_row_of_the_dataset(file_rows, hb, left_out) -> None:
    merged = add_data(MAIN, LABS, ["pid"], ["pid"], ["hb"], strategy="left", file_rows=file_rows)
    assert merged["hb"].tolist() == pytest.approx(hb, nan_ok=True)
    assert list(merged.index) == list(MAIN.index)  # each row once, with its own id
    counts = merged.attrs["join_counts"]
    assert counts["main_rows_repeated"] == 0 and counts["rows_after"] == len(MAIN)
    assert (counts["file_rows"], counts["file_rows_left_out"]) == (file_rows, left_out)


def test_every_row_of_the_file_repeats_the_dataset_as_before() -> None:
    merged = add_data(MAIN, LABS, ["pid"], ["pid"], ["hb"], strategy="left")
    counts = merged.attrs["join_counts"]
    assert (counts["file_rows"], counts["main_rows_repeated"], len(merged)) == ("all", 2, 6)


def test_a_file_expected_unique_that_repeats_an_identifier_is_refused(
    caplog: pytest.LogCaptureFixture,
) -> None:
    unique = LABS.drop_duplicates("pid")
    merged = add_data(MAIN, unique, ["pid"], ["pid"], ["hb"], strategy="left", file_rows="unique")
    assert merged.attrs["join_counts"]["file_rows"] == "unique" and len(merged) == len(MAIN)
    planted = LABS.assign(pid=["P-0042", "P-0042", "x", "y", "z", "w"])
    main = ensure_row_ids(pd.DataFrame({"pid": ["P-0042", "x"], "age": [50, 60]}))
    with caplog.at_level(logging.ERROR):
        refused = add_data(
            main, planted, ["pid"], ["pid"], ["hb"], strategy="left", file_rows="unique"
        )
    assert refused is main  # the dataset unchanged
    assert "MergeError" in caplog.text and "P-0042" not in caplog.text  # the keys are patient data
    with pytest.raises(ValueError, match="file_rows is one of"):
        add_data(MAIN, LABS, ["pid"], ["pid"], ["hb"], file_rows="some")


def test_the_flow_tells_repeated_rows_from_added_rows() -> None:
    population = evaluate_population(pd.DataFrame({"x": range(7)}), ())
    merge = {
        "function": "enrichment",
        "description": "Enriched with external data",
        "params": {"join_counts": {"main_rows_repeated": 2, "rows_added": 1}},
        "data": {"rows_in": 4, "rows_out": 7, "rows_added": 3},
    }
    flow = participant_flow(population, steps=[merge], n_loaded=4)
    (box,) = [b for b in flow.main if b.id == "step0"]
    assert box.details[1:] == (
        "2 rows repeat a row of the dataset (its identifier on several rows of the file)",
        "1 row added",
    )
    assert flow.notes == ()  # the rows add up
    other = {**merge, "function": "variable_computation"}  # only a merge repeats rows
    (box,) = [
        b for b in participant_flow(population, steps=[other], n_loaded=4).main if b.id == "step0"
    ]
    assert box.details[1:] == ("3 rows added",)


def _choice_app() -> None:
    import pandas as pd
    import streamlit as st

    from app_pages.data_enrichment import file_rows_choice

    main = pd.DataFrame({"pid": [1, 2, 3]})
    file = st.session_state.get("file", pd.DataFrame({"pid": [2, 2, 3], "hb": [11.0, 12.0, 13.0]}))
    st.session_state["chosen"] = file_rows_choice(
        main, file, ["pid"], ["pid"], st.session_state.get("how", "left")
    )


def test_the_page_asks_before_a_merge_that_repeats_rows() -> None:
    at = AppTest.from_function(_choice_app, default_timeout=120).run()
    assert not at.exception
    (warning,) = at.warning
    assert warning.value.startswith("1 row of the dataset matches more than one row of the file")
    assert "add 1 row, which the analyses would count as participants" in warning.value
    assert at.session_state["chosen"] is None  # nothing chosen: the merge waits
    at.radio(key="enrich_file_rows").set_value("first").run()
    assert at.session_state["chosen"] == "first"


@pytest.mark.parametrize(
    ("file", "how", "chosen"),
    [
        (
            pd.DataFrame({"pid": [2, 3], "hb": [11.0, 13.0]}),
            "left",
            "unique",
        ),  # one row per identifier
        (
            pd.DataFrame({"pid": [2, 2], "hb": [11.0, 12.0]}),
            "rows",
            "all",
        ),  # rows appended, no keys
    ],
)
def test_the_page_asks_nothing_when_nothing_repeats(file, how, chosen) -> None:
    at = AppTest.from_function(_choice_app, default_timeout=120)
    at.session_state["file"], at.session_state["how"] = file, how
    at.run()
    assert not at.exception and not at.warning and at.session_state["chosen"] == chosen
