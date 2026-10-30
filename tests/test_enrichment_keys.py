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
from enrich.external_data import KeyCheck, add_data, align_date_keys, key_check
from ingest.dates import convert

MAIN = ensure_row_ids(pd.DataFrame({"pid": [1.0, 2.0, 3.0, np.nan], "age": [50, 60, 70, 80]}))
# pid 2 on two rows, a missing pid on two rows (pandas matches missing keys; Enrich no longer), pid 5 for no one
LABS = pd.DataFrame(
    {"pid": [2.0, 2.0, 3.0, np.nan, np.nan, 5.0], "hb": [11.0, 12.0, 13.0, 9.0, 9.5, 14.0]}
)


def test_the_check_counts_the_rows_a_merge_would_repeat() -> None:
    assert key_check(MAIN, LABS, ["pid"], ["pid"]) == KeyCheck(
        file_keys_repeated=1, rows_repeated=1, rows_added=1, rows_without_key=1
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

    main = st.session_state.get("main", pd.DataFrame({"pid": [1, 2, 3]}))
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
        # the file repeats only an identifier no row of the dataset holds: "unique" was refused (pandas checks every
        # identifier of the file), the merge failed with no way round
        (pd.DataFrame({"pid": [2, 3, 9, 9], "hb": [11.0, 13.0, 8.0, 8.5]}), "left", "all"),
        # two rows without an identifier are no one's: one row per identifier, checked on the identified rows
        (pd.DataFrame({"pid": [2, 3, None, None], "hb": [11.0, 13.0, 8.0, 8.5]}), "left", "unique"),
    ],
)
def test_the_page_asks_nothing_when_nothing_repeats(file, how, chosen) -> None:
    at = AppTest.from_function(_choice_app, default_timeout=120)
    at.session_state["file"], at.session_state["how"] = file, how
    at.run()
    assert not at.exception and not at.warning and at.session_state["chosen"] == chosen


@pytest.mark.parametrize(("file_rows", "rows"), [("all", 5), ("first", 4), ("last", 4)])
def test_a_missing_identifier_matches_no_other(file_rows: str, rows: int) -> None:
    """The row without a pid took the file's rows without one (hb 9.0 and 9.5), with nothing asked."""
    merged = add_data(
        MAIN,
        LABS,
        ["pid"],
        ["pid"],
        ["hb"],
        strategy="left",
        file_rows=file_rows,
        missing_keys="never",
    )
    assert len(merged) == rows and merged.loc[merged["pid"].isna(), "hb"].isna().all()
    assert merged.attrs["join_counts"]["main_rows_without_key"] == 1
    before = add_data(MAIN, LABS, ["pid"], ["pid"], ["hb"], strategy="left", file_rows=file_rows)
    assert (
        before.loc[before["pid"].isna(), "hb"].notna().all()
    )  # a step recorded before replays as it did


def test_rows_without_an_identifier_in_the_file_are_no_ones() -> None:
    outer = add_data(MAIN, LABS, ["pid"], ["pid"], ["hb"], strategy="outer", missing_keys="never")
    assert sorted(outer.loc[outer["pid"].isna(), "hb"].fillna(0).tolist()) == [
        0.0,
        9.0,
        9.5,
    ]  # added, apart
    once = pd.DataFrame({"pid": [2.0, 3.0, np.nan, np.nan], "hb": [11.0, 13.0, 9.0, 9.5]})
    unique = add_data(
        MAIN, once, ["pid"], ["pid"], ["hb"], file_rows="unique", missing_keys="never"
    )
    assert unique["hb"].tolist() == pytest.approx([np.nan, 11.0, 13.0, np.nan], nan_ok=True)
    assert (
        add_data(MAIN, once, ["pid"], ["pid"], ["hb"], file_rows="unique") is MAIN
    )  # pandas refused it
    sites = ensure_row_ids(pd.DataFrame({"site": pd.Categorical(["a", "b", None]), "x": [1, 2, 3]}))
    file = pd.DataFrame({"site": ["a", None], "y": [10, 20]})
    merged = add_data(sites, file, ["site"], ["site"], ["y"], missing_keys="never")
    assert merged["y"].tolist() == pytest.approx([10, np.nan, np.nan], nan_ok=True)
    assert (
        isinstance(merged["site"].dtype, pd.CategoricalDtype) and merged["site"].isna().sum() == 1
    )
    with pytest.raises(ValueError, match="missing_keys is one of"):
        add_data(MAIN, LABS, ["pid"], ["pid"], ["hb"], missing_keys="sometimes")


def test_the_page_says_how_many_rows_have_no_identifier() -> None:
    at = AppTest.from_function(_choice_app, default_timeout=120)
    at.session_state["main"] = pd.DataFrame({"pid": [1, 2, None, None]})
    at.session_state["file"] = pd.DataFrame({"pid": [1, 2, None], "hb": [11.0, 12.0, 9.0]})
    at.run()
    assert not at.exception and at.session_state["chosen"] == "unique"
    (info,) = at.info
    assert info.value.startswith(
        "2 rows of the dataset have no identifier: they take nothing from the file"
    )


def test_a_file_repeating_only_identifiers_the_dataset_lacks_merges_again() -> None:
    main = ensure_row_ids(pd.DataFrame({"pid": [1, 2, 3], "age": [50, 60, 70]}))
    file = pd.DataFrame({"pid": [2, 3, 9, 9], "hb": [11.0, 13.0, 8.0, 8.5]})
    assert key_check(main, file, ["pid"], ["pid"]) == KeyCheck(
        1, 0, 0
    )  # repeated, but for no row of the dataset
    refused = add_data(main, file, ["pid"], ["pid"], ["hb"], strategy="left", file_rows="unique")
    assert refused is main  # what the page chose before
    merged = add_data(main, file, ["pid"], ["pid"], ["hb"], strategy="left", file_rows="all")
    assert merged["hb"].tolist() == pytest.approx([np.nan, 11.0, 13.0], nan_ok=True)
    assert list(merged.index) == list(main.index)  # each row of the dataset once


MONTH_FIRST = {"visit_date": {"format": "month_first"}}
VISITS = ["03/01/2024", "04/02/2024", "05/03/2024"]


def test_a_date_key_is_read_as_the_dataset_s_dates_were() -> None:
    """Date checks read visit_date month first, and the file writes it the same way. Read day first by position, as
    before, 03/01/2024 was 3 January: no row matched, while the page said "3 rows before, 3 after"."""
    main = ensure_row_ids(
        convert(pd.DataFrame({"pid": [1, 2, 3], "visit_date": VISITS}), MONTH_FIRST)
    )
    file = pd.DataFrame({"pid": [1, 2, 3], "visit_date": VISITS, "hb": [12.1, 13.4, 11.8]})
    keys = ["pid", "visit_date"]
    aligned = align_date_keys(main, file, MONTH_FIRST)
    assert aligned.attrs["date_formats"] == MONTH_FIRST
    merged = add_data(main, aligned, keys, keys, ["hb"], missing_keys="never")
    assert merged["hb"].tolist() == [12.1, 13.4, 11.8]
    assert merged.attrs["join_counts"]["main_rows_unmatched"] == 0
    before = add_data(main, align_date_keys(main, file), keys, keys, ["hb"], missing_keys="never")
    assert before["hb"].isna().all() and before.attrs["join_counts"]["main_rows_unmatched"] == 3


def test_date_cells_of_the_file_are_kept() -> None:
    main = convert(pd.DataFrame({"visit_date": VISITS}), MONTH_FIRST)
    file = pd.DataFrame(
        {"visit_date": pd.to_datetime(["2024-03-01", "2024-04-02"]), "hb": [12.1, 13.4]}
    )
    aligned = align_date_keys(main, file, MONTH_FIRST)
    pd.testing.assert_series_equal(aligned["visit_date"], file["visit_date"])
    assert aligned.attrs["date_formats"] == MONTH_FIRST  # recorded: the replay keeps them too


def test_rows_without_a_match_are_counted_apart_from_rows_without_an_identifier() -> None:
    merged = add_data(MAIN, LABS, ["pid"], ["pid"], ["hb"], file_rows="first", missing_keys="never")
    counts = merged.attrs["join_counts"]
    assert (counts["main_rows_without_key"], counts["main_rows_unmatched"]) == (
        1,
        1,
    )  # pid nan; pid 1
    inner = add_data(MAIN, LABS, ["pid"], ["pid"], ["hb"], strategy="inner", file_rows="first")
    assert inner.attrs["join_counts"]["main_rows_unmatched"] == 0  # left out, counted as dropped


def _merged_app() -> None:
    import streamlit as st

    from app_pages.data_enrichment import _merged

    _merged(st.session_state["merged"], st.session_state["main"])


def test_the_page_warns_when_no_row_of_the_dataset_matched() -> None:
    main = ensure_row_ids(
        convert(pd.DataFrame({"pid": [1, 2, 3], "visit_date": VISITS}), MONTH_FIRST)
    )
    file = pd.DataFrame({"pid": [1, 2, 3], "visit_date": VISITS, "hb": [12.1, 13.4, 11.8]})
    keys = ["pid", "visit_date"]
    at = AppTest.from_function(_merged_app, default_timeout=120)
    at.session_state["main"] = main
    at.session_state["merged"] = add_data(
        main, align_date_keys(main, file), keys, keys, ["hb"], missing_keys="never"
    )
    at.run()
    assert not at.exception and not at.success
    (warning,) = at.warning
    assert warning.value.startswith("No row of the dataset matched a row of the file")
    assert "3 rows of the dataset matched no row of the file" in warning.value
    at.session_state["merged"] = add_data(
        main, align_date_keys(main, file, MONTH_FIRST), keys, keys, ["hb"]
    )
    at.run()
    assert not at.warning and at.success[0].value == "Enrichment completed: 3 rows before, 3 after."
