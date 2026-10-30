"""
Date checks asks what the data do not decide (M6): the order of day and month when no value shows it (pre-filled from
the import's day-first setting), a pivot for two-digit years, whether whole numbers are Excel serial dates; it ticks
the placeholder dates no person has, and converts a column that loses more than 5 % of its values only when told to.
Each answer is recorded with the step, so the replay reads the column the same way.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from ingest.dates import PLACEHOLDERS
from manage.reproduction_manager import apply_steps

T = pd.Timestamp


def _app(frame: dict, dayfirst: bool) -> None:
    import pandas as pd
    import streamlit as st

    from app_pages.validate_dates import render_date_checks
    from core.state import DatasetRef, ParseOptions
    from ui.state import get_state

    st.session_state.setdefault("username", "dates_user")
    st.session_state.setdefault("user_role", "researcher")
    state = get_state()
    if state.base is None:
        df = pd.DataFrame(frame)
        state.set_dataset(
            df,
            DatasetRef(
                name="dates", origin="upload", file_hash="d", parse=ParseOptions(dayfirst=dayfirst)
            ),
            source_bytes=df.to_csv(index=False).encode("utf-8"),
            source_name="dates.csv",
        )
    render_date_checks()


@pytest.fixture(autouse=True)
def _data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))


def _open(frame: dict, dayfirst: bool = True) -> AppTest:
    at = AppTest.from_function(
        _app, kwargs={"frame": frame, "dayfirst": dayfirst}, default_timeout=120
    )
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    return at


def _convert(at: AppTest) -> dict:
    before = at.session_state["data"].copy()
    at.button(key="convert_dates").click().run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    step = at.session_state["transformation_manager"].history[-1]
    replayed = apply_steps(before, [step])  # the replay reads each column as recorded
    assert replayed.error is None and replayed.divergence is None
    return step


def test_an_order_no_value_shows_is_asked_pre_filled_from_the_import() -> None:
    at = _open({"visit": ["05/03/2021", "04/02/2021", "01/12/2021"]}, dayfirst=False)
    assert at.selectbox(key="date_format::visit").value == "month_first"
    step = _convert(at)
    assert step["params"]["formats"]["visit"]["order_source"] == "import_setting"
    assert at.session_state["data"]["visit"].tolist() == [
        T("2021-05-03"),
        T("2021-04-02"),
        T("2021-01-12"),
    ]


def test_an_order_chosen_is_recorded_as_the_users() -> None:
    at = _open({"visit": ["05/03/2021", "04/02/2021"]}, dayfirst=False)
    at.selectbox(key="date_format::visit").set_value("day_first").run()
    step = _convert(at)
    recorded = step["params"]["formats"]["visit"]
    assert (recorded["format"], recorded["order_source"]) == ("day_first", "user")
    assert at.session_state["data"]["visit"].tolist() == [T("2021-03-05"), T("2021-02-04")]


def test_two_digit_years_are_read_with_the_pivot_shown() -> None:
    at = _open({"birth": ["13/03/68", "05/03/21"]})
    pivot = at.number_input(key="date_pivot::birth")
    assert pivot.value == pd.Timestamp.today().year % 100
    pivot.set_value(26).run()
    step = _convert(at)
    assert step["params"]["formats"]["birth"]["pivot"] == 26
    assert at.session_state["data"]["birth"].tolist() == [T("1968-03-13"), T("2021-03-05")]


def test_whole_numbers_are_read_as_excel_dates_only_when_accepted() -> None:
    at = _open({"admission_date": [44197, 44562, 45657]})
    assert at.selectbox(key="date_format::admission_date").value == "text"
    assert not [b for b in at.button if b.key == "convert_dates"]  # nothing chosen to convert
    at.selectbox(key="date_format::admission_date").set_value("excel_serial").run()
    step = _convert(at)
    assert step["params"]["formats"]["admission_date"]["order_source"] == "user"
    assert at.session_state["data"]["admission_date"].tolist() == [
        T("2021-01-01"),
        T("2022-01-01"),
        T("2024-12-31"),
    ]


PRESELECTED = [p.date for p in PLACEHOLDERS if p.preselected]


def test_the_placeholder_dates_no_one_has_are_ticked_to_go_missing() -> None:
    at = _open({"death_date": ["01/01/1900", "15/03/2021", "01/01/1970", "13/04/2022"]})
    picked = at.multiselect(key="date_placeholders::death_date")
    assert (
        picked.value == PRESELECTED and "1970-01-01" not in picked.value
    )  # 1970 may be a real date
    step = _convert(at)
    recorded = step["params"]["formats"]["death_date"]
    assert (recorded["placeholders_to_missing"], recorded["set_to_missing"]) == (PRESELECTED, 1)
    read = at.session_state["data"]["death_date"]
    assert pd.isna(read[0]) and read.tolist()[1:] == [
        T("2021-03-15"),
        T("1970-01-01"),
        T("2022-04-13"),
    ]


def test_a_column_losing_more_than_5_percent_converts_only_when_told() -> None:
    at = _open({"onset": ["15/03/2021", "13/04/2022", "20/05/2023", "pending"]})
    assert any("1 of the 4 recorded values" in w.value for w in at.warning)
    assert at.button(key="convert_dates").disabled
    at.checkbox(key="dates_convert_anyway::onset").check().run()
    assert not at.button(key="convert_dates").disabled
    step = _convert(at)
    recorded = step["params"]["formats"]["onset"]
    assert (recorded["values"], recorded["unreadable"]) == (4, 1)
    assert at.session_state["data"]["onset"].isna().tolist() == [False, False, False, True]


def test_the_screen_shows_what_the_record_never_holds() -> None:
    """Values that cannot be read, and the earliest and latest dates, are shown; the step records counts only."""
    at = _open(
        {"visit": ["17/08/2019", "13/03/2021", "15/04/2021", "unknown P-0042", "20/05/2021"]}
    )
    assert any("P-0042" in c.value for c in at.caption)  # an example, drawn as code
    (table,) = at.dataframe
    assert table.value["Earliest"].tolist() == ["17/08/2019"]
    at.checkbox(key="dates_convert_anyway::visit").check().run()
    recorded = json.dumps(_convert(at)["params"])
    for value in ("P-0042", "17/08/2019", "2019-08-17", "20/05/2021", "2021-05-20"):
        assert value not in recorded, value


def test_planned_dates_are_not_counted_after_the_extraction_date() -> None:
    at = _open({"next_visit": ["15/03/2031", "13/04/2032", "20/05/2021"]})
    (table,) = at.dataframe
    assert table.value["After extraction"].tolist() == [2]
    at.multiselect(key="date_planned").select("next_visit").run()
    recorded = _convert(at)["params"]["formats"]["next_visit"]
    assert recorded["planned"] is True and recorded["after_extraction"] is None


def test_the_record_holds_the_rule_not_the_dates_a_column_holds() -> None:
    """A list of the placeholder dates found told the trace that a participant was born on 1 January 1970."""
    plain = _convert(_open({"birth": ["15/03/1950", "13/04/1962", "20/05/1971"]}))["params"][
        "formats"
    ]["birth"]
    planted = _convert(_open({"birth": ["15/03/1950", "01/01/1970", "20/05/1971"]}))["params"][
        "formats"
    ]["birth"]
    assert planted["placeholders_to_missing"] == plain["placeholders_to_missing"] == PRESELECTED
    assert (plain["placeholders"], planted["placeholders"]) == (
        0,
        1,
    )  # a count changes, nothing names the date
    assert {k for k in plain if plain[k] != planted[k]} == {"placeholders"}


def test_each_column_has_its_own_placeholder_dates() -> None:
    at = _open({"birth": ["01/01/1970", "13/04/1962"], "visit": ["01/01/1970", "13/04/2022"]})
    at.multiselect(key="date_placeholders::visit").select(
        "1970-01-01"
    ).run()  # no visit in 1970: a placeholder
    formats = _convert(at)["params"]["formats"]
    assert "1970-01-01" in formats["visit"]["placeholders_to_missing"]
    assert (
        "1970-01-01" not in formats["birth"]["placeholders_to_missing"]
    )  # a real birth date stays
    data = at.session_state["data"]
    assert pd.isna(data["visit"][0]) and data["birth"][0] == T("1970-01-01")


def test_convert_anyway_is_asked_per_column() -> None:
    """One "Convert anyway" for every column: an identifier column losing values blocked a real date column."""
    at = _open(
        {
            "onset": ["15/03/2021", "13/04/2022", "20/05/2023", "pending"],
            "visit": ["17/08/2019", "13/03/2021", "15/04/2021", "20/05/2021"],
        }
    )
    assert not at.button(
        key="convert_dates"
    ).disabled  # the visit converts; the onset waits for its own answer
    step = _convert(at)
    assert step["params"]["columns"] == ["visit"]


def test_a_column_the_data_decided_can_be_left_as_text() -> None:
    at = _open(
        {
            "visit": ["17/08/2019", "13/03/2021", "15/04/2021"],
            "onset": ["15/03/2021", "13/04/2022", "20/05/2023"],
        }
    )
    at.multiselect(key="date_leave_text").select("onset").run()
    assert _convert(at)["params"]["columns"] == ["visit"]


def test_whole_numbers_written_as_eight_digits_are_proposed_left_as_text() -> None:
    """20210305 read as a number never reached the tab; proposed now, as text until chosen (numbers are ambiguous)."""
    at = _open({"visit_date": [20210305, 20220414, 20230101]})
    assert at.selectbox(key="date_format::visit_date").value == "text"
    at.selectbox(key="date_format::visit_date").set_value("ymd_compact").run()
    _convert(at)
    assert at.session_state["data"]["visit_date"].tolist() == [
        T("2021-03-05"),
        T("2022-04-14"),
        T("2023-01-01"),
    ]


def test_dates_on_the_first_say_what_each_order_gives() -> None:
    """01/MM/YYYY reads both ways: month first, every date fell in January, silently (367 of 400 read wrong)."""
    at = _open({"visit": [f"01/{m:02d}/2021" for m in range(1, 13)] * 3}, dayfirst=False)
    said = " ".join(m.value for m in at.markdown)
    assert "Read day first, the dates over 12 months; month first, every date in one month." in said


def test_columns_already_dates_are_checked_and_their_placeholders_set_to_missing() -> None:
    """Date cells read on load (an Excel file's) had no check: a 1900-01-01, or a date after the extraction, stayed
    uncounted, and no placeholder could be set to missing."""
    visits = [T("1900-01-01"), T("2021-03-15 10:30"), T("2099-12-31"), T("2022-04-13")]
    at = _open({"visit": visits, "site": ["a", "b", "c", "d"]})
    assert "No text column looks like dates." in [c.value for c in at.caption]
    assert [s.value for s in at.subheader] == ["1 column holds dates"]
    row = at.dataframe[0].value.set_index("Column").loc["visit"]
    assert (row["Values"], row["With a time"], row["Placeholders"], row["After extraction"]) == (
        4,
        1,
        2,
        1,
    )
    assert at.multiselect(key="date_placeholders::visit").value == PRESELECTED
    button = at.button(key="dates_set_placeholders")
    assert button.label == "Set 2 placeholder dates to missing" and not button.disabled
    before = at.session_state["data"].copy()
    button.click().run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    step = at.session_state["transformation_manager"].history[-1]
    recorded = step["params"]["formats"]["visit"]
    assert (step["params"]["columns"], recorded["format"]) == (["visit"], "dates")
    assert (recorded["placeholders_to_missing"], recorded["set_to_missing"]) == (PRESELECTED, 2)
    read = at.session_state["data"]["visit"]
    assert read.isna().tolist() == [True, False, True, False] and read[1] == T("2021-03-15 10:30")
    replayed = apply_steps(before, [step])  # the replay reads the step as recorded
    assert replayed.error is None and replayed.divergence is None
    assert at.button(key="dates_set_placeholders").disabled  # nothing left to set to missing


def test_one_extraction_date_serves_both_kinds_of_columns() -> None:
    at = _open(
        {"death_date": ["15/03/2021", "13/04/2022"], "visit": [T("2021-03-15"), T("2099-12-31")]}
    )
    assert [s.value for s in at.subheader] == [
        "1 text column looks like dates",
        "1 column holds dates",
    ]
    assert len(at.date_input) == 1
    assert at.multiselect(key="date_planned").options == ["death_date", "visit"]
    at.multiselect(key="date_planned").set_value(["visit"]).run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    row = at.dataframe[1].value.set_index("Column").loc["visit"]
    assert pd.isna(
        row["After extraction"]
    )  # a column of planned dates: its later dates are not counted
