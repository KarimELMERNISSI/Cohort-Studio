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


def test_the_placeholder_dates_no_one_has_are_ticked_to_go_missing() -> None:
    at = _open({"death_date": ["01/01/1900", "15/03/2021", "01/01/1970", "13/04/2022"]})
    assert at.multiselect(key="date_placeholders").value == [
        "1900-01-01"
    ]  # 1970 may be a real date
    step = _convert(at)
    assert step["params"]["formats"]["death_date"]["placeholders_to_missing"] == ["1900-01-01"]
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
    at.checkbox(key="dates_convert_anyway").check().run()
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
    at.checkbox(key="dates_convert_anyway").check().run()
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
