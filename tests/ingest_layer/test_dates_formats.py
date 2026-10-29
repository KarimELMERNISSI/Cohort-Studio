"""
Dates read with a format (M6): a catalogue of formats read whole, never value by value; the format detected from the
shape of the values, the order of day and month from the values that show it, else asked; two-digit years only with a
pivot; Excel serial numbers only when accepted; partial dates never completed; placeholder dates set to missing on
request. The reader before M6 (smart_parse_dates) is pinned by test_dates_golden.py and unchanged.
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

from core.state import ParseOptions
from ingest.dates import (
    DATE_FORMATS,
    PLACEHOLDERS,
    Detection,
    convert,
    detect,
    partial_dates,
    placeholders_found,
    read_dates,
)
from ingest.loaders import read_table
from ui.state import TUTORIAL_COHORT_PATH

S = pd.Series
T = pd.Timestamp


@pytest.mark.parametrize(
    ("format_id", "values", "read"),
    [
        (
            "iso",
            ["2024-12-31", "2024/01/05 10:30", "2024.02.30"],
            [T("2024-12-31"), T("2024-01-05 10:30"), None],
        ),
        (
            "day_first",
            ["31/12/2024", "05-03-2021 14:30:15", "31.02.2021"],
            [T("2024-12-31"), T("2021-03-05 14:30:15"), None],
        ),
        ("month_first", ["12/31/2024", "03/05/2021"], [T("2024-12-31"), T("2021-03-05")]),
        (
            "ymd_compact",
            ["20241231", "17991231"],
            [T("2024-12-31"), None],
        ),  # a year before 1800 is not one
        ("dmy_compact", ["31122024", "05032021"], [T("2024-12-31"), T("2021-03-05")]),
        ("excel_serial", [45657, "45657", 1.5], [T("2024-12-31"), T("2024-12-31"), None]),
    ],
)
def test_each_format_reads_its_shape_and_real_dates_only(format_id, values, read) -> None:
    got = read_dates(S(values, dtype=object), format_id)
    assert [None if pd.isna(v) else v for v in got] == read
    assert str(got.dtype) == "datetime64[us]"


def test_the_formats_are_a_catalogue_of_ids_never_renamed() -> None:
    assert DATE_FORMATS.ids() == (
        "iso",
        "day_first",
        "month_first",
        "ymd_compact",
        "dmy_compact",
        "excel_serial",
    )
    assert DATE_FORMATS.get("day_first").other_order == "month_first"


def test_years_from_1_to_9999_hold_and_a_blank_is_missing() -> None:
    got = read_dates(S(["1582-10-14", "9999-12-31", "  ", None]), "iso")
    assert got.tolist()[:2] == [T("1582-10-14"), T("9999-12-31")] and got[2:].isna().all()


def test_a_two_digit_year_reads_only_with_a_pivot() -> None:
    values = S(["05/03/21", "05/03/26", "05/03/27", "05/03/68"])
    assert read_dates(values, "day_first").isna().all()
    assert [d.year for d in read_dates(values, "day_first", pivot=26)] == [
        2021,
        2026,
        1927,
        1968,
    ]  # never %y's 2068


def _slash(day_only: int, month_only: int, ambiguous: int) -> S:
    return S(["13/01/2021"] * day_only + ["01/13/2021"] * month_only + ["05/03/2021"] * ambiguous)


@pytest.mark.parametrize(
    ("day_only", "month_only", "ambiguous", "decided", "found"),
    [
        (5, 0, 95, True, "day_first"),  # 5 of 100 = ceil(0.05 n): the data decide
        (4, 0, 96, False, "day_first"),  # one fewer: asked, pre-filled day first
        (0, 5, 95, True, "month_first"),
        (600, 1, 0, False, "day_first"),  # a conflict: asked, the minority unread
        (0, 0, 100, False, "day_first"),  # no evidence (every visit on a day <= 12): asked
    ],
)
def test_the_data_decide_the_order_only_when_they_show_it(
    day_only, month_only, ambiguous, decided, found
) -> None:
    detection = detect(_slash(day_only, month_only, ambiguous), dayfirst=True)
    assert (detection.format, detection.decided) == (found, decided)
    assert (detection.day_first_only, detection.month_first_only) == (day_only, month_only)
    assert detection.order_source == ("data" if decided else "import_setting")


def test_an_order_not_shown_is_pre_filled_from_the_import_setting() -> None:
    assert detect(_slash(0, 0, 10), dayfirst=False).format == "month_first"
    assert detect(_slash(0, 0, 10), dayfirst=True).format == "day_first"


@pytest.mark.parametrize(("shaped", "decided"), [(950, True), (949, False)])
def test_a_shape_at_95_percent_is_chosen_and_below_it_proposed(shaped: int, decided: bool) -> None:
    values = S(["2021-03-05"] * shaped + ["unknown"] * (1000 - shaped))
    detection = detect(values, dayfirst=True)
    assert (detection.format, detection.decided, detection.shape) == ("iso", decided, shaped)


def test_two_digit_years_always_ask_for_a_pivot() -> None:
    detection = detect(S(["13/03/21", "05/03/21"]), dayfirst=True)
    assert (detection.format, detection.decided, detection.two_digit_years) == (
        "day_first",
        False,
        2,
    )


def test_partial_dates_are_counted_and_never_completed() -> None:
    values = S(["2021", "03/2021", "2021-03", "05/03/2021"])
    assert partial_dates(values) == 3
    assert read_dates(values, "day_first").notna().sum() == 1
    assert detect(S(["2021", "03/2021"]), dayfirst=True).format is None


def test_excel_serial_numbers_are_proposed_for_a_date_column_and_never_chosen() -> None:
    serials = S([44197, 44562, 45657])
    proposed = detect(serials, dayfirst=True, name="admission_date")
    assert (proposed.format, proposed.decided) == ("excel_serial", False)
    assert detect(serials, dayfirst=True, name="fu_days").format is None  # a duration is not a date
    assert detect(S([0, 1, 45657]), dayfirst=True, name="admission_date").format is None


def test_placeholder_dates_are_found_and_set_to_missing_when_chosen() -> None:
    assert {p.date for p in PLACEHOLDERS if not p.preselected} == {
        "1904-01-01",
        "1960-01-01",
        "1970-01-01",
    }
    dates = read_dates(S(["01/01/1900", "15/03/2021", "31/12/9999", "01/01/1900"]), "day_first")
    assert placeholders_found(dates) == {"1900-01-01": 2, "9999-12-31": 1}
    table = pd.DataFrame({"d": ["01/01/1900", "15/03/2021", "31/12/9999"]})
    read = convert(table, {"d": {"format": "day_first", "placeholders_to_missing": ["1900-01-01"]}})
    assert read["d"].tolist()[1:] == [T("2021-03-15"), T("9999-12-31")] and pd.isna(read["d"][0])
    assert table["d"][0] == "01/01/1900"  # the table given is left as it was


def test_a_column_of_dates_has_nothing_to_detect() -> None:
    assert detect(S(pd.to_datetime(["2021-03-05"])), dayfirst=True) == Detection(None, False, 1, 0)


def test_the_tutorials_dates_are_day_first_by_their_values() -> None:
    raw = TUTORIAL_COHORT_PATH.read_bytes()
    df = read_table(raw, TUTORIAL_COHORT_PATH.name, ParseOptions(dayfirst=True)).df
    for column, day_only in (("enrol_date", 740), ("last_contact_date", 1041)):
        detection = detect(
            df[column], dayfirst=False, name=column
        )  # the data decide, whatever the setting
        assert (detection.format, detection.decided, detection.order_source) == (
            "day_first",
            True,
            "data",
        )
        assert (detection.day_first_only, detection.month_first_only) == (day_only, 0)
        assert read_dates(df[column], "day_first").notna().sum() == df[column].notna().sum() == 1206


def test_a_large_column_reads_without_parsing_value_by_value() -> None:
    values = S(["05/03/2021", "13/03/2021", "2021-03-05"] * 10_000)
    start = time.perf_counter()
    detect(values, dayfirst=True)
    read_dates(values, "day_first")
    assert time.perf_counter() - start < 5  # 0.3 s measured; dateutil took seconds per column
