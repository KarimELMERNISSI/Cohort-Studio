"""
Dates read with a format (M6): a catalogue of formats read whole, never value by value; the format detected from the
shape of the values, the order of day and month from the values that show it, else asked; two-digit years only with a
pivot; Excel serial numbers only when accepted; partial dates never completed; placeholder dates set to missing on
request. The reader before M6 (smart_parse_dates) is pinned by test_dates_golden.py and unchanged.
"""

from __future__ import annotations

import time
from datetime import date

import numpy as np
import pandas as pd
import pytest

from core.state import ParseOptions
from ingest.dates import (
    DATE_FORMATS,
    PLACEHOLDERS,
    Detection,
    convert,
    date_report,
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
    one_way = bool(day_only) != bool(
        month_only
    )  # under 5 %, one way only: pre-filled from those values
    assert detection.order_source == (
        "data" if decided else "few_values" if one_way else "import_setting"
    )


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


def test_the_report_counts_what_reading_found() -> None:
    values = S(
        [
            "05/03/2021",
            "04/04/2021",
            "13/03/21",
            "15/03/2030 10:00",
            "01/01/1900",
            "1899",
            "01/01/1883",
            None,
        ]
    )
    report = date_report(values, "day_first", pivot=26, extraction=date(2026, 10, 29))
    assert report.to_record() == {
        "values": 7,
        "unreadable": 1,  # 1899, a year alone
        "partial": 1,
        "ambiguous": 1,  # 05/03; 04/04 and 01/01 read the same both ways
        "two_digit_years": 1,
        "with_time": 1,
        "placeholders": 1,  # 1900-01-01
        "before_1900": 1,  # 1883
        "after_extraction": 1,  # 2030
        "january_first": False,
        "month_first_day": False,
    }
    assert (
        date_report(values, "day_first", pivot=26).after_extraction is None
    )  # no extraction date given


@pytest.mark.parametrize(
    ("values", "january", "month"),
    [
        (
            ["01/01/2020"] * 5 + ["15/06/2020"] * 10,
            True,
            False,
        ),  # 1 January no longer among the 1sts
        (["01/01/2020"] * 4 + ["15/06/2020"], False, False),  # fewer than 5 on the day
        (["01/03/2020"] * 5 + ["15/06/2020"] * 20, False, True),
        (
            ["01/03/2020"] * 5 + ["15/06/2020"] * 100,
            False,
            False,
        ),  # 4.8 %: what the 1st of a month holds anyway
    ],
)
def test_dates_heaped_on_a_first_day_are_told(
    values: list[str], january: bool, month: bool
) -> None:
    report = date_report(S(values), "day_first")
    assert (report.january_first, report.month_first_day) == (january, month)


@pytest.mark.parametrize(
    ("values", "name", "found"),
    [
        (
            [f"2023{i:04d}" for i in range(1, 400)] + ["pending"],
            "patient_code",
            None,
        ),  # identifiers, not dates
        (
            [20210305, 20220414, 20230101] * 10,
            "visit_date",
            "ymd_compact",
        ),  # whole numbers, eight digits
        ([5032021, 14042022, 1012023] * 10, "date_visite", "dmy_compact"),  # a leading zero lost
        ([3200, 2800, 4100] * 10, "birth_weight", None),  # a measure named like a date
        ([1950, 1962, 1971] * 10, "annee_naissance", None),  # years, below 1910 as serials
        ([120, 3000, 45] * 10, "days_to_death", None),  # a duration
        (
            [44197, 44562, 45657] * 10,
            "stage_date",
            "excel_serial",
        ),  # "stage" holds "age", but is no measure
        ([44197, 44562, 45657] * 10, "décès", "excel_serial"),
    ],
)
def test_numbers_and_digits_are_dates_only_when_they_read_as_dates(values, name, found) -> None:
    detection = detect(pd.Series(values), dayfirst=True, name=name)
    assert detection.format == found
    assert (
        not detection.decided or found is None or isinstance(values[0], str)
    )  # numbers are only proposed


def test_an_order_shown_one_way_by_a_few_values_is_pre_filled_from_them() -> None:
    """30 of 1,000 values read only day first, none only month first: the import's month first swapped 888 others."""
    gen = np.random.default_rng(1)
    ambiguous = [
        f"{d:02d}/{m:02d}/2021"
        for d, m in zip(gen.integers(1, 13, 970), gen.integers(1, 13, 970), strict=True)
    ]
    day_only = [
        f"{d}/{m:02d}/2021"
        for d, m in zip(gen.integers(13, 29, 30), gen.integers(1, 13, 30), strict=True)
    ]
    found = detect(pd.Series(ambiguous + day_only), dayfirst=False, name="visit")
    assert (found.format, found.order_source, found.decided) == (
        "day_first",
        "few_values",
        False,
    )  # asked


def test_a_heap_on_1_january_is_not_a_heap_on_the_first_of_the_month() -> None:
    """50 more dates on 1 January among 1,000 spread over the year set the 1st-of-the-month flag too (87 > 65.7)."""
    spread = pd.date_range("2020-01-01", periods=1000, freq="9h").strftime("%d/%m/%Y").tolist()
    report = date_report(S(spread + ["01/01/2021"] * 50), "day_first")
    assert (report.january_first, report.month_first_day) == (True, False)


def test_dates_spread_over_the_year_are_not_told_heaped() -> None:
    """Five times the expected share flagged even dates by chance one time in ten at n = 76."""
    even = pd.date_range("2020-01-01", periods=76, freq="4D19h").strftime("%d/%m/%Y").tolist()
    report = date_report(S(even), "day_first")
    assert (report.january_first, report.month_first_day) == (False, False)
