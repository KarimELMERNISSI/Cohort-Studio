"""
Today's date parsing, pinned before M6 moves it into ingest and reads new conversions another way.

smart_parse_dates reads some dates wrong without a word: two-digit years in one order or the other within a column,
8-digit dates and years alone as Excel serials, a month and year completed to the 1st, a time that loses its row,
small numbers as nanoseconds after 1970. It stays as it is: a trace that recorded it ("method": "smart_parse_dates")
replays with it, and must give the same table. These pins show that it did not change.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from core.fingerprint import frame_fingerprint
from core.state import ParseOptions
from ingest.loaders import read_path, read_table
from manage.reproduction_manager import apply_steps
from utils.data_analyzer import DataAnalyzer
from utils.date_parser import smart_parse_dates

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parents[2]

# (case, values, what today's parser reads, its notes): the wrong readings on purpose
CASES = [
    (
        "iso",
        ["2021-03-05", "2021-12-31", None],
        ["2021-03-05T00:00:00", "2021-12-31T00:00:00", None],
        ["Strict Structure: Y=P0, M=P1, D=P2"],
    ),
    (
        "day_first_by_a_day_over_12",
        ["13/02/2021", "05/03/2021"],
        ["2021-02-13T00:00:00", "2021-03-05T00:00:00"],
        ["Strict Structure: Y=P2, M=P1, D=P0"],
    ),
    (
        "month_first_by_a_day_over_12",
        ["02/13/2021", "03/05/2021"],
        ["2021-02-13T00:00:00", "2021-03-05T00:00:00"],
        ["Strict Structure: Y=P2, M=P0, D=P1"],
    ),
    (
        "ambiguous_year_last",
        ["05/03/2021", "01/12/2021"],
        ["2021-03-05T00:00:00", "2021-12-01T00:00:00"],
        ["Strict Structure: Y=P2, M=P1, D=P0"],
    ),
    (
        "ambiguous_year_first",
        ["2021/05/03", "2021/01/12"],
        ["2021-05-03T00:00:00", "2021-01-12T00:00:00"],
        ["Strict Structure: Y=P0, M=P1, D=P2"],
    ),
    # month first twice, day first once, in one column
    (
        "two_digit_years",
        ["05/03/21", "13/03/21", "01/12/21"],
        ["2021-05-03T00:00:00", "2021-03-13T00:00:00", "2021-01-12T00:00:00"],
        ["Fallback: No identifiable Year"],
    ),
    (
        "with_a_time",
        ["05/03/2021 14:30", "13/03/2021 08:00"],
        ["2021-05-03T14:30:00", None],
        ["Fallback: No identifiable Year"],
    ),
    (
        "compact",
        ["20210305", "20211231"],
        ["57233-12-06T00:00:00", "57236-06-19T00:00:00"],
        ["Excel Serial Detected"],
    ),
    (
        "year_only",
        ["2021", "1999"],
        ["1905-07-13T00:00:00", "1905-06-21T00:00:00"],
        ["Excel Serial Detected"],
    ),
    (
        "month_and_year",
        ["03/2021", "12/2020"],
        ["2021-03-01T00:00:00", "2020-12-01T00:00:00"],
        ["Fallback: Non-standard separators"],
    ),
    (
        "excel_serials",
        [44260, 44561],
        ["2021-03-05T00:00:00", "2021-12-31T00:00:00"],
        ["Excel Serial Detected"],
    ),
    (
        "excel_serials_as_text",
        ["44260", "44561"],
        ["2021-03-05T00:00:00", "2021-12-31T00:00:00"],
        ["Excel Serial Detected"],
    ),
    (
        "empty_excel_cells",
        [0, 1, 2],
        ["1970-01-01T00:00:00", "1970-01-01T00:00:00.000000001", "1970-01-01T00:00:00.000000002"],
        ["Fallback: Non-standard separators"],
    ),
    (
        "mixed_shapes",
        ["2021-03-05", "05/03/2021"],
        ["2021-03-05T00:00:00", None],
        ["Strict Structure: Y=P0, M=P1, D=P2"],
    ),
    (
        "mixed_separators",
        ["05.03.2021", "13-03-2021", "01/12/2021"],
        ["2021-03-05T00:00:00", "2021-03-13T00:00:00", "2021-12-01T00:00:00"],
        ["Strict Structure: Y=P2, M=P1, D=P0"],
    ),
    ("garbage", ["abc", "xyz"], [None, None], ["Fallback: Non-standard separators"]),
    ("all_missing", [None, None], [None, None], ["Empty/All-Null Strings"]),
]


def _iso(value: object) -> str | None:
    return None if pd.isna(value) else pd.Timestamp(value).isoformat()


@pytest.mark.parametrize(
    ("case", "values", "read", "notes"), CASES, ids=[case[0] for case in CASES]
)
def test_todays_parser_reads_as_it_did(case: str, values: list, read: list, notes: list) -> None:
    parsed, used = smart_parse_dates(pd.Series(values))
    assert [_iso(value) for value in parsed] == read
    assert used == notes


def test_an_empty_column_is_returned_as_it_is() -> None:
    parsed, used = smart_parse_dates(pd.Series([], dtype=object))
    assert used == ["Empty Column"] and parsed.empty


def test_a_conversion_recorded_before_m6_replays_to_the_same_table() -> None:
    """A 20-row file and the conversion Validate recorded on it at c3aa6ca, with its fingerprints and what it read."""
    fixture = json.loads((FIXTURES / "dates_m5b_conversion.json").read_text(encoding="utf-8"))
    loaded = read_path(
        str(FIXTURES / "dates_m5b.csv"), ParseOptions.from_dict(fixture["parse_options"])
    ).df
    result = apply_steps(loaded, [fixture["step"]])
    assert result.error is None and result.divergence is None
    assert frame_fingerprint(result.df) == fixture["step"]["data"]["fp_out"]
    for column, values in fixture["parsed"].items():
        assert [_iso(value) for value in result.df[column]] == values, column


def test_the_tutorials_column_profile_is_as_it_was() -> None:
    path = ROOT / "assets" / "tutorial" / "tutorial_cohort.csv"
    frame = read_table(path.read_bytes(), path.name, ParseOptions(dayfirst=True)).df
    expected = json.loads((FIXTURES / "tutorial_profile.json").read_text(encoding="utf-8"))
    assert asdict(DataAnalyzer(frame).profile) == expected
