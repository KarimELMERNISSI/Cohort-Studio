"""
Reading text columns as dates.

Since M6 a column is read with an explicit format (:data:`DATE_FORMATS`), detected from the shape of its values or
chosen by the user, and recorded with the conversion: the replay reads the column again with that format, never
detecting it again. Every format is read with a regular expression and an explicit ``to_datetime`` format, never
value by value through dateutil. :func:`smart_parse_dates` is the reader before M6, kept unchanged to replay the
conversions recorded with it.
"""

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import pandas as pd

from core.registry import Catalogue


def smart_parse_dates(series):
    """
    Parses dates by enforcing a single consistent format across the column.

    Detects components (Day, Month, Year) by analyzing global numeric ranges.

    Strategy:
    1. Identify Separator & Split Strings
    2. Analyze each column position (0, 1, 2)
    3. Deduction Rules:
       - Value > 31 => YEAR
       - Value > 12 (and not Year) => DAY
       - Remaining => MONTH
    4. Reconstruct Date

    Parameters:
    -----------
    series : pd.Series
        The pandas Series to parse.

    Returns:
    --------
    tuple
        (parsed_series, notes) - The parsed datetime series and a list of notes explaining the parsing logic.
    """
    if series.empty:
        return series, ["Empty Column"]

    # -------------------------------------------------------------
    # 1. EXCEL SERIALS / NUMERIC HANDLING
    # -------------------------------------------------------------
    try:
        numeric_series = pd.to_numeric(series, errors="coerce")
        # If > 50% are valid numbers, chance of Serial Date
        if numeric_series.notna().sum() > 0.5 * len(series):
            # Check validity range (e.g. year > 1900)
            res_serial = pd.to_datetime(
                numeric_series, unit="D", origin="1899-12-30", errors="coerce"
            )
            if (res_serial.dt.year > 1900).sum() > 0.1 * len(series):
                return res_serial, ["Excel Serial Detected"]
    except Exception:
        pass  # not Excel serials: the string formats are analysed next

    # -------------------------------------------------------------
    # 2. STRICT STRING STRUCTURE ANALYSIS
    # -------------------------------------------------------------
    try:
        # Work on non-null copy
        s = series.dropna().astype(str).copy()
        if s.empty:
            return pd.to_datetime(series, errors="coerce"), ["Empty/All-Null Strings"]

        # Normalize Separators
        s = s.str.replace(r"[-.]", "/", regex=True)

        # Split
        splits = s.str.split("/", expand=True)

        # Must have at least 3 parts for standard dates
        if splits.shape[1] < 3:
            # Fallback for weird formats? Or try generic parser?
            # User requested "rebuild date properly", implying standard dates.
            # Let's fallback to generic for safety if split fails.
            return pd.to_datetime(series, errors="coerce"), ["Fallback: Non-standard separators"]

        # Convert cols to numeric
        cols = []
        for i in range(3):
            cols.append(pd.to_numeric(splits[i], errors="coerce"))

        p0, p1, p2 = cols[0], cols[1], cols[2]

        # Global Range Analysis
        p0_max = p0.max(skipna=True)
        p1_max = p1.max(skipna=True)
        p2_max = p2.max(skipna=True)

        # --- IDENTIFY YEAR ---
        year_idx = -1
        # Priority: If any column goes way above 31 (e.g. 1999, 2020)
        # Check standard positions first to be safe (0 or 2)
        if p0_max > 31:
            year_idx = 0
        elif p2_max > 31:
            year_idx = 2
        elif p1_max > 31:
            year_idx = 1

        # --- IDENTIFY DAY / MONTH (every branch below sets both) ---
        if year_idx != -1:
            # We found the year. Now distinguish Day vs Month in remaining cols.
            rem = [i for i in [0, 1, 2] if i != year_idx]
            a, b = rem[0], rem[1]
            max_a = cols[a].max(skipna=True)
            max_b = cols[b].max(skipna=True)

            if max_a > 12:  # Must be Day
                day_idx = a
                month_idx = b
            elif max_b > 12:  # Must be Day
                day_idx = b
                month_idx = a
            else:
                # Both <= 12. Ambiguous.
                # Default: Day comes before Month in Euro signals? Or Position Logic?
                # If Year is last (2), usually Day is first (0). (DD/MM/YYYY)
                # If Year is first (0), usually Month is first (1). (YYYY-MM-DD - ISO)
                if year_idx == 2:
                    # DD/MM/YYYY vs MM/DD/YYYY. Euro is safer default.
                    if a == 0:
                        day_idx, month_idx = 0, 1
                    else:
                        day_idx, month_idx = 1, 0
                else:
                    # YYYY-MM-DD vs YYYY-DD-MM. ISO (Start with Y) usually implies Month next.
                    # e.g. 2020-01-25. P1=1, P2=25. P2 is Day.
                    # If both <= 12 (e.g. 2020-01-01), assume ISO standard (Year-Month-Day).
                    # So Month is likely the "middle" one (1).
                    if a == 1:
                        month_idx, day_idx = 1, 2
                    else:
                        month_idx, day_idx = 2, 1
        else:
            # No Year > 31? Maybe 2-digit year? Or data is garbage.
            return pd.to_datetime(series, errors="coerce"), ["Fallback: No identifiable Year"]

        # Reconstruct
        # Filter valid rows? No, just map index.
        df_parts = pd.DataFrame(
            {"year": cols[year_idx], "month": cols[month_idx], "day": cols[day_idx]}
        )

        parsed = pd.to_datetime(df_parts, errors="coerce")

        # Realign
        final = pd.Series(pd.NaT, index=series.index)
        final[parsed.index] = parsed

        return final, [f"Strict Structure: Y=P{year_idx}, M=P{month_idx}, D=P{day_idx}"]

    except Exception:
        return pd.to_datetime(series, errors="coerce"), ["Fallback: Exception in Struct Parser"]


# ----------------------------------------------------------------------------------------------- read with a format

LEGACY_METHOD = "smart_parse_dates"  # the conversions recorded before M6, replayed with smart_parse_dates unchanged
METHOD = "parse_dates"  # since M6: each column's format is recorded, and the replay never detects it again
CHOSEN, PROPOSED = (
    0.95,
    0.50,
)  # share of a column's values with a format's shape: chosen, or proposed to the user
ORDER_EVIDENCE = (
    0.05  # values only one order reads, as a share of the values, for the data to decide the order
)

_TIME = r"(?:[ T](?P<H>\d{1,2}):(?P<M>\d{2})(?::(?P<S>\d{2})(?:[.,]\d+)?)?)?"
_ISO = rf"(?P<y>\d{{4}})[-/.](?P<m>\d{{1,2}})[-/.](?P<d>\d{{1,2}}){_TIME}"
_SLASH = rf"(?P<a>\d{{1,2}})[/.-](?P<b>\d{{1,2}})[/.-](?P<y>\d{{2}}|\d{{4}}){_TIME}"
_COMPACT = r"(?P<x>\d{8})"
# a month and a year, or a year alone: never completed to a day, so they stay missing
_PARTIAL = r"\d{1,2}[/.-]\d{4}|\d{4}[-/.]\d{1,2}|\d{4}"
_EXCEL_ORIGIN = pd.Timestamp("1899-12-30")
_EXCEL_LAST = 2958465  # 9999-12-31
# the names of columns that may hold dates written as Excel serial numbers
_DATE_NAME = re.compile(
    r"date|dt_|_dt|dob|birth|naiss|deces|death|admission|sortie|visit|enrol|onset", re.I
)


def _text(series: pd.Series) -> pd.Series:
    """The values as trimmed text, a blank one missing."""
    text = series.astype("string").str.strip()
    return text.mask(text == "")


def _number(parts: pd.Series) -> pd.Series:
    return pd.to_numeric(parts, errors="coerce")


def _datetimes(year, month, day, hour=None, minute=None, second=None) -> pd.Series:
    """
    Datetimes from their numbers, the time of day when the values hold one: a real calendar date, else NaT (31/02,
    a month 13, a number that did not read). Written as text and read with an explicit format, in microseconds,
    which hold any year from 1 to 9999.
    """
    frame = pd.DataFrame({"y": year, "m": month, "d": day})
    for name, values in (("H", hour), ("M", minute), ("S", second)):
        frame[name] = values if values is not None else 0
    read = frame[["y", "m", "d"]].notna().all(axis=1)
    whole = frame[read].fillna(0).astype("int64")
    text = pd.Series(pd.NA, index=frame.index, dtype="string")
    if not whole.empty:
        text[read] = (
            whole["y"].astype("string").str.zfill(4)
            + "-"
            + whole["m"].astype("string").str.zfill(2)
            + "-"
            + whole["d"].astype("string").str.zfill(2)
            + " "
            + whole["H"].astype("string").str.zfill(2)
            + ":"
            + whole["M"].astype("string").str.zfill(2)
            + ":"
            + whole["S"].astype("string").str.zfill(2)
        )
    return pd.to_datetime(text, format="%Y-%m-%d %H:%M:%S", errors="coerce").astype(
        "datetime64[us]"
    )


def _times(parts: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    return _number(parts["H"]), _number(parts["M"]), _number(parts["S"])


def _year(digits: pd.Series, pivot: int | None) -> pd.Series:
    """A year of four digits as written; one of two digits by the pivot (yy <= pivot: 20yy, else 19yy), or missing
    when there is no pivot. Never strptime's %y, whose 1969/2068 split makes a 1968 birth one in 2068."""
    year = _number(digits)
    short = digits.str.len() == 2
    if pivot is None:
        return year.mask(short)
    century = year.where(year > pivot, year + 2000).where(year <= pivot, year + 1900)
    return year.mask(short, century)


def _extract(series: pd.Series, pattern: str) -> pd.DataFrame:
    return _text(series).str.extract(f"^(?:{pattern})$")


def _read_iso(series: pd.Series, pivot: int | None) -> pd.Series:
    p = _extract(series, _ISO)
    return _datetimes(_number(p["y"]), _number(p["m"]), _number(p["d"]), *_times(p))


def _read_slash(day_first: bool) -> Callable[[pd.Series, int | None], pd.Series]:
    def read(series: pd.Series, pivot: int | None) -> pd.Series:
        p = _extract(series, _SLASH)
        day, month = (p["a"], p["b"]) if day_first else (p["b"], p["a"])
        return _datetimes(_year(p["y"], pivot), _number(month), _number(day), *_times(p))

    return read


def _compact_parts(digits: pd.Series, day_first: bool) -> tuple[pd.Series, pd.Series, pd.Series]:
    if day_first:
        year, month, day = digits.str[4:8], digits.str[2:4], digits.str[0:2]
    else:
        year, month, day = digits.str[0:4], digits.str[4:6], digits.str[6:8]
    year = _number(year)
    return year.where(year.between(1800, 2199)), _number(month), _number(day)


def _read_compact(day_first: bool) -> Callable[[pd.Series, int | None], pd.Series]:
    def read(series: pd.Series, pivot: int | None) -> pd.Series:
        return _datetimes(*_compact_parts(_extract(series, _COMPACT)["x"], day_first))

    return read


def _read_excel(series: pd.Series, pivot: int | None) -> pd.Series:
    serial = pd.to_numeric(series, errors="coerce")
    serial = serial.where((serial == serial.round()) & serial.between(0, _EXCEL_LAST))
    return (_EXCEL_ORIGIN + pd.to_timedelta(serial, unit="D")).astype("datetime64[us]")


@dataclass(frozen=True)
class DateFormat:
    """A way of writing dates, read whole. Its id is recorded with each conversion and never renamed."""

    id: str
    label: str
    read: Callable[
        [pd.Series, int | None], pd.Series
    ]  # (values, two-digit-year pivot) -> datetimes, NaT unread
    other_order: str = ""  # the format that reads the same values with day and month swapped


DATE_FORMATS: Catalogue[DateFormat] = Catalogue("date format")
DATE_FORMATS.register(DateFormat("iso", "Year first (2024-12-31)", _read_iso))
DATE_FORMATS.register(
    DateFormat("day_first", "Day first (31/12/2024)", _read_slash(True), "month_first")
)
DATE_FORMATS.register(
    DateFormat("month_first", "Month first (12/31/2024)", _read_slash(False), "day_first")
)
DATE_FORMATS.register(
    DateFormat(
        "ymd_compact", "Eight digits, year first (20241231)", _read_compact(False), "dmy_compact"
    )
)
DATE_FORMATS.register(
    DateFormat(
        "dmy_compact", "Eight digits, day first (31122024)", _read_compact(True), "ymd_compact"
    )
)
DATE_FORMATS.register(
    DateFormat("excel_serial", "Excel serial numbers (45657 is 31/12/2024)", _read_excel)
)


def read_dates(series: pd.Series, format_id: str, *, pivot: int | None = None) -> pd.Series:
    """
    A column read with one format: datetimes on its index, NaT where a value does not read as a real date. A two-digit
    year reads only with a ``pivot``.
    """
    return DATE_FORMATS.get(format_id).read(series, pivot)


@dataclass(frozen=True)
class Detection:
    """How a column's values are written, as far as their shape tells: counts only, never a value."""

    format: (
        str | None
    )  # the format to read them with, chosen when ``decided``, else proposed; None: not dates
    decided: bool
    values: int  # the recorded values (a blank is missing)
    shape: int  # those with the format's shape
    day_first_only: int = (
        0  # values only the day-first order reads (first number 13-31, second 1-12)
    )
    month_first_only: int = 0
    order_source: str = (
        ""  # "data" when the values show the order; "import_setting" when it is asked, pre-filled
    )
    two_digit_years: int = 0  # read only with a pivot, which is always asked


def _order(first: int, second: int, values: int) -> bool:
    """Whether the values show their order: one reading has no value of its own, the other enough of them."""
    return second == 0 and first >= max(1, math.ceil(ORDER_EVIDENCE * values))


def _ordered(
    n: int, shape: int, day_only: int, month_only: int, dayfirst: bool, formats: tuple[str, str]
) -> Detection:
    """A format of a family read in two orders, ``formats`` (day first, month first)."""
    if _order(day_only, month_only, n):
        chosen, source = formats[0], "data"
    elif _order(month_only, day_only, n):
        chosen, source = formats[1], "data"
    else:
        chosen, source = formats[0 if dayfirst else 1], "import_setting"
    decided = shape >= CHOSEN * n and source == "data"
    return Detection(chosen, decided, n, shape, day_only, month_only, source)


def _serials(series: pd.Series, name: str, n: int) -> Detection:
    """Excel serial numbers, proposed only for a column named like a date whose values nearly all are whole numbers
    between 1 January 1900 and today: a duration in days or an age is not a date."""
    if not _DATE_NAME.search(name):
        return Detection(None, False, n, 0)
    serial = pd.to_numeric(series, errors="coerce").dropna()
    today = (pd.Timestamp.today().normalize() - _EXCEL_ORIGIN).days
    whole = int(((serial == serial.round()) & serial.between(2, today)).sum())
    if whole >= CHOSEN * n:
        return Detection("excel_serial", False, n, whole)
    return Detection(None, False, n, 0)


def detect(series: pd.Series, *, dayfirst: bool, name: str = "") -> Detection:
    """
    The format of a column, from the shape of its values: the family that at least 95 % of the values have is
    chosen, one that 50 to 95 % have is proposed. The data decide day first or month first when values that only
    one order reads exist in that order alone, at least 5 % of the values; otherwise the order is asked, pre-filled
    from the import's day-first setting (``dayfirst``). Two-digit years always ask for a pivot. Excel serial
    numbers are never chosen, only proposed. A column already of dates has none.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return Detection(None, False, int(series.notna().sum()), 0)
    text = _text(series).dropna()
    n = len(text)
    if n == 0:
        return Detection(None, False, 0, 0)
    if pd.api.types.is_numeric_dtype(series):
        return _serials(series, name, n)
    iso = int(text.str.fullmatch(_ISO).sum())
    slash = _extract(text, _SLASH)
    n_slash = int(slash["a"].notna().sum())
    digits = _extract(text, _COMPACT)["x"].dropna()
    best = max(iso, n_slash, len(digits))
    if best < PROPOSED * n:
        return _serials(series, name, n)
    if best == iso:
        return Detection("iso", iso >= CHOSEN * n, n, iso)
    if best == n_slash:
        a, b = _number(slash["a"]), _number(slash["b"])
        day_only = int((a.between(13, 31) & b.between(1, 12)).sum())
        month_only = int((a.between(1, 12) & b.between(13, 31)).sum())
        found = _ordered(n, n_slash, day_only, month_only, dayfirst, ("day_first", "month_first"))
        short = int((slash["y"].str.len() == 2).sum())
        return Detection(
            found.format,
            found.decided and not short,
            n,
            n_slash,
            day_only,
            month_only,
            found.order_source,
            short,
        )
    day_reads = _datetimes(*_compact_parts(digits, True)).notna()
    month_reads = _datetimes(*_compact_parts(digits, False)).notna()
    return _ordered(
        n,
        len(digits),
        int((day_reads & ~month_reads).sum()),
        int((month_reads & ~day_reads).sum()),
        dayfirst,
        ("dmy_compact", "ymd_compact"),
    )


def partial_dates(series: pd.Series) -> int:
    """Values that give a month and a year, or a year alone: never completed to a day, they stay missing."""
    return int(_text(series).dropna().str.fullmatch(_PARTIAL).sum())


@dataclass(frozen=True)
class Placeholder:
    """A date some systems write for an unknown one."""

    date: str  # ISO
    source: str
    preselected: (
        bool  # set to missing unless the user unticks it (never one a person may really have)
    )


PLACEHOLDERS = (
    Placeholder("1582-10-14", "SPSS date 0", True),
    Placeholder("1753-01-01", "SQL Server's earliest date", True),
    Placeholder("1899-12-30", "Excel serial 0", True),
    Placeholder("1899-12-31", "Excel serial 1", True),
    Placeholder("1900-01-01", "Excel day 1, and the unknown date of many systems", True),
    Placeholder("1904-01-01", "Excel 1904 system, serial 0", False),
    Placeholder("1960-01-01", "SAS and Stata date 0", False),
    Placeholder("1970-01-01", "Unix time 0", False),
    Placeholder("2099-12-31", "a date for no end", True),
    Placeholder("2999-12-31", "a date for no end", True),
    Placeholder("9999-12-31", "a date for no end", True),
)


def set_to_missing(dates: pd.Series, days: Sequence[str]) -> int:
    """How many read values fall on one of ``days``, the placeholder dates set to missing (their calendar day)."""
    if not days:
        return 0
    normal = pd.to_datetime(dates, errors="coerce").dt.normalize()
    return int(normal.isin([pd.Timestamp(day) for day in days]).sum())


def placeholders_found(dates: pd.Series) -> dict[str, int]:
    """How many read values fall on each placeholder date (their calendar day), the dates none falls on left out."""
    days = pd.to_datetime(dates, errors="coerce").dt.normalize()
    found = {p.date: int((days == pd.Timestamp(p.date)).sum()) for p in PLACEHOLDERS}
    return {day: n for day, n in found.items() if n}


HEAPING_MIN = 5  # values on one day before they are told heaped there
_JANUARY_FIRST = 5 / 365.25  # the share of 1 January among dates spread over the year
_MONTH_FIRST = 2 * 12 / 365.25  # twice the share of the 1st of a month among them


@dataclass(frozen=True)
class DateReport:
    """
    What reading a column found: counts only, which the conversion records; never a date of the data, a value, or
    the earliest or latest date (those are shown on screen only).
    """

    values: int  # recorded values (a blank is missing)
    unreadable: int  # recorded, but not read as a real date with the format: they become missing
    partial: int  # a month and a year, or a year alone (among the unreadable: never completed)
    ambiguous: int  # read otherwise in the other order (05/03 is 5 March or 3 May; 04/04 is not)
    two_digit_years: int
    with_time: int  # read with a time of day
    placeholders: int  # read on a placeholder date (PLACEHOLDERS), before any is set missing
    before_1900: int
    after_extraction: (
        int | None
    )  # None: no extraction date given, or a column that may hold planned dates
    january_first: (
        bool  # more than 5/365.25 of the read dates on 1 January (and at least HEAPING_MIN)
    )
    month_first_day: (
        bool  # more than 2 x 12/365.25 of them on the 1st of a month (and at least HEAPING_MIN)
    )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def date_report(
    series: pd.Series,
    format_id: str,
    *,
    pivot: int | None = None,
    extraction: date | None = None,
    dates: pd.Series | None = None,
) -> DateReport:
    """
    What reading ``series`` with a format finds. ``extraction`` is the date the data were extracted: dates after it
    are counted (None: not counted, as for a column of planned dates). Heaping on 1 January or on the 1st of a month
    hints at partial dates the source completed. ``dates``: the column already read with that format and pivot.
    """
    if dates is None:
        dates = read_dates(series, format_id, pivot=pivot)
    read = dates.dropna()
    values = int(_text(series).notna().sum())
    other = DATE_FORMATS.get(format_id).other_order
    ambiguous = 0
    if other:
        swapped = read_dates(series, other, pivot=pivot)
        ambiguous = int((dates.notna() & swapped.notna() & (swapped != dates)).sum())
    two_digit = 0
    if format_id in ("day_first", "month_first"):
        two_digit = int((_extract(series, _SLASH)["y"].str.len() == 2).sum())
    days = read.dt.normalize()
    january = int(((read.dt.month == 1) & (read.dt.day == 1)).sum())
    first = int((read.dt.day == 1).sum())
    return DateReport(
        values=values,
        unreadable=values - len(read),
        partial=partial_dates(series),
        ambiguous=ambiguous,
        two_digit_years=two_digit,
        with_time=int((read != days).sum()),
        placeholders=sum(placeholders_found(read).values()),
        before_1900=int((read < pd.Timestamp("1900-01-01")).sum()),
        after_extraction=None
        if extraction is None
        else int((days > pd.Timestamp(extraction)).sum()),
        january_first=january >= HEAPING_MIN and january > _JANUARY_FIRST * len(read),
        month_first_day=first >= HEAPING_MIN and first > _MONTH_FIRST * len(read),
    )


def convert(df: pd.DataFrame, formats: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """
    The table with each named column read as dates. ``formats`` maps a column to how it is read, as a conversion
    records it: its "format", a "pivot" for two-digit years, and the "placeholders_to_missing" dates. The page and the
    replay both call it, so a replayed conversion never detects a format again.
    """
    out = df.copy()
    for column, how in formats.items():
        dates = read_dates(out[column], str(how["format"]), pivot=how.get("pivot"))
        for day in how.get("placeholders_to_missing") or ():
            dates = dates.mask(dates.dt.normalize() == pd.Timestamp(day))
        out[column] = dates
    return out


__all__ = [
    "CHOSEN",
    "DATE_FORMATS",
    "HEAPING_MIN",
    "LEGACY_METHOD",
    "METHOD",
    "ORDER_EVIDENCE",
    "PLACEHOLDERS",
    "PROPOSED",
    "DateFormat",
    "DateReport",
    "Detection",
    "Placeholder",
    "convert",
    "date_report",
    "detect",
    "partial_dates",
    "placeholders_found",
    "set_to_missing",
    "read_dates",
    "smart_parse_dates",
]
