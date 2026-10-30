"""
Clean > Dates (Validate's Date checks tab until M6's Clean page): the text columns that look like dates, and their
conversion. Each column is read with one
format (ingest.dates): detected from the shape of its values, the order of day and month from the values that show
it, else asked and pre-filled from the import's day-first setting; two-digit years with a pivot; Excel serial numbers
only when accepted; the placeholder dates the user keeps ticked set to missing. The conversion is one step of the
trace (method parse_dates, each column's format recorded), which the replay reads with the same function, never
detecting again. The columns already dates (date cells read on load, or converted here) are counted the same way, and
their placeholder dates set to missing on request: one step recorded as a conversion with the format "dates". A tab
of its own since M6: above Validate's tabs, the conversion read those columns whenever Validate opened on new data,
whichever tab was wanted.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from core.registry import as_code
from ingest.dates import (
    AS_DATES,
    CHOSEN,
    DATE_FORMATS,
    METHOD,
    PLACEHOLDERS,
    DateReport,
    Detection,
    convert,
    date_report,
    detect,
    placeholders_found,
    read_dates,
    set_to_missing,
)
from ui.components import no_dataset
from ui.state import get_state

TEXT = "text"  # the choice to leave a column as text
UNREADABLE = 0.05  # unreadable values above this share of a column's values: converting needs "Convert anyway"
_ORDERED = ("day_first", "month_first", "dmy_compact", "ymd_compact")
_CHOICES = {
    "day_first": ("day_first", "month_first"),
    "month_first": ("day_first", "month_first"),
    "dmy_compact": ("dmy_compact", "ymd_compact"),
    "ymd_compact": ("dmy_compact", "ymd_compact"),
    "iso": ("iso",),
    "excel_serial": ("excel_serial",),
}
_ORDER_FROM = {
    "data": "the data",
    "few_values": "a few values",
    "import_setting": "the import setting",
    "user": "your choice",
    "shape": "",
}


def _loaded_on(state) -> date:
    """The day the dataset was loaded: the extraction date proposed."""
    ref = state.dataset
    try:
        return date.fromisoformat(str(ref.loaded_at)[:10]) if ref is not None else date.today()
    except ValueError:
        return date.today()


def _dayfirst(state) -> bool:
    """The import's day-first setting, else the user's default for imports."""
    ref = state.dataset
    if ref is not None and ref.parse is not None:
        return bool(ref.parse.dayfirst)
    return bool(st.session_state.get("import_default_dayfirst"))


def detections(df: pd.DataFrame, *, dayfirst: bool) -> dict[str, Detection]:
    """The columns that look like dates (detect() gives them a format, chosen or proposed), and how."""
    found = {}
    for column in df.columns:
        if pd.api.types.is_bool_dtype(df[column]):
            continue
        detection = detect(df[column], dayfirst=dayfirst, name=str(column))
        if detection.format is not None:
            found[str(column)] = detection
    return found


def _detected(state) -> dict[str, Detection]:
    """The dataset's detections, kept until the data or the day-first setting change."""
    df, dayfirst = state.base, _dayfirst(state)
    return state.session.memo("date_detection", dayfirst, lambda: detections(df, dayfirst=dayfirst))


def date_candidates(state) -> list[str]:
    """
    The text columns that look like dates, by the shape of their values: nothing is read as a date, and the result is
    kept until the data change, so that the line above Validate's tabs can count them on every click (it links to
    Clean > Dates).
    """
    return list(_detected(state)) if state.base is not None else []


def dates_found(n: int) -> str:
    """How many text columns look like dates, in the line above Validate's tabs and at the top of the tab."""
    return "1 text column looks like dates" if n == 1 else f"{n} text columns look like dates"


def _label(format_id: str) -> str:
    return "Leave as text" if format_id == TEXT else DATE_FORMATS.get(format_id).label


def _spread(dates: pd.Series) -> str:
    months = int(dates.dropna().dt.month.nunique())
    return "every date in one month" if months == 1 else f"the dates over {months} months"


def _why(detection: Detection, values: pd.Series | None = None) -> str:
    """Why a column's reading is asked: counts only, never a value."""
    if detection.format == "excel_serial":
        return "whole numbers in a column named like a date. Read them as dates only if they are Excel serial numbers."
    reasons = []
    if detection.shape < CHOSEN * detection.values:
        reasons.append(
            f"{detection.shape:,} of its {detection.values:,} values have this shape; the others would be left missing."
        )
    if detection.order_source == "import_setting":
        if detection.day_first_only and detection.month_first_only:
            reasons.append(
                f"{detection.day_first_only:,} values read only day first and {detection.month_first_only:,} only "
                "month first: the order chosen leaves the others missing."
            )
        else:  # no value shows the order: what each gives (dates completed to the 1st read the same both ways)
            reasons.append(
                "No value shows whether the day or the month comes first: the order of the import setting is "
                "proposed."
            )
            if values is not None and detection.format in ("day_first", "month_first"):
                reasons.append(
                    f"Read day first, {_spread(read_dates(values, 'day_first'))}; month first, "
                    f"{_spread(read_dates(values, 'month_first'))}."
                )
    if detection.order_source == "few_values":
        shown = detection.day_first_only or detection.month_first_only
        way = "day" if detection.day_first_only else "month"
        reasons.append(
            f"Only {shown:,} values show the order, {way} first, fewer than 5 %: it is pre-filled from them. Check it."
        )
    if detection.two_digit_years:
        reasons.append(f"{detection.two_digit_years:,} values have a two-digit year.")
    return " ".join(reasons)


def _choose(
    column: str, detection: Detection, values: pd.Series | None = None
) -> dict[str, Any] | None:
    """How a column is read: as detected when the data decide, else as chosen here. None: left as text."""
    if detection.decided:
        return {"format": detection.format, "order_source": detection.order_source or "shape"}
    with st.container(border=True):
        st.markdown(f"{as_code(column)}: {_why(detection, values)}")
        options = [*_CHOICES[str(detection.format)], TEXT]
        # read from numbers (an Excel serial, eight digits): ambiguous, left as text until chosen
        default = TEXT if not detection.order_source else detection.format
        left, right = st.columns([3, 2])
        chosen = left.selectbox(
            "Read as",
            options,
            index=options.index(default),
            format_func=_label,
            key=f"date_format::{column}",
            persist_state="page",
        )
        pivot = None
        if detection.two_digit_years and chosen != TEXT:
            pivot = right.number_input(
                "Two-digit years up to (20yy)",
                min_value=0,
                max_value=99,
                value=date.today().year % 100,
                key=f"date_pivot::{column}",
                persist_state="page",
                help="A two-digit year up to this one is read 20yy, a later one 19yy (with 26: 21 is 2021, 45 is 1945). "
                "It is recorded with the conversion.",
            )
    if chosen == TEXT:
        return None
    if chosen not in _ORDERED:
        source = "user" if chosen == "excel_serial" else "shape"
    else:
        # the data's order when they showed it; a proposal read from numbers, accepted, is the user's choice
        source = (
            detection.order_source
            if chosen == detection.format and detection.order_source
            else "user"
        )
    how: dict[str, Any] = {"format": chosen, "order_source": source}
    if pivot is not None:
        how["pivot"] = int(pivot)
    return how


def _reports(
    df: pd.DataFrame, chosen: dict[str, dict[str, Any]], extraction: date | None, planned: list[str]
) -> tuple[dict[str, DateReport], dict[str, pd.Series]]:
    """
    Each chosen column read as it would be converted (before the placeholders are set to missing), and what the
    reading found; a column that may hold planned dates is not counted after the extraction date.
    """
    readings, reports = {}, {}
    for column, how in chosen.items():
        readings[column] = read_dates(df[column], how["format"], pivot=how.get("pivot"))
        reports[column] = date_report(
            df[column],
            how["format"],
            pivot=how.get("pivot"),
            extraction=None if column in planned else extraction,
            dates=readings[column],
        )
    return reports, readings


def _day(value) -> str:
    return "" if pd.isna(value) else pd.Timestamp(value).strftime("%d/%m/%Y")


def _placeholders(chosen: Collection[str], readings: dict[str, pd.Series]) -> dict[str, list[str]]:
    """
    Per column, the placeholder dates set to missing, chosen from the whole catalogue: a list of only the dates a
    column holds told the trace that a participant was born on 1 January 1970, and one choice for every column set a
    real birth date to missing with a placeholder visit. The counts are this column's, on screen only.
    """
    catalogue = {p.date: p for p in PLACEHOLDERS}
    found = {c: placeholders_found(readings[c]) for c in chosen}
    chosen_days: dict[str, list[str]] = {}
    with st.expander(
        "Placeholder dates set to missing",
        icon=":material/event_busy:",
        expanded=any(found.values()),
    ):
        st.caption(
            "Dates some systems write for an unknown one. Those no person really has are ticked in every column; "
            "1 January 1970, 1960 or 1904 can be a real birth date, and stay unless you tick them."
        )
        for column in chosen:
            here = found[column]
            picked = st.multiselect(
                as_code(column),
                list(catalogue),
                default=[day for day, p in catalogue.items() if p.preselected],
                format_func=lambda day, here=here: (
                    f"{day} ({catalogue[day].source})"
                    + (f": {here[day]:,} here" if day in here else "")
                ),
                key=f"date_placeholders::{column}",
                persist_state="page",
            )
            chosen_days[column] = [
                day for day in catalogue if day in picked
            ]  # in the catalogue's order
    return chosen_days


def _study_dates(state, options: list[str]) -> tuple[date | None, list[str]]:
    """The extraction date and the columns that may hold planned dates: asked once, for both kinds of columns."""
    left, right = st.columns(2)
    extraction = left.date_input(
        "Extraction date",
        value=_loaded_on(state),
        format="DD/MM/YYYY",
        key="date_extraction",
        persist_state="page",
        help="The day the data were taken from their source: the dates after it are counted. Pre-filled with the day "
        "the dataset was loaded.",
    )
    planned = right.multiselect(
        "Columns that may hold planned dates",
        options,
        key="date_planned",
        persist_state="page",
        help="A next appointment or a planned end: their dates after the extraction date are not counted.",
    )
    return extraction, planned


def _heaping(reports: dict[str, DateReport]) -> None:
    for column, report in reports.items():
        if report.january_first or report.month_first_day:
            where = "1 January" if report.january_first else "the 1st of a month"
            st.caption(
                f":material/info: {as_code(column)}: many dates fall on {where}. Did the source complete dates it "
                "knew only by their month or year?"
            )


def _extraction_held(extraction: date | None, dates: list[pd.Series]) -> None:
    """
    A warning when 1 to 4 values of the data fall on the extraction date: a participant's date typed in the box is
    recorded with the step as typed. A count, on screen only.
    """
    if extraction is None:
        return
    day = pd.Timestamp(extraction)
    held = sum(int((values.dt.normalize() == day).sum()) for values in dates)
    if 0 < held < 5:
        st.warning(
            f"{held} value{'s' if held != 1 else ''} of the data fall{'s' if held == 1 else ''} on the extraction "
            "date, which is recorded with the step as typed: check that it is the day the data were extracted, not a "
            "participant's date.",
            icon=":material/event:",
        )


def already_dates(df: pd.DataFrame) -> list[str]:
    """The columns that already hold dates without a time zone: date cells read on load, or a conversion's."""
    return [
        str(c)
        for c in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[c]) and getattr(df[c].dt, "tz", None) is None
    ]


def _dated_reports(
    df: pd.DataFrame, dated: list[str], extraction: date | None, planned: list[str]
) -> dict[str, DateReport]:
    """What each column already dates holds, counted as for a conversion: nothing to read, the dates themselves."""
    return {
        c: date_report(
            df[c], AS_DATES, extraction=None if c in planned else extraction, dates=df[c]
        )
        for c in dated
    }


def render_date_checks() -> None:
    """
    The Date checks tab: how the text columns that look like dates would be read, what is asked, and their conversion,
    one traced step, on request; then what the columns already dates hold. Dates used to be re-parsed in place on
    every visit of the page, untraced and invisible to the replay.
    """
    state = get_state()
    df = state.base
    if df is None:
        no_dataset()
        return
    found = _detected(state)
    dated = already_dates(df)
    study = None
    if found:
        study = _text_columns(state, df, found, dated)
    else:
        st.caption("No text column looks like dates.")
    if dated:
        _date_columns(state, df, dated, study)
    zoned = [
        str(c)
        for c in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[c]) and getattr(df[c].dt, "tz", None) is not None
    ]
    if zoned:
        n = len(zoned)
        st.caption(
            f"{n} column{'s hold' if n != 1 else ' holds'} dates with a time zone "
            f"({', '.join(as_code(c) for c in zoned[:5])}{' ...' if n > 5 else ''}): not checked here."
        )


def _text_columns(
    state, df: pd.DataFrame, found: dict[str, Detection], dated: list[str]
) -> tuple[date | None, list[str]] | None:
    """
    The text columns that look like dates: how each would be read, and their conversion. Returns the extraction date
    and the planned-date columns it asked for (None: no column chosen, nothing asked).
    """
    st.subheader(dates_found(len(found)), anchor=False)
    st.caption(
        "They stay as text until you convert them. Each is read with one format, recorded with the conversion: the "
        "replay reads it again the same way. A value is never completed: a month and a year, or a year alone, stays "
        "missing."
    )
    decided = [c for c, d in found.items() if d.decided]
    as_text = (
        st.multiselect(
            "Leave as text",
            decided,
            key="date_leave_text",
            persist_state="page",
            help="The columns the data decided to read as dates: one listed here stays text (an identifier made of "
            "digits, a column you will not analyse as dates).",
        )
        if decided
        else []
    )
    chosen = {}
    for column, detection in found.items():
        how = None if column in as_text else _choose(column, detection, df[column])
        if how is not None:
            chosen[column] = how
    if not chosen:
        st.caption("No column is chosen to be read as dates.")
        return None
    # every text column, those left as text too: an option that disappears takes its choice with it
    extraction, planned = _study_dates(state, [*found, *dated])
    key = tuple(
        sorted((c, how["format"], how.get("pivot"), c in planned) for c, how in chosen.items())
    )
    reports, readings = state.session.memo(
        "date_readings", (key, str(extraction)), lambda: _reports(df, chosen, extraction, planned)
    )
    _extraction_held(extraction, [*readings.values(), *(df[c] for c in dated)])
    # the earliest and latest dates are shown here only, never recorded
    st.dataframe(
        pd.DataFrame(
            {
                "Column": list(chosen),
                "Read as": [_label(how["format"]) for how in chosen.values()],
                "Order from": [_ORDER_FROM[how["order_source"]] for how in chosen.values()],
                "Values": [reports[c].values for c in chosen],
                "Cannot be read": [reports[c].unreadable for c in chosen],
                "Partial dates": [reports[c].partial for c in chosen],
                "Ambiguous": [reports[c].ambiguous for c in chosen],
                "Two-digit years": [reports[c].two_digit_years for c in chosen],
                "With a time": [reports[c].with_time for c in chosen],
                "Before 1900": [reports[c].before_1900 for c in chosen],
                "After extraction": [reports[c].after_extraction for c in chosen],
                "Earliest": [_day(readings[c].min()) for c in chosen],
                "Latest": [_day(readings[c].max()) for c in chosen],
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Ambiguous: dates that read otherwise in the other order (5 March or 3 May), read as *Order from* says. "
        "Partial dates are among those that cannot be read: left missing, never completed. For an age or a time since "
        "diagnosis, the usual way is a stated completion (the 15th, or 1 July) as a derived variable."
    )
    _heaping(reports)
    for column, report in reports.items():
        if report.unreadable:  # on screen only, never recorded, and only when asked for
            text = df[column].astype("string").str.strip()
            unread = (
                text[text.notna() & (text != "") & readings[column].isna()]
                .drop_duplicates()
                .head(5)
            )
            with st.expander(
                f"Values of {as_code(column)} that cannot be read", icon=":material/search:"
            ):
                st.caption("For example: " + ", ".join(as_code(v) for v in unread))

    to_missing = _placeholders(chosen, readings)

    held = []  # a column losing more than UNREADABLE converts only when told, apart from the others
    for column in [c for c in chosen if reports[c].unreadable > UNREADABLE * reports[c].values]:
        unread, values = reports[column].unreadable, reports[column].values
        st.warning(
            f"{unread:,} of the {values:,} recorded values of {as_code(column)} ({unread / values:.0%}) cannot be read "
            "as dates and would become missing."
        )
        if not st.checkbox(
            f"Convert {as_code(column)} anyway", key=f"dates_convert_anyway::{column}"
        ):
            held.append(column)
    chosen = {c: how for c, how in chosen.items() if c not in held}
    n = len(chosen)
    if st.button(
        f"Convert {n} column{'s' if n != 1 else ''} to dates (in place)",
        key="convert_dates",
        type="primary",
        disabled=n == 0,
    ):
        # counts only: never a date of the data (the extraction date is the study's)
        formats = {
            c: {
                **how,
                # the rule, the same whatever the column holds: the dates found are counted, never listed
                "placeholders_to_missing": to_missing[c],
                "set_to_missing": set_to_missing(readings[c], to_missing[c]),
                **({"planned": True} if c in planned else {}),
                **reports[c].to_record(),
            }
            for c, how in chosen.items()
        }
        state.commit_change(
            convert(df, formats),
            "data_type_conversion",
            {
                "columns": list(chosen),
                "new_type": "datetime",
                "method": METHOD,
                "formats": formats,
                "dayfirst": _dayfirst(state),
                "extraction": extraction.isoformat() if extraction else None,
            },
            f"Converted {n} column{'s' if n != 1 else ''} to dates",
        )
        st.rerun()
    return extraction, planned


def _date_columns(
    state, df: pd.DataFrame, dated: list[str], study: tuple[date | None, list[str]] | None
) -> None:
    """
    The columns already dates (date cells read on load, or converted here): what they hold, counted as for a
    conversion, and the placeholder dates ticked set to missing on request, one step recorded as a conversion with the
    format "dates", which the replay reads the same way. They had no check: an Excel file's 1900-01-01, or a date after
    the extraction, stayed uncounted.
    """
    n = len(dated)
    st.subheader(f"{n} column{'s hold' if n != 1 else ' holds'} dates", anchor=False)
    st.caption(
        "Read as dates on load (date cells) or converted here: what they hold. Their placeholder dates can be set to "
        "missing, one step of the trace."
    )
    extraction, planned = study if study is not None else _study_dates(state, dated)
    key = (tuple(dated), tuple(c in planned for c in dated), str(extraction))
    reports = state.session.memo(
        "date_reports_dated", key, lambda: _dated_reports(df, dated, extraction, planned)
    )
    if study is None:  # the text columns' section did not ask for the extraction date: checked here
        _extraction_held(extraction, [df[c] for c in dated])
    # the earliest and latest dates are shown here only, never recorded
    st.dataframe(
        pd.DataFrame(
            {
                "Column": dated,
                "Values": [reports[c].values for c in dated],
                "With a time": [reports[c].with_time for c in dated],
                "Placeholders": [reports[c].placeholders for c in dated],
                "Before 1900": [reports[c].before_1900 for c in dated],
                "After extraction": [reports[c].after_extraction for c in dated],
                "Earliest": [_day(df[c].min()) for c in dated],
                "Latest": [_day(df[c].max()) for c in dated],
            }
        ),
        hide_index=True,
        width="stretch",
    )
    _heaping(reports)
    to_missing = _placeholders(dated, {c: df[c] for c in dated})
    counts = {c: set_to_missing(df[c], to_missing[c]) for c in dated}
    total = sum(counts.values())
    if st.button(
        f"Set {total:,} placeholder date{'s' if total != 1 else ''} to missing",
        key="dates_set_placeholders",
        disabled=total == 0,
    ):
        columns = [c for c in dated if counts[c]]
        # counts only, as for a conversion: the rule, never a date of the data
        formats = {
            c: {
                "format": AS_DATES,
                "placeholders_to_missing": to_missing[c],
                "set_to_missing": counts[c],
                # no count after the extraction date: a conversion records its own, and two extraction dates for
                # one column would bound a date between them
                **{k: v for k, v in reports[c].to_record().items() if k != "after_extraction"},
            }
            for c in columns
        }
        state.commit_change(
            convert(df, formats),
            "data_type_conversion",
            {
                "columns": columns,
                "new_type": "datetime",
                "method": METHOD,
                "formats": formats,
                "dayfirst": _dayfirst(state),
            },
            f"Set {total} placeholder date{'s' if total != 1 else ''} to missing",
        )
        st.rerun()


__all__ = [
    "UNREADABLE",
    "already_dates",
    "date_candidates",
    "dates_found",
    "detections",
    "render_date_checks",
]
