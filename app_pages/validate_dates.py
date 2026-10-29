"""
Validate > Date checks: the text columns that look like dates, how they would be read, and their conversion as one
step of the trace, which the replay reads with the same method (smart_parse_dates). Its own tab since M6: above
Validate's tabs, it read those columns whenever Validate opened on new data, whichever tab was wanted.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.registry import as_code
from ingest.analyzer import DataAnalyzer
from ingest.dates import smart_parse_dates
from ui.components import no_dataset
from ui.state import get_state


def date_candidates(df: pd.DataFrame, analyzer: DataAnalyzer) -> list[str]:
    """The text columns the column profile takes for dates. Nothing is parsed: the profile is kept until the data
    change, so the line above Validate's tabs can count them on every click."""
    return [
        c
        for c in analyzer.date_cols
        if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]


def dates_found(n: int) -> str:
    """How many text columns look like dates, in the line above Validate's tabs and at the top of the tab."""
    return "1 text column looks like dates" if n == 1 else f"{n} text columns look like dates"


def _date_preview(
    df: pd.DataFrame, analyzer: DataAnalyzer
) -> tuple[list[str], dict[str, pd.Series], dict[str, str], dict[str, int]]:
    """The text columns that look like dates, their parsing, the format found and the values that would be lost."""
    candidates = date_candidates(df, analyzer)
    parsed, notes, lost = {}, {}, {}
    for col in candidates:
        series, used_notes = smart_parse_dates(df[col])
        parsed[col] = series
        notes[col] = ", ".join(used_notes)
        lost[col] = int(series.isna().sum() - df[col].isna().sum())
    return candidates, parsed, notes, lost


def render_date_checks() -> None:
    """
    The Date checks tab: how the text columns that look like dates would be read, and their conversion, one traced
    step, on request. Dates used to be re-parsed in place on every visit of the page, untraced and invisible to the
    replay.
    """
    state = get_state()
    df = state.base
    analyzer = state.analyzer("base")
    if df is None or analyzer is None:
        no_dataset()
        return
    # the parsing is kept until the data change (M5b)
    candidates, parsed, notes, lost = state.session.memo(
        "date_preview", "", lambda: _date_preview(df, analyzer)
    )
    if not candidates:
        st.caption("No text column looks like dates.")
        return
    n = len(candidates)
    st.subheader(dates_found(n), anchor=False)
    st.caption(
        "They stay as text until you convert them. The conversion is one step of the trace and is "
        "replayed with the others."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Column": candidates,
                "Detected format": [notes[c] for c in candidates],
                "Values that would become missing": [lost[c] for c in candidates],
            }
        ),
        hide_index=True,
        width="stretch",
    )
    for col, count in lost.items():
        # a share of the values recorded: a sparse column can lose half its dates and few of its rows
        recorded = int(df[col].notna().sum())
        if recorded and count > recorded * 0.05:
            st.warning(
                f"{count} of the {recorded} recorded values of {as_code(col)} ({count / recorded:.0%}) cannot be "
                "read as dates and would become missing."
            )
    if st.button(
        f"Convert {n} column{'s' if n != 1 else ''} to dates", key="convert_dates", type="primary"
    ):
        converted = df  # a copy-on-write copy of the dataset (AppState.base)
        for col in candidates:
            converted[col] = parsed[col]
        state.commit_change(
            converted,
            "data_type_conversion",
            {
                "columns": candidates,
                "new_type": "datetime",
                "method": "smart_parse_dates",
                "notes": notes,
                "values_lost": lost,
            },
            f"Converted {n} column{'s' if n != 1 else ''} to dates",
        )
        st.rerun()


__all__ = ["date_candidates", "dates_found", "render_date_checks"]
