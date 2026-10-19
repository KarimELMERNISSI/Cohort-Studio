"""
How a delimited text file is read (M5): the delimiter choice shared by Data > Import, Enrich's external file and
Validate's Compare. The usual delimiters are offered with space, and any other single character can be typed; the
engine (``ingest.loaders.separator_problem``) refuses one that cannot separate columns. The options chosen are
recorded with the step that uses the file, so the replay reads it the same way.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import streamlit as st

from core.state import ParseOptions
from ingest.loaders import (
    DELIMITER_CANDIDATES,
    delimiter_label,
    file_kind,
    read_table,
    separator_problem,
    sniff,
)

OTHER = "other"
CHOICES = (*DELIMITER_CANDIDATES, " ", OTHER)


def _label(choice: str) -> str:
    return "other…" if choice == OTHER else delimiter_label(choice)


def delimiter_input(current: str, *, key: str, container: Any = None) -> str:
    """
    The delimiter kept or chosen. One the list does not hold (detected, such as ':' or '^', or typed before) is
    shown in the 'other' box. Returns '' while that box is empty.
    """
    target = container if container is not None else st
    index = CHOICES.index(current) if current in CHOICES else CHOICES.index(OTHER)
    picked = target.radio(
        "Delimiter",
        CHOICES,
        index=index,
        format_func=_label,
        key=key,
        help="The character between the columns. Choose 'other…' for any other single character.",
    )
    if picked != OTHER:
        return picked
    typed = target.text_input(
        "Other delimiter",
        value="" if current in CHOICES else current,
        max_chars=1,
        key=f"{key}_other",
        placeholder="e.g. : or ^",
        help="The one character between the columns of this file.",
    )
    if typed:
        target.caption(f"Read with: {delimiter_label(typed)}")
    return typed


def csv_options(data: bytes, filename: str, *, key: str) -> ParseOptions | None:
    """
    For a delimited text file read outside Data > Import: the detected delimiter and decimal separator, which the
    user can change; None for an Excel or Parquet file, which carries its own structure. The section opens by itself
    when the detected delimiter finds one column only. While the chosen delimiter cannot separate columns, the
    problem is shown and the file is read with the detected options.
    """
    try:
        if file_kind(filename) != "csv":
            return None
    except ValueError:
        return None
    detected = sniff(data, filename)
    try:
        one_column = read_table(data, filename, detected, nrows=20).df.shape[1] <= 1
    except Exception:
        one_column = True  # the detected options cannot read it: let the user choose
    with st.expander(
        "How the file is read (delimiter, decimal separator)",
        expanded=one_column,
        icon=":material/table_view:",
    ):
        st.caption(
            f"Detected: {delimiter_label(detected.sep)}, decimal separator '{detected.decimal}'."
        )
        if one_column:
            st.warning(
                "The detected delimiter finds a single column: choose the one this file uses."
            )
        c1, c2 = st.columns(2)
        sep = delimiter_input(detected.sep, key=f"{key}_sep", container=c1)
        decimal = c2.radio(
            "Decimal separator",
            (".", ","),
            index=0 if detected.decimal == "." else 1,
            key=f"{key}_dec",
            help="Use ',' for files exported with French or German settings.",
        )
        chosen = replace(detected, sep=sep, decimal=decimal)
        if problem := separator_problem(chosen):
            st.warning(f"{problem} Meanwhile the file is read with the detected options.")
            return detected
    return chosen


__all__ = ["CHOICES", "OTHER", "csv_options", "delimiter_input"]
