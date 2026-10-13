"""
Legacy loader entry points used by the pages. Parsing lives in :mod:`ingest.loaders`; this module
has no Streamlit dependency (pages render their own messages).
"""

from __future__ import annotations

import hashlib
import io
import logging
from typing import Any

import pandas as pd

import manage.file_handling as mf
from core.state import ParseOptions
from ingest.loaders import (
    detect_decimal,
    detect_delimiter,
    excel_sheet_names,
    file_kind,
    read_table,
)
from ingest.loaders import detect_encoding as _detect_encoding
from utils.config_loader import (
    create_empty_config,
)

logger = logging.getLogger(__name__)


def detect_encoding(file_content: bytes, num_lines: int = 100) -> str:
    """Encoding of ``file_content`` (UTF-8, then Windows-1252, then charset-normalizer)."""
    return _detect_encoding(file_content)


def detect_separator(text: str) -> str:
    """The delimiter whose count is constant across the first lines."""
    return detect_delimiter(text)


def load_csv_with_separator(file_bytes: bytes, encoding: str, num_lines: int = 30) -> pd.DataFrame | None:
    """CSV bytes -> DataFrame with detected delimiter and decimal separator; ``None`` on failure."""
    try:
        text = file_bytes[:200_000].decode(encoding, errors="replace")
        sep = detect_delimiter(text)
        return pd.read_csv(io.BytesIO(file_bytes), sep=sep, encoding=encoding, decimal=detect_decimal(text, sep), on_bad_lines="warn")
    except Exception as exc:
        logger.error("Error loading CSV: %s", exc)
        return None


def load_dataframe(uploaded_file: Any) -> pd.DataFrame | dict | None:
    """
    DataFrame from a Streamlit upload or a file path. A multi-sheet workbook returns the sentinel
    ``{"__multi_sheet__": True, "sheet_names": [...]}`` so the caller can ask which sheets to load.
    """
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, str):
        return mf.load_dataframe(uploaded_file)
    name = getattr(uploaded_file, "name", "upload.csv")
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    try:
        kind = file_kind(name)
    except ValueError as exc:
        logger.error("%s", exc)
        return None
    try:
        if kind == "excel":
            sheets = excel_sheet_names(data)
            if len(sheets) > 1:
                return {"__multi_sheet__": True, "sheet_names": sheets}
        return read_table(data, name).df
    except Exception as exc:
        logger.error("Error loading file %s: %s", name, exc)
        return None


def _load_excel_with_selection(file_bytes: bytes, sheet_names: list, selected_sheets: list,
                               merge_key: str | None = None) -> pd.DataFrame | None:
    """One sheet, or several merged on ``merge_key`` (outer join) or stacked when no key is given."""
    try:
        options = ParseOptions(sheets=tuple(selected_sheets), merge_key=merge_key)
        return read_table(file_bytes, "workbook.xlsx", options).df
    except Exception as exc:
        logger.error("Error loading Excel sheets: %s", exc)
        return None


def get_file_hash(uploaded_file: Any) -> str:
    """MD5 of the upload's content."""
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


__all__ = ["_load_excel_with_selection", "create_empty_config", "detect_encoding", "detect_separator", "get_file_hash",
           "load_csv_with_separator", "load_dataframe"]
