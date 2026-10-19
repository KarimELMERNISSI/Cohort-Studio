"""
Loader entry points used by the pages. Parsing lives in :mod:`ingest.loaders`, the engine the import and the
replay use, so a file read here, by upload or by path, gives the table the replay will read again. Each table
carries the options it was parsed with in ``df.attrs["parse_options"]``, for the step that records it. This module
has no Streamlit dependency (pages render their own messages).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import pandas as pd

from core.state import ParseOptions
from ingest.loaders import (
    LoadResult,
    excel_sheet_names,
    file_kind,
    read_path,
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


def _with_options(result: LoadResult) -> pd.DataFrame:
    """The table, carrying the options it was parsed with (recorded by the step that uses it, for the replay)."""
    result.df.attrs["parse_options"] = result.options.to_dict()
    return result.df


def load_dataframe(uploaded_file: Any, options: ParseOptions | None = None) -> pd.DataFrame | dict | None:
    """
    DataFrame from a Streamlit upload or a file path, read with ``options`` (the ones the user chose in
    ``ui.components.csv_options``; detected when omitted). A multi-sheet workbook uploaded returns the sentinel
    ``{"__multi_sheet__": True, "sheet_names": [...]}`` so the caller can ask which sheets to load; a workbook read
    by path gives its first sheet, as the replay does. A file that cannot be read by path raises.
    """
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, str):
        return _with_options(read_path(uploaded_file, options))
    name = getattr(uploaded_file, "name", "upload.csv")
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    try:
        uploaded_file.seek(0)
    except Exception:
        pass  # not seekable: the bytes are already read
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
        return _with_options(read_table(data, name, options))
    except Exception as exc:
        logger.error("Error loading file %s: %s", name, exc)
        return None


def _load_excel_with_selection(file_bytes: bytes, sheet_names: list, selected_sheets: list,
                               merge_key: str | None = None) -> pd.DataFrame | None:
    """One sheet, or several merged on ``merge_key`` (outer join) or stacked when no key is given."""
    try:
        options = ParseOptions(sheets=tuple(selected_sheets), merge_key=merge_key)
        return _with_options(read_table(file_bytes, "workbook.xlsx", options))
    except Exception as exc:
        logger.error("Error loading Excel sheets: %s", exc)
        return None


def get_file_hash(uploaded_file: Any) -> str:
    """MD5 of the upload's content."""
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


__all__ = ["_load_excel_with_selection", "create_empty_config", "detect_encoding", "get_file_hash", "load_dataframe"]
