"""Tabular file ingestion: sniffing, parsing with explicit options, and parse reports (no Streamlit)."""

from ingest.loaders import (
    LoadResult,
    ParseReport,
    detect_decimal,
    detect_delimiter,
    detect_encoding,
    excel_sheet_names,
    file_kind,
    read_path,
    read_table,
    sniff,
)

__all__ = [
    "LoadResult",
    "ParseReport",
    "detect_decimal",
    "detect_delimiter",
    "detect_encoding",
    "excel_sheet_names",
    "file_kind",
    "read_path",
    "read_table",
    "sniff",
]
