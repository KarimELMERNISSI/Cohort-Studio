"""
File loaders with explicit, user-overridable parsing options.

Design (audit U3): the import flow *proposes* encoding, delimiter and decimal separator from the
file content, the user can override every option, and nothing is dropped silently: malformed
lines are counted and reported instead of being discarded.
"""

from __future__ import annotations

import csv
import functools
import importlib.util
import io
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.state import ParseOptions

logger = logging.getLogger(__name__)

CSV_EXTENSIONS = {".csv", ".tsv", ".txt"}
EXCEL_EXTENSIONS = {".xlsx", ".xls", ".xlsm"}
PARQUET_EXTENSIONS = {".parquet"}
SUPPORTED_EXTENSIONS = sorted(CSV_EXTENSIONS | EXCEL_EXTENSIONS | PARQUET_EXTENSIONS)
DELIMITER_CANDIDATES = (",", ";", "\t", "|")
DELIMITER_LABELS = {",": "comma ( , )", ";": "semicolon ( ; )", "\t": "tab", "|": "pipe ( | )"}


@dataclass
class ParseReport:
    """What happened while parsing; shown to the user and logged in the trace."""

    kind: str
    n_rows: int = 0
    n_cols: int = 0
    bad_lines: int = 0
    bad_line_examples: list[str] = field(default_factory=list)
    empty_columns: list[str] = field(default_factory=list)
    unnamed_columns: list[str] = field(default_factory=list)
    duplicate_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "bad_lines": self.bad_lines,
            "bad_line_examples": list(self.bad_line_examples),
            "empty_columns": list(self.empty_columns),
            "unnamed_columns": list(self.unnamed_columns),
            "duplicate_columns": list(self.duplicate_columns),
            "warnings": list(self.warnings),
        }


@dataclass
class LoadResult:
    df: pd.DataFrame
    options: ParseOptions
    report: ParseReport


def _xls_reader_installed() -> bool:
    """Legacy ``.xls`` workbooks are read by xlrd, which is not a dependency of the application."""
    return importlib.util.find_spec("xlrd") is not None


def file_kind(filename: str) -> str:
    """``csv`` | ``excel`` | ``parquet`` from the file extension; raises for anything else."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext in CSV_EXTENSIONS:
        return "csv"
    if ext in EXCEL_EXTENSIONS:
        if ext == ".xls" and not _xls_reader_installed():
            raise ValueError(
                "Legacy Excel files (.xls) need the xlrd package, which is not installed. Open the file in Excel "
                "or LibreOffice, save it as .xlsx and import it again."
            )
        return "excel"
    if ext in PARQUET_EXTENSIONS:
        return "parquet"
    raise ValueError(
        f"Unsupported file type '{ext or filename}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}."
    )


# ----------------------------------------------------------------------------- detection


def detect_encoding(data: bytes) -> str:
    """
    UTF-8 when it decodes (BOM-aware); otherwise Windows-1252 when it decodes (the usual encoding of
    Western-European Excel/CSV exports, and what short samples are most often mis-guessed as);
    otherwise the charset-normalizer best guess; otherwise latin-1. The user can always override it.
    """
    if data.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        data.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        data[:200_000].decode("cp1252")
        return "cp1252"
    except UnicodeDecodeError:
        pass
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(data[:200_000]).best()
        if best is not None and best.encoding:
            return best.encoding
    except Exception:  # pragma: no cover - optional dependency problems
        logger.debug("charset-normalizer unavailable or failed; falling back to latin-1")
    return "latin-1"


def _sample_lines(text: str, limit: int = 20) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[:limit]


def detect_delimiter(text: str) -> str:
    """
    The delimiter whose count is constant across the first 20 non-empty lines (the strongest
    structural signal). Ties between ',' and ';' prefer ';' when decimal commas are present
    (French/Luxembourgish exports). Falls back to csv.Sniffer, then to ','.
    """
    lines = _sample_lines(text)
    if not lines:
        return ","
    decimal_comma = re.search(r"\d,\d", text[:5000]) is not None
    consistent: dict[str, int] = {}
    for cand in DELIMITER_CANDIDATES:
        counts = [line.count(cand) for line in lines]
        if min(counts) > 0 and len(set(counts)) == 1:
            consistent[cand] = counts[0]
    if consistent:
        if "," in consistent and ";" in consistent and decimal_comma:
            return ";"
        return max(consistent, key=lambda c: consistent[c])
    try:
        return csv.Sniffer().sniff(text[:5000], delimiters="".join(DELIMITER_CANDIDATES)).delimiter
    except Exception:
        return ","


def detect_decimal(text: str, sep: str) -> str:
    """',' when the file uses decimal commas (only possible when the delimiter is not the comma)."""
    if sep == ",":
        return "."
    sample = text[:20000]
    commas = len(re.findall(r"(?<![\d.])\d+,\d+(?![\d,])", sample))
    dots = len(re.findall(r"(?<![\d,])\d+\.\d+(?![\d.])", sample))
    return "," if commas > dots else "."


def sniff(data: bytes, filename: str) -> ParseOptions:
    """Proposed parsing options for a file; the user can override every field."""
    kind = file_kind(filename)
    if kind != "csv":
        return ParseOptions()
    encoding = detect_encoding(data)
    try:
        text = data[:200_000].decode(encoding, errors="replace")
    except LookupError:
        encoding, text = "latin-1", data[:200_000].decode("latin-1", errors="replace")
    ext = os.path.splitext(filename)[1].lower()
    sep = "\t" if ext == ".tsv" else detect_delimiter(text)
    return ParseOptions(encoding=encoding, sep=sep, decimal=detect_decimal(text, sep))


def excel_sheet_names(data: bytes) -> list[str]:
    with pd.ExcelFile(io.BytesIO(data)) as xls:
        return [str(s) for s in xls.sheet_names]


# ----------------------------------------------------------------------------- parsing


def _column_report(df: pd.DataFrame, report: ParseReport) -> None:
    report.n_rows, report.n_cols = int(df.shape[0]), int(df.shape[1])
    cols = [str(c) for c in df.columns]
    report.unnamed_columns = [c for c in cols if c.startswith("Unnamed:")]
    report.empty_columns = [str(c) for c in df.columns if len(df) and bool(df[c].isna().all())]
    bases = [re.sub(r"\.\d+$", "", c) for c in cols]
    report.duplicate_columns = sorted({b for b in bases if bases.count(b) > 1})
    if report.unnamed_columns:
        report.warnings.append(
            f"{len(report.unnamed_columns)} column(s) have no header (shown as 'Unnamed: n'); check the header row."
        )
    if report.empty_columns:
        report.warnings.append(f"{len(report.empty_columns)} column(s) are entirely empty.")
    if report.duplicate_columns:
        report.warnings.append(
            "Duplicate column names were suffixed with .1, .2: "
            + ", ".join(report.duplicate_columns)
        )
    if report.bad_lines:
        report.warnings.append(
            f"{report.bad_lines} malformed line(s) were skipped (wrong number of fields); check the delimiter."
        )


def _read_csv(
    data: bytes, options: ParseOptions, report: ParseReport, nrows: int | None
) -> pd.DataFrame:
    bad: list[list[str]] = []

    def _on_bad_line(fields: list[str]) -> None:
        bad.append(fields)
        return None

    df = pd.read_csv(
        io.BytesIO(data),
        sep=options.sep,
        encoding=options.encoding,
        decimal=options.decimal,
        thousands=options.thousands or None,
        header=options.header,
        skiprows=options.skiprows or None,
        na_values=list(options.na_values),
        keep_default_na=True,
        on_bad_lines=_on_bad_line,
        engine="python",
        nrows=nrows,
    )
    report.bad_lines = len(bad)
    report.bad_line_examples = [
        options.sep.join(str(f) for f in fields)[:160] for fields in bad[:3]
    ]
    return df


def _read_excel(
    data: bytes, options: ParseOptions, report: ParseReport, nrows: int | None
) -> pd.DataFrame:
    with pd.ExcelFile(io.BytesIO(data)) as xls:
        names = [str(s) for s in xls.sheet_names]
        sheets = list(options.sheets) or names[:1]
        missing = [s for s in sheets if s not in names]
        if missing:
            raise ValueError(
                f"Sheet(s) not found: {', '.join(missing)}. Available: {', '.join(names)}."
            )
        frames = [
            pd.read_excel(
                xls,
                sheet_name=s,
                header=options.header,
                skiprows=options.skiprows or None,
                na_values=list(options.na_values),
                nrows=nrows,
            )
            for s in sheets
        ]
    if len(frames) == 1:
        return frames[0]
    if options.merge_key:
        usable = []
        for name, frame in zip(sheets, frames, strict=True):
            if options.merge_key not in frame.columns:
                report.warnings.append(
                    f"Sheet '{name}' has no column '{options.merge_key}' and was skipped."
                )
                continue
            usable.append(frame)
        if not usable:
            raise ValueError(f"No sheet contains the merge key '{options.merge_key}'.")
        merged = functools.reduce(
            lambda left, right: pd.merge(left, right, on=options.merge_key, how="outer"), usable
        )
        report.warnings.append(
            f"{len(usable)} sheet(s) merged on '{options.merge_key}' (outer join)."
        )
        return merged
    report.warnings.append(f"{len(frames)} sheet(s) stacked vertically (no merge key).")
    return pd.concat(frames, ignore_index=True)


def read_table(
    data: bytes, filename: str, options: ParseOptions | None = None, *, nrows: int | None = None
) -> LoadResult:
    """Parse ``data`` (the file's bytes) with ``options`` (sniffed when omitted) and report what happened."""
    kind = file_kind(filename)
    options = options or sniff(data, filename)
    report = ParseReport(kind=kind)
    if kind == "csv":
        df = _read_csv(data, options, report, nrows)
    elif kind == "excel":
        df = _read_excel(data, options, report, nrows)
    else:
        df = pd.read_parquet(io.BytesIO(data))
        if nrows is not None:
            df = df.head(nrows)
    df.columns = [str(c) for c in df.columns]
    _column_report(df, report)
    return LoadResult(df=df, options=options, report=report)


def read_path(path: str, options: ParseOptions | None = None) -> LoadResult:
    """Load a file from disk through the same engine (used by replay and path-based loading)."""
    with open(path, "rb") as handle:
        data = handle.read()
    return read_table(data, os.path.basename(path), options)


__all__ = [
    "DELIMITER_CANDIDATES",
    "DELIMITER_LABELS",
    "SUPPORTED_EXTENSIONS",
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
