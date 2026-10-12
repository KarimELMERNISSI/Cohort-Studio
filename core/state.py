"""Typed application state: value objects shared by the UI, the loaders and the trace (pure Python)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

DEFAULT_NA_TOKENS: tuple[str, ...] = ("", "NA", "N/A", "n/a", "NaN", "nan", "NULL", "null", "None", "-", "?", "#N/A")

_SAVED_NAME = re.compile(r"^(?P<base>.+)_v(?P<version>\d+)$")


@dataclass(frozen=True)
class ParseOptions:
    """How a tabular file is parsed. Every field is user-overridable in the import flow and logged in the trace."""

    encoding: str = "utf-8"
    sep: str = ","
    decimal: str = "."
    thousands: str | None = None
    dayfirst: bool = False
    header: int = 0
    skiprows: int = 0
    na_values: tuple[str, ...] = DEFAULT_NA_TOKENS
    sheets: tuple[str, ...] = ()
    merge_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ParseOptions:
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in known}
        if "na_values" in clean and clean["na_values"] is not None:
            clean["na_values"] = tuple(clean["na_values"])
        if "sheets" in clean and clean["sheets"] is not None:
            clean["sheets"] = tuple(clean["sheets"])
        return cls(**clean)


@dataclass(frozen=True)
class DatasetRef:
    """Identity of the dataset currently loaded in the session."""

    name: str
    origin: str = "upload"  # upload | saved | sample | path
    version: int | None = None
    file_hash: str | None = None
    source_name: str | None = None
    source_path: str | None = None
    n_rows: int = 0
    n_cols: int = 0
    loaded_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    parse: ParseOptions | None = None

    @property
    def label(self) -> str:
        return f"{self.name} (v{self.version})" if self.version else self.name

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["parse"] = self.parse.to_dict() if self.parse else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetRef:
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in known}
        clean["parse"] = ParseOptions.from_dict(clean.get("parse")) if clean.get("parse") else None
        return cls(**clean)

    @classmethod
    def from_saved_name(cls, saved_name: str, **extra: Any) -> DatasetRef:
        """``cohort_v3`` -> name ``cohort``, version 3 (the DuckDB store's naming convention)."""
        match = _SAVED_NAME.match(saved_name)
        if match:
            return cls(name=match.group("base"), version=int(match.group("version")), origin="saved", **extra)
        return cls(name=saved_name, origin="saved", **extra)


def dataset_base_name(filename: str) -> str:
    """A clean dataset name from a file name: ``My Cohort (2024).xlsx`` -> ``My_Cohort_2024``."""
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", filename or "").strip()
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    return stem or "dataset"


__all__ = ["DEFAULT_NA_TOKENS", "DatasetRef", "ParseOptions", "dataset_base_name"]
