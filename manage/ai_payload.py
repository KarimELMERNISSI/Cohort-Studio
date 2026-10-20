"""
What the AI features may send about a dataset to a cloud provider: a summary of each column, never a row.

The column renaming, the variable taxonomy, the unit hints of Enrich and the suggested criteria of Validate all
describe the columns with :func:`column_profile`, and the data-sharing notice of Settings lists exactly this:
- numeric columns: the 5th, 25th, 50th, 75th and 95th percentiles; never the minimum or the maximum, which can
  single out one participant;
- other columns: their most frequent values with their counts, only values shared by at least ``MIN_COUNT`` rows;
- date columns: the first and the last year.

Statistics that reach a prompt by another way (a variable taxonomy imported from a file, or built before this
rule) pass through :func:`safe_stats` first; :func:`taxonomy_for_prompt` applies it to a whole taxonomy.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

MIN_COUNT = 5  # a value shared by fewer rows is not sent
TOP_VALUES = 5
PERCENTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
FEW_LEVELS = 10  # a numeric column with this many distinct values or fewer is described by its values
PERCENTILE_KEYS = tuple(f"p{round(q * 100)}" for q in PERCENTILES)
# summary statistics that column_profile never sends: the extremes single out a participant
STATISTIC_NAMES = frozenset(
    {"min", "max", "mean", "median", "std", "sd", "var", "variance", "sum", "range", "count", "first", "last"}
)
# environment switches of the LangChain tracing service, which would receive every prompt
TRACING_VARIABLES = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING")


def _frequent_values(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts().head(TOP_VALUES)
    return {str(value): int(n) for value, n in counts.items() if n >= MIN_COUNT}


def column_profile(
    df: pd.DataFrame, columns: Iterable[str] | None = None, *, date_columns: Iterable[str] = ()
) -> list[dict[str, Any]]:
    """
    One summary per column (``name``, ``type`` and ``stats``, ``top_values`` or ``range``). ``date_columns`` are read
    as dates even when they hold text (only their years are sent).
    """
    as_dates = set(date_columns)
    profiles: list[dict[str, Any]] = []
    for col in list(columns) if columns is not None else list(df.columns):
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if col in as_dates and not pd.api.types.is_datetime64_any_dtype(series):
            series = pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=True).dropna()
        info: dict[str, Any] = {"name": str(col)}
        if pd.api.types.is_datetime64_any_dtype(series):
            info["type"] = "Date"
            if not series.empty:
                info["range"] = f"{series.min().year} to {series.max().year}"
        elif pd.api.types.is_bool_dtype(series) or (
            pd.api.types.is_numeric_dtype(series) and series.nunique() <= FEW_LEVELS
        ):
            info["type"] = "Categorical"
            info["top_values"] = _frequent_values(series)
        elif pd.api.types.is_numeric_dtype(series):
            info["type"] = "Numeric"
            if not series.empty:
                quantiles = series.astype(float).quantile(list(PERCENTILES))
                info["stats"] = {f"p{round(q * 100)}": round(float(v), 6) for q, v in quantiles.items()}
        else:
            info["type"] = "Categorical"
            info["top_values"] = _frequent_values(series)
        profiles.append(info)
    return profiles


def safe_stats(stats: Any) -> dict[str, Any]:
    """
    What may be sent of a variable's statistics: the percentiles of :func:`column_profile`, or values with their
    counts when each is shared by at least ``MIN_COUNT`` rows. The minimum, maximum, mean and the like are dropped.
    """
    if not isinstance(stats, Mapping):
        return {}
    if any(key in stats for key in PERCENTILE_KEYS):
        return {key: stats[key] for key in PERCENTILE_KEYS if key in stats}
    return {
        str(value): count
        for value, count in stats.items()
        if str(value).lower() not in STATISTIC_NAMES
        and isinstance(count, int | float)
        and not isinstance(count, bool)
        and count >= MIN_COUNT
    }


def taxonomy_for_prompt(taxonomy: Mapping[str, Any]) -> dict[str, Any]:
    """A variable taxonomy as a prompt may carry it: each variable's statistics through :func:`safe_stats`."""
    out: dict[str, Any] = {}
    for name, entry in taxonomy.items():
        if isinstance(entry, Mapping) and "stats" in entry:
            entry = {**entry, "stats": safe_stats(entry["stats"])}
        out[name] = entry
    return out


def disable_remote_tracing() -> list[str]:
    """
    Switches off the LangChain tracing service (LangSmith) at start-up: a tracing variable left in ``.env`` or the
    environment would send every prompt there, outside the data-sharing confirmation. Returns the variables found on.
    """
    switched = [name for name in TRACING_VARIABLES if os.environ.get(name, "").strip().lower() in ("1", "true", "yes")]
    for name in TRACING_VARIABLES:
        os.environ[name] = "false"
    if switched:
        logger.warning("Remote tracing of the AI prompts switched off (%s was on).", ", ".join(switched))
    return switched


__all__ = [
    "MIN_COUNT",
    "PERCENTILES",
    "PERCENTILE_KEYS",
    "column_profile",
    "disable_remote_tracing",
    "safe_stats",
    "taxonomy_for_prompt",
]
