"""
What the AI features may send about a dataset to a cloud provider: a summary of each column, never a row.

The column renaming, the variable taxonomy, the unit hints of Enrich and the suggested criteria of Validate all
describe the columns with :func:`column_profile`, and the data-sharing notice of Settings lists exactly this:
- numeric columns: the 5th, 25th, 50th, 75th and 95th percentiles; never the minimum or the maximum, which can
  single out one participant;
- other columns: their most frequent values with their counts, only values shared by at least ``MIN_COUNT`` rows;
- date columns: the first and the last year.

Statistics that reach a prompt by another way (a variable taxonomy imported from a file, edited, or built before
this rule) pass through allow-lists: :func:`safe_stats` keeps the percentiles only, and :func:`taxonomy_for_prompt`
drops the keys of a taxonomy that hold values of the data.

The module also names the recipients of the data (:func:`recipient`, :func:`store_recipient`, :func:`provider_label`):
the cloud providers, and an Ollama or a Qdrant server outside this machine and the private network, which need the
user's confirmation.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from core.workspace import file_name_of
from manage.rag.providers import AI_PROVIDERS, declared

logger = logging.getLogger(__name__)

MIN_COUNT = 5  # a value shared by fewer rows is not sent
TOP_VALUES = 5
PERCENTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
# below this many values, the 5th and 95th percentiles are interpolated with the minimum and the maximum, which
# can single out one participant: no percentile is sent
MIN_QUANTILE_ROWS = 21
FEW_LEVELS = 10  # a numeric column with this many distinct values or fewer is described by its values
PERCENTILE_KEYS = tuple(f"p{round(q * 100)}" for q in PERCENTILES)
# the keys of a taxonomy entry that describe the variable, not its values: only these reach a prompt (an imported or
# edited map may hold anything in the others)
DESCRIPTIVE_KEYS = frozenset(
    {
        "standard_name",
        "description",
        "unit",
        "node_type",
        "role",
        "relationships",
        "related_formula_ids",
        "clinical_usage",
        "category",
        "formula",
        "type",
    }
)
# environment switches of the LangChain tracing service, which would receive every prompt
TRACING_VARIABLES = ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING")
THIS_MACHINE = frozenset({"", "localhost", "127.0.0.1", "::1"})
PRIVATE_SUFFIXES = (".local", ".lan", ".internal", ".home.arpa", ".localdomain")


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
            pd.api.types.is_numeric_dtype(series) and int(series.nunique()) <= FEW_LEVELS
        ):
            info["type"] = "Categorical"
            info["top_values"] = _frequent_values(series)
        elif pd.api.types.is_numeric_dtype(series):
            info["type"] = "Numeric"
            if len(series) >= MIN_QUANTILE_ROWS:
                quantiles = series.astype(float).quantile(list(PERCENTILES))
                info["stats"] = {f"p{round(q * 100)}": round(float(v), 6) for q, v in quantiles.items()}
        else:
            info["type"] = "Categorical"
            info["top_values"] = _frequent_values(series)
        profiles.append(info)
    return profiles


def _two_digits(value: float) -> str:
    """A value to two significant digits, written plainly (120, 0.95, 2,500): enough to tell a unit."""
    rounded = float(f"{value:.2g}")
    return f"{int(rounded):,}" if rounded.is_integer() else f"{rounded:g}"


def typical_ranges(
    df: pd.DataFrame, columns: Iterable[str] | None = None, *, date_columns: Iterable[str] = ()
) -> dict[str, str]:
    """
    The typical values of each number column, from the summary of :func:`column_profile` (its 5th, 50th and 95th
    percentiles, never the minimum or the maximum, and none for a column of fewer than 21 values), to two
    significant digits: "52 to 98 (median 75)", a clue to the column's unit. ``date_columns`` (a date kept as a
    number) give none.
    """
    out: dict[str, str] = {}
    for info in column_profile(df, columns, date_columns=date_columns):
        stats = info.get("stats") or {}
        if {"p5", "p50", "p95"} <= stats.keys():
            low, mid, high = (_two_digits(stats[k]) for k in ("p5", "p50", "p95"))
            out[info["name"]] = f"{low} to {high} (median {mid})"
    return out


def column_kinds(
    df: pd.DataFrame, columns: Iterable[str], *, date_columns: Iterable[str] = ()
) -> dict[str, str]:
    """
    How each column is stored, for a prompt that writes formulas on them: ``number``, ``text``, ``date`` or
    ``yes/no``. ``date_columns`` are dates even when they hold text (formulas compute with them: a subtraction gives
    days). Like the column names, it sends no value: a model told that a column holds text compares it with text, not
    with a number.
    """
    as_dates = set(date_columns)
    kinds: dict[str, str] = {}
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            kinds[str(col)] = "yes/no"
        elif col in as_dates or pd.api.types.is_datetime64_any_dtype(series):
            kinds[str(col)] = "date"
        elif pd.api.types.is_numeric_dtype(series):
            kinds[str(col)] = "number"
        else:
            kinds[str(col)] = "text"
    return kinds


def safe_stats(stats: Any) -> dict[str, Any]:
    """
    What a prompt may carry of a variable's statistics: the percentiles of :func:`column_profile`, nothing else (an
    allow-list). The minimum, maximum, mean, "0%" or "100%" of an imported or edited map are dropped.
    """
    if not isinstance(stats, Mapping):
        return {}
    return {key: stats[key] for key in PERCENTILE_KEYS if key in stats}


def frequent_values(values: Any) -> dict[str, int]:
    """The values of :func:`column_profile` that may be sent: each shared by at least ``MIN_COUNT`` rows."""
    if not isinstance(values, Mapping):
        return {}
    return {
        str(value): int(count)
        for value, count in values.items()
        if isinstance(count, int | float) and not isinstance(count, bool) and count >= MIN_COUNT
    }


def taxonomy_for_prompt(taxonomy: Mapping[str, Any]) -> dict[str, Any]:
    """
    A variable taxonomy as a prompt may carry it (allow-lists): the keys that describe each variable
    (``DESCRIPTIVE_KEYS``) and the percentiles of its statistics. An entry that is not a mapping is left out. The
    user's map itself is unchanged.
    """
    out: dict[str, Any] = {}
    for name, entry in taxonomy.items():
        if not isinstance(entry, Mapping):
            continue
        kept = {key: value for key, value in entry.items() if key in DESCRIPTIVE_KEYS}
        if percentiles := safe_stats(entry.get("stats")):
            kept["stats"] = percentiles
        out[name] = kept
    return out


def document_name(metadata: Mapping[str, Any] | None) -> str:
    """
    The document a passage comes from, as a prompt names it: the file's name, never its path on the server, which
    names the user's account and the server's folders. Both separators are read: an index can move between systems.
    """
    return file_name_of((metadata or {}).get("source")) or "Unknown"  # a folder is never named


def category_values(
    counts: Mapping[str, int], provider: str | None, base_url: str | None
) -> tuple[list[str], int]:
    """
    The values of a categorical variable that may be sent to the AI engine for it to propose which are one category
    (Data > Clean): every value to an Ollama on this machine; to any other recipient, an Ollama of the private network
    included, only those held by MIN_COUNT rows or more, since a rarer one can single out a participant. The most
    frequent first; with how many were held back. No count is ever sent.
    """
    local = provider in local_providers() and on_this_machine(base_url)
    sent = [str(v) for v, n in counts.items() if local or n >= MIN_COUNT]
    return sent, len(counts) - len(sent)


def host_of(url: str | None) -> str:
    """The host of a service address ("" when there is none), as an HTTP client reads it: a backslash counts as a
    slash, so a name and an "@" before it cannot hide the host the client connects to."""
    url = (url or "").strip().replace("\\", "/")
    return urlparse(url if "//" in url else f"//{url}").hostname or ""


def _numeric_address(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """The address a numeric host stands for, as the system's resolver reads it without a lookup (134744072 and
    0x08.0x08.0x08.0x08 are 8.8.8.8), or None for a name."""
    try:
        found = socket.getaddrinfo(host, None, flags=socket.AI_NUMERICHOST)
    except OSError, UnicodeError, ValueError:
        return None
    return ipaddress.ip_address(str(found[0][4][0]).split("%")[0]) if found else None


def on_this_machine(url: str | None) -> bool:
    return host_of(url).lower() in THIS_MACHINE


def outside_private_network(url: str | None) -> bool:
    """
    Whether an address may be anywhere: neither this machine nor a private network (the private and link-local
    address ranges, a single-label name such as a Docker service, a .local, .lan or .internal name). Data sent
    there may leave the user's organisation.
    """
    host = host_of(url).lower()
    if host in THIS_MACHINE:
        return False
    address = _numeric_address(host)
    if address is None:
        return "." in host and not host.endswith(PRIVATE_SUFFIXES)
    return not (address.is_private or address.is_loopback or address.is_link_local)


def recipient(provider: str | None, base_url: str | None = None) -> str | None:
    """
    The recipient that needs the user's confirmation: a cloud provider (any provider not known as local), or an
    Ollama outside the private network ("ollama@<host>"); None for an Ollama on this machine or the private network.
    """
    if not provider:
        return None
    if provider not in local_providers():
        return provider
    return f"{provider}@{host_of(base_url)}" if outside_private_network(base_url) else None


QDRANT = "qdrant"  # the vector database whose server can receive the documents (manage.vectorstores, not imported)
_STORE_LABELS = {QDRANT: "Qdrant"}


def store_recipient(url: str | None) -> str | None:
    """
    A Qdrant server outside this machine and the private network ("qdrant@<host>"), which needs the confirmation as
    an Ollama there does: it keeps the text of every indexed PDF and receives the vector of every search. None for
    embedded Qdrant (no address) and a server on this machine or the private network.
    """
    return f"{QDRANT}@{host_of(url)}" if url and outside_private_network(url) else None


def provider_label(provider: str) -> str:
    """The name of a recipient: "Google Gemini", or "Ollama at <host>" and "Qdrant at <host>" outside the network."""
    name, _, host = provider.partition("@")
    label = declared(name).label if name in AI_PROVIDERS.names() else _STORE_LABELS.get(name, name)
    return f"{label} at {host}" if host else label


def local_providers() -> frozenset[str]:
    """The providers that run on this machine or the user's network, as they declare (manage.rag.providers); every
    other one, an unknown one included, needs the confirmation."""
    return frozenset(name for name in AI_PROVIDERS.names() if not declared(name).cloud)


def disable_remote_tracing() -> list[str]:
    """
    Switches off the LangChain tracing service (LangSmith) at start-up: a tracing variable left in ``.env`` or the
    environment would send every prompt there, outside the data-sharing confirmation. LangSmith's global switch is
    set off too, which wins over any variable it reads later. Returns the variables that were on.
    """
    on = ("1", "true", "yes")
    switched = [name for name in TRACING_VARIABLES if os.environ.get(name, "").strip().lower() in on]
    for name in TRACING_VARIABLES:
        os.environ[name] = "false"
    try:
        from langsmith import run_trees
        from langsmith import utils as langsmith_utils
    except ImportError:  # the AI stack is not installed: nothing traces
        pass
    else:
        run_trees.configure(enabled=False)
        getattr(langsmith_utils.get_env_var, "cache_clear", lambda: None)()  # it caches what it read
    if switched:
        logger.warning("Remote tracing of the AI prompts switched off (%s was on).", ", ".join(switched))
    return switched


__all__ = [
    "MIN_COUNT",
    "MIN_QUANTILE_ROWS",
    "PERCENTILES",
    "PERCENTILE_KEYS",
    "column_kinds",
    "column_profile",
    "disable_remote_tracing",
    "document_name",
    "frequent_values",
    "host_of",
    "local_providers",
    "on_this_machine",
    "outside_private_network",
    "provider_label",
    "recipient",
    "store_recipient",
    "safe_stats",
    "taxonomy_for_prompt",
    "typical_ranges",
]
