"""
Analysis results stamped with what they were computed on (audit E8, follow-up programme M1).

A page stores a result with :func:`remember` and shows it again with :func:`recall`. The stamp records the
data fingerprint, the analysis population, a hash of the analysis settings, the revision of the data and
the number of rows analysed. :func:`recall` returns the result only when all of them still hold: after a
change of the data, of the ticked criteria or of the settings, the old numbers are not shown as current;
a notice says why and the analysis is run again. :func:`provenance` prints the stamp under the result,
with a caveat when values filled by a single imputation were analysed (``columns``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import streamlit as st

RESULTS_KEY = "_results"
CURRENT, STALE_DATA, STALE_POPULATION, STALE_CONFIG, MISSING = (
    "current",
    "stale_data",
    "stale_population",
    "stale_config",
    "missing",
)


@dataclass(frozen=True)
class ResultStamp:
    data_fp: str
    population_fp: str
    config_hash: str
    revision: int
    dataset_label: str
    population_label: str
    n_population: int
    n_analysed: int | None
    not_used: str
    computed_at: str
    trace_head: str | None
    # (column, number of values filled by a single imputation) among the columns the analysis read
    single_imputed: tuple[tuple[str, int], ...] = ()

    @property
    def caption(self) -> str:
        parts = [
            f"Computed on {self.dataset_label}",
            f"revision {self.revision}",
            f"population: {self.population_label}",
        ]
        if self.n_analysed is not None:
            dropped = self.n_population - self.n_analysed
            analysed = f"N analysed {self.n_analysed:,}"
            if dropped > 0:
                analysed += f" ({dropped:,} not used: {self.not_used})"
            parts.append(analysed)
        parts.append(self.computed_at[11:16])
        if self.trace_head:
            parts.append(f"trace {self.trace_head[:8]}")
        return " · ".join(parts)


@dataclass(frozen=True)
class StoredResult:
    value: Any
    stamp: ResultStamp


def config_hash(config: Any) -> str:
    """Stable hash of the settings an analysis was run with (widget values, column names...)."""
    text = json.dumps(config, sort_keys=True, default=repr)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _store() -> dict[str, StoredResult]:
    store = st.session_state.get(RESULTS_KEY)
    if not isinstance(store, dict):
        store = {}
        st.session_state[RESULTS_KEY] = store
    return store


def remember(
    key: str,
    value: Any,
    *,
    config: Any,
    n_analysed: int | None = None,
    not_used: str = "missing values",
    columns: Sequence[str] | None = None,
) -> ResultStamp | None:
    """
    Store ``value`` with the identity of the data, the population and the settings it was computed on.

    ``columns`` (the variables the analysis reads) lets the stamp count the values a single imputation filled.
    """
    from ui.state import get_state

    state = get_state()
    identity = state.identity()
    if identity is None:
        return None
    state.record_population_use()
    ref = state.dataset
    tm = st.session_state.get("transformation_manager")
    stamp = ResultStamp(
        data_fp=identity.data_fp,
        population_fp=identity.population_fp,
        config_hash=config_hash(config),
        revision=identity.revision,
        dataset_label=ref.label if ref else "dataset",
        population_label=identity.population_label,
        n_population=identity.n_population,
        n_analysed=None if n_analysed is None else int(n_analysed),
        not_used=not_used,
        computed_at=datetime.now().isoformat(timespec="seconds"),
        trace_head=getattr(tm, "head_hash", None),
        single_imputed=tuple(
            state.single_imputed_counts([c for c in columns if c]).items() if columns else ()
        ),
    )
    _store()[key] = StoredResult(value, stamp)
    return stamp


def status(key: str, *, config: Any) -> str:
    stored = _store().get(key)
    if stored is None:
        return MISSING
    from ui.state import get_state

    identity = get_state().identity()
    if identity is None:
        return MISSING
    stamp = stored.stamp
    if stamp.data_fp != identity.data_fp:
        return STALE_DATA
    if stamp.population_fp != identity.population_fp:
        return STALE_POPULATION
    if stamp.config_hash != config_hash(config):
        return STALE_CONFIG
    return CURRENT


def recall(key: str, *, config: Any, notice: bool = True) -> Any | None:
    """The stored result when it still describes the current data, population and settings; else None.

    A result computed with other settings is simply not shown (the widgets changed); one computed on other
    data or another population is announced, so the user knows why it disappeared.
    """
    state = status(key, config=config)
    if state == CURRENT:
        return _store()[key].value
    if notice and state in (STALE_DATA, STALE_POPULATION):
        stamp = _store()[key].stamp
        if state == STALE_DATA:
            st.warning(
                "The data changed since this result was computed (revision "
                f"{stamp.revision}); it is no longer shown. Run the analysis again.",
                icon=":material/update:",
            )
        else:
            st.warning(
                f"This result was computed on another analysis population ({stamp.population_label}); "
                "it is no longer shown. Run the analysis again.",
                icon=":material/filter_alt:",
            )
    return None


def single_imputation_warning(counts: Mapping[str, int]) -> None:
    """Analysed values that a single imputation filled: the result treats them as observed."""
    if not counts:
        return
    total = sum(counts.values())
    detail = ", ".join(f"{col}: {n:,}" for col, n in counts.items())
    st.warning(
        f"{total:,} of the values analysed were filled by single imputation ({detail}). The analysis treats "
        "them as observed, so standard errors are too small and p-values too optimistic. For inference, use "
        "multiple imputation on the data before the single imputation.",
        icon=":material/warning:",
    )


def provenance(key: str) -> None:
    """Caption with what the stored result was computed on (and the single-imputation caveat)."""
    stored = _store().get(key)
    if stored is not None:
        single_imputation_warning(dict(stored.stamp.single_imputed))
        st.caption(f":material/fingerprint: {stored.stamp.caption}")


def stamp_of(key: str) -> ResultStamp | None:
    stored = _store().get(key)
    return stored.stamp if stored is not None else None


__all__ = [
    "CURRENT",
    "MISSING",
    "RESULTS_KEY",
    "STALE_CONFIG",
    "STALE_DATA",
    "STALE_POPULATION",
    "ResultStamp",
    "StoredResult",
    "config_hash",
    "provenance",
    "recall",
    "remember",
    "stamp_of",
    "status",
]
