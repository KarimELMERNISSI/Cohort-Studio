"""
Analysis results stamped with what they were computed on (audit E8, follow-up programme M1).

A page stores a result with :func:`remember` and shows it again with :func:`recall`. The stamp records the
data fingerprint, the analysis population, a hash of the analysis settings, the revision of the data and
the number of rows analysed. :func:`recall` returns the result only when all of them still hold: after a
change of the data, of the ticked criteria or of the settings, the old numbers are not shown as current;
a notice says why and the analysis is run again. :func:`provenance` prints the stamp under the result,
with a caveat when values filled by a single imputation were analysed (``columns``), and a button that adds
the result to the study dossier (follow-up programme M3): :func:`report_items` lists them.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import streamlit as st

RESULTS_KEY = "_results"
REPORT_KEY = "_report"  # the results added to the study dossier
CURRENT, STALE_DATA, STALE_POPULATION, STALE_CONFIG, MISSING = (
    "current",
    "stale_data",
    "stale_population",
    "stale_config",
    "missing",
)
STALE_CODE = (
    "stale_code"  # computed before the application was updated: never shown, never in the dossier
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
    config: Any = None  # the settings, kept to describe the result in the study dossier


@dataclass(frozen=True)
class ReportItem:
    """A result added to the study dossier, with the stamp of what it was computed on."""

    id: str
    key: str
    title: str
    settings: str
    value: Any
    stamp: ResultStamp
    added_at: str


RESULT_TITLES: dict[str, str] = {
    "survival": "Kaplan-Meier survival and log-rank test",
    "survival_mi": "Kaplan-Meier survival and log-rank test, pooled",
    "cox": "Cox proportional-hazards model",
    "cox_mi": "Cox proportional-hazards model, pooled",
    "logistic": "Logistic regression",
    "logistic_mi": "Logistic regression, pooled",
    "count": "Count regression",
    "count_mi": "Count regression, pooled",
    "univariate": "Group comparison",
    "univariate_mi": "Group comparison, pooled",
    "ancova": "ANCOVA",
    "ancova_mi": "ANCOVA, pooled",
    "paired": "Paired test",
    "paired_mi": "Paired t-test, pooled",
    "one_sample": "One-sample test",
    "one_sample_mi": "One-sample t-test, pooled",
    "mcnemar": "McNemar's test",
    "mantel_haenszel": "Mantel-Haenszel stratified analysis",
    "diagnostic": "Diagnostic accuracy",
    "bland_altman": "Bland-Altman agreement",
}


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
    _store()[key] = StoredResult(value, stamp, config)
    return stamp


def _outdated(value: Any) -> bool:
    """
    Whether ``value`` is of a class its module no longer defines: the application was updated (its modules reloaded)
    since the result was computed, and the page or the dossier would read fields the old object lacks.
    """
    cls = type(value)
    module = sys.modules.get(cls.__module__)
    if module is None or cls.__module__ == "builtins":
        return False
    current: Any = module
    for part in cls.__qualname__.split(
        "."
    ):  # a class defined in a function cannot be looked up: kept
        current = getattr(current, part, None)
    return current is not None and current is not cls


def status(key: str, *, config: Any) -> str:
    stored = _store().get(key)
    if stored is None:
        return MISSING
    if _outdated(stored.value):
        return STALE_CODE
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
    if notice and state == STALE_CODE:
        st.warning(
            "CohortStudio was updated since this result was computed; it is no longer shown. Run the analysis again.",
            icon=":material/update:",
        )
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


_TOKEN = re.compile(
    r"^[0-9a-f]{32}:"
)  # the identity of an imputation in a pooled result's settings


def describe_settings(config: Any) -> str:
    """The settings of an analysis in words: its variables and options, without internal tokens."""
    words: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, list | tuple):
            for v in value:
                walk(v)
        elif isinstance(value, str):
            if value and value != "(none)" and not _TOKEN.match(value):
                words.append(value)
        elif isinstance(value, int | float) and not isinstance(value, bool):
            words.append(f"{value:g}")

    walk(config)
    return ", ".join(dict.fromkeys(words))


def _report() -> dict[str, ReportItem]:
    basket = st.session_state.get(REPORT_KEY)
    if not isinstance(basket, dict):
        basket = {}
        st.session_state[REPORT_KEY] = basket
    return basket


def _item_id(key: str, stored: StoredResult) -> str:
    return f"{key}:{stored.stamp.config_hash}"


def add_to_report(key: str) -> None:
    """Put the stored result ``key`` in the study dossier (callback of the Add to report button)."""
    stored = _store().get(key)
    if stored is None:
        return
    item_id = _item_id(key, stored)
    _report()[item_id] = ReportItem(
        id=item_id,
        key=key,
        title=RESULT_TITLES.get(key, key.replace("_", " ").capitalize()),
        settings=describe_settings(stored.config),
        value=stored.value,
        stamp=stored.stamp,
        added_at=datetime.now().isoformat(timespec="seconds"),
    )


def remove_from_report(item_id: str) -> None:
    _report().pop(item_id, None)


def report_items() -> list[ReportItem]:
    """The results added to the study dossier, in the order they were added."""
    return list(_report().values())


def report_status(item: ReportItem) -> str:
    """
    CURRENT, or STALE_DATA / STALE_POPULATION when the data or the population changed since it was computed, or
    STALE_CODE when the application was updated since.
    """
    from ui.state import get_state

    if _outdated(item.value):
        return STALE_CODE

    identity = get_state().identity()
    if identity is None:
        return MISSING
    if item.stamp.data_fp != identity.data_fp:
        return STALE_DATA
    if item.stamp.population_fp != identity.population_fp:
        return STALE_POPULATION
    return CURRENT


def provenance(key: str) -> None:
    """Caption with what the stored result was computed on (and the single-imputation caveat), and the button
    that adds it to the study dossier."""
    stored = _store().get(key)
    if stored is None:
        return
    single_imputation_warning(dict(stored.stamp.single_imputed))
    st.caption(f":material/fingerprint: {stored.stamp.caption}")
    item_id = _item_id(key, stored)
    if item_id in _report():
        st.button(
            "In the study dossier (remove)",
            key=f"report_remove::{key}",
            icon=":material/check_circle:",
            type="tertiary",
            on_click=remove_from_report,
            args=(item_id,),
            help="Reproduce > Study dossier lists the results of the report.",
        )
    else:
        st.button(
            "Add to the study dossier",
            key=f"report_add::{key}",
            icon=":material/note_add:",
            type="tertiary",
            on_click=add_to_report,
            args=(key,),
            help="Adds this result, with what it was computed on, to the Word study dossier (Reproduce > Study "
            "dossier).",
        )


def stamp_of(key: str) -> ResultStamp | None:
    stored = _store().get(key)
    return stored.stamp if stored is not None else None


__all__ = [
    "CURRENT",
    "MISSING",
    "REPORT_KEY",
    "RESULTS_KEY",
    "RESULT_TITLES",
    "STALE_CONFIG",
    "STALE_DATA",
    "STALE_POPULATION",
    "ReportItem",
    "ResultStamp",
    "StoredResult",
    "add_to_report",
    "config_hash",
    "describe_settings",
    "provenance",
    "recall",
    "remember",
    "remove_from_report",
    "report_items",
    "report_status",
    "single_imputation_warning",
    "stamp_of",
    "status",
]
