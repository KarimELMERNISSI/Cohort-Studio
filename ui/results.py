"""
Analysis results stamped with what they were computed on (audit E8, follow-up programme M1).

A page stores a result with :func:`remember` and shows it again with :func:`recall`. The stamp records the
data fingerprint, the analysis population, a hash of the analysis settings, the revision of the data and
the number of rows analysed. :func:`recall` returns the result only when all of them still hold: after a
change of the data, of the ticked criteria or of the settings, the old numbers are not shown as current;
a notice says why and the analysis is run again. :func:`provenance` prints the stamp under the result,
with a caveat when values filled by a single imputation were analysed (``columns``), and a button that adds
the result to the study dossier (follow-up programme M3): :func:`report_items` lists them.

A calculation from numbers typed (a calculator's result) is no result of the dataset: it is never stored here with a
stamp, never stale and never in the participant flow. The dossier keeps it apart (:func:`add_typed`,
:func:`typed_stored`), as JSON only: its spec, where its numbers come from, and when it was added. One reader
serves the page's list and the Word file (``app_pages.study_dossier``), which compute it again from that JSON.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import streamlit as st

from biostats.plan import AnalysisSpec, ComputedOn, read_spec, titles
from manage.report_tables import (
    TABLE_KIND,
    result_title,
    table_held_back,
    typed_numbers,
)
from ui.components.text import plain

logger = logging.getLogger(__name__)

RESULTS_KEY = "_results"
REPORT_KEY = "_report"  # the results added to the study dossier
TYPED_KEY = "_report_typed"  # the calculations from numbers typed added to it: JSON only
# the four counts, sorted, of each accuracy result whose 2x2 table the dossier does not write (counts only)
HELD_TABLES_KEY = "_held_tables"
NOTE_MAX = 200  # characters of "Where the numbers come from"
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
    # a result of several analyses, each on its own rows (a group comparison of several variables): the largest
    # number analysed, n_analysed being the smallest; None when the result has one number analysed
    n_most: int | None = None
    # the rows of the population left out for another reason than a missing value (groups not compared, groups too
    # small to draw), and the words for them
    n_left_out: int = 0
    left_out: str = ""

    @property
    def caption(self) -> str:
        parts = [
            f"Computed on {self.dataset_label}",
            f"data revision {self.revision}",  # the data's own counter: a revision is the study definition's
            f"population: {self.population_label}",
        ]
        if self.n_analysed is not None:
            from core.attrition import not_used_words

            analysed = f"N analysed {self.n_analysed:,}"
            if self.n_most is not None and self.n_most != self.n_analysed:
                analysed += f" to {self.n_most:,} by variable"
            why = not_used_words(
                self.n_population,
                self.n_analysed,
                self.n_most,
                self.n_left_out,
                self.left_out,
                self.not_used,
            )
            if why:
                analysed += f" ({'; '.join(why)})"
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
    # what was asked and on what (biostats.plan): a study package computes the result again from it; None for a kind
    # that has no spec yet, or a result stored before specs existed
    spec: AnalysisSpec | None = None


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
    spec: AnalysisSpec | None = None  # as StoredResult.spec


# the title of every stored key, from the catalogue of analysis kinds (biostats.plan.KINDS)
RESULT_TITLES: dict[str, str] = titles()


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
    spec: AnalysisSpec | None = None,
    n_most: int | None = None,
    n_left_out: int = 0,
    left_out: str = "",
) -> ResultStamp | None:
    """
    Store ``value`` with the identity of the data, the population and the settings it was computed on.

    ``columns`` (the variables the analysis reads) lets the stamp count the values a single imputation filled.
    ``spec`` (the analysis as it ran, biostats.plan.run) is kept with what it was computed on (:func:`_computed_on`).
    ``n_most`` is the largest number analysed when the result holds several analyses, each on its own rows
    (``n_analysed`` is then the smallest). ``n_left_out`` counts the rows left out for another reason than a missing
    value, and ``left_out`` says which ("not in the compared groups").
    """
    from ui.state import get_state

    if spec is not None and spec.typed is not None:
        # stamped, it would read "Computed on <dataset>, N analysed" for numbers that are not the dataset's, and go
        # stale with data it never read
        raise ValueError(
            "a calculation from numbers typed is not a result of the dataset: it is kept with add_typed"
        )
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
        n_most=None if n_most is None else int(n_most),
        n_left_out=int(n_left_out),
        left_out=left_out if n_left_out else "",
    )
    if spec is not None:
        try:
            spec = spec.on(_computed_on(state, identity, spec.pooled, stamp, tm))
        except Exception as exc:
            # the result is computed: it is kept without its spec (a package then lists it as not run), rather than
            # reported as a model that could not be fitted
            logger.warning("the spec of %s could not be recorded: %s", key, type(exc).__name__)
            spec = None
    _store()[key] = StoredResult(value, stamp, config, spec)
    if key == TABLE_KIND:
        _keep_held_table(value)
    return stamp


def _keep_held_table(value: Any) -> None:
    """
    Keep, for the session's dataset, the four counts of an accuracy result whose 2x2 table the dossier does not
    write (``manage.report_tables.table_held_back``): sorted, without a name, cleared with the dataset. The page
    keeps one accuracy result at a time, so that the next run replaced it, and its table typed into the calculator
    afterwards was written in the dossier, with the predictive value "1 of 1" the dossier holds back.
    """
    try:
        cells = table_held_back(value)
    except Exception:  # a result of a shape this version does not read: nothing of it to keep
        cells = None
    if cells is None:
        return
    kept = st.session_state.get(HELD_TABLES_KEY)
    kept = list(kept) if isinstance(kept, list) else []
    if cells not in kept:
        kept.append(cells)
    st.session_state[HELD_TABLES_KEY] = kept


def held_tables_kept() -> list[tuple[int, ...]]:
    """The four counts, sorted, of each accuracy result computed on the session's dataset whose table the dossier
    does not write (:func:`_keep_held_table`)."""
    kept = st.session_state.get(HELD_TABLES_KEY)
    return [tuple(cells) for cells in kept] if isinstance(kept, list) else []


def _computed_on(
    state: Any, identity: Any, pooled: bool, stamp: ResultStamp, trace: Any
) -> ComputedOn:
    """
    What an analysis was computed on, for its spec: the data's fingerprint, the fingerprint of the analysis
    population's row ids (which does not depend on the pandas version, unlike the data's), the numbers of rows, the
    study definition's hash, the version of the rules that read missing values and criteria, the trace's head and
    its number of steps (saving a revision of the study definition adds a step that changes no data: the head then
    moves, and the number of steps says how far behind a result is), and for a pooled analysis the imputation by its
    content.
    """
    from core.criteria import RULES_VERSION
    from core.protocol import protocol_hash

    population = state.population
    imputation = None
    if pooled:
        from ui.components.mi_mode import imputation_reference

        imputation = imputation_reference(state)
    return ComputedOn(
        mode="imputed" if pooled else "observed",
        data_fp=identity.data_fp,
        n_population=identity.n_population,
        n_analysed=stamp.n_analysed,
        rows_fp=None if population is None else population.rows_fingerprint,
        protocol_hash=protocol_hash(state.protocol_body()),
        rules_version=RULES_VERSION,
        trace_head=stamp.trace_head,
        trace_steps=len(getattr(trace, "history", None) or []),
        imputation=imputation,
    )


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
                "The data changed since this result was computed (data revision "
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


# the identity of an imputation in a pooled result's settings (ui.components.mi_mode.mi_token): by its cells'
# fingerprint, or by its run (results stored before M8), never words of the settings
_TOKEN = re.compile(r"mi:[0-9a-f]{16}(?::subset)?\Z|[0-9a-f]{32}:")  # matched from the start


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
        title=result_title(
            key, stored.value, RESULT_TITLES.get(key, key.replace("_", " ").capitalize())
        ),
        settings=describe_settings(stored.config),
        value=stored.value,
        stamp=stored.stamp,
        added_at=datetime.now().isoformat(timespec="seconds"),
        spec=stored.spec,
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


# ----------------------------------------------------------------------------- calculations from numbers typed


def clean_note(note: Any) -> str:
    """
    Where the numbers come from, as it is kept: plain text on one line, ``NOTE_MAX`` characters at most, and only
    what is seen. Control and format characters are dropped: a soft hyphen or a zero-width space inside a name
    (a source pasted from a PDF) hides the name from a scan of rare values while the page shows it whole, and a
    right-to-left override shows another text than the one kept.
    """
    seen = "".join(
        " " if c.isspace() else c
        for c in str(note or "")
        if c.isspace() or unicodedata.category(c) not in ("Cc", "Cf")
    )
    return " ".join(seen.split())[:NOTE_MAX]


def typed_id(spec: AnalysisSpec) -> str:
    """
    The name of a calculation in the dossier: ``typed:<kind>:`` then the start of its request id. The same numbers
    and options are one item, whatever their source (a source typed after adding replaces the one kept); another
    number is another. Of the session only: a package names a calculation by its kind and its place, never by a
    hash of its numbers (a small count is found back from one in seconds).
    """
    return f"typed:{spec.kind}:{spec.request_id[:16]}"


def typed_note(note_key: str | None) -> str:
    """The source as its field holds it now (``note_key``: the field's key), cleaned; "" without a field."""
    return clean_note(st.session_state.get(note_key)) if note_key else ""


def _typed() -> dict[str, dict[str, Any]]:
    basket = st.session_state.get(TYPED_KEY)
    if not isinstance(basket, dict):
        basket = {}
        st.session_state[TYPED_KEY] = basket
    return basket


def add_typed(spec: Mapping[str, Any], note_key: str | None = None) -> None:
    """
    Put a calculation from numbers typed in the study dossier, or write its source anew when the dossier holds its
    numbers already (callback of the two buttons): ``spec`` as JSON values. The source is read from its field when
    the button is clicked, never as it stood when the button was drawn: a callback runs with the arguments of the
    run before, so a source typed just before the click was lost, and a second click kept the calculation twice.
    """
    found, _ = read_spec(spec)
    if found is None or found.typed is None:
        return
    note, item_id = typed_note(note_key), typed_id(found)
    stored = _typed().get(item_id)
    if stored is not None:
        stored["note"] = note
        return
    try:
        words = typed_numbers(found)
    except ValueError:
        words = ""
    _typed()[item_id] = {
        "id": item_id,
        "spec": found.to_dict(),
        "note": note,
        # the numbers in words, as added: what an item this version can no longer read still shows
        "words": words,
        "added_at": datetime.now().isoformat(timespec="seconds"),
    }


def remove_typed(item_id: str) -> None:
    _typed().pop(item_id, None)


def typed_stored() -> list[dict[str, Any]]:
    """The calculations of the dossier as the session keeps them (JSON values), in the order they were added: what
    the Word dossier is built from, after the page run."""
    return [json.loads(json.dumps(stored)) for stored in _typed().values()]


def typed_provenance(spec: AnalysisSpec, *, note_key: str | None = None) -> None:
    """
    Under a calculator's result, after the line of its numbers: the button that adds the calculation to the study
    dossier; "In the study dossier (remove)" once it is there with the source the field holds; "Update the source
    in the study dossier" when it is there with another. ``note_key`` is the key of the field "Where the numbers
    come from": the callbacks read it when clicked. The buttons are named by the numbers alone, so each stays one
    widget while the source is typed.
    """
    item_id = typed_id(spec)
    stored = _typed().get(item_id)
    if stored is None:
        st.button(
            "Add to the study dossier",
            key=f"report_typed_add::{item_id}",
            icon=":material/note_add:",
            type="tertiary",
            on_click=add_typed,
            args=(spec.to_dict(), note_key),
            help="Adds this calculation, with the numbers typed and where they come from, to the Word study "
            "dossier (Reproduce > Study dossier), apart from the results of the dataset.",
        )
    elif str(stored.get("note") or "") != typed_note(note_key):
        kept = str(stored.get("note") or "")
        st.button(
            "Update the source in the study dossier",
            key=f"report_typed_update::{item_id}",
            icon=":material/edit_note:",
            type="tertiary",
            on_click=add_typed,
            args=(spec.to_dict(), note_key),
            help="The study dossier keeps these numbers "
            + (f'with another source ("{plain(kept)}")' if kept else "without a source")
            + ". This writes the one above in its place, or none if the field is empty; the numbers and the result "
            "do not change.",
        )
    else:
        st.button(
            "In the study dossier (remove)",
            key=f"report_typed_remove::{item_id}",
            icon=":material/check_circle:",
            type="tertiary",
            on_click=remove_typed,
            args=(item_id,),
            help="Reproduce > Study dossier lists the calculations from numbers typed, apart from the results of "
            "the dataset.",
        )


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


def stored_value(key: str) -> Any | None:
    """The result kept under ``key`` as it is, whatever its settings and its age; None without one. For what reads
    a result without showing it (the dossier compares a table typed with the table of the accuracy result the
    session holds)."""
    stored = _store().get(key)
    return stored.value if stored is not None else None


def stored_settings(key: str) -> tuple[str, str] | None:
    """
    The settings of the stored result ``key`` in words, and the time it was computed (hours and minutes); None
    without one. A page says so where a result computed with other settings is no longer shown: the result did not
    vanish, it belongs to other choices.
    """
    stored = _store().get(key)
    if stored is None:
        return None
    return describe_settings(stored.config), stored.stamp.computed_at[11:16]


__all__ = [
    "CURRENT",
    "MISSING",
    "NOTE_MAX",
    "REPORT_KEY",
    "RESULTS_KEY",
    "RESULT_TITLES",
    "STALE_CONFIG",
    "STALE_DATA",
    "STALE_POPULATION",
    "TYPED_KEY",
    "ReportItem",
    "ResultStamp",
    "StoredResult",
    "add_to_report",
    "add_typed",
    "clean_note",
    "config_hash",
    "describe_settings",
    "held_tables_kept",
    "provenance",
    "recall",
    "remember",
    "remove_from_report",
    "remove_typed",
    "report_items",
    "report_status",
    "single_imputation_warning",
    "stamp_of",
    "status",
    "stored_settings",
    "stored_value",
    "typed_id",
    "typed_note",
    "typed_provenance",
    "typed_stored",
]
