"""
Session side of the workflow progress (see :mod:`core.progress`).

Pages call :func:`mark_step` when they perform a step on the current data; the Start page calls
:func:`current_progress` to tick boxes and pick the suggested next step. Marks are tied to the
trace session, so loading another dataset (a new trace) starts the workflow afresh.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import streamlit as st

from core.progress import MARKABLE, Progress, StepMark, data_revision, evaluate, revision_labels
from utils.session_scope import get_current_scope

_KEY = "workflow_marks"


def _trace() -> tuple[str | None, list[dict]]:
    tm = st.session_state.get("transformation_manager")
    if tm is None:
        return None, []
    return getattr(tm, "session_id", None), list(getattr(tm, "history", []) or [])


def _marks(session_id: str | None) -> dict[str, StepMark]:
    store = st.session_state.get(_KEY)
    if not isinstance(store, dict) or store.get("session_id") != session_id:
        return {}
    return dict(store.get("marks") or {})


def mark_step(step: str, detail: str = "") -> None:
    """Record that ``step`` (validate, analyse or report) was just performed on the current data."""
    if step not in MARKABLE or st.session_state.get("data") is None:
        return
    session_id, steps = _trace()
    marks = _marks(session_id)
    marks[step] = StepMark(
        revision=data_revision(steps),
        at=datetime.now().isoformat(timespec="seconds"),
        detail=detail,
        population_fp=_population_fp(),
    )
    st.session_state[_KEY] = {"session_id": session_id, "marks": marks}


def _population_fp() -> str:
    """Fingerprint of the analysis population (rows analysed), empty when it cannot be evaluated."""
    from ui.state import get_state

    try:
        identity = get_state().identity()
    except Exception:
        return ""
    return identity.population_fp if identity is not None else ""


def current_trace_session() -> str | None:
    return _trace()[0]


_QUALITY_KEY = "quality_history"


def record_quality_check(metrics: Mapping[str, float | None]) -> list[dict]:
    """
    Keep the audit metrics of the current data, one entry per revision (the latest check of a revision wins).

    Returns the checks of this dataset in order: ``{"revision", "label", "metrics", "at"}``. Storing the
    metrics, not the global score, lets the history follow the weights the user sets now.
    """
    session_id, steps = _trace()
    revision = data_revision(steps)
    labels = revision_labels(steps)
    store = st.session_state.get(_QUALITY_KEY)
    if not isinstance(store, dict) or store.get("session_id") != session_id:
        store = {"session_id": session_id, "checks": []}
    checks: list[dict] = list(store["checks"])
    entry = {
        "revision": revision,
        "label": labels[revision] if revision < len(labels) else f"revision {revision}",
        "metrics": {k: v for k, v in metrics.items() if v is not None},
        "at": datetime.now().isoformat(timespec="seconds"),
    }
    if checks and checks[-1]["revision"] == revision:
        checks[-1] = entry
    else:
        checks.append(entry)
    st.session_state[_QUALITY_KEY] = {"session_id": session_id, "checks": checks}
    return checks


def _count(directory: Path | None, pattern: str) -> int:
    try:
        return sum(1 for _ in directory.glob(pattern)) if directory and directory.exists() else 0
    except OSError:
        return 0


def _non_empty(directory: Path | None) -> bool:
    try:
        return bool(directory and directory.exists() and any(directory.iterdir()))
    except OSError:
        return False


def _pdfs(directory: Path | None) -> list[str]:
    try:
        if not directory or not directory.exists():
            return []
        return sorted(p.name for p in directory.iterdir() if p.suffix.lower() == ".pdf")
    except OSError:
        return []


def _indexed(vectors_dir: Path | None) -> set[str]:
    """The documents the vector indexes of the workspace hold, from their manifests (LanceDB or Qdrant)."""
    found: set[str] = set()
    try:
        manifests = (
            list(vectors_dir.glob("*/manifest.json"))
            if vectors_dir and vectors_dir.exists()
            else []
        )
    except OSError:
        return found
    for path in manifests:
        try:
            found.update(
                str(d) for d in json.loads(path.read_text(encoding="utf-8")).get("documents") or []
            )
        except OSError, ValueError, AttributeError:
            continue
    return found


def _names(names: list[str]) -> str:
    if len(names) <= 2:
        return " and ".join(names)
    if len(names) == 3:
        return f"{names[0]}, {names[1]} and {names[2]}"
    return f"{', '.join(names[:3])} and {len(names) - 3} more"


def knowledge_evidence(ai_on: bool | None = None) -> dict[str, tuple[bool, str]]:
    """
    What the user's workspace holds for the two knowledge paths (documents, variable map).

    The documents path is done when the AI engine is on (``ai_on``; None: read from the session) and the library
    holds indexed documents: without the engine, the documents serve nothing.
    """
    scope = get_current_scope()
    if scope is None:
        return {
            "docs": (False, "Sign in to keep a document library."),
            "varmap": (False, "Sign in to keep variable maps."),
        }
    if ai_on is None:
        ai_on = bool(getattr(st.session_state.get("rag_manager"), "initialized", False))
    pdfs = _pdfs(scope.documents_dir())
    indexed = sorted(set(pdfs) & _indexed(scope.vectors_dir()))  # indexed and still in the library
    n_graphs = _count(scope.knowledge_graphs_dir(), "*.json")
    n_maps = _count(scope.taxonomy_dir(), "v*/taxonomy_metadata.json")

    if not pdfs:
        docs = (False, "No document in your library yet.")
    elif not indexed:
        n = len(pdfs)
        docs = (
            False,
            f"{n} PDF{'s' if n != 1 else ''} in your library, not indexed yet: initialise the AI engine to index "
            f"{'it' if n == 1 else 'them'}.",
        )
    elif not ai_on:
        docs = (
            False,
            f"{_names(indexed)} indexed in your library: initialise the AI engine to use "
            f"{'it' if len(indexed) == 1 else 'them'}.",
        )
    else:
        graphs = (
            f"; {n_graphs} knowledge graph{'s' if n_graphs != 1 else ''} saved" if n_graphs else ""
        )
        docs = (True, f"{_names(indexed)} indexed for chat and search{graphs}.")
    if n_maps:
        varmap = (True, f"{n_maps} variable map version{'s' if n_maps != 1 else ''} saved.")
    else:
        varmap = (False, "No variable map saved yet.")
    return {"docs": docs, "varmap": varmap}


def current_progress() -> Progress:
    """Status of every workflow step for the session's current data."""
    session_id, steps = _trace()
    df = st.session_state.get("data")
    ref = st.session_state.get("dataset_ref")
    label = ""
    if df is not None:
        name = (
            getattr(ref, "label", None) or st.session_state.get("current_dataset_name") or "dataset"
        )
        label = f"{name} ({len(df):,} rows, {df.shape[1]} columns)"
    return evaluate(
        steps,
        _marks(session_id),
        has_data=df is not None,
        dataset_label=label,
        knowledge=knowledge_evidence(),
        population_fp=_population_fp() if df is not None else "",
    )


__all__ = [
    "current_progress",
    "current_trace_session",
    "knowledge_evidence",
    "mark_step",
    "record_quality_check",
]
