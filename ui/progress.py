"""
Session side of the workflow progress (see :mod:`core.progress`).

Pages call :func:`mark_step` when they perform a step on the current data; the Start page calls
:func:`current_progress` to tick boxes and pick the suggested next step. Marks are tied to the
trace session, so loading another dataset (a new trace) starts the workflow afresh.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st

from core.progress import MARKABLE, Progress, StepMark, data_revision, evaluate
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
    )
    st.session_state[_KEY] = {"session_id": session_id, "marks": marks}


def current_trace_session() -> str | None:
    return _trace()[0]


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


def knowledge_evidence() -> dict[str, tuple[bool, str]]:
    """What the user's workspace holds for the two knowledge paths (documents, variable map)."""
    scope = get_current_scope()
    if scope is None:
        return {
            "docs": (False, "Sign in to keep a document library."),
            "varmap": (False, "Sign in to keep variable maps."),
        }
    n_pdfs = _count(scope.documents_dir(), "*.pdf")
    n_graphs = _count(scope.knowledge_graphs_dir(), "*.json")
    indexed = _non_empty(scope.chroma_dir())
    n_maps = _count(scope.taxonomy_dir(), "v*/taxonomy_metadata.json")

    pdfs = f"{n_pdfs} PDF{'s' if n_pdfs != 1 else ''} in your library"
    if n_graphs:
        docs = (True, f"{pdfs}, {n_graphs} knowledge graph{'s' if n_graphs != 1 else ''} saved.")
    elif indexed and n_pdfs:
        docs = (True, f"{pdfs}, indexed for chat and search.")
    elif n_pdfs:
        docs = (False, f"{pdfs}, not indexed yet.")
    else:
        docs = (False, "No document in your library yet.")
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
    )


__all__ = ["current_progress", "current_trace_session", "knowledge_evidence", "mark_step"]
