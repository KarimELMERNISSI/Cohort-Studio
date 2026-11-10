"""
The study definition strip (M8, the study package): the study the session's dataset belongs to, whether its definition
is saved as a revision, and what can be done about it. It is the same component on each page that shows the study
definition (the Study dossier now, Versions & traces with the package's screens), and every page reads one
:class:`StudyStatus` (:func:`study_status`), so that two pages never tell two stories about one study.

A revision keeps the criteria with their descriptions, which of them are ticked, the follow-up and the declared derived
variables with their notes, as written when it is saved; it is never changed afterwards (``manage.protocol_store``),
holds no data and no result, and can be deleted. A corrected description or note is saved as the next revision, which
differs in its labels only. The strip names a revision by its number, never by its fingerprint.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import streamlit as st

from core.derived import CONFIG_KEY as DERIVED_KEY
from core.population import catalogue
from core.protocol import body_checksum, hashed_view, protocol_hash
from core.registry import as_code
from ui.components import confirm, page_link, status_chip
from ui.state import get_state

SAID_KEY = (
    "study_strip_said"  # {strip key: (kind, text)}: a strip's message, shown once (ui/keys.py)
)

StudyState = Literal["none", "saved", "labels", "changed", "not_loaded", "unreadable"]
# the sections of a definition, as a sentence names them
_PARTS = {
    "mask_families": "the criteria",
    "population": "which criteria are ticked",
    "follow_up": "the follow-up",
    DERIVED_KEY: "the derived variables",
}
TITLE_HELP = (
    "What the study decides apart from the data: its inclusion and exclusion criteria, which of them are ticked, the "
    "follow-up and the derived variables' formulas. A revision keeps it as it is when saved, and is never changed "
    "afterwards; it holds no data and no result. The study is named after the dataset: every version saved under this "
    "name belongs to it, and a file imported or saved under another name starts another study."
)
SAVE_HELP = (
    "Keeps the study definition as it is now: the criteria with their descriptions, which of them are ticked, the "
    "follow-up and the derived variables with their notes, as written (check that none names a person). It keeps "
    "neither the data (the sidebar's Save keeps a version of the data) nor the results. A later change, a corrected "
    "description included, is saved as the next revision."
)


@dataclass(frozen=True)
class StudyStatus:
    """The study definition as every page shows it: the study, its latest readable revision, the definition the
    session holds, how the two compare, what differs, the number the next revision takes, and the revisions that
    cannot be read."""

    study_id: str
    latest: Any  # core.protocol.StudyProtocol | None
    body: Mapping[str, Any]
    state: StudyState
    changed: tuple[str, ...] = ()
    next_number: int = 1
    unreadable: tuple[int, ...] = ()


def _holds_nothing(body: Mapping[str, Any]) -> bool:
    return not any(body.get(key) for key in ("mask_families", "follow_up", DERIVED_KEY))


def _changed_parts(old: Mapping[str, Any], new: Mapping[str, Any]) -> tuple[str, ...]:
    """The sections in which two definitions select differently (their labels left out)."""
    before, after = hashed_view(old), hashed_view(new)
    return tuple(words for key, words in _PARTS.items() if before.get(key) != after.get(key))


def study_status() -> StudyStatus | None:
    """The session's study definition against its study's latest revision; None without a dataset or a signed-in
    user (the revisions live in a user's workspace)."""
    from manage.protocol_store import ProtocolFileError

    state = get_state()
    study = state.study_id()
    if study is None:
        return None
    body = state.protocol_body()
    try:
        store = state.services.protocols
        latest = state.latest_revision()
        number = store.next_number(study)
        unreadable = tuple(store.unreadable(study))
    except ValueError:
        return None
    except (
        ProtocolFileError,
        OSError,
    ):  # a folder locked by another program: the rest of the page is drawn
        return StudyStatus(study, None, body, "unreadable")
    if latest is None:
        kind: StudyState = "none"
    elif latest.checksum == body_checksum(body):
        kind = "saved"
    elif _holds_nothing(body) and not _holds_nothing(latest.body):
        kind = "not_loaded"
    elif latest.hash == protocol_hash(body):
        kind = "labels"
    else:
        kind = "changed"
    changed = _changed_parts(latest.body, body) if kind == "changed" else ()
    return StudyStatus(study, latest, body, kind, changed, number, unreadable)


def definition_summary(body: Mapping[str, Any]) -> str:
    """What a study definition holds, counted as the sidebar counts (ticked first): "4 of 12 criteria ticked", its
    derived variables, whether a follow-up is defined. Nothing of the data, and no rule's text."""
    criteria = catalogue(body)
    population = body.get("population")
    ticked = set(population.get("ticked") or ()) if isinstance(population, Mapping) else set()
    on, total = len({c.id for c in criteria} & ticked), len(criteria)
    if total == 0:
        parts = ["no criteria yet"]
    elif total == 1:
        parts = ["1 criterion, ticked" if on else "1 criterion, not ticked"]
    elif on == total:
        parts = [f"all {total} criteria ticked"]
    elif on == 0:
        parts = [f"none of the {total} criteria ticked"]
    else:
        parts = [f"{on} of {total} criteria ticked"]
    derived = len(body.get(DERIVED_KEY) or ())
    if derived:
        parts.append(f"{derived} derived variable{'s' if derived != 1 else ''}")
    if body.get("follow_up"):
        parts.append("a follow-up defined")
    return " · ".join(parts)


def status_text(status: StudyStatus | None) -> str:
    """The study definition in one line, for a document (the Word dossier): the study and its revision."""
    if status is None:
        return "n/a"
    latest = status.latest.meta.revision if status.latest is not None else None
    said = {
        "none": "no revision saved",
        "saved": f"revision {latest}",
        "labels": f"revision {latest}, its descriptions or notes changed since (not saved)",
        "changed": f"changed since revision {latest} (not saved)",
        "not_loaded": f"revision {latest} saved, not loaded in this session",
        "unreadable": "its revisions cannot be read",
    }[status.state]
    return f"{status.study_id}, {said}"


def _when(saved_at: str) -> str:
    try:
        moment = datetime.fromisoformat(saved_at)
    except ValueError:
        return ""
    return f"{moment.day} {moment:%b %Y}, {moment:%H:%M}"


def _say(key: str, kind: str, text: str) -> None:
    said = st.session_state.get(SAID_KEY)
    said = dict(said) if isinstance(said, dict) else {}
    said[key] = (kind, text)
    st.session_state[SAID_KEY] = said


def _said(key: str) -> tuple[str, str] | None:
    said = st.session_state.get(SAID_KEY)
    found = said.pop(key, None) if isinstance(said, dict) else None
    ok = isinstance(found, tuple) and len(found) == 2 and found[0] in ("ok", "info", "error")
    return found if ok else None


def _save(key: str) -> None:
    """Save revision's callback: the definition as it is now becomes the next revision, and the strip says so."""
    from manage.protocol_store import ProtocolFileError, ProtocolUnchanged

    try:
        saved = get_state().save_protocol_revision()
    except ProtocolUnchanged as exc:
        _say(key, "info", f"Not saved: {exc}.")
    except ProtocolFileError as exc:  # its message names no path
        _say(
            key,
            "error",
            f"Not saved: {exc}. Nothing was changed. Try again; if it fails again, ask your administrator to check "
            "the workspace folder.",
        )
    except ValueError as exc:  # no dataset, or a study whose folder cannot be used: no path either
        _say(key, "error", f"Not saved: {exc}.")
    else:
        _say(
            key,
            "ok",
            f"Revision {saved.meta.revision} saved in your workspace: the study definition above. The data and the "
            "results are not part of it.",
        )


def _load(key: str, revision: Any) -> None:
    """Load revision's callback: the revision's definition becomes the session's (criteria are filters: no row
    changes)."""
    get_state().load_protocol_revision(revision)
    _say(
        key,
        "ok",
        f"Revision {revision.meta.revision} loaded: its criteria, which of them are ticked, the follow-up and the "
        "derived variables are this session's again. No row changed.",
    )


def _delete(key: str, number: int) -> None:
    """Delete's confirmed action: the revision's file goes; the session's definition stays."""
    from manage.protocol_store import ProtocolFileError

    try:
        deleted = get_state().delete_protocol_revision(number)
    except (ProtocolFileError, ValueError) as exc:
        _say(key, "error", f"Revision {number} was not deleted: {exc}.")
        return
    _say(
        key,
        "ok" if deleted else "info",
        f"Revision {number} deleted from your workspace, with the descriptions and notes it held; its number is never "
        "given again. The session's definition is unchanged."
        if deleted
        else f"Revision {number} is no longer in your workspace.",
    )


def study_strip(key: str) -> StudyStatus | None:
    """
    The strip: the study and its state, what differs, and the actions that fit. Save revision n when the definition
    differs from the latest revision (its labels included); Load revision n when the session holds no definition and
    the latest revision does (a new session, a saved version loaded again); Delete revision n otherwise. Returns the
    status it drew, for the page's other parts.
    """
    status = study_status()
    if status is None:
        return None
    latest = status.latest
    number = latest.meta.revision if latest is not None else None
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown(
                f":material/assignment: **Study definition** · {as_code(status.study_id)}",
                width="content",
                help=TITLE_HELP,
            )
            if status.state == "unreadable":
                status_chip("revisions cannot be read", "caution")
            elif latest is None:
                status_chip("no revision saved", "neutral")
            elif status.state == "saved":
                when = _when(latest.meta.saved_at)
                status_chip(f"revision {number}" + (f", saved {when}" if when else ""), "ok")
            elif status.state == "not_loaded":
                status_chip(f"revision {number} not loaded", "info")
            elif status.state == "labels":
                status_chip(f"labels changed since revision {number}", "caution")
            else:
                status_chip(f"changed since revision {number}", "caution")
        if status.state == "unreadable":
            st.caption(
                "The study definition's revisions cannot be read now (the folder is in use or locked). The rest of "
                "the page is not affected."
            )
            return status
        if status.state == "not_loaded":
            st.caption(
                f"This session holds no criteria: revision {number} holds "
                f"{definition_summary(latest.body)}."
            )
        else:
            st.caption(definition_summary(status.body))
        if status.state == "changed" and status.changed:
            st.caption(f"Changed since revision {number}: {' and '.join(status.changed)}.")
        elif status.state == "labels":
            st.caption(
                f"Changed since revision {number}: descriptions or notes only (the definition selects the same "
                "rows). Save them as the next revision, and delete the earlier one if it named a person."
            )
        if status.unreadable:
            shown = ", ".join(str(n) for n in status.unreadable)
            st.caption(
                f":material/warning: Revision{'s' if len(status.unreadable) > 1 else ''} {shown} cannot be read: "
                "changed outside CohortStudio. "
                + (f"The next revision follows revision {number}." if number else "")
            )
        if status.state == "none" and _holds_nothing(status.body):
            st.caption(
                "No criteria, derived variables or follow-up yet: the analyses use every row. Define criteria on "
                "Data > Validate."
            )
            page_link("validate", "Define criteria", icon=":material/rule:")
        said = _said(key)
        if said:
            {"ok": st.success, "info": st.info, "error": st.error}[said[0]](said[1])
        with st.container(horizontal=True, gap="small"):
            if status.state == "not_loaded":
                st.button(
                    f"Load revision {number}",
                    key=f"{key}_load",
                    type="primary",
                    icon=":material/download:",
                    on_click=_load,
                    args=(key, latest),
                    help="Makes the revision's criteria, which of them are ticked, the follow-up and the derived "
                    "variables this session's. Criteria are filters: no row changes.",
                )
            elif status.state in ("none", "changed", "labels"):
                st.button(
                    f"Save revision {status.next_number}",
                    key=f"{key}_save",
                    icon=":material/bookmark_add:",
                    on_click=_save,
                    args=(key,),
                    help=SAVE_HELP,
                )
            deletable = latest is not None and status.state != "not_loaded"
            if deletable and st.button(
                f"Delete revision {number}",
                key=f"{key}_delete",
                type="tertiary",
                icon=":material/delete:",
                help="Removes the revision from your workspace, with the descriptions and notes it holds. Its "
                "number is never given again, and the session's definition is unchanged.",
            ):
                confirm(
                    f"Delete revision {number}?",
                    f"Revision {number} of {status.study_id} is removed from your workspace, with the descriptions "
                    "and notes it holds; its number is never given again. The session's criteria are not changed. "
                    "A description the trace recorded for a ticked criterion stays in the trace.",
                    lambda: _delete(key, number),
                    confirm_label=f"Delete revision {number}",
                )
    return status


__all__ = [
    "SAID_KEY",
    "StudyStatus",
    "definition_summary",
    "status_text",
    "study_status",
    "study_strip",
]
