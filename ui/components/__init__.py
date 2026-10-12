"""
Reusable Streamlit components (audit U2/U5/U6): one page header per page, consistent empty
states, status chips, confirmations for destructive actions, and long-operation feedback.
Only native Streamlit elements are used (no raw HTML), so light and dark themes both work.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import streamlit as st

from ui import nav
from ui.theme import STATUS_COLORS


# ----------------------------------------------------------------------------- navigation helpers

def page_link(key: str, label: str | None = None, *, icon: str | None = None, width: str = "content") -> None:
    """Link to a registered page; degrades to a caption outside ``st.navigation`` (tests, bare mode)."""
    definition = nav.page(key)
    try:
        st.page_link(nav.page_path(key), label=label or definition.title, icon=icon or definition.icon, width=width)
    except Exception:
        st.caption(f"-> {label or definition.title}")


def switch_to(key: str) -> None:
    try:
        st.switch_page(nav.page_path(key))
    except Exception:
        st.rerun()


# ----------------------------------------------------------------------------- page structure

def page_header(title: str, subtitle: str | None = None, *, chapter: str | None = None) -> None:
    """The page's single H1, an optional one-line purpose, and the handbook link for this page."""
    left, right = st.columns([5, 1], vertical_alignment="center")
    with left:
        st.title(title)
        if subtitle:
            st.caption(subtitle)
    if chapter:
        st.session_state["handbook_chapter"] = chapter
        with right:
            page_link("handbook", "Handbook", icon=":material/menu_book:")


def section(title: str, help_text: str | None = None) -> None:
    """A level-2 heading (pages keep one H1 and use sections below it)."""
    st.subheader(title, help=help_text)


# ----------------------------------------------------------------------------- status

def status_chip(label: str, status: str = "neutral") -> None:
    """``ok`` | ``caution`` | ``violated`` | ``info`` | ``neutral`` rendered as a coloured badge with an icon."""
    icons = {"ok": ":material/check_circle:", "caution": ":material/warning:", "violated": ":material/error:",
             "info": ":material/info:", "neutral": ":material/radio_button_unchecked:"}
    st.badge(label, icon=icons.get(status, icons["neutral"]), color=STATUS_COLORS.get(status, "grey"))


# ----------------------------------------------------------------------------- empty states

def empty_state(title: str, body: str, *, links: tuple[tuple[str, str], ...] = (), action: Callable[[], None] | None = None) -> None:
    """A bordered placeholder with a clear next step (never a bare warning)."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(body)
        if links or action:
            cols = st.columns(max(1, len(links) + (1 if action else 0)))
            for col, (key, label) in zip(cols, links, strict=False):
                with col:
                    page_link(key, label)
            if action:
                with cols[-1]:
                    action()


def no_dataset() -> None:
    """Shown by every page that needs a dataset while none is loaded."""
    def _sample_button() -> None:
        if st.button("Load the sample cohort", key=f"empty_sample_{st.session_state.get('_empty_state_n', 0)}"):
            from ui.state import get_state

            get_state().load_sample_cohort()
            st.rerun()

    st.session_state["_empty_state_n"] = st.session_state.get("_empty_state_n", 0) + 1
    empty_state(
        "No dataset loaded",
        "Import a file (CSV, Excel or Parquet), reopen a saved version, or try the bundled sample cohort.",
        links=(("import", "Import a dataset"), ("start", "Saved versions")),
        action=_sample_button,
    )


def no_ai(feature: str = "This feature") -> None:
    empty_state(
        "AI engine not initialised",
        f"{feature} needs the AI engine. Choose a provider (local Ollama by default) and initialise it in Settings.",
        links=(("settings", "Open Settings"),),
    )


def no_results(message: str = "Run the analysis to see results here.") -> None:
    empty_state("No results yet", message)


# ----------------------------------------------------------------------------- confirmations and feedback

def confirm(title: str, body: str, on_confirm: Callable[[], Any], *, confirm_label: str = "Confirm", danger: bool = True) -> None:
    """Open a modal asking for confirmation before running ``on_confirm`` (destructive actions, audit U5)."""

    @st.dialog(title)
    def _dialog() -> None:
        st.write(body)
        col_ok, col_cancel = st.columns(2)
        with col_ok:
            if st.button(confirm_label, type="primary" if danger else "secondary", width="stretch"):
                on_confirm()
                st.rerun()
        with col_cancel:
            if st.button("Cancel", width="stretch"):
                st.rerun()

    _dialog()


@contextmanager
def long_op(label: str, *, expanded: bool = False) -> Iterator[Any]:
    """``st.status`` wrapper for anything that takes more than a couple of seconds."""
    status = st.status(label, expanded=expanded)
    try:
        yield status
    except Exception as exc:
        status.update(label=f"{label} - failed", state="error")
        status.error(str(exc))
        raise
    else:
        status.update(label=f"{label} - done", state="complete")


def error_card(user_message: str, exc: BaseException | None = None) -> None:
    """A readable error with the technical detail folded away."""
    st.error(user_message)
    if exc is not None:
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")


def help_popover(text: str, label: str = "How to read this") -> None:
    with st.popover(label, icon=":material/help:"):
        st.markdown(text)


__all__ = [
    "confirm", "empty_state", "error_card", "help_popover", "long_op", "no_ai", "no_dataset", "no_results",
    "page_header", "page_link", "section", "status_chip", "switch_to",
]
