"""
Reusable Streamlit components (audit U2/U5/U6): one page header per page, consistent empty
states, status chips, confirmations for destructive actions, and long-operation feedback.
Only native Streamlit elements are used (no raw HTML), so light and dark themes both work.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import streamlit as st

from ui import nav
from ui.theme import APP_NAME, STATUS_COLORS

# ----------------------------------------------------------------------------- navigation helpers


def page_link(
    key: str,
    label: str | None = None,
    *,
    icon: str | None = None,
    width: str = "content",
    tab: str | None = None,
) -> None:
    """
    Link to a registered page, opened on one of its tabs when ``tab`` names one it declares (``nav.PageDef.tabs``);
    degrades to a caption outside ``st.navigation`` (tests, bare mode).
    """
    definition = nav.page(key)
    query = nav.tab_query(
        key, tab
    )  # an undeclared tab is refused, here rather than swallowed below
    try:
        st.page_link(
            nav.page_path(key),
            label=label or definition.title,
            icon=icon or definition.icon,
            width=width,
            query_params=query,
        )
    except Exception:
        st.caption(f"-> {label or definition.title}")


def switch_to(key: str, *, tab: str | None = None) -> None:
    """Open a registered page, on one of its tabs when ``tab`` names one it declares (``nav.PageDef.tabs``)."""
    query = nav.tab_query(key, tab)
    try:
        st.switch_page(nav.page_path(key), query_params=query)
    except Exception:
        st.rerun()


def linked_tabs(
    page: str, labels: Sequence[str], *, depth: int = 0, **kwargs: Any
) -> Sequence[Any]:
    """
    ``st.tabs`` opened on the tab a link names (``nav.PageDef.tabs``, by the query parameter ``nav.TAB_PARAM``): the
    label at ``depth`` of its path, the outer row being 0. A label the row does not hold (a tab renamed since the link
    was made) opens the row as usual. The parameter stays in the address while the page is open, so the tab does not
    change between reruns.
    """
    opened = nav.opened_tabs(page, st.query_params.get(nav.TAB_PARAM))
    label = opened[depth] if len(opened) > depth else None
    return st.tabs(list(labels), default=label if label in labels else None, **kwargs)


# ----------------------------------------------------------------------------- page structure


def _page_meta(title: str) -> nav.PageDef | None:
    return next((p for p in nav.PAGES if p.title == title), None)


def page_header(
    title: str,
    subtitle: str | None = None,
    *,
    chapter: str | None = None,
    eyebrow: str | None = None,
    icon: str | None = None,
) -> None:
    """
    The page hero banner: section name, the page's single H1, a one-line purpose and the handbook
    link. Section and icon come from the navigation registry unless given explicitly.
    """
    meta = _page_meta(title)
    section_name = eyebrow or (meta.section if meta and meta.section else APP_NAME)
    icon = icon or (meta.icon if meta else None)
    if chapter:
        st.session_state["handbook_chapter"] = chapter
    with st.container(key="cmp-hero"):
        left, right = st.columns([5, 1.2], vertical_alignment="bottom")
        with left:
            with st.container(key="cmp-hero-eyebrow"):
                st.markdown(f"{icon} {section_name}" if icon else section_name)
            st.title(title, anchor=False)
            if subtitle:
                st.caption(subtitle)
        if chapter:
            with right, st.container(horizontal=True, horizontal_alignment="right"):
                page_link("handbook", "Handbook", icon=":material/menu_book:")


def section(title: str, help_text: str | None = None) -> None:
    """A level-2 heading (pages keep one H1 and use sections below it)."""
    st.subheader(title, help=help_text)


# ----------------------------------------------------------------------------- status


def status_chip(label: str, status: str = "neutral") -> None:
    """``ok`` | ``caution`` | ``violated`` | ``info`` | ``neutral`` rendered as a coloured badge with an icon."""
    icons = {
        "ok": ":material/check_circle:",
        "caution": ":material/warning:",
        "violated": ":material/error:",
        "info": ":material/info:",
        "neutral": ":material/radio_button_unchecked:",
    }
    st.badge(
        label, icon=icons.get(status, icons["neutral"]), color=STATUS_COLORS.get(status, "grey")
    )


# ----------------------------------------------------------------------------- empty states


def empty_state(
    title: str,
    body: str,
    *,
    links: tuple[tuple[str, str], ...] = (),
    action: Callable[[], None] | None = None,
) -> None:
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
        if st.button(
            "Load the sample cohort",
            key=f"empty_sample_{st.session_state.get('_empty_state_n', 0)}",
        ):
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
    from utils.optional import extra_installed, missing_message

    if not extra_installed("ai"):
        empty_state("AI features not installed", missing_message(feature, "ai"))
        return
    empty_state(
        "AI engine not initialised",
        f"{feature} needs the AI engine. Choose a provider (local Ollama by default) and initialise it in Settings.",
        links=(("settings", "Open Settings"),),
    )


def no_results(message: str = "Run the analysis to see results here.") -> None:
    empty_state("No results yet", message)


# ----------------------------------------------------------------------------- confirmations and feedback


def confirm(
    title: str,
    body: str,
    on_confirm: Callable[[], Any],
    *,
    confirm_label: str = "Confirm",
    danger: bool = True,
) -> None:
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


def glossary_popover(*terms: str, label: str = "Glossary") -> None:
    """Definitions of the given terms (all terms when none are given)."""
    from ui.guidance.glossary import GLOSSARY, define

    with st.popover(label, icon=":material/dictionary:"):
        items = [(t, define(t)) for t in terms] if terms else list(GLOSSARY.items())
        for term, text in items:
            if text:
                st.markdown(f"**{term}** - {text}")


# ----------------------------------------------------------------------------- results


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}g}"


def result_card(
    result: Any, *, title: str | None = None, group_variable: str | None = None
) -> None:
    """
    Estimate -> CI -> p (raw/adjusted) -> n -> assumption checks -> method and reference -> 'How to read this'
    (audit S9/U4). ``group_variable`` names the groups in the checks ("Normality, diabetes = No").
    """
    from biostats.assumptions import read_assumptions
    from ui.guidance.interpretation import how_to_read

    with st.container(border=True):
        head, chip = st.columns([4, 1], vertical_alignment="center")
        head.markdown(f"**{title or result.name}**")
        with chip:
            sig = result.significant
            if sig is None:
                status_chip("not computed", "neutral")
            else:
                status_chip("p < alpha" if sig else "p >= alpha", "info" if sig else "neutral")
        cols = st.columns(3)
        if result.estimate is not None:
            cols[0].metric(
                result.estimate_label or "estimate",
                _fmt(result.estimate),
                help="The quantity of interest on the original scale, with its confidence interval below.",
            )
            if result.estimate_ci is not None:
                cols[0].caption(
                    f"95% CI {result.estimate_ci.as_text(3)} ({result.estimate_ci.method})"
                )
        elif result.primary_effect is not None:
            eff = result.primary_effect
            cols[0].metric(eff.name, _fmt(eff.estimate))
            if eff.ci is not None:
                cols[0].caption(f"95% CI {eff.ci.as_text(3)} ({eff.ci.method})")
        p_text = _fmt(result.p_value, 2)
        if result.p_adjusted is not None:
            p_text += f" (adjusted {_fmt(result.p_adjusted, 2)}, {result.adjustment or 'adjusted'})"
        stat = f"{result.statistic_label or 'statistic'} = {_fmt(result.statistic)}"
        if result.df:
            stat += ", df = " + ", ".join(f"{d:.4g}" for d in result.df)
        cols[1].metric("p-value", p_text, help=stat)
        n_text = ", ".join(
            f"{g}: {n}" for g, n in zip(result.groups, result.n_per_group, strict=False)
        ) or str(result.n_total)
        cols[2].metric("n", str(result.n_total), help=n_text)
        if result.reference_group:
            cols[2].caption(
                f"reference: {result.reference_group}; comparison: {result.comparison_group}"
            )
        if result.effect_sizes:
            parts = []
            for eff in result.effect_sizes:
                text = f"{eff.name} = {_fmt(eff.estimate)}"
                if eff.ci is not None:
                    text += f" (95% CI {eff.ci.as_text(2)})"
                if eff.magnitude():
                    text += f", {eff.magnitude()}"
                parts.append(text)
            st.caption("Effect sizes: " + "; ".join(parts))
        if result.mi is not None:
            info = result.mi
            st.caption(
                f":material/stacks: Pooled over {info.m} imputed datasets with Rubin's rules: df = {info.df:.4g}, "
                f"FMI = {info.fmi:.3f}, RIV = {info.riv:.3f}."
            )
        if result.assumptions:
            # each check with the group it concerns, its finding and what it means for this test (biostats)
            st.caption("**Assumption checks**: diagnostics, not gates; the test ran regardless.")
            for reading in read_assumptions(result, group_variable):
                chip, text = st.columns([2, 5], vertical_alignment="center")
                with chip:
                    status_chip(
                        reading.check, "neutral" if reading.status == "na" else reading.status
                    )
                text.caption(reading.sentence)
        if result.references:
            st.caption("Method: " + " | ".join(result.references))
        help_popover(how_to_read(result))


__all__ = [
    "confirm",
    "empty_state",
    "error_card",
    "glossary_popover",
    "help_popover",
    "linked_tabs",
    "long_op",
    "no_ai",
    "no_dataset",
    "no_results",
    "page_header",
    "page_link",
    "result_card",
    "section",
    "status_chip",
    "switch_to",
]
