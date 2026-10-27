"""
One control for choosing a connector (M5b, step 4): the AI provider, the vector database and the graph engine.

The choice follows the kit's order (``core.registry``): the session's choice, then the user's saved choice, then the
server's setting, then the family's own order. The control is seeded once, when the session has no value for it yet
(from ``seed``, the session's own state such as the engine that runs, else from that order), so that seeding never
fires ``on_change``. A change is kept for the next session (``ui.preferences.save_from``); "automatic" forgets the
saved choice. Only the connectors that can run here are offered.

With ``explain`` (Settings), the control also says which connector is used and why, why a saved choice or the
server's setting is not used, and which connectors cannot run here, with the group to install
(:func:`explain_choice`, which a card can call itself further down). The identity providers are chosen by the
deployment's setting only: they have no control.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any, Literal

import streamlit as st

from core.registry import AUTO, Availability, ConnectorRegistry, Selection
from ui import preferences
from ui.components.text import plain


def connector_selector(
    registry: ConnectorRegistry[Any],
    *,
    key: str,
    label: str,
    preference: str | None,
    preference_name: str,
    setting: str | None = None,
    seed: str | None = None,
    offer_auto: bool = True,
    auto: str | None = None,
    auto_reasons: Iterable[str] = (),
    check: Callable[[str], Availability] | None = None,
    widget: Literal["selectbox", "segmented"] = "selectbox",
    persist: Literal["page", "session"] | None = "session",
    help_text: str | None = None,
    explain: bool = True,
    details: Callable[[Selection], None] | None = None,
    container: Any = None,
) -> Selection:
    """
    The control of the family ``registry`` under ``key``; returns the selection in use, whose warnings are the saved
    choice and the server's setting set aside (until the user chooses again).

    ``preference`` is the user's saved choice, kept as the preference ``preference_name``; ``setting`` is the server's
    (``CMP_*``); ``auto`` and ``auto_reasons`` are a recommendation for this machine; ``check`` is the family's own rule
    of availability (the offline mode). ``seed`` is the session's own choice, used when the control has no value yet:
    the engine that runs, or the graph engine chosen on another page. ``offer_auto`` adds "automatic", the server's
    default, as the first option. ``details`` adds to the explanation (the machine a recommendation is made for).
    """
    if registry.setting_only:  # the deployment chooses: a session or a saved choice never does
        raise ValueError(f"the {registry.family} is chosen by {registry.setting} only")
    found: dict[str, Availability] = {}

    def checked(name: str) -> Availability:  # each connector is checked once per run
        if name not in found:
            found[name] = check(name) if check is not None else registry.availability(name)
        return found[name]

    def choose(session: str | None = None, saved: str | None = None) -> Selection:
        return registry.select(
            session=session,
            preference=saved,
            setting=setting,
            auto=auto,
            auto_reasons=auto_reasons,
            check=checked,
        )

    server = choose()  # what "automatic" gives
    fallback = choose(saved=preference)  # what a session that has chosen nothing gets
    options = ([AUTO] if offer_auto else []) + [
        name for name, state in fallback.alternatives if state.ok
    ]
    ss = st.session_state
    current = ss.get(key)
    if current not in options:  # the session's first run of it, or a choice no longer offered
        start = choose(session=seed, saved=preference) if seed else fallback
        current = AUTO if offer_auto and start.source in ("setting", "auto") else start.chosen
    # set before the widget on every run: a widget's identity includes its page, so the control of another page
    # (the graph engine's, on four) takes the value only when it is set here; a value set so never fires on_change
    ss[key] = current

    def shown(name: str) -> str:
        if name != AUTO:
            return registry.display_name(name)
        if server.chosen is None:
            return "Automatic"
        used = registry.display_name(server.chosen)
        return f"Server default: {used}" if server.source == "setting" else f"Automatic: {used}"

    target = container if container is not None else st
    common: dict[str, Any] = {
        "options": options,
        "format_func": shown,
        "key": key,
        "help": help_text,
        "on_change": preferences.save_from,
        "args": (key, preference_name),
        "kwargs": {"auto": AUTO} if offer_auto else None,
        "persist_state": persist,
    }
    if widget == "segmented":
        value = target.segmented_control(label, required=bool(options), **common)
    else:
        value = target.selectbox(label, **common)

    if value is None:  # nothing can run here
        selection = fallback
    elif value == AUTO:
        selection = server
    elif value == fallback.chosen:  # seeded from the saved or server's choice, or chosen and kept
        selection = fallback
    else:
        selection = choose(session=value, saved=preference)
    # the saved choice stays in the file until the user chooses again: said until then
    selection = replace(selection, warnings=fallback.warnings)
    if explain:
        explain_choice(registry, selection, details=details, container=container)
    return selection


def explain_choice(
    registry: ConnectorRegistry[Any],
    selection: Selection,
    *,
    details: Callable[[Selection], None] | None = None,
    container: Any = None,
) -> None:
    """
    The connector used and why, each choice set aside (the selection's warnings), and what cannot run here. Every
    signed-in user sees it: names and reasons are shown as written, never as Markdown.
    """
    target = container if container is not None else st
    used = plain(registry.display_name(selection.chosen)) if selection.chosen else None
    outcome = f"{used} is used" if used else f"no {registry.family} can run here"
    for warning in selection.warnings:  # whole sentences, names as code (core.registry)
        target.caption(f"{warning}: {outcome}.")
    if used:
        with target.expander(f"Why {used}"):
            for reason in selection.reasons:
                st.markdown(f"- {plain(reason)}")
            if details is not None:
                details(selection)
    missing = [(name, state) for name, state in selection.alternatives if not state.ok]
    if missing:
        with target.expander(f"Not available here ({len(missing)})"):
            for name, state in missing:
                line = f"**{plain(registry.display_name(name))}**: {plain(state.reason)}."
                if state.install_hint:
                    line += f" Install it on the server with `{state.install_hint}`."
                st.markdown(line)


__all__ = ["connector_selector", "explain_choice"]
