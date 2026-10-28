"""
One box per rule of a family (M5b step 6): its name and its kind, and, opened, what makes it fire in the words of the
participant flow and the dossier (core.criteria.describe_definition). Validate's lists of criteria and the quality
dashboard's show the same boxes, so that every screen reads a rule the same way.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import streamlit as st

from core.criteria import describe_definition, rule_kind
from ui.components.text import plain


def criterion_cards(
    members: Mapping[str, Any], *, per_row: int = 3, expanded: bool = False
) -> None:
    """
    A box for each rule of ``members``, a family's rules (its "operator" is not a rule), closed unless ``expanded``, so
    that a long list stays short: its title gives the rule's name and its kind; opened, it gives what makes the rule
    fire and the user's description. A rule whose kind cannot be read says why. Every text is shown as written
    (plain): a name or a description is the user's text, never Markdown.
    """
    rules = [(str(name), spec) for name, spec in members.items() if name != "operator"]
    if not rules:
        st.caption("No criterion yet.")
        return
    grid = st.columns(per_row)
    for i, (name, definition) in enumerate(rules):
        title = plain(name, limit=200)
        try:
            label = rule_kind(name, definition).label
        # a rule loaded before the checks, or written by hand into the session
        except Exception as exc:
            with grid[i % per_row], st.expander(f"{title} · cannot be read", expanded=expanded):
                st.caption(plain(f"Cannot be read: {exc}", limit=500))
            continue
        with grid[i % per_row], st.expander(f"{title} · {label}", expanded=expanded):
            st.code(describe_definition(name, definition), language=None, wrap_lines=True)
            description = definition.get("description")
            if description:
                st.caption(plain(description, limit=500))


__all__ = ["criterion_cards"]
