"""
The levels of a grouping variable, as the lists of groups show them (M5).

"Luxembourg" and "Luxembourg " are two groups, which a browser shows identically: the tutorial cohort's ``site``
seemed to hold each site twice. The lists mark the hidden spaces (``␣``), and a warning names the levels that differ
only by spaces or capitals, with the formula of a cleaned copy (Enrich > Create New Variables). Nothing is merged
silently: the analysis uses the values as they are.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import streamlit as st

# what a browser hides: spaces at either end, runs of spaces, and whitespace other than a plain space
_HIDDEN = re.compile(r"^\s+|\s+$|\s{2,}|[^\S ]")


def _visible(spaces: str) -> str:
    return "".join("␣" if ch == " " else repr(ch)[1:-1] for ch in spaces)


def show_level(value: Any) -> str:
    """
    A level as a list shows it: "Luxembourg " reads "Luxembourg␣". Two levels never read the same, since the
    multiselect tells options apart by these labels.
    """
    return _HIDDEN.sub(lambda m: _visible(m.group()), str(value))


def _key(value: Any) -> str:
    return " ".join(str(value).split()).casefold()


def alike_levels(values: Iterable[Any]) -> list[list[Any]]:
    """The sets of two or more levels that are the same text but for spaces or capitals, in the order given."""
    groups: dict[str, list[Any]] = defaultdict(list)
    for value in values:
        groups[_key(value)].append(value)
    return [found for found in groups.values() if len(found) > 1]


def cleaning_formula(column: Any, alike: list[list[Any]]) -> str:
    """The Enrich formula of a cleaned copy of ``column``: spaces trimmed, and lower case when capitals differ."""
    name = str(column) if str(column).isidentifier() else f'""{column}""'
    by_case = any(len({" ".join(str(v).split()) for v in found}) > 1 for found in alike)
    return f"{name}.str.strip()" + (".str.lower()" if by_case else "")


def level_warning(column: Any, values: Iterable[Any]) -> None:
    """Warns when levels of ``column`` differ only by spaces or capitals: each is analysed as a separate group."""
    alike = alike_levels(values)
    if not alike:
        return
    examples = "; ".join(", ".join(f"`{show_level(v)}`" for v in found) for found in alike[:3])
    more = f" and {len(alike) - 3} more" if len(alike) > 3 else ""
    st.warning(
        f"Some groups of **{column}** differ only by spaces (␣) or capitals: {examples}{more}. Each is analysed as a "
        f"separate group. To merge them, add a cleaned copy in Enrich > Create New Variables, for example "
        f"`{column}_clean` = `{cleaning_formula(column, alike)}`, and group by it.",
        icon=":material/spellcheck:",
    )


__all__ = ["alike_levels", "cleaning_formula", "level_warning", "show_level"]
