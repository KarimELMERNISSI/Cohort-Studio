"""
The levels of a grouping variable, as the lists of groups show them (M5).

"Luxembourg" and "Luxembourg " are two groups, which a browser shows identically: the tutorial cohort's ``site``
seemed to hold each site twice. The lists mark the hidden spaces (``␣``), and a warning names the levels that differ
only by spaces or capitals, and where they are cleaned: Data > Clean > Categories, into a copy beside the variable.
Nothing is merged silently: the analysis uses the values as they are.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import streamlit as st

from core.registry import as_code
from ui import nav

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


# where a variable's values written several ways are cleaned
CLEAN_WHERE = nav.where("clean", "categories")


def level_warning(
    column: Any, values: Iterable[Any], *, source: str = "", copy: str = "", follows: bool = True
) -> bool:
    """
    Warns when levels of ``column`` differ only by spaces or capitals: each is analysed as a separate group, until
    the variable is cleaned on Data > Clean (a copy beside it, which the analysis can then group by). ``copy``: the copy
    that cleans ``source`` (``column`` itself, or the variable whose copy ``column`` is), when one was written;
    ``follows``: it was written from the values ``source`` holds now. Whether it warned: the analysis pages then offer
    to clean it there, or to group by the copy (app_pages.clean_dialog.group_warning).
    """
    alike = alike_levels(values)
    if not alike:
        return False
    # as code: a value is the data's text, and a backtick in it would end a code span written by hand
    examples = "; ".join(", ".join(as_code(show_level(v)) for v in found) for found in alike[:3])
    more = f" and {len(alike) - 3} more" if len(alike) > 3 else ""
    name, cleaned, original = as_code(column), as_code(copy), as_code(source or column)
    if copy and str(column) == copy:  # grouped by the copy: its cleaning kept these apart
        then = (
            f"{name} is the clean copy of {original}, and its cleaning kept these apart: to merge them, clean "
            f"{original} again on {CLEAN_WHERE}."
        )
    elif copy and follows:
        then = f"{name} is cleaned into {cleaned}: group by {cleaned}."
    elif copy:  # the copy misses a later change of the variable (an imputation, a merge's rows)
        then = (
            f"{cleaned} cleans {name} as it was before a later step changed it: clean {name} again on "
            f"{CLEAN_WHERE}, then group by the copy."
        )
    else:
        then = (
            f"To merge them, clean {name} on {CLEAN_WHERE}: its clean values go into a copy beside it, "
            f"{as_code(f'{column}_clean')} unless you name it otherwise, and you then group by the copy."
        )
    st.warning(
        f"Some groups of {name} differ only by spaces (␣) or capitals: {examples}{more}. Each is analysed as a "
        f"separate group. {then}",
        icon=":material/spellcheck:",
    )
    return True


__all__ = ["CLEAN_WHERE", "alike_levels", "level_warning", "show_level"]
