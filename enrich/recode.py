"""
The values of a categorical variable as written, recoded (Data > Clean > Categories): each value mapped to a clean value
or to missing, written into a new column beside the original, which is kept. One traced step (``recode_levels``) that
the replay redoes from the recorded mapping; a value the mapping does not list stays as written, and the replay says
how many.

Suggestions pre-fill only what cannot change a meaning: values that differ by spaces or capitals take the spelling the
most rows of their group hold. The rest are offers, on those spellings, never applied: the rarer of a one-letter code
and the only longer value it begins ("Female" and "F"), a rare near spelling of a frequent value (a one-row
"Luxembourgg" beside 397 "Luxembourg"), and the codes that usually mean missing ("unknown", "?"). One edit apart can be
two categories (HFmrEF and HFrEF, NYHA II and III, type 1 and 2): a near spelling is never offered across digits,
numerals or signs, nor between two frequent values. Nothing here reads an outcome: the values and their rows only.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

# why a value is given its clean value
SPACES, CAPITALS, SPACES_AND_CAPITALS, FIRST_LETTER, SPELLING, MISSING_CODE = (
    "spaces",
    "capitals",
    "spaces and capitals",
    "first letter",
    "spelling",
    "missing code",
)
SIMILAR = 0.85  # difflib's ratio from which two spellings are near
RARE = 0.05  # a near spelling is offered only when it holds at most this share of the rows of the one it could join
# what a value meaning "unknown" is usually written as (folded: spaces, capitals, accents); "M" and "N" never guessed
_MISSING_CODES = frozenset(
    {
        "?",
        "??",
        "-",
        "--",
        ".",
        "na",
        "n/a",
        "n.a.",
        "nan",
        "none",
        "null",
        "missing",
        "unknown",
        "unk",
        "not known",
        "nk",
        "not available",
        "inconnu",
        "non renseigne",
        "nr",
    }
)
_ROMAN = re.compile(r"\b[ivxlcdm]+\b")
_SIGNS = re.compile(r"[+\-<>=]")


@dataclass(frozen=True)
class Offer:
    """A spelling that could join another, or become missing (``clean`` None): offered, never applied."""

    value: str  # a spelling, after spaces and capitals
    clean: str | None
    why: str  # FIRST_LETTER, SPELLING or MISSING_CODE


@dataclass(frozen=True)
class Suggestions:
    spellings: dict[
        str, str
    ]  # a value as written -> its group's spelling, where they differ: pre-filled
    reasons: dict[str, str]  # the same values -> SPACES, CAPITALS or SPACES_AND_CAPITALS
    offers: tuple[Offer, ...]  # the most frequent first


def _spaced(value: str) -> str:
    return " ".join(value.split())


def _folded(value: str) -> str:
    return _spaced(value).casefold()


def _plain(value: str) -> str:
    return unicodedata.normalize("NFKD", _folded(value)).encode("ascii", "ignore").decode()


def _why(value: str, spelling: str) -> str:
    spaces = _spaced(value) != value
    capitals = _spaced(value) != spelling
    return SPACES_AND_CAPITALS if spaces and capitals else SPACES if spaces else CAPITALS


def level_counts(values: pd.Series) -> dict[str, int]:
    """The values as written and their rows, the most frequent first (ties by value); missing values left out."""
    counts = values.dropna().astype(str).value_counts()
    return dict(
        sorted(((str(v), int(n)) for v, n in counts.items()), key=lambda vn: (-vn[1], vn[0]))
    )


def _apart(a: str, b: str) -> bool:
    """Whether two spellings differ by what names a category: digits, Roman numerals, signs."""
    digits = re.sub(r"\D", "", a) != re.sub(r"\D", "", b)
    roman = sorted(_ROMAN.findall(a.casefold())) != sorted(_ROMAN.findall(b.casefold()))
    signs = sorted(_SIGNS.findall(a)) != sorted(_SIGNS.findall(b))
    return digits or roman or signs


def suggest(counts: Mapping[str, int]) -> Suggestions:
    """What each value as written could become (``level_counts``)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for value in counts:
        groups[_folded(value)].append(value)
    spellings: dict[str, str] = {}
    held: dict[str, int] = defaultdict(int)
    for members in groups.values():
        rows: dict[str, int] = defaultdict(int)
        for value in members:
            rows[_spaced(value)] += counts[value]
        spelling = min(rows, key=lambda s: (-rows[s], s))  # the most rows; ties: the first in order
        for value in members:
            held[spelling] += counts[value]
            if value != spelling:
                spellings[value] = spelling
    order = sorted(held, key=lambda s: (-held[s], s))
    offers: dict[str, Offer] = {}
    for spelling in order:
        if _plain(spelling) in _MISSING_CODES:
            offers[spelling] = Offer(spelling, None, MISSING_CODE)
    for short in order:
        if len(short) != 1 or not short.isalpha():
            continue
        longer = [s for s in order if len(s) > 1 and s.casefold().startswith(short.casefold())]
        if len(longer) == 1 and longer[0] not in offers:
            rare, common = sorted((short, longer[0]), key=lambda s: (held[s], s))
            offers.setdefault(rare, Offer(rare, common, FIRST_LETTER))
    for rare in order:
        if rare in offers or len(rare) < 4:
            continue
        for common in order:  # the most frequent first
            if (
                common == rare
                or common in offers
                or len(common) < 4
                or held[rare] > RARE * held[common]
                or _apart(rare, common)
            ):
                continue
            if difflib.SequenceMatcher(None, rare.casefold(), common.casefold()).ratio() >= SIMILAR:
                offers[rare] = Offer(rare, common, SPELLING)
                break
    ranked = tuple(sorted(offers.values(), key=lambda o: order.index(o.value)))
    return Suggestions(spellings, {v: _why(v, s) for v, s in spellings.items()}, ranked)


def recode(values: pd.Series, mapping: Mapping[str, str | None]) -> tuple[pd.Series, int]:
    """
    ``values`` with each value the mapping lists replaced by its clean value (None: missing), the others kept as
    written. Returns the recoded values and how many distinct values the mapping did not list.
    """

    def one(value: Any) -> Any:
        if pd.isna(value):
            return None
        return mapping[str(value)] if str(value) in mapping else value

    seen = {str(v) for v in values.dropna().unique()}
    return values.astype(object).map(one).astype(object), len(seen - set(mapping))


def recode_column(
    df: pd.DataFrame,
    source: str,
    target: str,
    mapping: Mapping[str, str | None],
    *,
    replaces: bool = False,
) -> tuple[pd.DataFrame, int]:
    """
    The table with ``target``, the recoded copy of ``source``, right after it (``replaces``: an earlier copy of that
    name is written anew). Returns the table and how many distinct values the mapping did not list.
    """
    if source not in df.columns:
        raise ValueError(f"{source!r} is not a column of the data")
    if target == source:
        raise ValueError("the clean values go into a new column: the original is kept")
    if target in df.columns and not replaces:
        raise ValueError(f"{target!r} is already a column of the data")
    recoded, unlisted = recode(df[source], mapping)
    out = df.drop(columns=[target], errors="ignore")
    out.insert(out.columns.get_loc(source) + 1, target, recoded)
    return out, unlisted


__all__ = [
    "CAPITALS",
    "FIRST_LETTER",
    "MISSING_CODE",
    "SPACES",
    "SPACES_AND_CAPITALS",
    "SPELLING",
    "Offer",
    "Suggestions",
    "level_counts",
    "recode",
    "recode_column",
    "suggest",
]
