"""
The values of a categorical variable as written, recoded (Data > Clean > Categories): each value given a clean value
or set to missing, and written into a new text column beside the original, which is kept. One traced step
(``recode_levels``) that the replay redoes.

What the step records, and what the replay recomputes:

- the fold rule (:data:`FOLD`), never the values it folds. Values that differ by spaces take their group's spelling,
  the one most rows hold. So do values that differ by capitals in words of 4 letters or more (Male, MALE, male).
  In shorter codes, case can carry meaning (genotypes AA, Aa, aa; Rh haplotypes Cde, cde; pH): a variant there is
  folded only when it holds under :data:`CASE_SHARE` of its group's rows and the group writes none of its spellings
  in mixed case (M and m, NR and nr); otherwise it is offered. The replay folds the values it reads with this rule;
- the user's choices (``mapping``): a ticked offer or a renamed value applies to its spelling's whole group, a value
  given its own clean value to itself only, and ``kept`` lists the folded values put back as written;
- what each choice is (``kinds``), what each value set to missing means (``missing_meaning``), and why two values
  that each hold :data:`GROUPING_SHARE` of the rows were merged (``groupings``);
- counts, never a count per value.

The offers, never applied:

- a one-letter code and the only longer value it begins (the rarer joins the more frequent);
- a rare near spelling of a frequent value (a one-row "Luxembourgg" beside 397 "Luxembourg");
- a case variant the fold leaves apart;
- the codes that may mean missing. Some are read as unknown; for others the user must say what they mean ("n/a" can
  mean not applicable, "none" an absence, "NR" non-reactive).

One edit apart can be two categories, so a near spelling is never offered in these cases:

- across a token holding a digit, a short token or an acronym (HFmrEF and HFrEF, Stage IIIB and IIIA, 3a and 3b);
- when the words differ in their first four letters (Unstable and Stable, Abnormal and Normal, Hypotension and
  Hypertension);
- between two values that are not rare against each other.

Nothing here reads an outcome: the values and their rows only.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

FOLD = "spaces_and_capitals"  # the fold rule's id, recorded with the step: a changed rule takes a new id
CASE_SHARE = (
    0.05  # in a code, a case variant under this share of its group's rows is folded; above, offered
)
SIMILAR = 0.85  # difflib's ratio from which two spellings are near
RARE_SHARE, RARE_ROWS = (
    0.01,
    5,
)  # a near spelling holds at most 1 % of its target's rows, and at most 5 rows
GROUPING_SHARE = 0.05  # two values that each hold this share of the rows, merged: a grouping, which needs a reason
MAX_VALUES = 200  # more distinct values: free text or an identifier, not listed
# why a value is given its clean value (stored with the step: never renamed)
SPACES, CAPITALS, SPACES_AND_CAPITALS = "spaces", "capitals", "spaces and capitals"
FIRST_LETTER, SPELLING, MISSING_CODE, CHOSEN, GROUPING = (
    "first letter",
    "spelling",
    "missing code",
    "chosen",
    "grouping",
)
# what a value set to missing means (stored with the step: never renamed)
UNKNOWN, NOT_APPLICABLE, DECLINED = "unknown or not done", "not applicable", "declined"
MEANINGS = (UNKNOWN, NOT_APPLICABLE, DECLINED)
# codes read as unknown (folded: spaces, capitals and accents)
_UNKNOWN_CODES = frozenset(
    {
        "?",
        "??",
        "unknown",
        "unk",
        "not known",
        "nk",
        "missing",
        "null",
        "not done",
        "not available",
        "inconnu",
        "inconnue",
        "ne sait pas",
        "nsp",
        "non renseigne",
        "non-renseigne",
    }
)
# codes that may mean missing, or not applicable, an absence, a result (ND: not detected; NR: non-reactive)
_ASKED_CODES = frozenset({"n/a", "na", "n.a.", "nan", "none", "nr", "nd", "-", "--", "."})
_ROMAN = re.compile(  # a whole token; the lookbehind refuses the empty match before a word such as "male"
    r"\b(?=[mdclxvi])m{0,4}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})\b(?<=[mdclxvi])"
)
_SIGNS = re.compile(
    r"[+<>=]|(?<![^\W\d_])-|-(?![^\W\d_])"
)  # a hyphen between two letters joins words: no sign
_JOINING_HYPHEN = re.compile(r"(?<=[^\W\d_])-(?=[^\W\d_])")


@dataclass(frozen=True)
class Offer:
    """A spelling that could join another, or become missing (``clean`` None): offered, never applied."""

    value: str  # a spelling, after the fold
    clean: str | None
    why: str  # FIRST_LETTER, SPELLING, CAPITALS or MISSING_CODE
    meaning: str | None = (
        None  # a missing code read as unknown: UNKNOWN; one whose meaning must be said: None
    )


@dataclass(frozen=True)
class Suggestions:
    spellings: dict[
        str, str
    ]  # a value as written -> the spelling the fold gives it, where they differ
    reasons: dict[str, str]  # the same values -> SPACES, CAPITALS or SPACES_AND_CAPITALS
    offers: tuple[Offer, ...]  # the most frequent first


@dataclass(frozen=True)
class Candidate:
    """A text column the Categories tab lists."""

    column: str
    values: int  # distinct values as written
    to_check: int  # the values the fold gives a spelling, and those an offer names
    cleaned_into: str = ""  # the copy a recode of it wrote, still a column


def _spaced(value: str) -> str:
    return " ".join(value.split())


def _folded(value: str) -> str:
    return _spaced(value).casefold()


def _plain(value: str) -> str:
    return unicodedata.normalize("NFKD", _folded(value)).encode("ascii", "ignore").decode()


def _words(text: str) -> list[str]:
    return _JOINING_HYPHEN.sub(" ", text).split()


def _why(value: str, spelling: str) -> str:
    spaces = _spaced(value) != value
    capitals = _spaced(value) != spelling
    return SPACES_AND_CAPITALS if spaces and capitals else SPACES if spaces else CAPITALS


def _case_rank(text: str) -> int:
    """Capitalised first (Female), then lower case (male), then the rest (MALE)."""
    letters = [c for c in text if c.isalpha()]
    if letters and letters[0].isupper() and any(c.islower() for c in letters):
        return 0
    return 1 if text == text.lower() else 2


def _plain_case(word: str) -> bool:
    return word in (word.lower(), word.upper(), word.capitalize())


def _case_free(variant: str, spelling: str) -> bool:
    """Whether two spellings differ only in words whose capitals carry no meaning: 4 letters or more, each in lower
    case, capitals or capitalised."""
    return all(
        a == b or (sum(c.isalpha() for c in a) >= 4 and _plain_case(a) and _plain_case(b))
        for a, b in zip(_words(variant), _words(spelling), strict=False)
    )


def level_counts(values: pd.Series) -> dict[str, int]:
    """The values as written (their text) and their rows, the most frequent first (ties by value); missing values
    left out."""
    counts = values.dropna().astype(str).value_counts()
    return dict(
        sorted(((str(v), int(n)) for v, n in counts.items()), key=lambda vn: (-vn[1], vn[0]))
    )


def _spelling(rows: Mapping[str, int]) -> str:
    """A group's spelling: the most rows; on a tie, capitalised, then lower case, then the first in order."""
    return min(rows, key=lambda s: (-rows[s], _case_rank(s), s))


def _groups(counts: Mapping[str, int]) -> list[dict[str, int]]:
    """The spellings (spaces collapsed) that differ by capitals only, and their rows: one dict per group."""
    groups: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for value, n in counts.items():
        groups[_folded(value)][_spaced(value)] += n
    return [dict(rows) for rows in groups.values()]


def fold(counts: Mapping[str, int]) -> dict[str, str]:
    """
    The rule :data:`FOLD` on the values as written (``level_counts``): each value -> the spelling it takes, where
    they differ.
    """
    takes: dict[str, str] = {}  # a spelling (spaces collapsed) -> the one it takes
    for rows in _groups(counts):
        spelling, total = _spelling(rows), sum(rows.values())
        cased = all(s in (s.lower(), s.upper()) for s in rows)
        for variant, n in rows.items():
            free = _case_free(variant, spelling) or (cased and n < CASE_SHARE * total)
            takes[variant] = spelling if free else variant
    return {value: takes[_spaced(value)] for value in counts if takes[_spaced(value)] != value}


def _marked(text: str) -> bool:
    """Holds a digit, a sign or a Roman numeral: what names a category of its own."""
    return any(c.isdigit() for c in text) or bool(
        _SIGNS.search(text) or _ROMAN.search(text.casefold())
    )


def _strict(token: str) -> bool:
    """A token that names a category by itself: a digit, four characters or fewer, or capitals inside (HFmrEF)."""
    return any(c.isdigit() for c in token) or len(token) <= 4 or any(c.isupper() for c in token[1:])


def _apart(a: str, b: str) -> bool:
    """Whether two spellings differ by what names a category: digits, Roman numerals, signs, a strict token."""
    digits = re.sub(r"\D", "", a) != re.sub(r"\D", "", b)
    roman = sorted(_ROMAN.findall(a.casefold())) != sorted(_ROMAN.findall(b.casefold()))
    signs = sorted(_SIGNS.findall(a)) != sorted(_SIGNS.findall(b))
    strict = sorted(t for t in _words(a) if _strict(t)) != sorted(
        t for t in _words(b) if _strict(t)
    )
    return digits or roman or signs or strict


def _near(rare: str, common: str) -> bool:
    """A near spelling: no category apart, a first difference past a word's first four letters, difflib >= 0.85."""
    if _apart(rare, common):
        return False
    a, b = " ".join(_words(rare)).casefold(), " ".join(_words(common)).casefold()
    first = next(
        (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b))
    )
    if (
        first - (a.rfind(" ", 0, first) + 1) < 4
    ):  # un-, ab-, non-, hypo- and hyper-: another category
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= SIMILAR


def suggest(counts: Mapping[str, int], *, near: bool = True) -> Suggestions:
    """
    What each value as written could become (``level_counts``): the fold, pre-filled, and the offers (near spellings
    only with ``near``: they compare every rare value with every frequent one).
    """
    spellings = fold(counts)
    held: dict[str, int] = defaultdict(int)
    for value, n in counts.items():
        held[spellings.get(value, value)] += n
    order = sorted(held, key=lambda s: (-held[s], s))
    offers: dict[str, Offer] = {}
    for spelling in order:
        plain = _plain(spelling)
        if plain in _UNKNOWN_CODES:
            offers[spelling] = Offer(spelling, None, MISSING_CODE, UNKNOWN)
        elif plain in _ASKED_CODES:
            offers[spelling] = Offer(spelling, None, MISSING_CODE)
    main: dict[str, str] = {}  # a spelling -> its group's (the fold left case variants apart)
    for rows in _groups(held):
        spelling = _spelling(rows)
        for variant in rows:
            main[variant] = spelling
            if variant != spelling and variant not in offers:
                offers[variant] = Offer(variant, spelling, CAPITALS)
    for short in order:
        if len(short) != 1 or not short.isalpha() or short in offers:
            continue
        longer = {
            main[s] for s in order if len(s) > 1 and s.casefold().startswith(short.casefold())
        }
        if len(longer) != 1:
            continue
        (word,) = longer
        if sum(c.isalpha() for c in word) >= 3 and not _marked(word) and word not in offers:
            rare, common = sorted((short, word), key=lambda s: (held[s], s))
            offers[rare] = Offer(rare, common, FIRST_LETTER)
    if near:
        for rare in order:
            if rare in offers or held[rare] > RARE_ROWS:
                continue
            for common in order:  # the most frequent first
                if held[rare] > RARE_SHARE * held[common]:
                    break
                if (
                    common not in offers
                    and common.casefold() != rare.casefold()
                    and _near(rare, common)
                ):
                    offers[rare] = Offer(rare, common, SPELLING)
                    break
    ranked = tuple(sorted(offers.values(), key=lambda o: order.index(o.value)))
    return Suggestions(spellings, {v: _why(v, s) for v, s in spellings.items()}, ranked)


def clean_value(
    value: str,
    spellings: Mapping[str, str],
    mapping: Mapping[str, str | None],
    kept: Collection[str],
) -> str | None:
    """
    What one value as written becomes: as written when ``kept``; its own choice when the fold gave it a spelling and
    the mapping names it; else the choice for its spelling (its group's), or that spelling.
    """
    if value in kept:
        return value
    spelling = spellings.get(value, value)
    if value != spelling and value in mapping:
        return mapping[value]
    return mapping.get(spelling, spelling)


def _spellings(counts: Mapping[str, int], fold_rule: str | None) -> dict[str, str]:
    if fold_rule not in (None, FOLD):
        raise ValueError(f"unknown fold rule {fold_rule!r}; the rule is {FOLD!r}")
    return fold(counts) if fold_rule == FOLD else {}


def plan(
    counts: Mapping[str, int],
    mapping: Mapping[str, str | None],
    *,
    fold_rule: str | None = FOLD,
    kept: Collection[str] = (),
) -> dict[str, str | None]:
    """Each value as written that a recode changes -> its clean value (None: missing). ``fold_rule`` None: no fold
    (the mapping lists every change)."""
    spellings, kept = _spellings(counts, fold_rule), set(kept)
    full = {value: clean_value(value, spellings, mapping, kept) for value in counts}
    return {value: clean for value, clean in full.items() if clean != value}


def recode(values: pd.Series, full: Mapping[str, str | None]) -> pd.Series:
    """The text of ``values`` with each value ``full`` lists replaced by its clean value (None: missing), as a text
    column: 2 and "2" are one value, as ``level_counts`` counts them."""
    text = values.astype(object).map(lambda v: None if pd.isna(v) else str(v))
    return text.map(lambda v: v if v is None else full.get(v, v)).astype("str")


def recode_column(
    df: pd.DataFrame,
    source: str,
    target: str,
    mapping: Mapping[str, str | None],
    *,
    fold_rule: str | None = FOLD,
    kept: Collection[str] = (),
    replaces: bool = False,
) -> tuple[pd.DataFrame, int]:
    """
    The table with ``target``, the recoded copy of ``source``, right after it (``replaces``: an earlier copy of that
    name is written anew). Returns the table and how many distinct values were kept as written.
    """
    if source not in df.columns:
        raise ValueError(f"{source!r} is not a column of the data")
    if target == source:
        raise ValueError("the clean values go into a new column: the original is kept")
    if target in df.columns and not replaces:
        raise ValueError(f"{target!r} is already a column of the data")
    counts = level_counts(df[source])
    full = plan(counts, mapping, fold_rule=fold_rule, kept=kept)
    out = df.drop(columns=[target], errors="ignore")
    out.insert(out.columns.get_loc(source) + 1, target, recode(df[source], full))
    return out, sum(1 for value in counts if value not in full)


def groupings(counts: Mapping[str, int], full: Mapping[str, str | None]) -> dict[str, list[str]]:
    """
    The clean values that merge two spellings (after the fold) each holding :data:`GROUPING_SHARE` of the rows or
    more (NYHA III and IV): a grouping of categories, not a correction. Clean value -> those spellings.
    """
    total = sum(counts.values())
    spellings = fold(counts)
    into: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for value, n in counts.items():
        clean = full.get(value, value)
        if clean is not None:
            into[clean][spellings.get(value, value)] += n
    return {
        clean: sorted(big)
        for clean, held in into.items()
        if len(big := [s for s, n in held.items() if n >= GROUPING_SHARE * total]) >= 2
    }


def changes(
    counts: Mapping[str, int],
    mapping: Mapping[str, str | None],
    kinds: Mapping[str, str],
    *,
    fold_rule: str | None = FOLD,
    kept: Collection[str] = (),
    meanings: Mapping[str, str] | None = None,
) -> dict[str, dict[str, list[int]]]:
    """
    The values a recode changes, counted, never named: ``{"by_kind": {kind: [values, rows]}, "set_to_missing":
    {meaning: [values, rows]}}``. A value takes the kind of the choice it follows, or the fold's reason.
    """
    spellings, kept, meanings = _spellings(counts, fold_rule), set(kept), meanings or {}
    by_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    missing: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for value, n in counts.items():
        clean = clean_value(value, spellings, mapping, kept)
        if clean == value:
            continue
        spelling = spellings.get(value, value)
        key = value if value != spelling and value in mapping else spelling
        kind = kinds.get(key, CHOSEN) if key in mapping else _why(value, spelling)
        for tally, name in (
            (by_kind, kind),
            *(((missing, meanings.get(key, "")),) if clean is None else ()),
        ):
            tally[name][0] += 1
            tally[name][1] += n
    return {"by_kind": dict(by_kind), "set_to_missing": dict(missing)}


def report_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """The step's parameters as the trace report prints them: the choices counted, never named (the trace keeps them
    for the replay)."""
    named = ("mapping", "kinds", "reasons", "kept", "missing_meaning", "groupings")
    out = {k: v for k, v in params.items() if k not in named}
    chosen = (
        params.get("kinds") or params.get("reasons") or {}
    )  # "reasons": every change, before the fold rule
    out["choices"] = {
        kind: sum(1 for k in chosen.values() if k == kind) for kind in sorted(set(chosen.values()))
    }
    out["kept as written"] = len(params.get("kept") or ())
    out["groupings"] = len(params.get("groupings") or {})
    return out


def recoded_columns(steps: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Each column recoded in a trace's steps -> the copy its last recode wrote."""
    out: dict[str, str] = {}
    for step in steps:
        if step.get("function") == "recode_levels":
            params = step.get("params") or {}
            out[str(params.get("source"))] = str(params.get("target"))
    return out


def last_record(
    steps: Iterable[Mapping[str, Any]], source: str, target: str
) -> dict[str, Any] | None:
    """The parameters of the last recode of ``source`` into ``target`` in a trace's steps: cleaning again starts from
    its choices."""
    found = None
    for step in steps:
        params = step.get("params") or {}
        if step.get("function") == "recode_levels" and (
            params.get("source"),
            params.get("target"),
        ) == (source, target):
            found = dict(params)
    return found


def candidates(df: pd.DataFrame, recoded: Mapping[str, str] | None = None) -> list[Candidate]:
    """
    The text and categorical columns with 2 to :data:`MAX_VALUES` values as written, the copies ``recoded`` names
    (source -> copy) left out: those with values to check first, then the cleaned ones, then the others, each in the
    dataset's order. Near spellings are not looked for here (they compare every pair of values): only in the open
    variable.
    """
    recoded = {s: t for s, t in (recoded or {}).items() if s in df.columns and t in df.columns}
    copies = set(recoded.values())
    found = []
    for position, column in enumerate(df.columns):
        values = df[column]
        if (
            str(column) in copies
            or pd.api.types.is_bool_dtype(values)
            or pd.api.types.is_numeric_dtype(values)
            or pd.api.types.is_datetime64_any_dtype(values)
        ):
            continue
        counts = level_counts(values)
        if not 2 <= len(counts) <= MAX_VALUES:
            continue
        hints = suggest(counts, near=False)
        into = recoded.get(str(column), "")
        found.append(
            (
                position,
                Candidate(str(column), len(counts), len(hints.spellings) + len(hints.offers), into),
            )
        )

    def rank(c: Candidate) -> int:  # to check, then cleaned, then the others
        return 1 if c.cleaned_into else 0 if c.to_check else 2

    return [c for _, c in sorted(found, key=lambda pc: (rank(pc[1]), pc[0]))]


__all__ = [
    "CAPITALS",
    "CASE_SHARE",
    "CHOSEN",
    "DECLINED",
    "FIRST_LETTER",
    "FOLD",
    "GROUPING",
    "GROUPING_SHARE",
    "MAX_VALUES",
    "MEANINGS",
    "MISSING_CODE",
    "NOT_APPLICABLE",
    "SPACES",
    "SPACES_AND_CAPITALS",
    "SPELLING",
    "UNKNOWN",
    "Candidate",
    "Offer",
    "Suggestions",
    "candidates",
    "changes",
    "clean_value",
    "fold",
    "groupings",
    "last_record",
    "level_counts",
    "plan",
    "recode",
    "recode_column",
    "recoded_columns",
    "report_params",
    "suggest",
]
