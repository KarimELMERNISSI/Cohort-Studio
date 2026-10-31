"""
The values of a categorical variable as written, recoded (Data > Clean > Categories): each value given a clean value
or set to missing, and written into a new text column beside the original, which is kept. One traced step
(``recode_levels``) that the replay redoes.

What the step records, and what the replay recomputes:

- the fold rule (:data:`FOLD`), never the values it folds: a value written with spaces before, after or repeated
  inside takes its spelling with the spaces collapsed. Capitals are never folded: case can name a category (Kell K
  and k, anti-k and anti-K, g/L and G/L, Rh Ccdee and ccdee), so a case variant is offered. The replay folds the
  values it reads with the rule the step recorded; a changed rule takes a new id, and the two earlier rules recorded
  under :data:`LEGACY_FOLD` are told apart by the record (``rule_of``);
- the user's choices (``mapping``): a choice for a spelling applies to its whole group, a folded value's own choice
  to itself, a clean value that is itself given another is followed (MALE → male → M), and ``kept`` lists the folds
  put back as written;
- what each choice is (``kinds``), what each value set to missing means (``missing_meaning``), and why each grouping
  of categories was made (``groupings``);
- counts by kind and by meaning (a kind that holds one value gives that value's rows: M8's disclosure rule reads
  them), and fingerprints of the values seen and of the copy written.

The offers, never applied:

- a one-letter code and the only longer value it begins (the rarer joins the more frequent);
- a variant: the same spelling once accents are stripped and ae or oe read as e (Pétange, Petange; Anaemia, Anemia);
- a rare spelling one letter from a frequent value (Damerau-Levenshtein distance 1: "Luxembourgg", "Daibetes"), and,
  to check first, a close spelling one letter apart that holds more rows;
- a case variant the fold leaves apart;
- the codes that may mean missing. Some are read as unknown; for others the user says what they mean ("n/a" can mean
  not applicable, "none" an absence, "NR" non-reactive, "ND" a result).

One letter can separate two categories, so a spelling is never offered across a token holding a digit, a short token
or an acronym (HFmrEF and HFrEF, Stage IIIB and IIIA, 3a and 3b), a sign or a Roman numeral, nor when the words
differ in their first four letters (Unstable and Stable, Hypotension and Hypertension), nor two letters apart
(Hypokalaemia and Hypocalcaemia, Craniectomy and Craniotomy). One letter apart can still separate two terms (Dysphasia
and Dysphagia): such an offer says to check it.

Nothing here reads an outcome: the values and their rows only.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
from rapidfuzz.distance import OSA

from core.fingerprint import frame_fingerprint

FOLD = "spaces"  # the fold rule's id, recorded with the step: a changed rule takes a new id
# the id of the two rules of 30 October 2026, which folded capitals too: kept for the replay of their steps
LEGACY_FOLD = "spaces_and_capitals"
_FIRST_FOLD = (
    "spaces_and_capitals, first rule"  # never recorded: the first of the two (``rule_of``)
)
CASE_SHARE = (
    0.05  # the earlier rules: a case variant under this share of its group's rows could be a typo
)
RARE_SHARE, RARE_ROWS = (
    0.01,
    5,
)  # a rare spelling: at most 1 % of its target's rows, and at most 5 rows
CLOSE_SHARE = (
    0.05  # a close spelling, to check first: one letter apart, under 5 % of its target's rows
)
GROUPING_SHARE = (
    0.05  # two values that each hold this share of the rows, merged by hand: a grouping
)
MAX_VALUES = 200  # more distinct values: free text or an identifier, not listed
# why a value is given its clean value (stored with the step: never renamed)
SPACES, CAPITALS, SPACES_AND_CAPITALS = "spaces", "capitals", "spaces and capitals"
FIRST_LETTER, VARIANT, SPELLING, CLOSE = "first letter", "variant", "spelling", "close spelling"
MISSING_CODE, CHOSEN, GROUPING = "missing code", "chosen", "grouping"
AI = "ai"  # proposed by the AI engine (a synonym, a translation): checked here, ticked by the user
# the offers, never a grouping: a variant, a spelling, a synonym the AI engine proposes (one category written two
# ways); a merge of two categories is made by hand, and says why
OFFER_KINDS = frozenset({CAPITALS, FIRST_LETTER, VARIANT, SPELLING, CLOSE, MISSING_CODE, AI})
# what a value set to missing means (stored with the step: never renamed)
UNKNOWN, NOT_APPLICABLE, DECLINED = "unknown or not done", "not applicable", "declined"
MEANINGS = (UNKNOWN, NOT_APPLICABLE, DECLINED)
RESULT = "a result: keep as a value"  # the other answer to a code that may mean missing ("ND": not detected)
# codes read as unknown (folded: spaces, capitals and accents)
_UNKNOWN_CODES = frozenset(
    {
        "?",
        "??",
        "unknown",
        "unk",
        "not known",
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
# codes that may mean missing, or not applicable, an absence, a result: NK (not known, or natural killer), NR
# (non-reactive), ND (not detected). "NA" is not here: the import reads it as missing, and "Na" is sodium
_ASKED_CODES = frozenset(
    {
        "n/a",
        "n.a.",
        "nan",
        "none",
        "nk",
        "nr",
        "nd",
        "nc",
        "so",
        "-",
        "--",
        ".",
        "not tested",
        "untested",
        "refused",
        "refus",
        "sans objet",
        "non communique",
    }
)
# what is no value of its own: the AI engine never joins it to one, or one to it ("Not tested" is not "Negative")
_NOT_VALUES = (
    _UNKNOWN_CODES | _ASKED_CODES | {"declined", "not applicable", "unknown or not done", "other"}
)
# prefixes that negate or oppose a word: a proposal never joins two words a prefix sets apart (Unstable, Stable;
# Hypotension, Hypertension), a broader class neither (STEMI and MI)
_OPPOSITE_PREFIXES = (
    "",
    "un",
    "non",
    "a",
    "an",
    "anti",
    "ab",
    "dis",
    "in",
    "im",
    "ir",
    "il",
    "ex",
    "hypo",
    "hyper",
    "micro",
    "macro",
    "brady",
    "tachy",
    "pre",
    "post",
    "intra",
    "extra",
    "sub",
    "supra",
    "mono",
    "poly",
    "hemi",
    "oligo",
    "pan",
    "de",
    "n",
)
# words that name opposite categories: the AI engine never joins them
_OPPOSITES = frozenset(
    frozenset(pair)
    for pair in (
        ("positive", "negative"),
        ("present", "absent"),
        ("yes", "no"),
        ("oui", "non"),
        ("high", "low"),
        ("increased", "decreased"),
        ("left", "right"),
        ("benign", "malignant"),
        ("acute", "chronic"),
        ("primary", "secondary"),
        ("early", "late"),
        ("mild", "severe"),
        ("upper", "lower"),
        ("alive", "dead"),
        ("true", "false"),
        ("reactive", "non reactive"),
        ("detected", "not detected"),
    )
)
# letters NFKD does not decompose; the minus sign read as a hyphen (a sign, for _SIGNS)
_TRANSLITERATE = str.maketrans(
    {"œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE", "ø": "o", "Ø": "O", "ł": "l", "Ł": "L", "ß": "ss"}
)
_ROMAN = re.compile(  # a whole token; the lookbehind refuses the empty match before a word such as "male"
    r"\b(?=[mdclxvi])m{0,4}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})\b(?<=[mdclxvi])"
)
_SIGNS = re.compile(  # ± ≤ ≥ and the minus sign − too
    r"[+<>=±≤≥−]|(?<![^\W\d_])-|-(?![^\W\d_])"
)  # a hyphen between two letters joins words: no sign
_JOINING_HYPHEN = re.compile(r"(?<=[^\W\d_])-(?=[^\W\d_])")
_NONE = object()  # no choice: a mapping can hold None (missing)


@dataclass(frozen=True)
class Offer:
    """A spelling that could join another, or become missing (``clean`` None): offered, never applied."""

    value: str  # a spelling, after the fold
    clean: str | None
    why: str  # one of OFFER_KINDS, or AI
    meaning: str | None = (
        None  # a missing code read as unknown: UNKNOWN; one whose meaning must be said: None
    )
    reason: str = ""  # an AI proposal's reason, in a few words (the model's text: drawn as text)


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
    return _unaccented(_folded(value))


def _unaccented(text: str) -> str:
    """Accents set aside (Pétange, Petange; Mère, Mere), every other character kept: a Greek letter, µ, ≤ and ≥ name
    categories (α- and β-blocker, IgG κ and λ, µg/L and g/L), so the ASCII form, which drops them, is never used."""
    decomposed = unicodedata.normalize("NFKD", text.translate(_TRANSLITERATE))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def sort_key(value: str) -> tuple[str, str]:
    """A to Z as a list of values reads it: case, spaces and accents set aside, so that a value's spellings sit
    together (Pétange, Petange, Petange␣), then the value as written."""
    return (_key(value), value)


def _key(value: str) -> str:
    """Two spellings of one word compare equal: spaces, capitals, accents and hyphens set aside, ae and oe read as e
    before two letters or a vowel (Anaemia, Anemia; Oedema, Œdème, Edema; Diarrhoea, Diarrhea), never before a
    word's last letter (Michael and Michel, Noel and Nel stay apart)."""
    text = " ".join(_unaccented(value).replace("-", " ").split()).casefold()
    return re.sub(r"[ao]e(?=[^\W\d_]{2}|[aeiouy])", "e", text)


def _words(text: str) -> list[str]:
    return _JOINING_HYPHEN.sub(" ", text).split()


def _why(value: str, spelling: str) -> str:
    spaces = _spaced(value) != value
    capitals = _spaced(value) != spelling
    return SPACES_AND_CAPITALS if spaces and capitals else SPACES if spaces else CAPITALS


def _case_rank(text: str, code: bool) -> int:
    """A word: capitalised first (Female), then lower case (male), then capitals (MALE); a code: capitals first."""
    letters = [c for c in text if c.isalpha()]
    upper, lower = text == text.upper(), text == text.lower()
    capitalised = bool(letters) and letters[0].isupper() and any(c.islower() for c in letters)
    order = (upper, capitalised, lower) if code else (capitalised, lower, upper)
    return next((i for i, holds in enumerate(order) if holds), len(order))


def _inner_capital(text: str) -> bool:
    """A capital right after a lower-case letter (CcDee, HFmrEF, pH, mg/dL): case that carries meaning."""
    return any(a.islower() and b.isupper() for a, b in zip(text, text[1:], strict=False))


def level_counts(values: pd.Series) -> dict[str, int]:
    """The values as written (their text) and their rows, the most frequent first (ties by value); missing values
    left out."""
    counts = values.dropna().astype(str).value_counts()
    return dict(
        sorted(((str(v), int(n)) for v, n in counts.items()), key=lambda vn: (-vn[1], vn[0]))
    )


def _spelling(rows: Mapping[str, int]) -> str:
    """A group's spelling: the most rows; on a tie, by case (a code takes capitals), then the first in order."""
    code = all(len(word) <= 3 for spelling in rows for word in _words(spelling))
    return min(rows, key=lambda s: (-rows[s], _case_rank(s, code), s))


def _groups(counts: Mapping[str, int]) -> list[dict[str, int]]:
    """The spellings (spaces collapsed) that differ by capitals only, and their rows: one dict per group."""
    groups: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for value, n in counts.items():
        groups[_folded(value)][_spaced(value)] += n
    return [dict(rows) for rows in groups.values()]


def fold(counts: Mapping[str, int]) -> dict[str, str]:
    """
    The rule :data:`FOLD` on the values as written (``level_counts``): each value -> the spelling it takes, where
    they differ. Spaces only: a value that differs from another by capitals is offered, never folded (case names
    categories in codes, antibodies and units: Kell K and k, anti-k and anti-K, g/L and G/L, Rh Ccdee and ccdee).
    """
    return {value: _spaced(value) for value in counts if _spaced(value) != value}


def rule_of(params: Mapping[str, Any]) -> str | None:
    """
    The fold rule a recorded step used (None: none, its mapping lists every change). Two rules were recorded under
    :data:`LEGACY_FOLD` on 30 October 2026: the first (3ea6198, 747ccc3), then the second (191a550 on), whose records
    alone hold ``values_fp``; each replays as it wrote.
    """
    rule = params.get("fold")
    return _FIRST_FOLD if rule == LEGACY_FOLD and "values_fp" not in params else rule


def _first_fold(counts: Mapping[str, int]) -> dict[str, str]:
    """
    The first rule recorded under :data:`LEGACY_FOLD` (replays only): spaces, and a case variant of a group's spelling
    (the most rows; on a tie capitalised, then lower case) whose differing words hold 4 letters or more, each in lower
    case, capitals or capitalised; in a group written in lower case or capitals only, any variant under
    :data:`CASE_SHARE` of its rows. No chain.
    """
    takes: dict[str, str] = {}
    for rows in _groups(counts):
        spelling, total = (
            min(rows, key=lambda s: (-rows[s], _first_case_rank(s), s)),
            sum(rows.values()),
        )
        cased = all(s in (s.lower(), s.upper()) for s in rows)
        for variant, n in rows.items():
            free = _case_free(variant, spelling) or (cased and n < CASE_SHARE * total)
            takes[variant] = spelling if free else variant
    return {value: takes[_spaced(value)] for value in counts if takes[_spaced(value)] != value}


def _first_case_rank(text: str) -> int:
    letters = [c for c in text if c.isalpha()]
    if letters and letters[0].isupper() and any(c.islower() for c in letters):
        return 0
    return 1 if text == text.lower() else 2


def _case_free(variant: str, spelling: str) -> bool:
    def plain_case(word: str) -> bool:
        return word in (word.lower(), word.upper(), word.capitalize())

    return all(
        a == b or (sum(c.isalpha() for c in a) >= 4 and plain_case(a) and plain_case(b))
        for a, b in zip(_words(variant), _words(spelling), strict=False)
    )


def _legacy_fold(counts: Mapping[str, int]) -> dict[str, str]:
    """The second rule recorded under :data:`LEGACY_FOLD` (replays only): spaces, and a word's case variant in a
    typo's number of rows."""
    meaningful = any(
        _inner_capital(value) for value in counts
    )  # the column's capitals carry meaning
    takes: dict[str, str] = {}  # a spelling (spaces collapsed) -> the one it takes
    for rows in _groups(counts):
        spelling, total = _spelling(rows), sum(rows.values())
        word = any(
            len(w) >= 4 for w in _words(spelling)
        )  # in a code (Kell K, k; g/l, G/L), case can mean
        for variant, n in rows.items():
            typo = n <= RARE_ROWS and n < CASE_SHARE * total
            takes[variant] = (
                spelling if variant == spelling or (typo and word and not meaningful) else variant
            )
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
    strict = sorted(_ascii(t) for t in _words(a) if _strict(t)) != sorted(
        _ascii(t) for t in _words(b) if _strict(t)
    )
    return digits or roman or signs or strict


def _ascii(text: str) -> str:
    """Accents set aside, case and every other character kept (Mère, Mere; HFrEF and HFREF, κ and λ stay apart)."""
    return _unaccented(text)


def _into_word(a: str, b: str) -> int:
    """How far into its word the first difference of two keys falls."""
    first = next(
        (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b))
    )
    return first - (a.rfind(" ", 0, first) + 1)


def _relation(rare: str, common: str, keys: Mapping[str, str] | None = None) -> str | None:
    """
    How two spellings may be one category: VARIANT (equal keys), SPELLING (one letter apart: Damerau-Levenshtein
    distance 1 on the keys, the first difference past a word's first four letters), or None. Never across what names
    a category (``_apart``).
    """
    a = keys[rare] if keys else _key(rare)
    b = keys[common] if keys else _key(common)
    if not any(c.isalnum() for c in a) or not any(c.isalnum() for c in b):
        return None
    if a == b:
        return None if _apart(rare, common) else VARIANT
    if abs(len(a) - len(b)) > 1 or _into_word(a, b) < 4 or OSA.distance(a, b) != 1:
        return None
    return None if _apart(rare, common) else SPELLING


def _first_letter(short: str, word: str) -> bool:
    """A one-letter code and a word of 3 letters or more it begins, which holds no digit, sign or Roman numeral."""
    return (
        len(short) == 1
        and short.isascii()  # a Latin letter: a Greek one names a category (α, β, κ)
        and short.isalpha()
        and sum(c.isalpha() for c in word) >= 3
        and not _marked(word)
        and _key(word).startswith(_key(short))
    )


def suggest(counts: Mapping[str, int], *, near: bool = True) -> Suggestions:
    """
    What each value as written could become (``level_counts``): the fold, pre-filled, and the offers (variants and
    spellings one letter apart only with ``near``: they compare the spellings two by two).
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
        if len(short) != 1 or not short.isascii() or not short.isalpha() or short in offers:
            continue
        longer = {main[s] for s in order if len(s) > 1 and _key(s).startswith(_key(short))}
        if len(longer) != 1:
            continue
        (word,) = longer
        if _first_letter(short, word) and word not in offers:
            rare, common = sorted((short, word), key=lambda s: (held[s], s))
            offers[rare] = Offer(rare, common, FIRST_LETTER)
    if near:
        keys = {s: _key(s) for s in order}
        for rare in order:
            if rare in offers:
                continue
            for common in order:  # the most frequent first; only into a more frequent spelling
                if held[common] <= held[rare]:
                    break
                if (
                    common in offers
                    or main[common] != common
                    or common.casefold() == rare.casefold()
                ):
                    continue
                relation = _relation(rare, common, keys)
                if relation == VARIANT:
                    offers[rare] = Offer(rare, common, VARIANT)
                elif relation == SPELLING and held[rare] <= min(
                    RARE_ROWS, RARE_SHARE * held[common]
                ):
                    offers[rare] = Offer(rare, common, SPELLING)
                elif relation == SPELLING and held[rare] < CLOSE_SHARE * held[common]:
                    offers[rare] = Offer(rare, common, CLOSE)
                if rare in offers:
                    break
    ranked = tuple(sorted(offers.values(), key=lambda o: order.index(o.value)))
    return Suggestions(spellings, {v: _why(v, s) for v, s in spellings.items()}, ranked)


def _hop(
    value: str,
    spellings: Mapping[str, str],
    mapping: Mapping[str, str | None],
    kept: Collection[str],
) -> tuple[str | None, str | None]:
    """One step: what a value becomes, and the key of the choice that says so (None: as written, or the fold)."""
    if value in kept:
        return value, None
    spelling = spellings.get(value, value)
    if value != spelling and value in mapping:
        return mapping[value], value
    if spelling in mapping:
        return mapping[spelling], spelling
    return spelling, None


def resolve(
    value: str,
    spellings: Mapping[str, str],
    mapping: Mapping[str, str | None],
    kept: Collection[str],
    *,
    chain: bool = True,
) -> tuple[str | None, str | None, str | None]:
    """
    What one value as written becomes, the key of the first choice it follows (its kind's) and of the last (what
    missing means): as written when ``kept``; its own choice when the fold gave it a spelling and the mapping names
    it; else its spelling's choice (its group's), or that spelling. With ``chain``, a clean value that is itself
    given another is followed (MALE → male → M); a cycle (F → Female → F) takes the first choice only.
    """
    clean, first = _hop(value, spellings, mapping, kept)
    if not chain or first is None or clean is None:
        return clean, first, first
    path, last, current = [value], first, clean
    while True:
        if current in path:
            first_clean = _hop(value, spellings, mapping, kept)[0]
            return first_clean, first, first
        path.append(current)
        nxt, key = _hop(current, spellings, mapping, kept)
        if key is None:
            return nxt, first, last
        if nxt is None:
            return None, first, key
        current, last = nxt, key


def clean_value(
    value: str,
    spellings: Mapping[str, str],
    mapping: Mapping[str, str | None],
    kept: Collection[str],
    *,
    chain: bool = True,
) -> str | None:
    """What one value as written becomes (``resolve``)."""
    return resolve(value, spellings, mapping, kept, chain=chain)[0]


def _spellings(counts: Mapping[str, int], fold_rule: str | None) -> dict[str, str]:
    rules = {FOLD: fold, LEGACY_FOLD: _legacy_fold, _FIRST_FOLD: _first_fold}
    if fold_rule is not None and fold_rule not in rules:
        raise ValueError(
            f"unknown fold rule {fold_rule!r}; the rules are {FOLD!r} and {LEGACY_FOLD!r}"
        )
    return rules[fold_rule](counts) if fold_rule is not None else {}


def _chains(fold_rule: str | None) -> bool:
    """Whether a clean value that is itself given another is followed: from the second rule of 30 October 2026 on."""
    return fold_rule in (FOLD, LEGACY_FOLD)


def plan(
    counts: Mapping[str, int],
    mapping: Mapping[str, str | None],
    *,
    fold_rule: str | None = FOLD,
    kept: Collection[str] = (),
) -> dict[str, str | None]:
    """Each value as written that a recode changes -> its clean value (None: missing). ``fold_rule`` None: no fold and
    no chain (a step recorded before the fold rule listed every change)."""
    spellings, kept = _spellings(counts, fold_rule), set(kept)
    chain = _chains(fold_rule)
    full = {value: clean_value(value, spellings, mapping, kept, chain=chain) for value in counts}
    return {value: clean for value, clean in full.items() if clean != value}


def outcome(
    counts: Mapping[str, int],
    mapping: Mapping[str, str | None],
    meanings: Mapping[str, str],
    *,
    kept: Collection[str] = (),
    fold_rule: str | None = FOLD,
) -> dict[str, tuple[str | None, str | None]]:
    """Each value as written -> what it becomes, and what missing means when it becomes missing (the preview; of a
    recorded step, with the rule it recorded)."""
    spellings, kept = _spellings(counts, fold_rule), set(kept)
    out: dict[str, tuple[str | None, str | None]] = {}
    for value in counts:
        clean, _, last = resolve(value, spellings, mapping, kept, chain=_chains(fold_rule))
        out[value] = (clean, meanings.get(last) if clean is None and last is not None else None)
    return out


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


def values_fp(counts: Mapping[str, int], fold_rule: str | None = FOLD) -> str:
    """A fingerprint of the values as written and the spelling the fold gives each (with the rule a step recorded):
    the replay says when the values it reads differ from those the step saw (a value swapped for another; under the
    rules of 30 October 2026, a spelling that lost its majority)."""
    spellings = _spellings(counts, fold_rule)
    text = json.dumps(sorted((v, spellings.get(v, v)) for v in counts), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# The choices of the page, until Apply: {"mapping", "kinds", "kept", "meanings", ...} (a session's dict). The page
# and the dialog of the analysis warnings change them only through these functions.


def new_choices() -> dict[str, Any]:
    # "ai_by": who proposed each AI choice ({provider, model, where, values_sent, values_held}), recorded with the step
    return {"mapping": {}, "kinds": {}, "kept": [], "meanings": {}, "reasons": {}, "ai_by": {}}


def _drop(choices: MutableMapping[str, Any], key: str) -> None:
    for part in ("mapping", "kinds", "meanings", "ai_by"):
        choices.setdefault(part, {}).pop(key, None)


def give(
    choices: MutableMapping[str, Any],
    counts: Mapping[str, int],
    value: str,
    clean: str | None,
    *,
    meaning: str | None = None,
) -> None:
    """
    One value as written given a clean value (None: missing, ``meaning`` what missing means there; the value itself:
    as written). The choice of a group's spelling is its group's; a folded value's own choice is its own, and one
    that agrees with its group, or puts its fold back, is no choice of its own.
    """
    spellings = fold(counts)
    mapping, kept = choices["mapping"], choices["kept"]
    spelling = spellings.get(value, value)
    if value in kept:
        kept.remove(value)
    if (
        clean is not None and clean == value
    ):  # as written: its own choice goes (a spelling's: its group's)
        _drop(choices, value)
        if resolve(value, spellings, mapping, kept)[0] != value:
            kept.append(value)  # its fold, or its group's choice, put back for this value
        return
    if clean is not None and value != spelling:
        without = {k: v for k, v in mapping.items() if k != value}
        if (
            clean == resolve(spelling, spellings, without, kept)[0]
        ):  # its group's clean value: no choice of its own
            _drop(choices, value)
            return
    if mapping.get(value, _NONE) != clean:
        choices["kinds"][value] = CHOSEN
    mapping[value] = clean
    if clean is None and meaning:
        choices["meanings"][value] = meaning
    elif clean is not None:
        choices["meanings"].pop(value, None)


def unset_missing(choices: MutableMapping[str, Any], counts: Mapping[str, int], value: str) -> None:
    """A value no longer missing: its own choice goes; when its group's choice still sets it to missing, it is kept as
    written."""
    _drop(choices, value)
    spellings = fold(counts)
    if resolve(value, spellings, choices["mapping"], choices["kept"])[0] is None:
        choices["kept"].append(value)


def tick(
    choices: MutableMapping[str, Any],
    counts: Mapping[str, int],
    offer: Offer,
    *,
    by: Mapping[str, Any] | None = None,
) -> None:
    """An offer ticked: its spelling's choice, for the whole group; the members' own choices that agree with it go.
    ``by``: who proposed an AI offer ({provider, model, where, values_sent, values_held}), kept with the choice."""
    spellings = fold(counts)
    for value in counts:
        same = value != offer.value and spellings.get(value, value) == offer.value
        if same and choices["mapping"].get(value, _NONE) == offer.clean:
            _drop(choices, value)
    if offer.value in choices["kept"]:
        choices["kept"].remove(offer.value)
    choices["mapping"][offer.value] = offer.clean
    choices["kinds"][offer.value] = offer.why
    if offer.clean is None and offer.meaning:
        choices["meanings"][offer.value] = offer.meaning
    if offer.why == AI and by:
        choices.setdefault("ai_by", {})[offer.value] = dict(by)


def untick(choices: MutableMapping[str, Any], counts: Mapping[str, int], offer: Offer) -> None:
    """An offer unticked, as ``tick`` ticks it: its spelling's choice goes, and the members' own choices that give what
    it proposes (a row given that clean value in the table), so that it reads unticked."""
    spellings = fold(counts)
    for value in (offer.value, *counts):
        if (
            spellings.get(value, value) == offer.value
            and choices["mapping"].get(value, _NONE) == offer.clean
        ):
            _drop(choices, value)


def swaps(counts: Mapping[str, int], choices: Mapping[str, Any]) -> list[str]:
    """The values whose choices lead back to themselves (Oui → Yes → Oui): the copy would swap their names, not merge
    them; the page asks for one name first."""
    spellings, mapping, kept = fold(counts), choices["mapping"], set(choices["kept"])
    found = []
    for value in counts:
        node, path = value, [value]
        clean, key = _hop(value, spellings, mapping, kept)
        while key is not None and clean is not None and clean != node:
            if clean in path:
                found.append(value)
                break
            path.append(clean)
            node = clean
            clean, key = _hop(node, spellings, mapping, kept)
    return found


def ticked(choices: Mapping[str, Any], counts: Mapping[str, int], offer: Offer) -> bool:
    """An offer reads ticked when its spelling's choice is the offer's, or every value of its group already becomes
    what it proposes (a row given that clean value by hand)."""
    if offer.value in choices["mapping"] and choices["mapping"][offer.value] == offer.clean:
        return True
    spellings = fold(counts)
    group = [v for v in counts if spellings.get(v, v) == offer.value]
    return bool(group) and all(
        resolve(v, spellings, choices["mapping"], choices["kept"])[0] == offer.clean for v in group
    )


def recorded(
    counts: Mapping[str, int], choices: Mapping[str, Any]
) -> tuple[dict[str, str | None], dict[str, str], list[str], dict[str, str]]:
    """
    What the step records of the choices: the mapping, its kinds, the folds put back and what each missing value
    means, without a choice the others already make (a choice its group or the fold makes, a member's choice that
    agrees with its group's). Choices are left out one at a time against the running mapping, comparing what each value
    becomes and what missing means, so that the record always gives the preview. Raises ValueError if it does not
    (never: the page then refuses Apply).
    """
    meanings = dict(choices["meanings"])
    mapping = {k: v for k, v in choices["mapping"].items() if v != ""}
    kept = [v for v in dict.fromkeys(choices["kept"]) if v in counts]
    target = outcome(counts, mapping, meanings, kept=kept)
    for key in sorted(mapping):
        trial = {k: v for k, v in mapping.items() if k != key}
        if outcome(counts, trial, meanings, kept=kept) == target:
            mapping = trial
    for value in list(kept):
        trial_kept = [k for k in kept if k != value]
        if outcome(counts, mapping, meanings, kept=trial_kept) == target:
            kept = trial_kept
    if outcome(counts, mapping, meanings, kept=kept) != target:
        raise ValueError("the recorded choices would not give the preview")
    kinds = {k: str(choices["kinds"].get(k, CHOSEN)) for k in mapping}
    return (
        mapping,
        kinds,
        kept,
        {k: meanings[k] for k, c in mapping.items() if c is None and k in meanings},
    )


def _related(a: str, b: str) -> bool:
    """Two spellings an offer would join: a one-letter code and its word, a variant, one letter apart."""
    return _first_letter(a, b) or _first_letter(b, a) or _relation(a, b) is not None


def groupings(
    counts: Mapping[str, int],
    mapping: Mapping[str, str | None],
    kinds: Mapping[str, str],
    *,
    kept: Collection[str] = (),
) -> dict[str, list[str]]:
    """
    The clean values that group categories: a spelling merged by hand (a choice, not an offer) with another that
    something naming a category sets apart (``_apart``: NYHA IV into III, Stage IVA into IV; ``_named_apart``:
    Unstable angina into Stable angina, Type 2 diabetes into Diabetes), unless an offer would join them (a first
    letter, a variant); or two spellings that each hold :data:`GROUPING_SHARE` of the rows. Clean value -> the
    spellings it merges.
    """
    spellings, kept = fold(counts), set(kept)
    held: dict[str, int] = defaultdict(int)
    for value, n in counts.items():
        held[spellings.get(value, value)] += n
    total = sum(counts.values())
    into: dict[str, set[str]] = defaultdict(set)
    hand: dict[str, set[str]] = defaultdict(set)
    for value in counts:
        clean, first, _ = resolve(value, spellings, mapping, kept)
        if clean is None:
            continue
        spelling = spellings.get(value, value)
        into[clean].add(spelling)
        if first is not None and kinds.get(first, CHOSEN) not in OFFER_KINDS:
            hand[clean].add(spelling)

    def grouping(a: str, b: str) -> bool:
        if _key(a) == _key(b):
            return False  # one word, written two ways
        big = min(held.get(a, 0), held.get(b, 0)) >= GROUPING_SHARE * total
        return big or ((_apart(a, b) or _named_apart(a, b)) and not _related(a, b))

    return {
        clean: sorted(into[clean])
        for clean in into
        if any(grouping(a, b) for a in hand.get(clean, ()) for b in into[clean] if a != b)
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
    The values a recode changes, counted by kind and meaning: ``{"by_kind": {kind: [values, rows]},
    "set_to_missing": {meaning: [values, rows]}}``. A value takes the kind of the first choice it follows, or the
    fold's reason; a kind or meaning holding one value gives that value's rows.
    """
    spellings, kept, meanings = _spellings(counts, fold_rule), set(kept), meanings or {}
    by_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    missing: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for value, n in counts.items():
        clean, first, last = resolve(value, spellings, mapping, kept, chain=_chains(fold_rule))
        if clean == value:
            continue
        kind = (
            kinds.get(first, CHOSEN)
            if first is not None
            else _why(value, spellings.get(value, value))
        )
        tallies = [(by_kind, kind)]
        if clean is None:
            tallies.append((missing, meanings.get(last, "") if last is not None else ""))
        for tally, name in tallies:
            tally[name][0] += 1
            tally[name][1] += n
    return {"by_kind": dict(by_kind), "set_to_missing": dict(missing)}


def report_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """
    The step's parameters as the trace report prints them: the choices counted, never named (the trace keeps them for
    the replay), and each grouping with its reason, naming its values only when each held 5 rows or more. The
    fingerprints are left out: one of a single column can be tested against a guessed cell.
    """
    named = (
        "mapping",
        "kinds",
        "reasons",
        "kept",
        "missing_meaning",
        "groupings",
        "target_fp",
        "values_fp",
    )
    out = {k: v for k, v in params.items() if k not in named}
    chosen = (
        params.get("kinds") or params.get("reasons") or {}
    )  # "reasons": every change, before the fold rule
    out["choices"] = {
        kind: sum(1 for k in chosen.values() if k == kind) for kind in sorted(set(chosen.values()))
    }
    out["folds put back as written"] = len(params.get("kept") or ())
    grouped = params.get("groupings") or {}
    # a reason is free text, which may name the values: printed only beside values each held by 5 rows or more
    out["groupings"] = [
        f"{', '.join(g['values'])} grouped as {clean}: {g['reason']}"
        if isinstance(g, Mapping) and g.get("each_5_rows")
        else f"{len(g.get('values', ())) if isinstance(g, Mapping) else 'some'} values grouped (the reason is in "
        "the trace)"
        for clean, g in grouped.items()
    ]
    proposed: dict[tuple[str, str], int] = defaultdict(int)
    for by in (params.get("ai_by") or {}).values():
        if isinstance(by, Mapping):
            proposed[(str(by.get("provider", "")), str(by.get("model", "")))] += 1
    if proposed:  # who proposed, how many choices: never which values
        out["proposed by the AI engine"] = [
            f"{provider or 'unknown provider'} {model or 'unknown model'}: {n} value{'s' if n != 1 else ''}"
            for (provider, model), n in sorted(proposed.items())
        ]
    out.pop("ai_by", None)
    return out


def free_text(counts: Mapping[str, int]) -> bool:
    """
    Values the AI engine is never asked about: free text (long, or several words each), mostly rare values (those
    held by fewer than 5 rows cover half the rows), or an identifier (many values, none frequent: names, numbers).
    """
    if not counts:
        return False
    lengths = [len(_spaced(v)) for v in counts]
    wordy = sum(1 for v in counts if len(v.split()) >= 5)
    rows = sum(counts.values())
    rare = sum(n for n in counts.values() if n < 5)
    identifier = len(counts) > 30 and max(counts.values()) < 0.05 * rows
    return (
        sum(lengths) / len(lengths) > 40
        or wordy > len(counts) / 2
        or rare * 2 >= rows
        or identifier
    )


def _ai_refused(value: str, clean: str) -> bool:
    """
    Whether a proposal of the AI engine joins what the rules keep apart: a number, a sign or two Roman numerals (Type
    1 and 2, NYHA II and III); words that name two categories (``_named_apart``); a value that is no value of its own
    on either side (Not tested and Negative, Refused and Unknown, Homme and unknown).
    """
    a, b = _key(value), _key(clean)
    return (
        _numbered_apart(value, clean)
        or a in _NOT_VALUES
        or b in _NOT_VALUES
        or _named_apart(value, clean)
    )


def _named_apart(value: str, clean: str) -> bool:
    """
    Whether the words of two values name two categories: a word in common and others apart (Stage IIIA and IIIB,
    Hepatitis B and C, Ex-smoker and Current smoker), unless those differ by a typo (a letter, in words of 5 letters
    or more: Diabetes melitus); two codes a letter or two apart (HFmrEF and HFrEF, pT1a and pT1b, IgG and IgM); two
    units (mg/dL and mmol/L); opposites (Positive and Negative); a prefix that negates or opposes (Unstable and
    Stable, Hypotension and Hypertension); a broader class, or a word inside the other (STEMI and MI, Type 2 diabetes
    and Diabetes, Female and Male). A translation, a synonym or an abbreviation crosses none (Femme and Female, Oui
    and Yes, MI and Myocardial infarction).
    """
    a, b = _key(value), _key(clean)
    if a == b:
        return False
    words_a, words_b = set(a.split()), set(b.split())
    only_a, only_b = words_a - words_b, words_b - words_a
    if all(w.isdigit() or _ROMAN.fullmatch(w) for w in (*only_a, *only_b)) and not _numbered_apart(
        value, clean
    ):
        only_a = only_b = set()  # one number written two ways (NYHA 3 and NYHA III)
    typo = (
        len(only_a) == len(only_b) == 1
        and min(len(w) for w in (*only_a, *only_b)) >= 5
        and OSA.distance(next(iter(only_a)), next(iter(only_b))) == 1
    )
    if words_a & words_b and only_a and only_b and not typo:
        return True
    codes = all(_code(t) for t in _words(value)) and all(_code(t) for t in _words(clean))
    if (codes and OSA.distance(a, b) <= 2) or ("/" in a and "/" in b):
        return True  # two codes a letter or two apart; two ratios (units: mg/dL and mmol/L)
    if frozenset((a, b)) in _OPPOSITES or words_a < words_b or words_b < words_a:
        return True
    if min(len(a), len(b)) >= 2 and (a.endswith(b) or b.endswith(a)):
        return True

    def stems(word: str) -> set[str]:
        return {
            word[len(p) :]
            for p in _OPPOSITE_PREFIXES
            if word.startswith(p) and len(word) - len(p) >= 4
        }

    return any(stems(x) & stems(y) for x in words_a for y in words_b if x != y)


def _code(token: str) -> bool:
    """A code rather than a word: a digit, or a capital after its first character (HFmrEF, pT1a, IgG, MI, NYHA); Oui,
    Yes and Female are words."""
    return any(c.isdigit() for c in token) or any(c.isupper() for c in token[1:])


def _numbered_apart(a: str, b: str) -> bool:
    """Whether a number or a sign sets two values apart (Type 1 and 2, NYHA II and III, HER2- and HER2+), a Roman
    numeral read as the number it names (NYHA 3 and NYHA III, Grade 2 and II are one class): a translation or an
    abbreviation crosses none (Oui and Yes; MI and myocardial infarction, where "MI" reads as a numeral only when the
    other value holds a number too)."""
    digits_a, digits_b = (
        [int(n) for n in re.findall(r"\d+", a)],
        [int(n) for n in re.findall(r"\d+", b)],
    )
    roman_a, roman_b = _ROMAN.findall(a.casefold()), _ROMAN.findall(b.casefold())
    if digits_b or roman_b:
        digits_a += [_roman_value(r) for r in roman_a]
    if digits_a or roman_a:
        digits_b += [_roman_value(r) for r in roman_b]
    signs = sorted(_SIGNS.findall(a)) != sorted(_SIGNS.findall(b))
    return signs or sorted(digits_a) != sorted(digits_b)


def _roman_value(numeral: str) -> int:
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    for here, after in zip(numeral, [*numeral[1:], ""], strict=True):
        total += -values[here] if after and values[after] > values[here] else values[here]
    return total


def ai_offers(
    proposals: Iterable[Mapping[str, Any]],
    counts: Mapping[str, int],
    *,
    sent: Collection[str],
    taken: Collection[str] = (),
) -> tuple[Offer, ...]:
    """
    The AI engine's proposals, read: only a value it was sent (as written, or its spelling), given another value or a
    short standard name, never to missing, never across what the rules keep apart (``_ai_refused``: the prompt
    forbids the rest, and the user ticks each); one per value, those an offer of the rules already names (``taken``)
    left out. A value proposed both ways (Oui → Yes, Yes → Oui) or along a chain (MALE → Male → M) keeps one direction,
    into the value holding the most rows, so that ticking them all never swaps two names. The model's text is kept to a
    short reason, drawn as text. In the order of the values' rows.
    """
    spellings = fold(counts)
    allowed = {spellings.get(v, v) for v in sent if v in counts}
    held: dict[str, int] = defaultdict(int)
    for value, n in counts.items():
        held[spellings.get(value, value)] += n
    found: dict[str, Offer] = {}
    for proposal in proposals:
        value = str(proposal.get("value", ""))
        spelling = spellings.get(value, value)
        clean = " ".join(str(proposal.get("clean", "")).split())
        if spelling not in allowed or spelling in taken or spelling in found:
            continue
        if not clean or len(clean) > 60 or clean == spelling or _ai_refused(spelling, clean):
            continue
        reason = _shortened(" ".join(str(proposal.get("reason", "")).split()), 80)
        found[spelling] = Offer(spelling, clean, AI, reason=reason)
    kept: dict[str, Offer] = {}
    into: set[str] = set()
    for offer in sorted(
        found.values(), key=lambda o: (-held.get(o.clean or "", 0), -held[o.value], o.value)
    ):
        if offer.value not in into and offer.clean not in kept:
            kept[offer.value] = offer
            into.add(offer.clean or "")
    return tuple(sorted(kept.values(), key=lambda o: (-held[o.value], o.value)))


def _shortened(text: str, limit: int) -> str:
    """At most ``limit`` characters, cut after a whole word."""
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0] if " " in text[: limit - 1] else text[: limit - 1]
    return cut.rstrip(" ,;:") + "…"


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
        ) == (
            source,
            target,
        ):
            found = dict(params)
    return found


def choices_of(record: Mapping[str, Any] | None) -> dict[str, Any]:
    """The choices a recorded step made, to clean again from them (its kinds kept)."""
    if record is None:
        return new_choices()
    mapping = dict(record.get("mapping") or {})
    kinds = dict(
        record.get("kinds") or record.get("reasons") or {}
    )  # "reasons": before the fold rule
    grouped = record.get("groupings") or {}
    return {
        "mapping": mapping,
        "kinds": {k: str(kinds.get(k, CHOSEN)) for k in mapping},
        "kept": list(record.get("kept") or []),
        "meanings": dict(record.get("missing_meaning") or {}),
        "reasons": {
            c: (g.get("reason", "") if isinstance(g, Mapping) else str(g))
            for c, g in grouped.items()
        },
        "ai_by": {k: dict(v) for k, v in (record.get("ai_by") or {}).items() if k in mapping},
    }


def replaceable(
    df: pd.DataFrame, steps: Iterable[Mapping[str, Any]], source: str, target: str
) -> bool:
    """
    Whether ``target`` can be written again: it is the copy of ``source`` that the last recode wrote, and no later step
    changed it (its fingerprint still matches, or the recorded recode of the source as it is now still gives it, as
    after a freeze that removed rows from both). A record without a fingerprint (before 747ccc3) is checked by the
    recode alone.
    """
    record = last_record(steps, source, target)
    if record is None or target not in df.columns:
        return False
    fp = record.get("target_fp")
    if fp is not None and frame_fingerprint(df[[target]]) == fp:
        return True
    try:
        again, _ = recode_column(
            df,
            source,
            target,
            record.get("mapping") or {},
            fold_rule=rule_of(record),
            kept=record.get("kept") or (),
            replaces=True,
        )
    except ValueError:
        return False
    return bool(again[target].equals(df[target]))


def numeric_codes(df: pd.DataFrame) -> list[str]:
    """The number columns holding 2 to 10 distinct values (1/2 codes): not listed, and said so."""
    return [
        str(c)
        for c, values in df.items()
        if pd.api.types.is_numeric_dtype(values)
        and not pd.api.types.is_bool_dtype(values)
        and 2 <= int(values.nunique(dropna=True)) <= 10
    ]


def candidates(df: pd.DataFrame, recoded: Mapping[str, str] | None = None) -> list[Candidate]:
    """
    The text and categorical columns with 2 to :data:`MAX_VALUES` values as written, the copies ``recoded`` names
    (source -> copy) left out: those with values to check first, then the cleaned ones, then the others, each in the
    dataset's order. The count of values to check holds every offer the page makes, variants and spellings one letter
    apart included (rapidfuzz keeps the two-by-two comparison to milliseconds).
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
        hints = suggest(
            counts
        )  # variants and spellings one letter apart too: what the page will offer
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
    "AI",
    "CAPITALS",
    "CASE_SHARE",
    "CHOSEN",
    "CLOSE",
    "DECLINED",
    "FIRST_LETTER",
    "FOLD",
    "GROUPING",
    "GROUPING_SHARE",
    "LEGACY_FOLD",
    "MAX_VALUES",
    "MEANINGS",
    "MISSING_CODE",
    "NOT_APPLICABLE",
    "OFFER_KINDS",
    "RESULT",
    "SPACES",
    "SPACES_AND_CAPITALS",
    "SPELLING",
    "UNKNOWN",
    "VARIANT",
    "Candidate",
    "Offer",
    "Suggestions",
    "ai_offers",
    "candidates",
    "changes",
    "choices_of",
    "clean_value",
    "fold",
    "free_text",
    "give",
    "groupings",
    "last_record",
    "level_counts",
    "new_choices",
    "numeric_codes",
    "outcome",
    "plan",
    "recode",
    "recode_column",
    "recoded_columns",
    "recorded",
    "replaceable",
    "report_params",
    "resolve",
    "rule_of",
    "sort_key",
    "suggest",
    "swaps",
    "tick",
    "ticked",
    "unset_missing",
    "untick",
    "values_fp",
]
