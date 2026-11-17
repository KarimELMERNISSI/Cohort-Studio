"""
The study package's formats (M8): the members of a ``.cstudio.zip``, the one writer of their JSON and its strict
reader, the manifest and the package's hash, the rule for a text of a package, and the policy block.

Standard library only, with ``core.privacy`` (itself standard library only): ``python -m manage.replay inspect``
reads a package through this module without the engine, numpy, pandas or Streamlit.

**The members**, in the order the zip holds them (``MEMBERS``): ``package.json`` (what the package is of, and the
policy it was made under), ``protocol.json`` (the study definition as shared), ``results.json`` (the disclosed
results), ``README.md``, ``uv.lock`` when the installation has one, and ``MANIFEST.sha256``, which lists the
SHA-256 of every other member. **The package's hash is the SHA-256 of the manifest's bytes**: it is what a sender
gives apart from the file, and what a reader prints first.

**One writing per meaning.** :func:`dumps` is the only writer of a JSON member: UTF-8 without a byte order mark,
line feeds, two spaces, keys sorted, no NaN, a final line feed, under an envelope ``format`` + ``schema``. One part
keeps the order it is given: the body of ``protocol.json``, where the order of the families and of their rules is
part of the study. Two packages of the same study, made on two machines, are then the same bytes.

**Strict readers.** :func:`loads` refuses what the writer never writes: a byte order mark, a key written twice, NaN
or an infinite number, a nest too deep, a number too long, a member over its size. Every reader here is closed: a
key or a closed value it does not know reads as written by a later version, counted and never named, and a reason
quotes nothing of a file (a key or a value of a received package can hold anything).

**No text of a package holds a control or a direction character, nor one that shows nothing**
(``core.privacy.hidden``, the rule of a study's own names):
such a character can make a name read as another. The writer refuses to write one and the reader refuses a member
that holds one; a line feed inside a text is the one exception (where a text is shown, it must not pass for a
line of its own). :func:`shown` is for what does not come from a member (a file's name, an argument of a command).

**The policy block** (:class:`Policy`) is ``package.json``'s ``policy``: under which privacy rules the package was
made. One closed shape with every key always written, for the three values of ``rules``: ``on`` (a package without
the data, made under the rules: what this version writes), ``off`` (made for a study whose rows are declared not
people) and ``none`` (a package that holds the data). It records what applied; of the study's k it also records
what was asked, ``k_asked`` being null when the study asked none, with the default beside it. The ids of the
tiers, of the members, of ``rules`` and of every closed value are stored ids: never renamed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeGuard

from core.privacy import (
    FLOOR_K,
    INVISIBLE,
    MOST_WITHOUT_THE_EVENT,
    NOT_PEOPLE_REASONS,
    OUTCOME_MAX,
    ROWS_DEFAULT,
    ROWS_WORDS,
    hidden,
    unseen,
)

# ----------------------------------------------------------------------------- tiers and members (stored ids)

SHAREABLE, RULES_OFF, REPRODUCTION = "shareable", "rules_off", "reproduction"
TIERS = (SHAREABLE, RULES_OFF, REPRODUCTION)
_SUFFIX = {SHAREABLE: "", RULES_OFF: "_rules_off", REPRODUCTION: "_with_data"}
EXTENSION = ".cstudio.zip"

PACKAGE, PROTOCOL, RESULTS = "package.json", "protocol.json", "results.json"
README, LOCK, MANIFEST = "README.md", "uv.lock", "MANIFEST.sha256"
MEMBERS = (PACKAGE, PROTOCOL, RESULTS, README, LOCK, MANIFEST)  # the order in the zip
OPTIONAL = (LOCK,)  # an installed package has no lock file
LISTED = tuple(
    sorted(name for name in MEMBERS if name != MANIFEST)
)  # what the manifest lists, in its order

FORMATS = {
    PACKAGE: "cohortstudio.study-package",
    PROTOCOL: "cohortstudio.shared-definition",
    RESULTS: "cohortstudio.study-results",
}
# A member's schema moves when a key or a closed value is added to it; the package's own (package.json's) also
# when the set of members changes.
SCHEMAS = {PACKAGE: 1, PROTOCOL: 1, RESULTS: 1}
ORDERED = {PROTOCOL: ("body",)}  # the parts that keep the order they are given

_MB = 1024 * 1024
CAPS = {
    PACKAGE: _MB,
    PROTOCOL: 8 * _MB,
    RESULTS: 32 * _MB,
    README: _MB,
    LOCK: 4 * _MB,
    MANIFEST: 4096,
}
PACKAGE_CAP = 64 * _MB  # the whole file
DEPTH = 64  # lists and mappings inside one another, at most: no member comes near it

STUDY_ID = re.compile(
    r"[a-z0-9][a-z0-9_-]{0,63}"
)  # as core.protocol.STUDY_ID, which this module does not import
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LINE = re.compile(r"([0-9a-f]{64})  ([A-Za-z0-9_.]+)")
_SCHEMA_MOST = 1_000_000  # a schema number above it is no schema: nothing that long is written back

# why something is refused, in words that quote nothing of a file
NOT_JSON = "cannot be read: it is not the JSON this version writes"
TOO_LARGE = "is larger than a member of a package may be"
KEY_TWICE = "cannot be read: it holds a key written twice"
NOT_A_NUMBER = "cannot be read: it holds a number that is not finite"
NOT_A_MEMBER = "is not the member of a study package it is named as"
NOT_A_MANIFEST = "cannot be read as a package's list of checksums"
NOT_A_POLICY = "its policy cannot be read by this version"
LATER_POLICY_VALUE = (
    "written by a later version: its policy holds a value this version does not know"
)
K_NOT_THE_LARGER = "its policy's k is not the larger of what was asked and the floor"


def hidden_reason(count: int) -> str:
    """Counted, never shown: the characters are what must not reach a screen."""
    return (
        f"holds {count} character{'s' if count != 1 else ''} that no text of a package may hold (they show "
        "nothing, or change the direction of what follows)"
    )


def later_schema(found: int, read: int) -> str:
    return f"written by a later version (schema {found}; this version reads {read})"


def later_keys(count: int) -> str:
    return (
        f"written by a later version: its policy holds {count} key{'s' if count != 1 else ''} this version does "
        "not read"
    )


def package_name(study_id: str, revision: int, tier: str = SHAREABLE) -> str:
    """``<study>_r<revision>.cstudio.zip``; ``_rules_off`` for a package made with the rules off, ``_with_data``
    for one that holds the data: each tier has its own name, so that one is never taken for another."""
    if not isinstance(study_id, str) or not STUDY_ID.fullmatch(study_id):
        raise ValueError("a package is named after its study's id")
    if not _is_count(revision) or revision < 1:
        raise ValueError("a package is of a saved revision")
    if tier not in TIERS:
        raise ValueError("unknown tier")
    return f"{study_id}_r{revision}{_SUFFIX[tier]}{EXTENSION}"


# ----------------------------------------------------------------------------- the texts of a package


def _is_count(value: Any) -> TypeGuard[int]:
    """A whole number as JSON writes one: not a yes or no, not a decimal."""
    return isinstance(value, int) and not isinstance(value, bool)


REPLACED = chr(0xFFFD)  # stands for a character that is not shown
CUT = chr(0x2026)  # the mark of a text cut short


def shown(text: object, limit: int = 120) -> str:
    """A text that does not come from a member (a file's name, an argument of a command), as a terminal or a page
    may show it: on one line, each character no package may hold (a line feed too) replaced by U+FFFD, cut at
    ``limit`` characters."""
    cleaned = "".join(REPLACED if unseen(char) else char for char in str(text))
    return cleaned if len(cleaned) <= limit else cleaned[: max(0, limit - 1)] + CUT


# ----------------------------------------------------------------------------- the one writer of a JSON member


def _written(value: Any, keep: bool) -> Any:
    """``value`` as the writer writes it: mappings with their keys sorted (in the order given under ``keep``),
    lists in their order. ValueError for what no member holds."""
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("a key of a member is a text")
        if any(hidden(key) for key in value):
            raise ValueError("a key " + hidden_reason(sum(hidden(key) for key in value)))
        keys = list(value) if keep else sorted(value)
        return {key: _written(value[key], keep) for key in keys}
    if isinstance(value, (list, tuple)):
        return [_written(item, keep) for item in value]
    if isinstance(value, str):
        if hidden(value):
            raise ValueError("a text " + hidden_reason(hidden(value)))
        return value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("a number of a member is finite")
        return value
    raise ValueError("a member holds texts, numbers, yes or no, nothing, lists and mappings only")


def dumps(member: str, payload: Mapping[str, Any]) -> bytes:
    """
    The bytes of a JSON member: ``payload`` under the member's envelope (``format``, ``schema``), in the one
    writing (see the module's words). ValueError for a payload that names the envelope's keys, or holds what no
    member may: a number that is not finite, a text with a control or a direction character, a key that is not a
    text, a value of another type. The message counts and never quotes.
    """
    if member not in FORMATS:
        raise ValueError("not a JSON member of a study package")
    if "format" in payload or "schema" in payload:
        raise ValueError("the envelope's keys are the writer's")
    kept = ORDERED.get(member, ())
    whole: dict[str, Any] = {"format": FORMATS[member], "schema": SCHEMAS[member]}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError("a key of a member is a text")
        if hidden(key):
            raise ValueError("a key " + hidden_reason(hidden(key)))
        whole[key] = _written(value, keep=key in kept)
    text = json.dumps(
        {key: whole[key] for key in sorted(whole)}, indent=2, ensure_ascii=False, allow_nan=False
    )
    return (text + "\n").encode("utf-8")


# ----------------------------------------------------------------------------- the strict reader


class _Refused(ValueError):
    """A member the strict reader refuses, with its fixed reason."""


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for key, value in items:
        if key in found:
            raise _Refused(KEY_TWICE)
        found[key] = value
    return found


def _no_constant(_name: str) -> Any:
    raise _Refused(NOT_A_NUMBER)


def _checked(value: Any) -> int:
    """How many characters no package may hold stand in ``value``'s keys and texts; _Refused for a number that is
    not finite (``1e999`` is read as one without a word) and for a nest of more than ``DEPTH`` levels (the reader's
    own limit refused only far deeper, and by the interpreter's). An explicit stack: no recursion of its own."""
    count, stack = 0, [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, (dict, list)):
            if depth > DEPTH:
                raise _Refused(NOT_JSON)
            if isinstance(item, dict):
                count += sum(hidden(key) for key in item)
                stack.extend((inner, depth + 1) for inner in item.values())
            else:
                stack.extend((inner, depth + 1) for inner in item)
        elif isinstance(item, str):
            count += hidden(item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise _Refused(NOT_A_NUMBER)
    return count


def loads(data: bytes, cap: int) -> tuple[Any | None, list[str]]:
    """
    A JSON member read strictly: its value, or None with why it is refused. Refused: more than ``cap`` bytes, a
    byte order mark, bytes that are not UTF-8, a key written twice, NaN or an infinite number, a nest of more than
    ``DEPTH`` levels, a number too long for the reader, and a key or a text holding a character no package may
    hold. The reasons quote nothing of the member. Nothing here says what the value means: each member has its
    closed reader.
    """
    if len(data) > cap:
        return None, [TOO_LARGE]
    if data.startswith(b"\xef\xbb\xbf"):
        return None, [NOT_JSON]
    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_no_constant)
        count = _checked(value)
    except _Refused as refused:
        return None, [str(refused)]
    except (
        ValueError,
        RecursionError,
    ):  # not UTF-8, not JSON, nested too deep, a number of too many digits
        return None, [NOT_JSON]
    if count:
        return None, [hidden_reason(count)]
    return value, []


def read_envelope(value: Any, member: str) -> list[str]:
    """Why ``value`` is not the member it is named as, or was written by a later version; [] when it is one this
    version reads. A schema is a whole number from 1: a later one is said with its number, which is bounded."""
    if member not in FORMATS:
        raise ValueError("not a JSON member of a study package")
    if not isinstance(value, dict) or value.get("format") != FORMATS[member]:
        return [NOT_A_MEMBER]
    schema = value.get("schema")
    if not _is_count(schema) or not 1 <= schema <= _SCHEMA_MOST:
        return [NOT_A_MEMBER]
    if schema > SCHEMAS[member]:
        return [later_schema(schema, SCHEMAS[member])]
    return []


# ----------------------------------------------------------------------------- the manifest and the package's hash


def checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_of(members: Mapping[str, bytes]) -> bytes:
    """
    The manifest of a package's other members: one line ``<sha256>  <name>`` each, sorted by name, ASCII, line
    feeds. ValueError for a name that is no member, for the manifest itself, or without a member every package
    holds.
    """
    unknown = set(members) - set(LISTED)
    if unknown:
        raise ValueError("the manifest lists the package's members, and not itself")
    missing = [name for name in LISTED if name not in OPTIONAL and name not in members]
    if missing:
        raise ValueError("a package holds " + ", ".join(missing))
    return "".join(f"{checksum(members[name])}  {name}\n" for name in sorted(members)).encode(
        "ascii"
    )


def read_manifest(data: bytes) -> tuple[dict[str, str] | None, list[str]]:
    """
    The checksums a manifest lists, by member, or None with why it is refused. Strict: at most ``CAPS[MANIFEST]``
    bytes of ASCII, each line ``<64 lower-case hex>  <name>`` ended by a line feed, each name a member of the
    package other than the manifest, each once, sorted, and every member that a package always holds.
    """
    if len(data) > CAPS[MANIFEST] or not data.endswith(b"\n"):
        return None, [NOT_A_MANIFEST]
    try:
        lines = data.decode("ascii").split("\n")[:-1]
    except UnicodeDecodeError:
        return None, [NOT_A_MANIFEST]
    listed: dict[str, str] = {}
    for line in lines:
        match = _LINE.fullmatch(line)
        if match is None or match[2] not in LISTED or match[2] in listed:
            return None, [NOT_A_MANIFEST]
        listed[match[2]] = match[1]
    if list(listed) != sorted(listed):
        return None, [NOT_A_MANIFEST]
    if any(name not in listed for name in LISTED if name not in OPTIONAL):
        return None, [NOT_A_MANIFEST]
    return listed, []


def package_hash(manifest: bytes) -> str:
    """The package's hash: the SHA-256 of its manifest's bytes, which hold the checksum of every other member."""
    return checksum(manifest)


def not_matching(members: Mapping[str, bytes], listed: Mapping[str, str]) -> list[str]:
    """The members whose bytes do not have the checksum the manifest lists, or that one holds and the other does
    not: by their names, which are the format's own."""
    names = sorted(set(members) | set(listed))
    return [
        name
        for name in names
        if name not in members or name not in listed or checksum(members[name]) != listed[name]
    ]


# ----------------------------------------------------------------------------- the policy block

ON, OFF, NONE = "on", "off", "none"  # the values of ``rules`` (stored ids)
RULES = (ON, OFF, NONE)
# who set a value of the rules: a level, never an account
SET_BY = ("code", "deployment", "administrator", "study")
LEVELS = ("study", "installation")  # for whom the rows were declared not people
OVERRIDE_RULES = ("scan_hit", "typed_equal", "cap")
PLACE_MAX = 200
# A place in a member, as an override names it: the member, then what leads to the text, by the format's own names
# and by positions ("protocol.json: body > mask_families > name 3 > description"), the shape the finder of rare
# values gives its places. Never a name the user or the data gave: such a name is said by its position. The list
# of the format's names is completed with the members' inner shapes, before the first package is written.
PLACE_NAMES = frozenset(
    {
        "body",
        "calculations",
        "column",
        "derived_variables",
        "description",
        "expression",
        "follow_up",
        "formula",
        "inverses",
        "mask_families",
        "note",
        "options",
        "outcome",
        "record",
        "results",
        "roles",
        "source",
        "spec",
        "title",
        "units",
    }
)
PLACE_NUMBERED = ("name", "item", "result", "calculation")
_POSITION = re.compile(r"[1-9][0-9]{0,8}")
POLICY_KEYS = (
    "rules",
    "version",
    "k",
    "k_asked",
    "k_default",
    "floor",
    "floor_set_by",
    "cap",
    "cap_set_by",
    "followed_to_the_event",
    "rows_not_people",
    "rows_word",
    "overrides",
)
_WORDS = (ROWS_DEFAULT, *ROWS_WORDS)


def _is_k(value: Any) -> TypeGuard[int]:
    return _is_count(value) and value >= FLOOR_K


def _is_name(value: Any, most: int) -> bool:
    """A name or a place: a text of 1 to ``most`` characters, not blank, on one line, without a character no
    package may hold."""
    return (
        isinstance(value, str)
        and 0 < len(value) <= most
        and bool(value.strip())
        and "\n" not in value
        and not hidden(value)
    )


def is_place(value: Any) -> bool:
    """Whether ``value`` names a place in a JSON member as the format allows: ASCII, at most ``PLACE_MAX``
    characters, the member's name, then ": " and segments joined by " > ", each one of the format's own names
    (``PLACE_NAMES``) or a position ("name 3", "item 2", "result 4", "calculation 1"). Nothing else: a place holds
    no value, no column's name and no words ("criterion age == 97 (...)" was read as one)."""
    if not isinstance(value, str) or not value.isascii() or not 0 < len(value) <= PLACE_MAX:
        return False
    member, colon, path = value.partition(": ")
    if member not in FORMATS:
        return False
    if not colon:
        return True  # the member as a whole
    for segment in path.split(" > "):
        word, space, position = segment.partition(" ")
        if space:
            if word not in PLACE_NUMBERED or not _POSITION.fullmatch(position):
                return False
        elif word not in PLACE_NAMES:
            return False
    return True


@dataclass(frozen=True)
class Policy:
    """
    Under which privacy rules a package was made, as ``package.json`` holds it (see the module's words).

    ``rules`` ``on``: ``version`` of the rules, the ``k`` applied, the k the study asked (``k_asked``, None when it
    asked none), ``k_default``, the installation's ``floor`` and who set it, the ``cap`` of "followed to the event"
    (one value for the package) and who set it, the outcomes declared followed to the event as pairs (outcome,
    override of the cap) sorted by outcome, the word for the rows, and the ``overrides`` kept at export as pairs
    (rule, place). k is the larger of the asked k (or the default) and the floor. ``rows_not_people`` is None: a
    package made under the rules names no declaration that did not apply.

    ``rules`` ``off``: ``version``, ``rows_not_people`` as (reason, level) and the word; nothing was withheld, so
    no k, no cap and no declaration. ``rules`` ``none``: a package that holds the data; only the word, and
    ``rows_not_people`` when the study's rows are declared not people (None otherwise).

    ValueError, in words that quote nothing, for values that no package holds.
    """

    rules: str = ON
    version: int | None = None
    k: int | None = None
    k_asked: int | None = None
    k_default: int | None = None
    floor: int | None = None
    floor_set_by: str | None = None
    cap: float | None = None
    cap_set_by: str | None = None
    followed_to_the_event: tuple[tuple[str, bool], ...] = ()
    rows_not_people: tuple[str, str] | None = None
    rows_word: str = ROWS_DEFAULT
    overrides: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        problems = self._problems()
        if problems:
            raise ValueError(problems[0])

    def _problems(self) -> list[str]:
        """Why these values are no policy a package holds ([] when they are one): the reader's reasons."""
        if self.rules not in RULES:
            return [LATER_POLICY_VALUE]
        if self.rows_word not in _WORDS:
            return [LATER_POLICY_VALUE if isinstance(self.rows_word, str) else NOT_A_POLICY]
        numbers = (self.k, self.k_asked, self.k_default, self.floor, self.cap)
        setters = (self.floor_set_by, self.cap_set_by)
        if self.rules == ON:
            if not (_is_count(self.version) and self.version >= 1):
                return [NOT_A_POLICY]
            if not (_is_k(self.k) and _is_k(self.k_default) and _is_k(self.floor)):
                return [NOT_A_POLICY]
            if self.k_asked is not None and not _is_k(self.k_asked):
                return [NOT_A_POLICY]
            cap = self.cap
            # a share, written as one (a float), and never above the code's: the cap is lowered, never raised
            if not isinstance(cap, float) or not 0 < cap <= MOST_WITHOUT_THE_EVENT:
                return [NOT_A_POLICY]
            for setter in setters:
                if not isinstance(setter, str):
                    return [NOT_A_POLICY]
                if setter not in SET_BY:
                    return [LATER_POLICY_VALUE]
            if self.rows_not_people is not None:
                return [NOT_A_POLICY]
            for outcome, beyond in self.followed_to_the_event:
                if not _is_name(outcome, OUTCOME_MAX) or not isinstance(beyond, bool):
                    return [NOT_A_POLICY]
            outcomes = [outcome for outcome, _ in self.followed_to_the_event]
            if outcomes != sorted(set(outcomes)):  # each once, in the one order
                return [NOT_A_POLICY]
            for rule, place in self.overrides:
                if not isinstance(rule, str) or not is_place(place):
                    return [NOT_A_POLICY]
                if rule not in OVERRIDE_RULES:
                    return [LATER_POLICY_VALUE]
            if list(self.overrides) != sorted(set(self.overrides)):  # each once, in the one order
                return [NOT_A_POLICY]
            asked = self.k_default if self.k_asked is None else self.k_asked
            if self.k != max(asked, self.floor):
                return [K_NOT_THE_LARGER]
            return []
        # nothing was withheld: no k, no cap, no declaration and no override
        if any(number is not None for number in numbers) or any(s is not None for s in setters):
            return [NOT_A_POLICY]
        if self.followed_to_the_event or self.overrides:
            return [NOT_A_POLICY]
        declared = self.rows_not_people
        if self.rules == NONE:
            # a package that holds the data states no rule; of a study whose rows are declared not people, it holds
            # that declaration in the place of its paragraph on personal data
            if self.version is not None:
                return [NOT_A_POLICY]
            if declared is None:
                return []
        elif not (_is_count(self.version) and self.version >= 1):
            return [NOT_A_POLICY]
        if (
            not isinstance(declared, tuple)
            or len(declared) != 2
            or not all(isinstance(part, str) for part in declared)
        ):
            return [NOT_A_POLICY]
        if declared[0] not in NOT_PEOPLE_REASONS or declared[1] not in LEVELS:
            return [LATER_POLICY_VALUE]
        return []

    def to_dict(self) -> dict[str, Any]:
        """The block as ``package.json`` holds it: every key, always."""
        declared = self.rows_not_people
        return {
            "rules": self.rules,
            "version": self.version,
            "k": self.k,
            "k_asked": self.k_asked,
            "k_default": self.k_default,
            "floor": self.floor,
            "floor_set_by": self.floor_set_by,
            "cap": self.cap,
            "cap_set_by": self.cap_set_by,
            "followed_to_the_event": [
                {"outcome": outcome, "beyond_the_cap": beyond}
                for outcome, beyond in self.followed_to_the_event
            ],
            "rows_not_people": None
            if declared is None
            else {"reason": declared[0], "level": declared[1]},
            "rows_word": self.rows_word,
            "overrides": [{"rule": rule, "place": place} for rule, place in self.overrides],
        }

    @classmethod
    def from_dict(cls, value: Any) -> tuple[Policy | None, list[str]]:
        """
        The policy a block holds, or None with why it is refused. Closed: every key of ``POLICY_KEYS`` and no
        other (a key more is a later version's, counted), every closed value among those this version knows, and
        the block written as :meth:`to_dict` writes it (a list in another order is another writing).
        """
        if not isinstance(value, Mapping):
            return None, [NOT_A_POLICY]
        later = len(set(value) - set(POLICY_KEYS))
        if later:
            return None, [later_keys(later)]
        if set(value) != set(POLICY_KEYS):
            return None, [NOT_A_POLICY]
        followed, overrides, declared = (
            value["followed_to_the_event"],
            value["overrides"],
            value["rows_not_people"],
        )
        if not isinstance(followed, list) or not isinstance(overrides, list):
            return None, [NOT_A_POLICY]
        pairs: list[tuple[Any, Any]] = []
        for entry in followed:
            if not isinstance(entry, Mapping) or set(entry) != {"outcome", "beyond_the_cap"}:
                return None, [NOT_A_POLICY]
            pairs.append((entry["outcome"], entry["beyond_the_cap"]))
        kept: list[tuple[Any, Any]] = []
        for entry in overrides:
            if not isinstance(entry, Mapping) or set(entry) != {"rule", "place"}:
                return None, [NOT_A_POLICY]
            kept.append((entry["rule"], entry["place"]))
        if declared is not None:
            if not isinstance(declared, Mapping) or set(declared) != {"reason", "level"}:
                return None, [NOT_A_POLICY]
            declared = (declared["reason"], declared["level"])
        # texts that are not texts never reach the comparisons of the checks below
        texts = [o for o, _ in pairs] + [part for pair in kept for part in pair]
        if not all(isinstance(text, str) for text in texts):
            return None, [NOT_A_POLICY]
        if not isinstance(value["rules"], str):
            return None, [NOT_A_POLICY]
        try:
            policy = cls(
                rules=value["rules"],
                version=value["version"],
                k=value["k"],
                k_asked=value["k_asked"],
                k_default=value["k_default"],
                floor=value["floor"],
                floor_set_by=value["floor_set_by"],
                cap=value["cap"],
                cap_set_by=value["cap_set_by"],
                followed_to_the_event=tuple(pairs),
                rows_not_people=declared,
                rows_word=value["rows_word"],
                overrides=tuple(kept),
            )
        except ValueError as refused:
            return None, [str(refused)]
        except TypeError:  # a value that cannot be compared or hashed: a list where a text stands
            return None, [NOT_A_POLICY]
        return policy, []


__all__ = [
    "CAPS",
    "CUT",
    "DEPTH",
    "EXTENSION",
    "FORMATS",
    "INVISIBLE",
    "KEY_TWICE",
    "K_NOT_THE_LARGER",
    "LATER_POLICY_VALUE",
    "LEVELS",
    "LISTED",
    "LOCK",
    "MANIFEST",
    "MEMBERS",
    "NONE",
    "NOT_A_MANIFEST",
    "NOT_A_MEMBER",
    "NOT_A_NUMBER",
    "NOT_A_POLICY",
    "NOT_JSON",
    "OFF",
    "ON",
    "OPTIONAL",
    "ORDERED",
    "OVERRIDE_RULES",
    "PACKAGE",
    "PACKAGE_CAP",
    "PLACE_MAX",
    "PLACE_NAMES",
    "PLACE_NUMBERED",
    "POLICY_KEYS",
    "PROTOCOL",
    "README",
    "REPLACED",
    "REPRODUCTION",
    "RESULTS",
    "RULES",
    "RULES_OFF",
    "SCHEMAS",
    "SET_BY",
    "SHAREABLE",
    "STUDY_ID",
    "TIERS",
    "TOO_LARGE",
    "Policy",
    "checksum",
    "dumps",
    "hidden",
    "hidden_reason",
    "is_place",
    "later_keys",
    "later_schema",
    "loads",
    "manifest_of",
    "not_matching",
    "package_hash",
    "package_name",
    "read_envelope",
    "read_manifest",
    "shown",
]
