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

**No text of a written member holds a control or a direction character, nor one that shows nothing**
(``core.privacy.hidden``, the rule of a study's own names):
such a character can make a name read as another. The writer refuses to write one and the reader refuses a member
that holds one; a line feed inside a text is the one exception (where a text is shown, it must not pass for a
line of its own). The members that carry the study's own texts in a package that holds the data are the
exception (see below). :func:`shown` is for what does not come from a member (a file's name, an argument of a
command), and for every key and text of a carried member.

**A package that holds the data** (the tier ``reproduction``) has other members (:func:`members_of`): a clear
notice ``NOTICE.txt`` (one of two fixed texts, read by its exact bytes: :func:`read_notice`), then, encrypted in
the zip, ``package.json``, ``protocol.json`` (the saved revision, whole), ``results.json`` (exact), ``trace.json``,
``profiles.json``, ``README.md``, ``uv.lock`` and the data (``source.<ext>`` or ``version.parquet``, then
``merge_NN.parquet`` and ``snapshot_NN.parquet``: :func:`data_role`), and the clear manifest last. Its hash is
still the SHA-256 of the manifest, which lists the checksums of the members before encryption.

**Two writings in that tier.** ``package.json`` and ``README.md`` are *written* as above, under the rule for a
text. The members that hold the study's own texts (``protocol.json``, ``results.json``, ``trace.json``,
``profiles.json``) are *carried*: the same writer in ASCII, every character outside ASCII as an escape, and no rule
for a text. A sealed trace can hold a tab or a direction mark in a value of the data, and a study must not be kept
from travelling for it; the bytes hold nothing that hides or turns in a viewer. The carried reader refuses a byte
that is not ASCII, a lone surrogate (which no sealed trace can hold), and all that the strict reader refuses but
the rule for a text: a carried key or text may hold any other character, so it reaches a screen only through
:func:`shown`. :func:`read_member` is the one reader of a member: it knows which writing each one has.

**The anchor.** A clear line of the manifest gives the SHA-256 of each member. A member whose every value could be
guessed (with a shareable package of the same study at hand) would be given away by trying them; so
``package.json`` holds an anchor, the SHA-256 of a fixed prefix and of the bytes of the data member the trace
starts from (:func:`anchor_of`), which nobody computes without the data.

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
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeGuard, get_args

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

# ----------------------------------------------------------------------------- the package that holds the data
# The names above (MEMBERS, LISTED, FORMATS, CAPS, PACKAGE_CAP) are those of a package without the data, in its two
# tiers. What follows is per tier.

NOTICE, TRACE, PROFILES = "NOTICE.txt", "trace.json", "profiles.json"
_WITH_DATA = (NOTICE, PACKAGE, PROTOCOL, RESULTS, TRACE, PROFILES, README, LOCK, MANIFEST)
_FIXED = {SHAREABLE: MEMBERS, RULES_OFF: MEMBERS, REPRODUCTION: _WITH_DATA}
# What the zip does not encrypt. A package without the data is not encrypted; one that holds them shows its notice
# and its manifest, so that its hash is read before any password.
_CLEAR = {SHAREABLE: MEMBERS, RULES_OFF: MEMBERS, REPRODUCTION: (NOTICE, MANIFEST)}
# One format name per shape: a member that is not the same in two tiers has a name in each, so that one tier's
# schema moves alone, and a member taken out of a package of one tier never reads as the other's.
_FORMATS = {
    SHAREABLE: FORMATS,
    RULES_OFF: FORMATS,
    REPRODUCTION: {
        PACKAGE: "cohortstudio.study-package-with-data",
        PROTOCOL: "cohortstudio.study-definition",  # the revision as saved, whole: its checksum can be checked
        RESULTS: "cohortstudio.exact-results",
        TRACE: "cohortstudio.trace",
        PROFILES: "cohortstudio.column-profiles",
    },
}
# Each format's schema: the one table that is read. A format's schema moves when a key or a closed value is added
# to it; the index's (package.json's) also when the set of members of its tier changes. The notice of a package
# that holds the data names its index's schema.
FORMAT_SCHEMAS = {
    "cohortstudio.study-package": 1,
    "cohortstudio.shared-definition": 1,
    "cohortstudio.study-results": 1,
    "cohortstudio.study-package-with-data": 1,
    "cohortstudio.study-definition": 1,
    "cohortstudio.exact-results": 1,
    "cohortstudio.trace": 1,
    "cohortstudio.column-profiles": 1,
}
# The zip that holds the members (manage/package_zip.py) is the container, frozen across the schemas: its layout (the
# members one after the other from the file's first byte, each local header saying what the directory says, the
# directory right after them and its end record last, without a comment and without ZIP64), at most 1,024 entries
# named in ASCII by at most 64 characters, no folder, no name twice, and the member that says whether a later version
# made the package (NOTICE.txt with the data, package.json without) stored and clear under its name, within this
# version's cap of it, and package.json JSON that this version's strict reader reads. A reader reads these before
# any schema: a later schema changes the members and what they hold, never these.
# The members that hold the study's own texts, written with escapes and read without the rule for a text
_CARRIED = {SHAREABLE: (), RULES_OFF: (), REPRODUCTION: (PROTOCOL, RESULTS, TRACE, PROFILES)}
_ORDERED = {SHAREABLE: ORDERED, RULES_OFF: ORDERED, REPRODUCTION: {PROTOCOL: ("revision",)}}

# What a member of a package that holds the data may hold, before encryption. The JSON members are parsed in
# memory, where they take several times their size: 64 MB of JSON bounds what a reader parses, whatever a received
# file declares (to be set against measured sizes before the first package is released). A zip writer adds a ZIP64
# field to a member larger than ``zipfile.ZIP64_LIMIT`` (with a 5% margin), and to one whose header starts beyond
# it: a data member and the whole stay below that bound, since the format holds no extra field but AES's.
_GB = 1024 * _MB
_DATA_CAPS = {
    NOTICE: 16 * 1024,
    PACKAGE: 4 * _MB,  # every step's hash
    PROTOCOL: 64 * _MB,  # the rules that list rows travel
    RESULTS: 64 * _MB,  # exact, with the estimates of each imputed dataset
    TRACE: 64 * _MB,
    PROFILES: 64 * _MB,
    README: _MB,
    LOCK: 4 * _MB,
    MANIFEST: 64 * 1024,
}
DATA_MEMBER_CAP = _GB  # one file of data, before encryption
_PACKAGE_CAPS = {SHAREABLE: PACKAGE_CAP, RULES_OFF: PACKAGE_CAP, REPRODUCTION: 3 * _GB // 2}
DATA_MEMBERS_MOST = 99  # merged files, and recorded tables: their two-digit numbers

# The roles of a file a trace reads (the study package's list of files, the replay's messages): stored ids, never
# renamed. ``ROLE_WORDS`` says them in a sentence.
FileRole = Literal["source", "saved_version", "replayed_state", "enrichment", "snapshot"]
FILE_ROLES: tuple[str, ...] = get_args(FileRole)
ROLE_WORDS: dict[str, str] = {
    "source": "the source file",
    "saved_version": "a saved version",
    "replayed_state": "a state replayed from another trace",
    "enrichment": "a merged file",
    "snapshot": "a step's recorded table",
}
# The files a source may be (ingest.loaders.SUPPORTED_EXTENSIONS, which this module does not import: a test
# compares them), in lower case and without their dot. The bytes of the first travel as they were loaded; a
# workbook of the second can hold code (a macro project), which no package holds: such a source travels only as
# the table the study read (``source.parquet``, ``as_read``).
SOURCE_EXTENSIONS = ("csv", "parquet", "tsv", "txt", "xlsx")
TABLE_ONLY_EXTENSIONS = ("xls", "xlsm")
_SOURCE = re.compile(r"source\.(?:" + "|".join(SOURCE_EXTENSIONS) + ")")
_VERSION = "version.parquet"
_NUMBERED = re.compile(r"(merge|snapshot)_(0[1-9]|[1-9][0-9])\.parquet")
_NUMBERED_ROLE = {"merge": "enrichment", "snapshot": "snapshot"}


def _tier(tier: str) -> str:
    if tier not in TIERS:
        raise ValueError("unknown tier")
    return tier


def members_of(tier: str) -> tuple[str, ...]:
    """The members every package of ``tier`` is made of, in the zip's order, its manifest last (``uv.lock`` only
    when the installation has one). A package that holds the data has its data members between them and the
    manifest (:func:`data_role`)."""
    return _FIXED[_tier(tier)]


def clear_members(tier: str) -> tuple[str, ...]:
    """The members the zip does not encrypt: all of a package without the data, the notice and the manifest of one
    that holds them."""
    return _CLEAR[_tier(tier)]


def holds_data(tier: str) -> bool:
    return _tier(tier) == REPRODUCTION


def formats_of(tier: str) -> Mapping[str, str]:
    """The JSON members of a package of ``tier``, each with its format's name."""
    return _FORMATS[_tier(tier)]


def is_carried(member: str, tier: str) -> bool:
    """Whether ``member`` holds the study's own texts in this tier (see the module's words on the two writings)."""
    return member in _CARRIED[_tier(tier)]


def cap_of(member: str, tier: str = SHAREABLE) -> int:
    """The most bytes ``member`` may hold in a package of ``tier``, before encryption. ValueError for a name that
    is no member of such a package."""
    if holds_data(tier):
        if member in _DATA_CAPS:
            return _DATA_CAPS[member]
        if data_role(member) is not None:
            return DATA_MEMBER_CAP
    elif member in CAPS:
        return CAPS[member]
    raise ValueError("not a member of a study package of this tier")


def package_cap(tier: str) -> int:
    """The most bytes the members of a package of ``tier`` may hold together, before encryption."""
    return _PACKAGE_CAPS[_tier(tier)]


def source_member(extension: str) -> str:
    """The member that holds the source's bytes: ``source.<ext>``, the extension in lower case. ValueError for a
    file whose bytes never travel (a workbook that can hold code: its table travels instead, as
    ``source.parquet``), and for a file the loader does not read."""
    bare = str(extension).lower().lstrip(".")
    if bare in TABLE_ONLY_EXTENSIONS:
        raise ValueError("a workbook that can hold code travels as the table the study read")
    if bare not in SOURCE_EXTENSIONS:
        raise ValueError("not a file a study is loaded from")
    return f"source.{bare}"


def version_member() -> str:
    """The member that holds the table of the saved version a trace starts from."""
    return _VERSION


def _numbered(kind: str, number: int) -> str:
    if not _is_count(number) or not 1 <= number <= DATA_MEMBERS_MOST:
        raise ValueError(f"a package holds 1 to {DATA_MEMBERS_MOST} of them")
    return f"{kind}_{number:02d}.parquet"


def merge_member(number: int) -> str:
    """The member of the ``number``-th merge step's file (1 is the first merge step): the table as read."""
    return _numbered("merge", number)


def snapshot_member(number: int) -> str:
    """The member of the ``number``-th recorded table that the replay reads in place of a step."""
    return _numbered("snapshot", number)


def data_role(name: object) -> str | None:
    """The role a data member's name says (``source``, ``saved_version``, ``enrichment``, ``snapshot``); None for a
    name that is not a data member's. The names are the format's own, in this exact case: none comes from a file
    of the study."""
    if not isinstance(name, str):
        return None
    if _SOURCE.fullmatch(name):
        return "source"
    if name == _VERSION:
        return "saved_version"
    numbered = _NUMBERED.fullmatch(name)
    return _NUMBERED_ROLE[numbered[1]] if numbered else None


NOT_THE_DATA = "does not hold the data members of a study package: one source, and files numbered without a gap"


def data_members_problem(names: Iterable[str]) -> str:
    """Why the data members named are not those of one package ("" when they are): exactly one source or saved
    version, and the merged files and the recorded tables each numbered from 01 without a gap."""
    found = [name for name in names if data_role(name) is not None]
    starts = [name for name in found if data_role(name) in ("source", "saved_version")]
    if len(starts) != 1 or len(set(found)) != len(found):
        return NOT_THE_DATA
    for kind in _NUMBERED_ROLE:
        numbers = sorted(int(m[2]) for m in map(_NUMBERED.fullmatch, found) if m and m[1] == kind)
        if numbers != list(range(1, len(numbers) + 1)):
            return NOT_THE_DATA
    return ""


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
# the same marks on a terminal, where a code page may lack both (a cp1252 console writes U+2026 as a byte that is
# not UTF-8, and has no U+FFFD)
REPLACED_ON_A_TERMINAL, CUT_ON_A_TERMINAL = "?", "..."


_BACKSLASH = chr(92)


def _on_a_terminal(char: str) -> str:
    """A character as a terminal shows it: itself when it is printable ASCII, ``?`` when no package may hold it
    (``?`` means only that), a backslash doubled (so that an escape is never a text's own characters), else the
    escape an ASCII stream writes for it (``backslashreplace``)."""
    if unseen(char):
        return REPLACED_ON_A_TERMINAL
    if char == _BACKSLASH:
        return _BACKSLASH * 2
    code = ord(char)
    if code < 0x7F:  # printable ASCII: the controls and DEL are hidden, above
        return char
    letter, width = ("x", 2) if code < 0x100 else ("u", 4) if code < 0x10000 else ("U", 8)
    return f"{_BACKSLASH}{letter}{code:0{width}x}"


def shown(text: object, limit: int = 120, *, terminal: bool = False) -> str:
    """
    A text that does not come from a member (a file's name, an argument of a command), or a key or a text of a
    carried member, as a page may show it: on one line, each character no package may hold (a line feed too)
    replaced by U+FFFD, cut at ``limit`` characters with U+2026.

    On a terminal (``terminal``): ASCII only, at most ``limit`` characters whatever the text holds, so that a line
    keeps its width on any console. A character no package may hold is ``?``; another character outside ASCII is
    the escape an ASCII stream would write for it ("é" is ``\\xe9``), and a backslash is doubled, each counted in
    the limit and never cut in two; the mark of a cut is ``...``. A literal ``?`` still reads as the mark: never
    show two texts side by side through it to compare them.
    """
    if not terminal:
        cleaned = "".join(REPLACED if unseen(char) else char for char in str(text))
        return cleaned if len(cleaned) <= limit else cleaned[: max(0, limit - len(CUT))] + CUT
    pieces = [_on_a_terminal(char) for char in str(text)]
    if sum(len(piece) for piece in pieces) <= limit:
        return "".join(pieces)
    room, kept = max(0, limit - len(CUT_ON_A_TERMINAL)), ""
    for piece in pieces:
        if len(kept) + len(piece) > room:
            break
        kept += piece
    return (kept + CUT_ON_A_TERMINAL)[: max(0, limit)]


# ----------------------------------------------------------------------------- the one writer of a JSON member


def _surrogates(text: str) -> int:
    """The lone surrogates in ``text``: a value UTF-8 cannot hold, which no sealed trace holds (its hash is taken
    over UTF-8)."""
    return sum(0xD800 <= ord(char) <= 0xDFFF for char in text)


def _not_allowed(text: str, carried: bool) -> int:
    """The characters of ``text`` a member may not hold: those of the rule for a text in a written member, a lone
    surrogate in a carried one."""
    return _surrogates(text) if carried else hidden(text)


def _written(value: Any, keep: bool, carried: bool = False) -> Any:
    """``value`` as the writer writes it: mappings with their keys sorted (in the order given under ``keep``),
    lists in their order. ValueError for what no member holds; a carried member's texts are written whatever they
    hold (the writer escapes them) but a lone surrogate."""
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("a key of a member is a text")
        found = sum(_not_allowed(key, carried) for key in value)
        if found:
            raise ValueError("a key " + hidden_reason(found))
        keys = list(value) if keep else sorted(value)
        return {key: _written(value[key], keep, carried) for key in keys}
    if isinstance(value, (list, tuple)):
        return [_written(item, keep, carried) for item in value]
    if isinstance(value, str):
        found = _not_allowed(value, carried)
        if found:
            raise ValueError("a text " + hidden_reason(found))
        return value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("a number of a member is finite")
        return value
    raise ValueError("a member holds texts, numbers, yes or no, nothing, lists and mappings only")


def dumps(member: str, payload: Mapping[str, Any], tier: str = SHAREABLE) -> bytes:
    """
    The bytes of a JSON member of a package of ``tier``: ``payload`` under the member's envelope (``format``,
    ``schema``), in the one writing (see the module's words). ValueError for a payload that names the envelope's
    keys, or holds what no member may: a number that is not finite, a key that is not a text, a value of another
    type, and in a written member a text with a control or a direction character. A carried member is written in
    ASCII, every other character as an escape. The message counts and never quotes.
    """
    formats = formats_of(tier)
    if member not in formats:
        raise ValueError("not a JSON member of a study package")
    if "format" in payload or "schema" in payload:
        raise ValueError("the envelope's keys are the writer's")
    carried = is_carried(member, tier)
    kept = _ORDERED[tier].get(member, ())
    whole: dict[str, Any] = {"format": formats[member], "schema": FORMAT_SCHEMAS[formats[member]]}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError("a key of a member is a text")
        if hidden(key):  # the member's own keys are the format's, in both writings
            raise ValueError("a key " + hidden_reason(hidden(key)))
        whole[key] = _written(value, keep=key in kept, carried=carried)
    text = json.dumps(
        {key: whole[key] for key in sorted(whole)}, indent=2, ensure_ascii=carried, allow_nan=False
    )
    return (text + "\n").encode("ascii" if carried else "utf-8")


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


def _checked(value: Any, carried: bool = False) -> int:
    """How many characters a member may not hold stand in ``value``'s keys and texts (those of the rule for a text;
    in a ``carried`` member, the lone surrogates only); _Refused for a number that is not finite (``1e999`` is read
    as one without a word) and for a nest of more than ``DEPTH`` levels (the reader's own limit refused only far
    deeper, and by the interpreter's). An explicit stack: no recursion of its own."""
    count, stack = 0, [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, (dict, list)):
            if depth > DEPTH:
                raise _Refused(NOT_JSON)
            if isinstance(item, dict):
                count += sum(_not_allowed(key, carried) for key in item)
                stack.extend((inner, depth + 1) for inner in item.values())
            else:
                stack.extend((inner, depth + 1) for inner in item)
        elif isinstance(item, str):
            count += _not_allowed(item, carried)
        elif isinstance(item, float) and not math.isfinite(item):
            raise _Refused(NOT_A_NUMBER)
    return count


def loads(data: bytes, cap: int, *, carried: bool = False) -> tuple[Any | None, list[str]]:
    """
    A JSON member read strictly: its value, or None with why it is refused. Refused: more than ``cap`` bytes, a
    byte order mark, bytes that are not UTF-8, a key written twice, NaN or an infinite number, a nest of more than
    ``DEPTH`` levels, a number too long for the reader, and a key or a text holding a character no package may
    hold. A ``carried`` member (see the module's words) is refused for a byte that is not ASCII and for a lone
    surrogate instead, and its texts may hold any other character. The reasons quote nothing of the member.
    Nothing here says what the value means: each member has its closed reader (:func:`read_member` reads a member
    with the writing it has).
    """
    if len(data) > cap:
        return None, [TOO_LARGE]
    if data.startswith(b"\xef\xbb\xbf") or (carried and not data.isascii()):
        return None, [NOT_JSON]
    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_no_constant)
        count = _checked(value, carried)
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


def read_envelope(value: Any, member: str, tier: str = SHAREABLE) -> list[str]:
    """Why ``value`` is not the member it is named as in a package of ``tier``, or was written by a later version;
    [] when it is one this version reads. A schema is a whole number from 1: a later one is said with its number,
    which is bounded."""
    formats = formats_of(tier)
    if member not in formats:
        raise ValueError("not a JSON member of a study package")
    if not isinstance(value, dict) or value.get("format") != formats[member]:
        return [NOT_A_MEMBER]
    schema, read = value.get("schema"), FORMAT_SCHEMAS[formats[member]]
    if not _is_count(schema) or not 1 <= schema <= _SCHEMA_MOST:
        return [NOT_A_MEMBER]
    if schema > read:
        return [later_schema(schema, read)]
    return []


NOT_A_TEXT = "cannot be read: it is not the text this version writes"


def read_member(member: str, data: bytes, tier: str) -> tuple[Any | None, list[str]]:
    """
    A member of a package of ``tier`` read by its one reader, with the writing it has: a JSON member strictly
    (carried or written, under its cap), then its envelope; ``README.md`` as a text under the rule for a text
    (UTF-8 without a byte order mark); ``uv.lock`` as UTF-8 under its cap. The notice and the manifest have their own
    readers (:func:`read_notice`, :func:`read_manifest`), and a data member is not read here: ValueError for a name
    that is none of these members of the tier.
    """
    if member in formats_of(tier):
        value, reasons = loads(data, cap_of(member, tier), carried=is_carried(member, tier))
        if value is None:
            return None, reasons
        reasons = read_envelope(value, member, tier)
        return (None, reasons) if reasons else (value, [])
    if member not in (README, LOCK):
        raise ValueError("not a member read as a text")
    if len(data) > cap_of(member, tier):
        return None, [TOO_LARGE]
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None, [NOT_A_TEXT]
    if text.startswith(chr(0xFEFF)):
        return None, [NOT_A_TEXT]
    found = hidden(text) if member == README else 0  # the lock file is uv's, never shown
    return (None, [hidden_reason(found)]) if found else (text, [])


# ----------------------------------------------------------------------------- the manifest and the package's hash


def checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


ANCHOR_PREFIX = b"CohortStudio study package anchor\n"


def anchor_of(chunks: Iterable[bytes]) -> str:
    """The anchor of a package that holds the data (see the module's words): the SHA-256 of ``ANCHOR_PREFIX`` and of
    the bytes of the data member the trace starts from, given in chunks (a writer and a reader stream them). It is
    not the member's checksum, which the clear manifest gives."""
    digest = hashlib.sha256(ANCHOR_PREFIX)
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _fixed_listed(tier: str) -> tuple[str, ...]:
    """The members of ``tier`` a manifest lists by their fixed names: every one but the manifest."""
    return tuple(name for name in members_of(tier) if name != MANIFEST)


def _is_listed(name: str, tier: str) -> bool:
    return name in _fixed_listed(tier) or (holds_data(tier) and data_role(name) is not None)


def _listing_problem(names: Iterable[str], tier: str) -> str:
    """Why these names are not what the manifest of a package of ``tier`` lists ("" when they are)."""
    names = list(names)
    if not all(isinstance(name, str) and _is_listed(name, tier) for name in names):
        return "the manifest lists the package's members, and not itself"
    missing = [name for name in _fixed_listed(tier) if name not in OPTIONAL and name not in names]
    if missing:
        return "a package holds " + ", ".join(missing)
    if holds_data(tier):
        return data_members_problem(names)
    return ""


def manifest_of(members: Mapping[str, bytes], tier: str = SHAREABLE) -> bytes:
    """
    The manifest of a package's other members: one line ``<sha256>  <name>`` each, sorted by name, ASCII, line
    feeds. A package that holds the data lists its notice and its data members too. ValueError for a name that is
    no member of ``tier``, for the manifest itself, without a member every package of the tier holds, or for data
    members that are not those of one package.
    """
    return manifest_from_checksums({name: checksum(data) for name, data in members.items()}, tier)


def manifest_from_checksums(checksums: Mapping[str, str], tier: str = SHAREABLE) -> bytes:
    """:func:`manifest_of` for members whose checksums are known already (a writer that streams them): the same
    lines, the same refusals, and a checksum that is not 64 lower-case hexadecimal characters refused too."""
    problem = _listing_problem(checksums, tier)
    if problem:
        raise ValueError(problem)
    if not all(_SHA256.fullmatch(str(value)) for value in checksums.values()):
        raise ValueError("a checksum of the manifest is a SHA-256 in hexadecimal")
    return "".join(f"{checksums[name]}  {name}\n" for name in sorted(checksums)).encode("ascii")


def read_manifest(data: bytes, tier: str = SHAREABLE) -> tuple[dict[str, str] | None, list[str]]:
    """
    The checksums a manifest lists, by member, or None with why it is refused. Strict: at most the manifest's cap
    of bytes of ASCII, each line ``<64 lower-case hex>  <name>`` ended by a line feed, each name a member of a
    package of ``tier`` other than the manifest, each once, sorted, every member that a package of the tier always
    holds, and (with the data) the data members of one package.
    """
    if len(data) > cap_of(MANIFEST, tier) or not data.endswith(b"\n"):
        return None, [NOT_A_MANIFEST]
    try:
        lines = data.decode("ascii").split("\n")[:-1]
    except UnicodeDecodeError:
        return None, [NOT_A_MANIFEST]
    listed: dict[str, str] = {}
    for line in lines:
        match = _LINE.fullmatch(line)
        if match is None or not _is_listed(match[2], tier) or match[2] in listed:
            return None, [NOT_A_MANIFEST]
        listed[match[2]] = match[1]
    if list(listed) != sorted(listed) or _listing_problem(listed, tier):
        return None, [NOT_A_MANIFEST]
    return listed, []


def in_zip_order(names: Iterable[str], tier: str) -> tuple[str, ...]:
    """
    The members named, in the order the zip of a package of ``tier`` holds them: the format's members in their
    order, the data members (with the data) before the manifest, the start of the trace first, then the merged
    files and the recorded tables by their numbers. ValueError for a name that is no member of the tier, or for
    data members that are not those of one package.
    """
    names = list(dict.fromkeys(names))
    problem = _listing_problem([name for name in names if name != MANIFEST], tier)
    if problem:
        raise ValueError(problem)
    if MANIFEST not in names:
        raise ValueError("a package holds " + MANIFEST)
    rank = {name: place for place, name in enumerate(members_of(tier))}
    kinds = {"source": 0, "saved_version": 0, "enrichment": 1, "snapshot": 2}

    def place(name: str) -> tuple[int, int, str]:
        role = data_role(name) if name not in rank else None
        if role is None:
            return (rank[name], 0, name)
        return (rank[MANIFEST] - 1, 1 + kinds[role], name)  # after the members before the manifest

    return tuple(sorted(names, key=place))


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


# ----------------------------------------------------------------------------- the notice of a package with the data

_PACKAGE_SCHEMA = FORMAT_SCHEMAS[_FORMATS[REPRODUCTION][PACKAGE]]
_NOTICE_HEAD = "CohortStudio study package with the data, schema "
NOTICE_PERSONAL_DATA, NOTICE_NOT_PEOPLE = (
    "personal_data",
    "rows_not_people",
)  # the two notices (stored ids)
_ENCRYPTED = (
    "Apart from this notice and its list of checksums, it is encrypted",
    "(AES-256). Its sender was asked to send the password apart from the",
    "package. CohortStudio keeps no copy of the password: if it is lost,",
    "ask the sender for a new package.",
)
# what a recipient does before typing the password; no menu of the application (its words are frozen with the
# schema, the application's are not)
_CHECK_IT_FIRST = (
    "Before you type the password, check that this is the package its",
    "sender made. With CohortStudio installed on your computer, run:",
    "    python -m manage.replay inspect <this file>",
    "It prints the package hash (the SHA-256 of MANIFEST.sha256, not of the",
    "zip file): compare it, or its first 16 characters, with the one the",
    "sender gave you apart from the file. If you use CohortStudio only in a",
    "browser, its handbook (the chapter on reproduction) says how to",
    "install it.",
)


def _notice(*paragraphs: tuple[str, ...]) -> bytes:
    lines = [f"{_NOTICE_HEAD}{_PACKAGE_SCHEMA}"]
    for paragraph in paragraphs:
        lines += ["", *paragraph]
    return ("\n".join(lines) + "\n").encode("ascii")


# The clear notice of a package that holds the data: fixed bytes, the same in every package of a schema (no study,
# no date, no count), read by their exact bytes, so that a clear member is never a place for words of anyone's.
# **Its words belong to its schema**: a reader takes any other bytes for a changed file, so a notice is never
# edited in place; a new wording comes with a new schema, and the readers keep the earlier notices' bytes (a test
# pins each notice's SHA-256).
NOTICES: dict[str, bytes] = {
    NOTICE_PERSONAL_DATA: _notice(
        (
            "This package holds personal data: a study's records about people, and",
            "the trace of every step applied to those records. Records about people",
            "who can be identified are personal data even without their names (GDPR",
            "Art. 4(1) and 4(5), Recital 26); where they concern health, they are a",
            "special category (Art. 9).",
        ),
        _ENCRYPTED,
        (
            "Open it only if a data sharing agreement, a data processing agreement",
            "or another legal basis covers you as its recipient for this study. If",
            "it reached you by mistake, do not open it: delete it and tell its",
            "sender.",
        ),
        _CHECK_IT_FIRST,
    ),
    # for a study whose rows are declared not people, by its author or for its whole installation: read by this
    # version, written from M8b on
    NOTICE_NOT_PEOPLE: _notice(
        (
            "The rows of this study were declared not people; the declaration and",
            "its reason are inside.",
        ),
        _ENCRYPTED,
        (
            "Open it only if its sender meant it for you. If it reached you by",
            "mistake, do not open it: delete it and tell its sender.",
        ),
        _CHECK_IT_FIRST,
    ),
}
NOT_A_NOTICE = "is not the notice of a study package"
NOTICE_NOT_THE_POLICY = "its notice does not say what its policy says"


def notice_for(policy: Policy) -> bytes:
    """The notice of a package made under ``policy``: of personal data, unless its rows are declared not people."""
    return NOTICES[NOTICE_PERSONAL_DATA if policy.rows_not_people is None else NOTICE_NOT_PEOPLE]


def read_notice(data: bytes) -> tuple[str | None, list[str]]:
    """Which notice ``data`` is, or None with why it is refused: anything but one of the fixed texts, except a
    notice that names a later schema in its first line, which says so (made by a later version, not changed)."""
    if len(data) > cap_of(NOTICE, REPRODUCTION):
        return None, [TOO_LARGE]
    for kind, text in NOTICES.items():
        if data == text:
            return kind, []
    first, head = data.split(b"\n", 1)[0], _NOTICE_HEAD.encode("ascii")
    number = first[len(head) :] if first.startswith(head) else b""
    # a whole number as the writer writes one, and bounded before it is read
    written = number.isdigit() and not number.startswith(b"0") and len(number) <= 7
    if written and _PACKAGE_SCHEMA < int(number) <= _SCHEMA_MOST:
        return None, [later_schema(int(number), _PACKAGE_SCHEMA)]
    return None, [NOT_A_NOTICE]


def notice_problem(kind: object, policy: Policy) -> str:
    """Why a package's notice disagrees with its policy ("" when it does not): the notice of personal data for a
    study whose rows are not declared not people, the other one for a study whose rows are, by either level. A kind
    that is no notice's is refused."""
    if not isinstance(kind, str) or kind not in NOTICES:
        return NOT_A_NOTICE
    declared = policy.rows_not_people is not None
    return "" if (kind == NOTICE_NOT_PEOPLE) == declared else NOTICE_NOT_THE_POLICY


# ----------------------------------------------------------------------------- the index of a package with the data
# package.json in the tier ``reproduction``: what the package is of, from what, made with what and how, under which
# policy and to whom. No text the user or the data gave but the study's id and the declaration's two (typed for the
# export, under the rule for a text): no title, no recorded path, no column's name. Every step's hash and the anchor
# are there: the clear manifest gives this member's SHA-256, and a member whose every value could be guessed would
# be given away by trying them.

NOT_AN_INDEX = "its index cannot be read by this version"
LATER_INDEX_VALUE = "written by a later version: its index holds a value this version does not know"


def later_index_keys(count: int) -> str:
    return (
        f"written by a later version: its index holds {count} key{'s' if count != 1 else ''} this version does "
        "not read"
    )


TRACE_SCHEMA = (
    2  # core.lineage.TRACE_SCHEMA, restated: this module does not import it (a test compares them)
)
STARTS_FROM = ("source", "saved_version")  # what the trace starts from: the roles of its first file
CUT_RULES = ("rows_taken", "every_row")  # the rows of a merged file a package holds (stored ids)
# what covers sending the data to the recipient (stored ids): the form offers the first five; the last is the
# declaration that the study's rows are not people (M8b)
BASES = (
    "data_sharing_agreement",
    "data_processing_agreement",
    "joint_controllers",
    "same_organisation",
    "other_basis",
    "rows_not_people",
)
NO_REFERENCE_NEEDED = ("same_organisation", "rows_not_people")
RECIPIENT_MAX, REFERENCE_MAX = 200, 100
# the version of the words the exporter declared on: a record, read whatever its number (the README carries the
# words, and a reworded form must not make older readers refuse newer packages)
DECLARATION_WORDS = 1
TEXT_MAX = 64  # a version, a system's or a machine's name
SEED_MOST = 2**32 - 1  # utils.settings' bound of CMP_RANDOM_SEED (a test compares them)
BYTE_ORDERS = ("little", "big")
# The steps whose recorded table a package holds in place of the step: core.steps' kinds replayed from their
# snapshot (restated; a test compares them), but ``multiple_imputation``. That legacy step adopted one imputed
# dataset as the data: no imputed table travels, and a trace that holds it does not travel with its data.
SNAPSHOT_STEPS = ("external_write", "outlier_handling")
ADOPTED_IMPUTATION = "multiple_imputation"
_COMMIT = re.compile(
    r"[0-9a-f]{40}(?:[0-9a-f]{24})?"
)  # SHA-1, or SHA-256 for a repository that uses it
_FUNCTION = re.compile(r"[a-z][a-z0-9_]{0,63}")  # a step's kind (core.steps' ids)
_LIBRARY = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
# a version: ASCII, no space, no path ("3.14.3", "2.5.0.dev3+g1a2b3c4", "1!2.0")
_SOFTWARE_TEXT = re.compile("[A-Za-z0-9][A-Za-z0-9._+!-]{0," + str(TEXT_MAX - 1) + "}")
# a system's or a machine's name, as the platform gives it ("Windows", "Linux", "AMD64", "x86_64", "aarch64"): a word,
# never a host's name ("JDUPONT-PC", "lab.example.org") nor a path
_PLATFORM_WORD = re.compile("[A-Za-z0-9_]{1,16}")
INDEX_KEYS = ("tier", "study", "software", "trace", "files", "policy", "declaration", "anchor")
_STUDY_KEYS = ("id", "revision", "revision_checksum", "protocol_hash", "rules_version")
_SOFTWARE_KEYS = (
    "cohortstudio",
    "commit",
    "code_sha256",
    "uv_lock_sha256",
    "python",
    "system",
    "machine",
    "byteorder",
    "random_seed",
    "libraries",
)
_TRACE_KEYS = ("schema_version", "head", "starts_from", "steps")
_STEP_KEYS = ("index", "function", "hash")
_FILE_KEYS = ("role", "step", "member", "as_read", "rows", "columns", "cut")
_CUT_KEYS = ("rule", "rows_of_original", "original_sha256")
_DECLARATION_KEYS = ("recipient", "basis", "reference", "words")


def _check(condition: bool, reason: str = NOT_AN_INDEX) -> None:
    if not condition:
        raise ValueError(reason)


def _hex64(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _closed_value(value: Any, known: tuple[str, ...]) -> None:
    """A closed value: refused as a later version's when it is a text this version does not know."""
    _check(isinstance(value, str))
    _check(value in known, LATER_INDEX_VALUE)


def _block(value: Any, keys: tuple[str, ...]) -> Mapping[str, Any]:
    """``value`` as a block of exactly ``keys``: a key more is a later version's (counted), one less no index."""
    _check(isinstance(value, Mapping))
    later = len(set(value) - set(keys))
    _check(not later, later_index_keys(later))
    _check(set(value) == set(keys))
    return value


@dataclass(frozen=True)
class StudyRef:
    """The study a package holds: its id, the revision of its definition with that revision's checksum (of the body
    as written) and fingerprint, and the version of the rules its criteria were read under."""

    id: str
    revision: int
    revision_checksum: str
    protocol_hash: str
    rules_version: int

    def __post_init__(self) -> None:
        _check(isinstance(self.id, str) and STUDY_ID.fullmatch(self.id) is not None)
        _check(_is_count(self.revision) and self.revision >= 1)
        _check(_hex64(self.revision_checksum) and _hex64(self.protocol_hash))
        _check(_is_count(self.rules_version) and self.rules_version >= 1)

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in _STUDY_KEYS}


@dataclass(frozen=True)
class Software:
    """What made the package: CohortStudio's version, its commit and the fingerprints of its code and of its lock
    file (each None when unknown), Python, the system, the machine and its byte order, the process's seed (for a
    step recorded before its own seed was kept), and the libraries a replay reads, by name (None: not installed)."""

    cohortstudio: str
    commit: str | None
    code_sha256: str | None
    uv_lock_sha256: str | None
    python: str
    system: str
    machine: str
    byteorder: str
    random_seed: int
    libraries: tuple[tuple[str, str | None], ...] = ()

    def __post_init__(self) -> None:
        # ASCII versions without a space or a path, and the platform's words: a host's name, an account or a folder
        # has no place here
        for text in (self.cohortstudio, self.python):
            _check(isinstance(text, str) and bool(_SOFTWARE_TEXT.fullmatch(text)))
        for text in (self.system, self.machine):
            _check(isinstance(text, str) and bool(_PLATFORM_WORD.fullmatch(text)))
        _check(
            self.commit is None
            or (isinstance(self.commit, str) and bool(_COMMIT.fullmatch(self.commit)))
        )
        _check(
            all(value is None or _hex64(value) for value in (self.code_sha256, self.uv_lock_sha256))
        )
        _check(isinstance(self.byteorder, str) and self.byteorder in BYTE_ORDERS)
        _check(_is_count(self.random_seed) and 0 <= self.random_seed <= SEED_MOST)
        _check(isinstance(self.libraries, tuple))
        for pair in self.libraries:
            _check(isinstance(pair, tuple) and len(pair) == 2)
            name, version = pair
            _check(isinstance(name, str) and bool(_LIBRARY.fullmatch(name)))
            _check(
                version is None
                or (isinstance(version, str) and bool(_SOFTWARE_TEXT.fullmatch(version)))
            )
        names = [name for name, _ in self.libraries]
        _check(names == sorted(set(names)))  # each once, in the one order

    def to_dict(self) -> dict[str, Any]:
        found: dict[str, Any] = {key: getattr(self, key) for key in _SOFTWARE_KEYS}
        found["libraries"] = dict(self.libraries)
        return found


@dataclass(frozen=True)
class TraceSummary:
    """The trace a package holds, as its index names it: its schema, what it starts from, and each step's kind and
    hash in order (its head is the last step's hash)."""

    schema_version: int
    starts_from: str
    steps: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        # a trace before the chain (schema 1) never travels with the data
        _check(_is_count(self.schema_version) and self.schema_version >= TRACE_SCHEMA)
        _check(self.schema_version == TRACE_SCHEMA, LATER_INDEX_VALUE)
        _closed_value(self.starts_from, STARTS_FROM)
        _check(isinstance(self.steps, tuple) and len(self.steps) >= 1)
        for pair in self.steps:
            _check(isinstance(pair, tuple) and len(pair) == 2)
            function, step_hash = pair
            _check(
                isinstance(function, str)
                and bool(_FUNCTION.fullmatch(function))
                and _hex64(step_hash)
            )
            _check(
                function != ADOPTED_IMPUTATION
            )  # a dataset of a multiple imputation adopted as the data

    @property
    def head(self) -> str:
        return self.steps[-1][1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "head": self.head,
            "starts_from": self.starts_from,
            "steps": [
                {"index": index, "function": function, "hash": step_hash}
                for index, (function, step_hash) in enumerate(self.steps, start=1)
            ],
        }


@dataclass(frozen=True)
class Cut:
    """The rows of a merged file that a package holds: those its merge took, or every row (a merge that keeps them
    all), out of the file's ``rows_of_original``, whose SHA-256 is the one its step recorded (None for a step
    recorded before it was kept)."""

    rule: str
    rows_of_original: int
    original_sha256: str | None = None

    def __post_init__(self) -> None:
        _closed_value(self.rule, CUT_RULES)
        _check(_is_count(self.rows_of_original) and self.rows_of_original >= 0)
        _check(self.original_sha256 is None or _hex64(self.original_sha256))

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in _CUT_KEYS}


@dataclass(frozen=True)
class DataFile:
    """
    A data member, as the index names it: its role, the step that reads it (1-based; None for what the trace starts
    from), its member, whether it holds the table as the step read it (``as_read``: a merged file always, cut by
    ``cut``; a source only as ``source.parquet``, a workbook's table) rather than the file's bytes, and its rows and
    columns.
    """

    role: str
    step: int | None
    member: str
    as_read: bool
    rows: int
    columns: int
    cut: Cut | None = None

    def __post_init__(self) -> None:
        _closed_value(self.role, FILE_ROLES)
        _check(self.role != "replayed_state")  # a state replayed from another trace never travels
        _check(data_role(self.member) == self.role)
        if self.role in STARTS_FROM:
            _check(self.step is None)
        else:
            _check(_is_count(self.step) and self.step >= 1)
        _check(isinstance(self.as_read, bool))
        _check(
            _is_count(self.rows)
            and self.rows >= 0
            and _is_count(self.columns)
            and self.columns >= 0
        )
        if self.role == "enrichment":
            _check(self.as_read and isinstance(self.cut, Cut))
            assert isinstance(self.cut, Cut)  # for the type checker
            _check(self.rows <= self.cut.rows_of_original)
            _check(self.cut.rule != "every_row" or self.rows == self.cut.rows_of_original)
        else:
            _check(self.cut is None)
            _check(not self.as_read or self.member == source_member("parquet"))

    def to_dict(self) -> dict[str, Any]:
        found: dict[str, Any] = {key: getattr(self, key) for key in _FILE_KEYS}
        found["cut"] = None if self.cut is None else self.cut.to_dict()
        return found


@dataclass(frozen=True)
class Declaration:
    """What the exporter declared: the recipient (a person and their organisation), the basis that covers sending
    the data (``BASES``) and its reference (not needed for the exporter's own organisation), on the words of version
    ``words`` (any number from 1). Under ``rows_not_people`` (M8b) the recipient and the reference may be None. The
    two texts are typed for the export, under the rule for a text: one line each, at most 200 and 100 characters,
    composed (NFC: the form normalises what is typed)."""

    basis: str
    recipient: str | None = None
    reference: str | None = None
    words: int = DECLARATION_WORDS

    def __post_init__(self) -> None:
        _closed_value(self.basis, BASES)
        _check(_is_count(self.words) and self.words >= 1)
        needed = (self.basis != "rows_not_people", self.basis not in NO_REFERENCE_NEEDED)
        for text, most, required in zip(
            (self.recipient, self.reference), (RECIPIENT_MAX, REFERENCE_MAX), needed, strict=True
        ):
            if text is None:
                _check(not required)
            else:
                # one way of writing each text (composed), so that one declaration gives one package's hash
                _check(_is_name(text, most) and unicodedata.is_normalized("NFC", text))

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in _DECLARATION_KEYS}


@dataclass(frozen=True)
class DataPackageIndex:
    """
    ``package.json`` of a package that holds the data (see the section's words). Its files: what the trace starts
    from (one, of the role the trace names), then the merged files, then the recorded tables, each numbered in the
    order of the steps that read them: a merged file read at a step that merges, a recorded table at a step of
    ``SNAPSHOT_STEPS``. Its policy states no rule (``rules`` ``none``); a declaration under ``rows_not_people``
    needs a policy that declares it. Its ``anchor`` is :func:`anchor_of` the member the trace starts from.

    ValueError, in words that quote nothing, for values that no package holds.
    """

    study: StudyRef
    software: Software
    trace: TraceSummary
    files: tuple[DataFile, ...]
    policy: Policy
    declaration: Declaration
    anchor: str

    def __post_init__(self) -> None:
        parts = ((self.study, StudyRef), (self.software, Software), (self.trace, TraceSummary))
        _check(all(isinstance(part, kind) for part, kind in parts))
        _check(isinstance(self.policy, Policy) and isinstance(self.declaration, Declaration))
        _check(
            isinstance(self.files, tuple)
            and all(isinstance(entry, DataFile) for entry in self.files)
        )
        _check(_hex64(self.anchor))
        _check(self.policy.rules == NONE)
        _check(
            self.declaration.basis != "rows_not_people" or self.policy.rows_not_people is not None
        )
        _check(not data_members_problem(self.data_members()))
        starts = [entry for entry in self.files if entry.role in STARTS_FROM]
        _check(starts[0].role == self.trace.starts_from)
        # the start, then the merged files and the recorded tables, each numbered in the order of its steps
        expected = [starts[0].member]
        steps = len(self.trace.steps)
        for role, member in (("enrichment", merge_member), ("snapshot", snapshot_member)):
            read = [entry for entry in self.files if entry.role == role]
            at = [entry.step for entry in read]
            _check(at == sorted(set(at)) and all(step is not None and step <= steps for step in at))
            expected += [member(number) for number in range(1, len(read) + 1)]
        _check(list(self.data_members()) == expected)
        # a merged file is read at a merge; a recorded table stands for a step whose table only it gives (never a
        # step that only documents, nor an imputation)
        kinds = {"enrichment": ("enrichment",), "snapshot": SNAPSHOT_STEPS}
        for entry in self.files:
            if entry.role in kinds:
                assert entry.step is not None  # for the type checker
                _check(self.trace.steps[entry.step - 1][0] in kinds[entry.role])

    def data_members(self) -> tuple[str, ...]:
        return tuple(entry.member for entry in self.files)

    def to_dict(self) -> dict[str, Any]:
        """The index as ``dumps(PACKAGE, ..., REPRODUCTION)`` writes it: every key, always."""
        return {
            "tier": REPRODUCTION,
            "study": self.study.to_dict(),
            "software": self.software.to_dict(),
            "trace": self.trace.to_dict(),
            "files": [entry.to_dict() for entry in self.files],
            "policy": self.policy.to_dict(),
            "declaration": self.declaration.to_dict(),
            "anchor": self.anchor,
        }

    @classmethod
    def from_dict(cls, value: Any) -> tuple[DataPackageIndex | None, list[str]]:
        """
        The index ``package.json`` holds (the whole member, its envelope included), or None with why it is refused.
        Closed: every key of each block and no other (a key more is a later version's, counted), every closed value
        among those this version knows, the steps numbered from 1, the head the last step's hash.
        """
        reasons = read_envelope(value, PACKAGE, REPRODUCTION)
        if reasons:
            return None, reasons
        try:
            whole = _block(value, ("format", "schema", *INDEX_KEYS))
            _closed_value(whole["tier"], TIERS)
            _check(whole["tier"] == REPRODUCTION)
            study = _block(whole["study"], _STUDY_KEYS)
            made = _block(whole["software"], _SOFTWARE_KEYS)
            _check(isinstance(made["libraries"], Mapping))
            trace = _block(whole["trace"], _TRACE_KEYS)
            _check(isinstance(trace["steps"], list) and isinstance(whole["files"], list))
            steps = []
            for position, item in enumerate(trace["steps"], start=1):
                step = _block(item, _STEP_KEYS)
                _check(_is_count(step["index"]) and step["index"] == position)
                steps.append((step["function"], step["hash"]))
            summary = TraceSummary(trace["schema_version"], trace["starts_from"], tuple(steps))
            _check(trace["head"] == summary.head)
            files = []
            for item in whole["files"]:
                entry = _block(item, _FILE_KEYS)
                cut = None if entry["cut"] is None else Cut(**_block(entry["cut"], _CUT_KEYS))
                files.append(DataFile(**{**entry, "cut": cut}))
            policy, refused = Policy.from_dict(whole["policy"])
            if policy is None:
                return None, refused
            index = cls(
                study=StudyRef(**study),
                software=Software(
                    **{**made, "libraries": tuple(sorted(made["libraries"].items()))}
                ),
                trace=summary,
                files=tuple(files),
                policy=policy,
                declaration=Declaration(**_block(whole["declaration"], _DECLARATION_KEYS)),
                anchor=whole["anchor"],
            )
        except ValueError as refused:
            return None, [str(refused)]
        except TypeError:  # a value that cannot be compared or sorted: a list where a text stands
            return None, [NOT_AN_INDEX]
        return index, []


__all__ = [
    "ADOPTED_IMPUTATION",
    "ANCHOR_PREFIX",
    "BASES",
    "BYTE_ORDERS",
    "CAPS",
    "CUT",
    "CUT_ON_A_TERMINAL",
    "CUT_RULES",
    "DATA_MEMBERS_MOST",
    "DATA_MEMBER_CAP",
    "DECLARATION_WORDS",
    "DEPTH",
    "EXTENSION",
    "FILE_ROLES",
    "FORMATS",
    "FORMAT_SCHEMAS",
    "INDEX_KEYS",
    "INVISIBLE",
    "KEY_TWICE",
    "K_NOT_THE_LARGER",
    "LATER_INDEX_VALUE",
    "LATER_POLICY_VALUE",
    "LEVELS",
    "LISTED",
    "LOCK",
    "MANIFEST",
    "MEMBERS",
    "NONE",
    "NOTICE",
    "NOTICES",
    "NOTICE_NOT_PEOPLE",
    "NOTICE_NOT_THE_POLICY",
    "NOTICE_PERSONAL_DATA",
    "NOT_A_MANIFEST",
    "NOT_A_MEMBER",
    "NOT_A_NOTICE",
    "NOT_A_NUMBER",
    "NOT_A_POLICY",
    "NOT_AN_INDEX",
    "NOT_A_TEXT",
    "NOT_JSON",
    "NOT_THE_DATA",
    "NO_REFERENCE_NEEDED",
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
    "PROFILES",
    "PROTOCOL",
    "README",
    "RECIPIENT_MAX",
    "REFERENCE_MAX",
    "REPLACED",
    "REPLACED_ON_A_TERMINAL",
    "REPRODUCTION",
    "RESULTS",
    "ROLE_WORDS",
    "RULES",
    "RULES_OFF",
    "SEED_MOST",
    "SET_BY",
    "SHAREABLE",
    "SNAPSHOT_STEPS",
    "SOURCE_EXTENSIONS",
    "STARTS_FROM",
    "STUDY_ID",
    "TABLE_ONLY_EXTENSIONS",
    "TEXT_MAX",
    "TIERS",
    "TOO_LARGE",
    "TRACE",
    "TRACE_SCHEMA",
    "Cut",
    "DataFile",
    "DataPackageIndex",
    "Declaration",
    "FileRole",
    "Policy",
    "Software",
    "StudyRef",
    "TraceSummary",
    "anchor_of",
    "cap_of",
    "checksum",
    "clear_members",
    "data_members_problem",
    "data_role",
    "dumps",
    "formats_of",
    "hidden",
    "hidden_reason",
    "holds_data",
    "in_zip_order",
    "is_carried",
    "is_place",
    "later_index_keys",
    "later_keys",
    "later_schema",
    "loads",
    "manifest_from_checksums",
    "manifest_of",
    "members_of",
    "merge_member",
    "not_matching",
    "notice_for",
    "notice_problem",
    "package_cap",
    "package_hash",
    "package_name",
    "read_envelope",
    "read_manifest",
    "read_member",
    "read_notice",
    "shown",
    "snapshot_member",
    "source_member",
    "version_member",
]
