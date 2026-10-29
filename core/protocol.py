"""
The study definition (M6): what a study decides, apart from the data it is applied to, and saved in the user's
workspace (manage.protocol_store).

Its *body* holds the decisions: the criteria (``mask_families``), the ticked population, the follow-up and the declared
derived variables. A section is written only when it is not empty, so a study keeps its fingerprint across versions of
the application. Its *metadata* stays outside: the study's id, title and description, the revision and the one it
follows, when and with which version it was saved.

The fingerprint (``protocol_hash``) is a SHA-256 of a canonical view of the body. The view leaves out the labels (a
rule's description, a derived variable's note, origin and kind, the follow-up's note), the order of keys and the order
of the ticks, and reads each rule as it selects rows (a bound of 18.0 is 18; see :func:`_canonical`). It keeps the
order of the families and of their rules, which decides the stages and the counts of the participant flow, and the
order of the derived variables, which decides which one fills a variable when two can; it reads a family's operator as
the population does ("All" is "all"), and holds the version of how rules are evaluated (core.criteria.RULES_VERSION).
It changes when a criterion, an operator, an order, a tick, a definition, the follow-up or that version changes. The
hash stays in the workspace and the trace; documents name a revision by its number.

The fingerprint is computed again whenever it is needed: a later version may read a study more finely, and its
fingerprint then changes. A file is checked by its *checksum* instead (:func:`body_checksum`), a SHA-256 of the body
exactly as written, whose computation never changes: a file saved by an earlier version is always read, and a body
changed outside the application never is. A revision is named, and linked to the one it follows, by its checksum.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from functools import cached_property
from types import MappingProxyType
from typing import Any

from core.criteria import RULES_VERSION
from core.derived import CONFIG_KEY as DERIVED_KEY
from core.lineage import canonical_json, normalise
from core.population import catalogue, family_operator
from core.workspace import is_device_name

SCHEMA = 1
# The sections of a body, in this order. M8 adds the analysis plan, the disclosure policy and the variables contract,
# M9 the concepts: each written only when not empty, so a study without them keeps its hash.
BODY_KEYS = ("mask_families", "population", "follow_up", DERIVED_KEY)
_DERIVED_LABELS = ("note", "origin", "kind")
# a study's id is its folder: lower-case ASCII letters and digits, with "_" and "-" after the first, 64 at most
STUDY_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_HASH = re.compile(r"[0-9a-f]{64}")
_META_TEXT = ("title", "description", "saved_at", "app_version")


def _filled(value: Any) -> bool:
    return value is not None and not (isinstance(value, (Mapping, list, tuple, str)) and not value)


def _frozen(value: Any) -> Any:
    """A read-only copy: mappings become mapping proxies, lists tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _frozen(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(v) for v in value)
    return value


def body_checksum(body: Mapping[str, Any]) -> str:
    """
    The checksum of a body as written: SHA-256 of its compact JSON, in the order written (the order of the families and
    their rules is part of a study), in full. Never change how it is computed: every saved file is checked with it.
    """
    text = json.dumps(normalise(body), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_study_id(text: object) -> bool:
    """Whether ``text`` can name a study (and its folder): see :data:`STUDY_ID`, and no name Windows keeps."""
    return isinstance(text, str) and bool(STUDY_ID.fullmatch(text)) and not is_device_name(text)


def study_id_for(name: str | None) -> str:
    """
    The id of the study of a dataset, from its base name: lower-case ASCII, accents dropped, other characters "_";
    "study" when nothing is left, "<name>_study" for a name Windows keeps for a device.
    """
    text = (
        unicodedata.normalize("NFKD", str(name or ""))
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    text = re.sub(r"[^a-z0-9_-]+", "_", text)
    text = re.sub(r"^[^a-z0-9]+", "", text)[:64].rstrip("_-")
    if not text:
        return "study"
    return f"{text[:58]}_study" if is_device_name(text) else text


@dataclass(frozen=True)
class ProtocolMeta:
    """What names a study and places a revision, outside the fingerprint."""

    study_id: str
    title: str = ""
    description: str = ""
    revision: int | None = None  # None: the working copy
    parent: str | None = None  # the checksum of the revision this one follows
    parent_revision: int | None = None
    saved_at: str = ""  # local time, to the minute
    app_version: str = ""

    def to_file(self) -> dict[str, Any]:
        out: dict[str, Any] = {"study_id": self.study_id}
        for name in (
            "title",
            "description",
            "revision",
            "parent",
            "parent_revision",
            "saved_at",
            "app_version",
        ):
            value = getattr(self, name)
            if value not in (None, ""):
                out[name] = value
        return out

    @classmethod
    def from_file(cls, data: Mapping[str, Any]) -> ProtocolMeta:
        """The metadata of a file; ValueError when it names no study or places the revision wrongly."""
        study_id = data.get("study_id")
        if not is_study_id(study_id):
            raise ValueError("the study's id is missing or is not one")
        numbers = {}
        for name in ("revision", "parent_revision"):
            value = data.get(name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"its {name.replace('_', ' ')} is not a number of a revision")
            numbers[name] = value
        parent = data.get("parent")
        if parent is not None and not (isinstance(parent, str) and _HASH.fullmatch(parent)):
            raise ValueError("its parent is not the checksum of a revision")
        texts = {name: str(data.get(name) or "") for name in _META_TEXT}
        return cls(study_id=str(study_id), parent=parent, **numbers, **texts)


@dataclass(frozen=True)
class StudyProtocol:
    """A study definition: its body, read-only, and its metadata."""

    body: Mapping[str, Any]
    meta: ProtocolMeta

    def __post_init__(self) -> None:
        object.__setattr__(self, "body", _frozen(body_of(self.body)))

    @cached_property
    def hash(self) -> str:
        """The body's fingerprint (:func:`protocol_hash`), as this version reads it."""
        return protocol_hash(self.body)

    @cached_property
    def checksum(self) -> str:
        """The body's checksum as written (:func:`body_checksum`)."""
        return body_checksum(self.body)

    def to_file(self) -> dict[str, Any]:
        """The file form of the working copy and of every revision."""
        return {
            "schema": SCHEMA,
            "checksum": self.checksum,
            "meta": self.meta.to_file(),
            "body": normalise(self.body),
        }

    @classmethod
    def from_file(cls, data: Any) -> tuple[StudyProtocol | None, list[str]]:
        """
        A study definition read from its file, or none with the reasons: a file of another schema or of a later
        version, metadata that place it wrongly, or a body that no longer matches its checksum (changed outside the
        application). The reasons quote nothing from the file, and nothing in it raises.
        """
        if not isinstance(data, Mapping) or not isinstance(data.get("body"), Mapping):
            return None, ["not a study definition"]
        if data.get("schema") != SCHEMA:
            return None, [
                f"written for another version of the study definition (this version reads schema {SCHEMA})"
            ]
        # counted, never named: a key of the file could hold anything
        later = len(set(data["body"]) - set(BODY_KEYS))
        if later:
            sections = f"{later} section{'s' if later != 1 else ''}"
            return None, [
                f"written by a later version: it holds {sections} this version does not read"
            ]
        meta = data.get("meta")
        if not isinstance(meta, Mapping):
            return None, ["it names no study"]
        try:
            meta = ProtocolMeta.from_file(meta)
        except ValueError as exc:
            return None, [str(exc)]
        try:  # the body as written, before this version reads it
            if data.get("checksum") != body_checksum(data["body"]):
                return None, [
                    "its body no longer matches its checksum: it was changed outside CohortStudio"
                ]
            return cls(data["body"], meta), []
        except (
            ArithmeticError,
            RecursionError,
            TypeError,
            ValueError,
        ):  # nested too deep, a number too large
            return None, ["not a study definition this version can read"]


def body_of(
    config: Mapping[str, Any] | None, ticked: Iterable[str] | None = None
) -> dict[str, Any]:
    """
    The study definition a configuration holds: its sections that are not empty, as a copy. ``ticked`` replaces the
    population's ticked criteria (the session keeps them apart from the configuration); they are kept sorted, and are
    all the population holds: a count or a note there would be a finding, which the fingerprint would not cover.
    """
    sections = {
        key: normalise(config[key]) for key in BODY_KEYS if config and _filled(config.get(key))
    }
    if ticked is None:
        population = sections.get("population")
        found = population.get("ticked") if isinstance(population, Mapping) else None
        ids = list(found) if isinstance(found, (list, tuple)) else []
    else:
        ids = list(ticked)
    sections["population"] = {"ticked": sorted({str(i) for i in ids})} if ids else {}
    return {key: sections[key] for key in BODY_KEYS if _filled(sections.get(key))}


def with_body(config: Mapping[str, Any] | None, body: Mapping[str, Any]) -> dict[str, Any]:
    """
    A configuration holding ``body``: its sections replace the configuration's and a section it lacks is removed,
    but ``mask_families`` stays (empty), as the pages expect it; the other keys (the description) are kept.
    """
    out = {
        key: copy.deepcopy(value) for key, value in (config or {}).items() if key not in BODY_KEYS
    }
    out["mask_families"] = {}
    out.update({key: normalise(body[key]) for key in BODY_KEYS if _filled(body.get(key))})
    return out


def _canonical(value: Any, key: str = "") -> Any:
    """
    A rule's definition as it selects rows, whatever its writing: without its descriptions (at any depth: a combined
    rule's members have theirs), a whole number written as one (a bound of 18.0 is 18), an expression without spaces
    at its ends, and no ``"numeric": false``, which is what a rule says by leaving it out.
    """
    if isinstance(value, dict):
        return {
            k: _canonical(v, k)
            for k, v in value.items()
            if k != "description" and not (k == "numeric" and v is False)
        }
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if key == "expression" and isinstance(value, str):
        return value.strip()
    return value


def _definition(definition: Any, labels: bool) -> Any:
    return normalise(definition) if labels else _canonical(normalise(definition))


def _derived(definition: Any, labels: bool) -> Any:
    """
    A derived variable as its fingerprint reads it: what it computes and how it is used, without its labels, read
    from the stored definition as written (formulas without spaces at their ends, a whole tolerance as an integer, an
    acceptance only when true), so that how the application parses a definition never moves it.
    """
    if labels or not isinstance(definition, Mapping):
        return normalise(definition)
    kept = {key: value for key, value in definition.items() if key not in _DERIVED_LABELS}
    formula, inverses = kept.pop("formula", None), kept.pop("inverses", None)
    accepted = kept.pop("accepted", None)
    out: dict[str, Any] = {}
    if formula is not None:
        out["formula"] = formula.strip() if isinstance(formula, str) else normalise(formula)
    if isinstance(inverses, Mapping) and inverses:
        out["inverses"] = {
            str(name): text.strip() if isinstance(text, str) else normalise(text)
            for name, text in inverses.items()
        }
    elif inverses:
        out["inverses"] = normalise(inverses)
    if accepted is True:
        out["accepted"] = True
    out.update(_canonical(normalise({k: v for k, v in kept.items() if v is not None})))
    return out


def _view(body: Mapping[str, Any], *, labels: bool) -> dict[str, Any]:
    view: dict[str, Any] = {}
    families = body.get("mask_families") or {}
    if isinstance(families, Mapping) and families:
        view["mask_families"] = [
            [
                str(family),
                list(family_operator(members)),
                [
                    [str(rule), _definition(spec, labels)]
                    for rule, spec in members.items()
                    if rule != "operator"
                ],
            ]
            if isinstance(members, Mapping)
            else [str(family), None, normalise(members)]
            for family, members in families.items()
        ]
    elif families:  # malformed: hashed as written, so that a change still shows
        view["mask_families"] = normalise(families)
    if families:
        view["rules_version"] = RULES_VERSION
    population = body.get("population")
    ids = population.get("ticked") if isinstance(population, Mapping) else None
    ticked = sorted({str(i) for i in ids}) if isinstance(ids, (list, tuple, set, frozenset)) else []
    if not labels:  # a tick naming no criterion of the body selects nothing
        known = (
            {c.id for c in catalogue({"mask_families": families})}
            if isinstance(families, Mapping)
            else set()
        )
        ticked = [i for i in ticked if i in known]
    if ticked:
        view["population"] = {"ticked": ticked}
    follow_up = body.get("follow_up")
    if follow_up:
        if isinstance(follow_up, Mapping) and not labels:
            follow_up = {key: value for key, value in follow_up.items() if key != "note"}
        view["follow_up"] = normalise(follow_up)
    derived = body.get(DERIVED_KEY)
    if derived:
        # in the order declared: when two definitions can fill one variable equally well, the first one does
        view[DERIVED_KEY] = (
            [[str(target), _derived(spec, labels)] for target, spec in derived.items()]
            if isinstance(derived, Mapping)
            else normalise(derived)
        )
    return view


def hashed_view(body: Mapping[str, Any]) -> dict[str, Any]:
    """What the fingerprint covers (see the module's description)."""
    return _view(body, labels=False)


def protocol_hash(body: Mapping[str, Any]) -> str:
    """The fingerprint of a body: SHA-256 of the canonical JSON of its :func:`hashed_view`, in full."""
    return hashlib.sha256(canonical_json(hashed_view(body)).encode("utf-8")).hexdigest()


def diff_lines(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """
    What changed from one body to the next, as the lines of a unified diff: indented JSON, keys sorted, the families
    and their rules in their order, the ticks sorted, labels included. None when nothing changed.
    """

    def lines(body: Mapping[str, Any]) -> list[str]:
        return json.dumps(
            _view(body, labels=True), indent=2, sort_keys=True, ensure_ascii=False
        ).splitlines()

    return list(difflib.unified_diff(lines(old), lines(new), lineterm="", n=2))[2:]


def next_revision(
    latest: StudyProtocol | None,
    body: Mapping[str, Any],
    meta: ProtocolMeta,
    *,
    number: int | None = None,
) -> StudyProtocol:
    """
    The revision that follows ``latest`` (the first when there is none), its parent the latest's checksum. Its number
    is the next one, or ``number``: the store's next free number, when a revision it cannot read holds one.
    """
    if number is None:
        number = (latest.meta.revision or 0) + 1 if latest is not None else 1
    return StudyProtocol(
        body,
        replace(
            meta,
            revision=number,
            parent=latest.checksum if latest is not None else None,
            parent_revision=latest.meta.revision if latest is not None else None,
        ),
    )


__all__ = [
    "BODY_KEYS",
    "SCHEMA",
    "STUDY_ID",
    "ProtocolMeta",
    "StudyProtocol",
    "body_checksum",
    "body_of",
    "diff_lines",
    "hashed_view",
    "is_study_id",
    "next_revision",
    "protocol_hash",
    "study_id_for",
    "with_body",
]
