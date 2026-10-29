"""
The study definition (M6): what a study decides, apart from the data it is applied to, and saved in the user's
workspace (manage.protocol_store).

Its *body* holds the decisions: the criteria (``mask_families``), the ticked population, the follow-up and the declared
derived variables. A section is written only when it is not empty, so a study keeps its fingerprint across versions of
the application. Its *metadata* stays outside: the study's id, title and description, the revision and the one it
follows, when and with which version it was saved.

The fingerprint (``protocol_hash``) is a SHA-256 of a canonical view of the body. The view leaves out the labels (a
rule's description, a derived variable's note, origin and kind, the follow-up's note), the order of keys and the order
of the ticks, but keeps the order of the families and of their rules, which decides the stages and the counts of the
participant flow, and reads a family's operator as the population does ("All" is "all"). It changes when a criterion,
an operator, that order, a tick, a definition or the follow-up changes. The hash stays in the workspace and the trace;
documents name a revision by its number.
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

from core.derived import CONFIG_KEY as DERIVED_KEY
from core.derived import DerivedVariable
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
    parent: str | None = None  # the fingerprint of the revision this one follows
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
            raise ValueError("its parent is not the fingerprint of a revision")
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
        """The body's fingerprint (:func:`protocol_hash`)."""
        return protocol_hash(self.body)

    def to_file(self) -> dict[str, Any]:
        """The file form of the working copy and of every revision."""
        return {
            "schema": SCHEMA,
            "hash": self.hash,
            "meta": self.meta.to_file(),
            "body": normalise(self.body),
        }

    @classmethod
    def from_file(cls, data: Any) -> tuple[StudyProtocol | None, list[str]]:
        """
        A study definition read from its file, or none with the reasons: a file of another schema or of a later
        version, metadata that place it wrongly, or a body that no longer matches its fingerprint (edited outside the
        application). The reasons quote nothing from the file.
        """
        if not isinstance(data, Mapping) or not isinstance(data.get("body"), Mapping):
            return None, ["not a study definition"]
        if data.get("schema") != SCHEMA:
            return None, [
                f"written for another version of the study definition (this version reads schema {SCHEMA})"
            ]
        later = sorted(set(data["body"]) - set(BODY_KEYS))
        if later:
            return None, [f"written by a later version: it holds {', '.join(later)}"]
        meta = data.get("meta")
        if not isinstance(meta, Mapping):
            return None, ["it names no study"]
        try:
            protocol = cls(data["body"], ProtocolMeta.from_file(meta))
        except ValueError as exc:
            return None, [str(exc)]
        if data.get("hash") != protocol.hash:
            return None, [
                "its body no longer matches its fingerprint: it was changed outside CohortStudio"
            ]
        return protocol, []


def body_of(
    config: Mapping[str, Any] | None, ticked: Iterable[str] | None = None
) -> dict[str, Any]:
    """
    The study definition a configuration holds: its sections that are not empty, as a copy. ``ticked`` replaces the
    population's ticked criteria (the session keeps them apart from the configuration); they are kept sorted.
    """
    sections = {
        key: normalise(config[key]) for key in BODY_KEYS if config and _filled(config.get(key))
    }
    population = sections.get("population")
    population = dict(population) if isinstance(population, Mapping) else {}
    ids = population.get("ticked") if ticked is None else ticked
    if isinstance(ids, (list, tuple, set, frozenset)) and ids:
        population["ticked"] = sorted({str(i) for i in ids})
    else:
        population.pop("ticked", None)
    sections["population"] = population
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


def _definition(definition: Any, labels: bool) -> Any:
    if isinstance(definition, Mapping) and not labels:
        definition = {key: value for key, value in definition.items() if key != "description"}
    return normalise(definition)


def _derived(target: str, definition: Any, labels: bool) -> Any:
    """A derived variable as its fingerprint reads it: what it computes and how it is used, without its labels."""
    if labels or not isinstance(definition, Mapping):
        return normalise(definition)
    try:
        parsed = DerivedVariable.from_config(target, definition)
    except KeyError, TypeError, ValueError:
        # a malformed definition: as written, without its labels
        kept = {key: value for key, value in definition.items() if key not in _DERIVED_LABELS}
        return normalise(kept)
    out: dict[str, Any] = {"formula": parsed.formula.strip()}
    if parsed.inverses:
        out["inverses"] = {name: formula.strip() for name, formula in parsed.inverses.items()}
    if parsed.tolerance is not None:
        out["tolerance"] = parsed.tolerance
    if parsed.accepted:
        out["accepted"] = True
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
    population = body.get("population")
    ids = population.get("ticked") if isinstance(population, Mapping) else None
    ticked = sorted({str(i) for i in ids}) if isinstance(ids, (list, tuple)) else []
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
        view[DERIVED_KEY] = (
            {str(target): _derived(str(target), spec, labels) for target, spec in derived.items()}
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
    The revision that follows ``latest`` (the first when there is none), its parent the latest's fingerprint. Its
    number is the next one, or ``number``: the store's next free number, when a revision it cannot read holds one.
    """
    if number is None:
        number = (latest.meta.revision or 0) + 1 if latest is not None else 1
    return StudyProtocol(
        body,
        replace(
            meta,
            revision=number,
            parent=latest.hash if latest is not None else None,
            parent_revision=latest.meta.revision if latest is not None else None,
        ),
    )


__all__ = [
    "BODY_KEYS",
    "SCHEMA",
    "STUDY_ID",
    "ProtocolMeta",
    "StudyProtocol",
    "body_of",
    "diff_lines",
    "hashed_view",
    "is_study_id",
    "next_revision",
    "protocol_hash",
    "study_id_for",
    "with_body",
]
