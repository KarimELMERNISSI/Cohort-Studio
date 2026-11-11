"""
The analysis plan (M8, the study package): what a stored analysis was asked to compute, the one call that computes it,
and the numbers it gave.

- :class:`AnalysisKind`: one entry per kind of analysis a page stores (``KINDS``). Their ids are stored with results,
  in dossiers and in packages: an id is never renamed.
- :class:`AnalysisSpec`: the variables by role, the options, the decisions read on the data when it ran, the random
  draws, what computed it, and what it was computed on (:class:`ComputedOn`: values the application passes in and
  this package never reads, since it imports nothing of the application). JSON values only. Its ``request_id`` names
  the request (kind, variables, options): the same on every machine, and what a package may carry. Its
  ``fingerprint`` also covers what it was computed on, row fingerprints included: for the session and for a package
  that holds the data, never for one meant to travel.
- :func:`run`: the kind's computation from its spec, on the observed rows or on imputed data. A page computes through
  it, so the result on the page and the one a package's check computes again come from the same call.
- :class:`ResultRecord` (:func:`record`): a result as named numbers. Each number has a comparison class (how closely
  a value computed again must agree) and each part the counts behind it (for the small-cell rule of a shared
  package). A part's id holds no value of the data: a factor's term is named by its factor and its level's place,
  the level's own name being a label that a shared package drops with the part it withholds. A record holds no
  residual and no point of a figure. Numbers at the resolution of one participant (the steps of a Kaplan-Meier
  curve, one per event; a median, which is an event time; the last follow-up time) are marked ``fine``: a shared
  package writes only what its rule for the kind names, and never a ``fine`` number as it is.

Specs and records exist for the time-to-event, logistic and count analyses; the other kinds have their titles and
get their runners later (until then their results have no spec and a package lists them as not run).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- comparison classes (stored ids)

EXACT = "exact"
DETERMINISTIC = "deterministic"
ITERATIVE = "iterative"
ITERATIVE_LOOSE = "iterative_loose"
SEEDED = "seeded"
CLASSES: dict[str, str] = {
    EXACT: "counts, names and decisions: equal",
    DETERMINISTIC: "closed-form arithmetic and special functions: equal up to rounding error",
    ITERATIVE: "Newton and IRLS fits: equal up to a small part of the standard error",
    ITERATIVE_LOOSE: "fits and statistics of an optional library, which stop earlier: equal up to their tolerance",
    SEEDED: "random draws: identical under the same seed, the rows in the same order, the same libraries and the "
    "same processor",
}

# how the difference between two values of a number is measured (stored ids)
PLAIN = "plain"
LOG = "log"  # of a p-value: in the tails it amplifies its statistic's rounding error
RECIPROCAL = (
    "reciprocal"  # through 1/x: a pooled term's df is infinite in one run and about 1e35 in another
)
RULES = (PLAIN, LOG, RECIPROCAL)

POOLED_SUFFIX = "_mi"  # of the key a pooled result is stored under ("cox_mi")

EXTREME_LOG, EXTREME_SE = (
    10.0,
    100.0,
)  # as biostats.regression: a log ratio and a standard error beyond them


class LaterVersion(ValueError):
    """A spec written by a later version of its kind than this installation knows: it is not run."""


# ----------------------------------------------------------------------------- JSON values


def _plain(value: Any) -> Any:
    """
    ``value`` as JSON values only (tuples as lists, numpy scalars and arrays as Python's), or TypeError: a spec or a
    record holding anything else could not be written, hashed or compared. A float that is not finite is written as
    a marker, ``{"float": "nan" | "inf" | "-inf"}``: JSON has no such number.
    """
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"float": "nan"}
        if math.isinf(value):
            return {"float": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("a JSON object needs text keys")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple | np.ndarray | pd.Series | pd.Index):
        return [_plain(item) for item in value]
    raise TypeError(
        f"not a JSON value: {type(value).__name__}"
    )  # the type only: a value may be data


_MARKERS = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


def _restored(value: Any) -> Any:
    """The reverse of :func:`_plain` for what was read back: the markers as floats."""
    if isinstance(value, dict):
        if set(value) == {"float"} and value["float"] in _MARKERS:
            return _MARKERS[value["float"]]
        return {key: _restored(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restored(item) for item in value]
    return value


def canonical(value: Any) -> str:
    """The canonical JSON of ``value``: sorted keys, no spaces, floats as Python writes them (they read back as the
    same 64-bit values)."""
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _typed(
    value: Mapping[str, Any], key: str, kinds: type | tuple[type, ...], *, optional: bool = False
) -> Any:
    """``value[key]`` when it is of the type a spec or a record holds there; ValueError otherwise, naming the key
    and never the value (a received file can hold anything)."""
    found = value.get(key)
    if found is None and optional:
        return None
    wanted = kinds if isinstance(kinds, tuple) else (kinds,)
    if isinstance(found, bool) and bool not in wanted:
        raise ValueError(f"{key} is not of the type expected")
    if not isinstance(found, wanted):
        raise ValueError(f"{key} is not of the type expected")
    return found


# ----------------------------------------------------------------------------- what it was computed on


@dataclass(frozen=True)
class ImputationUsed:
    """
    The imputation a pooled analysis used, by its content: the fingerprint of its imputed cells (the one the trace's
    ``mi_impute`` step records), the number of datasets, whether it was fitted on more rows than the analysis
    population, and that step's number and hash in the trace when the trace holds it.
    """

    cells_fp: str
    m: int
    subset: bool = False
    step_index: int | None = None
    step_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cells_fp": self.cells_fp,
            "m": self.m,
            "subset": self.subset,
            "step_index": self.step_index,
            "step_hash": self.step_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ImputationUsed:
        return cls(
            cells_fp=_typed(value, "cells_fp", str),
            m=_typed(value, "m", int),
            subset=_typed(value, "subset", bool),
            step_index=_typed(value, "step_index", int, optional=True),
            step_hash=_typed(value, "step_hash", str, optional=True),
        )


@dataclass(frozen=True)
class ComputedOn:
    """
    What an analysis was computed on, as the application says it (this package reads none of it): the data's
    fingerprint, the fingerprint of the analysis population's row ids, the rows of the population and those analysed,
    the study definition's hash, the version of the rules that read missing values and criteria, the trace's head
    and its number of steps then, and for a pooled analysis the imputation.

    ``rows_fp``, the trace's head, the study definition's hash and the imputation's fingerprints name rows: a guess
    of which rows a criterion excluded can be tried against them. A package meant to travel carries
    :meth:`shareable` only.
    """

    mode: str  # "observed" | "imputed"
    data_fp: str
    n_population: int
    n_analysed: int | None = None
    rows_fp: str | None = None
    protocol_hash: str | None = None
    rules_version: int | None = None
    trace_head: str | None = None
    trace_steps: int | None = None
    imputation: ImputationUsed | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("observed", "imputed"):
            raise ValueError("an analysis is computed on observed or on imputed data")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "data_fp": self.data_fp,
            "n_population": self.n_population,
            "n_analysed": self.n_analysed,
            "rows_fp": self.rows_fp,
            "protocol_hash": self.protocol_hash,
            "rules_version": self.rules_version,
            "trace_head": self.trace_head,
            "trace_steps": self.trace_steps,
            "imputation": None if self.imputation is None else self.imputation.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ComputedOn:
        imputation = _typed(value, "imputation", dict, optional=True)
        return cls(
            mode=_typed(value, "mode", str),
            data_fp=_typed(value, "data_fp", str),
            n_population=_typed(value, "n_population", int),
            n_analysed=_typed(value, "n_analysed", int, optional=True),
            rows_fp=_typed(value, "rows_fp", str, optional=True),
            protocol_hash=_typed(value, "protocol_hash", str, optional=True),
            rules_version=_typed(value, "rules_version", int, optional=True),
            trace_head=_typed(value, "trace_head", str, optional=True),
            trace_steps=_typed(value, "trace_steps", int, optional=True),
            imputation=None if imputation is None else ImputationUsed.from_dict(imputation),
        )

    def shareable(self) -> dict[str, Any]:
        """What a package meant to travel may say of it: no fingerprint of rows, of settings or of a step."""
        shared: dict[str, Any] = {
            "mode": self.mode,
            "data_fp": self.data_fp,
            "n_population": self.n_population,
            "n_analysed": self.n_analysed,
            "rules_version": self.rules_version,
            "trace_steps": self.trace_steps,
        }
        if self.imputation is not None:
            shared["imputation"] = {
                "m": self.imputation.m,
                "subset": self.imputation.subset,
                "step_index": self.imputation.step_index,
            }
        return shared


# ----------------------------------------------------------------------------- the spec


@dataclass(frozen=True)
class AnalysisSpec:
    """
    What an analysis was asked to compute.

    ``variables`` names the columns by their role (``kind``'s ``roles``); ``options`` the settings (``alpha``,
    ``ties``, ``family``, the event's level); ``decided`` what was read on the data when it ran (the levels of each
    factor with the reference first, the values read as an event), recorded so that a later run uses or compares
    them; ``random`` the seed and the number of draws, None for a kind that draws nothing; ``backend`` the libraries
    that computed it, with their versions; ``data`` what it was computed on.
    """

    kind: str
    pooled: bool = False
    variables: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)
    decided: Mapping[str, Any] = field(default_factory=dict)
    random: Mapping[str, Any] | None = None
    backend: Mapping[str, Any] = field(default_factory=dict)
    data: ComputedOn | None = None
    kind_version: int = 1

    def __post_init__(self) -> None:
        kind = kind_named(self.kind)
        if self.pooled and kind.pooled_title is None:
            raise ValueError(f"the analysis kind {self.kind} is never pooled")
        unknown = sorted(set(self.variables) - set(kind.roles))
        if unknown:
            raise ValueError(
                f"the analysis kind {self.kind} has no role {', '.join(unknown)}; its roles: "
                f"{', '.join(kind.roles) or 'none yet'}"
            )
        offered = {**kind.options, **(kind.pooled_options if self.pooled else {})}
        unknown = sorted(set(self.options) - set(offered))
        if unknown:
            raise ValueError(
                f"the analysis kind {self.kind} has no option {', '.join(unknown)}; its options: "
                f"{', '.join(offered) or 'none'}"
            )
        for name in ("variables", "options", "decided", "backend"):
            object.__setattr__(self, name, _plain(getattr(self, name)))
        if self.random is not None:
            object.__setattr__(self, "random", _plain(self.random))

    def request(self) -> dict[str, Any]:
        """What was asked, apart from what it ran on and with: the kind, the variables, the options, the draws."""
        return {
            "kind": self.kind,
            "kind_version": self.kind_version,
            "pooled": self.pooled,
            "variables": copy.deepcopy(dict(self.variables)),
            "options": copy.deepcopy(dict(self.options)),
            "random": None if self.random is None else copy.deepcopy(dict(self.random)),
        }

    def to_dict(self) -> dict[str, Any]:
        """A copy as JSON values: changing it changes nothing of the spec."""
        return {
            **self.request(),
            "decided": copy.deepcopy(dict(self.decided)),
            "backend": copy.deepcopy(dict(self.backend)),
            "data": None if self.data is None else self.data.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AnalysisSpec:
        """A spec read back. ValueError for a value of another type than a spec holds there (the key is named, never
        the value), or for a kind, a role or an option this version does not know."""
        data = _typed(value, "data", dict, optional=True)
        random = _typed(value, "random", dict, optional=True)
        return cls(
            kind=_typed(value, "kind", str),
            kind_version=_typed(value, "kind_version", int),
            pooled=_typed(value, "pooled", bool),
            variables=_restored(_typed(value, "variables", dict)),
            options=_restored(_typed(value, "options", dict)),
            decided=_restored(_typed(value, "decided", dict)),
            random=None if random is None else _restored(random),
            backend=_restored(_typed(value, "backend", dict)),
            data=None if data is None else ComputedOn.from_dict(data),
        )

    @property
    def request_id(self) -> str:
        """SHA-256 of the request's canonical JSON: the same analysis has the same id on every machine and whatever
        it ran on. It names a result in a package."""
        return hashlib.sha256(canonical(self.request()).encode("utf-8")).hexdigest()

    @property
    def fingerprint(self) -> str:
        """
        SHA-256 of the whole spec's canonical JSON: two specs with one fingerprint asked for the same computation on
        the same data with the same libraries. It covers the fingerprints of the rows and of the trace: for the
        session and for a package that holds the data, never for one meant to travel (a guess of the excluded rows
        could be tried against it).
        """
        return hashlib.sha256(canonical(self.to_dict()).encode("utf-8")).hexdigest()

    def on(self, data: ComputedOn) -> AnalysisSpec:
        """The spec with what it was computed on."""
        return replace(self, data=data)

    def covariates(self) -> list[str]:
        """A model's covariates: the comparison first, then the adjustment (a variable named twice, once)."""
        compare = self.variables.get("compare")
        first = [compare] if compare else []
        return first + [c for c in self.variables.get("adjust") or [] if c not in first]


def read_spec(value: Any) -> tuple[AnalysisSpec | None, str]:
    """
    A spec from a received file, or the reason it cannot be read here: a kind, a role or an option this version does
    not know (the package was made by a later one), a value of another type. The reason names keys and kinds, never
    a value: a check of the package then reads "not run" with it.
    """
    if not isinstance(value, Mapping):
        return None, "not an analysis spec"
    try:
        spec = AnalysisSpec.from_dict(value)
    except (ValueError, TypeError) as exc:
        return None, str(exc) if isinstance(exc, ValueError) else "holds a value that is not JSON"
    if spec.kind_version > kind_named(spec.kind).version:
        return None, f"the {spec.kind} analysis was specified by a later version of CohortStudio"
    return spec, ""


@dataclass(frozen=True)
class Computed:
    """
    A result with the spec as it ran (its decisions and its libraries filled in) and ``derived``, the decisions this
    run would have taken on its own. They are the spec's on a first run. On a run from a recorded spec the factor
    levels used are the recorded ones, and ``derived`` says what the data give now, so that a check can say when
    they differ.
    """

    result: Any
    spec: AnalysisSpec
    derived: Mapping[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------- the record


@dataclass(frozen=True)
class Number:
    """
    A named number of a result. ``value`` is a JSON scalar or a list of them (a series: one class for every entry);
    ``scale`` is the quantity's own scale, the floor of its tolerance (its standard error for an estimate, 1 for a
    probability); ``fine`` marks a value at the resolution of one participant (a series with one entry per event, a
    median, a largest time).
    """

    name: str
    value: Any
    cls: str = DETERMINISTIC
    scale: float | None = None
    rule: str = PLAIN
    fine: bool = False

    def __post_init__(self) -> None:
        if self.cls not in CLASSES:
            raise ValueError(f"unknown comparison class {self.cls}; known: {', '.join(CLASSES)}")
        if self.rule not in RULES:
            raise ValueError(f"unknown comparison rule {self.rule}; known: {', '.join(RULES)}")
        value = _plain(self.value)
        if isinstance(value, dict) and set(value) != {"float"}:
            raise TypeError("a number is a scalar or a list of scalars")
        object.__setattr__(self, "value", _restored(value))
        scale = None if self.scale is None else float(self.scale)
        object.__setattr__(
            self, "scale", scale if scale is not None and math.isfinite(scale) else None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": _plain(self.value),
            "class": self.cls,
            "scale": self.scale,
            "rule": self.rule,
            "fine": self.fine,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Number:
        return cls(
            name=_typed(value, "name", str),
            value=_restored(value.get("value")),
            cls=_typed(value, "class", str),
            scale=_typed(value, "scale", (int, float), optional=True),
            rule=_typed(value, "rule", str),
            fine=_typed(value, "fine", bool),
        )


@dataclass(frozen=True)
class Part:
    """
    A part of a result (the model, a term, a curve, a test) with its numbers.

    ``support`` holds the counts behind it (``rows``, ``events``, ``non_events``, ``rows_with_events``,
    ``df_resid``): the small-cell rule of a shared package reads it. A pooled result's counts are the smallest
    over the imputed datasets; those of its model and of its curves, which a shared package can show, are named
    so (``rows_min``), since they do not add up. ``labels`` holds the names the part's id leaves out: its
    ``variable``, and for a level or a group its ``level`` or ``group``, which are values of the data. A term
    whose variable has a part ``reference:<variable>`` compares two cells: a level, or the upper value of a
    covariate of two values, with the reference; the rows off the commonest value of a numeric covariate with the
    rows on it. A shared package shows it only when both cells can be shown, and drops the labels of a part it
    withholds.
    """

    id: str
    numbers: tuple[Number, ...] = ()
    support: Mapping[str, int] = field(default_factory=dict)
    labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "numbers", tuple(self.numbers))
        support = {}
        for key, count in dict(self.support).items():
            if isinstance(count, bool) or not isinstance(count, int | np.integer):
                raise ValueError("a part's support holds whole numbers")
            support[str(key)] = int(count)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "labels", {str(k): str(v) for k, v in dict(self.labels).items()})

    def number(self, name: str) -> Number:
        for found in self.numbers:
            if found.name == name:
                return found
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "labels": dict(self.labels),
            "support": dict(self.support),
            "numbers": [n.to_dict() for n in self.numbers],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Part:
        return cls(
            id=_typed(value, "id", str),
            numbers=tuple(Number.from_dict(n) for n in _typed(value, "numbers", list)),
            support=_typed(value, "support", dict),
            labels=_typed(value, "labels", dict),
        )


@dataclass(frozen=True)
class ResultRecord:
    """
    A result as named numbers, part by part. ``record_version`` is the version of its kind's record: a record of
    another version is not compared number by number. ``warnings`` is wording: never compared, and never copied into
    a shared package (a warning can name a level); what a check needs of them is in the numbers (``converged``,
    ``extreme``, the groups skipped). ``complete_case`` is the record of the complete-case analysis a pooled result
    carries for comparison.
    """

    kind: str
    pooled: bool = False
    parts: tuple[Part, ...] = ()
    warnings: tuple[str, ...] = ()
    complete_case: ResultRecord | None = None
    record_version: int = 1

    def __post_init__(self) -> None:
        kind_named(self.kind)
        object.__setattr__(self, "parts", tuple(self.parts))
        object.__setattr__(self, "warnings", tuple(str(w) for w in self.warnings))
        ids = [p.id for p in self.parts]
        if len(set(ids)) != len(ids):
            raise ValueError("two parts of a record share an id")

    def part(self, part_id: str) -> Part:
        for found in self.parts:
            if found.id == part_id:
                return found
        raise KeyError(part_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "record_version": self.record_version,
            "pooled": self.pooled,
            "parts": [p.to_dict() for p in self.parts],
            "warnings": list(self.warnings),
            "complete_case": None if self.complete_case is None else self.complete_case.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResultRecord:
        inner = _typed(value, "complete_case", dict, optional=True)
        return cls(
            kind=_typed(value, "kind", str),
            record_version=_typed(value, "record_version", int),
            pooled=_typed(value, "pooled", bool),
            parts=tuple(Part.from_dict(p) for p in _typed(value, "parts", list)),
            warnings=tuple(_typed(value, "warnings", list)),
            complete_case=None if inner is None else cls.from_dict(inner),
        )

    def to_json(self) -> str:
        return canonical(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> ResultRecord:
        return cls.from_dict(json.loads(text))


# ----------------------------------------------------------------------------- the kinds


@dataclass(frozen=True)
class AnalysisKind:
    """
    A kind of stored analysis. ``run`` and ``pool`` compute it from a spec, on a frame or on imputed data, and return
    the result with the decisions used and those derived from the data; ``record`` and ``pooled_record`` give a
    result's numbers. A kind without them has its titles only. ``optional`` names, per optional library, the parts
    (``"ph:"``) and numbers (``"model/concordance"``) that only it computes: where it is absent on one side of a
    comparison, they cannot be compared.
    """

    id: str
    title: str
    pooled_title: str | None = None  # None: never pooled
    roles: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)  # with their defaults
    pooled_options: Mapping[str, Any] = field(default_factory=dict)  # of the pooled analysis only
    libraries: tuple[str, ...] = ()  # whose versions the spec records
    run: (
        Callable[[AnalysisSpec, pd.DataFrame], tuple[Any, dict[str, Any], dict[str, Any]]] | None
    ) = None
    pool: Callable[[AnalysisSpec, Any], tuple[Any, dict[str, Any], dict[str, Any]]] | None = None
    record: Callable[[Any], tuple[list[Part], ResultRecord | None]] | None = None
    pooled_record: Callable[[Any], tuple[list[Part], ResultRecord | None]] | None = None
    optional: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    version: int = 1  # of the spec: its roles, its options, what its runner computes
    record_version: int = 1  # of the record: its parts, its numbers, their classes


def kind_named(kind_id: str) -> AnalysisKind:
    found = KINDS.get(kind_id)
    if found is None:
        raise ValueError(f"unknown analysis kind {str(kind_id)[:40]!r}; known: {', '.join(KINDS)}")
    return found


def kind_of(key: str) -> tuple[AnalysisKind, bool]:
    """The kind a result is stored under and whether it is pooled: ``cox_mi`` is the Cox model, pooled."""
    if key in KINDS:
        return KINDS[key], False
    if key.endswith(POOLED_SUFFIX):
        kind = KINDS.get(key[: -len(POOLED_SUFFIX)])
        if kind is not None and kind.pooled_title is not None:
            return kind, True
    raise ValueError(f"unknown result key {str(key)[:40]!r}")


def titles() -> dict[str, str]:
    """The title of every stored result key, pooled ones included."""
    out: dict[str, str] = {}
    for kind in KINDS.values():
        out[kind.id] = kind.title
        if kind.pooled_title is not None:
            out[kind.id + POOLED_SUFFIX] = kind.pooled_title
    return out


def specify(
    kind: str, *, pooled: bool = False, options: Mapping[str, Any] | None = None, **variables: Any
) -> AnalysisSpec:
    """A spec of ``kind`` for these variables (by role), with the kind's default options unless ``options`` says
    otherwise."""
    found = kind_named(kind)
    defaults = {**found.options, **(found.pooled_options if pooled else {})}
    return AnalysisSpec(
        kind=kind,
        kind_version=found.version,
        pooled=pooled,
        variables=variables,
        options={**defaults, **dict(options or {})},
    )


def runnable(kind: str, *, pooled: bool = False) -> bool:
    """Whether :func:`run` can compute this kind (the others have their titles only for now)."""
    found = KINDS.get(kind)
    return found is not None and (found.pool if pooled else found.run) is not None


def _versions(kind: AnalysisKind, pooled: bool) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for name in kind.libraries:
        try:
            found[name] = version(name)
        except (
            PackageNotFoundError
        ):  # an optional library (lifelines): its part of the result is absent too
            found[name] = None
    if pooled:
        from biostats.mi.engine import ENGINE

        found["imputation_engine"] = ENGINE
    return found


def run(spec: AnalysisSpec, frame: pd.DataFrame | None = None, *, mi: Any = None) -> Computed:
    """
    Compute ``spec``: on ``frame`` (the rows of the analysis population), or on the imputed data ``mi`` when it is
    pooled. The spec returned holds the decisions the run used and the libraries' versions. The factor levels a spec
    already holds are used as recorded (the same terms, the same reference); what the data give on their own is
    returned beside (``derived``), for the caller to compare. :class:`LaterVersion` for a spec written by a later
    version of its kind.
    """
    kind = kind_named(spec.kind)
    if spec.kind_version > kind.version:
        raise LaterVersion(
            f"the {spec.kind} analysis was specified by a later version of CohortStudio"
        )
    if spec.pooled:
        if kind.pool is None:
            raise ValueError(f"the pooled {spec.kind} analysis cannot be computed from a spec yet")
        if mi is None:
            raise ValueError("a pooled analysis needs the imputed data")
        result, used, derived = kind.pool(spec, mi)
    else:
        if kind.run is None:
            raise ValueError(f"the {spec.kind} analysis cannot be computed from a spec yet")
        if frame is None:
            raise ValueError("an analysis needs the rows to analyse")
        result, used, derived = kind.run(spec, frame)
    return Computed(
        result, replace(spec, decided=used, backend=_versions(kind, spec.pooled)), _plain(derived)
    )


def record(key: str, result: Any) -> ResultRecord | None:
    """The record of a result stored under ``key``, stamped with its kind's record version; None for a kind that
    has none yet."""
    kind, pooled = kind_of(key)
    build = kind.pooled_record if pooled else kind.record
    if build is None or result is None:
        return None
    parts, complete_case = build(result)
    return ResultRecord(
        kind.id,
        pooled=pooled,
        parts=tuple(parts),
        warnings=tuple(getattr(result, "warnings", ()) or ()),
        complete_case=complete_case,
        record_version=kind.record_version,
    )


# ----------------------------------------------------------------------------- runners
# Each returns the result, the decisions the run used and the decisions the data give on their own.


def _recorded_levels(spec: AnalysisSpec) -> dict[str, list[str]] | None:
    """The factor levels a spec recorded, the reference first; None for a spec that has not run."""
    recorded = spec.decided.get("levels")
    return (
        None
        if recorded is None
        else {str(k): [str(v) for v in levels] for k, levels in recorded.items()}
    )


def _levels_of(counts: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    return {factor: list(levels) for factor, levels in counts.items()}


def _without_term(result: Any, covariates: Sequence[str]) -> list[str]:
    """The covariates the fit holds no term for: a constant column is dropped, a factor of one level has none."""
    indicators = set(getattr(result, "indicator_counts", {}) or {})
    factors = {
        f for f, levels in (getattr(result, "level_counts", {}) or {}).items() if len(levels) > 1
    }
    terms = {str(t) for t in result.table["term"]}
    return [c for c in covariates if c not in factors and c not in indicators and c not in terms]


def _decisions(
    used: dict[str, Any], derived: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """What the run used (``used`` over what the data give) and what the data give."""
    return {**derived, **used}, derived


def _recorded_cells(
    rows: pd.DataFrame, covariates: Sequence[str], events: Any, levels: Mapping[str, Any]
) -> dict[str, Any]:
    """
    The cells of a pooled model's numeric covariates, counted on recorded values. ``rows`` are the observed rows that
    record the outcome and ``events`` each one's events; a covariate is counted on the rows that record it: each
    value of a covariate recorded with two values, and the rows on and off the commonest value of one that holds
    more (biostats.design). An imputed value is not a recorded one. The small-cell rule of a shared result reads
    these cells where the imputed datasets do not keep a covariate's two values (a normal draw gives it others: one
    participant coded 1 would carry the term under the model's counts), and for a commonest value always.
    """
    from biostats.design import dominant_counts, indicator_counts

    happened = np.asarray(events, dtype=float)
    cells: dict[str, Any] = {"indicator": {}, "dominant": {}}
    for name in covariates:
        if name in levels:
            continue
        held = rows[name].notna().to_numpy(dtype=bool)
        part = rows.loc[held, [name]]
        cells["indicator"].update(indicator_counts(part, [name], happened[held]))
        cells["dominant"].update(dominant_counts(part, [name], happened[held]))
    return cells


def _run_survival(
    spec: AnalysisSpec, frame: pd.DataFrame
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from biostats.survival import compare_survival, event_coding

    v = spec.variables
    result = compare_survival(
        frame, v["time"], v["event"], v.get("group"), alpha=spec.options["alpha"]
    )
    derived = {
        "groups": [c.label for c in result.curves],
        "groups_skipped": int(result.groups_skipped),
        "event_values": event_coding(frame[v["event"]]),
    }
    return result, *_decisions({}, derived)


def _pool_survival(spec: AnalysisSpec, mi: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from biostats.mi import analyze_mi
    from biostats.survival import compare_survival, event_coding

    v, alpha = spec.variables, spec.options["alpha"]
    group = v.get("group")
    result = analyze_mi(
        lambda d: compare_survival(d, v["time"], v["event"], group, alpha=alpha),
        mi,
        columns=[v["time"], v["event"], *([group] if group else [])],
        alpha=alpha,
    )
    derived = {
        "groups": list(result.extra.get("curves") or {}),
        "event_values": event_coding(mi.base[v["event"]]),
    }
    return result, *_decisions({}, derived)


def _run_cox(spec: AnalysisSpec, frame: pd.DataFrame) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from biostats.design import factor_levels
    from biostats.survival import cox_regression, event_coding, rows_read

    v, o = spec.variables, spec.options
    covariates = spec.covariates()
    recorded = _recorded_levels(spec)
    result = cox_regression(
        frame,
        v["time"],
        v["event"],
        covariates,
        alpha=o["alpha"],
        ties=o["ties"],
        levels=recorded,
        drop_constant=None if recorded is None else True,
    )
    own = (
        _levels_of(result.level_counts)
        if recorded is None
        else factor_levels(rows_read(frame, v["time"], v["event"], covariates), covariates)
    )
    derived = {
        "levels": own,
        "event_values": event_coding(frame[v["event"]]),
        "without_term": _without_term(result, covariates),
    }
    return result, *_decisions({} if recorded is None else {"levels": recorded}, derived)


def _pool_cox(spec: AnalysisSpec, mi: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from biostats.mi import analyze_mi
    from biostats.survival import cox_regression, event_coding, event_indicator, rows_read

    v, o = spec.variables, spec.options
    covariates = spec.covariates()
    own = mi.levels(covariates)
    levels = _recorded_levels(spec) or own
    result = analyze_mi(
        lambda d: cox_regression(
            d,
            v["time"],
            v["event"],
            covariates,
            alpha=o["alpha"],
            ties=o["ties"],
            levels=levels,
            # the proportional-hazards test and the concordance (lifelines) on the first imputed datasets only
            diagnostics=d.attrs["imputation"] <= o["ph_test_imputations"],
        ),
        mi,
        columns=[v["time"], v["event"], *covariates],
        alpha=o["alpha"],
    )
    read = mi.base.loc[rows_read(mi.base, v["time"], v["event"]).index]
    result.extra["recorded_cells"] = _recorded_cells(
        read, covariates, event_indicator(read[v["event"]]), levels
    )
    derived = {"levels": own, "event_values": event_coding(mi.base[v["event"]])}
    return result, *_decisions({"levels": levels}, derived)


def _run_logistic(
    spec: AnalysisSpec, frame: pd.DataFrame
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from biostats.design import factor_levels
    from biostats.regression import logistic_regression

    v, o = spec.variables, spec.options
    covariates = spec.covariates()
    recorded = _recorded_levels(spec)
    result = logistic_regression(
        frame,
        v["outcome"],
        covariates,
        event_level=o["event_level"],
        alpha=o["alpha"],
        levels=recorded,
        drop_constant=None if recorded is None else True,
    )
    own = (
        _levels_of(result.level_counts)
        if recorded is None
        else factor_levels(frame[[v["outcome"], *covariates]].dropna(), covariates)
    )
    outcome_levels = list(result.outcome_levels)
    derived = {
        "levels": own,
        "outcome_levels": outcome_levels,
        # the level counted as the event: the one asked for, or the second of the two when none was
        "event_level": str(o["event_level"]) if o["event_level"] is not None else outcome_levels[1],
        "without_term": _without_term(result, covariates),
    }
    return result, *_decisions({} if recorded is None else {"levels": recorded}, derived)


def _pool_logistic(spec: AnalysisSpec, mi: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from biostats.mi import analyze_mi
    from biostats.regression import logistic_regression

    v, o = spec.variables, spec.options
    covariates = spec.covariates()
    own = mi.levels(covariates)
    levels = _recorded_levels(spec) or own
    result = analyze_mi(
        lambda d: logistic_regression(
            d,
            v["outcome"],
            covariates,
            event_level=o["event_level"],
            alpha=o["alpha"],
            levels=levels,
        ),
        mi,
        columns=[v["outcome"], *covariates],
        alpha=o["alpha"],
    )
    outcome_levels = sorted(mi.base[v["outcome"]].dropna().astype(str).unique())
    derived = {
        "levels": own,
        "outcome_levels": outcome_levels,
        "event_level": str(o["event_level"])
        if o["event_level"] is not None
        else outcome_levels[-1],
    }
    read = mi.base[mi.base[v["outcome"]].notna()]
    result.extra["recorded_cells"] = _recorded_cells(
        read, covariates, read[v["outcome"]].astype(str) == derived["event_level"], levels
    )
    return result, *_decisions({"levels": levels}, derived)


def _run_count(
    spec: AnalysisSpec, frame: pd.DataFrame
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from biostats.design import factor_levels
    from biostats.regression import count_regression

    v, o = spec.variables, spec.options
    covariates, exposure = spec.covariates(), v.get("person_time")
    recorded = _recorded_levels(spec)
    result = count_regression(
        frame,
        v["count"],
        covariates,
        exposure_col=exposure,
        family=o["family"],
        alpha=o["alpha"],
        levels=recorded,
        drop_constant=None if recorded is None else True,
    )
    read = frame[[v["count"], *covariates, *([exposure] if exposure else [])]].dropna()
    own = _levels_of(result.level_counts) if recorded is None else factor_levels(read, covariates)
    derived = {"levels": own, "without_term": _without_term(result, covariates)}
    return result, *_decisions({} if recorded is None else {"levels": recorded}, derived)


def _pool_count(spec: AnalysisSpec, mi: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from biostats.mi import analyze_mi
    from biostats.regression import count_regression

    v, o = spec.variables, spec.options
    covariates, exposure = spec.covariates(), v.get("person_time")
    own = mi.levels(covariates)
    levels = _recorded_levels(spec) or own
    result = analyze_mi(
        lambda d: count_regression(
            d,
            v["count"],
            covariates,
            exposure_col=exposure,
            family=o["family"],
            alpha=o["alpha"],
            levels=levels,
        ),
        mi,
        columns=[v["count"], *covariates, *([exposure] if exposure else [])],
        alpha=o["alpha"],
    )
    read = mi.base.loc[mi.base[[v["count"], *([exposure] if exposure else [])]].dropna().index]
    result.extra["recorded_cells"] = _recorded_cells(read, covariates, read[v["count"]], levels)
    return result, *_decisions({"levels": levels}, {"levels": own})


# ----------------------------------------------------------------------------- records


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _end_scale(end: Any, se: Any) -> float | None:
    """The scale of a ratio, or of an end of its interval: the value times the standard error of its logarithm (the
    delta method), with a margin of a fifth for an end, which moves with 1.96 times the standard error's own
    change."""
    a, b = _finite(end), _finite(se)
    return None if a is None or b is None else 1.2 * abs(a) * b


def _series(name: str, values: Any, cls: str, *, scale: float | None = None) -> Number:
    return Number(name, np.asarray(values).tolist(), cls, scale=scale, fine=True)


def _is_extreme(log_estimate: Any, se: Any) -> bool:
    """A term without a maximum of the likelihood (biostats.regression.extreme_estimates): its numbers are not
    compared."""
    a, b = _finite(log_estimate), _finite(se)
    return a is not None and b is not None and abs(a) > EXTREME_LOG and b > EXTREME_SE


def _cell_support(cell: Sequence[int], kind: str) -> dict[str, int]:
    """
    The counts behind a cell (a factor's level, an indicator's value): its rows, its events and its rows without
    the event (``non_events`` for a logistic model, ``censored`` for a Cox model); for a count model the rows with
    an event and the rows without one, since its events are not people. ``cell`` is (rows, events, rows with an
    event) from a model, with the rows without an event as a fourth count from a pooled one (the smallest of each
    over the datasets). A cell where nobody, or everybody, had the event says so of each of its participants: the
    small-cell rule of a shared result reads both sides.
    """
    rows, events, with_event = int(cell[0]), int(cell[1]), int(cell[2])
    without = int(cell[3]) if len(cell) > 3 else rows - with_event
    found = {"rows": rows, "events": events}
    if kind == "logistic":
        found["non_events"] = without
    if kind == "cox":
        found["censored"] = without
    if kind == "count":
        found["rows_with_events"] = with_event
        found["rows_without_events"] = without
    return found


@dataclass(frozen=True)
class _Term:
    """What a term of a design stands for: its part's id (without the prefix), its labels, and its cell's counts:
    a level's, the higher value's of a covariate of two values, the rows off the commonest value of a covariate
    that at least half of the rows hold at one value (None for any other numeric covariate: the model's counts are
    behind it)."""

    id: str
    labels: dict[str, str]
    cell: Sequence[int] | None


def _terms_of(
    names: Sequence[str],
    levels: Mapping[str, Mapping[str, Sequence[int]]],
    indicators: Mapping[str, Mapping[str, Sequence[int]]],
    dominant: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
) -> list[_Term]:
    """
    What each term of a design stands for, by its place: the design holds the numeric covariates first, then each
    factor's levels after its reference, so the last terms are the factors' and the others are numeric, whatever
    their names (two factors can give one name to a term: the place tells them apart). ``dominant`` holds, for a
    numeric covariate of more than two values, its commonest value's cell and then the cell of its other rows
    (biostats.design.dominant_counts).
    """
    of_factors = [
        _Term(f"{factor}#{place}", {"variable": factor, "level": str(level)}, cells[level])
        for factor, cells in levels.items()
        for place, level in enumerate(cells)
        if place > 0
    ]
    numeric = []
    for name in list(names)[: len(names) - len(of_factors)]:
        cells = indicators.get(str(name))
        apart = (dominant or {}).get(str(name))
        if (
            cells
        ):  # two values: the term compares the higher with the lower, as a level with its reference
            high = list(cells)[1]
            numeric.append(_Term(str(name), {"variable": str(name), "level": high}, cells[high]))
        elif apart:
            # one value for at least half of the rows: the term is carried by the rows off it
            numeric.append(_Term(str(name), {"variable": str(name)}, apart[list(apart)[1]]))
        else:
            numeric.append(_Term(str(name), {"variable": str(name)}, None))
    return numeric + of_factors


def _reference_parts(
    levels: Mapping[str, Mapping[str, Sequence[int]]],
    indicators: Mapping[str, Mapping[str, Sequence[int]]],
    kind: str,
    dominant: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
) -> list[Part]:
    """A part per factor, per covariate of two values and per covariate with a commonest value, for the cell its
    terms are compared with: the first level, the lower value, the commonest value. It has no number of its own;
    its counts are behind every term of its variable."""
    parts = []
    for variable, cells in (*levels.items(), *indicators.items(), *(dominant or {}).items()):
        names = list(cells)
        if not names:
            continue
        parts.append(
            Part(
                id=f"reference:{variable}",
                support=_cell_support(cells[names[0]], kind),
                labels={"variable": str(variable), "level": str(names[0])},
            )
        )
    return parts


def _model_terms(
    table: pd.DataFrame,
    terms: Sequence[_Term],
    *,
    estimate: str,
    ratio: str,
    cls: str,
    whole: Mapping[str, int],
    kind: str,
) -> list[Part]:
    """A part per term of a fitted model: the estimate on the log scale (allowed a part of its standard error), the
    ratio and its interval, whether the estimate is extreme, and the counts behind the term (its cell's, or the
    model's for a numeric term)."""
    parts = []
    for row, term in zip(table.to_dict("records"), terms, strict=True):
        se = row["se"]
        parts.append(
            Part(
                id=f"term:{term.id}",
                labels=term.labels,
                support=dict(whole) if term.cell is None else _cell_support(term.cell, kind),
                numbers=(
                    Number(estimate, row[estimate], cls, scale=_finite(se)),
                    Number("se", se, cls),
                    Number(ratio, row[ratio], cls, scale=_end_scale(row[ratio], se)),
                    Number("ci_low", row["ci_low"], cls, scale=_end_scale(row["ci_low"], se)),
                    Number("ci_high", row["ci_high"], cls, scale=_end_scale(row["ci_high"], se)),
                    Number("z", row["z"], cls, scale=1.0),
                    Number("p", row["p"], cls, rule=LOG),
                    Number("extreme", _is_extreme(row[estimate], se), EXACT),
                ),
            )
        )
    return parts


def _curve_parts(curves: Sequence[Any]) -> list[Part]:
    parts = []
    for place, curve in enumerate(curves):
        steps = [
            _series("times", curve.times, EXACT),
            _series("at_risk", curve.at_risk, EXACT),
            _series("survival", curve.survival, DETERMINISTIC, scale=1.0),
            _series("lower", curve.lower, DETERMINISTIC, scale=1.0),
            _series("upper", curve.upper, DETERMINISTIC, scale=1.0),
        ]
        if curve.se is not None:
            steps.append(_series("se", curve.se, DETERMINISTIC, scale=1.0))
        if getattr(curve, "events_at", None) is not None:
            # the events at each step and the rows censored until the next: what a shared curve is thinned by
            steps += [
                _series("events_at", curve.events_at, EXACT),
                _series("censored_after", curve.censored_after, EXACT),
            ]
        if getattr(curve, "follow_times", None) is not None:
            # every time a follow-up ends and the rows still followed at it: a shared curve's last point and its
            # follow-up are read on them
            steps += [
                _series("follow_times", curve.follow_times, EXACT),
                _series("followed", curve.followed, EXACT),
            ]
        parts.append(
            Part(
                id=f"curve#{place}",
                labels={"group": str(curve.label)},
                support={
                    "rows": int(curve.n),
                    "events": int(curve.events),
                    "censored": int(curve.n) - int(curve.events),
                },
                numbers=(
                    Number("rows", curve.n, EXACT),
                    Number("events", curve.events, EXACT),
                    Number("confidence", curve.level, EXACT),
                    # event times, or the middle of two of them: one participant's
                    Number("median", curve.median, EXACT, fine=True),
                    Number("median_lower", curve.median_lower, EXACT, fine=True),
                    Number("median_upper", curve.median_upper, EXACT, fine=True),
                    Number("last_time", curve.last_time, EXACT, fine=True),
                    *steps,
                ),
            )
        )
    return parts


def _record_survival(result: Any) -> tuple[list[Part], None]:
    parts = _curve_parts(result.curves)
    parts.append(
        Part(
            id="data",
            numbers=(
                Number("groups_skipped", getattr(result, "groups_skipped", 0), EXACT),
                # the rows of groups of one row: a count of participants, like a level's
                Number("rows_skipped", getattr(result, "skipped_rows", 0), EXACT, fine=True),
                Number("rows_left_out", getattr(result, "rows_left_out", 0), EXACT, fine=True),
            ),
        )
    )
    if result.logrank_statistic is not None:
        parts.append(
            Part(
                id="test:log_rank",
                support={
                    "rows": sum(c.n for c in result.curves),
                    "events": sum(c.events for c in result.curves),
                },
                numbers=(
                    Number("statistic", result.logrank_statistic, DETERMINISTIC, scale=1.0),
                    Number("df", result.logrank_df, EXACT),
                    Number("p", result.logrank_p, DETERMINISTIC, rule=LOG),
                ),
            )
        )
    return parts, None


def _record_cox(result: Any) -> tuple[list[Part], None]:
    whole = {
        "rows": int(result.n),
        "events": int(result.events),
        "censored": int(result.n) - int(result.events),
    }
    levels = getattr(result, "level_counts", {}) or {}
    indicators = getattr(result, "indicator_counts", {}) or {}
    dominant = getattr(result, "dominant_counts", {}) or {}
    terms = _terms_of(list(result.table["term"]), levels, indicators, dominant)
    parts = [
        Part(
            id="model",
            support={**whole, **_df(result.df_resid)},
            numbers=(
                Number("rows", result.n, EXACT),
                Number("events", result.events, EXACT),
                Number("ties", result.ties, EXACT),
                Number("df_resid", result.df_resid, EXACT),
                Number("converged", bool(getattr(result, "converged", True)), EXACT),
                Number("rows_left_out", getattr(result, "rows_left_out", 0), EXACT, fine=True),
                # lifelines' own fit: it stops earlier than the model's, and the concordance moves by whole pairs
                Number("concordance", result.concordance, ITERATIVE_LOOSE, scale=1.0),
            ),
        ),
        *_reference_parts(levels, indicators, "cox", dominant),
        *_model_terms(
            result.table,
            terms,
            estimate="log_hr",
            ratio="hr",
            cls=ITERATIVE,
            whole=whole,
            kind="cox",
        ),
    ]
    if result.ph_test is not None:
        for row, term in zip(result.ph_test.to_dict("records"), terms, strict=True):
            parts.append(
                Part(
                    id=f"ph:{term.id}",
                    labels=term.labels,
                    support=dict(whole) if term.cell is None else _cell_support(term.cell, "cox"),
                    numbers=(
                        Number("statistic", row["statistic"], ITERATIVE_LOOSE, scale=1.0),
                        Number("p", row["p"], ITERATIVE_LOOSE, rule=LOG),
                        Number("flagged", bool(row["p"] < 0.05), EXACT),
                    ),
                )
            )
    return parts, None


def _df(df_resid: Any) -> dict[str, int]:
    """The residual degrees of freedom as a count behind the model, when there are any."""
    found = _finite(df_resid)
    return {} if found is None else {"df_resid": int(found)}


def _record_regression(kind: str) -> Callable[[Any], tuple[list[Part], None]]:
    def build(result: Any) -> tuple[list[Part], None]:
        logistic = result.kind == "logistic"
        whole = {"rows": int(result.n)}
        if result.events is not None:
            whole["events"] = int(result.events)
            if logistic:
                whole["non_events"] = int(result.n) - int(result.events)
        with_events = getattr(result, "rows_with_events", None)
        if not logistic and with_events is not None:
            # a count model's events are a sum of counts: the people behind them are its rows with an event
            whole["rows_with_events"] = int(with_events)
            whole["rows_without_events"] = int(result.n) - int(with_events)
        model = [
            Number("family", result.kind, EXACT),
            Number("rows", result.n, EXACT),
            Number("events", result.events, EXACT),
            Number("df_resid", result.df_resid, EXACT),
            Number("converged", bool(getattr(result, "converged", True)), EXACT),
            Number("llf", result.llf, ITERATIVE),
            Number("aic", result.aic, ITERATIVE),
        ]
        if logistic:
            model.append(Number("pseudo_r2", result.pseudo_r2, ITERATIVE, scale=1.0))
            model.append(Number("auc", result.auc, ITERATIVE, scale=1.0))
            low, high = result.auc_ci or (None, None)
            model.append(Number("auc_ci_low", low, ITERATIVE, scale=1.0))
            model.append(Number("auc_ci_high", high, ITERATIVE, scale=1.0))
        else:
            # a negative binomial model's dispersion is a parameter of the fit, with its standard error; a Poisson
            # model's is a ratio of two sums
            model.append(
                Number(
                    "dispersion",
                    result.dispersion,
                    ITERATIVE,
                    scale=_finite(getattr(result, "dispersion_se", None)),
                )
            )
        levels = getattr(result, "level_counts", {}) or {}
        indicators = getattr(result, "indicator_counts", {}) or {}
        dominant = getattr(result, "dominant_counts", {}) or {}
        terms = _terms_of(list(result.table["term"]), levels, indicators, dominant)
        return [
            Part(id="model", support={**whole, **_df(result.df_resid)}, numbers=tuple(model)),
            *_reference_parts(levels, indicators, kind, dominant),
            *_model_terms(
                result.table,
                terms,
                estimate="coef",
                ratio="odds_ratio" if logistic else "rate_ratio",
                cls=ITERATIVE,
                whole=whole,
                kind=kind,
            ),
        ], None

    return build


def _pooling(result: Any) -> Part:
    return Part(
        id="pooling",
        numbers=(
            Number("m", result.m, EXACT),
            Number("m_total", result.m_total, EXACT),
            Number("partial", bool(result.partial), EXACT),
        ),
    )


def _pooled_terms(
    table: Any, terms: Sequence[_Term], whole: Mapping[str, int], kind: str
) -> list[Part]:
    """A part per pooled term: Rubin's quantities, the estimate of each imputed dataset, whether it is extreme, and
    the counts behind it. The intercept is no term of a result: with factors only it is the reference cell's own
    proportion."""
    estimates = np.asarray(table.estimates, dtype=float)
    kept = [(j, term) for j, term in enumerate(table.terms) if str(term.term) != "const"]
    parts = []
    for (j, term), plan in zip(kept, terms, strict=True):
        se = _finite(term.se)
        natural = term.natural_estimate
        numbers = [
            Number("estimate", term.estimate, ITERATIVE, scale=se),
            Number("se", term.se, ITERATIVE),
            Number("statistic", term.statistic, ITERATIVE, scale=1.0),
            Number("df", term.df, ITERATIVE, rule=RECIPROCAL),
            Number("p", term.p_value, ITERATIVE, rule=LOG),
            Number("ci_low", term.ci.lower, ITERATIVE, scale=se),
            Number("ci_high", term.ci.upper, ITERATIVE, scale=se),
            Number("ubar", term.ubar, ITERATIVE),
            Number("b", term.b, ITERATIVE, scale=None if se is None else se**2),
            Number("t", term.t, ITERATIVE),
            Number("riv", term.riv, ITERATIVE, scale=1.0),
            Number("lambda", term.lam, ITERATIVE, scale=1.0),
            Number("fmi", term.fmi, ITERATIVE, scale=1.0),
            Number("dfcom", term.dfcom, EXACT),
            Number("m", term.m, EXACT),
            Number("scale", term.scale, EXACT),
            Number("estimates", estimates[:, j].tolist(), ITERATIVE, scale=se),
            Number("extreme", _is_extreme(term.estimate, term.se), EXACT),
        ]
        if natural is not None:
            numbers.append(
                Number("natural", natural, ITERATIVE, scale=_end_scale(natural, term.se))
            )
        if term.natural_ci is not None:
            for name, end in (
                ("natural_low", term.natural_ci.lower),
                ("natural_high", term.natural_ci.upper),
            ):
                numbers.append(Number(name, end, ITERATIVE, scale=_end_scale(end, term.se)))
        parts.append(
            Part(
                id=f"term:{plan.id}",
                labels=plan.labels,
                support=dict(whole) if plan.cell is None else _cell_support(plan.cell, kind),
                numbers=tuple(numbers),
            )
        )
    return parts


def _pooled_tests(
    tests: Mapping[str, Any], cls: str, whole: Mapping[str, int], named: Mapping[str, list[str]]
) -> list[Part]:
    """A part per pooled test (D1 of a factor's terms, D2 of a chi-square): the terms it tests are named by their
    parts' ids, never by the design's names, which hold levels."""
    parts = []
    for name, test in tests.items():
        parts.append(
            Part(
                id=f"test:{name}",
                labels={"variable": str(name)} if name in named else {},
                support=dict(whole),
                numbers=(
                    Number("method", test.method, EXACT),
                    Number("terms", named.get(name, []), EXACT),
                    Number("statistic", test.statistic, cls, scale=1.0),
                    Number("df1", test.df1, EXACT),
                    Number("df2", test.df2, cls, rule=RECIPROCAL),
                    Number("p", test.p_value, cls, rule=LOG),
                    Number("riv", test.riv, cls, scale=1.0),
                    Number("m", test.m, EXACT),
                ),
            )
        )
    return parts


def _pooled_model(kind: str) -> Callable[[Any], tuple[list[Part], ResultRecord | None]]:
    """The record of a pooled Cox, logistic or count model."""

    def build(result: Any) -> tuple[list[Part], ResultRecord | None]:
        extra = result.extra
        family = "cox" if kind == "cox" else str(extra.get("family", kind))
        rows, events = extra.get("n"), extra.get("events")
        whole: dict[str, int] = {}
        sizes: list[Number] = []
        # each count is the smallest over the imputed datasets, and is named so: they do not add up
        if rows is not None:
            whole["rows_min"] = int(rows[0])
            sizes += [Number("rows_min", rows[0], EXACT), Number("rows_max", rows[1], EXACT)]
        if events is not None:
            whole["events_min"] = int(events[0])
            sizes += [
                Number("events_min", events[0], EXACT),
                Number("events_max", events[1], EXACT),
            ]
            if kind in ("logistic", "cox") and rows is not None:
                # the rows without the event: at least this many in every dataset
                whole["non_events_min" if kind == "logistic" else "censored_min"] = int(
                    rows[0]
                ) - int(events[1])
        with_events = extra.get("rows_with_events")
        if kind == "count" and with_events is not None and rows is not None:
            whole["rows_with_events_min"] = int(with_events[0])
            whole["rows_without_events_min"] = int(rows[0]) - int(with_events[1])
        if extra.get("df_resid") is not None:
            whole["df_resid_min"] = int(extra["df_resid"])
        if kind == "cox":
            # the rows a dataset's model left out for a follow-up time below zero: a count of participants
            sizes.append(Number("rows_left_out", extra.get("rows_left_out", 0), EXACT, fine=True))
        levels = extra.get("level_counts") or {}
        names = (
            [str(t.term) for t in result.table.terms if str(t.term) != "const"]
            if result.table
            else []
        )
        # where no imputed dataset keeps the two values of a covariate, its cells are those of its recorded
        # values; the commonest value of a covariate of more values is read on recorded values always
        recorded = extra.get("recorded_cells") or {}
        indicators = {
            name: cells
            for name, cells in {
                **(recorded.get("indicator") or {}),
                **(extra.get("indicator_counts") or {}),
            }.items()
            if name in names
        }
        dominant = {
            name: cells
            for name, cells in (recorded.get("dominant") or {}).items()
            if name in names and name not in indicators
        }
        terms = _terms_of(names, levels, indicators, dominant)
        of_factor: dict[str, list[str]] = {}
        for term in terms:
            if "#" in term.id:
                of_factor.setdefault(term.labels["variable"], []).append(f"term:{term.id}")
        observed = KINDS[kind].record
        complete = result.complete_case
        return [
            _pooling(result),
            Part(
                id="model",
                support=whole,
                numbers=(
                    *([Number("family", family, EXACT)] if kind != "cox" else []),
                    *sizes,
                    # the datasets whose fit reached a maximum: a term of a fit that did not is not compared
                    Number("converged_in", extra.get("converged_in", result.m), EXACT),
                ),
            ),
            *_reference_parts(levels, indicators, kind, dominant),
            *(_pooled_terms(result.table, terms, whole, kind) if result.table is not None else []),
            *_pooled_tests(result.tests, ITERATIVE, whole, of_factor),
        ], (
            _record_of(kind, complete, observed)
            if observed is not None and complete is not None
            else None
        )

    return build


def _record_of(
    kind: str, result: Any, build: Callable[[Any], tuple[list[Part], Any]]
) -> ResultRecord:
    """The record of an observed result of ``kind``, as :func:`record` stamps it."""
    parts, _ = build(result)
    return ResultRecord(
        kind,
        parts=tuple(parts),
        warnings=tuple(getattr(result, "warnings", ()) or ()),
        record_version=KINDS[kind].record_version,
    )


def _pooled_survival(result: Any) -> tuple[list[Part], ResultRecord | None]:
    extra = result.extra
    sizes = extra.get("sizes") or {}
    events = extra.get("events") or {}
    medians = extra.get("medians") or {}
    intervals = extra.get("median_ci") or {}
    counts = extra.get("grid_counts") or {}
    censored = extra.get("censored") or {}
    followed = extra.get("followed") or {}
    parts = [_pooling(result)]
    for place, (label, curve) in enumerate((extra.get("curves") or {}).items()):
        low, high = intervals.get(label) or (None, None)
        # each the smallest over the imputed datasets, and named so: the three do not add up
        behind = {}
        if label in sizes:
            behind["rows_min"] = int(sizes[label])
        if label in events:
            behind["events_min"] = int(events[label])
        if label in censored:
            behind["censored_min"] = int(censored[label])
        steps = [
            _series("times", curve["time"], EXACT),
            _series("survival", curve["survival"], DETERMINISTIC, scale=1.0),
            _series("lower", curve["lower"], DETERMINISTIC, scale=1.0),
            _series("upper", curve["upper"], DETERMINISTIC, scale=1.0),
            _series("fmi", curve["fmi"], DETERMINISTIC, scale=1.0),
        ]
        if label in counts:
            # at each time: the fewest events so far and the fewest rows still at risk over the datasets
            steps += [
                _series("events_so_far_min", counts[label]["events"], EXACT),
                _series("at_risk_min", counts[label]["at_risk"], EXACT),
                # the same in each dataset: the events between two times are counted in each (never shared)
                Number(
                    "events_so_far_each", counts[label].get("events_each") or [], EXACT, fine=True
                ),
            ]
        if (followed.get(label) or {}).get("times"):
            # at every time a follow-up ends in a dataset: the fewest rows still followed
            steps += [
                _series("follow_times", followed[label]["times"], EXACT),
                _series("followed_min", followed[label]["rows"], EXACT),
            ]
        parts.append(
            Part(
                id=f"curve#{place}",
                labels={"group": str(label)},
                support=behind,
                numbers=(
                    *([Number("rows_min", sizes[label], EXACT)] if label in sizes else []),
                    *([Number("events_min", events[label], EXACT)] if label in events else []),
                    Number("median", medians.get(label), EXACT, fine=True),
                    Number("median_lower", low, EXACT, fine=True),
                    Number("median_upper", high, EXACT, fine=True),
                    *steps,
                ),
            )
        )
    parts.append(
        Part(
            id="data",
            numbers=(
                # the rows in no curve (a group too small to draw, a follow-up time below zero), the most over the
                # datasets: counts of participants, like a level's
                Number("rows_skipped", extra.get("rows_skipped", 0), EXACT, fine=True),
                Number("rows_left_out", extra.get("rows_left_out", 0), EXACT, fine=True),
            ),
        )
    )
    total = extra.get("total") or {}
    whole = {f"{key}_min": int(total[key]) for key in ("rows", "events") if key in total}
    parts += _pooled_tests(result.tests, DETERMINISTIC, whole, {})
    complete = result.complete_case
    return parts, (None if complete is None else _record_of("survival", complete, _record_survival))


# ----------------------------------------------------------------------------- the catalogue

_MODEL_LIBRARIES = ("statsmodels", "scipy", "numpy", "pandas")

KINDS: dict[str, AnalysisKind] = {
    kind.id: kind
    for kind in (
        AnalysisKind(
            "survival",
            "Kaplan-Meier survival and log-rank test",
            "Kaplan-Meier survival and log-rank test, pooled",
            roles=("time", "event", "group"),
            options={"alpha": 0.05},
            libraries=_MODEL_LIBRARIES,
            run=_run_survival,
            pool=_pool_survival,
            record=_record_survival,
            pooled_record=_pooled_survival,
        ),
        AnalysisKind(
            "cox",
            "Cox proportional-hazards model",
            "Cox proportional-hazards model, pooled",
            roles=("time", "event", "compare", "adjust"),
            options={"alpha": 0.05, "ties": "efron"},
            pooled_options={"ph_test_imputations": 5},
            libraries=(*_MODEL_LIBRARIES, "lifelines"),
            run=_run_cox,
            pool=_pool_cox,
            record=_record_cox,
            pooled_record=_pooled_model("cox"),
            optional={"lifelines": ("ph:", "model/concordance")},
        ),
        AnalysisKind(
            "logistic",
            "Logistic regression",
            "Logistic regression, pooled",
            roles=("outcome", "compare", "adjust"),
            options={"alpha": 0.05, "event_level": None},
            libraries=_MODEL_LIBRARIES,
            run=_run_logistic,
            pool=_pool_logistic,
            record=_record_regression("logistic"),
            pooled_record=_pooled_model("logistic"),
        ),
        AnalysisKind(
            "count",
            "Count regression",
            "Count regression, pooled",
            roles=("count", "person_time", "compare", "adjust"),
            options={"alpha": 0.05, "family": "poisson"},
            libraries=_MODEL_LIBRARIES,
            run=_run_count,
            pool=_pool_count,
            record=_record_regression("count"),
            pooled_record=_pooled_model("count"),
        ),
        AnalysisKind("univariate", "Group comparison", "Group comparison, pooled"),
        AnalysisKind("ancova", "ANCOVA", "ANCOVA, pooled"),
        AnalysisKind("paired", "Paired test", "Paired t-test, pooled"),
        AnalysisKind("one_sample", "One-sample test", "One-sample t-test, pooled"),
        AnalysisKind("mcnemar", "McNemar's test"),
        AnalysisKind("mantel_haenszel", "Mantel-Haenszel stratified analysis"),
        AnalysisKind("diagnostic", "Diagnostic accuracy"),
        AnalysisKind("bland_altman", "Bland-Altman agreement"),
    )
}


__all__ = [
    "CLASSES",
    "DETERMINISTIC",
    "EXACT",
    "ITERATIVE",
    "ITERATIVE_LOOSE",
    "KINDS",
    "LOG",
    "PLAIN",
    "POOLED_SUFFIX",
    "RECIPROCAL",
    "RULES",
    "SEEDED",
    "AnalysisKind",
    "AnalysisSpec",
    "Computed",
    "ComputedOn",
    "ImputationUsed",
    "LaterVersion",
    "Number",
    "Part",
    "ResultRecord",
    "canonical",
    "kind_named",
    "kind_of",
    "read_spec",
    "record",
    "run",
    "runnable",
    "specify",
    "titles",
]
