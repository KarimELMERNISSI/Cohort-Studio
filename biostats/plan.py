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

- **A calculation from numbers typed** (a share, rates, observed against expected) is the same kind with a third
  source, beside the observed rows and imputed data: its spec holds the numbers (``AnalysisSpec.typed``) in place of
  what it was computed on, and :func:`run` computes it from them alone. It is a function of its numbers: never stale,
  and never among the results of the dataset (its record is marked ``typed``).

Specs and records exist for the time-to-event, logistic and count analyses, and for the calculations from numbers
typed; the other kinds have their titles and get their runners later (until then their results have no spec and a
package lists them as not run).
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

# the type of a number typed into a calculation (stored ids): a count of participants or of events, or a quantity
# (person-time, an expected number)
COUNT, QUANTITY = "count", "quantity"
# what a spec is computed on (stored ids)
OBSERVED, IMPUTED, TYPED = "observed", "imputed", "typed"

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


# the keys each part of a spec or a record holds, as its to_dict writes them: a reader refuses any other, so that
# a key a later version writes is never read as if it were not there (its meaning lost without a word)
IMPUTATION_KEYS = frozenset({"cells_fp", "m", "subset", "step_index", "step_hash"})
COMPUTED_ON_KEYS = frozenset(
    {
        "mode",
        "data_fp",
        "n_population",
        "n_analysed",
        "rows_fp",
        "protocol_hash",
        "rules_version",
        "trace_head",
        "trace_steps",
        "imputation",
    }
)
SPEC_KEYS = frozenset(
    {
        "kind",
        "kind_version",
        "pooled",
        "variables",
        "options",
        "decided",
        "random",
        "backend",
        "data",
        "typed",  # written only for a calculation from numbers typed
    }
)
NUMBER_KEYS = frozenset({"name", "value", "class", "scale", "rule", "fine"})
PART_KEYS = frozenset({"id", "labels", "support", "numbers"})
RECORD_KEYS = frozenset(
    # "typed" is written only for a calculation from numbers typed
    {"kind", "record_version", "pooled", "parts", "warnings", "complete_case", "typed"}
)


def _closed(value: Mapping[str, Any], keys: frozenset[str], what: str) -> None:
    """ValueError when ``value`` holds a key that ``what`` never holds in this version (written by a later one).
    The key is not named: a received file can hold anything."""
    if isinstance(value, Mapping) and set(value) - keys:
        raise ValueError(
            f"the {what} holds a key this version does not know (written by a later version)"
        )


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
        _closed(value, IMPUTATION_KEYS, "imputation")
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
        _closed(value, COMPUTED_ON_KEYS, "description of what an analysis was computed on")
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

    ``typed`` holds the numbers of a calculation from numbers typed, by name (the kind's ``typed_inputs``): such a
    spec names no variable, is never pooled, draws nothing and is computed on no data; its options are the kind's
    ``typed_options``. Each number is converted to its declared type when the spec is built (1200 and 1200.0 are one
    calculation).
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
    typed: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        kind = kind_named(self.kind)
        if self.typed is not None:
            if not kind.typed_inputs:
                raise ValueError(
                    f"the analysis kind {self.kind} is not computed from numbers typed"
                )
            if self.pooled or self.variables or self.random is not None or self.data is not None:
                raise ValueError(
                    "a calculation from numbers typed names no variable, is never pooled, draws nothing and is "
                    "computed on no data"
                )
            object.__setattr__(self, "typed", _numbers_typed(kind, self.typed))
            offered = dict(kind.typed_options)
        else:
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
        options = dict(self.options)
        for name, known in kind.choices.items():
            if name not in options:
                continue
            # the option named, never the value: a value this version does not know was written by a later one
            members = [value for value in known if _same_value(options[name], value)]
            if not members:
                raise ValueError(
                    f"the {self.kind} analysis does not know this value of its option {name} (written by a later "
                    "version)"
                )
            options[name] = members[
                0
            ]  # 1000.0 is the multiplier 1000: one request, written one way
        if self.typed is not None:
            for name, value in options.items():
                if name in kind.choices or value is None or isinstance(value, str):
                    continue
                # a calculation's own number among its options (a reference share): finite, and of one type
                if not _holds_a_number(value):
                    raise ValueError(
                        f"the option {name} of the {self.kind} calculation is not a finite number"
                    )
                options[name] = float(value)
        object.__setattr__(self, "options", options)
        for name in ("variables", "options", "decided", "backend"):
            object.__setattr__(self, name, _plain(getattr(self, name)))
        if self.random is not None:
            object.__setattr__(self, "random", _plain(self.random))

    def request(self) -> dict[str, Any]:
        """What was asked, apart from what it ran on and with: the kind, the variables, the options, the draws;
        for a calculation, its numbers (written only then: a spec of the dataset keeps its JSON and its id)."""
        asked = {
            "kind": self.kind,
            "kind_version": self.kind_version,
            "pooled": self.pooled,
            "variables": copy.deepcopy(dict(self.variables)),
            "options": copy.deepcopy(dict(self.options)),
            "random": None if self.random is None else copy.deepcopy(dict(self.random)),
        }
        if self.typed is not None:
            asked["typed"] = dict(self.typed)
        return asked

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
        the value), or for a kind, a role, an option or a key this version does not know."""
        _closed(value, SPEC_KEYS, "analysis spec")
        data = _typed(value, "data", dict, optional=True)
        random = _typed(value, "random", dict, optional=True)
        numbers = _typed(value, "typed", dict, optional=True)
        return cls(
            typed=None if numbers is None else _restored(numbers),
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

    @property
    def source(self) -> str:
        """What the spec is computed on: the numbers typed, imputed data, or the observed rows."""
        if self.typed is not None:
            return TYPED
        return IMPUTED if self.pooled else OBSERVED

    def on(self, data: ComputedOn) -> AnalysisSpec:
        """The spec with what it was computed on."""
        if self.typed is not None:
            raise ValueError("a calculation from numbers typed is computed on no data")
        return replace(self, data=data)

    def covariates(self) -> list[str]:
        """A model's covariates: the comparison first, then the adjustment (a variable named twice, once)."""
        compare = self.variables.get("compare")
        first = [compare] if compare else []
        return first + [c for c in self.variables.get("adjust") or [] if c not in first]


def _holds_a_number(value: Any) -> bool:
    """Whether ``value`` is a number with a value a float can hold: not a yes or no, not a text, not NaN or
    infinite, and not a whole number of hundreds of digits (which raises where it is converted)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _same_value(given: Any, known: Any) -> bool:
    """Whether an option's value is this member of its closed set: equal, a yes or no never being a number."""
    return isinstance(given, bool) == isinstance(known, bool) and given == known


def _numbers_typed(kind: AnalysisKind, typed: Mapping[str, Any]) -> dict[str, int | float]:
    """
    The numbers of a calculation, each converted to its declared type before anything reads them: a count is a whole
    number of 0 or more (1200.0 is the count 1200; a fraction, a negative number, a yes or no are refused), a
    quantity a finite number. ValueError names the number, never its value.
    """
    if not isinstance(typed, Mapping):
        raise ValueError("the numbers typed are not of the type expected")
    if set(typed) - set(kind.typed_inputs):
        # not named: a received file can hold anything there
        raise ValueError(
            f"the {kind.id} calculation holds a number this version does not know; its numbers: "
            f"{', '.join(kind.typed_inputs)}"
        )
    optional = set(kind.typed_optional)
    missing = [name for name in kind.typed_inputs if name not in optional and name not in typed]
    if missing:
        raise ValueError(f"the {kind.id} calculation lacks its number {', '.join(missing)}")
    if optional & set(typed) not in (set(), optional):
        raise ValueError(
            f"the numbers {', '.join(kind.typed_optional)} of the {kind.id} calculation are given together or not "
            "at all"
        )
    numbers: dict[str, int | float] = {}
    for name, declared in kind.typed_inputs.items():
        if name not in typed:
            continue
        value = typed[name]
        if isinstance(value, np.generic):
            value = value.item()
        if not _holds_a_number(value):
            raise ValueError(
                f"the number {name} of the {kind.id} calculation is not a finite number"
            )
        if declared == COUNT:
            if value < 0 or value != int(value):
                raise ValueError(
                    f"the number {name} of the {kind.id} calculation is not a count (a whole number of 0 or more)"
                )
            numbers[name] = int(value)
        else:
            numbers[name] = float(value)
    return numbers


def read_spec(value: Any) -> tuple[AnalysisSpec | None, str]:
    """
    A spec from a received file, or the reason it cannot be read here: a kind, a role, an option or a key this
    version does not know (the package was made by a later one), a value of another type. The reason names keys and kinds, never
    a value: a check of the package then reads "not run" with it.
    """
    if not isinstance(value, Mapping):
        return None, "not an analysis spec"
    # a later version first: what it added (a number, an option) would otherwise be read as a defect of the file
    known, written = KINDS.get(str(value.get("kind"))), value.get("kind_version")
    if known is not None and isinstance(written, int) and not isinstance(written, bool):
        current = known.typed_version if value.get("typed") is not None else known.version
        if written > current:
            return None, f"the {known.id} analysis was specified by a later version of CohortStudio"
    try:
        spec = AnalysisSpec.from_dict(value)
    except (ValueError, TypeError, ArithmeticError, RecursionError) as exc:
        return None, str(exc) if isinstance(exc, ValueError) else "holds a value that is not JSON"
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
        _closed(value, NUMBER_KEYS, "number of a result")
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
        _closed(value, PART_KEYS, "part of a result")
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
    carries for comparison. ``typed`` marks the record of a calculation from numbers typed (written only then): it
    is never pooled, and no reader takes it for a result of the dataset.
    """

    kind: str
    pooled: bool = False
    parts: tuple[Part, ...] = ()
    warnings: tuple[str, ...] = ()
    complete_case: ResultRecord | None = None
    record_version: int = 1
    typed: bool = False

    def __post_init__(self) -> None:
        kind_named(self.kind)
        if self.typed and (self.pooled or self.complete_case is not None):
            raise ValueError("the record of a calculation from numbers typed is never pooled")
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
        written: dict[str, Any] = {
            "kind": self.kind,
            "record_version": self.record_version,
            "pooled": self.pooled,
            "parts": [p.to_dict() for p in self.parts],
            "warnings": list(self.warnings),
            "complete_case": None if self.complete_case is None else self.complete_case.to_dict(),
        }
        if self.typed:  # only then: a record of the dataset keeps its JSON
            written["typed"] = True
        return written

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResultRecord:
        _closed(value, RECORD_KEYS, "result record")
        inner = _typed(value, "complete_case", dict, optional=True)
        return cls(
            kind=_typed(value, "kind", str),
            record_version=_typed(value, "record_version", int),
            pooled=_typed(value, "pooled", bool),
            parts=tuple(Part.from_dict(p) for p in _typed(value, "parts", list)),
            warnings=tuple(_typed(value, "warnings", list)),
            complete_case=None if inner is None else cls.from_dict(inner),
            typed=bool(_typed(value, "typed", bool, optional=True)),
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

    A kind may declare its roles and options before its runner (for its record's labels). Its request id then moves
    with the declaration until the first spec of the kind is written (by its runner, a typed source or the package
    writer); from then on, a change of the declaration moves ``version``.

    ``choices`` closes the values of the options that hold stored ids (a sampling, a form, a design): a spec holding
    another value is refused with the option named, and a received one reads as written by a later version, where
    its runner would have raised on a word. A value added to a closed set moves ``version``.

    A kind with ``typed_inputs`` can be computed from numbers typed: each number by name (a stored id) with its type
    (``COUNT`` or ``QUANTITY``); ``typed_optional`` names those left out together (the reference group of one rate);
    ``typed_options`` are the options of the calculation, with their defaults (their closed values are in
    ``choices`` too); ``typed_run`` computes it from the spec alone. The kind's ``record`` is shared by its sources:
    the share of a column and the share of two numbers typed are one record shape.
    """

    id: str
    title: str
    pooled_title: str | None = None  # None: never pooled
    roles: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)  # with their defaults
    pooled_options: Mapping[str, Any] = field(default_factory=dict)  # of the pooled analysis only
    choices: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)  # the closed values
    libraries: tuple[str, ...] = ()  # whose versions the spec records
    run: (
        Callable[[AnalysisSpec, pd.DataFrame], tuple[Any, dict[str, Any], dict[str, Any]]] | None
    ) = None
    pool: Callable[[AnalysisSpec, Any], tuple[Any, dict[str, Any], dict[str, Any]]] | None = None
    record: Callable[[Any], tuple[list[Part], ResultRecord | None]] | None = None
    pooled_record: Callable[[Any], tuple[list[Part], ResultRecord | None]] | None = None
    optional: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    typed_title: str | None = (
        None  # of a calculation from numbers typed; None: not computed from them
    )
    typed_inputs: Mapping[str, str] = field(
        default_factory=dict
    )  # each number typed: COUNT or QUANTITY
    typed_optional: tuple[str, ...] = ()  # those left out together
    typed_options: Mapping[str, Any] = field(default_factory=dict)  # with their defaults
    typed_run: Callable[[AnalysisSpec], tuple[Any, dict[str, Any], dict[str, Any]]] | None = None
    # of a typed spec: its numbers, its options, their closed values, what typed_run computes. Apart from
    # ``version``: declaring the kind's roles for the dataset later must not make this version unable to read a
    # calculation it computes the same way
    typed_version: int = 1
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
    kind: str,
    *,
    pooled: bool = False,
    typed: Mapping[str, Any] | None = None,
    options: Mapping[str, Any] | None = None,
    **variables: Any,
) -> AnalysisSpec:
    """A spec of ``kind`` for these variables (by role), or for these numbers ``typed``, with the kind's default
    options (those of the calculation for numbers typed) unless ``options`` says otherwise."""
    found = kind_named(kind)
    if typed is not None:
        defaults = dict(found.typed_options)
    else:
        defaults = {**found.options, **(found.pooled_options if pooled else {})}
    return AnalysisSpec(
        kind=kind,
        kind_version=found.version if typed is None else found.typed_version,
        pooled=pooled,
        variables=variables,
        options={**defaults, **dict(options or {})},
        typed=typed,
    )


def runnable(kind: str, *, pooled: bool = False, typed: bool = False) -> bool:
    """Whether :func:`run` can compute this kind, on the rows, on imputed data or from numbers typed (the others
    have their titles only for now)."""
    found = KINDS.get(kind)
    if found is None:
        return False
    if typed:
        return found.typed_run is not None
    return (found.pool if pooled else found.run) is not None


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
    Compute ``spec``: on ``frame`` (the rows of the analysis population), on the imputed data ``mi`` when it is
    pooled, or from its numbers alone when they were typed (no frame and no imputed data are then taken: a
    calculation reads no data). The spec returned holds the decisions the run used and the libraries' versions. The factor levels a spec
    already holds are used as recorded (the same terms, the same reference); what the data give on their own is
    returned beside (``derived``), for the caller to compare. :class:`LaterVersion` for a spec written by a later
    version of its kind.
    """
    kind = kind_named(spec.kind)
    if spec.kind_version > (kind.version if spec.typed is None else kind.typed_version):
        raise LaterVersion(
            f"the {spec.kind} analysis was specified by a later version of CohortStudio"
        )
    if spec.typed is not None:
        if frame is not None or mi is not None:
            raise ValueError("a calculation from numbers typed reads no data")
        if kind.typed_run is None:
            raise ValueError(f"the {spec.kind} calculation cannot be computed from a spec yet")
        result, used, derived = kind.typed_run(spec)
    elif spec.pooled:
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


def record_of(spec: AnalysisSpec, result: Any) -> ResultRecord | None:
    """The record of a result computed from ``spec``, stamped with its kind's record version and with its source
    (a calculation from numbers typed is marked so); None for a kind that has no record yet."""
    kind = kind_named(spec.kind)
    build = kind.pooled_record if spec.pooled else kind.record
    if build is None or result is None:
        return None
    parts, complete_case = build(result)
    return ResultRecord(
        kind.id,
        pooled=spec.pooled,
        parts=tuple(parts),
        warnings=tuple(getattr(result, "warnings", ()) or ()),
        complete_case=complete_case,
        record_version=kind.record_version,
        typed=spec.typed is not None,
    )


# ----------------------------------------------------------------------------- calculations from numbers typed
# Each runner computes from the spec's numbers and options alone, through the function the page's calculator shows.


def _typed_proportion(spec: AnalysisSpec) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from biostats.paired import one_sample_proportion

    numbers, reference = spec.typed or {}, spec.options.get("reference")
    if isinstance(reference, bool) or not isinstance(reference, int | float):
        raise ValueError("a share is compared with a reference share, fixed beforehand")
    if not 0 < reference < 1:
        # at 0 or at 1 every count but one is impossible: nothing to test (the page refuses them too)
        raise ValueError("a reference share lies strictly between 0 and 1")
    if numbers["total"] < 1 or numbers["count"] > numbers["total"]:
        raise ValueError(
            "the number with the characteristic is at most the number of participants, which is at least 1"
        )
    found = one_sample_proportion(
        numbers["count"], numbers["total"], float(reference), alpha=float(spec.options["alpha"])
    )
    return found, {}, {}


def _typed_rates(spec: AnalysisSpec) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from biostats.rates import rates

    numbers, options = spec.typed or {}, spec.options
    found = rates(
        numbers["events"],
        numbers["person_time"],
        numbers.get("reference_events"),
        numbers.get("reference_person_time"),
        per=float(options["per"]),
        alpha=float(options["alpha"]),
        unit=str(options["unit"]),
    )
    return found, {}, {}


def _typed_standardised(spec: AnalysisSpec) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from biostats.rates import standardised_ratio

    numbers = spec.typed or {}
    found = standardised_ratio(
        numbers["observed"],
        numbers["expected"],
        alpha=float(spec.options["alpha"]),
        events_are=str(spec.options["events_are"]),
    )
    return found, {}, {}


def _record_proportion(result: Any) -> tuple[list[Part], None]:
    """
    A share against a reference fixed beforehand (``biostats.paired.one_sample_proportion``), whatever its source:
    the share with its Wilson interval, and the exact binomial test with whether the interval holds the reference
    (the two are two methods, and can part). The support counts those with the characteristic and those without it:
    a count typed is counted as a participant's.
    """
    total = int(result.n_total)
    count = round(float(result.statistic))
    support = {"with": count, "without": total - count, "rows": total}
    ci, reference = result.estimate_ci, result.reference_value
    holds = ci is not None and reference is not None and ci.lower <= reference <= ci.upper
    return [
        Part(
            "share",
            support=support,
            numbers=(
                Number("count", count, EXACT),
                Number("total", total, EXACT),
                *_with_interval(result.estimate, ci, scale=1.0),
            ),
        ),
        Part(
            "test:exact_binomial",
            support=support,
            numbers=(
                Number("reference", reference, EXACT),
                Number("p", result.p_value, DETERMINISTIC, rule=LOG),
                Number("interval_holds_reference", bool(holds), EXACT),
            ),
        ),
    ], None


def _record_rates(result: Any) -> tuple[list[Part], None]:
    """
    One rate, or two and their ratio (``biostats.rates.RatesResult``): ``rate#0`` is the group compared and
    ``rate#1`` the reference group, each with its events (the part's support), its person-time, and its rate per
    unit of person-time with the exact interval; ``ratio`` holds the ratio with its interval (``bounded`` false
    when the reference group has no event: the upper end is then infinite), the test and its p-value, and the
    difference of the two rates. No ``ratio`` part for one rate, nor for two groups without an event.
    """
    groups = ("compared", "reference")
    parts: list[Part] = []
    for place, rate in enumerate(result.rates):
        parts.append(
            Part(
                f"rate#{place}",
                labels={"group": groups[place]} if len(result.rates) > 1 else {},
                support={"events": int(rate.events)},
                numbers=(
                    Number("events", int(rate.events), EXACT),
                    Number("person_time", float(rate.person_time), DETERMINISTIC),
                    # how the rate is shown, and of what: without them "12.50" is no rate a reader can use
                    Number("unit", str(result.unit), EXACT),
                    Number(
                        "per", int(rate.per) if float(rate.per).is_integer() else rate.per, EXACT
                    ),
                    *_with_interval(
                        rate.rate,
                        rate.ci,
                        scale=math.sqrt(max(rate.events, 1)) / rate.person_time,
                    ),
                ),
            )
        )
    ratio = result.ratio
    if ratio is not None:
        first, second = ratio.exposed, ratio.reference
        spread = math.sqrt(
            max(first.events, 1) / first.person_time**2
            + max(second.events, 1) / second.person_time**2
        )
        parts.append(
            Part(
                "ratio",
                support={
                    "events": int(first.events),
                    "reference_events": int(second.events),
                },
                numbers=(
                    *_with_interval(ratio.ratio, ratio.ci, scale=None),
                    Number("p", ratio.p_value, DETERMINISTIC, rule=LOG),
                    Number("test", str(ratio.test), EXACT),
                    Number("difference", ratio.rate_difference, DETERMINISTIC, scale=spread),
                    Number(
                        "bounded",
                        bool(math.isfinite(ratio.ratio) and math.isfinite(ratio.ci.upper)),
                        EXACT,
                    ),
                ),
            )
        )
    return parts, None


def _record_standardised(result: Any) -> tuple[list[Part], None]:
    """Observed against expected (``biostats.rates.StandardisedRatio``): the observed events (the part's support),
    the expected number taken as known, their ratio with its exact Poisson interval, and the exact p-value."""
    return [
        Part(
            "ratio",
            support={"observed": int(result.observed)},
            numbers=(
                Number("observed", int(result.observed), EXACT),
                Number("expected", float(result.expected), DETERMINISTIC),
                Number("events_are", str(result.events_are), EXACT),  # mortality or incidence
                *_with_interval(result.ratio, result.ci, scale=None),
                Number("p", result.p_value, DETERMINISTIC, rule=LOG),
            ),
        )
    ], None


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
    The cells of a pooled model's numeric covariates, counted on recorded values. ``rows`` are the observed rows the
    analysis reads and ``events`` each one's events, missing where the outcome is not recorded. A covariate is
    counted on the rows that record it, whatever their outcome: each value of a covariate recorded with two values
    or with a few, and the rows on and off the commonest value of one that holds more (biostats.design). A cell is
    its rows, its recorded events, its rows with a recorded event and its rows recorded without one.

    An imputed value is not a recorded one. The small-cell rule of a shared result reads these cells where the
    imputed datasets do not keep a covariate's two values (a normal draw gives it others: one participant coded 1
    would carry the term under the model's counts), and for a covariate of more values always. A participant coded 1
    whose outcome is missing is a row of the cell all the same: counted where the outcome is recorded only, the
    cell did not exist and the term was written.
    """
    from biostats.design import dominant_counts, indicator_counts, value_counts

    happened = np.asarray(events, dtype=float)
    known = ~np.isnan(happened)
    had = np.where(known, happened, 0.0)
    had_none = (known & (happened == 0)).astype(float)
    cells: dict[str, Any] = {"indicator": {}, "values": {}, "dominant": {}}
    for name in covariates:
        if name in levels:
            continue
        held = rows[name].notna().to_numpy(dtype=bool)
        part = rows.loc[held, [name]]
        for key, count in (
            ("indicator", indicator_counts),
            ("values", value_counts),
            ("dominant", dominant_counts),
        ):
            with_events = count(part, [name], had[held])
            without = count(part, [name], had_none[held])
            for covariate, found in with_events.items():
                cells[key][covariate] = {
                    value: (*cell, without[covariate][value][1]) for value, cell in found.items()
                }
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
    from biostats.survival import cox_regression, event_coding, event_indicator

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
    # every row but those a dataset's model leaves out for a follow-up time below zero
    read = mi.base[~(pd.to_numeric(mi.base[v["time"]], errors="coerce") < 0)]
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
    outcome = mi.base[v["outcome"]]
    happened = (outcome.astype(str) == derived["event_level"]).astype(float).where(outcome.notna())
    result.extra["recorded_cells"] = _recorded_cells(mi.base, covariates, happened, levels)
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
    counted = pd.to_numeric(mi.base[v["count"]], errors="coerce")
    result.extra["recorded_cells"] = _recorded_cells(mi.base, covariates, counted, levels)
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


# what the cells of a numeric covariate's term are (the label ``of`` of its reference part): stored ids
VALUES = "values"  # each value of the covariate is a cell: a covariate of a few values
COMMONEST = (
    "commonest"  # the rows on and off its commonest value: a covariate of many values, half at one
)


@dataclass(frozen=True)
class _Term:
    """What a term of a design stands for: its part's id (without the prefix), its labels, and its cell's counts:
    a level's; the higher value's of a covariate of two values; the smallest of each count over the values of a
    covariate of a few values; the rows off the commonest value of a covariate that at least half of the rows hold
    at one value (None for any other numeric covariate: the model's counts are behind it)."""

    id: str
    labels: dict[str, str]
    cell: Sequence[int] | None
    # for a covariate of a few values: the counts of the rows off its commonest value. Its term is a slope, one
    # parameter for every value: the rows at a rare value stay in the fit, and what would leave it with a
    # cell's own parameter (a level, an indicator) leaves it here only when nearly every row holds one value
    off: Sequence[int] | None = None
    # and the fewest rows that carry half of the slope's weight (rows far from the others are fitted nearly
    # exactly by the slope, as a lone row by a level)
    half: int | None = None


def _off_commonest(cells: Sequence[Sequence[int]]) -> tuple[int, int, int, int]:
    """The counts of the rows off the commonest value, of the cells of a covariate's values (rows, events, rows
    with an event, rows without one): every cell but the one that holds the most rows (the first, of two that
    hold as many), added up."""
    whole = [
        (int(c[0]), int(c[1]), int(c[2]), int(c[3]) if len(c) > 3 else int(c[0]) - int(c[2]))
        for c in cells
    ]
    commonest = max(range(len(whole)), key=lambda place: (whole[place][0], -place))
    others = [cell for place, cell in enumerate(whole) if place != commonest]
    return (
        sum(c[0] for c in others),
        sum(c[1] for c in others),
        sum(c[2] for c in others),
        sum(c[3] for c in others),
    )


def _half_rows(cells: Mapping[str, Sequence[int]]) -> int | None:
    """
    Of the cells of a covariate's values: the fewest rows that carry half of its slope's weight or more. A
    value's weight is its rows times the square of its distance from the commonest value, so that each of its
    rows weighs that square: the values are taken from the farthest down (of two as far, the one of fewer rows
    first), and from the heaviest down, and the smaller count is kept (one value that carries half alone can
    hold fewer rows than the farthest ones together). A dose of 0 for 588 rows, 1 for 11 and 50 for one puts
    99.6% of the weight on the participant at 50, whom the slope then fits nearly exactly; doses of 40, 45 and
    50 for one row each put 74% of it on two of them, though none carries nine tenths; beside 40 participants
    at 1, doses of 4, 4 and 5 put 59% of it on three, whom the heaviest value taken first hid. At 600 and
    2,000 rows, half of the weight is carried by 13 rows or more in the usual covariates tried (counts,
    scores, Likert items, bands of age), and by 149 in the tutorial file's hospital visits (130 in the tutorial
    chapter's analysis population); in a study of 100 to 150 participants a skewed count is often carried by
    fewer than 10 (a negative binomial count of visits: 97% of draws at 100 rows, 83% at 150, 18% at 300). None
    when a value does not read as a number, or when every row holds one value.
    """
    try:
        held = [(float(value), int(cell[0])) for value, cell in cells.items()]
    except TypeError, ValueError:
        return None
    if not held or not all(math.isfinite(value) for value, _ in held):
        return None
    commonest = max(range(len(held)), key=lambda place: (held[place][1], -place))
    centre = held[commonest][0]
    weights = [rows * (value - centre) ** 2 for value, rows in held]
    total = sum(weights)
    if not total > 0:
        return None

    def rows_to_half(order: Sequence[int]) -> int:
        carried, rows = 0.0, 0
        for place in order:
            carried += weights[place]
            rows += held[place][1]
            if carried >= total / 2:
                break
        return rows

    places = range(len(held))
    farthest = sorted(
        places, key=lambda place: (-((held[place][0] - centre) ** 2), held[place][1], place)
    )
    heaviest = sorted(places, key=lambda place: (-weights[place], place))
    return min(rows_to_half(farthest), rows_to_half(heaviest))


def _term_support(term: _Term, whole: Mapping[str, int], kind: str) -> dict[str, int]:
    """The counts behind a term's part: the model's for a numeric term without cells, its cell's otherwise; and,
    for a covariate of a few values, those of the rows off its commonest value under names that end in
    ``_off``, and the fewest rows that carry half of the slope's weight (``half_rows``). They are not a cell
    of the term: a rule that reads them asks for them by name."""
    if term.cell is None:
        return dict(whole)
    found = _cell_support(term.cell, kind)
    if term.off is not None:
        found.update({f"{name}_off": n for name, n in _cell_support(term.off, kind).items()})
    if term.half is not None:
        found["half_rows"] = int(term.half)
    return found


def _least(cells: Sequence[Sequence[int]]) -> tuple[int, int, int, int]:
    """The smallest of each count over several cells (rows, events, rows with an event, rows without one): each of
    the cells holds at least these. They are not one cell's counts, and do not add up."""
    whole = [
        (int(c[0]), int(c[1]), int(c[2]), int(c[3]) if len(c) > 3 else int(c[0]) - int(c[2]))
        for c in cells
    ]
    return (
        min(c[0] for c in whole),
        min(c[1] for c in whole),
        min(c[2] for c in whole),
        min(c[3] for c in whole),
    )


def _one_unit_apart(values: Sequence[str]) -> bool:
    """Whether a covariate's two values, as the counts name them, are one unit apart (0 and 1, False and True): its
    ratio per unit is then the ratio of the higher value against the lower."""
    words = {"False": 0.0, "True": 1.0}
    numbers = []
    for value in values:
        text = str(value)
        try:
            numbers.append(words[text] if text in words else float(text))
        except ValueError:
            return False
    return len(numbers) == 2 and abs(abs(numbers[1] - numbers[0]) - 1.0) < 1e-12


PER_UNIT = {"per": "unit"}  # the label of a numeric covariate's term: a ratio for one unit more


def _terms_of(
    names: Sequence[str],
    levels: Mapping[str, Mapping[str, Sequence[int]]],
    indicators: Mapping[str, Mapping[str, Sequence[int]]],
    dominant: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
    values: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
) -> list[_Term]:
    """
    What each term of a design stands for, by its place: the design holds the numeric covariates first, then each
    factor's levels after its reference, so the last terms are the factors' and the others are numeric, whatever
    their names (two factors can give one name to a term: the place tells them apart). ``values`` holds the cell of
    each value of a numeric covariate of a few values (biostats.design.value_counts); ``dominant``, for one of more
    values, its commonest value's cell and then the cell of its other rows (biostats.design.dominant_counts).

    The term of a numeric covariate is a ratio per unit, and is labelled so. Only where its two values are one unit
    apart (0 and 1) is it also the higher value against the lower, and labelled as that contrast: a dose coded 0 or
    50 read "50 against 0" beside a ratio for one unit of dose.
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
        few = (values or {}).get(str(name))
        apart = (dominant or {}).get(str(name))
        slope = {"variable": str(name), **PER_UNIT}
        if cells:
            # two values: the term is carried by the cell of the higher one; one unit apart, it compares the
            # higher with the lower, as a level with its reference
            high = list(cells)[1]
            contrast = {"variable": str(name), "level": high}
            numeric.append(
                _Term(str(name), contrast if _one_unit_apart(list(cells)) else slope, cells[high])
            )
        elif few:
            # a few values: each is a cell, and the term is shown when the smallest of them can be; the rows off
            # the commonest value say whether the others stay in the fit
            cells_of = list(few.values())
            numeric.append(
                _Term(
                    str(name),
                    slope,
                    _least(cells_of[1:]),
                    _off_commonest(cells_of),
                    _half_rows(few),
                )
            )
        elif apart:
            # one value for at least half of the rows: the term is carried by the rows off it
            numeric.append(_Term(str(name), slope, apart[list(apart)[1]]))
        else:
            numeric.append(_Term(str(name), slope, None))
    return numeric + of_factors


def _reference_parts(
    levels: Mapping[str, Mapping[str, Sequence[int]]],
    indicators: Mapping[str, Mapping[str, Sequence[int]]],
    kind: str,
    dominant: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
    values: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
) -> list[Part]:
    """
    A part per variable whose terms have cells behind them, for the other cell: the first level of a factor, the
    lower value of a covariate of two values or of a few, the commonest value of a covariate that half of the rows
    hold at one value. It has no number of its own; its counts are behind every term of its variable.

    A factor's part, and that of a covariate whose two values are one unit apart, is the reference its terms are
    compared with. The others say what they are (``of``: "values", every value of the covariate is a cell;
    "commonest", the rows on and off one value): their term is a ratio per unit, not a contrast with this cell.
    """
    kinds = [
        *((variable, cells, "") for variable, cells in levels.items()),
        *(
            (variable, cells, "" if _one_unit_apart(list(cells)) else VALUES)
            for variable, cells in indicators.items()
        ),
        *((variable, cells, VALUES) for variable, cells in (values or {}).items()),
        *((variable, cells, COMMONEST) for variable, cells in (dominant or {}).items()),
    ]
    parts = []
    for variable, cells, of in kinds:
        names = list(cells)
        if not names:
            continue
        labels = {"variable": str(variable), "level": str(names[0])}
        parts.append(
            Part(
                id=f"reference:{variable}",
                support=_cell_support(cells[names[0]], kind),
                labels={**labels, "of": of} if of else labels,
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
                support=_term_support(term, whole, kind),
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


def _with_interval(value: Any, ci: Any, *, scale: float | None) -> tuple[Number, ...]:
    """An estimate, the ends of its interval, the interval's method and its level, when it has one."""
    numbers = [Number("estimate", value, DETERMINISTIC, scale=scale)]
    if ci is not None:
        numbers += [
            Number("lower", ci.lower, DETERMINISTIC, scale=scale),
            Number("upper", ci.upper, DETERMINISTIC, scale=scale),
            Number("method", str(ci.method), EXACT),
            Number("level", float(ci.level), EXACT),
        ]
    return tuple(numbers)


def _record_diagnostic(result: Any) -> tuple[list[Part], None]:
    """
    The accuracy of a test against a reference standard, as the page keeps it: a mapping with ``acc`` (the measures
    of the 2x2 table; None for a score without a cut-off), ``auc`` (a score), ``youden`` (a cut-off found on the
    data) and ``index_form`` (``number``, a score, when it is absent; ``yes_no``). One record for the forms.

    A part per measure, present when the result holds it: predictive values at the share in the data and the share
    classified correctly for a series only, those at a prevalence when one was given, a likelihood ratio and the
    diagnostic odds ratio when both test results occur. Each part of the table carries the four cells as its
    support, which a shared package's rule reads. A cut-off found on the data is an observed score, and the ends of
    its range are resampled scores: marked as one participant's. The part of the reference standard done unequally
    counts, in its support, the participants without the reference standard by test result: the rule judges each
    count of each support, these two included.
    """
    from biostats.diagnostic import FOUND_CUTOFF, SAMPLING_NOT_STATED, SCORE_FORM, reads_share

    acc, auc = result.get("acc"), result.get("auc")
    form = str(result.get("index_form") or SCORE_FORM)
    labels = {
        "reference": str(result.get("outcome")),
        "index": str(result.get("score")),
        "present": str(result.get("pos_label")),
    }
    if result.get("positive") is not None:
        labels["positive"] = str(result["positive"])
    parts: list[Part] = []
    if auc is not None:
        with_it, without = int(auc["n_positive"]), int(auc["n_negative"])
        parts.append(
            Part(
                id="auc",
                labels=labels,
                support={
                    "rows": with_it + without,
                    "with_condition": with_it,
                    "without_condition": without,
                },
                numbers=(
                    *_with_interval(auc["auc"], auc.get("ci"), scale=1.0),
                    Number("variance", auc.get("variance"), DETERMINISTIC, scale=1.0),
                    Number("with_condition", with_it, EXACT),
                    Number("without_condition", without, EXACT),
                ),
            )
        )
    if acc is None:
        return parts, None
    cells = {"tp": int(acc.tp), "fp": int(acc.fp), "fn": int(acc.fn), "tn": int(acc.tn)}
    support = {
        **cells,
        "rows": sum(cells.values()),
        "with_condition": cells["tp"] + cells["fn"],
        "without_condition": cells["fp"] + cells["tn"],
        "called_positive": cells["tp"] + cells["fp"],
        "called_negative": cells["fn"] + cells["tn"],
    }
    sampling = str(getattr(acc, "sampling", SAMPLING_NOT_STATED))

    def part(part_id: str, *numbers: Number) -> Part:
        return Part(id=part_id, labels=labels, support=support, numbers=numbers)

    parts.append(
        part(
            "table",
            *(Number(name, count, EXACT) for name, count in cells.items()),
            Number("index_form", form, EXACT),
            Number("sampling", sampling, EXACT),
        )
    )
    parts += [
        part("sensitivity", *_with_interval(acc.sensitivity, acc.sensitivity_ci, scale=1.0)),
        part("specificity", *_with_interval(acc.specificity, acc.specificity_ci, scale=1.0)),
    ]
    if reads_share(sampling):
        parts += [
            part("ppv", *_with_interval(acc.ppv, acc.ppv_ci, scale=1.0)),
            part("npv", *_with_interval(acc.npv, acc.npv_ci, scale=1.0)),
            part("accuracy", Number("estimate", acc.accuracy, DETERMINISTIC, scale=1.0)),
        ]
    if acc.prevalence_assumed is not None:
        given = Number("prevalence", float(acc.prevalence_assumed), EXACT)
        for part_id, value, ci in (
            ("ppv_at", acc.ppv_at_prevalence, getattr(acc, "ppv_at_prevalence_ci", None)),
            ("npv_at", acc.npv_at_prevalence, getattr(acc, "npv_at_prevalence_ci", None)),
        ):
            if value is not None:
                parts.append(part(part_id, given, *_with_interval(value, ci, scale=1.0)))
    if acc.lr_positive is not None:
        parts.append(
            part("lr_positive", *_with_interval(acc.lr_positive, acc.lr_positive_ci, scale=None))
        )
    if acc.lr_negative is not None:
        parts.append(
            part("lr_negative", *_with_interval(acc.lr_negative, acc.lr_negative_ci, scale=None))
        )
    if acc.dor is not None:
        parts.append(
            part(
                "dor",
                *_with_interval(acc.dor, acc.dor_ci, scale=None),
                Number("corrected", bool(getattr(acc, "dor_corrected", False)), EXACT),
            )
        )
    parts.append(part("youden", Number("estimate", acc.youden, DETERMINISTIC, scale=1.0)))
    if form == SCORE_FORM:
        found = result.get("mode") == FOUND_CUTOFF
        numbers = [
            Number("mode", str(result.get("mode")), EXACT),
            Number("direction", str(result.get("direction")), EXACT),
            # found on the data, it is a participant's score; typed beforehand, it is nobody's
            Number("value", result.get("cutoff"), EXACT, fine=found),
        ]
        youden = result.get("youden") or {}
        if found and youden.get("ci") is not None:
            numbers += [
                Number("range_lower", youden["ci"].lower, SEEDED, fine=True),
                Number("range_upper", youden["ci"].upper, SEEDED, fine=True),
                Number("resamples", int(youden.get("resamples") or 0), EXACT),
            ]
        parts.append(part("cutoff", *numbers))
    unequal = result.get("verified")
    if unequal is not None and (
        unequal.positive_with_reference < unequal.positive
        or unequal.negative_with_reference < unequal.negative
    ):
        # the rows that hold the test, by its result, and those of them that hold the reference standard; the
        # support counts those without it too: with 10 or more in each of the four numbers, 2 participants could
        # have tested positive without the reference standard
        counts = {
            "positive": int(unequal.positive),
            "positive_with_reference": int(unequal.positive_with_reference),
            "negative": int(unequal.negative),
            "negative_with_reference": int(unequal.negative_with_reference),
        }
        parts.append(
            Part(
                id="verification",
                labels=labels,
                support={
                    **counts,
                    "positive_without_reference": counts["positive"]
                    - counts["positive_with_reference"],
                    "negative_without_reference": counts["negative"]
                    - counts["negative_with_reference"],
                },
                numbers=tuple(Number(name, count, EXACT) for name, count in counts.items()),
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
    values = getattr(result, "value_counts", {}) or {}
    terms = _terms_of(list(result.table["term"]), levels, indicators, dominant, values)
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
        *_reference_parts(levels, indicators, "cox", dominant, values),
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
        values = getattr(result, "value_counts", {}) or {}
        terms = _terms_of(list(result.table["term"]), levels, indicators, dominant, values)
        return [
            Part(id="model", support={**whole, **_df(result.df_resid)}, numbers=tuple(model)),
            *_reference_parts(levels, indicators, kind, dominant, values),
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
                support=_term_support(plan, whole, kind),
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
        values = {
            name: cells
            for name, cells in (recorded.get("values") or {}).items()
            if name in names and name not in indicators
        }
        dominant = {
            name: cells
            for name, cells in (recorded.get("dominant") or {}).items()
            if name in names and name not in indicators and name not in values
        }
        terms = _terms_of(names, levels, indicators, dominant, values)
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
                    # the largest share of rows without the event over the datasets: the counts above are
                    # each the smallest, come from different datasets, and read too low together
                    *(
                        [Number("without_event_share_max", extra["without_share"], DETERMINISTIC)]
                        if extra.get("without_share") is not None
                        else []
                    ),
                    # a count model's dispersion, the smallest and the largest over the datasets: a pooled Poisson
                    # model recorded none, and so could not say that its standard errors are too small
                    *(
                        [
                            Number("dispersion_min", extra["dispersion"][0], ITERATIVE),
                            Number("dispersion_max", extra["dispersion"][1], ITERATIVE),
                        ]
                        if kind == "count" and extra.get("dispersion") is not None
                        else []
                    ),
                ),
            ),
            *_reference_parts(levels, indicators, kind, dominant, values),
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
                # the rows of the groups drawn, the fewest over the datasets: what the analysis read
                Number("rows_min", (extra.get("total") or {}).get("rows"), EXACT),
                # the largest share of rows without the event over the datasets
                *(
                    [
                        Number(
                            "without_event_share_max",
                            (extra.get("total") or {})["without_share"],
                            DETERMINISTIC,
                        )
                    ]
                    if (extra.get("total") or {}).get("without_share") is not None
                    else []
                ),
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
        AnalysisKind(
            "mantel_haenszel",
            "Mantel-Haenszel stratified analysis",
            # declared before its runner and its record, as the accuracy of a test is: the design's values are the
            # ids of biostats.study_designs (OFFERED), held equal by tests/biostats_layer/test_plan.py
            roles=("outcome", "compare", "stratum"),
            options={"alpha": 0.05, "event_level": None, "reference": None, "design": "unknown"},
            choices={"design": ("unknown", "cohort", "rct", "cross-sectional", "case-control")},
        ),
        AnalysisKind(
            "diagnostic",
            "Diagnostic accuracy",
            # declared for its record; its runner comes with those of the other remaining kinds. The values of
            # index_form, direction, mode and sampling are the ids of biostats.diagnostic (INDEX_FORMS, DIRECTIONS,
            # CUTOFF_MODES, SAMPLINGS), held equal by tests/test_accuracy_yes_no.py: this module reads every
            # kind's own module inside its functions, never at import, so the two defaults are written here
            roles=("reference", "index"),
            options={
                "alpha": 0.05,
                "present": None,
                "index_form": "number",  # SCORE_FORM; or YES_NO_FORM
                "positive": None,  # the value of a yes/no test read as positive
                "direction": None,
                "mode": None,
                "cutoff": None,
                "prevalence": None,
                "sampling": "not_stated",  # SAMPLING_NOT_STATED
            },
            choices={
                "index_form": ("number", "yes_no"),
                "direction": (None, "higher", "lower"),
                "mode": (None, "No cut-off", "Manual", "Youden-optimal (data-driven)"),
                # never "": that is how a result computed before the question reads, not what a request asks
                "sampling": ("not_stated", "series", "two_gate"),
            },
            record=_record_diagnostic,
        ),
        AnalysisKind("bland_altman", "Bland-Altman agreement"),
        AnalysisKind("icc", "Intraclass correlation"),
        AnalysisKind("kappa", "Cohen's kappa"),
        AnalysisKind(
            "proportion",
            "Share against a reference",
            libraries=("scipy", "statsmodels"),
            record=_record_proportion,
            # typed: the calculator "A share against a reference, numbers I type"; its dataset runner comes later
            typed_title="Share against a reference (numbers typed)",
            typed_inputs={"count": COUNT, "total": COUNT},
            typed_options={"alpha": 0.05, "reference": None},
            # one level: every writer says "95%" from the record, and another level is a later version's
            choices={"alpha": (0.05,)},
            typed_run=_typed_proportion,
        ),
        AnalysisKind(
            "rates",
            "Incidence rates",
            libraries=("scipy", "statsmodels"),
            record=_record_rates,
            # typed only for now: rates read from the dataset come with the incidence analyses, as a later version
            typed_title="Incidence rates (numbers typed)",
            typed_inputs={
                "events": COUNT,
                "person_time": QUANTITY,
                "reference_events": COUNT,
                "reference_person_time": QUANTITY,
            },
            typed_optional=("reference_events", "reference_person_time"),
            typed_options={"alpha": 0.05, "unit": "person-years", "per": 1000},
            choices={
                "alpha": (0.05,),
                "unit": ("person-years", "person-months", "person-days"),
                "per": (100, 1000, 10000, 100000),
            },
            typed_run=_typed_rates,
        ),
        AnalysisKind(
            "standardised_ratio",
            "Standardised ratio (observed against expected)",
            libraries=("scipy", "statsmodels"),
            record=_record_standardised,
            typed_title="Standardised ratio (numbers typed)",
            typed_inputs={"observed": COUNT, "expected": QUANTITY},
            typed_options={"alpha": 0.05, "events_are": "deaths"},
            choices={"alpha": (0.05,), "events_are": ("deaths", "cases")},
            typed_run=_typed_standardised,
        ),
    )
}


__all__ = [
    "CLASSES",
    "COUNT",
    "DETERMINISTIC",
    "EXACT",
    "IMPUTED",
    "ITERATIVE",
    "ITERATIVE_LOOSE",
    "KINDS",
    "LOG",
    "OBSERVED",
    "PLAIN",
    "POOLED_SUFFIX",
    "QUANTITY",
    "RECIPROCAL",
    "RULES",
    "SEEDED",
    "TYPED",
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
    "record_of",
    "run",
    "runnable",
    "specify",
    "titles",
]
