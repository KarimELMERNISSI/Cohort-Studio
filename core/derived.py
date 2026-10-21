"""
Derived variables declared for a study: a variable computed from other variables of the same row (a BMI from the
weight and the height, a total from its items, a unit conversion). The list is study data, kept with the criteria in
the configuration (``config["derived_variables"]``, by variable) and exported with it. A definition says how to compute
its variable, and its inverses how to recover one of the formula's variables from the others (a weight from the BMI and
the height).

Pure: the formulas are read and evaluated by ``enrich`` (enrich.derived checks a definition against the data). A
definition is never applied by being declared: it is checked, then used when the user fills or imputes with it.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from core.expressions import DEFAULT_POLICY
from core.lineage import canonical_json

CONFIG_KEY = "derived_variables"

# what a definition is, shown and asked of the AI engine ("" when not said); an unknown one is read as ""
KINDS = ("conversion", "ratio", "total", "index", "equation", "class", "")
ORIGINS = ("user", "ai", "import")  # an unknown one is read as "user"


def _digest(body: Mapping[str, str]) -> str:
    return hashlib.sha256(canonical_json(dict(body)).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class DerivedVariable:
    """One declared definition, named by its variable: one definition per variable."""

    target: str
    formula: str
    # one of the formula's variables -> the formula that recovers it from the target and the others
    inverses: Mapping[str, str] = field(default_factory=dict)
    kind: str = ""
    origin: str = "user"
    note: str = ""  # its source, the units it assumes
    tolerance: float | None = None  # None: from the precision the values are recorded with
    # used although the data do not fully agree, or cannot check it (fewer rows than the check needs)
    accepted: bool = False

    @property
    def semantic_hash(self) -> str:
        """What the definition computes: its variable and its formula. Not its inverses (each has its own), its
        origin, its note or its acceptance: a result derived from it changes when this does."""
        return _digest({"target": self.target, "formula": self.formula.strip()})

    def inverse_hash(self, name: str) -> str:
        """What the inverse recovering ``name`` computes (a step that uses it records this)."""
        return _digest({"target": name, "formula": self.inverses[name].strip(), "of": self.target})

    def to_config(self) -> dict[str, Any]:
        """The definition as kept in the configuration (by its variable, which is not repeated)."""
        out: dict[str, Any] = {"formula": self.formula}
        if self.inverses:
            out["inverses"] = dict(self.inverses)
        if self.kind:
            out["kind"] = self.kind
        if self.origin != "user":
            out["origin"] = self.origin
        if self.note:
            out["note"] = self.note
        if self.tolerance is not None:
            out["tolerance"] = self.tolerance
        if self.accepted:
            out["accepted"] = True
        return out

    @classmethod
    def from_config(cls, target: str, data: Mapping[str, Any]) -> DerivedVariable:
        kind, origin, tolerance = (
            data.get("kind") or "",
            data.get("origin") or "user",
            data.get("tolerance"),
        )
        return cls(
            target=str(target),
            formula=str(data["formula"]),
            inverses={str(k): str(v) for k, v in dict(data.get("inverses") or {}).items()},
            kind=kind if kind in KINDS else "",
            origin=origin if origin in ORIGINS else "user",
            note=str(data.get("note") or ""),
            tolerance=None if tolerance is None else float(tolerance),
            accepted=data.get("accepted") is True,
        )


def shape_problems(entries: Any) -> list[str]:
    """What makes a configuration's derived variables unreadable, in words (empty when sound): an imported file is
    refused with them, as with a criterion of an unknown shape. A kind or an origin the application does not know is
    read as the default: they are only shown."""
    if entries is None:
        return []
    if not isinstance(entries, Mapping):
        return [f"{CONFIG_KEY} must map each variable to its definition"]
    longest = DEFAULT_POLICY.max_length
    problems: list[str] = []
    for target, entry in entries.items():
        where = f"{CONFIG_KEY}: {target}"
        if not isinstance(target, str) or not target.strip():
            problems.append(f"{CONFIG_KEY}: a definition has no variable")
            continue
        if not isinstance(entry, Mapping):
            problems.append(f"{where} is not a definition")
            continue
        formula = entry.get("formula")
        if not isinstance(formula, str) or not formula.strip() or len(formula) > longest:
            problems.append(f"{where} has no formula, or one longer than {longest:,} characters")
        inverses = entry.get("inverses") or {}
        if not isinstance(inverses, Mapping) or not all(
            isinstance(k, str) and isinstance(v, str) and v.strip() and len(v) <= longest
            for k, v in inverses.items()
        ):
            problems.append(f"{where}: inverses must map a variable to its formula")
        tolerance = entry.get("tolerance")
        if tolerance is not None and (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, int | float)
            or not math.isfinite(tolerance)
            or tolerance < 0
        ):
            problems.append(f"{where}: the tolerance must be a number, zero or more")
        if entry.get("accepted") not in (None, True, False):
            problems.append(f"{where}: accepted must be true or false")
    return problems


def from_config(config: Mapping[str, Any] | None) -> tuple[DerivedVariable, ...]:
    """The declared definitions, in the order they were declared (none when the configuration has none, or an
    unreadable list)."""
    entries = (config or {}).get(CONFIG_KEY)
    if not entries or shape_problems(entries):
        return ()
    return tuple(DerivedVariable.from_config(target, entry) for target, entry in entries.items())


def with_definitions(
    config: Mapping[str, Any] | None, definitions: Iterable[DerivedVariable]
) -> dict[str, Any]:
    """``config`` with these definitions (a new dict: the configuration is replaced, never changed in place)."""
    out = dict(config or {})
    entries = {d.target: d.to_config() for d in definitions}
    if entries:
        out[CONFIG_KEY] = entries
    else:
        out.pop(CONFIG_KEY, None)
    return out


def merge_definitions(
    current: Mapping[str, Any] | None, uploaded: Mapping[str, Any] | None
) -> tuple[dict[str, Any], list[str]]:
    """
    The definitions of an uploaded configuration added to the session's, whole: a variable defined in both keeps the
    session's definition, never a mix of the two, and is named among the conflicts when the file's differs (an
    uploaded configuration adds to the session's, as the criteria do).
    """
    merged = {str(k): dict(v) for k, v in dict(current or {}).items()}
    conflicts: list[str] = []
    for target, entry in dict(uploaded or {}).items():
        if target not in merged:
            merged[str(target)] = dict(entry)
        elif canonical_json(merged[target]) != canonical_json(dict(entry)):
            conflicts.append(str(target))
    return merged, conflicts


def _graph(
    definitions: Iterable[DerivedVariable], inputs: Mapping[str, Iterable[str]]
) -> dict[str, list[str]]:
    targets = {d.target for d in definitions}
    return {
        d.target: [n for n in inputs.get(d.target, ()) if n in targets and n != d.target]
        for d in definitions
    }


def cycles(
    definitions: Iterable[DerivedVariable], inputs: Mapping[str, Iterable[str]]
) -> list[list[str]]:
    """The definitions that compute each other (``inputs``: each variable's formula's variables): no order of filling
    can start with them."""
    graph = _graph(definitions, inputs)
    found: list[list[str]] = []
    state: dict[str, int] = {}  # 1 visiting, 2 done

    def visit(node: str, path: list[str]) -> None:
        state[node] = 1
        for nxt in graph[node]:
            if state.get(nxt) == 1:
                found.append([*path[path.index(nxt) :], nxt])
            elif state.get(nxt) is None:
                visit(nxt, [*path, nxt])
        state[node] = 2

    for target in graph:
        if state.get(target) is None:
            visit(target, [target])
    return found


def fill_order(
    definitions: Iterable[DerivedVariable], inputs: Mapping[str, Iterable[str]]
) -> list[str] | None:
    """The variables in an order where each comes after the declared variables its formula uses (a BMI before a BMI
    class), the declaration order otherwise; None when definitions compute each other. Filling, single and multiple
    imputation follow this one order."""
    definitions = list(definitions)
    graph = _graph(definitions, inputs)
    if cycles(definitions, inputs):
        return None
    order: list[str] = []

    def place(node: str) -> None:
        if node in order:
            return
        for nxt in graph[node]:
            place(nxt)
        order.append(node)

    for d in definitions:
        place(d.target)
    return order


__all__ = [
    "CONFIG_KEY",
    "KINDS",
    "ORIGINS",
    "DerivedVariable",
    "cycles",
    "fill_order",
    "from_config",
    "merge_definitions",
    "shape_problems",
    "with_definitions",
]
