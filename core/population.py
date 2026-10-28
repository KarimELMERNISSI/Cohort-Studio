"""
The analysis population: which rows of the dataset an analysis uses.

Inclusion and exclusion criteria are filters the user ticks or unticks; they never delete rows. The
study configuration groups criteria in families (``config["mask_families"]``): a family with the operator
``any`` (clinical anomalies, outliers) is an *exclusion* family (a row is excluded when a ticked rule fires),
a family with the operator ``all`` (study inclusion) an *inclusion* family (a row is excluded when it fails
a ticked criterion), and a family with a threshold operator one exclusion rule "at least k of these".

:func:`evaluate_population` applies the ticked criteria family by family, in the order of the
configuration, and counts what each excludes: the stages and counts of a STROBE participant-flow diagram
(item 13). The result is identified by a fingerprint of the data and of the rows kept, so results computed
on another population are recognised as stale.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd

from core.criteria import (
    CriterionError,
    criterion_columns,
    criterion_mask,
    criterion_problems,
)
from core.derived import CONFIG_KEY as DERIVED_KEY
from core.derived import shape_problems as derived_shape_problems
from core.fingerprint import ids_fingerprint, ids_to_ranges

EXCLUSION, INCLUSION = "exclusion", "inclusion"
_INLINE_IDS = 20_000  # row-id ranges are written into the trace up to this many rows


def _digest(payload: Any, length: int = 16) -> str:
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def criterion_id(family: str, name: str) -> str:
    return f"{family}::{name}"


@dataclass(frozen=True)
class Criterion:
    """One criterion of the study configuration, with a snapshot of its definition."""

    id: str
    family: str
    kind: str  # EXCLUSION | INCLUSION
    name: str
    definition: Mapping[str, Any]
    description: str = ""

    @cached_property
    def fingerprint(self) -> str:
        """Identity of what the criterion selects (the description is not part of it)."""
        spec = {k: v for k, v in self.definition.items() if k != "description"}
        return _digest([self.family, self.kind, self.name, spec])

    @property
    def columns(self) -> tuple[str, ...]:
        """Columns the criterion reads (empty when they cannot be determined)."""
        return criterion_columns(self.name, self.definition)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "kind": self.kind,
            "name": self.name,
            "definition": dict(self.definition),
        }


def family_operator(members: Mapping[str, Any]) -> tuple[str, Any]:
    """
    How a family combines its rules: ("any", None), ("all", None) or ("threshold", k). Text is read without regard
    to case or spaces ("All" is "all"); any other operator reads as "any", the default, and
    :func:`configuration_problems` reports it.
    """
    operator = members.get("operator", "any")
    if isinstance(operator, Mapping):
        return "threshold", operator.get("threshold")
    text = str(operator).strip().lower()
    if text == "threshold":
        return "threshold", None
    return ("all" if text == "all" else "any"), None


def _rules(members: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(name): spec
        for name, spec in members.items()
        if name != "operator" and isinstance(spec, Mapping)
    }


def catalogue(config: Mapping[str, Any] | None) -> tuple[Criterion, ...]:
    """Every criterion of the configuration, family by family, in configuration order."""
    out: list[Criterion] = []
    for family, members in ((config or {}).get("mask_families") or {}).items():
        if not isinstance(members, Mapping):
            continue
        operator, k = family_operator(members)
        rules = _rules(members)
        if operator == "threshold":
            name = f"At least {k} rules of {family}"
            out.append(
                Criterion(
                    criterion_id(family, name),
                    family,
                    EXCLUSION,
                    name,
                    {"threshold": k, "members": rules},
                    f"Rows breaking at least {k} of the {len(rules)} rules of the family.",
                )
            )
            continue
        kind = INCLUSION if operator == "all" else EXCLUSION
        for name, spec in rules.items():
            out.append(
                Criterion(
                    criterion_id(family, name),
                    family,
                    kind,
                    name,
                    dict(spec),
                    str(spec.get("description") or ""),
                )
            )
    return tuple(out)


def family_kind(config: Mapping[str, Any] | None, family: str) -> str | None:
    kinds = {c.kind for c in catalogue(config) if c.family == family}
    return kinds.pop() if len(kinds) == 1 else None


@dataclass(frozen=True)
class FamilyMasks:
    """The rules of one family evaluated on a table, as the analysis population evaluates them."""

    operator: str  # any | all | threshold
    threshold: Any
    rules: Mapping[str, pd.Series]  # rule name -> rows where it fires
    errors: Mapping[str, str]  # rule name -> why it cannot be evaluated
    # any: a rule fires; all: every rule fires; threshold: at least k fire (None while a rule has an error)
    flag: pd.Series | None


def family_masks(df: pd.DataFrame, members: Mapping[str, Any]) -> FamilyMasks:
    """
    Every rule of a family on ``df`` with :func:`core.criteria.criterion_mask`, and the family result read with
    its operator. The Validate checks and the quality dashboard use it, so they count what the population counts.
    A rule that cannot be evaluated is listed in ``errors`` and leaves the family result unset: the population
    refuses such a rule while it is ticked.
    """
    operator, k = family_operator(members)
    rules: dict[str, pd.Series] = {}
    errors: dict[str, str] = {}
    for name, spec in _rules(members).items():
        try:
            rules[name] = criterion_mask(df, name, spec)
        except CriterionError as exc:
            errors[name] = str(exc)
    if operator == "threshold" and (isinstance(k, bool) or not isinstance(k, int | float)):
        errors["operator"] = 'a threshold family needs a number of rules, as {"threshold": 2}'
    flag = None
    if rules and not errors:
        fired = pd.DataFrame(rules, index=df.index)
        if operator == "any":
            flag = fired.any(axis=1)
        elif operator == "all":
            flag = fired.all(axis=1)
        else:
            flag = fired.sum(axis=1) >= int(k)
    return FamilyMasks(operator, k, rules, errors, flag)


def without_row_lists(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """
    A copy of the configuration for sharing: without the rules that list rows (outliers found by a multivariate
    method, frozen as row ids), which only mean something for this dataset; a family left empty is dropped.
    """
    shared = json.loads(json.dumps(dict(config or {}), default=str))
    families = shared.get("mask_families")
    if isinstance(families, dict):
        for family in list(families):
            members = families[family]
            if not isinstance(members, dict):
                continue
            kept = {
                name: spec
                for name, spec in members.items()
                if name == "operator" or not (isinstance(spec, dict) and "row_ids" in spec)
            }
            if any(name != "operator" for name in members) and all(
                name == "operator" for name in kept
            ):
                del families[family]
            else:
                families[family] = kept
    return shared


def merge_families(
    session: Mapping[str, Any] | None, uploaded: Mapping[str, Any] | None
) -> tuple[dict[str, Any], list[str]]:
    """
    An uploaded configuration's criteria added to the session's, never merged into them key by key: a family or a
    rule the session lacks is added whole; a rule it holds keeps the session's definition whole, and is named when the
    file's differs; so is the family's operator. A numeric rule and an uploaded expression of the same name made one
    rule holding both, which was refused while the page reported success. The result shares nothing with the file.
    Returns the merged families and the names that differ ("family: rule", "family: the operator").
    """
    merged: dict[str, Any] = copy.deepcopy(dict(session or {}))
    differing: list[str] = []
    for family, members in (uploaded or {}).items():
        if not isinstance(members, Mapping):
            continue
        held = merged.get(family)
        if not isinstance(held, Mapping):
            merged[family] = copy.deepcopy(dict(members))
            continue
        for name, spec in members.items():
            if name not in held:
                held[name] = copy.deepcopy(spec)
            elif held[name] != spec:
                differing.append(f"{family}: {'the operator' if name == 'operator' else name}")
    return merged, differing


def configuration_problems(config: Mapping[str, Any] | None) -> list[str]:
    """
    What makes a study configuration ambiguous or unusable, in words (empty when it is sound): an unknown family
    operator, a threshold without a number, a family entry that is not a rule, a rule of an unknown shape or with
    both a numeric range and an expression. An uploaded configuration with problems is refused.
    """
    # the derived variables are kept with the criteria (core.derived)
    problems: list[str] = derived_shape_problems((config or {}).get(DERIVED_KEY))
    families = (config or {}).get("mask_families")
    if families is None:
        return problems
    if not isinstance(families, Mapping):
        return ["mask_families must map each family name to its rules"]
    for family, members in families.items():
        if not isinstance(members, Mapping):
            problems.append(f"family '{family}' must map each rule name to its definition")
            continue
        operator = members.get("operator", "any")
        if isinstance(operator, Mapping):
            k = operator.get("threshold")
            if isinstance(k, bool) or not isinstance(k, int) or k < 1:
                problems.append(
                    f"family '{family}': the threshold must be a whole number of rules, as {{\"threshold\": 2}}"
                )
        elif str(operator).strip().lower() not in ("any", "all"):
            problems.append(
                f'family \'{family}\': unknown operator {operator!r}; use "any", "all" or {{"threshold": k}}'
            )
        for name, spec in members.items():
            if name == "operator":
                continue
            if not isinstance(spec, Mapping):
                problems.append(
                    f"family '{family}': '{name}' is not a rule (only rules and \"operator\" belong in a family)"
                )
                continue
            # each shape's own checks: what fails on any table (a bound in words, a range of three numbers, a
            # threshold rule without its rules) passed this check, then stopped every page once loaded
            problems += [
                f"family '{family}': {problem}" for problem in criterion_problems(str(name), spec)
            ]
    return problems


@dataclass(frozen=True)
class Stage:
    """One family applied to the rows left by the previous families."""

    family: str
    kind: str
    n_in: int
    n_excluded: int
    n_out: int


@dataclass(frozen=True)
class CriterionCount:
    criterion: Criterion
    marginal_base: int  # rows of the whole table the criterion excludes
    marginal_in_stage: int  # rows entering its family's stage that it excludes
    sequential: int  # rows it excludes that no earlier criterion of the stage excluded


@dataclass(frozen=True)
class PopulationResult:
    data_fp: str
    n_base: int
    criteria: tuple[Criterion, ...]
    included: pd.Index
    stages: tuple[Stage, ...] = ()
    counts: tuple[CriterionCount, ...] = ()
    excluded_by: Mapping[str, pd.Index] = field(default_factory=dict)
    errors: Mapping[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def n_included(self) -> int:
        return len(self.included)

    @cached_property
    def spec_fingerprint(self) -> str:
        return _digest([c.fingerprint for c in self.criteria])

    @cached_property
    def rows_fingerprint(self) -> str:
        return ids_fingerprint(self.included)

    @cached_property
    def fingerprint(self) -> str:
        """Identity of the analysed rows: the data and the row ids kept."""
        return _digest([self.data_fp, self.rows_fingerprint])

    @property
    def label(self) -> str:
        if not self.criteria:
            return f"all {self.n_base:,} rows"
        n = len(self.criteria)
        return f"{self.n_included:,} of {self.n_base:,} rows ({n} criteri{'on' if n == 1 else 'a'})"

    def reasons(self, row_id: Any) -> tuple[str, ...]:
        """Names of the ticked criteria that exclude ``row_id``."""
        return tuple(
            c.name
            for c in self.criteria
            if c.id in self.excluded_by and row_id in self.excluded_by[c.id]
        )

    def to_trace(self, *, inline_limit: int = _INLINE_IDS) -> dict[str, Any]:
        """JSON-able record of the population for the transformation trace."""
        block: dict[str, Any] = {
            "fingerprint": self.fingerprint,
            "data_fp": self.data_fp,
            "rows_fp": self.rows_fingerprint,
            "spec_fp": self.spec_fingerprint,
            "n_base": self.n_base,
            "n_included": self.n_included,
            "criteria": [c.snapshot() for c in self.criteria],
            "stages": [vars(s) for s in self.stages],
            "counts": [
                {
                    "id": c.criterion.id,
                    "marginal_base": c.marginal_base,
                    "marginal_in_stage": c.marginal_in_stage,
                    "sequential": c.sequential,
                }
                for c in self.counts
            ],
        }
        if self.n_base - self.n_included <= inline_limit:
            block["excluded_by"] = {
                cid: ids_to_ranges(ids) for cid, ids in self.excluded_by.items() if len(ids)
            }
        if self.n_included <= inline_limit:
            block["included"] = ids_to_ranges(self.included)
        if self.errors:
            block["errors"] = dict(self.errors)
        return block


def evaluate_population(
    df: pd.DataFrame, criteria: Sequence[Criterion], *, data_fp: str = ""
) -> PopulationResult:
    """Apply ``criteria`` (ticked criteria, in catalogue order) family by family; never raises on a bad
    criterion: its error is recorded and the result is not ``ok`` (callers must refuse to analyse it)."""
    n = len(df)
    excluded: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}
    for c in criteria:
        try:
            fires = criterion_mask(df, c.name, c.definition).to_numpy(dtype=bool)
        except CriterionError as exc:
            errors[c.id] = str(exc)
            continue
        excluded[c.id] = fires if c.kind == EXCLUSION else ~fires

    families: list[str] = []
    for c in criteria:
        if c.family not in families:
            families.append(c.family)
    remaining = np.ones(n, dtype=bool)
    stages: list[Stage] = []
    counts: list[CriterionCount] = []
    for family in families:
        members = [c for c in criteria if c.family == family]
        n_in = int(remaining.sum())
        stage_excluded = np.zeros(n, dtype=bool)
        for c in members:
            mask = excluded.get(c.id)
            if mask is None:
                continue
            in_stage = mask & remaining
            counts.append(
                CriterionCount(
                    c,
                    marginal_base=int(mask.sum()),
                    marginal_in_stage=int(in_stage.sum()),
                    sequential=int((in_stage & ~stage_excluded).sum()),
                )
            )
            stage_excluded |= in_stage
        remaining &= ~stage_excluded
        n_out = int(remaining.sum())
        stages.append(Stage(family, members[0].kind, n_in, n_in - n_out, n_out))

    return PopulationResult(
        data_fp=data_fp,
        n_base=n,
        criteria=tuple(criteria),
        included=df.index[remaining],
        stages=tuple(stages),
        counts=tuple(counts),
        excluded_by={cid: df.index[mask] for cid, mask in excluded.items()},
        errors=errors,
    )


def criteria_from_trace(block: Mapping[str, Any] | None) -> tuple[Criterion, ...]:
    """The criteria recorded in a trace's population block, with their definitions as they were used."""
    return tuple(
        Criterion(
            id=str(c["id"]),
            family=str(c["family"]),
            kind=str(c["kind"]),
            name=str(c["name"]),
            definition=dict(c.get("definition") or {}),
        )
        for c in ((block or {}).get("criteria") or [])
    )


def active_criteria(
    config: Mapping[str, Any] | None, ticked: Iterable[str]
) -> tuple[Criterion, ...]:
    """The ticked criteria that still exist in the configuration, in catalogue order."""
    chosen = set(ticked)
    return tuple(c for c in catalogue(config) if c.id in chosen)


__all__ = [
    "EXCLUSION",
    "INCLUSION",
    "Criterion",
    "CriterionCount",
    "FamilyMasks",
    "PopulationResult",
    "Stage",
    "active_criteria",
    "catalogue",
    "configuration_problems",
    "criteria_from_trace",
    "criterion_id",
    "evaluate_population",
    "family_kind",
    "family_masks",
    "family_operator",
    "merge_families",
    "without_row_lists",
]
