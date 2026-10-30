"""
The analysis population: which rows of the dataset an analysis uses.

Inclusion and exclusion criteria are filters the user ticks or unticks; they never delete rows. The
study configuration groups criteria in families (``config["mask_families"]``): a family with the operator
``any`` (implausible values, outliers) is an *exclusion* family (a row is excluded when a ticked rule fires),
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
    RuleCheck,
    criterion_columns,
    criterion_mask,
    criterion_problems,
    holds_row_list,
    rule_checks,
    shared_definition,
    threshold_rule,
)
from core.derived import CONFIG_KEY as DERIVED_KEY
from core.derived import KINDS as DERIVED_KINDS
from core.derived import ORIGINS as DERIVED_ORIGINS
from core.derived import DerivedVariable
from core.derived import shape_problems as derived_shape_problems
from core.fingerprint import ids_fingerprint, ids_to_ranges

EXCLUSION, INCLUSION = "exclusion", "inclusion"
_INLINE_IDS = 20_000  # row-id ranges are written into the trace up to this many rows


def _digest(payload: Any, length: int = 16) -> str:
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def criterion_id(family: str, name: str) -> str:
    return f"{family}::{name}"


# What the pages, the participant flow and the dossier call a family. Its id is stored (config.json, criterion ids,
# traces, the population fingerprint) and never changes; its words may.
FAMILY_WORDS: dict[str, str] = {"clinical_anomalies": "implausible values"}


def family_words(family: str) -> str:
    """A family in words: "implausible values" for ``clinical_anomalies``, else its id with spaces."""
    return FAMILY_WORDS.get(family, family.replace("_", " "))


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
                    threshold_rule(k, rules),
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


@dataclass(frozen=True)
class FamilyChecks:
    """
    The checks of a family of plausibility rules (Domain Validity): with "any", each rule's checks; with "all" or
    "at least k", one check per row, on the rows where the rules settle the family's result (Kleene logic).
    """

    operator: str
    checks: int  # C: the checks made
    fails: int  # F: those where the rule, or the family, fires
    units: tuple[RuleCheck, ...]  # each check of each rule (one per column for bounds)
    errors: Mapping[str, str]  # rule name -> why it cannot be evaluated: listed, not counted


def family_checks(
    df: pd.DataFrame, members: Mapping[str, Any], *, imputed: Mapping[str, Any] | None = None
) -> FamilyChecks:
    """
    Domain Validity's checks of a family on ``df``: a check is one rule on one row that holds the values the rule
    reads (core.criteria.rule_checks), so that the score, 100 x (1 - fails / checks), never counts a missing value as
    a pass; an imputed value (``imputed``: row ids per column) is no recorded value either. With "all" or "at least
    k", one check per row where the rules settle the family, read as a rule "at least k of" them. A rule that cannot be
    evaluated is left out and listed.
    """
    operator, k = family_operator(members)
    observed = _observed(df, imputed)
    units: list[RuleCheck] = []
    evaluable: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name, spec in _rules(members).items():
        try:
            units += rule_checks(observed, name, spec)
        except CriterionError as exc:
            errors[name] = str(exc)
            continue
        evaluable[name] = spec
    if operator == "threshold" and (isinstance(k, bool) or not isinstance(k, int | float)):
        errors["operator"] = 'a threshold family needs a number of rules, as {"threshold": 2}'
        return FamilyChecks(operator, 0, 0, tuple(units), errors)
    if operator == "any" or not evaluable:
        checks = sum(int(c.checked.sum()) for c in units)
        fails = sum(int(c.fired.sum()) for c in units)
        return FamilyChecks(operator, checks, fails, tuple(units), errors)
    need = len(evaluable) if operator == "all" else int(k)
    family = {
        "threshold": need,
        "members": evaluable,
    }  # Kleene logic, as a rule "at least k of" reads it
    (whole,) = rule_checks(observed, "the family", family)
    return FamilyChecks(
        operator, int(whole.checked.sum()), int(whole.fired.sum()), tuple(units), errors
    )


def _observed(df: pd.DataFrame, imputed: Mapping[str, Any] | None) -> pd.DataFrame:
    """``df`` with its imputed cells missing: a check reads recorded values only."""
    if not imputed:
        return df
    out = df.copy()
    for column, ids in imputed.items():
        if column in out.columns:
            out[column] = out[column].mask(out.index.isin(np.asarray(ids)))
    return out


# What a configuration shared with someone else holds. The other keys are this installation's settings: its folders
# (folder_names, under DATA_ROOT), the paths of the files uploaded for an enrichment (data_enrichments), and defaults
# that nothing reads.
SHARED_KEYS = ("description", "mask_families", DERIVED_KEY)


@dataclass(frozen=True)
class SharedConfiguration:
    """A configuration for someone else, and what it left out of the session's (names as written)."""

    config: dict[str, Any]
    # "family: rule": a rule that lists row ids of this dataset (outliers frozen as rows), at any depth
    with_rows: tuple[str, ...] = ()
    # a family counted "at least k of" its rules, left out whole because one of its rules was
    families: tuple[str, ...] = ()
    # "family: rule", a family or "derived_variables: target" that cannot be read, or holds a value of another type
    unreadable: tuple[str, ...] = ()
    # "family: rule" or "family: the operator", rebuilt without keys that this version does not read
    trimmed: tuple[str, ...] = ()
    # the derived variables whose note was left out
    notes: tuple[str, ...] = ()
    # the other keys of the configuration
    settings: tuple[str, ...] = ()


def _shared_operator(operator: Any) -> Any:
    """A family's operator as a shared copy holds it: "any" or "all", or {"threshold": k}; None for anything else."""
    if isinstance(operator, str) and operator.strip().lower() in ("any", "all"):
        return operator
    if isinstance(operator, Mapping):
        k = operator.get("threshold")
        if isinstance(k, int) and not isinstance(k, bool):
            return {"threshold": k}
    return None


def _shared_family(
    family: str, members: Mapping[str, Any], left: dict[str, list[str]]
) -> dict[str, Any] | None:
    """A family as a shared copy holds it, None when it is left out whole; what it leaves out goes into ``left``."""
    operator = _shared_operator(members.get("operator", "any"))
    if operator is None:
        left["unreadable"].append(family)
        return None
    rules = {name: spec for name, spec in members.items() if name != "operator"}
    shared = {name: shared_definition(str(name), spec) for name, spec in rules.items()}
    rows = [name for name, spec in rules.items() if holds_row_list(spec)]
    unread = [name for name, rule in shared.items() if rule is None and name not in rows]
    if family_operator(members)[0] == "threshold" and (rows or unread):
        # "at least 2 of" these rules would count others without the ones left out: the family goes whole
        left["families"].append(family)
        return None
    left["with_rows"] += [f"{family}: {name}" for name in rows]
    left["unreadable"] += [f"{family}: {name}" for name in unread]
    kept = {name: rule for name, rule in shared.items() if rule is not None}
    left["trimmed"] += [f"{family}: {name}" for name, rule in kept.items() if rule != rules[name]]
    if rules and not kept:
        return None  # each of its rules is named
    if "operator" in members:
        if operator != members["operator"]:
            left["trimmed"].append(f"{family}: the operator")
        kept = {"operator": operator, **kept}
    return kept


def _shared_derived(target: str, definition: Any) -> dict[str, Any] | None:
    """
    A derived variable as a shared copy holds it: read as the application reads it (core.derived), without its note,
    the user's commentary, which no shared output holds. None when it cannot be read: a file can hold anything there.
    """
    if not isinstance(definition, Mapping) or derived_shape_problems({target: definition}):
        return None
    try:
        shared = DerivedVariable.from_config(target, definition).to_config()
    except Exception:
        return None
    shared.pop("note", None)
    return shared


def _derived_trimmed(definition: Mapping[str, Any]) -> bool:
    """Whether a derived variable is shared without something it holds besides its note: a key core.derived does not
    read, or a kind or an origin it does not know (read as none)."""
    known = ("formula", "inverses", "kind", "origin", "note", "tolerance", "accepted")
    return (
        bool(set(definition) - set(known))
        or definition.get("kind") not in (None, *DERIVED_KINDS)
        or definition.get("origin") not in (None, *DERIVED_ORIGINS)
    )


def configuration_for_sharing(config: Mapping[str, Any] | None) -> SharedConfiguration:
    """
    A copy of the configuration to share with someone else, and what it leaves out.

    It holds the description, the criteria and the derived variables (SHARED_KEYS): each rule rebuilt from the keys
    its kind holds, with values of the types it reads (core.criteria.shared_definition), each family's operator as
    "any", "all" or {"threshold": k}, each derived variable as core.derived reads it, without its note. Left out, and
    named: a rule that lists row ids of this dataset (outliers frozen as rows, by any method), also inside a threshold
    rule, or that cannot be read; a family counted with a threshold, whole, when one of its rules is left out ("at
    least 2 of" would count other rules); a family left without a rule; the other keys, which hold this installation's
    folders and the paths of uploaded files. A rule rebuilt without keys that no kind reads is named too. The copy
    never relies on the upload check, and shares nothing with the session's configuration.
    """
    source = json.loads(json.dumps(dict(config or {}), default=str))
    shared: dict[str, Any] = {}
    left: dict[str, list[str]] = {
        "with_rows": [],
        "families": [],
        "unreadable": [],
        "trimmed": [],
        "notes": [],
    }
    if isinstance(source.get("description"), str):
        shared["description"] = source["description"]
    families = source.get("mask_families")
    if isinstance(families, dict):
        kept_families: dict[str, Any] = {}
        for family, members in families.items():
            kept = _shared_family(str(family), members, left) if isinstance(members, dict) else None
            if kept is not None:
                kept_families[family] = kept
            elif not isinstance(members, dict):
                left["unreadable"].append(str(family))
        shared["mask_families"] = kept_families
    derived = source.get(DERIVED_KEY)
    if isinstance(derived, dict):
        kept_derived: dict[str, Any] = {}
        for target, definition in derived.items():
            rebuilt = _shared_derived(str(target), definition)
            if rebuilt is None:
                left["unreadable"].append(f"{DERIVED_KEY}: {target}")
                continue
            kept_derived[target] = rebuilt
            if definition.get("note"):
                left["notes"].append(str(target))
            if _derived_trimmed(definition):
                left["trimmed"].append(f"{DERIVED_KEY}: {target}")
        shared[DERIVED_KEY] = kept_derived
    settings = tuple(
        str(key)
        for key, value in source.items()
        if key not in SHARED_KEYS and value not in (None, "", [], {})
    )
    return SharedConfiguration(
        shared,
        with_rows=tuple(left["with_rows"]),
        families=tuple(left["families"]),
        unreadable=tuple(left["unreadable"]),
        trimmed=tuple(left["trimmed"]),
        notes=tuple(left["notes"]),
        settings=settings,
    )


def merge_families(
    session: Mapping[str, Any] | None, uploaded: Mapping[str, Any] | None
) -> tuple[dict[str, Any], list[str]]:
    """
    An uploaded configuration's criteria added to the session's, never merged into them key by key: a family the
    session lacks, or holds without a rule, is added whole; a rule it holds keeps the session's definition whole, and
    is named when the file's differs; a family that holds rules keeps how it reads, and its operator is named when the
    file's reads otherwise (one without an operator reads "any": given the file's, its ticked criteria would change or
    vanish). A numeric rule and an uploaded expression of the same name made one rule holding both, which was refused
    while the page reported success. The result shares nothing with the file. Returns the merged families and the
    names that differ ("family: rule", "family: the operator").
    """
    merged: dict[str, Any] = copy.deepcopy(dict(session or {}))
    differing: list[str] = []
    for family, members in (uploaded or {}).items():
        if not isinstance(members, Mapping):
            continue
        held = merged.get(family)
        if not isinstance(held, dict) or not _rules(held):
            merged[family] = copy.deepcopy(dict(members))
            continue
        if family_operator(held) != family_operator(members):
            differing.append(f"{family}: the operator")
        for name, spec in members.items():
            if name == "operator":
                continue
            if name not in held:
                held[name] = copy.deepcopy(spec)
            elif held[name] != spec:
                differing.append(f"{family}: {name}")
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
    "FAMILY_WORDS",
    "INCLUSION",
    "SHARED_KEYS",
    "Criterion",
    "CriterionCount",
    "FamilyMasks",
    "PopulationResult",
    "SharedConfiguration",
    "Stage",
    "active_criteria",
    "catalogue",
    "configuration_for_sharing",
    "configuration_problems",
    "criteria_from_trace",
    "criterion_id",
    "evaluate_population",
    "family_kind",
    "family_masks",
    "family_operator",
    "family_words",
    "merge_families",
]
