"""
Checking a study package (M8): the outcomes of a check, the catalogue of checks, how a number computed again is
compared with the recorded one, and the report.

A check reads one of six outcomes:

- ``identical``: the same values (for a number, the same 64-bit value after the JSON round trip);
- ``within_tolerance``: equal up to the tolerance stated for its class, with the largest difference and where;
- ``within_monte_carlo_error``: a pooled result whose imputations were drawn again (another machine draws other
  values): its pooled estimates agree as two draws of the same imputation model do. It says less than "within
  tolerance": a changed imputation model can stay inside it;
- ``not_comparable``: nothing can be concluded, with the reason (another environment, an optional library absent,
  what it rests on did not read identical);
- ``not_run``: the package or the installation lacks what the check reads (no rows, no specification for this kind
  of analysis);
- ``failed``.

Tolerances belong to the comparison classes of ``biostats.plan`` and are stated here once (``TOLERANCES``): a relative
part, a part of the quantity's own scale (its standard error for an estimate), the log scale for a p-value, the
reciprocal for the degrees of freedom of a pooled term. A tolerance cannot show that the data changed by one row: the
counts, the files' checksums and the fingerprints of the rows do.

The ids of the checks and of the outcomes are stored in reports: never renamed. A reason gives counts, step numbers
and variable names, never a value of the data (a term of a model is named by its place, since its name can hold a
level).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from biostats.plan import (
    DETERMINISTIC,
    EXACT,
    ITERATIVE,
    ITERATIVE_LOOSE,
    KINDS,
    LOG,
    RECIPROCAL,
    Number,
    ResultRecord,
)
from core.profiles import NUMBER, PROFILE_VERSION
from core.registry import Catalogue

# ----------------------------------------------------------------------------- outcomes (stored ids)

IDENTICAL = "identical"
WITHIN_TOLERANCE = "within_tolerance"
WITHIN_MONTE_CARLO = "within_monte_carlo_error"
NOT_COMPARABLE = "not_comparable"
NOT_RUN = "not_run"
FAILED = "failed"
OUTCOMES = (IDENTICAL, WITHIN_TOLERANCE, WITHIN_MONTE_CARLO, NOT_COMPARABLE, NOT_RUN, FAILED)
_WORST_FIRST = (FAILED, NOT_COMPARABLE, WITHIN_MONTE_CARLO, WITHIN_TOLERANCE, IDENTICAL)

# what a command line returns
EXIT_OK = 0  # no check failed
EXIT_FAILED = 1  # a check failed
EXIT_UNVERIFIABLE = (
    2  # the package could not be checked (unreadable, unsafe, a later schema, a wrong password)
)

REPORT_FORMAT = "cohortstudio.verification-report"
REPORT_SCHEMA = 1


def worst(outcomes: Iterable[str]) -> str:
    """
    The outcome of a check from its items': the worst of them, in the order failed, not comparable, within Monte
    Carlo error, within tolerance, identical. An item not run does not count beside one that ran; ``not_run`` when
    none ran.
    """
    seen = set(outcomes)
    unknown = seen - set(OUTCOMES)
    if unknown:
        raise ValueError(f"unknown outcome {sorted(unknown)[0]!r}; known: {', '.join(OUTCOMES)}")
    for outcome in _WORST_FIRST:
        if outcome in seen:
            return outcome
    return NOT_RUN


# ----------------------------------------------------------------------------- the checks

SHAREABLE, REPRODUCTION = "shareable", "reproduction"  # the tiers of a package (stored ids)
TIERS = (SHAREABLE, REPRODUCTION)


@dataclass(frozen=True)
class CheckKind:
    """
    A check of a package. ``tiers`` are the tiers whose packages it runs on (elsewhere it reads not_run);
    ``shareable_part`` says what it checks of a shareable package when that is less than of a reproduction one.
    """

    id: str
    title: str
    tiers: tuple[str, ...] = TIERS
    shareable_part: str = ""


CHECKS: Catalogue[CheckKind] = Catalogue("verification check")

for _kind in (
    CheckKind("package.files", "Each file of the package against the manifest's checksums"),
    CheckKind(
        "environment",
        "The code, the libraries and the machine, against those that computed the study",
    ),
    # the checksum covers the notes and the rows a rule lists, which a shareable package does not carry
    CheckKind(
        "protocol.checksum", "The study definition against its checksum", tiers=(REPRODUCTION,)
    ),
    CheckKind("protocol.fingerprint", "The study definition against its fingerprint"),
    CheckKind(
        "results.provenance",
        "Each result names this package's trace, data and population",
        shareable_part="each result computed on the package's data and rules, with the population's rows; no "
        "fingerprint of rows and no head of the trace is read",
    ),
    CheckKind(
        "flow.counts",
        "The participant flow's counts",
        shareable_part="their arithmetic, on the counts that are shown",
    ),
    CheckKind(
        "derived.definitions",
        "The definitions of the derived variables against those the steps used",
        shareable_part="the study definition's side",
    ),
    CheckKind(
        "data.sha256",
        "Each data file against the checksum its step recorded",
        shareable_part="when the data files are given",
    ),
    CheckKind("trace.chain", "The chain of the trace's steps", tiers=(REPRODUCTION,)),
    CheckKind("steps.shape", "The rows and columns of each step", tiers=(REPRODUCTION,)),
    CheckKind("population.rows", "The rows of each analysis population", tiers=(REPRODUCTION,)),
    CheckKind(
        "columns.profiles", "The columns after each step that changes data", tiers=(REPRODUCTION,)
    ),
    CheckKind(
        "results.values", "Each result computed again from its specification", tiers=(REPRODUCTION,)
    ),
    CheckKind(
        "frames.fingerprint",
        "The fingerprints of the tables and of the imputed cells",
        tiers=(REPRODUCTION,),
    ),
):
    CHECKS.register(_kind)


@dataclass(frozen=True)
class CheckItem:
    """One thing a check looked at (a file, a step by its number, a result), with the largest difference found:
    ``{"quantity", "difference", "allowed", "class"}``."""

    name: str
    outcome: str
    reason: str = ""
    largest: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        worst([self.outcome])  # a known outcome

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "outcome": self.outcome, "reason": self.reason}
        if self.largest is not None:
            out["largest"] = dict(self.largest)
        return out


@dataclass(frozen=True)
class CheckResult:
    id: str
    outcome: str
    reason: str = ""
    items: tuple[CheckItem, ...] = ()

    def __post_init__(self) -> None:
        CHECKS.get(self.id)
        worst([self.outcome])

    @classmethod
    def of(cls, check_id: str, items: Sequence[CheckItem], reason: str = "") -> CheckResult:
        """The check's result from its items: their worst outcome. Items that were not run are counted in the
        reason: a check that reads identical on two results of three says that one was not looked at."""
        skipped = sum(i.outcome == NOT_RUN for i in items)
        if skipped and skipped < len(items):
            reason = "; ".join(filter(None, [reason, f"{skipped} of {len(items)} not run"]))
        return cls(check_id, worst(i.outcome for i in items), reason, tuple(items))

    @classmethod
    def not_run(cls, check_id: str, reason: str) -> CheckResult:
        return cls(check_id, NOT_RUN, reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "outcome": self.outcome,
            "reason": self.reason,
            "items": [i.to_dict() for i in self.items],
        }


@dataclass(frozen=True)
class VerificationReport:
    """
    The report of a package's check: what was checked (``package``), what checked it (``verifier``), and each check's
    result. ``run_at`` is the one field in which two runs on the same package and machine differ.
    """

    package: Mapping[str, Any]
    verifier: Mapping[str, Any]
    checks: tuple[CheckResult, ...]
    run_at: str = ""

    def counts(self) -> dict[str, int]:
        found = Counter(c.outcome for c in self.checks)
        return {outcome: found.get(outcome, 0) for outcome in OUTCOMES}

    def exit_code(self) -> int:
        return EXIT_FAILED if any(c.outcome == FAILED for c in self.checks) else EXIT_OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": REPORT_FORMAT,
            "schema": REPORT_SCHEMA,
            "run_at": self.run_at,
            "package": dict(self.package),
            "verifier": dict(self.verifier),
            "counts": self.counts(),
            "exit_code": self.exit_code(),
            "checks": [c.to_dict() for c in self.checks],
        }


def for_tier(tier: str) -> tuple[CheckKind, ...]:
    """The checks that run on a package of this tier, in the catalogue's order."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; known: {', '.join(TIERS)}")
    return tuple(kind for kind in CHECKS if tier in kind.tiers)


# ----------------------------------------------------------------------------- tolerances


@dataclass(frozen=True)
class Tolerance:
    """
    How far a number computed again may be from the recorded one: ``of_scale`` times the quantity's own scale (its
    standard error for an estimate) plus ``relative`` times the larger of the two values. One form for every number
    of a class, so that a number computed from an estimate and its standard error (a z, an end of an interval) is
    allowed what both are. ``log_p``: of the natural logarithm of a p-value; ``reciprocal``: of 1/x, for the degrees
    of freedom of a pooled term.
    """

    relative: float
    of_scale: float
    log_p: float
    reciprocal: float


TOLERANCES: dict[str, Tolerance] = {
    # closed-form arithmetic: rounding error, seen at 2e-12 when the rows are permuted
    DETERMINISTIC: Tolerance(relative=1e-8, of_scale=1e-8, log_p=1e-6, reciprocal=1e-8),
    # Newton and IRLS fits: an estimate within 1e-4 of its standard error, the rest to 1e-6
    ITERATIVE: Tolerance(relative=1e-6, of_scale=1e-4, log_p=1e-3, reciprocal=1e-6),
    # what an optional library fits on its own and stops earlier (lifelines: up to 4e-4 standard error from the
    # model's optimum; a concordance that moves by whole pairs). At the floor of a p-value, ln p moves 8 times as much
    # as its z, which may move by 1e-3 + 8 x 1e-4
    ITERATIVE_LOOSE: Tolerance(relative=1e-4, of_scale=1e-3, log_p=2e-2, reciprocal=1e-4),
}
P_FLOOR = 1e-15  # two p-values below it are equal: nothing is known of a p-value there but that it is tiny
MONTE_CARLO = (
    "monte_carlo"  # the basis of a comparison of two draws (stored id, beside the classes)
)
MONTE_CARLO_LEVEL = 0.005  # each side, shared between the terms of a model


def _same(a: Any, b: Any) -> bool:
    """Equal values; two missing numbers (NaN) are the same."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return type(a) is type(b) and a == b


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def difference(recorded: Any, computed: Any, number: Number) -> tuple[float, float] | None:
    """
    The difference between two values of ``number`` as its rule measures it, and what its class allows; None when
    the values cannot be measured against each other (a text, a missing or an infinite value on one side only) or
    when its class has no tolerance (a count, a draw: equal or not).
    """
    tolerance = TOLERANCES.get(number.cls)
    if tolerance is None or not (_is_number(recorded) and _is_number(computed)):
        return None
    a, b = float(recorded), float(computed)
    finite = math.isfinite(a) and math.isfinite(b)
    if not finite and (number.rule != RECIPROCAL or math.isnan(a) or math.isnan(b)):
        return None  # an infinite df is 1/x = 0: measured; anything else that is not finite is not
    if number.rule == LOG:
        if a < P_FLOOR and b < P_FLOOR:
            return 0.0, tolerance.log_p
        if a <= 0 or b <= 0:
            return math.inf, tolerance.log_p
        return abs(math.log(a) - math.log(b)), tolerance.log_p
    if number.rule == RECIPROCAL:
        if a == 0 or b == 0:
            return (0.0 if a == b else math.inf), tolerance.reciprocal
        return abs(1 / a - 1 / b), tolerance.reciprocal
    allowed = tolerance.relative * max(abs(a), abs(b)) + tolerance.of_scale * (number.scale or 0.0)
    return abs(a - b), allowed


def compare_values(recorded: Any, computed: Any, number: Number) -> tuple[str, float, float]:
    """
    One value against the recorded one: the outcome, the difference and what was allowed (0, 0 when identical or
    when the values cannot be measured). ``number`` gives the class, the rule and the scale: the recorded one's. A
    count is equal or it fails; so is a seeded number, which is identical under the same seed, rows and libraries
    (whether they are the same is for the caller to say before comparing).
    """
    if _same(recorded, computed):
        return IDENTICAL, 0.0, 0.0
    measured = difference(recorded, computed, number)
    if measured is None:
        return FAILED, 0.0, 0.0
    found, allowed = measured
    return (WITHIN_TOLERANCE if found <= allowed else FAILED), found, allowed


def compare_numbers(recorded: Number, computed: Number) -> tuple[str, float, float]:
    """A number computed again against the recorded one; a series is compared entry by entry (its worst entry)."""
    if recorded.cls != computed.cls or recorded.rule != computed.rule:
        return FAILED, 0.0, 0.0
    a, b = recorded.value, computed.value
    if isinstance(a, list) != isinstance(b, list):
        return FAILED, 0.0, 0.0
    if not isinstance(a, list):
        return compare_values(a, b, recorded)
    if len(a) != len(b):
        return FAILED, 0.0, 0.0
    outcome, found, allowed = IDENTICAL, 0.0, 0.0
    for one, other in zip(a, b, strict=True):
        this, d, within = compare_values(one, other, recorded)
        if this == FAILED:
            return FAILED, d, within
        if this == WITHIN_TOLERANCE and (
            outcome == IDENTICAL or _ratio(d, within) > _ratio(found, allowed)
        ):
            outcome, found, allowed = this, d, within
    return outcome, found, allowed


def _ratio(found: float, allowed: float) -> float:
    return math.inf if allowed == 0 and found > 0 else (found / allowed if allowed else 0.0)


# ----------------------------------------------------------------------------- a result against its record

NO_MAXIMUM = "no maximum of the likelihood: such an estimate cannot be compared"
REDRAWN = "the imputations were drawn again: only the pooled estimates are compared"


def _prefix(part_id: str) -> str:
    """The kind of a part: what its id holds before its colon or its place ("term", "curve", "model")."""
    return part_id.split(":", 1)[0].split("#", 1)[0]


def place(record: ResultRecord, part_id: str, name: str = "") -> str:
    """
    Where a part is in its record, for a reader to find it without a value of the data: its kind, its place among
    the parts of that kind, its variable, and for a level its place among the variable's levels: "term 4 of 4 (nyha,
    level 3 of 3): log_hr". A level's own name is never written; a variable's is (a package holds the columns'
    names).
    """
    kind = _prefix(part_id)
    same = [p for p in record.parts if _prefix(p.id) == kind]
    part = next((p for p in same if p.id == part_id), None)
    where = (
        kind if part is None or len(same) == 1 else f"{kind} {same.index(part) + 1} of {len(same)}"
    )
    variable = part.labels.get("variable") if part is not None else None
    if variable:
        where += f" ({variable}"
        if "#" in part_id:
            level = int(part_id.rsplit("#", 1)[1])
            levels = 1 + sum(p.id.startswith(f"{kind}:{variable}#") for p in same)
            where += f", level {level + 1} of {levels}"
        where += ")"
    return f"{where}: {name}" if name else where


def _skipped(part_id: str, number: str, skip: Mapping[str, str]) -> str:
    """The reason a part (``number`` empty) or one of its numbers is left out of a comparison, or ''."""
    for key, reason in skip.items():
        if "/" in key:
            if number and key == f"{part_id}/{number}":
                return reason
        elif part_id.startswith(key) and (key.endswith(":") or part_id == key):
            return reason
    return ""


def _flag(part: Any, name: str) -> Any:
    """An exact number of a part that says how to read the others (``extreme``, ``converged``), or None."""
    return next((n.value for n in part.numbers if n.name == name), None)


def _unconverged(record: ResultRecord) -> bool:
    """Whether the fit behind a record has no maximum of the likelihood to compare: it did not converge, or not
    in every imputed dataset."""
    for part in record.parts:
        if part.id != "model":
            continue
        if _flag(part, "converged") is False:
            return True
        within = _flag(part, "converged_in")
        pooling = next((p for p in record.parts if p.id == "pooling"), None)
        m = None if pooling is None else _flag(pooling, "m")
        if within is not None and m is not None and within < m:
            return True
    return False


def compare_records(
    recorded: ResultRecord,
    computed: ResultRecord,
    *,
    name: str = "",
    not_comparable: Mapping[str, str] | None = None,
    absent: Sequence[str] = (),
    draws: str = "same",
) -> CheckItem:
    """
    A result computed again against its record: every number by its class, the counts behind each part exactly.

    Not compared, and the item then reads not comparable at best:

    - a record written by another version of its kind's record;
    - the terms of a fit that did not converge, and a term whose estimate is extreme (``extreme``, ``converged``:
      compared exactly themselves): the likelihood has no maximum there, and any two runs differ;
    - what an optional library computes when it is ``absent`` on one side (the kind says which parts and numbers:
      ``biostats.plan.AnalysisKind.optional``), and what ``not_comparable`` names with its reason (the start of a
      part's id, ``"ph:"``, or a number's place, ``"model/concordance"``).

    ``draws="other"`` is for a pooled result whose imputations were drawn again (their cells' fingerprint differs):
    each pooled estimate is held to the Monte Carlo error of two draws (:func:`monte_carlo_bound`), and nothing
    else of it is compared (the counts at an imputed level, the standard errors, the tests and the curves move
    with the draws). The complete-case analysis rests on the observed rows: it is compared as usual.

    A failure is placed within its kind of part (:func:`place`) and says how many other numbers differ.
    """
    label = name or recorded.kind
    if (recorded.kind, recorded.pooled) != (computed.kind, computed.pooled):
        return CheckItem(label, FAILED, "another kind of analysis was computed")
    if recorded.record_version != computed.record_version:
        return CheckItem(
            label,
            NOT_COMPARABLE,
            f"the result was recorded by another version of the {recorded.kind} record "
            f"({recorded.record_version}, here {computed.record_version})",
        )
    skip = dict(not_comparable or {})
    for library in absent:
        for key in KINDS[recorded.kind].optional.get(library, ()):
            skip.setdefault(key, f"{library} is not installed on one side")
    others = {p.id: p for p in computed.parts}
    only_recorded = [
        p.id for p in recorded.parts if p.id not in others and not _skipped(p.id, "", skip)
    ]
    known = {p.id for p in recorded.parts}
    only_computed = [i for i in others if i not in known and not _skipped(i, "", skip)]
    if only_recorded or only_computed:
        return CheckItem(
            label,
            FAILED,
            f"the parts of the result differ: {len(only_recorded)} recorded only, {len(only_computed)} computed only",
        )
    redrawn = draws == "other" and recorded.pooled
    flat = _unconverged(recorded) or _unconverged(computed)
    reasons: list[str] = [REDRAWN] if redrawn else []
    failures: list[tuple[str, dict[str, Any] | None]] = []
    outcome, largest = IDENTICAL, None
    estimates: list[tuple[str, Any, Any]] = []
    for part in recorded.parts:
        why = _skipped(part.id, "", skip)
        if why:
            reasons.append(why)
            continue
        other = others[part.id]
        term = _prefix(part.id) in ("term", "ph")
        if redrawn:
            if part.id == "pooling":
                if [n.value for n in part.numbers] != [n.value for n in other.numbers]:
                    failures.append(
                        (f"{place(recorded, part.id)}: the datasets pooled differ", None)
                    )
            elif _prefix(part.id) == "term":
                estimates.append((part.id, part, other))
            continue
        if dict(part.support) != dict(other.support):
            failures.append((f"{place(recorded, part.id, 'the counts behind it')} differ", None))
            continue
        if [n.name for n in part.numbers] != [n.name for n in other.numbers]:
            failures.append(
                (f"{place(recorded, part.id, 'its numbers')} are not the same ones", None)
            )
            continue
        extreme = _flag(part, "extreme") is True or _flag(other, "extreme") is True
        for number, again in zip(part.numbers, other.numbers, strict=True):
            why = _skipped(part.id, number.name, skip)
            if why:
                reasons.append(why)
                continue
            if term and (flat or extreme) and number.cls != EXACT:
                reasons.append(NO_MAXIMUM)
                continue
            this, found, allowed = compare_numbers(number, again)
            measured = {
                "quantity": place(recorded, part.id, number.name),
                "difference": found,
                "allowed": allowed,
                "class": number.cls,
            }
            if this == FAILED:
                failures.append((f"{measured['quantity']} differs", measured))
            elif this == WITHIN_TOLERANCE and (
                largest is None
                or _ratio(found, allowed) > _ratio(largest["difference"], largest["allowed"])
            ):
                outcome, largest = WITHIN_TOLERANCE, measured
    if redrawn and estimates and not failures:
        measured = _monte_carlo(recorded, estimates)
        if measured["difference"] > measured["allowed"]:
            failures.append(
                (
                    f"{measured['quantity']} differs by more than two draws of one imputation model do",
                    measured,
                )
            )
        else:
            outcome, largest = WITHIN_MONTE_CARLO, measured
    if (recorded.complete_case is None) != (computed.complete_case is None):
        failures.append(("the complete-case analysis is on one side only", None))
    elif recorded.complete_case is not None and computed.complete_case is not None:
        inner = compare_records(
            recorded.complete_case, computed.complete_case, name=label, not_comparable=skip
        )
        if inner.outcome == FAILED:
            failures.append((f"complete-case analysis: {inner.reason}", inner.largest))
        elif inner.outcome == NOT_COMPARABLE:
            reasons.append(inner.reason)
        elif (
            inner.outcome == WITHIN_TOLERANCE
            and outcome != WITHIN_MONTE_CARLO
            and inner.largest is not None
            and (
                largest is None
                or _ratio(inner.largest["difference"], inner.largest["allowed"])
                > _ratio(largest["difference"], largest["allowed"])
            )
        ):
            outcome, largest = WITHIN_TOLERANCE, dict(inner.largest)
    if failures:
        first, measured = failures[0]
        more = len(failures) - 1
        return CheckItem(
            label, FAILED, first + (f", and {more} other difference(s)" if more else ""), measured
        )
    if redrawn and not estimates:
        return CheckItem(label, NOT_COMPARABLE, REDRAWN)
    if reasons and outcome != WITHIN_MONTE_CARLO:
        return CheckItem(label, NOT_COMPARABLE, "; ".join(dict.fromkeys(reasons)), largest)
    return CheckItem(label, outcome, "; ".join(dict.fromkeys(reasons)), largest)


def monte_carlo_bound(b1: float, m1: int, b2: float, m2: int, terms: int) -> float:
    """
    How far apart two pooled estimates of one quantity may lie when their imputations were drawn independently from
    one model: the difference of two means of m draws has the variance B1/m1 + B2/m2 (B: the between-imputation
    variance), read against a t quantile on m1 + m2 - 2 degrees of freedom at 0.5% each side, shared between the
    ``terms`` of the model (3.9 standard errors of the difference for 9 terms and two sets of 10 datasets).
    """
    from scipy import stats

    df = max(m1 + m2 - 2, 1)
    quantile = float(stats.t.ppf(1 - MONTE_CARLO_LEVEL / max(terms, 1), df))
    return quantile * math.sqrt(max(b1, 0.0) / m1 + max(b2, 0.0) / m2)


def _monte_carlo(record: ResultRecord, estimates: Sequence[tuple[str, Any, Any]]) -> dict[str, Any]:
    """The pooled estimate furthest from its Monte Carlo bound, among the terms of a model."""
    worst_found: dict[str, Any] | None = None
    for part_id, one, other in estimates:
        values = {n.name: n.value for n in one.numbers}
        again = {n.name: n.value for n in other.numbers}
        if not all(_is_number(v.get(k)) for v in (values, again) for k in ("estimate", "b", "m")):
            continue
        allowed = monte_carlo_bound(
            values["b"], values["m"], again["b"], again["m"], len(estimates)
        )
        measured = {
            "quantity": place(record, part_id, "estimate"),
            "difference": abs(values["estimate"] - again["estimate"]),
            "allowed": allowed,
            "class": MONTE_CARLO,
        }
        if worst_found is None or _ratio(measured["difference"], allowed) > _ratio(
            worst_found["difference"], worst_found["allowed"]
        ):
            worst_found = measured
    return worst_found or {"quantity": "", "difference": 0.0, "allowed": 0.0, "class": MONTE_CARLO}


# ----------------------------------------------------------------------------- column profiles


ROUNDING = "rounding"  # the basis of a profile's tolerance (stored ids)
WIDER = "iterative step"
ULPS = 16  # units in the last place allowed per halving of a sum's terms
_EPSILON = 2.220446049250313e-16


def rounding_share(values: int) -> float:
    """
    The share of the sum of the absolute values by which a sum over ``values`` numbers may move when each number is
    off by rounding error and the sum is taken in another order: 16 units in the last place for each halving of the
    terms (7e-14 for a million values). It allows a column computed again by another BLAS; it does not allow one
    value changed by 0.2 among a million, which moves the sum by a share of 7e-9.
    """
    return ULPS * math.log2(values + 1) * _EPSILON


def _numeric_columns_agree(
    was: Mapping[str, Any], now: Mapping[str, Any], share: float
) -> tuple[bool, str, float, float]:
    """
    Whether two numeric columns of the same rows hold the same values up to a relative error of ``share`` on each:
    the sum, the sum of the absolute values and the two weighted sums within ``share`` of the sum of the absolute
    values, the sum of the squares within twice that share of itself, each end within ``share`` of itself (never
    less than 4 units in its last place). With the verdict: the sum that fails, or the one closest to its limit,
    its difference and what it was allowed.
    """
    total = max(abs(was.get("sum_abs") or 0.0), abs(now.get("sum_abs") or 0.0))
    squares = max(abs(was.get("sum_sq") or 0.0), abs(now.get("sum_sq") or 0.0))
    closest = ("sum", 0.0, 0.0)
    for key in ("sum", "sum_abs", "sum_sq", "min", "max", "mix", "mix2"):
        a, b = was.get(key), now.get(key)
        if a is None or b is None:
            if key in was or key in now:
                # a sum that could not be computed (too large for a float) on either side: nothing was compared
                return False, key, math.inf, 0.0
            continue  # a column without a finite value has no sums: its digest and its counts said it all
        if key == "sum_sq":
            allowed = 2 * share * squares
        elif key in ("min", "max"):
            allowed = max(share, 4 * _EPSILON) * max(abs(a), abs(b))
        else:
            allowed = share * total
        found = abs(a - b)
        if found > allowed:
            return False, key, found, allowed
        if _ratio(found, allowed) >= _ratio(closest[1], closest[2]):
            closest = (key, found, allowed)
    return True, *closest


def compare_profiles(
    recorded: Mapping[str, Any],
    computed: Mapping[str, Any],
    *,
    name: str = "",
    iterative: Sequence[str] = (),
) -> CheckItem:
    """
    A table's profile (``core.profiles.frame_profile``) against the recorded one.

    Identical when every column's digest is. Within tolerance when the columns whose digests differ hold real
    numbers and their sums agree up to the rounding error of a sum (:func:`rounding_share`): the same values computed
    by another BLAS or added in another order. A changed value, or two values exchanged between rows, is a failure
    however large the table (the weighted sums move with the rows). Whole numbers, text, dates and booleans are
    equal or they are not. The columns named in ``iterative`` were written by a step that stops at a tolerance (a
    fitted imputation): they are allowed the tolerance of a fitted number instead, and the reason says so. A failure
    names a column by its place.
    """
    if recorded.get("version") != PROFILE_VERSION or computed.get("version") != PROFILE_VERSION:
        return CheckItem(
            name, NOT_COMPARABLE, "the profiles were written by another version of the profiles"
        )
    if recorded.get("rows") != computed.get("rows") or recorded.get("rows_fp") != computed.get(
        "rows_fp"
    ):
        return CheckItem(name, FAILED, "the rows are not the same")
    names = list(recorded.get("names") or [])
    if names != list(computed.get("names") or []):
        return CheckItem(name, FAILED, "the columns are not the same, or not in the same order")
    largest: dict[str, Any] | None = None
    rounded, wider = 0, 0
    for index, (column, was, now) in enumerate(
        zip(names, recorded["columns"], computed["columns"], strict=True)
    ):
        if was == now:
            continue
        failed = CheckItem(name, FAILED, f"column {index + 1} does not hold the same values")
        same_shape = all(
            was.get(k) == now.get(k) for k in ("kind", "rows", "missing", "plus_inf", "minus_inf")
        )
        if was.get("kind") != NUMBER or not same_shape or (was.get("whole") and now.get("whole")):
            return failed
        loose = column in iterative
        values = int(was["rows"]) - int(was["missing"])
        share = TOLERANCES[ITERATIVE].relative if loose else rounding_share(values)
        agree, key, found, allowed = _numeric_columns_agree(was, now, share)
        measured = {
            "quantity": f"column {index + 1}: {key}",
            "difference": found,
            "allowed": allowed,
            "class": WIDER if loose else ROUNDING,
        }
        if not agree:
            return CheckItem(name, FAILED, failed.reason, measured)
        if largest is None or _ratio(found, allowed) > _ratio(
            largest["difference"], largest["allowed"]
        ):
            largest = measured
        rounded, wider = rounded + (not loose), wider + loose
    reasons = []
    if rounded:
        reasons.append(
            f"{rounded} numeric column(s) differ within the rounding error of their sums"
        )
    if wider:
        reasons.append(
            f"{wider} column(s) written by an iterative step differ within its tolerance"
        )
    return CheckItem(
        name,
        WITHIN_TOLERANCE if reasons else IDENTICAL,
        "; ".join(reasons),
        largest if reasons else None,
    )


__all__ = [
    "CHECKS",
    "EXIT_FAILED",
    "EXIT_OK",
    "EXIT_UNVERIFIABLE",
    "FAILED",
    "IDENTICAL",
    "MONTE_CARLO",
    "NOT_COMPARABLE",
    "NOT_RUN",
    "OUTCOMES",
    "P_FLOOR",
    "REPORT_FORMAT",
    "REPORT_SCHEMA",
    "REPRODUCTION",
    "ROUNDING",
    "SHAREABLE",
    "TIERS",
    "TOLERANCES",
    "WIDER",
    "WITHIN_MONTE_CARLO",
    "WITHIN_TOLERANCE",
    "CheckItem",
    "CheckKind",
    "CheckResult",
    "Tolerance",
    "VerificationReport",
    "compare_numbers",
    "compare_profiles",
    "compare_records",
    "compare_values",
    "difference",
    "for_tier",
    "monte_carlo_bound",
    "place",
    "rounding_share",
    "worst",
]
