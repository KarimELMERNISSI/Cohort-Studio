"""
The small-cell rule of a shared package (M8): a count of participants is written exactly, or as "fewer than k".

A package meant to travel (the shareable tier) holds no row, and no count from 1 to k - 1 of participants with a
characteristic: an exclusion reason, a level, a group, an event, a cell of a table. k is 10 unless the study says
otherwise, never below 5 (``FLOOR_K``), and never below the installation's own floor, which the caller passes
(``DisclosurePolicy.for_study``: this module reads no setting). A count is exact or withheld, never rounded, so
that the numbers shown still add up and a reader holding the data can compare them. Zeros are shown.

A withheld count must not come back by subtraction, so the participant flow is disclosed with complementary
suppression (:func:`disclose_flow`): the reasons of an exit that are too small are shown as one line with the next
smallest ones until the line reaches k (the line names none of them: a line of 10 naming ten criteria would give
each), an exit that is too small is merged with the next transition (the box between them is not shown), and when
every exclusion together stays below k only the population is shown. :func:`recoverable` then works out what a
reader can from what is shown (a box is the previous one minus its exits plus the rows added; an exit is the sum of
its reasons; no count is negative) and names any withheld count that comes back: a package for which it finds one is
never written.

Exempt, and always exact: the rows of the analysis population, the number each analysis used, and so the number of
values missing in a variable (reporting asks for them, STROBE items 13 and 14). The rows an analysis left out for
another reason (a group that is not compared, a group too small to draw, a follow-up time below zero) are a count
of participants with a characteristic: an analysis that left out 1 to k - 1 such rows is withheld.

A result is disclosed from its record (:func:`disclose_record`, ``biostats.plan.ResultRecord``), never from the
result itself, with its spec and with its analysis as the package's flow shows it. A result whose analysis the flow
withholds, or does not list, is withheld: its own counts beside the flow's would give back what the flow withheld.
So is a result whose record counts 1 to k - 1 rows left out, whatever the flow shows. Then, by the rule of its
kind:

- a model (Cox, logistic, count) is withheld whole when its rows, its rows with the event, its rows without it or
  its residual degrees of freedom are below k (a count model's events are a sum of counts: the people behind them
  are its rows with an event). A term that compares two cells is shown only when both hold at least k rows, k with
  the event and k without: a factor's level, or the upper value of a covariate coded with two values, against the
  reference; the rows off the commonest value of a numeric covariate that at least half of the rows hold at one
  value, against the rows on it. The coefficient of a level held by one participant gives that participant's
  outcome, and a level where nobody, or everybody, had the event says so of each of its participants. A term
  without a maximum of the likelihood is not shown either;
- no count travels beside a term, but in a model of categories alone the numbers of a shown term are its two
  cells' counts in another form (the coefficient is their log odds ratio, the square of its standard error the sum
  of their reciprocals), and the model's totals, less the cells shown, give the withheld ones. So the cells
  withheld of a variable hold k of each count together, or the smallest cell shown is withheld with them, and the
  next, until they do or none is left. The numbers of the fit as a whole (likelihood, AIC, pseudo-R2, AUC,
  concordance, dispersion, whether it converged) are equations in every cell of the model: they are written only
  when no term is withheld. A model none of whose terms can be shown, or no term of whose comparison can, is
  withheld whole. A withheld term is counted by its variable, never named by its level;
- a Kaplan-Meier curve is given at times that are multiples of one step for the whole result: at least a
  hundredth of the follow-up shown and five times the data's own unit, so that no time given is a participant's.
  At least k events lie between two of them and none or at least k after the last (in every imputed dataset, for
  a pooled curve), and k rows are still followed at the last. The result says until when k rows were followed;
  the median is rounded up to the step, its lower limit down and its upper limit up, or said to be later than
  that time. A group of fewer than k rows, events or rows without the event is withheld whole, and with it the
  smallest groups shown until the groups withheld hold k of each together: the number analysed, less the groups
  shown, gives their sum. A pooled result and its complete cases withhold the same groups. The log-rank test is
  shown when every group is;
- beside a pooled result, its complete-case analysis is shown only when the two differ by none, or by at least k,
  of their rows, their rows with the event and their rows without it (group by group for curves): they differ by
  the rows that lack a value, and by their outcomes.

Only the numbers a kind's rule names are written: a number added to a record later does not travel until the rule
says so, and a kind without a rule is withheld. No warning's wording travels: what a reader needs of the cautions
is written as numbers (a model's parameters and events per parameter, a curve's step and follow-up), and
:func:`reading_notes` gives the sentences a package says them with.

Before a package is offered, the texts that come from its user and from the data are read for a value held by 1 to
k - 1 rows (:class:`RareFinder`): a column named after a level, a criterion that names a site, a title. And every
text that reads as an expression, for a comparison that sets fewer than k rows apart by a value written in it:
"age == 97" selects one participant and says by what. A test that cannot be run safely is reported as not checked.
A hit names its place and its column, never the value.

What the writer of a package must do, which this module cannot:

- run one closure over every file, and write nothing while a count comes back (:func:`flow_closure` gives the
  flow's; :class:`Closure` takes the rest). Its unknowns are the cells of the cross-classification of every
  grouping variable and factor the package shows, each count apart (rows, events, rows without the event),
  computed from the data: each group, each level and each analysis shown is a sum of cells, and the shown terms of
  a model of categories alone are shown cells. Two groupings of one population give back a cell that neither
  shows (the groups of a country, less the sites shown in it);
- carry only results computed on the package's own population and data, compared by fingerprint and never by a
  count, and pass ``disclose_record`` and ``disclose_spec`` the analysis as the flow shows it;
- compare the numbers a package shows with those of earlier packages of the same data (two packages exported with
  different ticks differ by a withheld count): the export log keeps them;
- make the policy with ``DisclosurePolicy.for_study`` and the installation's floor, write it as it is recorded,
  and refuse a written policy that lacks what the study asked for and the floor;
- give the finder the user's and the data's texts only (its own sentences are made of fixed words and of disclosed
  numbers, never of a box's details, an analysis's note, a stored label or a warning), the format's fixed names
  as ``own``, each criterion stored as numbers through :meth:`RareFinder.bounds`, the identifying columns the
  person who shares marks, and copies of the datasets; run it in a process with a time limit, and show what it
  did not finish as not checked;
- check every member against a closed schema: no fingerprint of rows, of cells or of a spec, no intermediate head
  of the trace, no number marked as one participant's;
- name in the export log what is left out: the results by kind and place, the hits overridden, the texts not
  checked.
"""

from __future__ import annotations

import ast
import bisect
import copy
import importlib
import math
import re
import unicodedata
import warnings
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import accumulate
from typing import Any

import numpy as np
import pandas as pd

from biostats.plan import DETERMINISTIC, EXACT, AnalysisSpec, Number, Part, ResultRecord
from core.attrition import DATA, LOADED, SOURCE, AnalysisCount, FlowBox, ParticipantFlow
from core.expressions.safe_eval import DEFAULT_POLICY, evaluate_tree

DEFAULT_K = 10
FLOOR_K = 5  # nothing lower is accepted, whatever a study or an installation asks
POLICY_VERSION = 1  # of the rules below: a package records it with its k


@dataclass(frozen=True)
class DisclosurePolicy:
    """
    The smallest count of participants a shared document gives (``k``), and the version of the rules. A policy made
    for a study (:meth:`for_study`) also records the k the study asked for and the installation's floor, so that a
    package says why its k is not the study's.
    """

    k: int = DEFAULT_K
    version: int = POLICY_VERSION
    asked: int | None = field(default=None, compare=False)
    floor: int | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.k, bool) or not isinstance(self.k, int):
            raise ValueError("k is a whole number")
        if self.k < FLOOR_K:
            raise ValueError(f"k is never below {FLOOR_K}")

    @classmethod
    def for_study(cls, k: int | None, *, floor: int) -> DisclosurePolicy:
        """
        The policy of a study that asks for ``k`` (None: the default, 10) on an installation whose floor is
        ``floor``: the larger of the two. The floor has no default: the caller reads it from the installation's
        settings (``CMP_DISCLOSURE_MIN``), and a package written without it would follow no installation. A k or a
        floor below 5, or that is not a whole number, is refused.
        """
        if isinstance(floor, bool) or not isinstance(floor, int):
            raise ValueError("an installation's floor is a whole number")
        if floor < FLOOR_K:
            raise ValueError(f"an installation's floor is never below {FLOOR_K}")
        asked = cls(DEFAULT_K if k is None else k).k
        return cls(max(asked, floor), asked=asked, floor=floor)

    def shows(self, count: int) -> bool:
        """Whether a count of participants is given: none, or at least k."""
        return count == 0 or count >= self.k

    def to_dict(self) -> dict[str, int]:
        """``{"k": 10, "asked": 5, "floor": 10, "version": 1}``; without ``asked`` and ``floor`` for a policy that
        was not made for a study."""
        made_for = {} if self.asked is None else {"asked": self.asked, "floor": self.floor}
        return {"k": self.k, **made_for, "version": self.version}


@dataclass(frozen=True)
class Shown:
    """A count as a shared document gives it: the number, or that it is fewer than k (``n`` is None)."""

    n: int | None
    k: int

    @property
    def text(self) -> str:
        return f"fewer than {self.k}" if self.n is None else f"{self.n:,}"

    def to_json(self) -> int | dict[str, int]:
        return {"fewer_than": self.k} if self.n is None else self.n


def shown(count: int, policy: DisclosurePolicy) -> Shown:
    return Shown(int(count) if policy.shows(int(count)) else None, policy.k)


def part_of(count: int, total: int, policy: DisclosurePolicy) -> str:
    """
    A count beside its total, in words: a total gives the rest back, so both the count and the rest must be none or
    at least k. "412 of the 415" has a rest of 3: it reads "all but fewer than 10 of the 415".
    """
    count, total = int(count), int(total)
    if not 0 <= count <= total:
        raise ValueError("a part is between nothing and its total")
    if policy.shows(count) and policy.shows(total - count):
        return f"{count:,} of the {total:,}"
    if not policy.shows(count):
        return f"fewer than {policy.k} of the {total:,}"
    return f"all but fewer than {policy.k} of the {total:,}"


# ----------------------------------------------------------------------------- the participant flow


@dataclass(frozen=True)
class DisclosedReason:
    """
    A line of reasons: one criterion with the rows it was the first reason for (``names`` holds its name), or the
    other criteria of the step together (``names`` is empty: naming them would say each holds at least one row, and
    a line of 10 for ten names would give each). ``lacking`` is how many of a single criterion's rows lack a value
    it reads, when that can be said beside the line's count. ``parts`` says which reasons of which exits the line
    covers, for the check of what a reader can work out; it is never written.
    """

    names: tuple[str, ...]
    n: int
    lacking: int | None = None
    parts: tuple[str, ...] = ()


@dataclass(frozen=True)
class DisclosedTransition:
    """
    What happens between two boxes that are shown: the rows leaving and the rows added, each none or at least k.
    ``exits`` and ``labels`` are those of the exits it covers (several when exits were merged).
    """

    after: str
    before: str
    exits: tuple[str, ...]
    labels: tuple[str, ...]
    excluded: int
    added: int
    reasons: tuple[DisclosedReason, ...] = ()


@dataclass(frozen=True)
class DisclosedBox:
    id: str
    label: str
    n: int
    kind: str


@dataclass(frozen=True)
class DisclosedAnalysis:
    """
    What a reported analysis used. ``n`` is the fewest rows it analysed and ``n_most`` the most when it analyses
    several variables, each on its own rows; ``missing`` the rows it could not use for a missing value (the most,
    for several variables); ``left_out`` the rows it left out for another reason, none or at least k, with the words
    for them; ``imputations`` the datasets it pooled. All exact.
    """

    label: str
    n: int
    n_most: int | None = None
    missing: int = 0
    left_out: int = 0
    left_out_words: str = ""
    imputations: int | None = None


@dataclass(frozen=True)
class DisclosedFlow:
    """
    The participant flow as a shared document shows it. ``withheld`` says what is not shown at all ("fewer than 10
    rows excluded in all", "1 analysis withheld: ..."): when it is the exclusions, the flow starts at the analysis
    population, and the number of rows loaded appears nowhere.
    """

    k: int
    boxes: tuple[DisclosedBox, ...]
    transitions: tuple[DisclosedTransition, ...] = ()
    analyses: tuple[DisclosedAnalysis, ...] = ()
    withheld: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "k": self.k,
            "boxes": [vars(b) for b in self.boxes],
            "transitions": [
                {
                    **{key: value for key, value in vars(t).items() if key != "reasons"},
                    "exits": list(t.exits),
                    "labels": list(t.labels),
                    "reasons": [
                        {"names": list(r.names), "n": r.n, "lacking": r.lacking} for r in t.reasons
                    ],
                }
                for t in self.transitions
            ],
            "analyses": [vars(a) for a in self.analyses],
            "withheld": list(self.withheld),
        }


@dataclass(frozen=True)
class _Reason:
    part: str  # "<exit id>#<place>", or the exit's id when the exit has no reasons of its own
    name: str
    rows: int
    lacking: int = 0  # of which lack a value the criterion reads


def _parts(exit_) -> list[_Reason]:
    """The reasons of an exit; an exit without reasons that sum to it is its own reason."""
    if exit_.reasons and sum(n for _, n in exit_.reasons) == exit_.n:
        lacking = dict(exit_.missing)
        return [
            _Reason(f"{exit_.id}#{place}", name, int(n), int(lacking.get(name, 0)))
            for place, (name, n) in enumerate(exit_.reasons)
        ]
    return [_Reason(exit_.id, exit_.label, int(exit_.n))] if exit_.n else []


@dataclass(frozen=True)
class _Step:
    """A transition of the flow as built: from one box of the main column to the next."""

    index: int  # of the box it leads to
    exits: tuple[str, ...]
    labels: tuple[str, ...]
    excluded: int
    added: int
    reasons: tuple[_Reason, ...]


def _steps(flow: ParticipantFlow) -> list[_Step]:
    steps = []
    for index in range(1, len(flow.main)):
        leaving = flow.exits_after(flow.main[index - 1].id)
        reasons = [reason for exit_ in leaving for reason in _parts(exit_)]
        steps.append(
            _Step(
                index,
                tuple(e.id for e in leaving),
                tuple(e.label for e in leaving),
                sum(int(e.n) for e in leaving),
                int(flow.main[index].added),
                tuple(reasons),
            )
        )
    return steps


def _fine(group: Sequence[_Step], policy: DisclosurePolicy) -> bool:
    return policy.shows(sum(s.excluded for s in group)) and policy.shows(
        sum(s.added for s in group)
    )


def _grouped(steps: Sequence[_Step], policy: DisclosurePolicy) -> list[list[_Step]]:
    """
    The transitions in groups whose rows leaving and rows added are each none or at least k: a transition that is too
    small takes the next one, and the last group, when still too small, joins the one before it.
    """
    groups: list[list[_Step]] = []
    current: list[_Step] = []
    for step in steps:
        current.append(step)
        if _fine(current, policy):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    while len(groups) > 1 and not _fine(groups[-1], policy):
        last = groups.pop()
        groups[-1] = groups[-1] + last
    return groups


def _reason_lines(
    reasons: Sequence[_Reason], policy: DisclosurePolicy
) -> tuple[DisclosedReason, ...]:
    """
    The reasons of a transition: those of at least k each on its own line; those below k on one line with the next
    smallest ones until the line reaches k, which names none of them. When that line would take every reason, none
    is listed: the transition's total says it all.
    """
    counted = [r for r in reasons if r.rows]
    together = {i for i, r in enumerate(counted) if not policy.shows(r.rows)}
    if not together:
        return tuple(_line(r, policy) for r in counted)
    by_size = sorted(
        (i for i in range(len(counted)) if i not in together),
        key=lambda i: (counted[i].rows, counted[i].name, i),
    )
    while by_size and sum(counted[i].rows for i in together) < policy.k:
        together.add(by_size.pop(0))
    if not by_size:
        return ()
    merged = [r for i, r in enumerate(counted) if i in together]
    return (
        *(_line(r, policy) for i, r in enumerate(counted) if i not in together),
        DisclosedReason((), sum(r.rows for r in merged), None, tuple(r.part for r in merged)),
    )


def _line(reason: _Reason, policy: DisclosurePolicy) -> DisclosedReason:
    # the rows lacking a value are a part of the line's count: given only when the rest could be given too
    said = (
        reason.lacking
        if policy.shows(reason.lacking) and policy.shows(reason.rows - reason.lacking)
        else None
    )
    return DisclosedReason((reason.name,), reason.rows, said, (reason.part,))


OTHER_POPULATION, FEW_LEFT_OUT = "population", "left_out"  # why an analysis is withheld from a flow


def _analysed(a: AnalysisCount) -> int:
    """The rows an analysis used. A pooled analysis uses every row it did not leave out: the imputation fills the
    missing values."""
    pooled = a.n_analysed is None or bool(a.imputations)
    return int(a.n_population) - int(a.n_left_out) if pooled else int(a.n_analysed or 0)


def _analysis_withheld(a: AnalysisCount, policy: DisclosurePolicy, population: int) -> str:
    """Why the flow does not show an analysis ("" when it does)."""
    if int(a.n_population) != int(population):
        return OTHER_POPULATION
    return "" if policy.shows(a.n_left_out) else FEW_LEFT_OUT


def _analyses(
    counts: Sequence[AnalysisCount], policy: DisclosurePolicy, population: int
) -> tuple[tuple[DisclosedAnalysis, ...], tuple[str, ...]]:
    """
    The analyses as a shared document shows them, and what is said of those withheld. The rows an analysis could not
    use for a missing value are exact. The rows it left out for another reason are a count of participants with a
    characteristic (the size of a group): an analysis that left out 1 to k - 1 of them is withheld whole, since its
    number analysed would give them back. An analysis computed on another population than the flow's (before a
    criterion was ticked) is withheld too: its number beside the flow's gives the rows between the two.
    """
    shown, hidden, other = [], 0, 0
    for a in counts:
        analysed = _analysed(a)
        why = _analysis_withheld(a, policy, population)
        if why:
            hidden += why == FEW_LEFT_OUT
            other += why == OTHER_POPULATION
            continue
        shown.append(
            DisclosedAnalysis(
                label=a.label,
                n=analysed,
                n_most=None if a.n_most is None or a.n_most == analysed else int(a.n_most),
                missing=int(a.n_population) - int(a.n_left_out) - analysed,
                left_out=int(a.n_left_out),
                left_out_words=a.left_out if a.n_left_out else "",
                imputations=a.imputations,
            )
        )
    said = []
    if hidden:
        said.append(
            f"{hidden} analys{'es' if hidden != 1 else 'is'} withheld: fewer than {policy.k} rows were left out "
            "for another reason than a missing value"
        )
    if other:
        said.append(
            f"{other} analys{'es' if other != 1 else 'is'} withheld: computed on another population than this one "
            "(run again to include)"
        )
    return tuple(shown), tuple(said)


def disclose_flow(
    flow: ParticipantFlow, policy: DisclosurePolicy, analyses: Sequence[AnalysisCount] = ()
) -> DisclosedFlow:
    """
    The flow as a shared document shows it under ``policy`` (see the module's description), with the reported
    ``analyses`` from their counts (the flow's own analysis boxes hold written details, which are never copied).
    ValueError when the flow's numbers do not add up (the trace did not record every change of the rows), when the
    population itself is below k (a shared package is refused then), or when what would be shown gives a withheld
    count back (:func:`recoverable`: never expected, and never written).
    """
    if not flow.main:
        raise ValueError("the flow has no box")
    if flow.inconsistencies() or flow.notes:
        raise ValueError(
            "the flow's numbers do not add up: a shared package needs a trace that records every change of the rows"
        )
    population = flow.main[-1]
    if 0 < population.n < policy.k:
        raise ValueError(
            f"the analysis population holds fewer than {policy.k} rows: nothing of it can be shared"
        )
    shown, said_of_analyses = _analyses(analyses, policy, population.n)
    groups = _grouped(_steps(flow), policy)
    if len(groups) == 1 and not _fine(groups[0], policy):
        said = []
        if not policy.shows(sum(s.excluded for s in groups[0])):
            said.append(f"fewer than {policy.k} rows excluded in all")
        if not policy.shows(sum(s.added for s in groups[0])):
            said.append(f"fewer than {policy.k} rows added in all")
        disclosed = DisclosedFlow(
            policy.k, (_box(population),), (), shown, (*said, *said_of_analyses)
        )
    else:
        boxes = [_box(flow.main[0])]
        transitions = []
        for group in groups:
            start, end = flow.main[group[0].index - 1], flow.main[group[-1].index]
            transitions.append(
                DisclosedTransition(
                    after=start.id,
                    before=end.id,
                    exits=tuple(i for s in group for i in s.exits),
                    labels=tuple(label for s in group for label in s.labels),
                    excluded=sum(s.excluded for s in group),
                    added=sum(s.added for s in group),
                    reasons=_reason_lines([r for s in group for r in s.reasons], policy),
                )
            )
            boxes.append(_box(end))
        disclosed = DisclosedFlow(
            policy.k, tuple(boxes), tuple(transitions), shown, said_of_analyses
        )
    found = recoverable(flow, disclosed, analyses)
    if found and disclosed.transitions:
        # merged or not, the steps as listed give a count back (five exits listed for five rows: each holds one):
        # the population alone is shown
        disclosed = DisclosedFlow(
            policy.k,
            (_box(population),),
            (),
            shown,
            (
                f"the steps of the flow are withheld: as listed, they would give back a count below {policy.k}",
                *said_of_analyses,
            ),
        )
        found = recoverable(flow, disclosed, analyses)
    if found:
        raise ValueError(
            f"the flow as it would be shown gives back {len(found)} count(s) below {policy.k}: not written"
        )
    return disclosed


# What a box is called in a shared document, by its place in the flow: the label a session shows can hold the name of
# a file or of a dataset ("Records read in patients_dupont.csv"), which is not written; the source is identified by
# its checksum. The other labels are made of fixed words and of the names of the criteria's families.
_BOX_WORDS = {
    "source": "Records read in the source file",
    "loaded": "Records loaded",
    "dataset": "Dataset",
}


def _box(box: FlowBox) -> DisclosedBox:
    label = box.label
    if box.id in _BOX_WORDS and box.kind in (SOURCE, LOADED, DATA):
        label = _BOX_WORDS["source" if box.kind == SOURCE else box.id]
    return DisclosedBox(box.id, label, int(box.n), box.kind)


# ----------------------------------------------------------------------------- what a reader can work out


def _solved(rows: list[list[Fraction]], width: int) -> dict[int, Fraction]:
    """The unknowns that the equations ``rows`` (coefficients, then the constant) fix to one value: Gauss-Jordan
    elimination in exact arithmetic; an unknown is fixed when its row holds no other unknown."""
    pivots: dict[int, int] = {}
    r = 0
    for column in range(width):
        pivot = next((i for i in range(r, len(rows)) if rows[i][column] != 0), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        scale = rows[r][column]
        rows[r] = [v / scale for v in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][column] != 0:
                factor = rows[i][column]
                rows[i] = [a - factor * b for a, b in zip(rows[i], rows[r], strict=True)]
        pivots[column] = r
        r += 1
        if r == len(rows):
            break
    if any(row[width] != 0 and not any(row[:width]) for row in rows):
        raise ValueError("the flow's numbers contradict each other")
    return {
        column: rows[row][width]
        for column, row in pivots.items()
        if all(rows[row][other] == 0 for other in range(width) if other != column)
    }


def _bounded(
    equations: Sequence[tuple[dict[str, int], int]], names: Sequence[str], least: Mapping[str, int]
) -> dict[str, tuple[Fraction, Fraction | None]]:
    """
    The range each unknown can lie in, knowing that no count is negative (and ``least`` of some of them): the
    equations are read one unknown at a time (it is what the others leave of the constant), the ranges narrowed until
    nothing moves. An unknown whose range closes on one value is known to the reader although no elimination gives
    it (a sum of 0, or a sum that equals the least its members can hold).
    """
    low: dict[str, Fraction] = {name: Fraction(least.get(name, 0)) for name in names}
    high: dict[str, Fraction | None] = dict.fromkeys(names)  # None: no upper limit known
    for _ in range(4 * len(names) + 8):
        moved = False
        for terms, constant in equations:
            for name, coefficient in terms.items():
                # coefficient * x = constant - sum of the others: the others' extremes bound it
                least_here: Fraction | None = Fraction(constant)
                most: Fraction | None = Fraction(constant)
                for other, c in terms.items():
                    if other == name:
                        continue
                    top = high[other]
                    if c > 0:
                        most = None if most is None else most - c * low[other]
                        least_here = (
                            None if least_here is None or top is None else least_here - c * top
                        )
                    else:
                        least_here = None if least_here is None else least_here - c * low[other]
                        most = None if most is None or top is None else most - c * top
                if coefficient < 0:
                    least_here, most = (
                        None if most is None else most / coefficient,
                        None if least_here is None else least_here / coefficient,
                    )
                else:
                    least_here = None if least_here is None else least_here / coefficient
                    most = None if most is None else most / coefficient
                if least_here is not None and least_here > low[name]:
                    low[name], moved = least_here, True
                if most is not None and (high[name] is None or most < high[name]):
                    high[name], moved = most, True
        if not moved:
            break
    return {name: (low[name], high[name]) for name in names}


class Closure:
    """
    What a reader can work out of counts that add up. Each unknown is a count of participants: never negative, and
    at least 1 when the reader knows it holds a row (an exit that is listed only because it does). Each equation
    says that a sum of unknowns, each times a whole number, is a number the reader is shown or knows by how the
    documents are made. :meth:`found` gives the unknowns that the equations, and the ranges they leave, fix to one
    value; :meth:`comes_back`, those among them from 1 to k - 1.

    The flow's own closure is :func:`flow_closure`. The writer of a package adds to it what each result shows, and
    writes nothing while a count comes back. For that, the unknowns are the cells of the cross-classification of
    every grouping variable and factor the package shows, each count apart (rows, events, rows without the event):
    a group, a level, an analysis's number are each a sum of cells. With one unknown per group of each variable,
    two groupings of one population look independent, and the cell that only their difference shows (the rows of a
    country, less the sites shown in it) is not seen to come back.
    """

    def __init__(self) -> None:
        self._names: list[str] = []
        self._truth: dict[str, int] = {}
        self._least: dict[str, int] = {}
        self._equations: list[tuple[dict[str, int], int]] = []

    def unknown(self, name: str, value: int, *, at_least: int = 0) -> str:
        """Declare a count by its name and its true value (which no reader is told by this); again to raise what
        the reader knows it holds at least."""
        if name not in self._truth:
            self._names.append(name)
            self._truth[name] = int(value)
        elif self._truth[name] != int(value):
            raise ValueError(f"{name} is declared with two values")
        if at_least > self._truth[name]:
            raise ValueError(f"{name} holds less than the reader is said to know")
        self._least[name] = max(self._least.get(name, 0), int(at_least))
        return name

    def equation(self, terms: Mapping[str, int], constant: int) -> None:
        """Something the reader knows: the sum of ``terms`` (an unknown's name, its whole-number factor) is
        ``constant``. Refused when the true counts do not satisfy it: a reader is told nothing false."""
        kept = {name: int(factor) for name, factor in terms.items() if factor}
        if not kept:
            return
        if sum(factor * self._truth[name] for name, factor in kept.items()) != int(constant):
            raise ValueError("an equation that the counts do not satisfy")
        self._equations.append((kept, int(constant)))

    def shown(self, name: str) -> None:
        """The reader is given this count."""
        self.equation({name: 1}, self._truth[name])

    def found(self) -> set[str]:
        """The unknowns the reader can fix to one value."""
        position = {name: i for i, name in enumerate(self._names)}
        rows = []
        for terms, constant in self._equations:
            row = [Fraction(0)] * (len(self._names) + 1)
            for name, factor in terms.items():
                row[position[name]] += factor
            row[-1] = Fraction(constant)
            rows.append(row)
        fixed = {self._names[index] for index in _solved(rows, len(self._names))}
        ranges = _bounded(self._equations, self._names, self._least)
        return fixed | {name for name, (low, high) in ranges.items() if low == high}

    def comes_back(self, k: int) -> list[str]:
        """The counts from 1 to k - 1 that the reader can work out, in the order they were declared."""
        fixed = self.found()
        return [name for name in self._names if name in fixed and 0 < self._truth[name] < k]


def flow_closure(
    flow: ParticipantFlow, disclosed: DisclosedFlow, analyses: Sequence[AnalysisCount] = ()
) -> Closure:
    """
    What a reader of ``disclosed`` knows of the flow's counts (``box:<id>``, ``exit:<id>``, ``added:<box>``,
    ``reason:<exit>#<place>``, ``left_out:<place of the analysis>``). The reader knows how a flow is made: a box is
    the previous one minus its exits plus the rows added, an exit is the sum of its reasons, an exit that is listed
    holds a row, and an analysis used the population minus its missing values minus the rows it left out. They
    know every number shown: the boxes, each transition's rows leaving and added (a sum over the exits it covers),
    each line of reasons (a sum over the reasons it covers), each analysis's rows and missing values.
    """
    closure = Closure()
    steps = _steps(flow)
    sizes: dict[str, int] = {}
    for box in flow.main:
        closure.unknown(f"box:{box.id}", box.n)
    for step in steps:
        before, after = flow.main[step.index - 1], flow.main[step.index]
        terms = {f"box:{after.id}": 1, f"box:{before.id}": -1}
        for exit_ in flow.exits_after(before.id):
            terms[closure.unknown(f"exit:{exit_.id}", exit_.n)] = 1
            sizes[exit_.id] = int(exit_.n)
            parts = [r for r in _parts(exit_) if r.part != exit_.id]
            if parts:  # an exit is the sum of its reasons
                for reason in parts:
                    closure.unknown(f"reason:{reason.part}", reason.rows)
                closure.equation(
                    {f"exit:{exit_.id}": 1, **{f"reason:{r.part}": -1 for r in parts}}, 0
                )
        terms[closure.unknown(f"added:{after.id}", after.added)] = -1
        closure.equation(terms, 0)  # after - before + exits - added = 0
    # what is shown
    for box in disclosed.boxes:
        closure.shown(f"box:{box.id}")
    ids = [b.id for b in flow.main]
    for transition in disclosed.transitions:
        covered = steps[ids.index(transition.after) : ids.index(transition.before)]
        closure.equation({f"exit:{i}": 1 for s in covered for i in s.exits}, transition.excluded)
        closure.equation({f"added:{flow.main[s.index].id}": 1 for s in covered}, transition.added)
        exits = {i for s in covered for i in s.exits}
        for (
            exit_id
        ) in transition.exits:  # listed, so it holds a row: an exit exists only when it does
            if sizes.get(exit_id, 0) >= 1:
                closure.unknown(f"exit:{exit_id}", sizes[exit_id], at_least=1)
        for line in transition.reasons:
            # a part is a reason of an exit, or a whole exit that has no reasons
            members = [f"exit:{part}" if part in exits else f"reason:{part}" for part in line.parts]
            closure.equation(dict.fromkeys(members, 1), line.n)
    # each analysis shown: the population, less its missing values, less the rows it left out, is what it used
    population = f"box:{flow.main[-1].id}"
    pairs = _shown_analyses(analyses, disclosed)
    for place, (count, shown_as) in enumerate(pairs):
        left = closure.unknown(f"left_out:{place}", count.n_left_out)
        closure.equation({population: 1, left: -1}, shown_as.n + shown_as.missing)
    return closure


def recoverable(
    flow: ParticipantFlow, disclosed: DisclosedFlow, analyses: Sequence[AnalysisCount] = ()
) -> list[str]:
    """
    The counts from 1 to k - 1 that a reader of ``disclosed`` can work out from the flow alone
    (:func:`flow_closure`), by their ids. Empty when nothing comes back. A box is not one of them: the rows of each
    box shown are given.
    """
    closure = flow_closure(flow, disclosed, analyses)
    return [name for name in closure.comes_back(disclosed.k) if not name.startswith("box:")]


def _shown_analyses(
    analyses: Sequence[AnalysisCount], disclosed: DisclosedFlow
) -> list[tuple[AnalysisCount, DisclosedAnalysis]]:
    """
    Each analysis the disclosed flow shows, with its count. The shown ones are in the order given, and each is
    recognised by what is shown of it (its label, its rows, the rows it left out), so that a flow that was not made
    by the rule is read too. ValueError when an analysis is shown that no count explains: nothing can be said of
    what it gives back.
    """
    pairs, shown = [], list(disclosed.analyses)
    for count in analyses:
        if not shown:
            break
        first = shown[0]
        if (first.label, first.n, first.left_out) == (
            count.label,
            _analysed(count),
            int(count.n_left_out),
        ):
            pairs.append((count, shown.pop(0)))
    if shown:
        raise ValueError("the flow shows an analysis that no count explains")
    return pairs


# ----------------------------------------------------------------------------- results


@dataclass(frozen=True)
class DisclosedRecord:
    """
    What a shared package shows of a result: ``record`` holds the parts and the numbers that can be shown (None when
    the whole result is withheld), ``withheld`` says what is not, by count and by variable, never by a level's name.
    """

    kind: str
    pooled: bool
    record: ResultRecord | None
    withheld: tuple[str, ...] = ()


# What is written of a model as a whole: its totals, and, apart, what its fit says of all its rows together. A
# number of the fit is an equation in every cell of the model, a withheld one included: with its only term withheld,
# a model's log-likelihood, its pseudo-R2 or its AUC gives the cell back, and two models of one package give the
# withheld term's likelihood-ratio statistic. They travel only when no term is withheld. Whether the fit converged
# goes with them: beside a withheld term it says that the term had no maximum.
_MODEL_NUMBERS = {
    "cox": ("rows", "events", "ties", "df_resid"),
    "logistic": ("family", "rows", "events", "df_resid"),
    "count": ("family", "rows", "events", "df_resid"),
}
_FIT_NUMBERS = {
    "cox": ("converged", "concordance"),
    "logistic": ("converged", "llf", "aic", "pseudo_r2", "auc", "auc_ci_low", "auc_ci_high"),
    "count": ("converged", "llf", "aic", "dispersion"),
}
_POOLED_MODEL_NUMBERS = ("family", "rows_min", "rows_max", "events_min", "events_max")
_POOLED_FIT_NUMBERS = ("converged_in",)
_TERM_NUMBERS = (
    "log_hr",
    "coef",
    "se",
    "hr",
    "odds_ratio",
    "rate_ratio",
    "ci_low",
    "ci_high",
    "z",
    "p",
    "extreme",
)
_POOLED_TERM_NUMBERS = (
    "estimate",
    "se",
    "statistic",
    "df",
    "p",
    "ci_low",
    "ci_high",
    "ubar",
    "b",
    "t",
    "riv",
    "lambda",
    "fmi",
    "dfcom",
    "m",
    "scale",
    "extreme",
    "natural",
    "natural_low",
    "natural_high",
)  # fmt: skip  (the estimate of each imputed dataset stays: Rubin's quantities say what is needed)
_PH_NUMBERS = ("statistic", "p", "flagged")
_TEST_NUMBERS = ("method", "terms", "statistic", "df", "df1", "df2", "p", "riv", "m")
_POOLING_NUMBERS = ("m", "m_total", "partial")
# What each kind counts of a cell (a level, a value, a group) and of a model: its rows, those with the event and
# those without. Each must reach k for the cell, or the model, to be shown; a count of none is no exception (a
# level where nobody, or everybody, had the event says so of each of its participants), and a record that lacks one
# of them is not shown. A pooled record names its counts as the smallest over the imputed datasets ("rows_min").
_COUNTED = {
    "cox": ("rows", "events", "censored"),
    "logistic": ("rows", "events", "non_events"),
    "count": ("rows", "rows_with_events", "rows_without_events"),
    "survival": ("rows", "events", "censored"),
}
_TOTAL_NAMES = (
    "rows",
    "events",
    "non_events",
    "censored",
    "rows_with_events",
    "rows_without_events",
    "df_resid",
)
_TOTALS = frozenset(_TOTAL_NAMES) | {f"{name}_min" for name in _TOTAL_NAMES}
SPARSE_BELOW = 10  # events per parameter below which the application itself warns that a model's ratios are unstable
_SIDES = "rows, rows with the event or rows without it"
_ADVICE = (
    "join a small level or group to another, or take its rows out of the analysis population by a criterion, and "
    "run the analysis again"
)


def _count(support: Mapping[str, Any], name: str) -> Any:
    """A count behind a part: under its name, or under its pooled name (the smallest over the imputed datasets)."""
    return support[name] if name in support else support.get(f"{name}_min")


def _kept(part: Part, names: Sequence[str], *, totals: bool = False) -> Part:
    """
    The part with the numbers its kind's rule names, in the record's order. The counts behind it travel only for
    the model as a whole (``totals``): beside a term they would be subtracted from the model's.
    """
    support = {name: n for name, n in part.support.items() if name in _TOTALS} if totals else {}
    return Part(part.id, tuple(n for n in part.numbers if n.name in names), support, part.labels)


def _passes(support: Mapping[str, Any], kind: str, policy: DisclosurePolicy) -> bool:
    """Whether a cell, or a model, holds at least k of each count its kind counts."""
    held = [_count(support, name) for name in _COUNTED[kind]]
    return all(isinstance(n, int) and n >= policy.k for n in held)


def _model_shown(part: Part, kind: str, policy: DisclosurePolicy) -> bool:
    residual = _count(part.support, "df_resid")
    return (
        _passes(part.support, kind, policy) and isinstance(residual, int) and residual >= policy.k
    )


def _together(parts: Sequence[Part], kind: str) -> dict[str, int]:
    """The counts a kind counts, of several cells or groups together (a count one lacks counts as none)."""
    return {
        name: sum(n for p in parts if isinstance(n := _count(p.support, name), int))
        for name in _COUNTED[kind]
    }


def _said(count: int, what: str, why: str, *, of: str = "", more: bool = False) -> str:
    """ "2 terms of site withheld: ..." """
    return (
        f"{count} {'more ' if more else ''}{what}{'s' if count != 1 else ''}{f' of {of}' if of else ''} "
        f"withheld: {why}"
    )


def _differ_by_few(
    whole: Mapping[str, Any], part: Mapping[str, Any], kind: str, policy: DisclosurePolicy
) -> bool:
    """
    Whether two sets of counts, shown side by side, give a count from 1 to k - 1 by subtraction: the rows, the rows
    with the event or the rows without it that the first holds and the second lacks (a pooled analysis and its
    complete cases differ by the rows that lack a value, and by their outcomes).
    """
    for name in _COUNTED[kind]:
        a, b = _count(whole, name), _count(part, name)
        if not isinstance(a, int) or not isinstance(b, int) or not policy.shows(a - b) or a < b:
            return True
    return False


def _withheld_whole(record: ResultRecord, *why: str) -> DisclosedRecord:
    return DisclosedRecord(record.kind, record.pooled, None, tuple(why))


# why a term is withheld
_REFERENCE, _CELL, _COVER = "reference", "cell", "cover"


def _extreme(part: Part) -> bool:
    """A term without a maximum of the likelihood: the application itself says that such a ratio cannot be read."""
    return any(n.name == "extreme" and n.value is True for n in part.numbers)


def _withheld_terms(
    record: ResultRecord, policy: DisclosurePolicy
) -> tuple[dict[str, str], list[str]]:
    """
    The terms of a model that are not written, each with the reason, and the model's variables in its order.

    A term that compares two cells (a factor's level or the upper value of a covariate of two values with the
    reference; the rows off the commonest value of a numeric covariate with the rows on it) is written only when
    both cells hold k of each count the kind counts. A term without a maximum is not written. And the cells
    withheld of a variable must hold k of each count together, or the smallest cell shown goes with them, and the
    next, until they do or none is left: in a model of categories alone the coefficient of a shown term and its
    standard error are its two cells' counts in another form, so the model's totals, less the cells shown, give
    the withheld ones together.
    """
    kind = record.kind
    parts = {p.id: p for p in record.parts}
    terms = [p for p in record.parts if p.id.startswith("term:")]
    variables = list(dict.fromkeys(p.labels.get("variable", "") for p in terms))
    hidden: dict[str, str] = {}
    for variable in variables:
        own = [p for p in terms if p.labels.get("variable", "") == variable]
        reference = parts.get(f"reference:{variable}")
        if reference is None:
            # no cell behind it (a numeric covariate that no value dominates); a level without its reference's
            # counts cannot be judged, and is not shown
            hidden.update({p.id: _CELL for p in own if "level" in p.labels or _extreme(p)})
            continue
        if not _passes(reference.support, kind, policy):
            hidden.update({p.id: _REFERENCE for p in own})
            continue
        hidden.update(
            {p.id: _CELL for p in own if _extreme(p) or not _passes(p.support, kind, policy)}
        )
        while True:
            gone = [p for p in own if p.id in hidden]
            left = [p for p in own if p.id not in hidden]
            if not gone or not left or all(n >= policy.k for n in _together(gone, kind).values()):
                break
            smallest = min(left, key=lambda p: (_count(p.support, "rows") or 0, p.id))
            hidden[smallest.id] = _COVER
    return hidden, variables


def _term_sentences(
    record: ResultRecord,
    hidden: Mapping[str, str],
    variables: Sequence[str],
    policy: DisclosurePolicy,
) -> list[str]:
    """What is said of the withheld terms: counted by variable and by reason, never named by a level."""
    reasons = {
        _CELL: f"fewer than {policy.k} {_SIDES} in a cell compared, or an estimate without a maximum",
        _REFERENCE: f"its reference cell holds fewer than {policy.k} {_SIDES}",
        _COVER: f"the model's counts, less the cells shown, would give back fewer than {policy.k} {_SIDES}",
    }
    said = []
    for variable in variables:
        own = [
            hidden[p.id]
            for p in record.parts
            if p.id in hidden and p.labels.get("variable", "") == variable
        ]
        for reason in (_REFERENCE, _CELL, _COVER):
            if count := own.count(reason):
                said.append(
                    _said(count, "term", reasons[reason], of=variable, more=reason == _COVER)
                )
    return said


def _model_facts(model: Part, kind: str, pooled: bool, parameters: int) -> list[Number]:
    """
    What a reader needs beside a model's ratios, as numbers (no warning's wording travels): its parameters apart
    from the intercept, and for a logistic or a Cox model the events per parameter (the rarer side of a logistic
    outcome) and whether that is below 10, where the application warns that the ratios may be unstable. From the
    totals that are written; a pooled model's are the smallest over the imputed datasets, and named so.
    """
    facts = [Number("parameters", int(parameters), EXACT)]
    events = _count(model.support, "events")
    other = _count(model.support, "non_events") if kind == "logistic" else events
    if (
        kind == "count"
        or not parameters
        or not isinstance(events, int)
        or not isinstance(other, int)
    ):
        return facts
    per = min(events, other) / parameters
    return [
        *facts,
        Number(f"events_per_parameter{'_min' if pooled else ''}", per, DETERMINISTIC),
        Number("sparse", bool(per < SPARSE_BELOW), EXACT),
    ]


def _disclose_model(
    record: ResultRecord, policy: DisclosurePolicy, spec: AnalysisSpec
) -> DisclosedRecord:
    """A Cox, logistic or count model, observed or pooled (see the module's description)."""
    kind = record.kind
    parts = {p.id: p for p in record.parts}
    model = parts.get("model")
    if model is None or not _model_shown(model, kind, policy):
        return _withheld_whole(
            record,
            f"the model is withheld: fewer than {policy.k} {_SIDES}, or residual degrees of freedom",
        )
    hidden, variables = _withheld_terms(record, policy)
    written = [
        p.labels.get("variable", "")
        for p in record.parts
        if p.id.startswith("term:") and p.id not in hidden
    ]
    compare = spec.variables.get("compare")
    if not written:
        return _withheld_whole(
            record, f"the model is withheld: none of its terms can be shown ({_ADVICE})"
        )
    if compare and compare not in written:
        # the coefficients of the adjustment are not effects on their own, and with a term withheld what is left
        # of the fit carries it
        return _withheld_whole(
            record,
            f"the model is withheld: no term of its comparison can be shown, and what is left is its adjustment "
            f"({_ADVICE})",
        )
    # the comparison first: what is withheld of it is what a reader looks for
    ordered = sorted(variables, key=lambda v: v != compare)
    withheld = _term_sentences(record, hidden, ordered, policy)
    names = _POOLED_MODEL_NUMBERS if record.pooled else _MODEL_NUMBERS[kind]
    if hidden:
        withheld.append(
            "the numbers of the fit as a whole are withheld with them: each would give back what the terms "
            "withheld hold"
        )
    else:
        names = (*names, *(_POOLED_FIT_NUMBERS if record.pooled else _FIT_NUMBERS[kind]))
    term_names = _POOLED_TERM_NUMBERS if record.pooled else _TERM_NUMBERS
    shown: list[Part] = []
    for part in record.parts:
        prefix, _, rest = part.id.partition(":")
        if part.id == "model":
            kept = _kept(part, names, totals=True)
            facts = _model_facts(part, kind, record.pooled, len(written) + len(hidden))
            shown.append(Part(kept.id, (*kept.numbers, *facts), kept.support, kept.labels))
        elif part.id == "pooling":
            shown.append(_kept(part, _POOLING_NUMBERS))
        elif prefix == "reference":
            # a reference with no term left to compare with it says nothing, and its label is a value of the data
            if part.labels.get("variable", "") in written:
                shown.append(_kept(part, ()))
        elif prefix == "term":
            if part.id not in hidden:
                shown.append(_kept(part, term_names))
        elif prefix == "ph":
            if f"term:{rest}" in parts and f"term:{rest}" not in hidden:
                shown.append(_kept(part, _PH_NUMBERS))
        elif prefix == "test":
            tested = next((n.value for n in part.numbers if n.name == "terms"), None) or []
            if any(term in hidden for term in tested):
                continue  # a test of a factor whose level is withheld would speak for that level
            shown.append(_kept(part, _TEST_NUMBERS))
    complete = None
    if record.complete_case is not None:
        inner = _disclose_model(record.complete_case, policy, spec)
        observed = next((p for p in record.complete_case.parts if p.id == "model"), None)
        if (
            inner.record is None
            or observed is None
            or not policy.shows(_left_out(record.complete_case))
            or _differ_by_few(model.support, observed.support, kind, policy)
        ):
            # the two analyses differ by the rows that lack a value and by their outcomes: a few of either, and
            # the pair is a subtraction
            withheld.append("the complete-case analysis is withheld")
        else:
            complete = inner.record
            withheld += [f"complete-case analysis: {w}" for w in inner.withheld]
    return DisclosedRecord(
        record.kind,
        record.pooled,
        ResultRecord(record.kind, record.pooled, tuple(shown), (), complete, record.record_version),
        tuple(withheld),
    )


# --- the times of a curve

STEPS_OF_FOLLOW_UP = 100  # a step is at least this part of the follow-up shown
UNITS_A_STEP = (
    5  # and at least this many of the data's own unit (the smallest gap between two recorded times)
)


def _clean(value: float) -> float:
    return float(f"{value:.12g}")


def step_at_least(value: float) -> float:
    """The smallest of 1, 2, 5 x 10^n that is at least ``value`` (above 0)."""
    exponent = math.floor(math.log10(value))
    for mantissa in (1, 2, 5, 10):
        step = mantissa * 10.0**exponent
        if step >= value * (1 - 1e-12):
            return _clean(step)
    raise AssertionError("no step")  # pragma: no cover


def round_up(value: float, step: float) -> float:
    """``value`` rounded up to a multiple of ``step``, and never below ``value``: a time one last bit above a
    multiple goes to the next one."""
    shown = _clean(math.ceil(value / step - 1e-9) * step)
    return shown if shown >= value else _clean(shown + step)


def round_down(value: float, step: float) -> float:
    """``value`` rounded down to a multiple of ``step``, and never above ``value``."""
    shown = _clean(math.floor(value / step + 1e-9) * step)
    return shown if shown <= value else _clean(shown - step)


@dataclass(frozen=True)
class _Timeline:
    """
    A group's curve as its record holds it, for the rule of its times. ``times`` are its event times (a pooled
    curve's: every time a dataset has an event) and ``so_far`` the events up to each, in each imputed dataset (one
    list for an observed curve). ``follow`` holds every time a follow-up ends and ``followed`` the rows whose
    follow-up ends then or later (the fewest over the datasets).
    """

    times: tuple[float, ...]
    so_far: tuple[tuple[int, ...], ...]
    follow: tuple[float, ...]
    followed: tuple[int, ...]

    def last_step(self, time: float, step: float) -> int:
        """The place of the last event time at or before ``time`` (-1: none)."""
        return bisect.bisect_right(self.times, time + 1e-9 * step) - 1

    def events_to(self, place: int) -> tuple[int, ...]:
        """The events up to a place, in each dataset."""
        return tuple(0 if place < 0 else one[place] for one in self.so_far)

    def since(self, start: int, end: int) -> int:
        """The events after one place up to another: the fewest over the datasets of each one's own count (the
        difference of two smallest counts can be more than any dataset holds)."""
        return min(b - a for a, b in zip(self.events_to(start), self.events_to(end), strict=True))

    def after(self, place: int) -> tuple[int, ...]:
        """The events after a place, in each dataset."""
        return tuple(
            one[-1] - done for one, done in zip(self.so_far, self.events_to(place), strict=True)
        )

    def alive(self, time: float, step: float) -> int:
        """The rows still followed and without the event at the end of ``time``: those whose follow-up ends then or
        later, less the events at that very time (never more than the truth for a pooled curve)."""
        place = bisect.bisect_left(self.follow, time - 1e-9 * step)
        rows = self.followed[place] if place < len(self.follow) else 0
        at = self.last_step(time, step)
        if at >= 0 and abs(self.times[at] - time) <= 1e-9 * step:
            rows -= max(
                b - a for a, b in zip(self.events_to(at - 1), self.events_to(at), strict=True)
            )
        return rows

    def horizon(self, k: int) -> float | None:
        """The latest time at which k rows are still followed (None: never)."""
        held = [time for time, rows in zip(self.follow, self.followed, strict=True) if rows >= k]
        return held[-1] if held else None


def _values(part: Part) -> dict[str, Any]:
    return {n.name: n.value for n in part.numbers}


def _timeline(part: Part, pooled: bool) -> _Timeline | None:
    """A curve's record read for the rule; None when it lacks what the rule reads (a record made before its events
    and its follow-up times were kept: nothing of the curve can be given)."""
    held = _values(part)
    times = held.get("times")
    if pooled:
        each, followed = held.get("events_so_far_each"), held.get("followed_min")
    else:
        events, followed = held.get("events_at"), held.get("followed")
        each = None if events is None else [list(accumulate(int(e) for e in events))]
    follow = held.get("follow_times")
    if times is None or not each or not follow or followed is None or len(follow) != len(followed):
        return None
    if any(len(one) != len(times) for one in each):
        return None
    return _Timeline(
        tuple(float(t) for t in times),
        tuple(tuple(int(v) for v in one) for one in each),
        tuple(float(t) for t in follow),
        tuple(int(v) for v in followed),
    )


def time_step(curves: Sequence[_Timeline], k: int) -> float | None:
    """
    The step every time of a result is a multiple of (its groups, its complete cases): the smallest of 1, 2, 5 x
    10^n that is at least a hundredth of the follow-up shown (the latest time at which k rows of a group are still
    followed) and at least five times the data's own unit (the smallest gap between two recorded times). The second
    is what makes a time given coarser than the record: no time of a curve is a participant's. None when no group
    has k rows followed at a time above 0.
    """
    recorded = sorted({time for curve in curves for time in curve.follow})
    gaps = [b - a for a, b in zip(recorded, recorded[1:], strict=False)]
    horizons = [h for curve in curves if (h := curve.horizon(k)) is not None]
    least = max(
        (max(horizons) / STEPS_OF_FOLLOW_UP if horizons else 0.0),
        (UNITS_A_STEP * min(gaps) if gaps else 0.0),
    )
    return step_at_least(least) if horizons and least > 0 else None


def _points(curve: _Timeline, step: float, k: int) -> list[tuple[float, int]]:
    """
    The times a curve is given at, each with the place of the last event time it covers. The last one first: the
    latest event time, rounded up to a multiple of the step, at which k rows are still followed and without the
    event, with at least k events before it and none or at least k after it, in every imputed dataset. Before it, a
    time once k events lie since the one before (rounded up likewise), as long as k events lie between it and the
    last. So at least k events lie between two times, and none or at least k after the last: no count from 1 to
    k - 1 can be read between two values of the curve.
    """
    final = None
    for place in range(len(curve.times) - 1, -1, -1):
        shown_at = round_up(curve.times[place], step)
        if curve.alive(shown_at, step) < k:
            continue
        covered = curve.last_step(shown_at, step)
        if min(curve.events_to(covered)) >= k and all(
            n == 0 or n >= k for n in curve.after(covered)
        ):
            final = (shown_at, covered)
            break
    if final is None:
        return []
    points: list[tuple[float, int]] = []
    last = -1
    for place in range(len(curve.times)):
        if place <= last or curve.since(last, place) < k:
            continue
        shown_at = round_up(curve.times[place], step)
        covered = curve.last_step(shown_at, step)
        if shown_at >= final[0] or curve.since(covered, final[1]) < k:
            break  # the last time takes what is left: at least k events lie between the two
        points.append((shown_at, covered))
        last = covered
    return [*points, final]


def _followed_until(curve: _Timeline, step: float, k: int) -> float | None:
    """The latest multiple of the step at which k rows are still followed and without the event (None: none)."""
    horizon = curve.horizon(k)
    if horizon is None:
        return None
    until = round_down(horizon, step)
    while until > 0 and curve.alive(until, step) < k:
        until = _clean(until - step)
    return until if curve.alive(until, step) >= k else None


def _curve(
    part: Part, timeline: _Timeline | None, policy: DisclosurePolicy, step: float | None
) -> Part | None:
    """
    A curve as a shared package gives it, or None when its group cannot be shown on its own: fewer than k rows,
    events or rows without the event, or a record without what the rule of its times reads.

    Its rows, its events and its rows without the event travel with it (the table under a curve), with the step of
    its times, until when k rows were followed (``followed_until``) and every how many events a time is given
    (``points_every``). Where k rows are followed beyond the last time with no event between, one more time is
    given at ``followed_until``, with the same value: the curve is flat to there. The median is rounded up to the
    step, its lower limit down and its upper limit up (outwards), each given when k rows are still followed at
    that time; otherwise it is later than ``followed_until`` (``later_than_followed_until`` names those), or not
    reached (no value, and not named there).
    """
    if timeline is None or step is None or not _passes(part.support, "survival", policy):
        return None
    k = policy.k
    held = _values(part)
    by_name = {n.name: n for n in part.numbers}
    places = _points(timeline, step, k)
    until = _followed_until(timeline, step, k)
    # k rows are followed beyond the last time with no event between: the curve is flat to there
    if (
        places
        and until is not None
        and until > places[-1][0]
        and timeline.last_step(until, step) == places[-1][1]
    ):
        places.append((until, places[-1][1]))
    later: list[str] = []

    def quantile(name: str, direction: Callable[[float, float], float]) -> float | None:
        value = held.get(name)
        if value is None:
            return None  # not reached
        shown_at = direction(float(value), step)
        if until is None or shown_at > until or timeline.alive(shown_at, step) < k:
            later.append(name)
            return None
        return shown_at

    quantiles = [
        Number(name, quantile(name, direction), EXACT)
        for name, direction in (
            ("median", round_up),
            ("median_lower", round_down),
            ("median_upper", round_up),
        )
        if name in by_name
    ]
    numbers = [
        n
        for n in part.numbers
        if n.name in ("rows", "events", "rows_min", "events_min", "confidence")
    ]
    numbers += [
        Number("points_every", k, EXACT),
        Number("time_step", step, EXACT),
        Number("followed_until", until, EXACT),
        *quantiles,
        Number("later_than_followed_until", later, EXACT),
        Number("times", [time for time, _ in places], EXACT),
    ]
    numbers += [
        Number(
            name, [held[name][j] for _, j in places], by_name[name].cls, scale=by_name[name].scale
        )
        for name in ("survival", "lower", "upper", "fmi")
        if held.get(name) is not None
    ]
    counted = {
        name: n
        for name, n in part.support.items()
        if name.removesuffix("_min") in _COUNTED["survival"]
    }
    return Part(part.id, tuple(numbers), counted, part.labels)


@dataclass(frozen=True)
class _Groups:
    """The curves of a survival record, each with what the rule of its times reads."""

    record: ResultRecord
    curves: dict[str, Part]
    timelines: dict[str, _Timeline | None]

    @classmethod
    def of(cls, record: ResultRecord) -> _Groups:
        curves = {p.id: p for p in record.parts if p.id.startswith("curve#")}
        return cls(record, curves, {i: _timeline(p, record.pooled) for i, p in curves.items()})

    def label(self, part_id: str) -> str:
        return self.curves[part_id].labels.get("group", part_id)


def _withheld_groups(
    groups: _Groups,
    built: Mapping[str, Part | None],
    policy: DisclosurePolicy,
    also: Collection[str],
) -> tuple[list[str], int]:
    """
    The groups of a survival record that are not shown, and how many of them cannot be shown on their own. The
    number the analysis used, and any model on the same rows, give the withheld groups' rows, events and rows
    without the event together: together they hold at least k of each, or the smallest group shown goes with
    them, and the next, until they do or none is left. ``also`` names groups that the analysis beside this one (a
    pooled result's complete cases, or the reverse) withholds: one cover serves both.
    """
    small = [i for i in groups.curves if built[i] is None]
    hidden = [
        *small,
        *(i for i in groups.curves if built[i] is not None and groups.label(i) in also),
    ]
    shown = [i for i in groups.curves if i not in hidden]
    while (
        hidden
        and shown
        and any(
            n < policy.k for n in _together([groups.curves[i] for i in hidden], "survival").values()
        )
    ):
        smallest = min(shown, key=lambda i: (_count(groups.curves[i].support, "rows") or 0, i))
        shown.remove(smallest)
        hidden.append(smallest)
    return hidden, len(small)


def _shown_curves(
    groups: _Groups,
    built: Mapping[str, Part | None],
    hidden: Sequence[str],
    small: int,
    policy: DisclosurePolicy,
    *,
    paired: bool = False,
) -> DisclosedRecord:
    """A survival record with the groups that are shown, without its complete cases (``paired``: it is a pooled
    analysis or its complete cases, which withhold the same groups)."""
    record = groups.record
    withheld = []
    if small:
        withheld.append(
            _said(small, "group", f"fewer than {policy.k} rows, events or rows without the event")
        )
    if len(hidden) > small:
        withheld.append(
            _said(
                len(hidden) - small,
                "group",
                f"the number analysed, less the groups shown, would give back fewer than {policy.k} rows, events "
                "or rows without the event"
                + (", here or in the analysis beside this one" if paired else ""),
                more=True,
            )
        )
    if hidden:
        withheld.append(f"to share the other groups together: {_ADVICE}")
    shown = [part for i in groups.curves if i not in hidden and (part := built[i]) is not None]
    if not shown:
        return DisclosedRecord(record.kind, record.pooled, None, tuple(withheld))
    parts = [_kept(p, _POOLING_NUMBERS) for p in record.parts if p.id == "pooling"]
    parts += shown
    test = next((p for p in record.parts if p.id == "test:log_rank"), None)
    if test is not None:
        if hidden:
            withheld.append("the log-rank test is withheld: it compares a group that is")
        else:
            parts.append(_kept(test, _TEST_NUMBERS))
    return DisclosedRecord(
        record.kind,
        record.pooled,
        ResultRecord(record.kind, record.pooled, tuple(parts), (), None, record.record_version),
        tuple(withheld),
    )


def _disclose_survival(
    record: ResultRecord, policy: DisclosurePolicy, spec: AnalysisSpec
) -> DisclosedRecord:
    """Kaplan-Meier curves and the log-rank test, observed or pooled (see the module's description)."""
    main = _Groups.of(record)
    beside = None if record.complete_case is None else _Groups.of(record.complete_case)
    both = [main] if beside is None else [main, beside]
    # one step for every time of the result, and one set of groups withheld
    step = time_step([t for g in both for t in g.timelines.values() if t is not None], policy.k)
    built = [
        {i: _curve(p, g.timelines[i], policy, step) for i, p in g.curves.items()} for g in both
    ]
    # one cover for both: the result's own first, which its complete cases then take (two covers chosen apart
    # could be the two arms of the comparison); what the complete cases need beyond it goes back to the result
    also: set[str] = set()
    while True:
        found = []
        labels = set(also)
        for groups, parts in zip(both, built, strict=True):
            found.append(_withheld_groups(groups, parts, policy, labels))
            labels |= {groups.label(i) for i in found[-1][0]}
        if labels <= also:
            break
        also = labels
    disclosed = _shown_curves(main, built[0], *found[0], policy, paired=beside is not None)
    if beside is None or disclosed.record is None:
        return disclosed
    inner = _shown_curves(beside, built[1], *found[1], policy, paired=True)
    withheld = list(disclosed.withheld)
    observed = {p.labels.get("group"): p for p in beside.curves.values()}
    shown = [] if inner.record is None else [p.labels.get("group") for p in inner.record.parts]
    pairs = [
        (main.curves[p.id], observed[p.labels.get("group")])
        for p in disclosed.record.parts
        if p.id in main.curves and p.labels.get("group") in shown
    ]
    every = [p for p in main.curves.values() if p.labels.get("group") in observed]
    complete = None
    if (
        inner.record is None
        or not policy.shows(_left_out(beside.record))
        or any(_differ_by_few(p.support, o.support, "survival", policy) for p, o in pairs)
        or _differ_by_few(
            _together(every, "survival"),
            _together(list(observed.values()), "survival"),
            "survival",
            policy,
        )
    ):
        # the two analyses differ, group by group, by the rows that lack a value and by their outcomes
        withheld.append("the complete-case analysis is withheld")
    else:
        complete = inner.record
        withheld += [f"complete-case analysis: {w}" for w in inner.withheld]
    return DisclosedRecord(
        record.kind,
        record.pooled,
        ResultRecord(
            record.kind,
            record.pooled,
            disclosed.record.parts,
            (),
            complete,
            record.record_version,
        ),
        tuple(withheld),
    )


# the rule of each kind of result (the kinds' ids are biostats.plan.KINDS'): a kind without one is withheld
RULES: dict[str, Callable[[ResultRecord, DisclosurePolicy, AnalysisSpec], DisclosedRecord]] = {
    "survival": _disclose_survival,
    "cox": _disclose_model,
    "logistic": _disclose_model,
    "count": _disclose_model,
}
_LEFT_OUT = (
    "rows_skipped",
    "rows_left_out",
)  # the rows a record says it left out, a missing value apart


def _left_out(record: ResultRecord) -> int:
    """The rows a result left out for another reason than a missing value, as its record counts them (a group too
    small to draw, a follow-up time below zero)."""
    return sum(int(n.value or 0) for p in record.parts for n in p.numbers if n.name in _LEFT_OUT)


def _rows_analysed(record: ResultRecord) -> int | None:
    """The rows an observed result read, as its record counts them (None when it does not say: a pooled result's
    datasets can hold other rows than the flow counts)."""
    if record.pooled:
        return None
    if record.kind == "survival":
        rows = [p.support.get("rows") for p in record.parts if p.id.startswith("curve#")]
        return sum(rows) if rows and all(isinstance(n, int) for n in rows) else None
    model = next((p for p in record.parts if p.id == "model"), None)
    rows = None if model is None else model.support.get("rows")
    return rows if isinstance(rows, int) else None


def disclose_record(
    record: ResultRecord,
    policy: DisclosurePolicy,
    *,
    spec: AnalysisSpec,
    analysis: DisclosedAnalysis | None,
) -> DisclosedRecord:
    """
    What a shared package shows of ``record`` under ``policy``, by the rule of its kind. A kind without a rule is
    withheld whole. No warning's wording is ever copied (a warning can name a level), and no number marked as one
    participant's is written as it is, in the record or in the complete-case record beside it.

    ``spec`` is the result's own (ValueError when it is another kind's): a model's comparison is read in it.
    ``analysis`` is the result's analysis as the package's flow shows it (``DisclosedFlow.analyses``), or None when
    the flow withholds it or does not list it: the result is then withheld, since its own counts beside the flow's
    would give back what the flow withheld. It has no default. Withheld too: a result whose record counts 1 to
    k - 1 rows left out for another reason than a missing value (whatever the flow shows), and one whose rows, or
    whose rows left out, are not those of the analysis it is given with.
    """
    if spec.kind != record.kind or bool(spec.pooled) != bool(record.pooled):
        raise ValueError("a result is disclosed with its own spec")
    rule = RULES.get(record.kind)
    if rule is None:
        return _withheld_whole(record, "this kind of result has no rule for sharing yet")
    if analysis is None:
        return _withheld_whole(
            record, "the result is withheld: the participant flow does not show its analysis"
        )
    left_out = _left_out(record)
    if not policy.shows(left_out):
        return _withheld_whole(
            record,
            f"the result is withheld: fewer than {policy.k} rows were left out for another reason than a missing "
            "value",
        )
    rows = _rows_analysed(record)
    if left_out != analysis.left_out or (rows is not None and rows != analysis.n):
        return _withheld_whole(
            record,
            "the result is withheld: its rows are not those the participant flow shows for its analysis",
        )
    disclosed = rule(record, policy, spec)
    for written in (disclosed.record, getattr(disclosed.record, "complete_case", None)):
        if written is None:
            continue
        fine = [n.name for p in written.parts for n in p.numbers if n.fine]
        if fine:  # a rule that lets one through is a defect: nothing is written
            raise ValueError(
                f"the rule of {record.kind} would share {len(fine)} number(s) of one participant"
            )
    return disclosed


def disclose_spec(
    spec: AnalysisSpec, disclosed: DisclosedRecord, *, analysis: DisclosedAnalysis | None
) -> dict[str, Any] | None:
    """
    What a shared package says of a result's spec (``biostats.plan.AnalysisSpec``), given what it shows of the
    result: the request as it was asked (kind, variables, options, draws: its request id can be computed again
    from them), the libraries, and of what it was computed on the view that names no row
    (``ComputedOn.shareable``). The levels the run recorded are values of the data: each factor's levels are
    replaced by those of its shown terms, in their order, with the number withheld; the groups of the curves
    likewise. The level counted as the event travels with a shown model only. None when the result is withheld.

    ``analysis`` is the analysis as the package's flow shows it (``DisclosedFlow.analyses``), or None when the flow
    withholds it or does not list it. The rows of the population and the rows analysed are written only when they
    are the flow's own for that analysis, and as the flow gives them (a pooled analysis: the population less the
    rows it left out): any other number beside the flow's would be a subtraction.
    """
    if disclosed.record is None:
        return None
    shown = spec.request()
    decided: dict[str, Any] = {}
    for key in ("event_values", "without_term", "event_level"):
        if key in spec.decided:
            decided[key] = spec.decided[key]
    parts = disclosed.record.parts
    if "levels" in spec.decided:
        decided["levels"] = {}
        for factor, levels in spec.decided["levels"].items():
            kept = [p.labels["level"] for p in parts if p.id == f"reference:{factor}"]
            kept += [p.labels["level"] for p in parts if p.id.startswith(f"term:{factor}#")]
            decided["levels"][factor] = {"shown": kept, "withheld": len(levels) - len(kept)}
    if "groups" in spec.decided:
        kept = [p.labels["group"] for p in parts if p.id.startswith("curve#")]
        decided["groups"] = {"shown": kept, "withheld": len(spec.decided["groups"]) - len(kept)}
    if "outcome_levels" in spec.decided:
        # both hold at least k rows: the model is shown
        decided["outcome_levels"] = list(spec.decided["outcome_levels"])
    data = None if spec.data is None else spec.data.shareable()
    if data is not None:
        own = analysis is not None and (
            data.get("n_population") == analysis.n + analysis.missing + analysis.left_out
            and (spec.pooled or data.get("n_analysed") == analysis.n)
        )
        if own and analysis is not None:
            data["n_analysed"] = analysis.n
        else:
            data = {
                key: value
                for key, value in data.items()
                if key not in ("n_population", "n_analysed")
            }
    return {
        **shown,
        "request_id": spec.request_id,
        "decided": decided,
        "backend": dict(spec.backend),
        "data": data,
    }


def reading_notes(policy: DisclosurePolicy) -> tuple[str, ...]:
    """
    What a reader of a shared package's results must know to read them, in fixed words: the package's own text
    carries these sentences with its k, since no caution of a result travels as wording.
    """
    k = policy.k
    return (
        f"A count of participants is given exactly, or as fewer than {k}.",
        f"A term of a model is written only when both of the cells it compares hold {k} rows, {k} with the event "
        f"and {k} without. Over studies, the terms written for small levels are selected by their outcome, and the "
        "withheld ones are not missing at random: join small levels on subject-matter grounds before the "
        "analysis.",
        "A term that is written is the full model's, adjusted for every variable and every level, those withheld "
        "included. When a term is withheld, the numbers of the fit as a whole are withheld with it.",
        f"A survival curve is given at chosen times, multiples of its time step, with at least {k} events between "
        "two of them: its value and its limits at each are the full curve's. Between two times given, the curve "
        "lies between the two values. It is given until the time that followed_until names, when it stays flat "
        "to there.",
        "A median survival time is rounded up to the time step, its lower limit down and its upper limit up. One "
        "that is named in later_than_followed_until is later than that time; one without a value that is not "
        "named there was not reached.",
        f"Events per parameter below {SPARSE_BELOW} (sparse) mean that a model's ratios may be unstable.",
    )


# ----------------------------------------------------------------------------- rare values in a package's texts

MIN_LENGTH = 3  # a value of one or two characters ("F", "II") is in every text: looked for after its column's name only
CODE_LENGTH = (
    4  # a number of this many digits, in a text column or in an identifier, is a code: looked for
)
NEARLY_UNIQUE = (
    0.9  # distinct values per row above which a column reads as an identifier or as free text
)
ONE_HOT_LEVELS = (
    20  # a number column of more values than this is not one that gets a column per value
)
MOST_LISTED = 1000  # the values of an ``isin`` list that are counted one by one
# the kinds of hit: stored ids (an export log names the hits it overrides)
VALUE = "value"  # a rare value stands in the text
LITERAL = "literal"  # the text compares the column with a value that few rows hold
UNCHECKED = (
    "not_checked"  # the text holds a test that could not be run safely: nothing can be said of it
)
_WORD = re.compile(r"[^\W_]+")  # letters and digits; anything else separates, an underscore too
_WORDS = frozenset(
    {"true", "false", "none", "null"}
)  # in every text, and no participant's value on their own
_EQUALITY = frozenset({"eq", "ne", "isin"})
_PATTERN_TESTS = frozenset(
    {"contains", "match", "fullmatch"}
)  # their first argument is a regular expression
_PLAIN_TESTS = frozenset({"startswith", "endswith"})
_SAME_VALUES = frozenset(
    {"astype", "strip", "lower", "upper", "casefold", "fillna"}
)  # a column's values, still
_PATTERN_SIGNS = frozenset(
    ".^$*+?{}[]\\|()"
)  # a text without any of them is searched as it is written
# what the engine without backtracking (RE2, behind text stored by Arrow) does not take, and what makes Python's own
# engine try every split of a text: a look-around, a reference to a group
_BACKTRACKS = re.compile(r"\(\?[=!<]|\\[1-9]|\(\?P=")
# an ordering sign: a column's name, the sign and a number are a bound, not the name a column per value would have
_ORDERING = re.compile(r"[<>]=?")
_BOUND_WORD = (
    "zzboundzz"  # what stands for the sign in the text's normal form: no name and no value holds it
)


@dataclass(frozen=True)
class RareHit:
    """
    A value of one participant, or of a few, in a text of a package: where it is (the file, then the names that
    lead to the text) and the column that holds it. The value itself is never written, here or in the place; a
    column whose own name is, or holds, such a value is written by its position ("column 12").
    """

    place: str
    column: str
    kind: str = VALUE


def normal(text: str) -> str:
    """
    The form a value and a text are compared in: Unicode's compatibility form, without the case of letters and
    without accents, each run of anything else than letters and digits read as one space. "Wiltz-Canary",
    "WILTZ CANARY" (written with an ordinary or a no-break space) and the end of "site_Wiltz-Canary" are the same
    two words, whether an accent is one character or two.
    """
    folded = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", str(text)).casefold())
    return " ".join(_WORD.findall("".join(c for c in folded if not unicodedata.combining(c))))


def _reads_as_text(series: pd.Series) -> bool:
    kind = series.dtype
    return not (
        pd.api.types.is_bool_dtype(kind)
        or pd.api.types.is_numeric_dtype(kind)
        or pd.api.types.is_datetime64_any_dtype(kind)
        or pd.api.types.is_timedelta64_dtype(kind)
    )


def _whole_numbers(series: pd.Series) -> bool:
    kind = series.dtype
    if pd.api.types.is_bool_dtype(kind) or not pd.api.types.is_numeric_dtype(kind):
        return False
    values = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    return len(values) > 0 and bool((values == values.round()).all())


def identifying_columns(df: pd.DataFrame) -> list[str]:
    """
    The columns that read as an identifier or as a free text: texts or whole numbers with nearly one value per row.
    A first answer only, for the person who shares to correct: a laboratory value in whole numbers reads so in a
    small cohort, and an identifier repeated at each visit does not.
    """
    found = []
    for column in dict.fromkeys(df.columns):
        series = df[column]
        if isinstance(series, pd.DataFrame):
            continue
        held = series.dropna()
        if not len(held) or not (_reads_as_text(held) or _whole_numbers(held)):
            continue
        if held.astype(str).str.strip().nunique() / len(held) > NEARLY_UNIQUE:
            found.append(str(column))
    return found


def _forms(series: pd.Series) -> dict[str, int]:
    """The rows that hold each value, the values in their normal form: "Luxembourg" and "LUXEMBOURG" are one."""
    counts: dict[str, int] = {}
    for value, n in series.dropna().astype(str).value_counts().items():
        form = normal(value)
        if form:
            counts[form] = counts.get(form, 0) + int(n)
    return counts


def _days(series: pd.Series, k: int) -> set[str]:
    """The days that 1 to k - 1 rows hold, as year-month-day, day-month-year and month-day-year, with and without
    a zero before a day or a month of one digit."""
    forms: set[str] = set()
    days = series.dropna().dt.normalize().value_counts()
    for day, n in days.items():
        if 0 < n < k:
            y, m, d = str(day.year), day.month, day.day
            for month, date in ((f"{m:02d}", f"{d:02d}"), (str(m), str(d))):
                forms |= {f"{y} {month} {date}", f"{date} {month} {y}", f"{month} {date} {y}"}
    return forms


def rare_values(
    df: pd.DataFrame,
    policy: DisclosurePolicy,
    columns: Sequence[str] | None = None,
    *,
    identifying: Collection[str] | None = None,
    shown: Collection[str] | None = None,
) -> dict[str, frozenset[str]]:
    """
    For each column, what is looked for in a package's texts, in its normal form (:func:`normal`):

    - a text or category column: the values held by 1 to k - 1 rows, counted without case, accents or punctuation
      (a spelling in capitals held by one row is not a rare value of a column that holds the word 44 times);
    - an identifier held as whole numbers (``identifying``; None: :func:`identifying_columns`): its values held by
      1 to k - 1 rows, an identifier repeated at each visit included;
    - a date column: the days held by 1 to k - 1 rows, in three orders.

    Left out: a value of fewer than 3 characters, a number of fewer than 4 digits, True, False, and a value that is
    the name of a column the package shows, or one of its words (``shown``; None: every column): it is in every
    table of the package. A level left out here is still looked for after its column's name
    (:func:`one_hot_names`).
    """
    marked = set(identifying_columns(df) if identifying is None else identifying)
    names = [str(c) for c in df.columns if shown is None or str(c) in shown]
    own_words = {normal(c) for c in names} | {w for c in names for w in normal(c).split()}
    found: dict[str, frozenset[str]] = {}
    for column in dict.fromkeys(columns if columns is not None else df.columns):
        series = df[column]
        if isinstance(series, pd.DataFrame):  # two columns of one name: not read
            continue
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            rare = _days(series, policy.k)
        elif _reads_as_text(series):
            rare = {form for form, n in _forms(series).items() if 0 < n < policy.k}
        elif str(column) in marked and _whole_numbers(series):
            whole = pd.to_numeric(series.dropna(), errors="coerce").dropna().astype("int64")
            rare = {str(v) for v, n in whole.value_counts().items() if 0 < n < policy.k}
        else:
            continue
        kept = set()
        for form in rare:
            digits = form.replace(" ", "")
            if digits.isdigit():
                if len(digits) < CODE_LENGTH:
                    continue
            elif len(form) < MIN_LENGTH or form in _WORDS:
                continue
            if form not in own_words:
                kept.add(form)
        if kept:
            found[str(column)] = frozenset(kept)
    return found


def one_hot_names(df: pd.DataFrame, policy: DisclosurePolicy) -> dict[str, frozenset[str]]:
    """
    For each column, the names a column per value would have for the values held by 1 to k - 1 rows, in their
    normal form: the column's name, then the value ("grade iv" for ``grade_IV``, "stage 3" for ``stage_3``).
    Whatever the value's length, numbers included: after its column's name, "IV" is a level. Text, category and
    True/False columns, and number columns of at most 20 values.
    """
    found: dict[str, frozenset[str]] = {}
    for column in dict.fromkeys(df.columns):
        series = df[column]
        if isinstance(series, pd.DataFrame) or pd.api.types.is_datetime64_any_dtype(series.dtype):
            continue
        name = normal(column)
        if not name:
            continue
        if _reads_as_text(series) or pd.api.types.is_bool_dtype(series.dtype):
            values = {form for form, n in _forms(series).items() if 0 < n < policy.k}
        else:
            counts = pd.to_numeric(series, errors="coerce").dropna().value_counts()
            if len(counts) > ONE_HOT_LEVELS:
                continue
            values = set()
            for value, n in counts.items():
                if 0 < n < policy.k:
                    values.add(normal(str(value)))  # "3.0" reads "3 0"
                    if float(value).is_integer():
                        values.add(str(int(value)))
        if values:
            found[str(column)] = frozenset(f"{name} {value}" for value in values)
    return found


def _parsed(text: str) -> ast.AST | None:
    """The text as an expression, or None when it is not one."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )  # a text that is no expression may hold "\\d": no warning for it
            return ast.parse(text.strip(), mode="eval")
    except SyntaxError, ValueError, RecursionError, MemoryError:
        return None


def _literal(node: ast.AST) -> bool:
    """Whether an expression holds a number or a text written in it (True and False are not values to name)."""
    return any(
        isinstance(n, ast.Constant)
        and isinstance(n.value, str | int | float)
        and not isinstance(n.value, bool)
        for n in ast.walk(node)
    )


def _column_of(node: ast.AST) -> str | None:
    """The column an expression reads as it is: ``df['site']``, or a bare ``site``, through ``.str`` and the
    methods that keep a column's values."""
    while True:
        if isinstance(node, ast.Attribute) and node.attr == "str":
            node = node.value
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _SAME_VALUES
        ):
            node = node.func.value
        else:
            break
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        return node.slice.value if isinstance(node.slice.value, str) else None
    if isinstance(node, ast.Name):
        return node.id
    return None


def _bound(node: ast.AST) -> tuple[str, float] | None:
    """The column and the number of a bound, ``x >= 97`` or ``97 <= x``, either way round and strict or not."""
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1):
        return None
    if not isinstance(node.ops[0], ast.GtE | ast.LtE | ast.Gt | ast.Lt):
        return None
    for one, other in ((node.left, node.comparators[0]), (node.comparators[0], node.left)):
        column = _column_of(one)
        if (
            column is not None
            and isinstance(other, ast.Constant)
            and isinstance(other.value, int | float)
        ):
            return column, float(other.value)
    return None


class _Subscripts(ast.NodeTransformer):
    """``df.age`` written ``df['age']``, the way the application's evaluator reads a column."""

    def __init__(self, columns: Collection[str], frame_names: Collection[str]) -> None:
        self.columns, self.frame_names = columns, frame_names

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in self.frame_names
            and node.attr in self.columns
        ):
            return ast.copy_location(
                ast.Subscript(value=node.value, slice=ast.Constant(node.attr), ctx=ast.Load()), node
            )
        return node


@dataclass(frozen=True)
class _Test:
    """
    A part of an expression that sets rows apart by a value written in it: its tree and the columns it reads.

    - ``on``: the column, when the test is one column against that value (an equality, a text test, a range);
    - ``equality``: ``==``, ``!=``, ``isin`` (and whether the value is the empty text, which names nobody);
    - ``holders``: which rows hold the value, those the test takes or those it leaves (``!=``);
    - ``among``: for ``isin``, what is tested and its list, so that each value of the list is counted on its own;
    - ``range``: a range of values (``between``, two bounds joined by ``&``): it names a value when the rows it
      takes all hold the same one;
    - ``pattern``: a text test's pattern as written (None when it is not a text written in the expression), and
      ``plain`` when the test searches it as it is written (``startswith``, or a pattern without a sign).
    """

    tree: ast.AST
    columns: tuple[str, ...]
    on: str | None = None
    equality: bool = False
    empty: bool = False
    holders: bool = True
    among: tuple[ast.AST, ast.AST] | None = None
    range: bool = False
    text_test: bool = False
    pattern: str | None = None
    plain: bool = False
    flags: bool = False  # a text test given flags: pandas then runs it with Python's own engine


def _tests(tree: ast.AST, columns: Collection[str], frame_names: Collection[str]) -> list[_Test]:
    """
    The comparisons and tests of an expression that name a value: ``==``, ``!=``, ``in``, ``isin``, the text tests
    (``str.contains``...), and a range (``between``, two bounds of one column joined by ``&``). A bound alone
    selects a side, not a value: it is not one of them.
    """

    def read(node: ast.AST) -> tuple[str, ...]:
        found: dict[str, None] = {}
        for n in ast.walk(node):
            name = None
            if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant):
                name = n.slice.value if isinstance(n.slice.value, str) else None
            elif isinstance(n, ast.Name) and n.id not in frame_names:
                name = n.id
            if name in columns:
                found[name] = None
        return tuple(found)

    out: list[_Test] = []
    for node in ast.walk(tree):
        test: _Test | None = None
        if isinstance(node, ast.Compare):
            if not any(isinstance(op, ast.Eq | ast.NotEq | ast.In | ast.NotIn) for op in node.ops):
                continue
            test = _Test(node, read(node))
            if len(node.ops) == 1:
                left, right = node.left, node.comparators[0]
                for one, other in ((left, right), (right, left)):
                    column = _column_of(one)
                    if column in columns and _literal(other) and not read(other):
                        test = _Test(
                            node,
                            read(node),
                            on=column,
                            equality=True,
                            empty=isinstance(other, ast.Constant) and other.value == "",
                            holders=not isinstance(node.ops[0], ast.NotEq | ast.NotIn),
                        )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            method = node.func.attr
            column = _column_of(node.func.value)
            on = column if column in columns else None
            first = node.args[0] if node.args else None
            written = (
                first.value
                if isinstance(first, ast.Constant) and isinstance(first.value, str)
                else None
            )
            if method in _EQUALITY:
                among = (node.func.value, first) if method == "isin" and first is not None else None
                test = _Test(
                    node,
                    read(node),
                    on=on,
                    equality=True,
                    empty=written == "",
                    holders=method != "ne",
                    among=among,
                )
            elif method == "between":
                test = _Test(node, read(node), on=on, range=True)
            elif method in _PATTERN_TESTS or method in _PLAIN_TESTS:
                as_written = any(
                    kw.arg == "regex"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is False
                    for kw in node.keywords
                )
                plain = method in _PLAIN_TESTS or as_written
                if written is not None and not plain:
                    plain = not (set(written) & _PATTERN_SIGNS)
                test = _Test(
                    node,
                    read(node),
                    on=on,
                    text_test=True,
                    pattern=written,
                    plain=plain,
                    flags=any(kw.arg == "flags" for kw in node.keywords),
                )
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
            low, high = _bound(node.left), _bound(node.right)
            if low is not None and high is not None and low[0] == high[0] and low[0] in columns:
                test = _Test(node, read(node), on=low[0], range=True)
        if test is not None and test.columns and _literal(node):
            out.append(test)
    return out


def _linear_time(pattern: str) -> bool:
    """
    Whether the engine that cannot backtrack (RE2, behind text stored by Arrow) takes the pattern. pandas runs any
    other pattern with Python's own engine, which can try every split of a text: minutes for a value of 30
    characters, days for one of 40. The engine itself is asked; without it, no pattern is run.
    """
    if _BACKTRACKS.search(pattern):
        return False
    try:
        arrow = importlib.import_module("pyarrow")
        importlib.import_module("pyarrow.compute").match_substring_regex(
            arrow.array([""], type=arrow.string()), pattern
        )
    except Exception:  # not installed, or a pattern it refuses
        return False
    return True


class RareFinder:
    """
    What a package says of one participant, or of a few, in its texts. Built on the datasets the package describes
    (its source, its analysis base, its population: a value is rare when one of them holds it in 1 to k - 1 rows),
    then given the texts that come from the user or from the data (names, labels, titles, descriptions, criteria,
    formulas), not the sentences the package's writer makes from fixed words:

    - a rare value that stands in a text, compared in a normal form (:func:`normal`): a criterion that names a
      site, a title, a code, an identifier, a date; and the name a column per value would have
      (:func:`one_hot_names`: ``grade_IV``, "stage 3"), which is not read across an ordering sign
      (``df['score'] >= 8`` is a bound, and its rows are the flow's);
    - a text that reads as an expression and names a value that fewer than k rows hold: each of its comparisons
      and tests is run by the application's own evaluator (``core.expressions.safe_eval``) on a copy of each
      dataset, and the rows that hold the value are counted (those an equality or a text test takes, those an
      inequality leaves, those of each value of an ``isin`` list, those of a range when they all hold one value),
      so that ``.str.lower().str.contains(...)``, ``.dt.year == 1931`` or ``.round() == 97`` are counted as pandas
      counts them. The rows a comparison sets apart without naming their value (the two participants who were not
      enrolled on the day it names) are a count of the flow, which has its own rule;
    - on an identifying column (``identifying``: the columns the person who shares marks; None: those
      :func:`identifying_columns` finds), an equality with any value but the empty text, and a text searched as
      it is written unless it takes k rows or more: the value may be a participant who is no longer in the data;
    - a test that cannot be run safely is a hit of its own kind, "not checked", since nothing can be said of it:
      a pattern that the engine without backtracking does not take or that is not run by it (a look-around, flags,
      a column not stored by Arrow), which Python's own engine could try for hours, and a form the evaluator
      refuses.

    A hit names its place and its column, never the value: the writer of a package stops on a hit, and its message
    is read by the person who shares. Not looked for: one word of a value of several words, a value joined to
    another word, and a number of fewer than 4 digits outside an expression.
    """

    def __init__(
        self,
        frames: Sequence[pd.DataFrame],
        policy: DisclosurePolicy,
        *,
        identifying: Collection[str] | None = None,
        shown: Collection[str] | None = None,
    ) -> None:
        self.policy = policy
        self._frames = tuple(frames)
        guessed = {column for frame in self._frames for column in identifying_columns(frame)}
        self.identifying = frozenset(guessed if identifying is None else identifying)
        rare: dict[str, set[str]] = {}
        names: dict[str, set[str]] = {}
        for frame in self._frames:
            found = rare_values(frame, policy, identifying=self.identifying, shown=shown)
            for column, values in found.items():
                rare.setdefault(column, set()).update(values)
            for column, values in one_hot_names(frame, policy).items():
                names.setdefault(column, set()).update(values)
        self.rare: dict[str, frozenset[str]] = {c: frozenset(v) for c, v in rare.items()}
        self.one_hot: dict[str, frozenset[str]] = {c: frozenset(v) for c, v in names.items()}
        self._values = self._indexed(self.rare)
        self._names = self._indexed(self.one_hot)
        self._position: dict[str, int] = {}
        for frame in self._frames:
            for place, column in enumerate(frame.columns, start=1):
                self._position.setdefault(str(column), place)
        self._cache: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {}
        self._labels: dict[str, str] = {}

    @staticmethod
    def _indexed(forms_of: Mapping[str, frozenset[str]]) -> dict[str, list[tuple[str, str]]]:
        """Each form under the word that fewest forms share: a text is read once, whatever the number of values,
        and 20,000 notes that all hold "patient" are not all tried on a text that holds it."""
        forms = [(column, form) for column, values in forms_of.items() for form in values]
        shared: dict[str, int] = {}
        for _, form in forms:
            for word in set(form.split()):
                shared[word] = shared.get(word, 0) + 1
        index: dict[str, list[tuple[str, str]]] = {}
        for column, form in forms:
            key = min(form.split(), key=lambda word: (shared[word], -len(word), word))
            index.setdefault(key, []).append((column, f" {form} "))
        return index

    @staticmethod
    def _held(index: Mapping[str, list[tuple[str, str]]], form: str) -> set[str]:
        padded = f" {form} "
        found: set[str] = set()
        for word in set(form.split()) & index.keys():
            for column, sought in index[word]:
                if column not in found and sought in padded:
                    found.add(column)
        return found

    def named(self, text: str) -> tuple[str, ...]:
        """The columns one of whose rare values stands in the text, as whole words in the normal form."""
        return self._read(text)[0]

    def compared(self, text: str) -> tuple[str, ...]:
        """The columns the text, read as an expression, compares with a value that fewer than k rows hold."""
        return self._read(text)[1]

    def unchecked(self, text: str) -> tuple[str, ...]:
        """The columns a test of the text reads, when that test could not be run safely."""
        return self._read(text)[2]

    def _read(self, text: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        if not text:
            return (), (), ()
        if text not in self._cache:
            tree = _parsed(text)
            found = self._held(self._values, normal(text))
            # the name a column per value would have: not across an ordering sign, which makes a bound of it
            found |= self._held(self._names, normal(_ORDERING.sub(f" {_BOUND_WORD} ", text)))
            few: dict[str, None] = {}
            unknown: dict[str, None] = {}
            if tree is not None:
                for place, frame in enumerate(self._frames):
                    columns = frozenset(str(c) for c in frame.columns)
                    names = DEFAULT_POLICY.frame_names
                    rewritten = ast.fix_missing_locations(
                        _Subscripts(columns, names).visit(copy.deepcopy(tree))
                    )
                    for test in _tests(rewritten, columns, names):
                        answer = self._judge(place, frame, test)
                        if answer is None:
                            unknown.update(dict.fromkeys(test.columns))
                        elif answer:
                            few.update(
                                dict.fromkeys(
                                    (test.on,) if test.on and test.equality else test.columns
                                )
                            )
            not_run = tuple(column for column in unknown if column not in few)
            self._cache[text] = (tuple(sorted(found)), tuple(few), not_run)
        return self._cache[text]

    def _judge(self, place: int, frame: pd.DataFrame, test: _Test) -> bool | None:
        """
        Whether the test names a value that fewer than k rows of the dataset hold; None when it could not be run
        safely. On an identifying column an equality is one whatever it selects (but with the empty text), and so
        is a text searched as it is written, unless it takes k rows or more.
        """
        marked = test.on is not None and test.on in self.identifying
        if test.equality and marked and not test.empty:
            return True
        if test.text_test and not test.plain:
            # a pattern: run only by the engine that cannot backtrack, on a text it can take
            series = frame[test.on] if test.on is not None and test.on in frame.columns else None
            arrow = series is not None and getattr(series.dtype, "storage", None) == "pyarrow"
            if test.pattern is None or test.flags or not arrow or not _linear_time(test.pattern):
                return None
        own = frame.copy(
            deep=False
        )  # whatever the test does in place, the caller's dataset stays as it is

        def run(tree: ast.AST) -> Any:
            bare = {
                n.id: own[n.id]
                for n in ast.walk(tree)
                if isinstance(n, ast.Name)
                and n.id in own.columns
                and n.id not in DEFAULT_POLICY.frame_names
            }
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return evaluate_tree(
                    ast.Expression(body=copy.deepcopy(tree)), own, extra_locals=bare
                )

        def answers(given: Any) -> pd.Series | None:
            """True, False and missing for each row, or None when the test gave something else."""
            found = pd.Series(given)
            if len(found) != len(frame):
                return None
            if not pd.api.types.is_bool_dtype(found.dtype):
                held = found.dropna()
                if not len(held) or not all(isinstance(v, bool | np.bool_) for v in held):
                    return None
            return found

        try:
            given = answers(run(test.tree))
            if given is None:
                return False  # no test of rows: a value compared with a value
            taken = given == test.holders  # a missing answer is neither
            rows = int(taken.sum())
            if test.range:
                # a range names a value when the rows it takes all hold the same one
                column = frame[test.on].reset_index(drop=True) if test.on in frame.columns else None
                same = (
                    column is not None
                    and column[taken.to_numpy(dtype=bool, na_value=False)].nunique() == 1
                )
                return bool(0 < rows < self.policy.k and same)
            if test.text_test and marked and test.plain:
                return rows < self.policy.k
            if 0 < rows < self.policy.k:
                return True
            if test.among is not None:
                tested, values = pd.Series(run(test.among[0])), run(test.among[1])
                if isinstance(values, str) or not isinstance(values, Collection):
                    return False
                for value in list(values)[:MOST_LISTED]:
                    if 0 < int((tested == value).sum()) < self.policy.k:
                        return True
            return False
        except (
            Exception
        ):  # the evaluator refuses the form, or the test fails on these data: not checked
            return None

    def _label(self, column: str) -> str:
        """The column as a hit names it: by its name, or by its position when its name is, or holds, a rare value
        that is looked for (a column the package does not show, named after a level of another)."""
        if column not in self._labels:
            by_position = bool(self._held(self._values, normal(column)))
            self._labels[column] = (
                f"column {self._position.get(column, 0)}" if by_position else column
            )
        return self._labels[column]

    def text(self, place: str, text: str) -> list[RareHit]:
        """The hits of one text (a title, a description, an expression), under the place the caller gives it."""
        named, compared, unchecked = self._read(text)
        return [
            *(RareHit(place, self._label(c), VALUE) for c in named),
            *(RareHit(place, self._label(c), LITERAL) for c in compared),
            *(RareHit(place, self._label(c), UNCHECKED) for c in unchecked),
        ]

    def bounds(self, place: str, column: str, lower: Any, upper: Any) -> list[RareHit]:
        """
        The hits of a criterion stored as a range of a column (a numeric rule, a bounds rule), which holds its
        values as numbers and not as a text: read as the comparison it stands for, an equality when its two ends
        are the same value.
        """
        ends = [v for v in (lower, upper) if isinstance(v, int | float) and not isinstance(v, bool)]
        if len(ends) < 2:
            return []  # one bound, or none: a side, not a value
        of = f"df[{str(column)!r}]"
        asked = (
            f"{of} == {ends[0]!r}"
            if ends[0] == ends[1]
            else f"{of}.between({ends[0]!r}, {ends[1]!r})"
        )
        return [hit for hit in self.text(place, asked) if hit.kind != VALUE]

    def document(self, name: str, document: Any, own: Collection[str] = ()) -> list[RareHit]:
        """
        The hits of a file read from JSON: every text it holds, the names of its items as well as their values. A
        place is the file, then what leads to the text: the file format's own names (``own``: "criteria",
        "expression", the closed list its writer gives) by name, every other name by its position ("name 3"),
        since a name the user or the data gave can hold a value that this finder does not see. A name in ``own``
        is not read: it holds the format's fixed names only.
        """
        hits: dict[RareHit, None] = {}

        def walk(value: Any, path: tuple[str, ...]) -> None:
            if isinstance(value, str):
                hits.update(dict.fromkeys(self.text(_place(name, path), value)))
            elif isinstance(value, Mapping):
                for position, (key, item) in enumerate(value.items(), start=1):
                    label = str(key)
                    if label not in own:
                        by_position = f"name {position}"
                        hits.update(
                            dict.fromkeys(self.text(_place(name, (*path, by_position)), label))
                        )
                        label = by_position
                    walk(item, (*path, label))
            elif isinstance(value, list | tuple):
                for position, item in enumerate(value, start=1):
                    walk(item, (*path, f"item {position}"))

        walk(document, ())
        return list(hits)


def _place(name: str, path: tuple[str, ...]) -> str:
    return name if not path else f"{name}: {' > '.join(path)}"


__all__ = [
    "DEFAULT_K",
    "FEW_LEFT_OUT",
    "FLOOR_K",
    "LITERAL",
    "OTHER_POPULATION",
    "POLICY_VERSION",
    "SPARSE_BELOW",
    "UNCHECKED",
    "VALUE",
    "Closure",
    "DisclosedAnalysis",
    "DisclosedBox",
    "DisclosedFlow",
    "DisclosedReason",
    "DisclosedRecord",
    "DisclosedTransition",
    "DisclosurePolicy",
    "RareFinder",
    "RareHit",
    "Shown",
    "disclose_flow",
    "disclose_record",
    "disclose_spec",
    "flow_closure",
    "identifying_columns",
    "normal",
    "one_hot_names",
    "part_of",
    "rare_values",
    "reading_notes",
    "recoverable",
    "round_down",
    "round_up",
    "shown",
    "step_at_least",
    "time_step",
]
