"""
How the participants of a study were sampled, and the measures of an association each sampling supports (Rothman,
Greenland & Lash 2008, chapters 6 and 15; the same split as R's ``epiR::epi.2by2`` methods). The ids are those the
2x2 panel stores: never renamed; their words are the page's (``ui.guidance.measures``).

- **The odds ratio and the test of the association** mean the same whatever the sampling: a design not stated gets
  them only.
- **A case-control sample** gets nothing else: the share with the outcome is the sampling's, so risks, their ratio
  and their difference cannot be estimated.
- **A cohort or a randomised trial:** risks, their ratio and their difference; the number needed to treat for a trial
  only (it reads as the effect of an intervention).
- **A cross-sectional sample:** the same, as prevalences.

A result kept before the design was asked has none (``""``): it reads as it was then, every measure shown.
"""

from __future__ import annotations

from dataclasses import dataclass

# the measures, by stored id: the ratio and the difference are of risks, or of prevalences in a cross-sectional sample
ODDS_RATIO = "odds_ratio"
RATIO = "ratio"
DIFFERENCE = "difference"
NNT = "nnt"

NOT_STATED = "unknown"
BEFORE = ""  # a result kept before the design was asked


@dataclass(frozen=True)
class StudyDesign:
    """A sampling, and what it supports: the measures, what the share with the outcome in a group is (``"risk"``,
    ``"prevalence"``, or ``""`` when it is the sampling's), and whether the groups were formed by chance."""

    id: str
    measures: tuple[str, ...]
    shares: str = ""
    randomised: bool = False

    def supports(self, measure: str) -> bool:
        return measure in self.measures


DESIGNS: dict[str, StudyDesign] = {
    NOT_STATED: StudyDesign(NOT_STATED, (ODDS_RATIO,)),
    "cohort": StudyDesign("cohort", (RATIO, ODDS_RATIO, DIFFERENCE), "risk"),
    "rct": StudyDesign("rct", (RATIO, ODDS_RATIO, DIFFERENCE, NNT), "risk", randomised=True),
    "cross-sectional": StudyDesign(
        "cross-sectional", (RATIO, ODDS_RATIO, DIFFERENCE), "prevalence"
    ),
    "case-control": StudyDesign("case-control", (ODDS_RATIO,)),
}

# the order the page offers them in
OFFERED = (NOT_STATED, "cohort", "rct", "cross-sectional", "case-control")

# The accuracy of a test asks the same question with answers of its own (biostats.diagnostic.SAMPLINGS: two
# catalogues, since the answers differ). Where a design declared for an outcome later fills that answer in, it is by
# this table, and only when the reference standard is that outcome:
# - sampled by it (case-control), those with and without the condition were sampled apart (two-gate);
# - followed over time (a cohort, a trial), the participants entered before the condition was known, consecutively
#   or at random (a series). The share with the outcome is then a risk only with complete follow-up: a participant
#   lost before the condition could be seen counts as without it, unchecked; in a trial it is the arms' mix;
# - taken at one point in time, they were not selected by the condition (a series);
# - a design not stated is a sampling not stated, and so is a result kept before the design was asked
#   (:func:`accuracy_sampling` reads the table for any id).
ACCURACY_SAMPLING: dict[str, str] = {
    NOT_STATED: "not_stated",
    "cohort": "series",
    "rct": "series",
    "cross-sectional": "series",
    "case-control": "two_gate",
}


def accuracy_sampling(design: str) -> str:
    """The sampling of an accuracy study for a study design's id (``ACCURACY_SAMPLING``); not stated for a result
    kept before the design was asked, and for any id this version does not know."""
    return ACCURACY_SAMPLING.get(design, "not_stated")


_BEFORE_DESIGNS = StudyDesign(BEFORE, (RATIO, ODDS_RATIO, DIFFERENCE, NNT), "risk")


def design_named(design: str | None) -> StudyDesign:
    """The design of a stored id; ``""`` or None (a result kept before the design was asked): every measure, as it
    was read then. ValueError for an id this version does not know, naming the ids it knows (never the value)."""
    if not design:
        return _BEFORE_DESIGNS
    found = DESIGNS.get(design)
    if found is None:
        raise ValueError(f"unknown study design; the designs: {', '.join(DESIGNS)}")
    return found


__all__ = [
    "ACCURACY_SAMPLING",
    "BEFORE",
    "DESIGNS",
    "DIFFERENCE",
    "NNT",
    "NOT_STATED",
    "ODDS_RATIO",
    "OFFERED",
    "RATIO",
    "StudyDesign",
    "accuracy_sampling",
    "design_named",
]
