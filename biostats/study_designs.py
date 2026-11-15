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
    "BEFORE",
    "DESIGNS",
    "DIFFERENCE",
    "NNT",
    "NOT_STATED",
    "ODDS_RATIO",
    "OFFERED",
    "RATIO",
    "StudyDesign",
    "design_named",
]
