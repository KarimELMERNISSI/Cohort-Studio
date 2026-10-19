"""
What each assumption check of a test means for the test that ran (M5).

A check is a diagnostic, never a gate: the test runs regardless. The same finding matters differently from one test
to another. Unequal variances are the reason Welch's t-test is the default, but they undermine Student's t-test and
the one-way ANOVA. Non-normality matters for a test of means only with small groups, and not at all for a rank test.
The result card shows each check with the group it concerns, its finding, and this reading.
"""

from __future__ import annotations

from dataclasses import dataclass

from biostats._types import AssumptionCheck, TestResult

# values per group from which a test of means is robust to non-normality (central limit theorem)
LARGE_GROUP = 30


@dataclass(frozen=True)
class AssumptionReading:
    """One check as the result card shows it."""

    check: str  # what was checked, with the group it concerns: "Normality, diabetes = No (n = 600)"
    finding: str  # what the check found: "no evidence against normality (p = 0.21)"
    status: str  # ok | caution | violated | na
    meaning: str  # what the finding means for the test that ran

    @property
    def sentence(self) -> str:
        """The finding and its meaning as one sentence ("p" stays lower case)."""
        finding = self.finding.rstrip(". ")
        if finding and not finding.startswith("p "):
            finding = finding[0].upper() + finding[1:]
        text = ". ".join(part for part in (finding, self.meaning.strip()) if part)
        return text if text.endswith(".") else text + "."


def _test_kind(result: TestResult) -> str:
    name = result.name.lower()
    if result.family == "nonparametric" or any(
        rank in name for rank in ("mann-whitney", "kruskal", "wilcoxon")
    ):
        return "rank"
    if "welch" in name:
        return "welch"
    return "means"  # Student's t-test, the one-way ANOVA, paired and one-sample t-tests


def _normality(kind: str, check: AssumptionCheck, n: int | None) -> str:
    if kind == "rank":
        return "Not assumed by a rank test."
    if check.status == "na":
        return "Too few values to judge; look at the values themselves."
    if check.status == "ok":
        return "Consistent with a normal distribution."
    if n is not None and n >= LARGE_GROUP:
        meaning = (
            f"Not critical here: with {n} values, a test of means is robust to non-normality (central limit "
            "theorem)."
        )
        if "Q-Q" not in check.detail:  # a large sample's finding already says to judge the Q-Q plot
            meaning += " Look at the Q-Q plot for strong skew or outliers."
        return meaning
    return (
        "Matters with so few values: if the Q-Q plot shows clear skew or outliers, prefer the rank-based test "
        "(Mann-Whitney U for two groups, Kruskal-Wallis for more)."
    )


def _variances(kind: str, check: AssumptionCheck, test: str) -> str:
    if kind == "welch":
        return "Not required: Welch's test allows the groups different variances."
    if kind == "rank":
        return (
            "Not assumed by a rank test, but different spreads change what it compares: whether one group tends "
            "to have larger values, not only a shift in location."
        )
    if check.status == "ok":
        return f"Required by {test}, and met."
    if check.status == "na":
        return f"Required by {test}; it could not be checked."
    return f"Required by {test}: the variances differ, so prefer Welch's test, which does not assume equal variances."


def _sample_size(check: AssumptionCheck) -> str:
    if check.status == "ok":
        return "Large enough for the test's approximations."
    return "With so few values the approximations are rough; an exact or rank-based test is safer."


def read_assumptions(
    result: TestResult, group_variable: str | None = None
) -> list[AssumptionReading]:
    """
    The checks of ``result``, each with the group it concerns (``group_variable = level``), its finding and what it
    means for the test that ran.
    """
    sizes = dict(zip(result.groups, result.n_per_group, strict=False))
    kind = _test_kind(result)
    readings: list[AssumptionReading] = []
    for check in result.assumptions:
        name, meaning = check.name, ""
        group = (
            name[len("normality (") : -1]
            if name.startswith("normality (") and name.endswith(")")
            else None
        )
        if group is not None and group in sizes:
            where = f"{group_variable} = {group}" if group_variable else f"group {group}"
            name = f"Normality, {where} (n = {sizes[group]})"
            meaning = _normality(kind, check, sizes[group])
        elif name.startswith("normality"):
            name = name[0].upper() + name[1:]
            meaning = _normality(kind, check, result.n_total or None)
        elif name.startswith("equal variances"):
            method = name[len("equal variances") :].strip(" ()")
            name = "Equal variances (Levene)" if "Levene" in method else "Equal variances"
            meaning = _variances(kind, check, result.name)
            if method and check.detail:  # the test in the sentence keeps the label short
                readings.append(
                    AssumptionReading(name, f"{method}: {check.detail}", check.status, meaning)
                )
                continue
        elif name == "sample size":
            name = "Sample size"
            meaning = _sample_size(check)
        readings.append(AssumptionReading(name, check.detail, check.status, meaning))
    return readings


__all__ = ["LARGE_GROUP", "AssumptionReading", "read_assumptions"]
