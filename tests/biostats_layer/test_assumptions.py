"""
What each assumption check means for the test that ran (biostats/assumptions.py, M5). The result card showed
"normality (No)" and "normality (Yes)" side by side, without the group variable, the finding, or whether the check
matters for the test: a red "equal variances" under Welch's t-test, which is designed for unequal variances.
"""

from __future__ import annotations

import numpy as np

from biostats._types import AssumptionCheck, TestResult
from biostats.assumptions import read_assumptions
from biostats.inference import compare_groups


def _groups(seed: int = 2, n: tuple[int, int] = (60, 45)) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "No": rng.normal(5.5, 0.4, n[0]),
        "Yes": rng.lognormal(2.0, 0.5, n[1]),
    }  # unequal, skewed


def _by_check(readings, start: str):
    return next(r for r in readings if r.check.startswith(start))


def test_each_check_names_its_group_and_its_finding() -> None:
    readings = read_assumptions(compare_groups(_groups(), kind="welch"), "diabetes")
    names = [r.check for r in readings]
    assert names[:2] == ["Normality, diabetes = No (n = 60)", "Normality, diabetes = Yes (n = 45)"]
    assert "Equal variances (Levene)" in names and "Sample size" in names
    variances = _by_check(readings, "Equal variances")
    assert variances.finding.startswith("Levene, median-centred: p = ")
    assert variances.sentence.startswith(
        "Levene, median-centred: p = "
    ) and variances.sentence.endswith(".")
    assert all(r.finding for r in readings)  # the test and its p-value, not just a colour
    unnamed = read_assumptions(compare_groups(_groups(), kind="welch"))
    assert unnamed[0].check == "Normality, group No (n = 60)"


def test_unequal_variances_do_not_matter_for_welch_but_do_for_student() -> None:
    welch = _by_check(read_assumptions(compare_groups(_groups(), kind="welch")), "Equal variances")
    assert welch.status in ("caution", "violated") and welch.meaning.startswith("Not required")
    student = _by_check(
        read_assumptions(compare_groups(_groups(), kind="student")), "Equal variances"
    )
    assert (
        "Required by Student's t-test" in student.meaning
        and "prefer Welch's test" in student.meaning
    )


def test_non_normality_matters_with_small_groups_only_and_never_for_a_rank_test() -> None:
    large = _by_check(
        read_assumptions(compare_groups(_groups(), kind="welch"), "d"), "Normality, d = Yes"
    )
    assert large.status != "ok" and large.meaning.startswith("Not critical here: with 45 values")
    few = {
        "No": [5.1, 5.4, 5.0, 5.6, 5.3, 5.2, 5.5, 5.4],
        "Yes": [1.0, 1.1, 1.2, 1.1, 1.3, 1.0, 6.0, 9.0],
    }
    small = _by_check(
        read_assumptions(compare_groups(few, kind="welch"), "d"), "Normality, d = Yes"
    )
    assert small.status != "ok" and small.meaning.startswith("Matters with so few values")
    rank = _by_check(
        read_assumptions(compare_groups(_groups(), kind="mannwhitney"), "d"), "Normality"
    )
    assert rank.meaning == "Not assumed by a rank test."


def test_a_check_that_is_not_about_a_group_keeps_its_name() -> None:
    paired = TestResult(
        name="Paired t-test",
        assumptions=[
            AssumptionCheck(name="normality of the differences", status="ok", detail="p = 0.4")
        ],
    )
    (reading,) = read_assumptions(paired, "visit")
    assert reading.check == "Normality of the differences"
    assert reading.meaning == "Consistent with a normal distribution."
    assert (
        reading.sentence == "p = 0.4. Consistent with a normal distribution."
    )  # p stays lower case


def test_a_large_group_is_not_told_twice_to_look_at_the_q_q_plot() -> None:
    from biostats.assumptions import LARGE_GROUP

    rng = np.random.default_rng(7)
    groups = {"a": rng.normal(0, 1, 400), "b": rng.exponential(1.0, 400)}
    reading = _by_check(
        read_assumptions(compare_groups(groups, kind="welch"), "g"), "Normality, g = b"
    )
    assert 400 >= LARGE_GROUP and "Q-Q" in reading.finding
    assert reading.sentence.count("Q-Q") == 1
