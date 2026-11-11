"""
What is drawn at random in a stored result, and what its settings say of it (M8, the study package).

A group comparison draws nothing: its rank-based intervals are computed (biostats.rank_intervals), so its summary
table and its result card give the same numbers, read in the same direction, whatever the order of the rows. The one
stored interval that is drawn is the range of the Youden cut-off: its resamples and seed are given to the computation
and recorded in the result's settings from the same object, in Youden mode only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import utils.analysis_utils as au
from biostats._types import RANDOM_STATE, YOUDEN_BOOTSTRAP, Bootstrap
from biostats.diagnostic import youden_cutoff


def _frame() -> pd.DataFrame:
    """Two arms; the first row belongs to the arm that sorts last, as in the tutorial's analysis population."""
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "arm": ["Intervention"] * 40 + ["Control"] * 45,
            "crp": np.r_[rng.lognormal(1.4, 1, 40), rng.lognormal(1, 1, 45)],
            "sbp": np.r_[rng.normal(128, 14, 40), rng.normal(135, 15, 45)].round(),
        }
    )
    assert frame["arm"].iloc[0] == sorted(frame["arm"].unique())[-1]
    return frame


@pytest.mark.parametrize("test", ["Mann-Whitney U", "Welch's t-test"])
@pytest.mark.parametrize("target", ["crp", "sbp"])
def test_the_summary_row_and_the_result_card_give_the_same_numbers(test: str, target: str) -> None:
    """Each took its own order of the groups: the summary read "Intervention - Control" and the card the reverse,
    with other intervals."""
    frame = _frame()
    row = au.analyze_variable(frame, "arm", target, ["crp", "sbp"], ["arm"], [], manual_test=test)
    card = au.compare_variable_result(frame, "arm", target, manual_test=test)
    assert card is not None
    assert card.reference_group == "Control" and card.comparison_group == "Intervention"
    assert (
        row["Estimate of"] == card.estimate_label
        and "(Intervention - Control)" in card.estimate_label
    )
    assert row["Estimate"] == card.estimate
    assert row["Estimate CI"] == card.estimate_ci.as_text(3)
    assert row["Effect Size"] == card.primary_effect.estimate
    assert row["Effect CI"] == card.primary_effect.ci.as_text(3)
    assert row["P-Value"] == card.p_value
    # the same rows in another order: the same numbers
    shuffled = frame.sample(frac=1, random_state=4)
    again = au.compare_variable_result(shuffled, "arm", target, manual_test=test)
    assert again.estimate_label == card.estimate_label
    if test == "Mann-Whitney U":  # order statistics: the very same numbers
        assert (again.estimate, again.estimate_ci) == (card.estimate, card.estimate_ci)
    else:  # a mean is a sum: its last digit follows the order of the rows
        assert again.estimate == pytest.approx(card.estimate, rel=1e-12)
        assert (again.estimate_ci.lower, again.estimate_ci.upper) == pytest.approx(
            (card.estimate_ci.lower, card.estimate_ci.upper), rel=1e-12
        )


def test_a_group_comparison_draws_nothing() -> None:
    frame = _frame()
    state = np.random.get_state()[1].copy()
    first = au.compare_variable_result(frame, "arm", "crp", manual_test="Mann-Whitney U")
    second = au.compare_variable_result(frame, "arm", "crp", manual_test="Mann-Whitney U")
    assert first == second
    assert "bootstrap" not in (first.estimate_ci.method + first.primary_effect.ci.method)
    assert (
        np.random.get_state()[1] == state
    ).all()  # numpy's global generator is not touched either


def test_the_settings_of_a_comparison_say_nothing_of_a_bootstrap() -> None:
    from ui.results import describe_settings

    shown = describe_settings(
        ("arm", ("Control", "Intervention"), ("diabetes", "crp"), "Auto-Detect", "holm", None)
    )
    assert "bootstrap" not in shown


def test_the_draws_and_the_recorded_setting_come_from_one_object() -> None:
    from ui.results import config_hash, describe_settings

    assert YOUDEN_BOOTSTRAP == Bootstrap(2000, RANDOM_STATE)
    assert YOUDEN_BOOTSTRAP.arguments() == {"n_boot": 2000, "seed": RANDOM_STATE}
    setting = YOUDEN_BOOTSTRAP.setting
    assert setting == f"bootstrap of 2000 resamples (seed {RANDOM_STATE})"
    assert "," not in setting  # one setting in a line of settings separated by commas
    youden = ("death", "troponin", "1", "Youden-optimal (data-driven)", 30.0, 0.2, setting)
    assert describe_settings(youden).endswith(setting)
    # another seed or another number of resamples is another result
    other = (
        "death",
        "troponin",
        "1",
        "Youden-optimal (data-driven)",
        30.0,
        0.2,
        Bootstrap(2000, 7).setting,
    )
    assert config_hash(youden) != config_hash(other)
    with pytest.raises(AttributeError):
        YOUDEN_BOOTSTRAP.resamples = 10  # type: ignore[misc]


def test_the_youden_range_follows_its_resamples_and_seed() -> None:
    rng = np.random.default_rng(5)
    outcome = ["no"] * 150 + ["yes"] * 150
    score = np.r_[rng.normal(0, 1, 150), rng.normal(1.2, 1, 150)]
    a = youden_cutoff(outcome, score, pos_label="yes", n_boot=200, seed=1)
    b = youden_cutoff(outcome, score, pos_label="yes", n_boot=200, seed=1)
    c = youden_cutoff(outcome, score, pos_label="yes", n_boot=200, seed=2)
    assert a["ci"] == b["ci"] and a["ci"] != c["ci"]
    assert "bootstrap percentile" in a["ci"].method
    assert (
        youden_cutoff(outcome, score, pos_label="yes", n_boot=0)["ci"] is None
    )  # a manual cut-off draws nothing
    assert a["cutoff"] == c["cutoff"]  # the cut-off itself is not drawn
    assert "rough when the ROC curve is flat" in a["note"]
    assert (
        "not a confidence interval" in a["note"]
    )  # the middle of the resampled cut-offs: its level is not known
    assert a["resamples"] == 200 and a["caution"] == ""
    # the same rows in another order: the same range (the rows are put in order before they are resampled)
    shuffled = np.random.default_rng(0).permutation(len(score))
    again = youden_cutoff(
        [outcome[i] for i in shuffled], score[shuffled], pos_label="yes", n_boot=200, seed=1
    )
    assert (again["ci"], again["cutoff"]) == (a["ci"], a["cutoff"])
    # a level of fewer than 5 rows: nothing is drawn
    few = youden_cutoff(["no"] * 30 + ["yes"] * 4, np.arange(34.0), pos_label="yes", n_boot=200)
    assert few["ci"] is None and few["resamples"] == 0


def test_the_youden_cut_off_is_an_observed_score() -> None:
    """A marker that points the other way gave an infinite cut-off: the ROC curve's first point, above every score,
    won the maximum with a sensitivity of 0."""
    rng = np.random.default_rng(8)
    outcome = ["no"] * 80 + ["yes"] * 80
    score = np.r_[rng.normal(1.0, 1, 80), rng.normal(0, 1, 80)]  # lower in those with the condition
    found = youden_cutoff(outcome, score, pos_label="yes", n_boot=0)
    assert np.isfinite(found["cutoff"]) and found["cutoff"] in set(score)
    assert found["youden"] <= 0  # nothing better than chance in this direction, and it says so
    assert found["caution"].startswith("The marker points the other way (area under the curve 0.")
    # a marker that separates nothing: the best cut-off found is its largest score, and no one is called positive
    flat = youden_cutoff(["no", "yes"] * 20, [1.0] * 38 + [2.0, 2.0], pos_label="yes", n_boot=0)
    assert flat["cutoff"] == 2.0 and "largest observed score" in flat["caution"]
    good = youden_cutoff(outcome, -score, pos_label="yes", n_boot=0)
    assert good["caution"] == "" and good["youden"] > 0.2
