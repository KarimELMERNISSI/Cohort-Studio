"""
How the participants entered the study (biostats.study_designs), read by the stratified analysis, the 2x2 panel and
the dossier: each design gets the measures it supports and nothing else; a case-control sample, or a design not
stated, gets the odds ratio only. Numbers from the UGDP tolbutamide table stratified by age (Rothman, Greenland &
Lash 2008): the Mantel-Haenszel risk ratio and risk difference (Sato's variance), and the Woolf test of the risk
ratios.
"""

from __future__ import annotations

import math
import typing

import pandas as pd
import pytest

from app_pages.analysis_extras import strata_choices
from biostats.contingency import Design, TwoByTwo, analyze_twobytwo
from biostats.rates import mantel_haenszel
from biostats.study_designs import DESIGNS, NOT_STATED, OFFERED, design_named
from manage.report_tables import DESIGN_WORDS, result_blocks
from ui.guidance import measures as m

UGDP = {
    "<55": [[8, 98], [5, 115]],
    "55+": [[22, 76], [16, 69]],
}  # tolbutamide, placebo; dead, alive


def _frame(tables: dict[str, list[list[int]]]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for stratum, table in tables.items():
        for i, exposure in enumerate(("tolbutamide", "placebo")):
            for j, outcome in enumerate(("dead", "alive")):
                rows += [{"s": stratum, "e": exposure, "y": outcome}] * table[i][j]
    return pd.DataFrame(rows)


def _ugdp(design: str) -> typing.Any:
    return mantel_haenszel(
        _frame(UGDP), "e", "y", "s", exposure_level="tolbutamide", event_level="dead", design=design
    )


def test_the_designs_are_the_ids_the_2x2_panel_stores_and_the_page_names_each() -> None:
    assert set(typing.get_args(Design)) == set(DESIGNS) == set(OFFERED)
    assert OFFERED[0] == NOT_STATED and design_named(NOT_STATED).measures == ("odds_ratio",)
    assert set(m.DESIGN_LABELS) == set(m.DESIGN_SENTENCE) == set(m.DESIGN_READING) == set(DESIGNS)
    # the dossier's words are the page's labels in a sentence (manage cannot import the page's module)
    assert {d: words[0].upper() + words[1:] for d, words in DESIGN_WORDS.items()} == m.DESIGN_LABELS
    with pytest.raises(ValueError, match="unknown study design"):
        design_named("matched")


def test_a_cohort_gets_the_published_risk_ratio_difference_and_woolf_test() -> None:
    res = _ugdp("cohort")
    assert res.measures == ("ratio", "odds_ratio", "difference")
    assert (res.rr_mh, res.rr_mh_ci.lower, res.rr_mh_ci.upper) == pytest.approx(
        (1.3256, 0.7979, 2.2021), abs=5e-5
    )
    assert res.rd_mh_ci is not None and res.rd_mh_ci.method == "Sato"
    assert (res.rd_mh, res.rd_mh_ci.lower, res.rd_mh_ci.upper) == pytest.approx(
        (0.0349, -0.0276, 0.0974), abs=5e-5
    )
    assert (res.woolf_statistic, res.woolf_p, res.n_woolf) == pytest.approx(
        (0.4442, 0.5051, 2), abs=5e-5
    )
    assert "risk_ratio" in res.per_stratum


@pytest.mark.parametrize("design", ["case-control", NOT_STATED])
def test_a_case_control_sample_or_no_design_gets_the_odds_ratio_only(design: str) -> None:
    """The risk ratio of a case-control sample measures the share of controls drawn: never computed, never in its
    table or a record."""
    res = _ugdp(design)
    assert res.measures == ("odds_ratio",)
    assert res.or_mh == pytest.approx(1.4031, abs=5e-5)
    assert math.isnan(res.rr_mh) and math.isnan(res.crude_rr) and math.isnan(res.rd_mh)
    assert res.rd_mh_ci is None and res.n_woolf is None and math.isnan(res.woolf_p)
    assert "risk_ratio" not in res.per_stratum
    assert not any("risk" in line for line in res.warnings)


def test_a_result_of_before_reads_as_it_was_then() -> None:
    """Without a design (a caller of before, a result kept before the design was asked): every measure, and the
    dossier says the design was not recorded."""
    res = mantel_haenszel(
        _frame(UGDP), "e", "y", "s", exposure_level="tolbutamide", event_level="dead"
    )
    assert res.design == "" and res.measures == ()
    assert res.rr_mh == pytest.approx(1.3256, abs=5e-5)
    lines = result_blocks(res).lines
    assert "Study design: not recorded (computed before the design was asked)" in lines
    assert any(line.startswith("Mantel-Haenszel risk ratio") for line in lines)


def test_the_change_in_estimate_is_read_as_the_design_reads_it() -> None:
    """A trial: chance, not confounding; a case-control sample: the odds ratio, with the matching's caveat."""
    # crude risk ratio 2.00, adjusted 1.64; crude odds ratio 3.00, adjusted 2.25
    tables = {"a": [[40, 60], [20, 80]], "b": [[60, 40], [10, 10]]}
    trial = mantel_haenszel(
        _frame(tables),
        "e",
        "y",
        "s",
        exposure_level="tolbutamide",
        event_level="dead",
        design="rct",
    )
    assert any("by chance only" in line for line in trial.warnings)
    case = mantel_haenszel(
        _frame(tables),
        "e",
        "y",
        "s",
        exposure_level="tolbutamide",
        event_level="dead",
        design="case-control",
    )
    assert any("matched" in line for line in case.warnings)
    assert not any("risk ratio" in line for line in case.warnings)
    cohort = mantel_haenszel(
        _frame(tables),
        "e",
        "y",
        "s",
        exposure_level="tolbutamide",
        event_level="dead",
        design="cohort",
    )
    assert any("may be a confounder" in line for line in cohort.warnings)


@pytest.mark.parametrize(
    ("design", "shown", "said", "unsaid"),
    [
        (NOT_STATED, ["odds_ratio"], "Study design not stated", "risk ratio"),
        ("case-control", ["odds_ratio"], "Case-control sample", "NNT ="),
        ("cohort", ["ratio", "odds_ratio", "difference"], "risk ratio", "NNT ="),
        (
            "cross-sectional",
            ["ratio", "odds_ratio", "difference"],
            "prevalence ratio",
            "risk ratio",
        ),
        ("rct", ["ratio", "odds_ratio", "difference", "nnt"], "NNT =", "Study design not stated"),
    ],
)
def test_the_2x2_panel_lists_and_says_what_its_design_supports(
    design: str, shown: list[str], said: str, unsaid: str
) -> None:
    epi = analyze_twobytwo(
        TwoByTwo(
            a=87, b=227, c=114, d=191, exposure_level="Intervention", reference_level="Control"
        ),
        design=design,  # type: ignore[arg-type]
    ).legacy_dict()
    words = " ".join(epi["guidance"])
    assert epi["measures"] == shown and said in words and unsaid not in words
    assert (epi["nnt"] is not None) == ("nnt" in shown)


@pytest.mark.parametrize(
    ("design", "written", "unwritten"),
    [
        (
            "cohort",
            ["Study design: followed over time (cohort)", "Mantel-Haenszel risk ratio", "Woolf"],
            "prevalence",
        ),
        (
            "cross-sectional",
            ["Mantel-Haenszel prevalence ratio", "prevalence difference"],
            "risk ratio",
        ),
        ("case-control", ["Study design: by their outcome (case-control)", "odds ratio"], "risk"),
        (NOT_STATED, ["Study design not stated: odds ratios only"], "risk"),
    ],
)
def test_the_dossier_writes_the_design_and_only_what_it_supports(
    design: str, written: list[str], unwritten: str
) -> None:
    text = "\n".join(result_blocks(_ugdp(design)).lines)
    assert all(part in text for part in written), text
    assert unwritten not in text.replace("Study design: followed over time (cohort)", ""), text


def test_the_2x2_panel_s_design_is_carried_to_the_stratified_analysis() -> None:
    df = _frame(UGDP).rename(columns={"e": "arm", "y": "outcome", "s": "age"})
    carried = strata_choices(
        df, outcome="outcome", event="dead", compare="arm", reference="placebo", design="rct"
    )
    assert carried["mh_design"] == "rct" and carried["mh_design::for"] == "outcome"
    assert "mh_design" not in strata_choices(
        df, outcome="outcome", event="dead", compare="arm", reference="placebo", design="matched"
    )


# ----------------------------------------------------------------------------- the strata each test reads


def _counted(counts: dict[str, tuple[int, int, int, int]], design: str) -> typing.Any:
    """Strata written as (events among the exposed, the exposed, events in the reference group, the reference
    group), the exposed being "yes" and the event "case"."""
    return mantel_haenszel(
        _counted_frame(counts),
        "e",
        "y",
        "s",
        exposure_level="yes",
        event_level="case",
        design=design,
    )


def _counted_frame(counts: dict[str, tuple[int, int, int, int]]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for stratum, (a, n1, c, n0) in counts.items():
        rows += [{"s": stratum, "e": "yes", "y": "case"}] * a
        rows += [{"s": stratum, "e": "yes", "y": "no"}] * (n1 - a)
        rows += [{"s": stratum, "e": "no", "y": "case"}] * c
        rows += [{"s": stratum, "e": "no", "y": "no"}] * (n0 - c)
    return pd.DataFrame(rows)


# risk ratios of 0, 2 and 2: Woolf's test reads the two strata with an event in both groups, Breslow-Day all three
THREE = {"s1": (0, 10, 3, 10), "s2": (4, 10, 2, 10), "s3": (6, 10, 3, 10)}
# risk differences of -50, +50 and 0 points around a combined risk ratio of 1
OPPOSITE = {"s1": (0, 20, 10, 20), "s2": (10, 20, 0, 20), "s3": (5, 20, 5, 20)}
# a risk ratio of 2 in both strata, at baseline risks of 5% and 20%: the odds ratios differ (2.11, 2.67)
POPULATION = {"young": (300, 3000, 350, 7000), "old": (2400, 6000, 800, 4000)}
# an odds ratio of 2 in both strata, at baseline risks of 10% and 50%: the risk ratios differ (1.82, 1.33)
COMMON_ODDS = {"young": (545, 3000, 300, 3000), "old": (2000, 3000, 1500, 3000)}
DIFFER = ("the strata differ: read each stratum", "caution")


def test_a_stratum_the_test_of_the_ratios_cannot_read_is_counted_and_breslow_day_stands_in() -> (
    None
):
    """The page said the risk ratios "do not" differ, explained by a common outcome, beside a stratum of risk ratio
    0 against 2.0 in the others: the stratum Woolf's test cannot read is counted, and Breslow-Day, which reads it
    and finds that the strata differ, decides the lines and the chip. Woolf's line then keeps its numbers without a
    mark or a verdict: a green "no evidence that they differ" stood above "the odds ratios of the 3 strata differ"."""
    from app_pages.analysis_extras import strata_homogeneity

    res = _counted(THREE, "cohort")
    assert (res.n_woolf, res.n_tested) == (2, 3)
    assert list(res.per_stratum["risk_ratio"]) == pytest.approx([0.0, 2.0, 2.0])
    assert res.woolf_statistic == 0.0 and res.woolf_p == 1.0  # not the rounding's 1.2e-29
    lines, chip = strata_homogeneity(res, "risk ratio")
    assert lines == [
        ":material/info: The risk ratios of the 2 strata with an event in both groups are both 2.000 (p = 1.000, "
        "Woolf's test); 1 stratum has events in one group only (a risk ratio of 0), which that test cannot read.",
        ":material/warning: The odds ratios of the 3 strata differ (Breslow-Day p = 0.049), including the stratum "
        "the test of the risk ratios cannot read. Report the risk ratio of each stratum, not one combined risk "
        "ratio.",
    ]
    text = " ".join(lines)
    assert (
        "check_circle" not in text
        and "no evidence" not in text
        and "while the risk ratios do not" not in text
    )
    assert chip == DIFFER


def test_strata_of_opposite_directions_are_not_left_untested_when_breslow_day_can_read_them() -> (
    None
):
    """Woolf's test cannot run (one stratum has an event in both groups): "strata not tested" stood in the
    headline while Breslow-Day found p < 0.001. A stratum's risk ratio without an upper bound reads so, not "not
    estimable"."""
    from app_pages.analysis_extras import strata_homogeneity

    res = _counted(OPPOSITE, "cohort")
    assert list(res.per_stratum["risk_ratio"]) == [0.0, math.inf, 1.0]
    assert math.isnan(res.woolf_p) and res.breslow_day_p < 0.001
    lines, chip = strata_homogeneity(res, "risk ratio")
    assert lines[0] == (
        ":material/help: Whether the risk ratios of the strata differ was not tested: fewer than two strata have an "
        "event in both groups. 2 strata have events in one group only (risk ratios of 0, or without an upper "
        "bound): the test cannot read them; see the table."
    )
    assert lines[1].startswith(
        ":material/warning: The odds ratios of the strata run from 0.000 to 1.000 in 2 of"
    )
    assert "they differ (Breslow-Day p < 0.001)" in lines[1]
    assert chip == DIFFER


def test_the_two_tests_parting_is_put_down_to_a_common_outcome_only_on_the_same_strata() -> None:
    """Both tests read both strata, and the outcome is common: the odds ratios differ where the risk ratios do not,
    and the page says why; the risk ratios' span says "are both" when they are one value."""
    from app_pages.analysis_extras import strata_homogeneity

    lines, chip = strata_homogeneity(_counted(POPULATION, "cohort"), "risk ratio")
    assert lines[0].startswith(
        ":material/check_circle: The risk ratios of the strata are both 2.000: no evidence that they differ"
    )
    # the reason shows its evidence: the reference group's risks differ between the strata
    assert lines[1] == (
        ":material/info: The odds ratios differ between strata (Breslow-Day p = 0.013) while the risk ratios do "
        "not: with a common outcome and risks in `no` that differ between the strata (5.0% to 20.0%), the odds ratio "
        "changes with each stratum's risk even when the risk ratio does not. The combined risk ratio sums the strata "
        "up; the combined odds ratio does not."
    )
    assert chip is None


def test_the_two_tests_parting_the_other_way_is_said_too() -> None:
    """A common odds ratio at baseline risks of 10% and 50%: the risk ratios differ (Woolf) while the odds ratios
    show no evidence of it; the page said nothing of the turn from one verdict to the other."""
    from app_pages.analysis_extras import strata_homogeneity

    res = _counted(COMMON_ODDS, "cohort")
    assert res.woolf_p < 0.001 and res.breslow_day_p > 0.9
    lines, chip = strata_homogeneity(res, "risk ratio")
    assert lines[0].startswith(
        ":material/warning: The risk ratios of the strata run from 1.333 to 1.817: they differ"
    )
    assert lines[1].startswith(
        ":material/info: The odds ratios of the strata show no evidence that they differ (Breslow-Day p = 0.9"
    )
    assert lines[1].endswith(
        "with a common outcome and risks in `no` that differ between the strata (10.0% to 50.0%), the risk ratio "
        "changes with each stratum's risk even when the odds ratio does not. The combined odds ratio sums the "
        "strata up; the combined risk ratio does not."
    )
    assert chip == DIFFER


def test_no_reason_of_scale_at_one_baseline_risk() -> None:
    """Risk ratios of 1.4 to 2.6 at one baseline risk of 25%: the two tests part (Woolf p = 0.167, Breslow-Day p =
    0.032), but not by scale, since at one baseline risk a common risk ratio is a common odds ratio. The page said the
    risk ratios "do not" differ and that their combined ratio sums the strata up, with no chip; now the warning, and
    its chip."""
    from app_pages.analysis_extras import strata_homogeneity
    from biostats.rates import reference_risks

    one_baseline = {
        "s1": (35, 100, 25, 100),
        "s2": (45, 100, 25, 100),
        "s3": (55, 100, 25, 100),
        "s4": (65, 100, 25, 100),
    }
    res = _counted(one_baseline, "cohort")
    assert reference_risks(res.per_stratum) == pytest.approx((1.0, 0.25, 0.25))
    lines, chip = strata_homogeneity(res, "risk ratio")
    assert lines == [
        ":material/info: The risk ratios of the strata run from 1.400 to 2.600 (p = 0.167, Woolf's test).",
        ":material/warning: The odds ratios of the 4 strata differ (Breslow-Day p = 0.032). Report the risk ratio "
        "of each stratum, not one combined risk ratio.",
    ]
    assert chip == DIFFER
    p, low, high = reference_risks(_counted(POPULATION, "cohort").per_stratum)
    assert p < 0.001 and (low, high) == pytest.approx((0.05, 0.20))


def test_the_adjusted_difference_stands_beside_the_strata_own_on_the_page_and_in_the_dossier() -> (
    None
):
    """A risk ratio of 2 in both strata: their differences are 5.0 and 20.0 points, where the adjusted one is 13.0.
    The dossier writes points, as the page, and the strata Woolf's test read."""
    from app_pages.analysis_extras import _difference_line

    res = _counted(POPULATION, "cohort")
    assert _difference_line(res, False, "risk ratio") == (
        "Risk difference, adjusted (Mantel-Haenszel): 13.0 percentage points, 95% CI 11.9 to 14.1 (Sato's "
        "variance). The strata's own run from 5.0 to 20.0 points: one combined difference sums them up only when "
        "they are alike; with a common risk ratio, differences change with each stratum's risk."
    )
    text = "\n".join(result_blocks(res).lines)
    assert (
        "Mantel-Haenszel risk difference = 13.0 percentage points (95% CI 11.9 to 14.1; Sato's variance). The "
        "strata's own run from 5.0 to 20.0 points" in text
    )
    assert (
        "Woolf test that the risk ratios of the strata are alike: chi2 = 0, p = 1.000, on 2 strata"
        in text
    )


def test_no_interval_when_every_stratum_lies_at_the_edge() -> None:
    """Every stratum's difference is -100 points: Sato's variance is 0, and the interval read "-1 to 1"."""
    from app_pages.analysis_extras import _difference_line

    res = _counted({"s1": (0, 1, 1, 1), "s2": (0, 1, 1, 1)}, "cohort")
    assert res.rd_mh == -1.0 and res.rd_mh_ci is None
    assert "without an interval: every stratum lies at the edge" in _difference_line(
        res, False, "risk ratio"
    )
    assert "-100.0 percentage points (no interval: every stratum lies at the edge" in "\n".join(
        result_blocks(res).lines
    )


def test_the_published_rabbits_table_gives_the_odds_ratio_interval_and_test_of_r() -> None:
    """Penicillin and the delay of the injection (Agresti 1990, the example of mantelhaen.test in R): common odds
    ratio 7, 95% CI 1.026713 to 47.725133, Mantel-Haenszel X-squared = 3.9286, p = 0.04747. The two strata where
    every rabbit lived, or every one died, add nothing."""
    rabbits = {  # (cured without delay, without delay, cured after 1.5 h, after 1.5 h)
        "1/8": (0, 6, 0, 5),
        "1/4": (3, 6, 0, 6),
        "1/2": (6, 6, 2, 6),
        "1": (5, 6, 6, 6),
        "4": (2, 2, 5, 5),
    }
    res = _counted(rabbits, NOT_STATED)
    assert res.n_tested == 3
    assert (res.or_mh, res.or_mh_ci.lower, res.or_mh_ci.upper) == pytest.approx(
        (7.0, 1.026713, 47.725133), abs=5e-6
    )
    assert (res.mh_test_statistic, res.mh_test_p) == pytest.approx((3.9286, 0.04747), abs=5e-5)
    assert any("Robins J, Breslow N, Greenland S (1986)" in ref for ref in res.references)


def test_the_plan_names_the_strata_the_test_of_the_ratios_cannot_read_and_the_case_control_line() -> (
    None
):
    df = _counted_frame(THREE)
    cohort = m.plan_strata(df, "y", "case", "e", "no", "s", "cohort")
    # which case it is, as the result writes it: no event in the group compared
    assert (
        "1 of 3 strata has events in one group only (a risk ratio of 0): the test that the risk ratios are alike "
        "cannot read it (Breslow-Day's test of the odds ratios can)."
        in [advice.text for advice in cohort.advice]
    )
    case = m.plan_strata(df, "y", "case", "e", "no", "s", "case-control")
    assert m.DESIGN_CASE_CONTROL in [advice.text for advice in case.advice]
    assert "case-control" not in case.steps[0].text
    assert not any("events in one group only" in advice.text for advice in case.advice)


def test_the_number_needed_to_treat_says_which_way_the_difference_goes() -> None:
    """27.7% against 37.4%: the panel said "about 10 people ... per one additional event", of an intervention that
    prevents one."""
    fewer = analyze_twobytwo(
        TwoByTwo(
            a=87, b=227, c=114, d=191, exposure_level="Intervention", reference_level="Control"
        ),
        design="rct",
    ).guidance
    said = [line for line in fewer if line.startswith("NNT = ")]
    assert len(said) == 1 and said[0].endswith(
        ": for about every 10 participants in 'Intervention' rather than 'Control', one event fewer."
    ), fewer
    more = analyze_twobytwo(
        TwoByTwo(
            a=114, b=191, c=87, d=227, exposure_level="Control", reference_level="Intervention"
        ),
        design="rct",
    ).guidance
    assert any(line.startswith("NNH = ") and line.endswith("one event more.") for line in more), (
        more
    )
    assert not any("additional" in line for line in fewer + more)


@pytest.mark.parametrize(
    ("design", "named"),
    [
        (NOT_STATED, "applied to the odds ratio because a cell is zero: read it with caution."),
        ("case-control", "applied to the odds ratio because a cell is zero: read it with caution."),
        ("cohort", "applied to the odds ratio and the risk ratio because a cell is zero"),
        ("rct", "applied to the odds ratio and the risk ratio because a cell is zero"),
        (
            "cross-sectional",
            "applied to the odds ratio and the prevalence ratio because a cell is zero",
        ),
    ],
)
def test_a_zero_cell_names_only_the_ratios_its_design_reads(design: str, named: str) -> None:
    guidance = analyze_twobytwo(
        TwoByTwo(a=6, b=20, c=0, d=30, exposure_level="A", reference_level="B"),
        design=design,  # type: ignore[arg-type]
    ).guidance
    said = [line for line in guidance if "Haldane-Anscombe" in line]
    assert len(said) == 1 and named in said[0], guidance
    if design not in ("cohort", "rct"):
        assert "risk ratio" not in " ".join(guidance)
