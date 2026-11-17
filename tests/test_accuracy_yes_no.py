"""
A yes/no index test against a reference standard, and what the accuracy of a test reads by the sampling: the measures
of a 2x2 table at its edges (a likelihood ratio of 0, or without an upper bound, with the one bound the data leave;
a predictive value at a prevalence with its interval), predictive values at the share in the data for a series only,
the table read from two columns, the check of partial verification, the plan's lines, the dossier's block and the
result's record.

Numbers: the liver scan of Altman & Bland's Statistics Notes (BMJ 1994: 231, 32, 27, 54) and Deeks & Altman (2004)
for the likelihood ratios; Koopman's score interval against statsmodels where the library is right; Mercaldo's logit
interval computed by hand.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep

from app_pages import analysis_diagnostic as page
from app_pages.analysis_diagnostic import accuracy_of, accuracy_settings, accuracy_yes_no_of
from biostats import plan
from biostats._types import CI
from biostats.diagnostic import (
    CUTOFF_MODES,
    DELONG,
    DIRECTIONS,
    INDEX_FORMS,
    KOOPMAN,
    MERCALDO,
    SAMPLING_NOT_STATED,
    SAMPLINGS,
    SCORE_CARRIED,
    SERIES,
    SIMEL,
    TWO_GATE,
    WILSON,
    WOOLF,
    WOOLF_CORRECTED,
    diagnostic_accuracy,
    reads_share,
    score_ratio_interval,
    verification,
    yes_no_table,
)
from manage.report_tables import SAMPLING_WORDS, result_blocks, sampling_line
from ui.guidance import measures as m
from ui.guidance.numbers import share_parts

LIVER = (231, 32, 27, 54)  # tp, fp, fn, tn


# ----------------------------------------------------------------------------- the measures of a table


def test_a_published_table_gives_its_published_measures() -> None:
    acc = diagnostic_accuracy(*LIVER, prevalence=0.10, sampling=SERIES)
    assert (acc.sensitivity, acc.specificity) == pytest.approx((0.8953, 0.6279), abs=5e-5)
    assert (acc.ppv, acc.npv) == pytest.approx((0.8783, 0.6667), abs=5e-5)
    assert (acc.lr_positive, acc.lr_negative, acc.dor) == pytest.approx(
        (2.406, 0.1667, 14.44), abs=5e-3
    )
    assert acc.lr_positive_ci is not None and acc.lr_negative_ci is not None
    assert (acc.lr_positive_ci.lower, acc.lr_positive_ci.upper) == pytest.approx(
        (1.82, 3.18), abs=5e-3
    )
    assert (acc.lr_negative_ci.lower, acc.lr_negative_ci.upper) == pytest.approx(
        (0.11, 0.25), abs=5e-3
    )
    # the paper's predictive values hold at its series' share of 75%; at 10% they are others, with an interval
    assert acc.ppv_at_prevalence == pytest.approx(0.211, abs=5e-4)
    assert acc.npv_at_prevalence == pytest.approx(0.982, abs=5e-4)
    assert acc.ppv_at_prevalence_ci is not None and acc.npv_at_prevalence_ci is not None
    assert (acc.ppv_at_prevalence_ci.lower, acc.ppv_at_prevalence_ci.upper) == pytest.approx(
        (0.168, 0.261), abs=5e-4
    )
    assert (acc.npv_at_prevalence_ci.lower, acc.npv_at_prevalence_ci.upper) == pytest.approx(
        (0.973, 0.988), abs=5e-4
    )
    assert not acc.warnings and not acc.dor_corrected


def test_a_predictive_value_at_a_prevalence_carries_mercaldo_s_logit_interval() -> None:
    """The likelihood ratio's log interval carried through Bayes' rule is the logit interval of Mercaldo, Lau & Zhou
    (2007), the prevalence taken as known: computed here by hand on the reference table at 10%."""
    tp, fp, fn, tn, share = 90, 10, 20, 80, 0.10
    acc = diagnostic_accuracy(tp, fp, fn, tn, prevalence=share)
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    z = stats.norm.ppf(0.975)
    logit = math.log(sens * share / ((1 - spec) * (1 - share)))
    half = z * math.sqrt((1 - sens) / (sens * (tp + fn)) + spec / ((1 - spec) * (tn + fp)))
    by_hand = [1 / (1 + math.exp(-(logit + sign * half))) for sign in (-1, 1)]
    assert acc.ppv_at_prevalence == pytest.approx(0.45, abs=5e-4)
    assert acc.ppv_at_prevalence_ci is not None
    assert (acc.ppv_at_prevalence_ci.lower, acc.ppv_at_prevalence_ci.upper) == pytest.approx(
        by_hand
    )
    assert by_hand == pytest.approx([0.312, 0.596], abs=5e-4)
    assert acc.ppv_at_prevalence_ci.method == "logit (Mercaldo)"


def test_the_score_interval_of_a_ratio_is_the_library_s_where_the_library_is_right() -> None:
    """Koopman's interval by root-finding. statsmodels' own returns complex numbers when the second share is 100%
    (the negative likelihood ratio of a perfect table read "0.5 to 0.5")."""
    for cells in (
        (90, 110, 10, 90),
        (20, 110, 80, 90),
        (3, 15, 9, 15),
        (10, 12, 2, 8),
        (7, 30, 21, 30),
    ):
        theirs = confint_proportions_2indep(
            *cells, method="score", compare="ratio", correction=False
        )
        assert score_ratio_interval(*cells) == pytest.approx(theirs, abs=1e-9)
    assert score_ratio_interval(0, 20, 20, 20) == pytest.approx((0.0, 0.161), abs=5e-4)
    assert score_ratio_interval(20, 20, 0, 20)[1] == math.inf
    assert all(math.isnan(end) for end in score_ratio_interval(0, 10, 0, 10))


@pytest.mark.parametrize(
    ("cells", "ratio", "value", "bound", "said"),
    [
        # sensitivity 100%: LR- is 0, and the 30 participants with the condition allow up to 0.14
        ((30, 6, 0, 24), "lr_negative", 0.0, ("upper", 0.1425), "LR- is 0"),
        # specificity 100%: LR+ has no upper bound, and is at least 2.5
        ((10, 0, 2, 8), "lr_positive", math.inf, ("lower", 2.4995), "LR+ has no upper bound"),
        ((0, 3, 12, 9), "lr_positive", 0.0, ("upper", 1.1132), "LR+ is 0"),
        ((12, 9, 3, 0), "lr_negative", math.inf, ("lower", 0.5611), "LR- has no upper bound"),
    ],
)
def test_a_zero_cell_leaves_a_likelihood_ratio_its_one_bound(
    cells: tuple[int, int, int, int], ratio: str, value: float, bound: tuple[str, float], said: str
) -> None:
    """ "LR- 0" stood without an interval or a word, and a likelihood ratio without an upper bound read "undefined"."""
    acc = diagnostic_accuracy(*cells)
    ci = getattr(acc, f"{ratio}_ci")
    assert getattr(acc, ratio) == value and ci is not None and ci.method == "Koopman score"
    assert getattr(ci, bound[0]) == pytest.approx(bound[1], abs=5e-5)
    assert (ci.lower == 0.0) if bound[0] == "upper" else (ci.upper == math.inf)
    assert any(text.startswith(said) for text in acc.warnings), acc.warnings
    assert acc.dor is not None and acc.dor_corrected
    assert any("Koopman" in ref for ref in acc.references)


def test_the_bound_reaches_the_predictive_value_at_a_prevalence() -> None:
    """Specificity 8 of 8 at a prevalence of 2%: the positive predictive value read "100%" with no interval, where
    eight participants without the condition allow 4.9%."""
    acc = diagnostic_accuracy(10, 0, 2, 8, prevalence=0.02)
    assert acc.ppv_at_prevalence == 1.0 and acc.ppv_at_prevalence_ci is not None
    assert (acc.ppv_at_prevalence_ci.lower, acc.ppv_at_prevalence_ci.upper) == pytest.approx(
        (0.0485, 1.0), abs=5e-5
    )
    perfect = diagnostic_accuracy(30, 6, 0, 24, prevalence=0.02)
    assert perfect.npv_at_prevalence == 1.0 and perfect.npv_at_prevalence_ci is not None
    assert perfect.npv_at_prevalence_ci.lower == pytest.approx(0.9971, abs=5e-5)


def test_a_test_result_nobody_has_leaves_no_ratio() -> None:
    """No one called positive: "LR- 1.00 (1.00 to 1.00)" was written, an interval of no width."""
    acc = diagnostic_accuracy(0, 0, 12, 9)
    assert acc.lr_positive is None and acc.lr_negative is None and acc.dor is None
    assert acc.lr_negative_ci is None
    assert any(text.startswith("no one is called positive") for text in acc.warnings)
    everyone = diagnostic_accuracy(12, 9, 0, 0)
    assert everyone.lr_positive is None and everyone.dor is None
    assert any(text.startswith("everyone is called positive") for text in everyone.warnings)
    # a side of the reference standard without participants: no diagnostic odds ratio (2.71 was computed)
    assert diagnostic_accuracy(0, 3, 0, 9).dor is None


@pytest.mark.parametrize("prevalence", [0.0, 1.0, -0.1, 1.5])
def test_a_prevalence_is_strictly_between_0_and_1(prevalence: float) -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        diagnostic_accuracy(90, 10, 20, 80, prevalence=prevalence)


def test_predictive_values_at_the_share_in_the_data_are_for_a_series_only() -> None:
    """One hundred with and one hundred without the condition, sampled apart: "89.5% at the share in these data"
    where the setting's value, at 2%, is 14.8%. Only the prevalence given reads it."""
    assert [reads_share(s) for s in SAMPLINGS] == [False, True, False]
    # a caller that names no sampling gets what every sampling supports, never the measures kept for a series
    unsaid = diagnostic_accuracy(85, 10, 15, 90)
    assert unsaid.sampling == SAMPLING_NOT_STATED and math.isnan(unsaid.ppv)
    series = diagnostic_accuracy(85, 10, 15, 90, sampling=SERIES)
    assert series.ppv == pytest.approx(0.8947, abs=5e-5) and series.accuracy == pytest.approx(0.875)
    for sampling in (TWO_GATE, SAMPLING_NOT_STATED):
        acc = diagnostic_accuracy(85, 10, 15, 90, sampling=sampling, prevalence=0.02)
        assert math.isnan(acc.ppv) and math.isnan(acc.npv) and math.isnan(acc.accuracy)
        assert math.isnan(acc.ppv_ci.lower) and acc.sampling == sampling
        assert acc.ppv_at_prevalence == pytest.approx(0.148, abs=5e-4)
        # what every sampling supports is unchanged
        assert (acc.sensitivity, acc.specificity, acc.lr_positive) == (
            series.sensitivity,
            series.specificity,
            series.lr_positive,
        )
    for unknown in ("cohort", ""):
        with pytest.raises(ValueError, match="unknown sampling"):
            diagnostic_accuracy(85, 10, 15, 90, sampling=unknown)
    # sampled apart, the spectrum of the participants is not the setting's: cited with the result
    apart = diagnostic_accuracy(85, 10, 15, 90, sampling=TWO_GATE)
    assert any(ref.startswith("Lijmer") for ref in apart.references)
    assert not any(ref.startswith("Lijmer") for ref in series.references)


# ----------------------------------------------------------------------------- the table of two columns


def _cohort(rows: int = 400, seed: int = 3) -> pd.DataFrame:
    """Made-up participants: a condition, a score that rises with it, and a yes/no test cut from the score at 52."""
    rng = np.random.default_rng(seed)
    ill = rng.random(rows) < 0.25
    score = np.where(ill, rng.normal(60, 15, rows), rng.normal(40, 15, rows))
    return pd.DataFrame(
        {
            "ill": np.where(ill, "Yes", "No"),
            "score": score,
            "rapid": np.where(score >= 52, "Positive", "Negative"),
            "reader": np.where(score >= 52, "Reactive", "Non-reactive"),
        }
    )


def test_the_table_of_a_yes_no_test_is_counted_by_its_two_values() -> None:
    df = _cohort()
    df.loc[:9, "rapid"] = None
    df.loc[10:14, "ill"] = None
    tp, fp, fn, tn = yes_no_table(df, "ill", "rapid", present="Yes", positive="Positive")
    crossed = pd.crosstab(df["rapid"], df["ill"])
    assert (tp, fp, fn, tn) == (
        crossed.loc["Positive", "Yes"],
        crossed.loc["Positive", "No"],
        crossed.loc["Negative", "Yes"],
        crossed.loc["Negative", "No"],
    )
    assert tp + fp + fn + tn == len(df) - 15  # the rows that hold both
    # the value read as positive decides the rows, never the order of the levels
    assert yes_no_table(df, "ill", "rapid", present="Yes", positive="Negative") == (fn, tn, tp, fp)
    three = df.assign(rapid=np.where(df.index % 7 == 0, "Indeterminate", df["rapid"]))
    with pytest.raises(ValueError, match="holds 3 values"):
        yes_no_table(three, "ill", "rapid", present="Yes", positive="Positive")


def test_a_yes_no_test_cut_from_a_score_gives_the_score_s_numbers_at_that_cut_off() -> None:
    """The same table three ways: the score at a cut-off typed beforehand and the yes/no column cut from it."""
    df = _cohort()
    yes_no = accuracy_yes_no_of(df, "ill", "Yes", "rapid", "Positive", 0.05, SERIES)
    score = accuracy_of(df, "ill", "Yes", "score", m.HIGHER, m.FIXED, 52.0, 0.05, SERIES)
    a, b = yes_no["acc"], score["acc"]
    assert (a.tp, a.fp, a.fn, a.tn) == (b.tp, b.fp, b.fn, b.tn)
    for name in ("sensitivity", "specificity", "ppv", "npv", "lr_positive", "lr_negative", "dor"):
        assert getattr(a, name) == getattr(b, name), name
    assert a.ppv_at_prevalence_ci == b.ppv_at_prevalence_ci
    # a yes/no test has one point: no curve, no area, no cut-off
    assert yes_no["auc"] is None and yes_no["fig"] is None and yes_no["cutoff"] is None
    assert (yes_no["index_form"], score["index_form"]) == (m.YES_NO, m.NUMBER)
    assert (yes_no["positive"], yes_no["negative"], yes_no["absent"]) == (
        "Positive",
        "Negative",
        "No",
    )


def test_the_reference_standard_done_unequally_by_test_result_is_said_with_its_numbers() -> None:
    """Every participant who tested positive and a fifth of those who tested negative have the reference standard:
    sensitivity reads too high (Begg & Greenes 1983). The table holds only the rows with both; the dataset knows
    the others."""
    called = pd.Series([True] * 100 + [False] * 400)
    reference = pd.Series(["Yes"] * 100 + ["No"] * 80 + [None] * 320)
    found = verification(called, reference)
    assert (
        found.positive,
        found.positive_with_reference,
        found.negative,
        found.negative_with_reference,
    ) == (100, 100, 400, 80)
    assert found.ratio == pytest.approx(5.0) and found.partial
    assert not found.one_sided and not found.may_be_chance
    # counts, not rounded shares; the measures that err with sensitivity and specificity, and those that do not
    assert m.verification_words(found, 100 / 180) == (
        "Those who tested positive had the reference standard 5.0 times as often as those who tested negative (100 "
        "of 100 against 80 of 400). If that followed the test result, sensitivity is overestimated and specificity "
        "underestimated (verification bias). The likelihood ratios and the predictive values at a prevalence given "
        "are read from sensitivity and specificity, and err with them. The predictive values among those who had "
        "the reference standard do not, if it followed the test result alone; their share with the condition "
        "(55.6%) is then not the prevalence."
    )
    alike = verification(called, pd.Series(["Yes"] * 90 + [None] * 10 + ["No"] * 350 + [None] * 50))
    assert not alike.partial and m.verification_words(alike) == ""
    # a test not recorded is no test result
    some = verification(
        pd.Series([True, None, False, False]), pd.Series(["Yes", "Yes", None, "No"])
    )
    assert (some.positive, some.negative, some.negative_with_reference) == (1, 2, 1)


# ----------------------------------------------------------------------------- the plan


def _text(found: m.ModelPlan) -> str:
    return " ".join([*(step.text for step in found.steps), *(item.text for item in found.advice)])


def test_the_plan_of_a_yes_no_test_asks_in_order_and_names_both_values() -> None:
    df = _cohort()
    ask = m.plan_accuracy_yes_no
    assert (
        _text(ask(df, None, None, None, None, None))
        == "Choose the reference standard and the test."
    )
    assert _text(ask(df, "ill", None, "rapid", None, None)) == (
        "Choose the value that means the condition is present."
    )
    assert _text(ask(df, "ill", "Yes", "rapid", None, None)) == (
        "Choose the value that means a positive test."
    )
    assert "two different columns" in _text(ask(df, "ill", "Yes", "ill", "Yes", None))
    assert _text(ask(df, "ill", "Yes", "rapid", "Positive", 1.0)) == m.PREVALENCE_BOUNDS
    ready = ask(df, "ill", "Yes", "rapid", "Positive", None, SERIES)
    assert ready.steps[0].text.startswith(
        "The accuracy of `rapid` (positive when `Positive`, negative when `Negative`) against `ill` = `Yes` ("
    )
    assert "a series, so that share is the prevalence of its setting" in _text(ready)
    assert "the test points the other way" not in _text(ready)
    # the value read as positive swapped: said before Run
    swapped = _text(ask(df, "ill", "Yes", "rapid", "Negative", None, SERIES))
    assert "Counted as a positive test, `rapid` = `Negative` is found in" in swapped
    assert "the test points the other way" in swapped
    # a test of one value in the rows used is no yes/no test
    one = df.assign(rapid="Positive")
    assert _text(ask(one, "ill", "Yes", "rapid", "Positive", None)) == (
        "`rapid` holds 1 value in the rows where the reference standard is recorded: a yes/no test holds two "
        "(positive or negative)."
    )


def test_the_plan_reads_the_predictive_values_by_the_sampling() -> None:
    df = _cohort()
    not_stated = _text(m.plan_accuracy_yes_no(df, "ill", "Yes", "rapid", "Positive", None))
    assert (
        "No predictive values yet: if the participants were taken as they came (a series), say so to read them at "
        "the share in these data (24.0%); otherwise give the prevalence where the test will be used."
    ) in not_stated
    # the step promises what Run gives: no predictive value without a series or a prevalence
    assert "Sensitivity, specificity and likelihood ratios from the 2x2 table" in not_stated
    apart = _text(m.plan_accuracy_yes_no(df, "ill", "Yes", "rapid", "Positive", None, TWO_GATE))
    assert "was set by the sampling, and predictive values are not read at it" in apart
    assert m.TWO_GATE_SPECTRUM in apart and "Lijmer et al. 1999" in m.TWO_GATE_SPECTRUM
    typed = _text(m.plan_accuracy_yes_no(df, "ill", "Yes", "rapid", "Positive", 0.02, TWO_GATE))
    assert "predictive values are read at the prevalence typed (2%)" in typed
    assert "Sensitivity, specificity, predictive values and likelihood ratios" in typed
    # a prevalence typed is written with the figures it has: 1 in 2,500 read "0.0%"
    rare = _text(m.plan_accuracy_yes_no(df, "ill", "Yes", "rapid", "Positive", 0.0004, TWO_GATE))
    assert "the prevalence typed (0.04%)" in rare
    # the score's plan says the same, and a prevalence of 1 waits there too
    score = _text(m.plan_accuracy(df, "ill", "Yes", "score", m.HIGHER, m.FIXED, 52.0, None))
    assert "No predictive values yet" in score
    assert _text(m.plan_accuracy(df, "ill", "Yes", "score", m.HIGHER, m.FIXED, 52.0, 1.0)) == (
        m.PREVALENCE_BOUNDS
    )
    assert (
        set(m.SAMPLING_LABELS)
        == set(m.SAMPLING_SENTENCE)
        == set(m.SAMPLING_SETTING)
        == set(SAMPLINGS)
    )
    assert set(SAMPLING_WORDS) == set(SAMPLINGS)


def test_the_plan_says_when_the_reference_standard_followed_the_test() -> None:
    df = _cohort()
    rng = np.random.default_rng(5)
    df.loc[(df["rapid"] == "Negative") & (rng.random(len(df)) < 0.6), "ill"] = None
    said = _text(m.plan_accuracy_yes_no(df, "ill", "Yes", "rapid", "Positive", None, SERIES))
    assert "Those who tested positive had the reference standard" in said
    assert "verification bias" in said
    # at a cut-off typed beforehand, the score's plan says it too
    score = _text(m.plan_accuracy(df, "ill", "Yes", "score", m.HIGHER, m.FIXED, 52.0, None, SERIES))
    assert "verification bias" in score
    result = accuracy_yes_no_of(df, "ill", "Yes", "rapid", "Positive", None, SERIES)
    assert "verification bias" in result["verification"] and result["verified"].partial


# ----------------------------------------------------------------------------- the dossier


def _table(tp: int, fp: int, fn: int, tn: int) -> pd.DataFrame:
    rows = (
        [("Yes", "Positive")] * tp
        + [("No", "Positive")] * fp
        + [("Yes", "Negative")] * fn
        + [("No", "Negative")] * tn
    )
    return pd.DataFrame(rows, columns=["ill", "rapid"])


def _written(value: dict) -> str:
    blocks = result_blocks(value)
    tables = " ".join(f"{title} {table.to_string()}" for title, table in blocks.tables)
    return " ".join([tables, *blocks.lines, *blocks.warnings])


def _yes_no(
    cells: tuple[int, int, int, int], sampling: str = SERIES, prevalence: float | None = None
):
    return accuracy_yes_no_of(
        _table(*cells), "ill", "Yes", "rapid", "Positive", prevalence, sampling
    )


def test_the_dossier_writes_a_yes_no_test_s_table_with_its_measures() -> None:
    """Thirty with the condition among 928: the table and every measure, the cell of 5 exactly; no AUC, which a
    yes/no test has none of (the block read the AUC first and raised, and the whole Word download failed)."""
    blocks = result_blocks(_yes_no((25, 40, 5, 858)))
    (title, measures), (_, counts) = blocks.tables
    assert title == "rapid = Positive (positive) for ill = Yes"
    assert list(measures["Measure"])[:2] == ["Sensitivity", "Specificity"]
    assert "Area under the ROC curve (AUC)" not in list(measures["Measure"])
    assert "Diagnostic odds ratio" in list(measures["Measure"])
    assert counts.iloc[1].tolist() == ["Test negative (Negative)", 5, 858, 863]
    assert list(counts.columns) == ["", "Condition present (Yes)", "Condition absent (No)", "Total"]
    assert blocks.lines == [sampling_line(SERIES)]


@pytest.mark.parametrize(
    ("cells", "side"),
    [
        (
            (2, 0, 1, 397),
            "have the condition",
        ),  # three with the condition: "sensitivity 0.667" says which
        ((290, 5, 0, 4), "are without the condition"),  # nine without it
        ((5, 30, 4, 300), "have the condition"),  # nine
    ],
)
def test_a_small_side_of_the_reference_standard_holds_the_table_and_the_measures_back(
    cells: tuple[int, int, int, int], side: str
) -> None:
    """One rule for a side of the reference standard, whatever the index test: a yes/no column cut from a score
    would otherwise write what the score's block holds back. The sides are read on the table."""
    blocks = result_blocks(_yes_no(cells))
    assert not blocks.tables and not blocks.warnings
    assert blocks.lines == [
        "rapid = Positive (positive) for ill = Yes",
        f"Fewer than 10 participants {side}: the table and the measures of the test read them one by one, and are "
        "not written in the dossier; the page shows them.",
    ]
    said = _written(_yes_no(cells))
    assert "Sensitivity" not in said and "0.667" not in said and "2 of 3" not in said


def test_ten_on_each_side_are_written_and_the_side_the_test_calls_does_not_count() -> None:
    """A yes/no test's positive value is a code of the data, not a score found on them: a test that calls 2
    participants positive is written, as a cut-off typed beforehand is."""
    blocks = result_blocks(_yes_no((2, 0, 8, 390)))
    assert len(blocks.tables) == 2 and "Sensitivity" in list(blocks.tables[0][1]["Measure"])
    # at the edge of a cell: the one bound, in words
    measures = blocks.tables[0][1].set_index("Measure")
    assert measures.loc["Positive likelihood ratio (LR+)", "Estimate"] == "no upper bound"
    assert measures.loc["Positive likelihood ratio (LR+)", "95% CI"].startswith("at least ")


def test_the_dossier_writes_predictive_values_by_the_sampling() -> None:
    series = list(result_blocks(_yes_no((25, 40, 5, 858), SERIES)).tables[0][1]["Measure"])
    assert any(
        name.startswith("Positive predictive value, at the share in these data") for name in series
    )
    for sampling in (TWO_GATE, SAMPLING_NOT_STATED):
        blocks = result_blocks(_yes_no((25, 40, 5, 858), sampling, 0.02))
        names = list(blocks.tables[0][1]["Measure"])
        assert not any("at the share in these data" in name for name in names), sampling
        assert "Positive predictive value, at a prevalence of 2%" in names
        at = (
            blocks.tables[0][1]
            .set_index("Measure")
            .loc["Positive predictive value, at a prevalence of 2%"]
        )
        assert at["95% CI"]  # the likelihood ratio's interval, carried
        assert blocks.lines == [sampling_line(sampling, 0.02)]


def test_a_score_and_the_yes_no_column_cut_from_it_write_the_same_measures_or_none() -> None:
    df = _cohort()
    for frame in (df, df[df["ill"].eq("No") | (df.index < 12)]):  # then few with the condition
        yes_no = result_blocks(
            accuracy_yes_no_of(frame, "ill", "Yes", "rapid", "Positive", None, SERIES)
        )
        score = result_blocks(
            accuracy_of(frame, "ill", "Yes", "score", m.HIGHER, m.FIXED, 52.0, None, SERIES)
        )
        mine = yes_no.tables[0][1].set_index("Measure") if yes_no.tables else None
        theirs = score.tables[0][1].set_index("Measure")
        if mine is None:
            assert list(theirs.index) == ["Area under the ROC curve (AUC)"]
        else:
            pd.testing.assert_frame_equal(mine, theirs.drop(index="Area under the ROC curve (AUC)"))


# ----------------------------------------------------------------------------- the record


def test_the_record_of_a_yes_no_test_holds_its_table_and_what_its_sampling_supports() -> None:
    record = plan.record("diagnostic", _yes_no((25, 40, 5, 858), SERIES, 0.02))
    assert record is not None and record.kind == "diagnostic"
    parts = {part.id: part for part in record.parts}
    assert list(parts) == [
        "table",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "accuracy",
        "ppv_at",
        "npv_at",
        "lr_positive",
        "lr_negative",
        "dor",
        "youden",
    ]
    table = parts["table"]
    assert {n.name: n.value for n in table.numbers} == {
        "tp": 25,
        "fp": 40,
        "fn": 5,
        "tn": 858,
        "index_form": "yes_no",
        "sampling": "series",
    }
    assert all(n.cls == plan.EXACT for n in table.numbers)
    # the four cells behind every part: what a shared package's rule reads
    assert all(part.support["fn"] == 5 and part.support["rows"] == 928 for part in record.parts)
    assert table.labels == {
        "reference": "ill",
        "index": "rapid",
        "present": "Yes",
        "positive": "Positive",
    }
    assert not [n for part in record.parts for n in part.numbers if n.fine]
    assert plan.ResultRecord.from_dict(json.loads(json.dumps(record.to_dict()))).to_dict() == (
        record.to_dict()
    )
    apart = plan.record("diagnostic", _yes_no((25, 40, 5, 858), TWO_GATE))
    assert apart is not None
    assert not {"ppv", "npv", "accuracy", "ppv_at"} & {part.id for part in apart.parts}


def test_the_record_of_a_score_marks_a_found_cut_off_as_one_participant_s() -> None:
    df = _cohort()
    found = plan.record(
        "diagnostic", accuracy_of(df, "ill", "Yes", "score", m.HIGHER, m.YOUDEN, None, None, SERIES)
    )
    assert found is not None
    cutoff = next(part for part in found.parts if part.id == "cutoff")
    assert [n.name for n in cutoff.numbers if n.fine] == ["value", "range_lower", "range_upper"]
    typed = plan.record(
        "diagnostic", accuracy_of(df, "ill", "Yes", "score", m.HIGHER, m.FIXED, 52.0, None, SERIES)
    )
    assert typed is not None
    assert not [n for p in typed.parts if p.id == "cutoff" for n in p.numbers if n.fine]
    assert [part.id for part in found.parts][:2] == ["auc", "table"]
    alone = plan.record(
        "diagnostic", accuracy_of(df, "ill", "Yes", "score", m.HIGHER, m.NO_CUTOFF, None, None)
    )
    assert alone is not None and [part.id for part in alone.parts] == ["auc"]
    # a zero cell: the ratio without an upper bound is written with its marker and reads back
    edge = plan.record("diagnostic", _yes_no((30, 0, 0, 60), SAMPLING_NOT_STATED))
    assert edge is not None
    ratio = next(part for part in edge.parts if part.id == "lr_positive")
    assert ratio.number("estimate").value == math.inf and ratio.number("method").value == KOOPMAN
    assert '{"float":"inf"}' in edge.to_json()


# ----------------------------------------------------------------------------- the reference standard done unequally


def _programme() -> pd.DataFrame:
    """A made-up screening programme: every participant who screened positive had the biopsy, none of the others."""
    rng = np.random.default_rng(11)
    rows = 5000
    value = rng.normal(50, 10, rows)
    screen = np.where(value >= 59, "Positive", "Negative")
    biopsy = np.where(rng.random(rows) < np.where(value >= 59, 0.18, 0.02), "Yes", "No").astype(
        object
    )
    biopsy[screen == "Negative"] = None
    return pd.DataFrame({"biopsy": biopsy, "screen": screen, "value": value})


def test_a_reference_standard_done_after_one_test_result_only_is_said_on_both_paths() -> None:
    """Biopsies follow a positive screen only. The check said nothing (a ratio without a value), the yes/no plan
    blamed the test's column ("holds 1 value: a yes/no test holds two"), and the score's plan asked "the right
    unit?" beside a share called the prevalence of the setting."""
    df = _programme()
    positives = int((df["screen"] == "Positive").sum())
    found = verification(df["screen"] == "Positive", df["biopsy"])
    assert found.ratio == math.inf and found.partial and found.one_sided
    words = (
        f"None of the {len(df) - positives:,} who tested negative had the reference standard, against "
        f"{positives:,} of {positives:,} who tested positive: sensitivity and specificity cannot be read from these "
        "data (verification bias). Among those who tested positive, the share with the condition (the positive "
        "predictive value) can."
    )
    assert m.verification_words(found) == words
    # with the count of those with the condition, the one predictive value these data give is written as a count
    ill = int((df["biopsy"] == "Yes").sum())
    words = words.replace(
        "the share with the condition (the positive predictive value) can.",
        f"{ill:,} of {positives:,} had the condition: that share (the positive predictive value) can be read.",
    )
    assert m.verification_words(found, with_condition=ill) == words
    refused = m.plan_accuracy_yes_no(df, "biopsy", "Yes", "screen", "Positive", None, SERIES)
    assert not refused.steps and _text(refused) == words
    # before the positive value is chosen, the column is not blamed either
    unchosen = _text(m.plan_accuracy_yes_no(df, "biopsy", "Yes", "screen", None, None, SERIES))
    assert (
        "`screen` holds two values, and one only in the rows where the reference standard is recorded"
        in unchosen
    )
    score = _text(
        m.plan_accuracy(df, "biopsy", "Yes", "value", m.HIGHER, m.FIXED, 59.0, None, SERIES)
    )
    assert words in score and "The right variable, the right unit?" not in score
    assert "the prevalence of its setting" not in score
    assert "; that share is not the prevalence of the setting." in score
    result = accuracy_of(df, "biopsy", "Yes", "value", m.HIGHER, m.FIXED, 59.0, None, SERIES)
    assert result["verification"] == words
    # mirrored: no one who tested positive had it
    mirrored = verification(
        pd.Series([True] * 30 + [False] * 70), pd.Series([None] * 30 + ["No"] * 70)
    )
    assert mirrored.ratio == 0 and mirrored.one_sided
    assert m.verification_words(mirrored).startswith(
        "None of the 30 who tested positive had the reference standard, against 70 of 70 who tested negative: "
        "sensitivity and specificity cannot be read from these data (verification bias)."
    )
    assert m.verification_words(mirrored, with_condition=4).endswith(
        "Among those who tested negative, 66 of 70 did not have the condition: that share (the negative "
        "predictive value) can be read."
    )
    # nobody on a side, or nobody with the reference standard: nothing to compare
    assert not verification(pd.Series([True] * 5), pd.Series(["Yes"] * 5)).partial
    assert not verification(pd.Series([True, False]), pd.Series([None, None])).partial


def _partly_verified() -> pd.DataFrame:
    """A made-up series in which four fifths of those who tested negative lack the reference standard."""
    df = _cohort(600)
    rng = np.random.default_rng(5)
    df.loc[(df["rapid"] == "Negative") & (rng.random(len(df)) < 0.8), "ill"] = None
    return df


def test_under_partial_verification_a_series_does_not_call_its_share_a_prevalence() -> None:
    """The share among those who had the reference standard was called "the prevalence of its setting" directly
    under the caution of partial verification; the caution named sensitivity and specificity only, where the
    likelihood ratios and the predictive values at a prevalence given err with them (Begg & Greenes 1983)."""
    df = _partly_verified()
    said = _text(m.plan_accuracy_yes_no(df, "ill", "Yes", "rapid", "Positive", None, SERIES))
    assert "Predictive values are read among those who had the reference standard (" in said
    assert "so that share is the prevalence of its setting" not in said
    # the caution says that this share is no prevalence, once: the line under it does not repeat it
    assert (
        said.count("is then not the prevalence") == 1
        and "that share is not the prevalence" not in said
    )
    assert (
        "The likelihood ratios and the predictive values at a prevalence given are read from sensitivity and "
        "specificity, and err with them. The predictive values among those who had the reference standard do not, "
        "if it followed the test result alone"
    ) in said
    result = accuracy_yes_no_of(df, "ill", "Yes", "rapid", "Positive", None, SERIES)
    caption = page._predictive_caption(result["acc"], None, page._unequal(result))
    assert caption.startswith("Predictive values among those who had the reference standard (")
    assert "a series" not in caption
    # one pair of numbers, one name: the rows of the page and of the dossier, and the dossier's line
    rows = list(page._measure_rows(result["acc"], None, page._unequal(result))["Measure"])
    assert any(
        name.startswith("Positive predictive value, among those who had the reference standard (")
        for name in rows
    ) and not any("at the share in these data" in name for name in rows)
    blocks = result_blocks(result)
    names = list(blocks.tables[0][1]["Measure"])
    assert any(
        name.startswith("Positive predictive value, among those who had the reference standard (")
        for name in names
    )
    assert not any("at the share in these data" in name for name in names)
    assert blocks.lines[-1].endswith(
        "predictive values are read among those who had the reference standard, whose share with the condition "
        "is not the prevalence."
    )
    # the caution's reference goes with it, on the page and in the dossier
    assert "Begg CB" in page._methods(result)
    assert any(ref.startswith("Begg CB") for ref in result_blocks(result).references)
    whole = accuracy_yes_no_of(_cohort(600), "ill", "Yes", "rapid", "Positive", None, SERIES)
    assert not whole["verification"] and "Begg CB" not in page._methods(whole)
    assert not any(ref.startswith("Begg CB") for ref in result_blocks(whole).references)
    assert page._predictive_caption(whole["acc"], None, page._unequal(whole)).startswith(
        "Predictive values at the share with the condition in these participants ("
    )


def test_the_caution_writes_counts_and_says_when_it_may_be_chance() -> None:
    """A reference standard missing at random fired the caution in half the samples of 20 + 80; "100% against 0%"
    stood for 250 of 251 against 1 of 1,000, and "999.0 times as often"."""
    small = verification(
        pd.Series([True] * 20 + [False] * 80),
        pd.Series(["Yes"] * 8 + [None] * 12 + ["No"] * 20 + [None] * 60),
    )
    assert small.ratio == pytest.approx(1.6) and small.partial and small.may_be_chance
    said = m.verification_words(small)
    assert "1.6 times as often" in said and "(8 of 20 against 20 of 80)" in said
    assert said.endswith("With so few participants this may be chance.")
    large = verification(
        pd.Series([True] * 251 + [False] * 4000),
        pd.Series(["Yes"] * 250 + [None] + ["No"] * 20 + [None] * 3980),
    )
    said = m.verification_words(large)
    assert "more than 10 times as often" in said and "(250 of 251 against 20 of 4,000)" in said
    assert not large.may_be_chance and "chance" not in said
    # the other direction errs the other way
    other = verification(
        pd.Series([True] * 100 + [False] * 400),
        pd.Series(["Yes"] * 20 + [None] * 80 + ["No"] * 400),
    )
    assert m.verification_words(other).startswith(
        "Those who tested negative had the reference standard 5.0 times as often as those who tested positive (400 "
        "of 400 against 20 of 100). If that followed the test result, sensitivity is underestimated and "
        "specificity overestimated (verification bias)."
    )


def test_the_record_s_verification_part_counts_those_without_the_reference_standard() -> None:
    """50 tested positive, 48 of them with the reference standard: each count is 10 or more, and their difference
    says that 2 participants tested positive without it. The support holds that count too."""
    rows = (
        [("Yes", "Positive")] * 30
        + [("No", "Positive")] * 18
        + [(None, "Positive")] * 2
        + [("Yes", "Negative")] * 20
        + [("No", "Negative")] * 180
        + [(None, "Negative")] * 150
    )
    df = pd.DataFrame(rows, columns=["ill", "rapid"])
    record = plan.record(
        "diagnostic", accuracy_yes_no_of(df, "ill", "Yes", "rapid", "Positive", None, SERIES)
    )
    assert record is not None
    part = next(part for part in record.parts if part.id == "verification")
    counts = {
        "positive": 50,
        "positive_with_reference": 48,
        "negative": 350,
        "negative_with_reference": 200,
    }
    assert {n.name: n.value for n in part.numbers} == counts
    assert dict(part.support) == {
        **counts,
        "positive_without_reference": 2,
        "negative_without_reference": 150,
    }
    # every row with the reference standard: no such part
    whole = plan.record("diagnostic", _yes_no((25, 40, 5, 858)))
    assert whole is not None and "verification" not in {part.id for part in whole.parts}


# ----------------------------------------------------------------------------- at the edges of what is written


def test_a_test_result_nobody_has_leaves_no_predictive_value_at_a_prevalence_on_any_surface() -> (
    None
):
    """Everyone called positive, at a prevalence of 10%: "10.0%" was written without an interval, as if exact (the
    likelihood ratio of 1 that the same function withholds, carried through Bayes' rule); the caption read "negative
    n/a" and the dossier "n/a to n/a"."""
    result = accuracy_of(_cohort(), "ill", "Yes", "score", m.HIGHER, m.FIXED, -1000.0, 0.10, SERIES)
    acc = result["acc"]
    assert (acc.fn, acc.tn) == (0, 0)
    assert acc.ppv_at_prevalence is None and acc.npv_at_prevalence is None
    assert not any("at a prevalence" in name for name in page._measure_rows(acc, 0.10)["Measure"])
    caption = page._predictive_caption(acc, 0.10)
    assert caption.endswith("no negative predictive value: no one tested negative.")
    assert "At a prevalence" not in caption and "n/a" not in caption
    assert "Mercaldo" not in page._intervals(acc, 0.10)
    written = _written(result)
    assert "at a prevalence of" not in written and "n/a to n/a" not in written
    assert "at the prevalence given" not in written
    record = plan.record("diagnostic", result)
    assert record is not None
    assert not {"ppv_at", "npv_at"} & {part.id for part in record.parts}


def test_the_sampling_line_of_the_dossier_says_what_was_written() -> None:
    assert sampling_line(SERIES) == (
        "The participants entered the study as a series, taken before the condition was known (consecutive or "
        "random): predictive values are read at their share with the condition."
    )
    assert sampling_line(SERIES, 0.02).endswith(
        "at their share with the condition, and at the prevalence given (2%)."
    )
    assert sampling_line(SAMPLING_NOT_STATED) == (
        "How the participants entered the study is not stated: no predictive value is written, none was asked at a "
        "prevalence."
    )
    assert sampling_line(SAMPLING_NOT_STATED, 0.0004) == (
        "How the participants entered the study is not stated: predictive values are read at the prevalence given "
        "(0.04%) only."
    )
    # sampled apart: the spectrum caveat, whatever is read (Lijmer et al. 1999)
    caveat = (
        " Sensitivity and specificity may read higher than where the test will be used: sampled apart, those with "
        "the condition are often clearer cases, and those without healthier."
    )
    assert sampling_line(TWO_GATE, 0.02) == (
        "Participants with and without the condition were sampled apart (two-gate): predictive values are read at "
        "the prevalence given (2%) only." + caveat
    )
    assert sampling_line(TWO_GATE).endswith("none was asked at a prevalence." + caveat)
    # a prevalence asked whose predictive values have no value: not "none was asked"
    assert sampling_line(SAMPLING_NOT_STATED, 0.02, read=False) == (
        "How the participants entered the study is not stated: no predictive value is written."
    )


def test_a_share_is_written_with_the_figures_that_tell_it_from_all_or_none() -> None:
    """At a prevalence of 1 in 2,500 the page read "At a prevalence of 0.0%" and "negative 100.0% (95% CI 100.0% to
    100.0%)"."""
    acc = diagnostic_accuracy(85, 10, 15, 90, prevalence=0.0004, sampling=TWO_GATE)
    caption = page._predictive_caption(acc, 0.0004)
    assert caption.startswith(
        "At a prevalence of 0.04%: positive 0.34% (95% CI 0.19% to 0.61%), negative 99.9933% (95% CI 99.9893% "
        "to 99.9958%), the prevalence taken as known."
    )
    rows = page._measure_rows(acc, 0.0004).set_index("Measure")
    assert rows.loc["Negative predictive value, at a prevalence of 0.04%"].tolist() == [
        "99.9933%",
        "99.9893% to 99.9958%",
    ]
    # without an interval: the decimals that show the distance to all or to none ("more than 99.9%" stood for
    # 2,499 of 2,500), and past six decimals never "100%" for less than all, nor "0%" for some
    assert page._share(0.9996, None) == "99.960%" and page._share(0.0004, None) == "0.040%"
    assert page._share(1 - 1e-9, None) == "more than 99.999999%"
    assert page._share(1e-9, None) == "less than 0.000001%"
    assert page._share(1.0, None) == "100.0%" and page._share(0.0, None) == "0.0%"
    assert page._share(None, None) == "n/a"
    assert m.prevalence_text(0.0004) == "0.04%" and m.prevalence_text(0.109) == "10.9%"


def test_a_zero_cell_s_bound_is_written_one_way_and_its_method_named() -> None:
    """The caution said "it may be as large as 0.14" over the table's "at most 0.142", and the methods line named
    Mercaldo's interval where the score bound was carried."""
    acc = diagnostic_accuracy(30, 0, 6, 60, prevalence=0.02, sampling=SERIES)
    rows = page._measure_rows(acc, 0.02).set_index("Measure")
    assert rows.loc["Positive likelihood ratio (LR+)"].tolist() == [
        "no upper bound",
        "at least 13.8",
    ]
    assert any("it is at least 13.8 (95% interval)" in text for text in acc.warnings)
    assert acc.ppv_at_prevalence_ci is not None and acc.ppv_at_prevalence_ci.method == SCORE_CARRIED
    assert acc.npv_at_prevalence_ci is not None and acc.npv_at_prevalence_ci.method == MERCALDO
    line = page._intervals(acc, 0.02)
    assert (
        "(Mercaldo's logit interval; from the score bound where a likelihood ratio is 0 or has no upper bound)"
        in line
    )
    assert "the one bound of Koopman's score interval at a zero cell" in line
    usual = page._intervals(diagnostic_accuracy(85, 10, 15, 90, prevalence=0.02), 0.02)
    assert "(Mercaldo's logit interval)" in usual and "Koopman" not in usual
    other = diagnostic_accuracy(30, 6, 0, 24)
    rows = page._measure_rows(other, None).set_index("Measure")
    bound = rows.loc["Negative likelihood ratio (LR-)", "95% CI"]
    assert bound.startswith("at most ") and any(
        bound.removeprefix("at most ") in text for text in other.warnings
    )


def test_the_methods_a_record_can_hold_are_these_words() -> None:
    """The record holds the method of each interval by its words: a rewording changes what a record says of the same
    numbers, and moves the record's version."""
    methods = (DELONG, WILSON, SIMEL, KOOPMAN, WOOLF, WOOLF_CORRECTED, MERCALDO, SCORE_CARRIED)
    assert methods == (
        "DeLong",
        "Wilson",
        "Simel log",
        "Koopman score",
        "Woolf log",
        "Woolf log, Haldane-Anscombe 0.5 added",
        "logit (Mercaldo)",
        "from the score bound of the likelihood ratio",
    )
    held: set[str] = set()
    results = [
        _yes_no(cells, SERIES, 0.02) for cells in ((25, 40, 5, 858), (30, 0, 6, 60), (30, 6, 0, 24))
    ]
    # a score's record holds the method of its AUC's interval too
    results.append(accuracy_of(_cohort(), "ill", "Yes", "score", m.HIGHER, m.NO_CUTOFF, None, None))
    for result in results:
        record = plan.record("diagnostic", result)
        assert record is not None
        held |= {str(n.value) for part in record.parts for n in part.numbers if n.name == "method"}
    assert held == set(methods)


# ----------------------------------------------------------------------------- the settings of a result


def _settings_grid() -> list[dict]:
    grid = []
    for sampling in SAMPLINGS:
        for prevalence in (None, 0.02, 0.0200001):
            grid.append(
                accuracy_settings(
                    m.YES_NO, "Yes", positive="Positive", prevalence=prevalence, sampling=sampling
                )
            )
            for mode in CUTOFF_MODES:
                for cutoff in (None, 52.0):
                    grid.append(
                        accuracy_settings(
                            m.NUMBER,
                            "Yes",
                            direction=m.HIGHER,
                            mode=mode,
                            cutoff=cutoff,
                            prevalence=prevalence,
                            sampling=sampling,
                        )
                    )
    return grid


def test_the_settings_of_a_result_are_declared_options_of_its_kind() -> None:
    """The identity of a result on the page is built from the options of its request, by stored id."""
    options = plan.KINDS["diagnostic"].options
    for settings in _settings_grid():
        assert set(settings) <= set(options), settings
        # and their values are among the closed ones: the page's settings make a spec the kind accepts
        made = plan.specify("diagnostic", options=settings, reference="ill", index="test")
        assert {name: made.options[name] for name in settings} == settings
    # the ids the page, the record and the dossier read are the engine's, and the kind's defaults among them
    assert (m.NUMBER, m.YES_NO) == INDEX_FORMS and set(m.TEST_RESULTS) == set(INDEX_FORMS)
    assert (m.NO_CUTOFF, m.FIXED, m.YOUDEN) == CUTOFF_MODES and set(m.CUTOFFS) == set(CUTOFF_MODES)
    assert (m.HIGHER, m.LOWER) == DIRECTIONS
    assert options["index_form"] == m.NUMBER and options["sampling"] == SAMPLING_NOT_STATED
    assert options["sampling"] in SAMPLINGS and options["mode"] is None


def test_two_results_share_their_settings_line_exactly_when_their_settings_are_the_same() -> None:
    grid = _settings_grid()
    for one in grid:
        for other in grid:
            same = page._config(("ill", "test"), one) == page._config(("ill", "test"), other)
            assert same == (one == other), (one, other)
    # a score without a cut-off reads no table: one analysis whatever the sampling question holds
    alone = [
        accuracy_settings(
            m.NUMBER,
            "Yes",
            direction=m.HIGHER,
            mode=m.NO_CUTOFF,
            sampling=sampling,
            prevalence=share,
        )
        for sampling in SAMPLINGS
        for share in (None, 0.3)
    ]
    assert all(settings == alone[0] for settings in alone) and "sampling" not in alone[0]
    # a cut-off typed for another mode is no part of the result
    assert "cutoff" not in accuracy_settings(
        m.NUMBER, "Yes", direction=m.HIGHER, mode=m.YOUDEN, cutoff=52.0
    )
    assert page._config(
        ("ill", "score"),
        accuracy_settings(
            m.NUMBER, "Yes", direction=m.HIGHER, mode=m.FIXED, cutoff=52.0, sampling=SERIES
        ),
    ) == (
        "ill",
        "score",
        "Yes",
        "a score",
        "higher",
        "A value fixed beforehand",
        52.0,
        None,
        "a series of participants",
    )


def test_the_words_of_a_setting_tell_its_ids_apart() -> None:
    for words in (
        m.SAMPLING_SETTING,
        m.SAMPLING_LABELS,
        m.SAMPLING_SENTENCE,
        m.CUTOFFS,
        m.TEST_RESULTS,
        page.FORM_SETTING,
        SAMPLING_WORDS,
    ):
        assert len(set(words.values())) == len(words), words
    assert set(page.FORM_SETTING) == set(INDEX_FORMS)
    # two prevalences a person can type are two results
    assert page._prevalence_setting(0.1234561) != page._prevalence_setting(0.1234564)
    assert page._prevalence_setting(0.02) == "at a prevalence of 0.02"
    assert page._prevalence_setting(None) is None


# ----------------------------------------------------------------------------- what the dossier holds back, or writes


def test_a_held_back_block_writes_no_caution_that_gives_its_shares() -> None:
    """One of three participants with the condition tested positive, and 200 who tested negative lack the reference
    standard: the result keeps the caution of a test that points the other way ("33.3% of the participants with
    the condition") and the note of partial verification; the dossier writes neither beside a side too small."""
    rows = (
        [("Yes", "Positive")] * 1
        + [("No", "Positive")] * 300
        + [("Yes", "Negative")] * 2
        + [("No", "Negative")] * 97
        + [(None, "Negative")] * 200
    )
    df = pd.DataFrame(rows, columns=["ill", "rapid"])
    result = accuracy_yes_no_of(df, "ill", "Yes", "rapid", "Positive", None, SERIES)
    assert any("33.3%" in text for text in result["cautions"])
    assert "had the reference standard" in result["verification"]
    blocks = result_blocks(result)
    said = " ".join([*blocks.lines, *blocks.warnings])
    assert not blocks.tables and "33.3%" not in said and "had the reference standard" not in said
    assert not any(ref.startswith("Begg CB") for ref in blocks.references)


def test_ten_without_the_condition_are_written_and_nine_are_held_back() -> None:
    assert len(result_blocks(_yes_no((300, 1, 50, 9))).tables) == 2
    assert not result_blocks(_yes_no((300, 0, 50, 9))).tables


def _session_parts(*values: dict) -> dict:
    """The parts of a made-up session, as ``session_parts`` gives them, with these results kept."""
    from types import SimpleNamespace

    from core.attrition import FlowBox, ParticipantFlow
    from ui.results import CURRENT

    kept = [
        (
            SimpleNamespace(
                key="diagnostic",
                value=value,
                title="Diagnostic accuracy",
                settings="ill, rapid",
                stamp=SimpleNamespace(caption="computed on made-up rows", single_imputed=()),
            ),
            CURRENT,
        )
        for value in values
    ]
    return {
        "flow": ParticipantFlow(main=(FlowBox("loaded", "Loaded: made-up", 928, "loaded"),)),
        "kept": kept,
        "omitted": [],
        "population_label": "all 928 rows",
        "provenance": [("Dataset", "made-up")],
        "criteria": None,
        "data_steps": [],
        "missing": None,
        "single_imputed": {},
        "exact_imputed": {},
        "mi": None,
    }


def _document_text(buffer: object) -> str:
    from docx import Document

    doc = Document(buffer)
    cells = [cell.text for table in doc.tables for row in table.rows for cell in row.cells]
    return "\n".join([*(paragraph.text for paragraph in doc.paragraphs), *cells])


def test_one_result_that_fails_does_not_stop_the_dossier(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A defect in one result's block, or in its writing, stopped the download of every section. The fixed line
    stands in its place; the log holds the stored key and the exception's type, never its message."""
    import manage.report_tables as report_tables
    from app_pages import study_dossier
    from manage.dossier import NOT_WRITTEN, build_dossier
    from manage.report_tables import ResultBlocks

    real = report_tables.result_blocks

    def broken(value: dict) -> object:
        if value.get("broken"):
            raise KeyError("a value of the data: Luxembourg")
        return real(value)

    monkeypatch.setattr("manage.report_tables.result_blocks", broken)
    monkeypatch.setattr("manage.flow_figure.flow_png", lambda flow: None)
    parts = _session_parts({"broken": True}, _yes_no((25, 40, 5, 858)))
    with caplog.at_level("WARNING"):
        inp = study_dossier.finish_input(parts, title="A made-up study", author="a")
    assert inp.results[0].blocks.lines == [NOT_WRITTEN] and not inp.results[0].blocks.tables
    assert len(inp.results[1].blocks.tables) == 2
    assert "the block of diagnostic could not be written in the dossier: KeyError" in caplog.text
    assert "Luxembourg" not in caplog.text
    text = _document_text(build_dossier(inp))
    assert text.count(NOT_WRITTEN) == 1 and "Sensitivity" in text
    # a block that builds and cannot be written: the line follows what was written, the next result is whole
    inp.results[0].blocks = ResultBlocks(tables=[("a table that is none", None)])  # type: ignore[list-item]
    caplog.clear()
    with caplog.at_level("WARNING"):
        text = _document_text(build_dossier(inp))
    assert text.count(NOT_WRITTEN) == 1 and "Sensitivity" in text and "7. References" in text
    assert "result 1 of the dossier could not be written: AttributeError" in caplog.text


def test_a_character_a_word_document_refuses_does_not_stop_the_dossier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A positive value holding a vertical tab (a manual line break pasted from a word processor; fixed-width
    exports carry NUL characters): the block was built, then the document refused the character in the settings
    line, the title and the 2x2 table, and the download of every section failed."""
    from app_pages import study_dossier
    from manage.dossier import NOT_WRITTEN, build_dossier

    tab, shown = chr(11), chr(0xFFFD)
    frame = _table(25, 40, 5, 858).replace({"rapid": {"Positive": f"Pos{tab}itive"}})
    result = accuracy_yes_no_of(frame, "ill", "Yes", "rapid", f"Pos{tab}itive", None, SERIES)
    monkeypatch.setattr("manage.flow_figure.flow_png", lambda flow: None)
    parts = _session_parts(result)
    parts["kept"][0][0].settings = f"ill, rapid, Yes, positive when Pos{tab}itive{chr(0)}"
    inp = study_dossier.finish_input(parts, title=f"A made-up{tab} study", author="a")
    text = _document_text(build_dossier(inp))
    assert tab not in text and chr(0) not in text and NOT_WRITTEN not in text
    assert (
        f"Test positive (Pos{shown}itive)" in text
        and f"positive when Pos{shown}itive{shown}" in text
    )
    assert "Sensitivity" in text and f"A made-up{shown} study" in text


# ----------------------------------------------------------------------------- what the second reads found


def test_a_cut_off_found_on_the_data_calls_the_share_a_prevalence_on_a_condition() -> None:
    """Whether the reference standard followed the test is checked at the cut-off, which the plan of a cut-off found
    on the data does not know yet. With rows that lack the reference standard, the plan called the share among the
    others "the prevalence of its setting" (44.4% for a made-up 30%)."""
    df = _partly_verified()
    lacking = int(df["ill"].isna().sum())
    said = _text(m.plan_accuracy(df, "ill", "Yes", "score", m.HIGHER, m.YOUDEN, None, None, SERIES))
    assert (
        f"{lacking:,} of {len(df):,} rows with the test lack it: that share is the prevalence of the setting only "
        "if the reference standard was not done because of the score (the result checks it at the cut-off found)."
    ) in said
    assert "so that share is the prevalence of its setting" not in said
    whole = _text(
        m.plan_accuracy(_cohort(600), "ill", "Yes", "score", m.HIGHER, m.YOUDEN, None, None, SERIES)
    )
    assert "a series, so that share is the prevalence of its setting" in whole


def test_a_share_next_to_all_is_written_one_way_on_the_page_and_in_the_dossier() -> None:
    """2,499 of 2,500 read "100.0%" in the headline and "more than 99.9%" in the table of every measure; at a
    prevalence of 1 in 2,500 the dossier wrote "NPV | 1 | 1 to 1"; at 1 in 100,000 the page wrote "more than 99.999%
    (95% CI 100.000% to 100.000%)"; and 9,990 of 9,991 read "more than 99.9% (95% CI 99.9% to 100.0%)", its upper
    bound being 99.998%."""
    metric = page._share_metric("Sensitivity", 2499 / 2500, None, 2499, 2500)
    assert (metric.value, metric.caption) == ("99.960%", "2,499 of 2,500")
    usual = diagnostic_accuracy(85, 10, 15, 90)
    metric = page._share_metric("Sensitivity", usual.sensitivity, usual.sensitivity_ci, 85, 100)
    assert (metric.value, metric.caption) == ("85.0%", "85 of 100 · 95% CI 76.7% to 90.7%")
    rare = diagnostic_accuracy(48, 3, 2, 947, prevalence=0.00001, sampling=TWO_GATE)
    estimate, bounds = share_parts(rare.npv_at_prevalence, rare.npv_at_prevalence_ci)
    lower, upper = bounds.split(" to ")
    # the decimals that tell the two bounds apart; the share itself is never written as all
    assert lower != upper and not estimate.startswith("100")
    # a bound below 100% is not written as 100% either, nor one above 0 as 0%
    near = diagnostic_accuracy(9990, 1, 10, 9999, sampling=SERIES)
    assert share_parts(near.ppv, near.ppv_ci) == ("99.990%", "99.943% to 99.998%")
    for share, low, high in ((0.95, 0.90, 0.9996), (0.05, 0.0004, 0.10)):
        estimate, bounds = share_parts(share, CI(lower=low, upper=high, level=0.95, method=""))
        assert not any(end in ("0.0%", "100.0%") for end in bounds.split(" to ")), bounds
        assert estimate in ("95.00%", "5.00%")  # the three with the same decimals


def test_a_bound_in_a_caution_is_written_as_the_table_writes_it_at_any_size() -> None:
    """A perfect test in 5,000 + 5,000 participants: "at least 1.3e+03" stood over the table's "at least 1,303"."""
    import re

    perfect = diagnostic_accuracy(5000, 0, 0, 5000)
    said = " ".join(perfect.warnings)
    rows = page._measure_rows(perfect, None).set_index("Measure")
    least = rows.loc["Positive likelihood ratio (LR+)", "95% CI"].removeprefix("at least ")
    most = rows.loc["Negative likelihood ratio (LR-)", "95% CI"].removeprefix("at most ")
    assert "," in least and f"it is at least {least} (95% interval)" in said
    assert f"it may be as large as {most} (95% interval)" in said
    assert not re.search(r"\de[+-]\d", said)  # no exponent


def test_a_dataclass_that_computes_a_field_passes_through_the_writable_text() -> None:
    """Every text of the dossier's input is cleaned by rebuilding its dataclasses: a field a class computes for
    itself was handed back to its constructor, which refuses it, before any guard."""
    import dataclasses

    from manage.dossier import _writable

    @dataclasses.dataclass
    class Holder:
        text: str
        length: int = dataclasses.field(init=False)

        def __post_init__(self) -> None:
            self.length = len(self.text)

    cleaned = _writable([Holder("a" + chr(11) + "b")])[0]
    assert cleaned.text == "a" + chr(0xFFFD) + "b" and cleaned.length == 3


def test_the_caveat_of_a_sample_taken_apart_is_one_sentence_on_the_page_and_in_the_dossier() -> (
    None
):
    from manage.report_tables import TWO_GATE_SPECTRUM

    assert m.TWO_GATE_SPECTRUM == TWO_GATE_SPECTRUM.removesuffix(".") + " (Lijmer et al. 1999)."


@pytest.mark.parametrize(
    ("cells", "sampling", "prevalence", "only_on_the_page"),
    [
        ((85, 10, 15, 90), TWO_GATE, 0.0004, ["Youden's J"]),
        ((85, 10, 15, 90), SERIES, None, ["Youden's J", "Share classified correctly"]),
        ((25, 40, 5, 858), SERIES, 0.02, ["Youden's J", "Share classified correctly"]),
        # a zero cell: one bound for a likelihood ratio, in words, and the odds ratio with 0.5 added to every cell
        ((30, 0, 6, 60), SAMPLING_NOT_STATED, 0.02, ["Youden's J"]),
    ],
)
def test_the_dossier_writes_a_measure_as_the_page_writes_it(
    cells: tuple[int, int, int, int],
    sampling: str,
    prevalence: float | None,
    only_on_the_page: list[str],
) -> None:
    """The block read "Sensitivity | 0.861 | 0.781 to 0.916", "NPV, ..." and "LR- | 0.164 | 0.1 to 0.266" under a
    page that read 86.1% (95% CI 78.1% to 91.6%), "Negative predictive value, ..." and 0.100 to 0.266: the rows are
    now one list, name for name and number for number."""
    result = _yes_no(cells, sampling, prevalence)
    measures = result_blocks(result).tables[0][1].set_index("Measure")
    shown = page._measure_rows(result["acc"], prevalence).set_index("Measure")
    assert [name for name in shown.index if name not in measures.index] == only_on_the_page
    pd.testing.assert_frame_equal(measures, shown.loc[list(measures.index)])
    assert all(
        text.endswith("%") for text in measures.loc[["Sensitivity", "Specificity"], "Estimate"]
    )
    assert not any("PPV" in name or "NPV" in name for name in measures.index)


def test_the_dossier_s_block_reads_as_the_page_for_the_tutorial_s_table() -> None:
    tutorial = result_blocks(_yes_no((87, 126, 14, 701))).tables[0][1].set_index("Measure")
    assert tutorial.loc["Sensitivity"].tolist() == ["86.1%", "78.1% to 91.6%"]
    assert tutorial.loc["Specificity"].tolist() == ["84.8%", "82.2% to 87.1%"]
    assert tutorial.loc["Negative likelihood ratio (LR-)"].tolist() == ["0.164", "0.100 to 0.266"]
    assert tutorial.loc[
        "Negative predictive value, at the share in these data (10.9%)"
    ].tolist() == [
        "98.0%",
        "96.7% to 98.8%",
    ]


def test_an_area_of_1_is_written_without_an_interval_of_no_width() -> None:
    """Forty and forty participants that a score separates without overlap read "1.000 (95% CI 1.000 to 1.000)", on
    the page and in the dossier, without a word: DeLong's variance is 0 there, and the interval says nothing of how
    precisely the area is known. An area of 0.9997 read "1.000"."""
    from manage.report_tables import AUC_NAME, auc_row

    frame = pd.DataFrame(
        {"ill": ["Yes"] * 40 + ["No"] * 40, "score": [*range(100, 140), *range(40)]}
    ).astype({"score": float})
    result = accuracy_of(frame, "ill", "Yes", "score", m.HIGHER, m.FIXED, 50.0, None, SERIES)
    (said,) = [text for text in result["cautions"] if "no width" in text]
    assert said.startswith(
        "The two groups do not overlap: every participant with the condition has a higher value than every "
        "participant without it."
    )
    assert (
        "With 40 with the condition and 40 without, an area of 1 does not show a test without error."
        in said
    )
    assert auc_row(result["auc"]) == (AUC_NAME, "1.000", "")
    blocks = result_blocks(result)
    assert blocks.tables[0][1].set_index("Measure").loc[AUC_NAME].tolist() == ["1.000", ""]
    assert said in blocks.warnings
    # lower values pointing to the condition: the words follow the direction asked
    reverse = accuracy_of(
        frame.assign(score=-frame["score"]),
        "ill",
        "Yes",
        "score",
        m.LOWER,
        m.FIXED,
        -50.0,
        None,
        SERIES,
    )
    assert any(
        "has a lower value than every participant without it" in text
        for text in reverse["cautions"]
    )
    # one tie between the two groups (half a pair of 1,600): the area keeps the decimals that tell it from 1, and
    # its interval
    near = frame.copy()
    near.loc[0, "score"] = 39.0
    name, area, bounds = auc_row(
        accuracy_of(near, "ill", "Yes", "score", m.HIGHER, m.FIXED, 50.0, None, SERIES)["auc"]
    )
    assert (name, area) == (AUC_NAME, "0.9997") and bounds.endswith(" (DeLong)")
    assert not bounds.startswith("1.000") and len(bounds.split(" to ")[0]) == len(area)
    # an ordinary area: three decimals, the method with its bounds, as the page's headline writes it
    usual = accuracy_of(_cohort(), "ill", "Yes", "score", m.HIGHER, m.FIXED, 52.0, None, SERIES)
    name, area, bounds = auc_row(usual["auc"])
    assert len(area) == 5 and bounds.count(" to ") == 1 and bounds.endswith(" (DeLong)")
    assert not any("no width" in text for text in usual["cautions"])
