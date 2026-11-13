"""
Two tabs of Analyze > Hypothesis tests & epidemiology through the page itself (AppTest, the tutorial cohort with its two
families ticked: 928 rows): Diagnostic accuracy & agreement, and Paired, rates & stratified.

One analysis is open at a time, chosen by the tab's Question; nothing is preselected; every analysis of the dataset
keeps its result with what it was computed on. The numbers are those the tabs gave before they were laid out this
way (commit 9adfdd1), every digit: a layout changes no result. The few that did change are corrections, each with
its own test.
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from ui import nav
from ui.components.population import _family_key
from ui.guidance import measures as m

DIAGNOSTIC, PAIRED = "Diagnostic accuracy & agreement", "Paired, rates & stratified"
TIGHT = {"rel": 1e-12, "abs": 1e-12}


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:400] for e in at.exception]


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory):
    previous = dp.DATA_ROOT
    dp.DATA_ROOT = str(tmp_path_factory.mktemp("data"))
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    at = AppTest.from_file(str(root / "main.py"), default_timeout=600)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "measure_tabs_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_tutorial").click().run()
    _ok(at)
    for family in ("clinical_anomalies", "study_inclusion"):
        at.checkbox(key=_family_key(family)).check().run()
    assert len(at.session_state["data"]) == 928
    at.switch_page(nav.page_path("analyze")).run()
    _ok(at)
    yield at
    dp.DATA_ROOT = previous


def _open(at: AppTest, tab: str, question: str, **choices: str) -> None:
    """Open a tab on one of its analyses, as its Question selector does (and the second choices of that analysis)."""
    at.session_state["epi_tab"] = tab
    at.run()
    _ok(at)
    key = "diag_question" if tab == DIAGNOSTIC else "paired_question"
    if at.segmented_control(key=key).value != question:
        at.segmented_control(key=key).set_value(question).run()
    for name, value in choices.items():
        if at.segmented_control(key=name).value != value:
            at.segmented_control(key=name).set_value(value).run()
    _ok(at)


def _pick(at: AppTest, key: str, value: Any) -> None:
    at.selectbox(key=key).select(value).run()
    _ok(at)


def _type(at: AppTest, key: str, value: float | None) -> None:
    at.number_input(key=key).set_value(value).run()
    _ok(at)


def _run(at: AppTest, key: str) -> None:
    button = at.button(key=key)
    assert not button.disabled, f"{key} is disabled"
    button.click().run()
    _ok(at)


def _stored(at: AppTest, key: str) -> Any:
    held = at.session_state["_results"].get(key) if "_results" in at.session_state else None
    return None if held is None else held.value


def _texts(at: AppTest) -> str:
    return "\n".join(
        [
            *(c.value for c in at.caption),
            *(md.value for md in at.markdown),
            *(w.value for w in at.warning),
        ]
    )


def _metrics(at: AppTest) -> dict[str, str]:
    return {metric.label: metric.value for metric in at.metric}


# ----------------------------------------------------------------------------- what opens


def test_a_tab_opens_on_one_analysis_with_nothing_chosen(app: AppTest) -> None:
    """The first columns of the file were preselected: a treatment arm as reference standard, age against height."""
    _open(app, DIAGNOSTIC, m.ACCURACY)
    assert [app.selectbox(key=key).value for key in ("diag_outcome", "diag_pos", "diag_score")] == [
        None
    ] * 3
    assert app.radio(key="diag_direction").value is None
    assert app.radio(key="diag_mode").value == m.NO_CUTOFF
    assert app.button(key="diag_run").disabled
    assert not any(
        key in {b.key for b in app.button} for key in ("agree_run", "kappa_run")
    )  # one at a time
    assert not app.metric
    _open(app, PAIRED, m.PAIRED, pair_measure=m.NUMBER)
    assert (
        app.selectbox(key="pair_before").value is None
        and app.selectbox(key="pair_after").value is None
    )
    assert app.button(key="pair_run").disabled
    assert not any(key in {b.key for b in app.button} for key in ("mc_run", "one_run", "mh_run"))


def test_the_question_and_its_choices_are_kept_while_another_is_open(app: AppTest) -> None:
    _open(app, PAIRED, m.PAIRED, pair_measure=m.NUMBER)
    _pick(app, "pair_before", "sbp_baseline")
    _open(app, PAIRED, m.STRATA)
    assert "pair_before" not in {s.key for s in app.selectbox}
    _open(app, DIAGNOSTIC, m.RATERS)
    _open(app, PAIRED, m.PAIRED)
    assert app.selectbox(key="pair_before").value == "sbp_baseline"


# ----------------------------------------------------------------------------- accuracy of a test


def _accuracy(
    app: AppTest, mode: str, cutoff: float | None = None, prevalence: float | None = None
) -> dict:
    _open(app, DIAGNOSTIC, m.ACCURACY)
    _pick(app, "diag_outcome", "mi_adjudicated")
    assert (
        app.selectbox(key="diag_pos").value == "Yes"
    )  # the value that says the condition is present, offered
    _pick(app, "diag_score", "troponin_ng_l")
    app.radio(key="diag_direction").set_value(m.HIGHER).run()
    app.radio(key="diag_mode").set_value(mode).run()
    if mode == m.FIXED:
        _type(app, "diag_cut", cutoff)
    _type(app, "diag_prev", prevalence)
    _run(app, "diag_run")
    return _stored(app, "diagnostic")


def test_accuracy_without_a_cut_off_gives_the_curve_and_its_area(app: AppTest) -> None:
    result = _accuracy(app, m.NO_CUTOFF)
    assert result["auc"]["auc"] == pytest.approx(0.9273229015767356, **TIGHT)
    assert result["auc"]["ci"].lower == pytest.approx(0.9019274929803464, **TIGHT)
    assert result["auc"]["ci"].upper == pytest.approx(0.9527183101731248, **TIGHT)
    assert (result["auc"]["n_positive"], result["auc"]["n_negative"], result["n"]) == (
        101,
        827,
        928,
    )
    assert result["acc"] is None and result["cutoff"] is None and result["youden"] is None
    assert _metrics(app)["Area under the ROC curve (AUC)"] == "0.927"
    assert "Sensitivity" not in _metrics(app)  # no cut-off, no table: none was asked


def test_accuracy_at_a_cut_off_found_on_the_data(app: AppTest) -> None:
    result = _accuracy(app, m.YOUDEN)
    youden, acc = result["youden"], result["acc"]
    assert (youden["cutoff"], result["cutoff"]) == (30.9, 30.9)
    assert (youden["ci"].lower, youden["ci"].upper, youden["resamples"]) == (22.4, 42.1, 2000)
    assert (acc.tp, acc.fp, acc.fn, acc.tn) == (87, 126, 14, 701)
    assert acc.sensitivity == pytest.approx(0.8613861386138614, **TIGHT)
    assert acc.specificity == pytest.approx(0.8476420798065296, **TIGHT)
    assert acc.ppv == pytest.approx(0.4084507042253521, **TIGHT)
    assert acc.npv == pytest.approx(0.9804195804195804, **TIGHT)
    assert acc.lr_positive == pytest.approx(5.653701084394153, **TIGHT)
    assert acc.lr_negative == pytest.approx(0.1635287637180266, **TIGHT)
    # no prevalence was typed: none is reported (0.502, the share of another column's first level, was)
    assert acc.prevalence_assumed is None and acc.ppv_at_prevalence is None
    assert "0.502" not in _texts(app)
    shown = _metrics(app)
    assert (shown["Sensitivity"], shown["Specificity"]) == ("86.1%", "84.8%")
    assert "exploratory" in _texts(app)  # a cut-off found on the data it is judged on


def test_accuracy_at_a_cut_off_fixed_beforehand_with_another_prevalence(app: AppTest) -> None:
    result = _accuracy(app, m.FIXED, cutoff=40.0, prevalence=0.05)
    acc = result["acc"]
    assert result["cutoff"] == 40.0 and result["youden"] is None
    assert (acc.tp, acc.fp, acc.fn, acc.tn) == (76, 72, 25, 755)
    assert acc.sensitivity == pytest.approx(0.7524752475247525, **TIGHT)
    assert acc.specificity == pytest.approx(0.9129383313180169, **TIGHT)
    assert acc.ppv == pytest.approx(0.5135135135135135, **TIGHT)
    assert acc.npv == pytest.approx(0.967948717948718, **TIGHT)
    assert acc.ppv_at_prevalence == pytest.approx(0.31266540642722124, **TIGHT)
    assert acc.npv_at_prevalence == pytest.approx(0.9859307801186782, **TIGHT)
    assert acc.lr_positive == pytest.approx(8.643014301430144, **TIGHT)
    assert acc.dor == pytest.approx(31.877777777777776, **TIGHT)
    assert result["auc"]["auc"] == pytest.approx(0.9273229015767356, **TIGHT)


def test_a_cut_off_typed_for_one_score_does_not_follow_to_another(app: AppTest) -> None:
    """The cut-off 67 (the median of the first numeric column, age) stayed when the score became troponin."""
    _accuracy(app, m.FIXED, cutoff=40.0)
    _pick(app, "diag_score", "nt_probnp_pg_ml")
    assert app.number_input(key="diag_cut").value is None
    assert app.button(key="diag_run").disabled  # no cut-off: nothing to compute
    assert "Last result here" in _texts(
        app
    )  # the result of the other choices did not vanish unsaid


def test_a_marker_that_falls_with_the_condition_is_read_the_right_way(app: AppTest) -> None:
    """Lower ejection fraction, more hospitalisation: read as "higher", the cut-off found called everyone positive."""
    _open(app, DIAGNOSTIC, m.ACCURACY)
    _pick(app, "diag_outcome", "hf_hospitalisation")
    _pick(app, "diag_score", "lvef_pct")
    app.radio(key="diag_direction").set_value(m.LOWER).run()
    app.radio(key="diag_mode").set_value(m.FIXED).run()
    _type(app, "diag_cut", 35.0)
    _type(app, "diag_prev", None)
    _run(app, "diag_run")
    result = _stored(app, "diagnostic")
    data = app.session_state["data"][["hf_hospitalisation", "lvef_pct"]].dropna()
    positive = data["lvef_pct"] <= 35.0
    ill = data["hf_hospitalisation"] == "Yes"
    acc = result["acc"]
    assert (acc.tp, acc.fp) == (int((positive & ill).sum()), int((positive & ~ill).sum()))
    assert (acc.fn, acc.tn) == (int((~positive & ill).sum()), int((~positive & ~ill).sum()))
    assert (
        result["auc"]["auc"] > 0.5 and result["cutoff"] == 35.0 and result["direction"] == m.LOWER
    )
    assert "at most 35" in _texts(app)


def test_a_cut_off_found_for_a_falling_marker_is_on_the_markers_own_scale(app: AppTest) -> None:
    """The functions read the reversed marker: the cut-off, its range and the table come back on the marker's scale,
    no other cut-off has a larger Youden's J, and the curve is the "higher" one read the other way."""
    from app_pages.analysis_diagnostic import accuracy_of

    data = app.session_state["data"][["hf_hospitalisation", "lvef_pct"]].dropna()
    found = accuracy_of(
        data, "hf_hospitalisation", "Yes", "lvef_pct", m.LOWER, m.YOUDEN, None, None
    )
    ill = (data["hf_hospitalisation"] == "Yes").to_numpy()
    values = data["lvef_pct"].to_numpy(dtype=float)
    cut, acc = found["cutoff"], found["acc"]
    positive = values <= cut
    assert (acc.tp, acc.fp) == (int((positive & ill).sum()), int((positive & ~ill).sum()))
    assert (acc.fn, acc.tn) == (int((~positive & ill).sum()), int((~positive & ~ill).sum()))

    def youden(limit: float) -> float:
        called = values <= limit
        return float((called & ill).sum() / ill.sum() + (~called & ~ill).sum() / (~ill).sum() - 1)

    assert acc.youden == pytest.approx(max(youden(limit) for limit in np.unique(values)), abs=1e-12)
    drawn = found["youden"]["ci"]
    assert drawn.lower <= cut <= drawn.upper and values.min() <= cut <= values.max()
    rising = accuracy_of(
        data, "hf_hospitalisation", "Yes", "lvef_pct", m.HIGHER, m.NO_CUTOFF, None, None
    )
    assert found["auc"]["auc"] + rising["auc"]["auc"] == pytest.approx(1.0, abs=1e-12)
    assert found["auc"]["auc"] > 0.5 > rising["auc"]["auc"]


def test_the_caution_of_a_direction_the_data_contradict_is_in_the_words_of_the_page(
    app: AppTest,
) -> None:
    """Troponin read as "lower", cut-off found on the data. The functions read the reversed marker, and their own
    caution was shown as it was: "higher values go with the other level ... Take the other level as positive", the
    opposite of the data and an advice that gives the accuracy for the absence of the condition."""
    from app_pages.analysis_diagnostic import accuracy_of

    data = app.session_state["data"]
    found = accuracy_of(
        data, "mi_adjudicated", "Yes", "troponin_ng_l", m.LOWER, m.YOUDEN, None, None
    )
    assert found["auc"]["auc"] < 0.1
    (caution,) = found["cautions"]
    assert caution.startswith("These data contradict the direction chosen: read with lower values")
    assert "Choose higher above" in caution and "other level" not in caution
    right = accuracy_of(
        data, "mi_adjudicated", "Yes", "troponin_ng_l", m.HIGHER, m.YOUDEN, None, None
    )
    assert right["cautions"] == []
    # through the page: the caution is under the result, and the engine's own is not
    _open(app, DIAGNOSTIC, m.ACCURACY)
    _pick(app, "diag_outcome", "mi_adjudicated")
    _pick(app, "diag_score", "troponin_ng_l")
    app.radio(key="diag_direction").set_value(m.LOWER).run()
    app.radio(key="diag_mode").set_value(m.YOUDEN).run()
    _type(app, "diag_prev", None)
    _run(app, "diag_run")
    warnings = " ".join(w.value for w in app.warning)
    assert "These data contradict the direction chosen" in warnings
    assert (
        "Take the other level as positive" not in warnings and "reverse the marker" not in warnings
    )


def test_a_score_of_few_values_is_not_said_to_call_no_one_positive() -> None:
    """A grade from 1 to 5 whose best cut-off is 5, the largest score: more than a hundred of 400 are called positive,
    and the caution read "nearly no one is called positive. The marker hardly separates the two levels"."""
    from app_pages.analysis_diagnostic import accuracy_of

    rng = np.random.default_rng(4)
    ill = rng.random(400) < 0.3
    grade = np.where(
        ill, rng.choice([3, 4, 5], 400, p=[0.02, 0.03, 0.95]), rng.choice([1, 2, 3, 4, 5], 400)
    )
    frame = __import__("pandas").DataFrame(
        {"ill": np.where(ill, "Yes", "No"), "grade": grade.astype(float)}
    )
    found = accuracy_of(frame, "ill", "Yes", "grade", m.HIGHER, m.YOUDEN, None, None)
    assert found["cutoff"] == 5.0 and found["acc"].tp + found["acc"].fp > 100
    assert found["cautions"] == [] and found["acc"].warnings == []
    assert found["youden"]["caution"] == ""  # the engine's own words: silent too


def test_the_sentence_under_the_auc_counts_ties_apart() -> None:
    """A grade in three values, AUC 0.66: "in 66 of 100 pairs the one with the condition has the higher value" was
    said of 51 pairs, with 30 ties counted for a half."""
    import pandas as pd

    from app_pages.analysis_diagnostic import _in_100, accuracy_of

    frame = pd.DataFrame(
        {
            "ill": ["Yes"] * 10 + ["No"] * 10,
            "grade": [3.0] * 5 + [2.0] * 3 + [1.0] * 2 + [3.0] * 2 + [2.0] * 3 + [1.0] * 5,
        }
    )
    found = accuracy_of(frame, "ill", "Yes", "grade", m.HIGHER, m.NO_CUTOFF, None, None)
    further, same, other = found["pairs"]
    # by hand over the 100 pairs: 5*8 + 3*5 higher, 5*2 + 3*3 + 2*5 the same, the rest lower
    assert (round(100 * further), round(100 * same), round(100 * other)) == (55, 29, 16)
    assert found["auc"]["auc"] == pytest.approx(further + same / 2)
    turned = accuracy_of(frame, "ill", "Yes", "grade", m.LOWER, m.NO_CUTOFF, None, None)["pairs"]
    assert turned == pytest.approx((other, same, further))
    assert (_in_100(0.9996), _in_100(1.0), _in_100(0.0004), _in_100(0.0), _in_100(0.514)) == (
        "more than 99",
        "100",
        "fewer than 1",
        "0",
        "51",
    )


# ----------------------------------------------------------------------------- agreement of two measurements


def _measurements(app: AppTest, columns: list[str], model: str = m.ABSOLUTE) -> None:
    _open(app, DIAGNOSTIC, m.MEASUREMENTS)
    app.multiselect(key="agree_cols").set_value(columns).run()
    app.radio(key="icc_model").set_value(model).run()
    _run(app, "agree_run")


def test_two_measurements_give_the_limits_of_agreement_and_the_icc_in_one_run(app: AppTest) -> None:
    _measurements(app, ["sbp_baseline", "sbp_ambulatory"])
    ba = _stored(app, "bland_altman")
    assert ba.n == 928
    assert ba.bias == pytest.approx(5.987068965517241, **TIGHT)
    assert (ba.bias_ci.lower, ba.bias_ci.upper) == pytest.approx(
        (5.521036056529904, 6.453101874504578), **TIGHT
    )
    assert ba.sd == pytest.approx(7.233942887205186, **TIGHT)
    assert (ba.loa_lower, ba.loa_upper) == pytest.approx(
        (-8.191198559624617, 20.1653364906591), **TIGHT
    )
    assert ba.loa_lower_ci.lower == pytest.approx(-8.998391235989807, **TIGHT)
    assert ba.loa_upper_ci.upper == pytest.approx(20.97252916702429, **TIGHT)
    assert ba.within_loa_pct == pytest.approx(95.79741379310344, **TIGHT)
    assert (
        _stored(app, "icc") is not None
    )  # it vanished at the next click: kept now, with what it was computed on
    shown = _metrics(app)
    assert shown["Mean difference"] == "5.99"
    assert "The difference changes with the size of the measurement" in _texts(
        app
    )  # the result's own warning


def test_the_icc_of_two_readers_in_both_forms(app: AppTest) -> None:
    _measurements(app, ["lvef_pct", "lvef_reader2_pct"], m.ABSOLUTE)
    absolute = _stored(app, "icc")
    assert (round(absolute.icc, 3), absolute.n_subjects, absolute.n_raters) == (0.942, 928, 2)
    assert (round(absolute.ci.lower, 3), round(absolute.ci.upper, 3)) == (0.927, 0.952)
    _measurements(app, ["lvef_pct", "lvef_reader2_pct"], m.CONSISTENCY)
    consistency = _stored(app, "icc")
    assert round(consistency.icc, 3) == 0.945
    assert (round(consistency.ci.lower, 3), round(consistency.ci.upper, 3)) == (0.938, 0.952)
    assert "subjects" not in " ".join(c.value for c in app.caption)


# ----------------------------------------------------------------------------- agreement of two raters


def _raters(app: AppTest, scale: str, weights: str | None = None) -> dict:
    _open(app, DIAGNOSTIC, m.RATERS)
    _pick(app, "kappa_a", "nyha_class")
    _pick(app, "kappa_b", "nyha_class_central")
    app.radio(key="kappa_scale").set_value(scale).run()
    if weights is not None:
        app.radio(key="kappa_w").set_value(weights).run()
    _run(app, "kappa_run")
    return _stored(app, "kappa")


def test_kappa_of_two_raters_without_and_with_an_order(app: AppTest) -> None:
    plain = _raters(app, m.NAMES)
    assert (
        round(plain["kappa"].kappa, 3) == 0.707
        and round(plain["kappa"].observed_agreement, 3) == 0.822
    )
    assert (round(plain["kappa"].ci.lower, 3), round(plain["kappa"].ci.upper, 3)) == (0.667, 0.746)
    assert plain["weights"] is None and plain["kappa"].n == 928
    table = plain["table"]
    assert list(table.index) == list(
        table.columns
    )  # the same categories both ways: the diagonal is agreement
    assert int(table.to_numpy().sum()) == 928
    linear = _raters(app, m.GRADES, m.LINEAR)
    assert round(linear["kappa"].kappa, 3) == 0.759 and linear["order"] == ["I", "II", "III", "IV"]
    assert (round(linear["kappa"].ci.lower, 3), round(linear["kappa"].ci.upper, 3)) == (
        0.727,
        0.792,
    )
    quadratic = _raters(app, m.GRADES, m.QUADRATIC)
    assert round(quadratic["kappa"].kappa, 3) == 0.824
    assert (round(quadratic["kappa"].ci.lower, 3), round(quadratic["kappa"].ci.upper, 3)) == (
        0.8,
        0.849,
    )
    assert "I < II < III < IV" in _texts(app)  # the order the weights read, said with the result


# ----------------------------------------------------------------------------- paired measurements


def _paired(app: AppTest, kind: str) -> Any:
    _open(app, PAIRED, m.PAIRED, pair_measure=m.NUMBER)
    _pick(app, "pair_before", "sbp_baseline")
    _pick(app, "pair_after", "sbp_6m")
    app.radio(key="pair_kind").set_value(kind).run()
    _run(app, "pair_run")
    return _stored(app, "paired")


def test_the_change_between_two_measurements(app: AppTest) -> None:
    mean = _paired(app, m.T_TEST)
    assert mean.statistic == pytest.approx(-14.742770318130594, **TIGHT)
    assert mean.p_value == pytest.approx(1.477367353713424e-43, rel=1e-9)
    assert mean.estimate == pytest.approx(-5.010282776349614, **TIGHT)
    assert (mean.estimate_ci.lower, mean.estimate_ci.upper) == pytest.approx(
        (-5.677409375208041, -4.3431561774911875), **TIGHT
    )
    assert mean.effect_sizes[0].estimate == pytest.approx(-0.5285539236415279, **TIGHT)
    assert mean.n_total == 778
    shown = _metrics(app)
    assert shown["Mean change"] == "-5.01" and shown["Pairs"] == "778"
    captions = " ".join(c.value for c in app.caption)
    assert "p < 0.001" in captions and "e-43" not in captions
    assert "e-43" not in _texts(app)  # nor in "How to read this"
    ranks = _paired(app, m.SIGNED_RANK)
    assert ranks.statistic == 60962.5 and ranks.estimate == -5.5 and ranks.n_total == 744
    assert ranks.p_value == pytest.approx(5.169548445645164e-40, rel=1e-9)
    assert (ranks.estimate_ci.lower, ranks.estimate_ci.upper) == (-6.0, -4.5)
    assert _metrics(app)["Pairs that differ"] == "744"
    assert "34" in " ".join(
        w.value for w in app.warning
    )  # the pairs a rank test leaves out: its warning was not shown


def _yes_no(app: AppTest) -> dict:
    _open(app, PAIRED, m.PAIRED, pair_measure=m.YES_NO)
    _pick(app, "mc_before", "smoker_baseline")
    _pick(app, "mc_after", "smoker_6m")
    assert app.selectbox(key="mc_counted").value == "Yes"
    _run(app, "mc_run")
    return _stored(app, "mcnemar")


def test_a_change_in_a_yes_or_no_characteristic(app: AppTest) -> None:
    held = _yes_no(app)
    result, table = held["res"], held["table"]
    # without it the dossier's "difference in proportions (after - before) = -0.0625" did not say of which value
    assert result.estimate_label.startswith("change in the share with smoker_baseline = Yes")
    assert table.to_numpy().tolist() == [[744, 0], [58, 126]]
    assert [str(v) for v in table.index] == ["No", "Yes"] == [str(v) for v in table.columns]
    assert result.estimate == pytest.approx(-0.0625, **TIGHT)
    assert result.p_value == pytest.approx(6.938893903907228e-18, rel=1e-9)
    assert (result.estimate_ci.lower, result.estimate_ci.upper) == pytest.approx(
        (-0.07820017696367572, -0.04679982303632428), **TIGHT
    )
    shown = _metrics(app)
    assert shown["Change in the share with Yes"] == "-6.2 points" and shown["Pairs"] == "928"
    assert "58 participants changed from `Yes` to `No`" in _texts(app)


# ----------------------------------------------------------------------------- one group against a reference


def _one_mean(app: AppTest, kind: str) -> Any:
    _open(app, PAIRED, m.ONE_GROUP, one_compare=m.MEAN)
    _pick(app, "one_col", "hba1c_pct")
    _type(app, "one_ref", 7.0)
    app.radio(key="one_kind").set_value(kind).run()
    _run(app, "one_run")
    return _stored(app, "one_sample")


def test_a_mean_against_a_reference_value(app: AppTest) -> None:
    mean = _one_mean(app, m.ONE_T)
    assert mean.statistic == pytest.approx(-9.367279560456472, **TIGHT)
    assert mean.p_value == pytest.approx(9.44896001414053e-20, rel=1e-9)
    assert mean.estimate == pytest.approx(-0.44777158774373227, **TIGHT)
    assert (mean.estimate_ci.lower, mean.estimate_ci.upper) == pytest.approx(
        (-0.5416195551632113, -0.3539236203242533), **TIGHT
    )
    assert mean.n_total == 718
    assert _metrics(app)["Difference from 7"] == "-0.448"
    ranks = _one_mean(app, m.SIGNED_RANK)
    assert ranks.statistic == 73787.0 and ranks.n_total == 703
    assert ranks.estimate == pytest.approx(-0.5499999999999998, **TIGHT)
    assert ranks.p_value == pytest.approx(1.7971531729270538e-20, rel=1e-9)


def test_a_reference_typed_for_one_measurement_does_not_follow_to_another(app: AppTest) -> None:
    """The reference 67 (the mean of age) stayed when the measurement changed: it was tested against, and reported."""
    _one_mean(app, m.ONE_T)
    _pick(app, "one_col", "sbp_6m")
    assert app.number_input(key="one_ref").value is None
    assert app.button(key="one_run").disabled


def test_a_share_against_a_reference_from_numbers_typed(app: AppTest) -> None:
    _open(app, PAIRED, m.ONE_GROUP, one_compare=m.SHARE, prop_source=m.TYPED)
    assert not app.metric  # no example numbers: nothing is shown before something is typed
    _type(app, "prop_k", 30)
    _type(app, "prop_n", 100)
    _type(app, "prop_ref", 0.2)
    shown = _metrics(app)  # no button: the result follows the numbers
    assert shown["Share with the characteristic"] == "30.0%"
    text = _texts(app)
    assert "21.9% to 39.6%" in text and "p = 0.017" in text
    assert "From the numbers typed: 30 of 100." in text and "Not from the dataset" in text
    _type(app, "prop_k", 130)  # more than the participants counted
    assert not app.metric


def _share(app: AppTest) -> Any:
    _open(app, PAIRED, m.ONE_GROUP, one_compare=m.SHARE, prop_source=m.DATASET)
    _pick(app, "prop_col", "hf_hospitalisation")
    assert app.selectbox(key="prop_level").value == "Yes"
    _type(app, "prop_ref", 0.3)
    _run(app, "prop_run")
    return _stored(app, "proportion")


def test_a_share_against_a_reference_read_from_the_dataset(app: AppTest) -> None:
    result = _share(app)
    column = app.session_state["data"]["hf_hospitalisation"].dropna()
    assert result.n_total == len(column)
    assert result.estimate == pytest.approx(float((column == "Yes").mean()), **TIGHT)
    assert app.session_state["_results"]["proportion"].stamp.n_analysed == len(column)


# ----------------------------------------------------------------------------- an association within strata


def _strata(app: AppTest, stratum: str) -> Any:
    _open(app, PAIRED, m.STRATA)
    _pick(app, "mh_outcome", "hf_hospitalisation")
    assert app.selectbox(key="mh_ev_level").value == "Yes"
    _pick(app, "mh_exposure", "arm")
    _pick(app, "mh_reference", "Control")
    _pick(app, "mh_stratum", stratum)
    _run(app, "mh_run")
    return _stored(app, "mantel_haenszel")


def test_an_association_within_strata_whose_ratios_are_alike(app: AppTest) -> None:
    res = _strata(app, "nyha_class")
    assert res.n_strata == 3
    assert res.or_mh == pytest.approx(0.6278365838812753, **TIGHT)
    assert (res.or_mh_ci.lower, res.or_mh_ci.upper) == pytest.approx(
        (0.47420348918899285, 0.83124393861769), **TIGHT
    )
    assert res.rr_mh == pytest.approx(0.7353842796027623, **TIGHT)
    assert (res.rr_mh_ci.lower, res.rr_mh_ci.upper) == pytest.approx(
        (0.6103213064411115, 0.8860743234417187), **TIGHT
    )
    assert (res.crude_or, res.crude_rr) == pytest.approx(
        (0.6394572991886932, 0.7392641841772309), **TIGHT
    )
    assert res.breslow_day_p == pytest.approx(0.28182201913335003, **TIGHT)
    assert res.mh_test_p == pytest.approx(0.001427918019357266, **TIGHT)
    assert res.per_stratum[
        ["exposed_events", "exposed_total", "reference_events", "reference_total"]
    ].to_numpy().tolist() == [
        [47, 240, 80, 244],
        [61, 182, 72, 175],
        [21, 44, 21, 43],
    ]
    text = _texts(app)
    assert "no evidence that they differ" in text and "no proof that they are alike" in text
    assert (
        "Computed on" in text
    )  # the line of what the result was computed on: this analysis had none
    assert _metrics(app)["Odds ratio, adjusted for nyha_class"] == "0.628"


def test_an_association_within_strata_whose_ratios_differ(app: AppTest) -> None:
    res = _strata(app, "diabetes")
    assert res.n_strata == 2
    assert res.or_mh == pytest.approx(0.6438268815584676, **TIGHT)
    assert res.rr_mh == pytest.approx(0.7423136160876451, **TIGHT)
    assert res.breslow_day_p == pytest.approx(0.043600686623524454, **TIGHT)
    assert res.mh_test_p == pytest.approx(0.002154080463419472, **TIGHT)
    assert "Report the ratio of each stratum" in _texts(app)


def test_strata_without_information_are_left_out_of_the_tests_and_cleaned_from_the_plan(
    app: AppTest,
) -> None:
    """`site` as recorded: 7 strata, three of them one site written with a space, two of those holding one group
    only. The tests had no value and the page read "heterogeneous across strata"; the tests now read the five strata
    that hold both groups and both outcomes, and the plan offers to clean the variable where it stands."""
    _open(app, PAIRED, m.STRATA)
    _pick(app, "mh_outcome", "hf_hospitalisation")
    _pick(app, "mh_exposure", "arm")
    _pick(app, "mh_reference", "Control")
    _pick(app, "mh_stratum", "site")
    text = _texts(app)
    assert "written several ways" in text and "site_clean" in text
    assert "2 of 7 strata hold one group only or one outcome only" in text
    assert any("Clean" in str(b.label) for b in app.button), [b.label for b in app.button]
    _run(app, "mh_run")  # advice never blocks a run
    res = _stored(app, "mantel_haenszel")
    assert (res.n_strata, res.n_tested, res.n) == (7, 5, 928)
    assert np.isfinite(res.breslow_day_p) and np.isfinite(res.mh_test_p)
    text = _texts(app)
    assert "the two tests read the other 5" in text  # the result's own warning, shown
    assert "heterogeneous" not in text and "928 of 928 rows" in text
    assert app.session_state["_results"]["mantel_haenszel"].stamp.n_analysed == 928


def test_strata_that_differ_are_said_beside_the_combined_ratio(app: AppTest) -> None:
    """The headline gave one combined ratio with no mark, and the line under it said not to report it."""
    _strata(app, "diabetes")
    badges = [md.value for md in app.markdown if "badge" in md.value]
    assert any("the strata differ: read each stratum" in badge for badge in badges), badges
    _strata(app, "nyha_class")
    badges = [md.value for md in app.markdown if "badge" in md.value]
    assert not any("the strata differ" in badge or "not compared" in badge for badge in badges)


def test_the_reference_group_decides_which_way_the_ratios_read(app: AppTest) -> None:
    """ "Exposed level" asked for the group compared; the 2x2 panel of the same page asks for the reference group."""
    _strata(app, "nyha_class")
    _pick(app, "mh_reference", "Intervention")
    _run(app, "mh_run")
    turned = _stored(app, "mantel_haenszel")
    assert turned.or_mh == pytest.approx(1 / 0.6278365838812753, rel=1e-12)


# ----------------------------------------------------------------------------- rates, from counts


def _rates(app: AppTest, calculation: str) -> None:
    _open(app, PAIRED, m.RATES, rate_calc=calculation)


def test_two_rates_and_their_ratio_follow_the_numbers_typed(app: AppTest) -> None:
    _rates(app, m.TWO_RATES)
    for key in ("rate_e1", "rate_t1", "rate_e0", "rate_t0"):
        _type(app, key, None)
    assert not app.metric and "rate_run" not in {
        b.key for b in app.button
    }  # no examples, no button
    _type(app, "rate_e1", 15)
    _type(app, "rate_t1", 1200.0)
    _type(app, "rate_e0", 4)
    assert not app.metric  # a number is still missing: no result on half the numbers
    _type(app, "rate_t0", 1000.0)
    shown = _metrics(app)
    assert shown["Rate ratio"] == "3.12"
    assert shown["Group compared"] == "12.50"
    assert shown["Reference group"] == "4.00"
    text = _texts(app)
    assert (
        "7.00 to 20.62" in text and "1.09 to 10.24" in text
    )  # exact Poisson intervals of the two rates
    assert (
        "95% CI 0.996 to 12.935" in text
    )  # the exact conditional interval of the ratio (0.996 to 12.9)
    assert "From the numbers typed: 15 events over 1,200 person-years against 4 over 1,000." in text
    _type(app, "rate_e0", 0)  # no event in the reference group: no "inf"
    assert _metrics(app)["Rate ratio"] == "no upper bound"
    assert not re.search(r"inf", _texts(app))


def test_one_rate(app: AppTest) -> None:
    _rates(app, m.ONE_RATE)
    _type(app, "rate_e1", 15)
    _type(app, "rate_t1", 1200.0)
    assert _metrics(app) == {"Rate": "12.50"}
    assert "per 1,000 person-years · 15 events / 1,200 · 95% CI 7.00 to 20.62" in _texts(app)
    assert "rate_e0" not in {
        n.key for n in app.number_input
    }  # one group: no reference to leave at 0


def test_observed_against_expected_events(app: AppTest) -> None:
    _rates(app, m.EXPECTED)
    assert not app.metric
    _type(app, "smr_obs", 25)
    _type(app, "smr_exp", 18.4)
    shown = _metrics(app)
    assert shown["Standardised mortality ratio (SMR)"] == "1.36"
    assert "95% CI 0.879 to 2.006" in _texts(app)
    app.radio(key="smr_kind").set_value(m.CASES).run()
    assert "Standardised incidence ratio (SIR)" in _metrics(app)


# ----------------------------------------------------------------------------- the study dossier


def test_every_result_kept_by_the_two_tabs_can_go_to_the_dossier(app: AppTest) -> None:
    """Kappa, the ICC and the share against a reference vanished at the next click: they could not be reported."""
    from biostats.plan import KINDS
    from manage.report_tables import result_blocks

    _accuracy(app, m.NO_CUTOFF)  # a result without a cut-off: no 2x2 table to write
    _measurements(app, ["sbp_baseline", "sbp_ambulatory"])
    _raters(app, m.GRADES, m.LINEAR)
    _paired(app, m.T_TEST)
    _yes_no(app)
    _one_mean(app, m.ONE_T)
    _share(app)
    _strata(app, "nyha_class")
    held = app.session_state["_results"]
    keys = (
        "diagnostic",
        "bland_altman",
        "icc",
        "kappa",
        "paired",
        "mcnemar",
        "one_sample",
        "proportion",
        "mantel_haenszel",
    )
    assert set(keys) <= set(held), sorted(set(keys) - set(held))
    for key in keys:
        assert key in KINDS, key  # a kind without a title showed its raw key in the dossier
        blocks = result_blocks(held[key].value)
        written = " ".join([*blocks.lines, *(frame.to_string() for _, frame in blocks.tables)])
        assert written.strip(), key
        assert "subjects" not in written and "object at 0x" not in written, (key, written[:200])
    titles = [title for title, _ in result_blocks(held["diagnostic"].value).tables]
    assert any("(no cut-off)" in title for title in titles), titles


# ----------------------------------------------------------------------------- links


def test_a_link_opens_its_analysis_after_another_was_open(app: AppTest) -> None:
    """ "Which test?" named a section of a long tab, and its link opened the page on its first tab. A link now opens
    the analysis it names, whatever was open on the page before (every declared link: test_tab_links.py)."""
    _open(app, PAIRED, m.STRATA)
    app.switch_page(
        nav.page_path("which_test")
    ).run()  # another page: the choices of this one are let go
    app.switch_page(nav.page_path("analyze")).run()
    app.query_params[nav.TAB_PARAM] = "paired_yes_no"
    app.run()
    _ok(app)
    assert app.segmented_control(key="paired_question").value == m.PAIRED
    assert app.segmented_control(key="pair_measure").value == m.YES_NO
    assert "mc_run" in {b.key for b in app.button}
    del app.query_params[nav.TAB_PARAM]
    app.run()
    _ok(app)


def test_every_entry_of_the_two_tabs_can_be_linked_to() -> None:
    """A label renamed on one side only fails here, not as a link that opens the page on its first tab."""
    from app_pages.epidemiology import TABS

    paths = dict(nav.page("analyze").tabs)
    assert {labels[0] for labels in paths.values()} <= set(TABS)
    entries = {
        DIAGNOSTIC: {e.label for e in m.DIAGNOSTIC_ENTRIES},
        PAIRED: {e.label for e in m.PAIRED_ENTRIES},
    }
    seconds = {
        m.ENTRY[m.PAIRED].label: set(m.MEASURES.values()),
        m.ENTRY[m.ONE_GROUP].label: set(m.COMPARED.values()),
        m.ENTRY[m.RATES].label: set(m.CALCULATIONS.values()),
    }
    linked = {tab: set() for tab in entries}
    for labels in paths.values():
        if labels[0] not in entries:
            continue
        assert len(labels) >= 2 and labels[1] in entries[labels[0]], labels
        linked[labels[0]].add(labels[1])
        if len(labels) > 2:
            assert labels[2] in seconds[labels[1]], labels
    assert linked == entries  # every entry has a link


def test_which_test_opens_the_analysis_it_recommends() -> None:
    from ui.guidance.test_chooser import OTHER_QUESTIONS, all_combinations, recommend

    paths = dict(nav.page("analyze").tabs)
    for combination in all_combinations():
        recommendation = recommend(*combination)
        if recommendation.page == "analyze" and recommendation.available:
            assert recommendation.tab in paths, combination
    paired = recommend(
        "Continuous (measurement)",
        "Two groups",
        "Paired / repeated measures",
        "Roughly symmetric, at least 15 per group",
    )
    assert paired.tab == "paired_number"
    assert paired.where == (
        "Analyze > Hypothesis tests & epidemiology > Paired, rates & stratified > Paired measurements > A number "
        "(continuous)"
    )
    share = recommend(
        "Binary (yes/no)",
        "One group vs a reference value",
        "Independent groups",
        "Roughly symmetric, at least 15 per group",
    )
    assert share.tab == "one_share"
    assert {tab for _, tab in OTHER_QUESTIONS} <= set(paths)
    # the four questions compare groups: what they cannot reach is linked beside them
    assert {"accuracy", "measurements", "raters", "strata", "rates", "expected"} == {
        t for _, t in OTHER_QUESTIONS
    }


def test_the_orientation_opens_what_it_names(app: AppTest) -> None:
    app.session_state["epi_tab"] = "Analysis"
    app.run()
    app.button(key=f"epi_open::{PAIRED}::{m.STRATA}").click().run()
    _ok(app)
    assert app.session_state["epi_tab"] == PAIRED
    assert app.segmented_control(key="paired_question").value == m.STRATA
    assert "mh_run" in {b.key for b in app.button}
    app.button(key="epi_open::Power & Sample Size::").click().run()
    _ok(app)
    assert app.session_state["epi_tab"] == "Power & Sample Size"


def test_the_controls_of_the_analysis_tab_are_on_that_tab(app: AppTest) -> None:
    """ "Analysis type" and "Transform numeric variables" stood above the six tabs, and acted on the first only."""
    app.session_state["epi_tab"] = DIAGNOSTIC
    app.run()
    _ok(app)
    assert "epi_transformation" not in {s.key for s in app.selectbox}
    assert "epi_analysis_mode" not in {s.key for s in app.segmented_control}
    assert (
        app.toggle(key="epi_guidance").value is True
    )  # Guidance governs every tab: it stays above them
    app.session_state["epi_tab"] = "Analysis"
    app.run()
    _ok(app)
    assert app.selectbox(key="epi_transformation").value == "None (Raw Data)"
    assert np.isfinite(len(app.segmented_control(key="epi_analysis_mode").options))


def test_guidance_off_hides_the_plan_and_keeps_what_would_make_a_result_wrong(app: AppTest) -> None:
    _open(app, PAIRED, m.PAIRED, pair_measure=m.NUMBER)
    _pick(app, "pair_before", "sbp_baseline")
    _pick(app, "pair_after", "sbp_6m")
    assert "Before you run: the plan" in _texts(app)
    app.toggle(key="epi_guidance").set_value(False).run()
    _ok(app)
    assert "Before you run: the plan" not in _texts(app)
    assert not app.button(key="pair_run").disabled
    app.toggle(key="epi_guidance").set_value(True).run()
