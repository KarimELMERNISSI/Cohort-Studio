"""
Guidance of the Survival & regression page (ui/guidance/outcomes.py): the plan says what Run will do, on the rows
and events the models use, and advises without deciding; the comparison's terms are read by their names; it names
what would make a model's terms unstable or impossible to estimate before the model runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biostats.regression import count_regression, logistic_regression
from biostats.survival import compare_survival, cox_regression
from ui.guidance.glossary import define
from ui.guidance.outcomes import (
    BINARY,
    COUNT,
    HELP,
    NONE,
    OUTCOME,
    OUTCOME_TYPES,
    TIME_TO_EVENT,
    comparison_terms,
    covariates_of,
    diagnostics,
    listing,
    plan_binary,
    plan_count,
    plan_time_to_event,
)

rng = np.random.default_rng(11)


def _cohort(n: int = 240) -> pd.DataFrame:
    arm = np.where(np.arange(n) % 2 == 0, "Placebo", "Treatment")
    age = rng.normal(65, 10, n).round(1)
    age[::17] = np.nan  # missing in a few rows: the models leave them out
    time = rng.exponential(np.where(arm == "Treatment", 900, 600)).round()
    time[3] = -12  # a planted negative follow-up time
    event = (rng.random(n) < 0.35).astype(float)
    event[5] = np.nan
    return pd.DataFrame(
        {
            "fu_days": time,
            "death": event,
            "arm": arm,
            "age": age,
            "sex": np.where(rng.random(n) < 0.5, "F", "M"),
            "hosp": np.where(rng.random(n) < 0.3, "Yes", "No"),
            "visits": rng.poisson(2, n).astype(float),
            "years": rng.uniform(0.5, 3, n).round(2),
            "smoker": (rng.random(n) < 0.4).astype(int),
        }
    )


def _advice(plan, level: str) -> list[str]:
    return [a.text for a in plan.advice if a.level == level]


def test_every_outcome_type_names_terms_the_glossary_defines() -> None:
    assert [o.id for o in OUTCOME_TYPES] == [TIME_TO_EVENT, BINARY, COUNT]
    for outcome in OUTCOME_TYPES:
        assert outcome.gives and OUTCOME[outcome.id] is outcome
        for term in outcome.terms:
            assert define(term), term
    assert all(len(text) > 40 for text in HELP.values())


def test_the_comparison_comes_first_and_once() -> None:
    assert covariates_of("arm", ["age", "arm", "sex"]) == ["arm", "age", "sex"]
    assert covariates_of(NONE, ["age"]) == ["age"] and covariates_of(None, []) == []
    assert listing(["age"]) == "`age`" and listing(["a", "b", "c"]) == "`a`, `b` and `c`"


def test_the_time_to_event_plan_counts_the_rows_and_events_the_models_use() -> None:
    df = _cohort()
    plan = plan_time_to_event(df, "fu_days", "death", "arm", ["age"], group=True)
    km, cox = plan.steps
    curves = compare_survival(df, "fu_days", "death", "arm")
    assert (km.used, km.events) == (
        sum(c.n for c in curves.curves),
        sum(c.events for c in curves.curves),
    )
    assert "`arm` (2 groups)" in km.text and "log-rank" in km.text
    model = cox_regression(df, "fu_days", "death", ["arm", "age"], diagnostics=False)
    assert (cox.used, cox.events, cox.terms) == (model.n, model.events, len(model.table))
    assert "`arm`, adjusted for `age`" in cox.text
    caution = _advice(plan, "caution")
    assert any("negative `fu_days`" in text for text in caution)  # the planted -12
    assert any("missing `age`" in text for text in _advice(plan, "info"))  # complete cases, said
    assert plan.status == "caution"


def test_a_code_the_event_cannot_hold_is_refused_before_the_run() -> None:
    df = _cohort()
    df.loc[df.index[:4], "death"] = 2.0
    plan = plan_time_to_event(df, "fu_days", "death", "arm", [], group=True)
    assert plan.status == "violated" and not plan.steps
    assert "`death`" in plan.advice[0].text and r"0 \(censored\)" in plan.advice[0].text  # as text


def test_a_numeric_comparison_draws_one_curve_and_a_model_without_variables_none() -> None:
    df = _cohort()
    plan = plan_time_to_event(df, "fu_days", "death", "age", [], group=False)
    assert "all the rows" in plan.steps[0].text and "per unit" in _advice(plan, "info")[0]
    alone = plan_time_to_event(df, "fu_days", "death", NONE, [], group=False)
    assert len(alone.steps) == 1 and any("needs a variable" in t for t in _advice(alone, "info"))
    model_only = plan_time_to_event(df, "fu_days", "death", "arm", [], group=True, curves=False)
    assert len(model_only.steps) == 1 and "unadjusted" in model_only.steps[0].text


def test_few_events_per_term_are_flagged_not_blocked() -> None:
    df = _cohort(60)
    plan = plan_time_to_event(
        df, "fu_days", "death", "arm", ["age", "sex"], group=True, curves=False
    )
    (step,) = plan.steps
    assert step.events is not None and step.terms is not None
    assert step.events < 10 * step.terms
    assert any("per term" in text and "Peduzzi" in text for text in _advice(plan, "caution"))
    assert plan.status == "caution"  # a warning, never a refusal


def test_the_binary_plan_matches_the_logistic_model() -> None:
    df = _cohort()
    plan = plan_binary(df, "hosp", "Yes", "arm", ["age", "sex"])
    (step,) = plan.steps
    model = logistic_regression(df, "hosp", ["arm", "age", "sex"], event_level="Yes")
    assert (step.used, step.events, step.terms) == (model.n, model.events, len(model.table))
    assert "`hosp` = `Yes`, comparing `arm`, adjusted for `age` and `sex`" in step.text
    assert not plan_binary(df, "hosp", "Yes", NONE, []).steps  # nothing to fit yet
    three = df.assign(hosp=np.resize(["a", "b", "c"], len(df)))
    assert plan_binary(three, "hosp", "a", "arm", []).status == "violated"


def test_separation_and_the_fewer_of_events_and_non_events() -> None:
    df = _cohort(80)
    df.loc[df["sex"] == "F", "hosp"] = "Yes"  # every woman has the event
    plan = plan_binary(df, "hosp", "Yes", "arm", ["sex"])
    assert any("separation" in text and "`F`" in text for text in _advice(plan, "caution"))
    rare = _cohort(80).assign(hosp="No")
    rare.loc[rare.index[:6], "hosp"] = "Yes"
    texts = _advice(plan_binary(rare, "hosp", "Yes", "arm", ["sex"]), "caution")
    assert any(text.startswith("6 events for 2 terms") for text in texts)


def test_the_count_plan_matches_the_count_model_and_asks_for_person_time() -> None:
    df = _cohort()
    plan = plan_count(df, "visits", "years", "arm", ["age"], "poisson")
    (step,) = plan.steps
    model = count_regression(df, "visits", ["arm", "age"], exposure_col="years")
    assert (step.used, step.events, step.terms) == (model.n, model.events, len(model.table))
    assert "Poisson model of `visits` per `years` of person-time" in step.text
    without = plan_count(df, "visits", NONE, "arm", [], "negative_binomial")
    assert "Negative binomial model" in without.steps[0].text
    assert any("mean counts, not rates" in t for t in _advice(without, "info"))
    bad = df.assign(
        visits=df["visits"].where(df.index > 1, -1.0), years=df["years"].where(df.index > 2, 0.0)
    )
    texts = _advice(plan_count(bad, "visits", "years", "arm", [], "poisson"), "violated")
    assert any("below 0" in t for t in texts) and any("must be positive" in t for t in texts)
    half = df.assign(visits=df["visits"] + 0.5)
    assert any(
        "whole numbers" in t
        for t in _advice(plan_count(half, "visits", NONE, "arm", [], "poisson"), "caution")
    )


@pytest.mark.parametrize(
    ("compare", "terms", "values", "reference", "expected"),
    [
        (
            "arm",
            ["arm_Treatment", "age"],
            ["Placebo", "Treatment"],
            "Placebo",
            [("arm_Treatment", "Treatment vs Placebo")],
        ),
        # a covariate whose name starts like the comparison's terms is not one of them
        (
            "arm",
            ["arm_size", "arm_B", "arm_C"],
            ["A", "B", "C"],
            None,
            [("arm_B", "B vs A"), ("arm_C", "C vs A")],
        ),
        ("age", ["age", "sex_M"], [60.0, 71.5, 80.0], None, [("age", "per unit of age")]),
        ("smoker", ["smoker"], [0, 1, 1], None, [("smoker", "1 vs 0")]),
        ("flag", ["flag"], [True, False], None, [("flag", "1 vs 0")]),
        (NONE, ["arm_B"], ["A", "B"], None, []),
        ("age", ["arm_B"], [1.0, 2.0, 3.0], None, []),  # not among the model's terms
    ],
)
def test_the_comparison_terms_and_how_they_read(
    compare, terms, values, reference, expected
) -> None:
    found = comparison_terms(terms, compare, pd.Series(values), reference)
    assert [(t.term, t.label) for t in found] == expected


def test_the_diagnostics_of_each_model() -> None:
    df = _cohort()
    cox = cox_regression(df, "fu_days", "death", ["arm", "age"], diagnostics=False)
    names = [row[0] for row in diagnostics(cox)]
    assert names == ["Events per term", "Ties"]  # no concordance without the diagnostics
    logit = logistic_regression(df, "hosp", ["arm", "age"], event_level="Yes")
    rows = {row[0]: row[1] for row in diagnostics(logit)}
    assert {"Events per term", "Apparent AUC (in-sample)", "Pseudo-R² (McFadden)", "AIC"} <= set(
        rows
    )
    poisson = count_regression(df, "visits", ["arm"], exposure_col="years")
    assert [row[0] for row in diagnostics(poisson)] == ["Dispersion (Pearson chi²/df)", "AIC"]
    nb = count_regression(df, "visits", ["arm"], exposure_col="years", family="negative_binomial")
    assert diagnostics(nb)[0][0] == "Alpha (negative binomial)"
    assert diagnostics(compare_survival(df, "fu_days", "death", "arm")) == []


def test_one_value_written_several_ways_is_named_before_the_cox_model_fails() -> None:
    """
    The tutorial's sex holds F, "F ", Female, f, female, M, "M ", MALE, m and male, eight of them on one row, four
    without an event: a hazard ratio goes to infinity, statsmodels returned missing values for every term and
    lifelines said "high collinearity". The plan names the levels; the model refuses with the reason.
    """
    df = pd.read_csv(
        Path(__file__).resolve().parents[1] / "assets" / "tutorial" / "tutorial_cohort.csv"
    )
    plan = plan_time_to_event(df, "fu_days", "death", "arm", ["hypertension", "sex"], group=True)
    caution = _advice(plan, "caution")
    assert any(t.startswith("`sex` holds one value written several ways") for t in caution)
    assert any("`F␣`" in t for t in caution)  # "F " reads as the lists show it, not as "F"
    assert any(
        "`sex`: 8 levels held by fewer than 10 rows" in t and "4 of them without an event" in t
        for t in caution
    )
    assert not any("collinear" in t for t in caution)  # the design is of full rank
    with pytest.raises(ValueError, match="no finite estimate"):
        cox_regression(df, "fu_days", "death", ["arm", "hypertension", "sex"])
    curves = plan_time_to_event(df, "fu_days", "death", "site", [], group=True, model=False)
    assert any(
        t.startswith("`site` holds one value written several ways")
        for t in _advice(curves, "caution")
    )


def test_a_variable_with_one_value_and_exact_and_strong_collinearity() -> None:
    df = _cohort()
    df["consent"] = "Yes"
    plan = plan_time_to_event(df, "fu_days", "death", "arm", ["age", "consent"], group=True)
    assert any("`consent` takes one value in the rows used" in t for t in _advice(plan, "info"))
    df["age_months"] = df["age"] * 12 + 6  # the same variable in another unit
    exact = plan_binary(df, "hosp", "Yes", "arm", ["age", "age_months"])
    assert any("exact combination" in t for t in _advice(exact, "violated"))
    df["age_noisy"] = df["age"] + rng.normal(0, 1, len(df))
    strong = plan_count(df, "visits", "years", "arm", ["age", "age_noisy"], "poisson")
    collinear = [t for t in _advice(strong, "caution") if "strongly collinear" in t]
    assert (
        len(collinear) == 1 and "`age`" in collinear[0] and "`age_noisy`" in collinear[0]
    )  # one line per pair
    calm = plan_count(df, "visits", "years", "arm", ["age", "sex"], "poisson")
    assert not any("collinear" in t for t in _advice(calm, "caution"))


def test_an_event_code_from_the_file_is_drawn_as_text() -> None:
    """The refusal quotes the codes it found: drawn as Markdown, an image among them made the browser fetch it."""
    df = _cohort(40)
    df["death"] = df["death"].astype(object)
    df.loc[df.index[:2], "death"] = "![i](https://z.invalid/p.png)"
    plan = plan_time_to_event(df, "fu_days", "death", "arm", [], group=True)
    (text,) = _advice(plan, "violated")
    assert "![i](" not in text and r"\!\[i\]" in text
