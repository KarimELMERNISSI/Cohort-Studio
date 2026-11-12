"""
The analysis plan (M8): the kinds and their titles, a spec that is JSON with an id for its request, one call that
computes a kind from its spec (what the page shows and what a package's check computes again), and a result's record
as named numbers with the counts behind them, in parts whose ids hold no value of the data.
"""

from __future__ import annotations

import importlib.util
import json
import math

import numpy as np
import pandas as pd
import pytest

from biostats import plan
from biostats.mi import MIData, analyze_mi
from biostats.plan import (
    DETERMINISTIC,
    EXACT,
    ITERATIVE,
    ITERATIVE_LOOSE,
    AnalysisSpec,
    ComputedOn,
    ImputationUsed,
    LaterVersion,
    Number,
    Part,
    ResultRecord,
    read_spec,
    specify,
)
from biostats.regression import count_regression, logistic_regression
from biostats.survival import compare_survival, cox_regression, event_coding

TITLES = {
    "survival": "Kaplan-Meier survival and log-rank test",
    "survival_mi": "Kaplan-Meier survival and log-rank test, pooled",
    "cox": "Cox proportional-hazards model",
    "cox_mi": "Cox proportional-hazards model, pooled",
    "logistic": "Logistic regression",
    "logistic_mi": "Logistic regression, pooled",
    "count": "Count regression",
    "count_mi": "Count regression, pooled",
    "univariate": "Group comparison",
    "univariate_mi": "Group comparison, pooled",
    "ancova": "ANCOVA",
    "ancova_mi": "ANCOVA, pooled",
    "paired": "Paired test",
    "paired_mi": "Paired t-test, pooled",
    "one_sample": "One-sample test",
    "one_sample_mi": "One-sample t-test, pooled",
    "mcnemar": "McNemar's test",
    "mantel_haenszel": "Mantel-Haenszel stratified analysis",
    "diagnostic": "Diagnostic accuracy",
    "bland_altman": "Bland-Altman agreement",
}

# the ids a file can hold: a class, a rule, a part's prefix or a number's name is added, never renamed
CLASS_IDS = ("exact", "deterministic", "iterative", "iterative_loose", "seeded")
RULE_IDS = ("plain", "log", "reciprocal")
COX_PARTS = {
    "model": ["rows", "events", "ties", "df_resid", "converged", "rows_left_out", "concordance"],
    "reference": [],
    "term": ["log_hr", "se", "hr", "ci_low", "ci_high", "z", "p", "extreme"],
    "ph": ["statistic", "p", "flagged"],
}
SURVIVAL_PARTS = {
    "curve": [
        "rows", "events", "confidence", "median", "median_lower", "median_upper", "last_time", "times", "at_risk",
        "survival", "lower", "upper", "se", "events_at", "censored_after", "follow_times", "followed",
    ],
    "data": ["groups_skipped", "rows_skipped", "rows_left_out"],
    "test": ["statistic", "df", "p"],
}  # fmt: skip


def _cohort(n: int = 320, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.normal(62, 9, n).round(1),
            "arm": rng.choice(["control", "drug"], n),
            "nyha": rng.choice(["I", "II", "III"], n, p=[0.3, 0.5, 0.2]),
        }
    )
    risk = -3.2 + 0.04 * df["age"] + 0.7 * (df["nyha"] == "III") - 0.5 * (df["arm"] == "drug")
    df["died"] = (rng.random(n) < 1 / (1 + np.exp(-risk))).astype(int)
    df["days"] = rng.exponential(
        np.exp(6.2 - 0.01 * df["age"] + 0.4 * (df["arm"] == "drug"))
    ).round()
    df["years"] = rng.uniform(0.5, 4, n).round(2)
    frailty = rng.gamma(
        2.0, 0.5, n
    )  # over-dispersed counts: the negative binomial model has something to fit
    df["admissions"] = rng.poisson(
        df["years"] * np.exp(-0.8 + 0.3 * (df["nyha"] == "III")) * frailty
    )
    return df


def _imputed(m: int = 5, seed: int = 4) -> MIData:
    truth = _cohort()
    rng = np.random.default_rng(seed)
    base = truth.copy()
    gaps = {"age": rng.random(len(base)) < 0.12, "nyha": rng.random(len(base)) < 0.10}
    base["nyha"] = base["nyha"].astype(object)
    for column, gap in gaps.items():
        base.loc[gap, column] = np.nan
    completed = []
    for _ in range(m):
        one = base.copy()
        one.loc[gaps["age"], "age"] = truth.loc[gaps["age"], "age"] + rng.normal(
            0, 4, gaps["age"].sum()
        )
        one.loc[gaps["nyha"], "nyha"] = rng.choice(["I", "II", "III"], gaps["nyha"].sum())
        completed.append(one)
    return MIData.from_completed(base, completed, methods={"age": "pmm", "nyha": "polyreg"})


def _values(record: ResultRecord) -> dict[str, object]:
    return {f"{p.id}/{n.name}": n.value for p in record.parts for n in p.numbers}


LIFELINES = importlib.util.find_spec("lifelines") is not None  # the optional 'survival' extra


def _without_optional(kind: str, shape: dict[str, list[str]]) -> dict[str, list[str]]:
    """The shape a record has without the optional libraries that are not installed: the parts the kind declares
    they compute ("ph:") are absent; a number of a part they compute ("model/concordance") keeps its name, with no
    value."""
    out = {prefix: list(names) for prefix, names in shape.items()}
    for library, computed in plan.KINDS[kind].optional.items():
        if importlib.util.find_spec(library) is None:
            for item in computed:
                if item.endswith(":"):
                    out.pop(item[:-1], None)
    return out


def _shape(record: ResultRecord) -> dict[str, list[str]]:
    """The names of the numbers of each kind of part (the part's id before its colon or its place)."""
    found: dict[str, list[str]] = {}
    for part in record.parts:
        prefix = part.id.split(":")[0].split("#")[0]
        found.setdefault(prefix, [n.name for n in part.numbers])
    return found


# ----------------------------------------------------------------------------- the kinds and the stored ids


def test_every_stored_key_keeps_its_title() -> None:
    assert plan.titles() == TITLES


def test_a_stored_key_names_its_kind() -> None:
    assert plan.kind_of("cox")[0].id == "cox" and plan.kind_of("cox")[1] is False
    assert plan.kind_of("cox_mi") == (plan.KINDS["cox"], True)
    with pytest.raises(ValueError, match="unknown result key"):
        plan.kind_of("mcnemar_mi")  # never pooled
    with pytest.raises(ValueError, match="unknown result key"):
        plan.kind_of("cox_mi_mi")


def test_the_ids_a_file_can_hold_are_kept() -> None:
    assert tuple(plan.CLASSES) == CLASS_IDS and plan.RULES == RULE_IDS
    df = _cohort()
    cox = plan.record("cox", cox_regression(df, "days", "died", ["arm", "age", "nyha"]))
    assert cox is not None and _shape(cox) == _without_optional("cox", COX_PARTS)
    curves = plan.record("survival", compare_survival(df, "days", "died", "arm"))
    assert curves is not None and _shape(curves) == SURVIVAL_PARTS
    assert [p.id for p in curves.parts] == ["curve#0", "curve#1", "data", "test:log_rank"]


def test_only_the_kinds_with_a_runner_can_be_run() -> None:
    assert [k for k in plan.KINDS if plan.runnable(k)] == ["survival", "cox", "logistic", "count"]
    assert plan.runnable("cox", pooled=True) and not plan.runnable("ancova", pooled=True)
    with pytest.raises(ValueError, match="cannot be computed from a spec yet"):
        plan.run(AnalysisSpec("ancova"), _cohort())


# ----------------------------------------------------------------------------- the spec


def test_a_spec_is_json_with_an_id_for_its_request() -> None:
    spec = specify("cox", time="days", event="died", compare="arm", adjust=("age", "nyha"))
    assert spec.options == {"alpha": 0.05, "ties": "efron"}
    assert spec.variables["adjust"] == ["age", "nyha"]  # a tuple is written as a list
    assert spec.covariates() == ["arm", "age", "nyha"]
    again = AnalysisSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert again == spec and again.fingerprint == spec.fingerprint
    assert len(spec.fingerprint) == 64 and len(spec.request_id) == 64
    other = specify("cox", time="days", event="died", compare="arm", adjust=("age",))
    assert other.request_id != spec.request_id


def test_the_request_id_is_the_same_whatever_it_ran_on_and_the_fingerprint_is_not() -> None:
    """A package meant to travel names a result by its request: the whole spec's fingerprint covers the rows'."""
    df = _cohort()
    spec = specify("cox", time="days", event="died", compare="arm", adjust=["age"])
    done = plan.run(spec, df)
    assert (
        done.spec.request_id == spec.request_id
    )  # the decisions and the libraries are no part of the request
    here = done.spec.on(
        ComputedOn("observed", "aaaa", 320, 320, rows_fp="1111", trace_head="h1", trace_steps=3)
    )
    there = done.spec.on(
        ComputedOn("observed", "aaaa", 320, 320, rows_fp="2222", trace_head="h2", trace_steps=4)
    )
    assert here.request_id == there.request_id == spec.request_id
    assert here.fingerprint != there.fingerprint
    assert specify(
        "cox", time="days", event="died", compare=None, adjust=["arm", "age"]
    ).request_id != (
        spec.request_id
    )  # the same model, another request: which variable is compared is part of it


def test_a_spec_s_copy_can_be_changed_without_changing_the_spec() -> None:
    spec = specify("cox", time="days", event="died", compare="arm", adjust=["age", "nyha"])
    done = plan.run(spec, _cohort())
    fingerprint = done.spec.fingerprint
    written = done.spec.to_dict()
    written["variables"]["adjust"].append("sex")
    written["decided"]["levels"]["nyha"].pop()
    assert done.spec.variables["adjust"] == ["age", "nyha"]
    assert done.spec.decided["levels"]["nyha"] == ["I", "II", "III"]
    assert done.spec.fingerprint == fingerprint


def test_the_comparison_is_named_once_among_the_covariates() -> None:
    spec = specify("logistic", outcome="died", compare="arm", adjust=["arm", "age"])
    assert spec.covariates() == ["arm", "age"]
    assert specify("logistic", outcome="died", compare=None, adjust=["age"]).covariates() == ["age"]


def test_a_spec_refuses_what_its_kind_does_not_know() -> None:
    with pytest.raises(ValueError, match="unknown analysis kind"):
        AnalysisSpec("kaplan")
    with pytest.raises(ValueError, match="has no role outcome"):
        specify("cox", time="days", event="died", outcome="x")
    with pytest.raises(ValueError, match="has no option family"):
        specify("cox", time="days", event="died", options={"family": "poisson"})
    with pytest.raises(ValueError, match="never pooled"):
        AnalysisSpec("mcnemar", pooled=True)
    with pytest.raises(TypeError, match="not a JSON value: DataFrame"):
        specify("cox", time="days", event=pd.DataFrame())


def test_the_pooled_options_belong_to_the_pooled_spec_only() -> None:
    assert specify("cox", pooled=True, time="t", event="e").options["ph_test_imputations"] == 5
    with pytest.raises(ValueError, match="has no option ph_test_imputations"):
        specify("cox", time="t", event="e", options={"ph_test_imputations": 3})


def test_what_it_was_computed_on_has_a_view_that_can_travel() -> None:
    """The fingerprints of the rows, of the trace and of the imputed cells name rows: a guess can be tried against
    them. The view for a package meant to travel holds none of them."""
    on = ComputedOn(
        "imputed",
        "datafp",
        928,
        928,
        rows_fp="rowsfp",
        protocol_hash="p" * 64,
        rules_version=2,
        trace_head="head",
        trace_steps=7,
        imputation=ImputationUsed("cellsfp", 20, True, 5, "stephash"),
    )
    assert ComputedOn.from_dict(json.loads(json.dumps(on.to_dict()))) == on
    shared = on.shareable()
    assert shared == {
        "mode": "imputed",
        "data_fp": "datafp",
        "n_population": 928,
        "n_analysed": 928,
        "rules_version": 2,
        "trace_steps": 7,
        "imputation": {"m": 20, "subset": True, "step_index": 5},
    }
    text = json.dumps(shared)
    assert not any(secret in text for secret in ("rowsfp", "cellsfp", "stephash", "head", "pppp"))
    with pytest.raises(ValueError, match="observed or on imputed"):
        ComputedOn("both", "d", 1)


def test_a_received_spec_is_read_strictly_and_says_why_it_cannot_be() -> None:
    done = plan.run(
        specify("cox", time="days", event="died", compare="arm", adjust=["age"]), _cohort()
    )
    written = json.loads(json.dumps(done.spec.to_dict()))
    assert read_spec(written) == (done.spec, "")
    assert read_spec("cox") == (None, "not an analysis spec")
    assert read_spec({**written, "pooled": "false"}) == (None, "pooled is not of the type expected")
    assert read_spec({**written, "variables": ["days"]}) == (
        None,
        "variables is not of the type expected",
    )
    later_kind = read_spec({**written, "kind": "incidence"})
    assert later_kind[0] is None and later_kind[1].startswith("unknown analysis kind 'incidence'")
    later_option = read_spec({**written, "options": {**written["options"], "robust": True}})
    assert later_option[0] is None and "has no option robust" in later_option[1]
    later = read_spec({**written, "kind_version": 2})
    assert later == (None, "the cox analysis was specified by a later version of CohortStudio")
    with pytest.raises(LaterVersion):
        plan.run(AnalysisSpec.from_dict({**written, "kind_version": 2}), _cohort())


# ----------------------------------------------------------------------------- one call computes a kind


def test_a_spec_computes_what_the_direct_call_computes() -> None:
    df = _cohort()
    done = plan.run(
        specify("cox", time="days", event="died", compare="arm", adjust=["age", "nyha"]), df
    )
    direct = cox_regression(df, "days", "died", ["arm", "age", "nyha"])
    pd.testing.assert_frame_equal(done.result.table, direct.table)
    assert done.spec.decided == {
        "levels": {"arm": ["control", "drug"], "nyha": ["I", "II", "III"]},
        "event_values": {"event": ["1"], "censored": ["0"]},
        "without_term": [],
    }
    assert dict(done.derived) == dict(done.spec.decided)  # a first run: what the data give
    assert done.spec.backend["statsmodels"] and "lifelines" in done.spec.backend

    km = plan.run(specify("survival", time="days", event="died", group="arm"), df)
    curves = compare_survival(df, "days", "died", "arm")
    assert km.result.logrank_p == curves.logrank_p
    assert (
        km.spec.decided["groups"] == ["control", "drug"] and km.spec.decided["groups_skipped"] == 0
    )

    logit = plan.run(specify("logistic", outcome="died", compare="arm", adjust=["age"]), df)
    pd.testing.assert_frame_equal(
        logit.result.table, logistic_regression(df, "died", ["arm", "age"]).table
    )
    assert logit.spec.decided["outcome_levels"] == ["0", "1"]
    assert (
        logit.spec.decided["event_level"] == "1"
    )  # none was asked for: the second level, and it is recorded

    counts = plan.run(
        specify(
            "count",
            count="admissions",
            person_time="years",
            compare="nyha",
            adjust=[],
            options={"family": "negative_binomial"},
        ),
        df,
    )
    direct_count = count_regression(
        df, "admissions", ["nyha"], exposure_col="years", family="negative_binomial"
    )
    pd.testing.assert_frame_equal(counts.result.table, direct_count.table)


def test_a_run_from_a_recorded_spec_uses_the_levels_it_recorded() -> None:
    """Replayed as recorded: the same terms and the same reference, whatever rule a later version reads the levels
    by; what the data give on their own is returned beside, for a check to say when it differs."""
    df = _cohort()
    first = plan.run(
        specify("cox", time="days", event="died", compare="arm", adjust=["age", "nyha"]), df
    )
    again = plan.run(first.spec, df)
    pd.testing.assert_frame_equal(again.result.table, first.result.table)
    assert again.spec.decided == first.spec.decided and dict(again.derived) == dict(
        first.spec.decided
    )
    # a spec whose recorded reference is another level: the model is fitted with it
    other = AnalysisSpec.from_dict(
        {
            **first.spec.to_dict(),
            "decided": {
                **first.spec.to_dict()["decided"],
                "levels": {"arm": ["drug", "control"], "nyha": ["I", "II", "III"]},
            },
        }
    )
    turned = plan.run(other, df)
    assert "arm_control" in set(turned.result.table["term"]) and "arm_drug" not in set(
        turned.result.table["term"]
    )
    assert turned.spec.decided["levels"]["arm"] == ["drug", "control"]
    assert turned.derived["levels"]["arm"] == [
        "control",
        "drug",
    ]  # the data's own order, said beside
    # a constant covariate is dropped on the run from the record as on the first run
    constant = df.assign(centre=1.0)
    start = plan.run(
        specify("logistic", outcome="died", compare="arm", adjust=["age", "centre"]), constant
    )
    assert start.spec.decided["without_term"] == ["centre"]
    replay = plan.run(start.spec, constant)
    pd.testing.assert_frame_equal(replay.result.table, start.result.table)


def test_a_pooled_spec_computes_what_the_direct_pooling_computes() -> None:
    mi = _imputed()
    spec = specify(
        "cox", pooled=True, time="days", event="died", compare="arm", adjust=["age", "nyha"]
    )
    done = plan.run(spec, mi=mi)
    levels = mi.levels(["arm", "age", "nyha"])
    direct = analyze_mi(
        lambda d: cox_regression(
            d,
            "days",
            "died",
            ["arm", "age", "nyha"],
            levels=levels,
            diagnostics=d.attrs["imputation"] <= 5,
        ),
        mi,
        columns=["days", "died", "arm", "age", "nyha"],
    )
    assert done.result.table.estimates == direct.table.estimates
    assert [t.p_value for t in done.result.table.terms] == [t.p_value for t in direct.table.terms]
    assert done.spec.decided["levels"] == levels
    assert done.spec.backend["imputation_engine"]
    # the decisions a spec holds are used: a level the data does not hold stops the run with the reason
    written = done.spec.to_dict()
    written["decided"]["levels"] = {"arm": ["control", "drug"], "nyha": ["I", "II"]}
    with pytest.raises(Exception, match="outside the levels"):
        plan.run(AnalysisSpec.from_dict(written), mi=mi)
    with pytest.raises(ValueError, match="needs the imputed data"):
        plan.run(spec, _cohort())


def test_the_event_values_are_recorded_as_they_are_read() -> None:
    assert event_coding(["Died", "alive", None, "ALIVE", "dead"]) == {
        "event": ["dead", "died"],
        "censored": ["alive"],
    }
    assert event_coding([1.0, 0.0, 1.0, np.nan]) == {"event": ["1"], "censored": ["0"]}
    assert event_coding([True, False]) == {"event": ["true"], "censored": ["false"]}


# ----------------------------------------------------------------------------- the record


def test_a_cox_record_names_its_numbers_and_the_counts_behind_each_term() -> None:
    df = _cohort()
    result = cox_regression(df, "days", "died", ["arm", "age", "nyha"])
    record = plan.record("cox", result)
    assert (
        record is not None
        and record.kind == "cox"
        and not record.pooled
        and record.record_version == 1
    )
    model = record.part("model")
    assert model.number("rows").value == len(df) and model.number("rows").cls == EXACT
    assert model.support == {
        "rows": len(df),
        "events": int(df["died"].sum()),
        "censored": len(df)
        - int(df["died"].sum()),  # a cohort where everybody had the event says so of each
        "df_resid": int(result.df_resid),
    }
    assert model.number("converged").value is True
    concordance = model.number("concordance")  # lifelines' own fit: named without it, with no value
    assert concordance.cls == ITERATIVE_LOOSE and (concordance.value is not None) == LIFELINES
    third = record.part("term:nyha#2")
    assert third.labels == {"variable": "nyha", "level": "III"}
    at_level = df["nyha"] == "III"
    assert third.support == {
        "rows": int(at_level.sum()),
        "events": int(df.loc[at_level, "died"].sum()),
        "censored": int(at_level.sum()) - int(df.loc[at_level, "died"].sum()),
    }
    row = result.table.set_index("term").loc["nyha_III"]
    estimate = third.number("log_hr")
    assert (
        estimate.value == row["log_hr"]
        and estimate.cls == ITERATIVE
        and estimate.scale == row["se"]
    )
    assert third.number("p").rule == plan.LOG and third.number("extreme").value is False
    age = record.part("term:age")
    # a covariate of many values: a ratio for one unit more, with the model's counts behind it
    assert age.labels == {"variable": "age", "per": "unit"} and age.support == {
        "rows": len(df),
        "events": int(df["died"].sum()),
        "censored": len(df) - int(df["died"].sum()),
    }
    reference = record.part("reference:nyha")
    assert reference.labels == {"variable": "nyha", "level": "I"} and reference.numbers == ()
    assert reference.support["rows"] == int((df["nyha"] == "I").sum())
    if LIFELINES:
        assert record.part("ph:nyha#2").number("flagged").cls == EXACT
    else:  # the proportional-hazards tests are lifelines': without it, no such part
        assert not [p.id for p in record.parts if p.id.startswith("ph:")]


def test_no_part_s_id_holds_a_value_of_the_data() -> None:
    """A level's name is a value of the data: it is a label, which a shared package drops with a part it withholds,
    never a part of an id that a file stores."""
    df = _cohort()
    df["site"] = np.where(np.arange(len(df)) < 5, "CANARY-SITE", "Esch")
    df["group"] = np.where(np.arange(len(df)) % 2 == 0, "CANARY-GROUP", "other")
    records = [
        plan.record("cox", cox_regression(df, "days", "died", ["site", "age"])),
        plan.record("logistic", logistic_regression(df, "died", ["site", "age"])),
        plan.record("survival", compare_survival(df, "days", "died", "group")),
    ]
    for record in records:
        assert record is not None
        ids = " ".join(p.id for p in record.parts)
        assert "CANARY" not in ids and "Esch" not in ids
        without_labels = [{**p.to_dict(), "labels": {}} for p in record.parts]
        assert "CANARY" not in json.dumps(without_labels)  # nowhere but in the labels


def test_two_factors_that_give_one_name_to_a_term_keep_their_own_parts() -> None:
    """`a` with a level `b_d` and `a_b` with a level `d` both give the design a column `a_b_d`: the record stopped on
    two parts of one id, so that study had no package."""
    df = _cohort()
    rng = np.random.default_rng(2)
    df["a"] = rng.choice(["a0", "b_d"], len(df))
    df["a_b"] = rng.choice(["a0", "d"], len(df))
    record = plan.record("logistic", logistic_regression(df, "died", ["a", "a_b"]))
    assert record is not None
    assert [p.id for p in record.parts if p.id.startswith("term:")] == ["term:a#1", "term:a_b#1"]
    assert record.part("term:a#1").labels == {"variable": "a", "level": "b_d"}
    assert record.part("term:a_b#1").labels == {"variable": "a_b", "level": "d"}
    assert record.part("term:a#1").support["rows"] == int((df["a"] == "b_d").sum())


def test_a_covariate_coded_0_and_1_has_the_counts_of_its_two_cells() -> None:
    """A 0/1 exposure is read as a number: its term carried the whole model's counts, so one participant coded 1
    passed any small-cell rule while the coefficient's sign gave their outcome."""
    df = _cohort()
    rare = df.index[:6]
    df["transplanted"] = 0
    df.loc[rare, "transplanted"] = 1
    df["smoker"] = np.arange(len(df)) % 3 == 0  # True and False: numbers too
    df["as_text"] = np.where(df["transplanted"] == 1, "yes", "no")
    result = logistic_regression(df, "died", ["transplanted", "smoker", "age"])
    record = plan.record("logistic", result)
    assert record is not None
    events = int(df.loc[rare, "died"].sum())
    term = record.part("term:transplanted")
    assert term.labels == {"variable": "transplanted", "level": "1"}
    assert term.support == {"rows": 6, "events": events, "non_events": 6 - events}
    reference = record.part("reference:transplanted")
    assert reference.labels == {"variable": "transplanted", "level": "0"}
    assert reference.support["rows"] == len(df) - 6
    assert record.part("term:smoker").labels == {"variable": "smoker", "level": "True"}
    assert record.part("reference:smoker").support["rows"] == int((~df["smoker"]).sum())
    # many values: a ratio per unit, with the model's counts
    assert record.part("term:age").labels == {"variable": "age", "per": "unit"}
    assert record.part("term:age").support["rows"] == len(df)
    # the same variable as text has the same counts behind its term
    as_text = plan.record("logistic", logistic_regression(df, "died", ["as_text", "age"]))
    assert as_text is not None and as_text.part("term:as_text#1").support == term.support
    cox = plan.record(
        "cox", cox_regression(df, "days", "died", ["transplanted", "age"], diagnostics=False)
    )
    assert cox is not None and cox.part("term:transplanted").support == {
        "rows": 6,
        "events": events,
        "censored": 6 - events,
    }


def test_a_count_model_counts_the_rows_with_an_event() -> None:
    """In a count model the events of a level can all be one participant's: events are not people."""
    df = _cohort()
    df["admissions"] = 0
    df.loc[df.index[0], "admissions"] = (
        12  # one participant of class III holds every admission of the class
    )
    df.loc[df.index[0], "nyha"] = "III"
    df.loc[df["nyha"] != "III", "admissions"] = np.arange(int((df["nyha"] != "III").sum())) % 3
    record = plan.record(
        "count", count_regression(df, "admissions", ["nyha"], exposure_col="years")
    )
    assert record is not None
    third = record.part("term:nyha#2")
    assert third.support["events"] == 12 and third.support["rows_with_events"] == 1
    assert "non_events" not in third.support
    # and the rows without one, in the level and in the model: both sides of what a shared result must hold
    assert third.support["rows_without_events"] == third.support["rows"] - 1
    with_events = int((df["admissions"] > 0).sum())
    model = record.part("model").support
    assert (model["rows_with_events"], model["rows_without_events"]) == (
        with_events,
        len(df) - with_events,
    )
    assert model["events"] == int(df["admissions"].sum()) > with_events


def test_a_logistic_record_counts_the_rows_without_the_event() -> None:
    df = _cohort()
    result = logistic_regression(df, "died", ["nyha", "age"], event_level="1")
    record = plan.record("logistic", result)
    assert record is not None
    second = record.part("term:nyha#1")
    at_level = df["nyha"] == "II"
    events = int(df.loc[at_level, "died"].sum())
    assert second.support == {
        "rows": int(at_level.sum()),
        "events": events,
        "non_events": int(at_level.sum()) - events,
    }
    assert record.part("model").number("auc").value == result.auc


def test_a_negative_binomial_fit_ends_at_its_optimum() -> None:
    """Its record is compared like any other fit: finished with Newton steps, the same rows in another order give
    the same estimates (the BFGS fit alone stopped some thousandth of a standard error away, another one each
    path)."""
    df = _cohort()
    one = count_regression(
        df, "admissions", ["nyha", "age"], exposure_col="years", family="negative_binomial"
    )
    two = count_regression(
        df.sample(frac=1, random_state=3),
        "admissions",
        ["nyha", "age"],
        exposure_col="years",
        family="negative_binomial",
    )
    a, b = one.table.set_index("term"), two.table.set_index("term")
    assert float(((a["coef"] - b["coef"]).abs() / a["se"]).max()) < 1e-8
    assert one.dispersion == pytest.approx(two.dispersion, rel=1e-8)
    assert one.converged and one.dispersion_se is not None and one.dispersion_se > 0
    record = plan.record("count", one)
    assert record is not None
    assert record.part("term:nyha#2").number("coef").cls == ITERATIVE
    dispersion = record.part("model").number("dispersion")
    assert dispersion.cls == ITERATIVE and dispersion.scale == one.dispersion_se
    assert record.part("model").number("family").value == "negative_binomial"


def test_a_survival_record_marks_what_is_one_participant_s() -> None:
    df = _cohort()
    df.loc[df.index[7], ["days", "died"]] = [98765.0, 1]  # one participant's day of death
    record = plan.record("survival", compare_survival(df, "days", "died", "arm"))
    assert record is not None
    holding = [
        (part.id, number.name, number.fine)
        for part in record.parts
        for number in part.numbers
        if 98765.0 in (number.value if isinstance(number.value, list) else [number.value])
    ]
    # the curve's steps, the times its follow-ups end and the last of them: a shared package coarsens or drops
    # them
    assert {name for _, name, _ in holding} == {"times", "last_time", "follow_times"}
    assert all(fine for _, _, fine in holding)
    curve = record.part("curve#0")
    assert curve.labels == {"group": "control"}
    assert curve.number("survival").cls == DETERMINISTIC and curve.number("times").cls == EXACT
    assert curve.number("confidence").value == 0.95 and not curve.number("confidence").fine
    # a median is an event time, or the middle of two: one participant's too
    assert all(curve.number(name).fine for name in ("median", "median_lower", "median_upper"))
    # the events at each step and the censorings until the next add up to the group
    events, censored = curve.number("events_at").value, curve.number("censored_after").value
    assert sum(events) == curve.support["events"]
    assert curve.number("at_risk").value[0] - sum(events) - sum(censored) == 0
    assert record.part("test:log_rank").number("df").value == 1
    assert record.part("data").number("groups_skipped").value == 0
    # every time a follow-up ends, by an event or a censoring, with the rows still followed at it
    ends, followed = curve.number("follow_times").value, curve.number("followed").value
    assert curve.number("follow_times").fine and curve.number("followed").fine
    assert ends == sorted(set(ends)) and set(curve.number("times").value) <= set(ends)
    assert followed[0] == curve.support["rows"] and followed == sorted(followed, reverse=True)
    control = df[df["arm"] == "control"]
    assert followed[len(ends) // 2] == int((control["days"] >= ends[len(ends) // 2]).sum())


def test_a_record_holds_no_residual_and_no_covariate_value() -> None:
    df = _cohort()
    df.loc[df.index[3], "age"] = 71.123456  # no other row holds it
    for key, result in (
        ("cox", cox_regression(df, "days", "died", ["arm", "age"])),
        ("logistic", logistic_regression(df, "died", ["arm", "age"], event_level="1")),
        ("count", count_regression(df, "admissions", ["age"], exposure_col="years")),
    ):
        record = plan.record(key, result)
        assert record is not None and "71.123456" not in record.to_json(), key


def test_a_record_reads_back_as_it_was_written() -> None:
    df = _cohort()
    df.loc[df["arm"] == "drug", "died"] = df.loc[df["arm"] == "drug", "died"].where(
        df["days"] < df["days"].max(), 1
    )
    for key, result in (
        ("survival", compare_survival(df, "days", "died", "arm")),
        ("cox", cox_regression(df, "days", "died", ["arm", "age", "nyha"])),
        ("logistic", logistic_regression(df, "died", ["arm", "age"], event_level="1")),
        ("count", count_regression(df, "admissions", ["nyha"], exposure_col="years")),
    ):
        record = plan.record(key, result)
        assert record is not None
        text = record.to_json()
        again = ResultRecord.from_json(text)
        assert again.to_json() == text, key
        for name, value in _values(record).items():
            read = _values(again)[name]
            if isinstance(value, float) and math.isnan(value):
                assert isinstance(read, float) and math.isnan(read), name
            elif isinstance(value, list):
                assert len(read) == len(value), name
                assert all(
                    (a == b) or (isinstance(a, float) and math.isnan(a) and math.isnan(b))
                    for a, b in zip(value, read, strict=True)
                ), name
            else:
                assert read == value and type(read) is type(value), name


def test_a_number_that_is_not_finite_is_written_as_a_marker() -> None:
    part = Part("p", numbers=(Number("a", math.nan), Number("b", [1.5, math.inf, -math.inf, None])))
    text = ResultRecord("cox", parts=(part,)).to_json()
    assert '{"float":"nan"}' in text and '{"float":"-inf"}' in text and "NaN" not in text
    again = ResultRecord.from_json(text).part("p")
    assert math.isnan(again.number("a").value)
    assert again.number("b").value == [1.5, math.inf, -math.inf, None]
    with pytest.raises(ValueError, match="unknown comparison class"):
        Number("a", 1.0, "approximate")
    with pytest.raises(ValueError, match="share an id"):
        ResultRecord("cox", parts=(Part("p"), Part("p")))
    with pytest.raises(ValueError, match="whole numbers"):
        Part("p", support={"rows": 9.9})
    with pytest.raises(ValueError, match="fine is not of the type expected"):
        Number.from_dict(
            {
                "name": "a",
                "value": 1,
                "class": "exact",
                "scale": None,
                "rule": "plain",
                "fine": "no",
            }
        )


def test_a_pooled_record_keeps_each_imputation_and_the_smallest_counts() -> None:
    mi = _imputed()
    done = plan.run(
        specify(
            "cox", pooled=True, time="days", event="died", compare="arm", adjust=["age", "nyha"]
        ),
        mi=mi,
    )
    record = plan.record("cox_mi", done.result)
    assert record is not None and record.pooled
    pooling = record.part("pooling")
    assert pooling.number("m").value == 5 and pooling.number("partial").value is False
    model = record.part("model")
    # the model's counts are the smallest over the imputed datasets, and are named so
    assert model.number("converged_in").value == 5
    assert set(model.support) == {"rows_min", "events_min", "censored_min", "df_resid_min"}
    assert model.number("rows_left_out").value == 0 and model.number("rows_left_out").fine
    term = record.part("term:nyha#2")
    assert term.labels == {"variable": "nyha", "level": "III"}
    assert len(term.number("estimates").value) == 5
    assert term.number("df").rule == plan.RECIPROCAL
    smallest = min(
        int((data["nyha"] == "III").sum()) for _, data in mi.completed(["days", "died", "nyha"])
    )
    assert term.support["rows"] == smallest
    test = record.part("test:nyha")
    assert test.number("method").value == "D1"
    assert test.number("terms").value == [
        "term:nyha#1",
        "term:nyha#2",
    ]  # by their parts, never by a level's name
    assert test.number("df1").cls == EXACT
    assert record.complete_case is not None and record.complete_case.kind == "cox"
    assert ResultRecord.from_json(record.to_json()).to_json() == record.to_json()

    logit = plan.run(
        specify("logistic", pooled=True, outcome="died", compare="arm", adjust=["nyha"]), mi=mi
    )
    pooled = plan.record("logistic_mi", logit.result)
    assert pooled is not None
    assert pooled.part("model").number("family").value == "logistic"
    assert set(pooled.part("term:nyha#1").support) == {"rows", "events", "non_events"}
    assert "term:const" not in [p.id for p in pooled.parts]  # the intercept is no term of a result

    curves = plan.run(
        specify("survival", pooled=True, time="days", event="died", group="arm"), mi=mi
    )
    km = plan.record("survival_mi", curves.result)
    assert km is not None
    drug = next(p for p in km.parts if p.labels.get("group") == "drug")
    assert drug.id == "curve#1" and drug.number("times").fine and drug.number("median").fine
    # the smallest group, the fewest events and the fewest rows without the event over the datasets: three
    # counts that do not add up, and are named for what they are
    assert set(drug.support) == {"rows_min", "events_min", "censored_min"}
    # the fewest rows still followed at every time a follow-up ends in a dataset
    ends, followed = drug.number("follow_times"), drug.number("followed_min")
    assert ends.fine and followed.fine and len(ends.value) == len(followed.value)
    assert followed.value[0] == drug.support["rows_min"]
    assert followed.value == sorted(followed.value, reverse=True)
    data = km.part("data")
    assert [(n.name, n.value, n.fine) for n in data.numbers] == [
        ("rows_skipped", 0, True),
        ("rows_left_out", 0, True),
        (
            "rows_min",
            len(mi.base),
            False,
        ),  # the rows the analysis read, in the dataset that has the fewest
    ]
    # the events up to each time in each of the 5 datasets: the events between two times are counted in each
    each = drug.number("events_so_far_each")
    assert each.fine and len(each.value) == 5
    assert all(len(one) == len(drug.number("times").value) for one in each.value)
    assert [min(column) for column in zip(*each.value, strict=True)] == drug.number(
        "events_so_far_min"
    ).value
    so_far, at_risk = drug.number("events_so_far_min").value, drug.number("at_risk_min").value
    assert len(so_far) == len(drug.number("times").value) == len(at_risk)
    assert so_far == sorted(so_far) and at_risk == sorted(at_risk, reverse=True)
    assert so_far[-1] == drug.support["events_min"]
    test = km.part("test:log_rank")
    assert test.number("method").value == "D2"
    assert test.support == {
        "rows_min": len(mi.base),
        "events_min": int(mi.base["died"].sum()),
    }  # every row, in every dataset


def test_a_covariate_of_a_few_values_has_the_counts_of_each_value() -> None:
    """A dose that is 0 for all but two participants is read as a number: its term carried the whole model's counts,
    while its coefficient is those two participants' outcome. A score from 0 to 8 is a table as well: each of its
    values is a cell, which the written numbers of a model of such covariates give back."""
    from biostats.design import FEW_VALUES, value_counts

    df = _cohort()
    df["dose"] = 0.0
    df.loc[df.index[:2], "dose"] = [1.0, 3.0]
    df["score"] = np.arange(len(df)) % 8  # 0 to 7
    df.loc[df.index[:2], "score"] = 8  # and two participants at 8
    df["transplanted"] = (np.arange(len(df)) % 9 == 0).astype(int)
    cells = value_counts(df, ["dose", "score", "age", "transplanted", "arm"], df["died"])
    # neither a factor, nor a covariate of two values (an indicator), nor one of more than twenty values
    assert list(cells) == ["dose", "score"] and df["age"].nunique() > FEW_VALUES
    died = [int(df.loc[row, "died"]) for row in df.index[:2]]
    others = int(df["died"].sum()) - sum(died)
    assert cells["dose"] == {
        "0": (len(df) - 2, others, others),
        "1": (1, died[0], died[0]),
        "3": (1, died[1], died[1]),
    }
    assert list(cells["score"]) == [str(value) for value in range(9)]  # in increasing order
    assert cells["score"]["8"] == (2, sum(died), sum(died))
    record = plan.record("logistic", logistic_regression(df, "died", ["dose", "age"]))
    assert record is not None
    # a slope: a ratio per unit, with the smallest of each count over its values behind it
    term = record.part("term:dose")
    assert term.labels == {"variable": "dose", "per": "unit"}
    assert term.support == {"rows": 1, "events": min(died), "non_events": 1 - max(died)}
    # the other cell says what it is: every value of the covariate is a cell, and the term is no contrast with it
    reference = record.part("reference:dose")
    assert reference.labels == {"variable": "dose", "level": "0", "of": "values"}
    assert reference.support["rows"] == len(df) - 2
    by_score = plan.record("logistic", logistic_regression(df, "died", ["score", "age"]))
    assert by_score is not None and by_score.part("term:score").support["rows"] == 2
    assert by_score.part("reference:score").labels == {
        "variable": "score",
        "level": "0",
        "of": "values",
    }
    for key, result in (
        ("cox", cox_regression(df, "days", "died", ["dose", "age"], diagnostics=False)),
        ("count", count_regression(df, "admissions", ["dose", "age"], exposure_col="years")),
    ):
        other = plan.record(key, result)
        assert other is not None and other.part("term:dose").support["rows"] == 1, key
        assert other.part("reference:dose").support["rows"] == len(df) - 2, key


def test_a_covariate_that_most_rows_hold_at_one_value_has_the_counts_on_and_off_it() -> None:
    """A covariate of many values that is 0 for all but thirty participants: its coefficient is carried by the rows
    off its commonest value, while its term would carry the whole model's counts."""
    from biostats.design import OTHER_VALUES, dominant_counts

    df = _cohort()
    off = df.index[:30]
    df["dose"] = 0.0
    df.loc[off, "dose"] = 10.0 + np.arange(30)  # thirty values, one participant each
    df["packs"] = np.where(np.arange(len(df)) % 2 == 0, 0.0, 1.0 + np.arange(len(df)))
    df["score"] = np.arange(len(df)) % 9
    cells = dominant_counts(df, ["dose", "packs", "age", "score", "arm"], df["died"])
    # neither a factor, nor a covariate of a few values (counted value by value), nor one that no value holds for
    # half of the rows
    assert list(cells) == ["dose", "packs"]
    died = int(df.loc[off, "died"].sum())
    rest = int(df["died"].sum()) - died
    assert cells["dose"] == {"0": (len(df) - 30, rest, rest), OTHER_VALUES: (30, died, died)}
    assert cells["packs"]["0"][0] == cells["packs"][OTHER_VALUES][0] == len(df) // 2  # exactly half
    record = plan.record("logistic", logistic_regression(df, "died", ["dose", "age"]))
    assert record is not None
    assert record.part("term:dose").labels == {"variable": "dose", "per": "unit"}
    assert record.part("term:dose").support == {
        "rows": 30,
        "events": died,
        "non_events": 30 - died,
    }
    assert record.part("reference:dose").labels == {
        "variable": "dose",
        "level": "0",
        "of": "commonest",
    }
    for key, result in (
        ("cox", cox_regression(df, "days", "died", ["dose", "age"], diagnostics=False)),
        ("count", count_regression(df, "admissions", ["dose", "age"], exposure_col="years")),
    ):
        other = plan.record(key, result)
        assert other is not None and other.part("term:dose").support["rows"] == 30, key
        assert other.part("reference:dose").support["rows"] == len(df) - 30, key


def test_two_values_that_are_not_one_unit_apart_make_a_ratio_per_unit() -> None:
    """A dose coded 0 or 50 has two values, and its term read "50 against 0" beside an odds ratio for one unit of
    dose: the ratio of 50 against 0 is that one to the power 50."""
    df = _cohort()
    df["dose"] = np.where(np.arange(len(df)) % 4 == 0, 50, 0)
    df["exposed"] = (df["dose"] > 0).astype(int)
    by_dose = plan.record("logistic", logistic_regression(df, "died", ["dose", "age"]))
    by_flag = plan.record("logistic", logistic_regression(df, "died", ["exposed", "age"]))
    assert by_dose is not None and by_flag is not None
    term, flag = by_dose.part("term:dose"), by_flag.part("term:exposed")
    assert term.labels == {"variable": "dose", "per": "unit"}
    assert flag.labels == {
        "variable": "exposed",
        "level": "1",
    }  # one unit apart: the higher against the lower
    # the cell behind the term is the higher value's either way, and the other cell says that it is no reference
    assert term.support == flag.support and term.support["rows"] == len(df) // 4
    assert by_dose.part("reference:dose").labels == {
        "variable": "dose",
        "level": "0",
        "of": "values",
    }
    assert by_flag.part("reference:exposed").labels == {"variable": "exposed", "level": "0"}
    # one fit, written in two units
    assert flag.number("coef").value == pytest.approx(50 * term.number("coef").value, rel=1e-4)
    for key, result in (
        ("cox", cox_regression(df, "days", "died", ["dose", "age"], diagnostics=False)),
        ("count", count_regression(df, "admissions", ["dose", "age"], exposure_col="years")),
    ):
        other = plan.record(key, result)
        assert other is not None, key
        assert other.part("term:dose").labels == {"variable": "dose", "per": "unit"}, key
        assert other.part("reference:dose").labels["of"] == "values", key
    # True and False are one unit apart
    df["smoker"] = np.arange(len(df)) % 3 == 0
    smoker = plan.record("logistic", logistic_regression(df, "died", ["smoker", "age"]))
    assert smoker is not None
    assert smoker.part("term:smoker").labels == {"variable": "smoker", "level": "True"}
    assert "of" not in smoker.part("reference:smoker").labels


def test_a_pooled_term_s_cells_are_those_of_its_recorded_values() -> None:
    """One participant coded 1, and gaps filled by a normal draw: no imputed dataset keeps the covariate's two
    values, its term took the model's counts, and an odds ratio that is one participant's outcome passed any
    small-cell rule."""
    truth = _cohort()
    truth["transplanted"] = 0.0
    truth.loc[truth.index[5], "transplanted"] = 1.0
    truth["prior"] = 0.0
    truth.loc[truth.index[10:16], "prior"] = [1, 1, 2, 1, 3, 1]
    rng = np.random.default_rng(9)
    base = truth.copy()
    gap = rng.random(len(base)) < 0.12
    gap[:20] = False
    base.loc[gap, ["transplanted", "prior"]] = np.nan
    completed = []
    for _ in range(5):
        one = base.copy()
        one.loc[gap, "transplanted"] = rng.normal(0.0, 0.05, gap.sum())
        one.loc[gap, "prior"] = rng.normal(0.1, 0.05, gap.sum())
        completed.append(one)
    mi = MIData.from_completed(base, completed, methods={"transplanted": "norm", "prior": "norm"})
    recorded = int((~gap).sum())
    for kind, roles, without in (
        ("logistic", {"outcome": "died"}, "non_events"),
        ("cox", {"time": "days", "event": "died"}, "censored"),
    ):
        done = plan.run(
            specify(
                kind, pooled=True, compare="arm", adjust=["transplanted", "prior", "age"], **roles
            ),
            mi=mi,
        )
        record = plan.record(f"{kind}_mi", done.result)
        assert record is not None
        term = record.part("term:transplanted")
        assert term.labels == {"variable": "transplanted", "level": "1"}, kind
        died = int(truth.loc[truth.index[5], "died"])
        assert term.support == {"rows": 1, "events": died, without: 1 - died}, kind
        assert record.part("reference:transplanted").support["rows"] == recorded - 1, kind
        # a few values (0 to 3): each is a cell, on recorded values too, and the term has the smallest of each
        # count over the values above the first (four rows at 1, one at 2, one at 3)
        prior = record.part("term:prior")
        assert prior.labels == {"variable": "prior", "per": "unit"}, kind
        assert prior.support["rows"] == 1, kind
        assert record.part("reference:prior").labels == {
            "variable": "prior",
            "level": "0",
            "of": "values",
        }, kind
        assert record.part("reference:prior").support["rows"] == recorded - 6, kind
    # a covariate whose two values every dataset keeps is counted in the datasets, as before
    kept = plan.run(
        specify("logistic", pooled=True, outcome="died", compare="arm", adjust=["age"]),
        mi=_imputed(),
    )
    assert "recorded_cells" in kept.result.extra


def test_a_pooled_cell_counts_the_rows_whose_outcome_is_missing() -> None:
    """The one participant coded 1 had no recorded outcome. Counted on the rows that record the outcome, the cell did
    not exist: the covariate read as one value, and its term was written with the model's counts."""
    truth = _cohort()
    truth["transplanted"] = 0.0
    truth.loc[truth.index[5], "transplanted"] = 1.0
    rng = np.random.default_rng(9)
    base = truth.copy()
    gap = rng.random(len(base)) < 0.12
    gap[:20] = False
    base.loc[gap, "transplanted"] = np.nan
    base["died"] = base["died"].astype(float)
    base.loc[base.index[5], "died"] = np.nan  # the participant coded 1: no outcome recorded
    completed = []
    for _ in range(5):
        one = base.copy()
        one.loc[gap, "transplanted"] = rng.normal(0.0, 0.05, gap.sum())
        one.loc[one.index[5], "died"] = float(rng.random() < 0.5)
        completed.append(one)
    mi = MIData.from_completed(base, completed, methods={"transplanted": "norm", "died": "logreg"})
    for kind, roles, without in (
        ("logistic", {"outcome": "died"}, "non_events"),
        ("cox", {"time": "days", "event": "died"}, "censored"),
    ):
        done = plan.run(
            specify(kind, pooled=True, compare="arm", adjust=["transplanted", "age"], **roles),
            mi=mi,
        )
        record = plan.record(f"{kind}_mi", done.result)
        assert record is not None
        term = record.part("term:transplanted")
        assert term.labels == {"variable": "transplanted", "level": "1"}, kind
        # one row, no recorded event and no row recorded without one: no side of the cell reaches any k
        assert term.support == {"rows": 1, "events": 0, without: 0}, kind
        assert record.part("reference:transplanted").support["rows"] == int((~gap).sum()) - 1, kind


def test_a_pooled_result_counts_the_rows_its_datasets_left_out() -> None:
    """A pooled Kaplan-Meier counted no row left out: with a group of one row, its curves held one row less than
    the flow said it had analysed."""
    truth = _cohort()
    truth["site"] = "b"
    truth.loc[truth.index[:150], "site"] = "a"
    truth.loc[truth.index[0], "site"] = "c"  # a group of one row: no curve
    truth.loc[truth.index[1:3], "days"] = -4.0  # two follow-up times below zero
    rng = np.random.default_rng(4)
    base = truth.copy()
    gap = rng.random(len(base)) < 0.1
    base.loc[gap, "age"] = np.nan
    completed = []
    for _ in range(5):
        one = base.copy()
        one.loc[gap, "age"] = truth.loc[gap, "age"] + rng.normal(0, 4, gap.sum())
        completed.append(one)
    mi = MIData.from_completed(base, completed, methods={"age": "pmm"})
    km = plan.run(specify("survival", pooled=True, time="days", event="died", group="site"), mi=mi)
    assert (km.result.extra["rows_skipped"], km.result.extra["rows_left_out"]) == (1, 2)
    record = plan.record("survival_mi", km.result)
    assert record is not None
    data = record.part("data")
    assert (data.number("rows_skipped").value, data.number("rows_left_out").value) == (1, 2)
    # the rows the analysis read: every row but the group of one and the two left out
    assert data.number("rows_min").value == len(truth) - 3 and not data.number("rows_min").fine
    assert [p.labels["group"] for p in record.parts if p.id.startswith("curve#")] == ["a", "b"]
    cox = plan.run(
        specify("cox", pooled=True, time="days", event="died", compare="arm", adjust=["age"]), mi=mi
    )
    pooled = plan.record("cox_mi", cox.result)
    assert pooled is not None
    left = pooled.part("model").number("rows_left_out")
    assert left.value == 2 and left.fine
    assert pooled.part("model").support["rows_min"] == len(truth) - 2
