"""
A calculation from numbers typed as a source of an analysis spec (biostats.plan): the share against a reference, one
rate or two and their ratio, observed against expected, the 2x2 table of a test against a reference standard. Its
spec holds the numbers and names no data; ``run`` computes it from them alone, through the function the page's
calculator shows; its record is marked typed.

Numbers: 30 of 100 against 20% (Wilson 21.9% to 39.6%, exact binomial p = 0.017); 15 events over 1,200 person-years
against 4 over 1,000 (ratio 3.125, exact conditional); 25 observed against 18.4 expected (1.36, exact Poisson); a
table of 90 and 30 positive tests, 10 and 270 negative (sensitivity and specificity 90%).
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from biostats import plan
from biostats.paired import one_sample_proportion
from biostats.plan import AnalysisSpec, ComputedOn, ResultRecord, read_spec, specify
from biostats.rates import incidence_rate, rate_ratio, rates, standardised_ratio

SHARE = {"count": 30, "total": 100}
TWO_RATES = {
    "events": 15,
    "person_time": 1200,
    "reference_events": 4,
    "reference_person_time": 1000,
}
EXPECTED = {"observed": 25, "expected": 18.4}
TABLE = {"tp": 90, "fp": 30, "fn": 10, "tn": 270}
CALCULATIONS = (
    ("proportion", SHARE, {"reference": 0.2}),
    ("rates", TWO_RATES, {}),
    ("rates", {"events": 15, "person_time": 1200}, {"per": 100000, "unit": "person-days"}),
    ("standardised_ratio", EXPECTED, {"events_are": "cases"}),
    ("diagnostic", TABLE, {}),
    ("diagnostic", TABLE, {"sampling": "two_gate", "prevalence": 0.02}),
    ("diagnostic", {"tp": 12, "fp": 0, "fn": 3, "tn": 40}, {"sampling": "series"}),
)


def _numbers(part: plan.Part) -> dict:
    return {number.name: number.value for number in part.numbers}


# ----------------------------------------------------------------------------- the catalogue


def test_the_kinds_computed_from_numbers_typed_are_declared_whole() -> None:
    typed = [kind for kind in plan.KINDS.values() if kind.typed_run is not None]
    assert [kind.id for kind in typed] == [
        "diagnostic",
        "proportion",
        "rates",
        "standardised_ratio",
    ]
    assert [k for k in plan.KINDS if plan.runnable(k, typed=True)] == [kind.id for kind in typed]
    for kind in typed:
        assert kind.typed_title and kind.typed_title.endswith("(numbers typed)")
        assert kind.typed_inputs and set(kind.typed_inputs.values()) <= {plan.COUNT, plan.QUANTITY}
        assert set(kind.typed_optional) <= set(kind.typed_inputs)
        assert kind.record is not None and "alpha" in kind.typed_options
        # a closed option of a calculation is one of its options, and its default is among its values
        offered = {**kind.options, **kind.typed_options}
        for option, known in kind.choices.items():
            assert option in offered and offered[option] in known, (kind.id, option)
        # one level for every interval: the writers say "95%" from the record, another level is a later version's
        assert kind.choices["alpha"] == (0.05,) and kind.typed_version == 1
    # a kind without numbers to type has none of the rest
    for kind in plan.KINDS.values():
        if not kind.typed_inputs:
            assert kind.typed_run is None and kind.typed_title is None and not kind.typed_options
    # what runs on the rows is as before
    assert [k for k in plan.KINDS if plan.runnable(k)] == ["survival", "cox", "logistic", "count"]
    assert not plan.runnable("cox", typed=True) and not plan.runnable("unknown", typed=True)


def test_no_kind_s_id_can_be_read_as_a_typed_item_s() -> None:
    """A calculation kept in the dossier is named ``typed:<kind>:<...>``: no kind's id starts with "typed" or holds
    a colon, so that such a name is never read two ways."""
    for kind in plan.KINDS:
        assert not kind.startswith("typed") and ":" not in kind


# ----------------------------------------------------------------------------- the spec


@pytest.mark.parametrize(("kind", "numbers", "options"), CALCULATIONS)
def test_a_typed_spec_reads_back_as_written(kind: str, numbers: dict, options: dict) -> None:
    spec = specify(kind, typed=numbers, options=options)
    assert spec.source == plan.TYPED and spec.data is None and not spec.variables
    written = json.loads(json.dumps(spec.to_dict()))
    assert set(written) == plan.SPEC_KEYS and written["typed"] == dict(spec.typed or {})
    again = AnalysisSpec.from_dict(written)
    assert again == spec and again.request_id == spec.request_id
    assert again.fingerprint == spec.fingerprint
    assert read_spec(written) == (spec, "")
    # the options are the calculation's, with their defaults
    assert set(spec.options) == set(plan.KINDS[kind].typed_options)
    done = plan.run(spec)
    assert done.spec.typed == spec.typed and done.spec.request_id == spec.request_id
    assert set(done.spec.backend) == {"scipy", "statsmodels"} and done.derived == {}


def test_a_number_is_converted_to_its_declared_type_before_anything_reads_it() -> None:
    """1200 and 1200.0 are one calculation: without the conversion they were two request ids."""
    whole = specify("rates", typed={"events": 15, "person_time": 1200})
    written = specify("rates", typed={"events": 15.0, "person_time": 1200.0})
    assert whole == written and whole.request_id == written.request_id
    assert whole.typed == {"events": 15, "person_time": 1200.0}
    assert isinstance(whole.typed["events"], int) and isinstance(whole.typed["person_time"], float)


@pytest.mark.parametrize(
    ("numbers", "named"),
    [
        ({"events": 15}, "lacks its number person_time"),
        (
            {"events": 15, "person_time": 1200, "deaths": 3},
            "the rates calculation holds a number this version does not know",
        ),
        (
            {"events": 10**400, "person_time": 1200},
            "the number events of the rates calculation is not a finite number",
        ),
        (
            {"events": 15, "person_time": 1200, "reference_events": 4},
            "are given together or not at all",
        ),
        (
            {"events": 1.5, "person_time": 1200},
            "the number events of the rates calculation is not a count",
        ),
        (
            {"events": -1, "person_time": 1200},
            "the number events of the rates calculation is not a count",
        ),
        (
            {"events": True, "person_time": 1200},
            "the number events of the rates calculation is not a finite number",
        ),
        (
            {"events": "15", "person_time": 1200},
            "the number events of the rates calculation is not a finite number",
        ),
        (
            {"events": 15, "person_time": math.nan},
            "the number person_time of the rates calculation is not a finite",
        ),
        (
            {"events": 15, "person_time": math.inf},
            "the number person_time of the rates calculation is not a finite",
        ),
        (
            {"events": 15, "person_time": None},
            "the number person_time of the rates calculation is not a finite",
        ),
    ],
)
def test_a_number_that_is_not_what_was_declared_is_refused_by_its_name(
    numbers: dict, named: str
) -> None:
    """The reason names the number, never its value: a received file can hold anything."""
    with pytest.raises(ValueError) as refused:
        specify("rates", typed=numbers)
    assert named in str(refused.value)
    # neither a value nor a name the file brought: a received file can hold anything in both
    assert "deaths" not in str(refused.value)
    for value in numbers.values():
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if not isinstance(value, bool) and value not in (15, 1200, 4, None):
            assert str(value) not in str(refused.value)


def test_a_typed_spec_names_no_variable_no_pooling_no_draw_and_no_data() -> None:
    spec = specify("proportion", typed=SHARE, options={"reference": 0.2})
    written = spec.to_dict()
    on = ComputedOn("observed", "f" * 64, 10)
    for change in (
        {"variables": {"column": "died"}},
        {"pooled": True},
        {"random": {"seed": 1}},
        {"data": on.to_dict()},
    ):
        found, reason = read_spec({**written, **change})
        assert found is None and "names no variable, is never pooled, draws nothing" in reason
    with pytest.raises(ValueError, match="computed on no data"):
        spec.on(on)
    # an option of the dataset's analysis is not a calculation's, nor a value outside a closed set
    with pytest.raises(ValueError, match="has no option ties"):
        specify("proportion", typed=SHARE, options={"ties": "efron"})
    with pytest.raises(ValueError, match="does not know this value of its option per"):
        specify("rates", typed=TWO_RATES, options={"per": 500})
    with pytest.raises(ValueError, match="does not know this value of its option events_are"):
        specify("standardised_ratio", typed=EXPECTED, options={"events_are": "admissions"})
    # a kind that has no number to type
    with pytest.raises(
        ValueError, match="the analysis kind cox is not computed from numbers typed"
    ):
        specify("cox", typed={"with": 3, "of": 928})


def test_a_spec_of_the_dataset_is_written_as_before() -> None:
    """ "typed" is written only for a calculation: every spec and record of the dataset keeps its JSON, its id and
    its fingerprint."""
    spec = specify("cox", time="days", event="died", compare="arm", adjust=("age",))
    assert "typed" not in spec.to_dict() and "typed" not in spec.request()
    assert spec.source == plan.OBSERVED and specify("cox", pooled=True).source == plan.IMPUTED
    assert spec.request_id == "4410b23e8bb7bd9b8f1e9732409280b7b99f205161fa704b9fcdf290dd3b9d28"
    # the share's dataset spec, declared later, is not moved by its numbers typed
    assert (
        specify("proportion").request_id
        == "6d63a6a20240efe84020ca819c62b2d51dc14a36cb55042aa7abbecf08bbc469"
    )


# ----------------------------------------------------------------------------- the computation


def test_a_calculation_is_computed_by_the_function_its_calculator_shows() -> None:
    share = plan.run(specify("proportion", typed=SHARE, options={"reference": 0.2})).result
    assert share == one_sample_proportion(30, 100, 0.2)
    assert share.estimate == 0.3 and share.reference_value == 0.2
    assert share.p_value == pytest.approx(0.0169, abs=5e-5)
    two = plan.run(specify("rates", typed=TWO_RATES)).result
    direct = rate_ratio(15, 1200, 4, 1000)
    assert two.ratio is not None and two.ratio.ratio == direct.ratio == 3.125
    assert (two.ratio.ci, two.ratio.p_value) == (direct.ci, direct.p_value)
    assert [rate.rate for rate in two.rates] == [15 / 1200, 4 / 1000]
    one = plan.run(specify("rates", typed={"events": 15, "person_time": 1200})).result
    assert one.ratio is None and len(one.rates) == 1
    assert one.rates[0].ci == incidence_rate(15, 1200).ci
    expected = plan.run(specify("standardised_ratio", typed=EXPECTED)).result
    assert (expected.ratio, expected.ci, expected.p_value) == (
        standardised_ratio(25, 18.4).ratio,
        standardised_ratio(25, 18.4).ci,
        standardised_ratio(25, 18.4).p_value,
    )
    assert expected.ratio == pytest.approx(1.359, abs=5e-4)


def test_two_groups_without_an_event_have_two_rates_and_no_ratio() -> None:
    """The ratio of two rates of 0 has no value: decided where the rates are computed, not on the page."""
    none = rates(0, 1200, 0, 1000)
    assert none.ratio is None and [rate.rate for rate in none.rates] == [0.0, 0.0]
    assert none.warnings == ["No event in either group: two rates of 0 have no ratio."]
    numbers = {
        "events": 0,
        "person_time": 1200,
        "reference_events": 0,
        "reference_person_time": 1000,
    }
    done = plan.run(specify("rates", typed=numbers))
    record = plan.record_of(done.spec, done.result)
    assert record is not None and [part.id for part in record.parts] == ["rate#0", "rate#1"]
    assert record.warnings == tuple(none.warnings)
    # no event in the reference group: a ratio without an upper bound, said by a number
    open_ended = plan.run(specify("rates", typed={**numbers, "events": 6}))
    ratio = plan.record_of(open_ended.spec, open_ended.result).part("ratio")  # type: ignore[union-attr]
    assert _numbers(ratio)["estimate"] == math.inf and _numbers(ratio)["bounded"] is False
    with pytest.raises(ValueError, match="needs its events and its person-time"):
        rates(15, 1200, 4, None)


def test_a_calculation_reads_no_data() -> None:
    spec = specify("proportion", typed=SHARE, options={"reference": 0.2})
    with pytest.raises(ValueError, match="reads no data"):
        plan.run(spec, pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="reads no data"):
        plan.run(spec, mi=object())
    # what the fields of the page refuse is refused here too, by words that hold no number
    with pytest.raises(ValueError, match="compared with a reference share"):
        plan.run(specify("proportion", typed=SHARE))
    for outside in (20, 0, 1):  # at 0 or at 1 nothing can be tested: the page refuses them too
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            plan.run(specify("proportion", typed=SHARE, options={"reference": outside}))
    with pytest.raises(ValueError, match="at most the number of participants"):
        plan.run(
            specify("proportion", typed={"count": 30, "total": 20}, options={"reference": 0.2})
        )
    with pytest.raises(ValueError, match="person-time must be positive"):
        plan.run(specify("rates", typed={"events": 3, "person_time": 0}))


# ----------------------------------------------------------------------------- the record


def test_the_record_of_a_share_holds_its_counts_its_interval_and_its_test() -> None:
    done = plan.run(specify("proportion", typed=SHARE, options={"reference": 0.2}))
    record = plan.record_of(done.spec, done.result)
    assert record is not None and record.typed and record.kind == "proportion"
    assert [part.id for part in record.parts] == ["share", "test:exact_binomial"]
    share, test = record.parts
    assert list(_numbers(share)) == [
        "count",
        "total",
        "estimate",
        "lower",
        "upper",
        "method",
        "level",
    ]
    assert _numbers(share)["level"] == 0.95
    assert (_numbers(share)["count"], _numbers(share)["total"]) == (30, 100)
    assert _numbers(share)["method"] == "Wilson"
    assert (_numbers(share)["lower"], _numbers(share)["upper"]) == pytest.approx(
        (0.219, 0.396), abs=5e-4
    )
    assert list(_numbers(test)) == ["reference", "p", "interval_holds_reference"]
    assert (
        _numbers(test)["reference"] == 0.2 and _numbers(test)["interval_holds_reference"] is False
    )
    # a count typed is counted as a participant's: with the characteristic, and without it
    assert dict(share.support) == {"with": 30, "without": 70, "rows": 100}
    assert dict(test.support) == dict(share.support)
    # the same record shape for a share counted in the dataset, which is not marked typed
    counted = plan.record("proportion", one_sample_proportion(30, 100, 0.2))
    assert counted is not None and not counted.typed
    assert [part.to_dict() for part in counted.parts] == [part.to_dict() for part in record.parts]


def test_the_record_of_rates_names_its_groups_by_place() -> None:
    done = plan.run(specify("rates", typed=TWO_RATES))
    record = plan.record_of(done.spec, done.result)
    assert record is not None and record.typed
    assert [part.id for part in record.parts] == ["rate#0", "rate#1", "ratio"]
    first, second, ratio = record.parts
    assert (dict(first.labels), dict(second.labels)) == (
        {"group": "compared"},
        {"group": "reference"},
    )
    assert list(_numbers(first)) == [
        "events",
        "person_time",
        "unit",
        "per",
        "estimate",
        "lower",
        "upper",
        "method",
        "level",
    ]
    # how a rate is shown, and of what, is in its record: its block is written without its spec
    assert (_numbers(first)["unit"], _numbers(first)["per"]) == ("person-years", 1000)
    assert (_numbers(first)["events"], _numbers(first)["person_time"]) == (15, 1200.0)
    assert (
        _numbers(first)["estimate"] == 15 / 1200
    )  # per unit of person-time: "per" is a way to show it
    assert dict(first.support) == {"events": 15} and dict(second.support) == {"events": 4}
    assert list(_numbers(ratio)) == [
        "estimate",
        "lower",
        "upper",
        "method",
        "level",
        "p",
        "test",
        "difference",
        "bounded",
    ]
    assert _numbers(ratio)["estimate"] == 3.125 and _numbers(ratio)["bounded"] is True
    assert _numbers(ratio)["test"] == "exact conditional"
    assert _numbers(ratio)["difference"] == pytest.approx(15 / 1200 - 4 / 1000)
    assert dict(ratio.support) == {"events": 15, "reference_events": 4}
    # one rate: one part, and no group to name
    alone = plan.run(specify("rates", typed={"events": 15, "person_time": 1200}))
    (only,) = plan.record_of(alone.spec, alone.result).parts  # type: ignore[union-attr]
    assert only.id == "rate#0" and not only.labels


def test_the_record_of_observed_against_expected() -> None:
    done = plan.run(specify("standardised_ratio", typed=EXPECTED))
    record = plan.record_of(done.spec, done.result)
    assert record is not None and record.typed
    (ratio,) = record.parts
    assert ratio.id == "ratio" and dict(ratio.support) == {"observed": 25}
    assert list(_numbers(ratio)) == [
        "observed",
        "expected",
        "events_are",
        "estimate",
        "lower",
        "upper",
        "method",
        "level",
        "p",
    ]
    assert (_numbers(ratio)["observed"], _numbers(ratio)["expected"]) == (25, 18.4)
    assert _numbers(ratio)["events_are"] == "deaths"
    assert _numbers(ratio)["method"] == "exact Poisson"


@pytest.mark.parametrize(("kind", "numbers", "options"), CALCULATIONS)
def test_a_typed_record_reads_back_and_is_the_same_computed_again(
    kind: str, numbers: dict, options: dict
) -> None:
    spec = specify(kind, typed=numbers, options=options)
    first = plan.run(spec)
    record = plan.record_of(first.spec, first.result)
    assert record is not None
    written = json.loads(json.dumps(record.to_dict()))
    assert set(written) == plan.RECORD_KEYS and written["typed"] is True
    assert ResultRecord.from_dict(written).to_dict() == record.to_dict()
    # from the spec as a file holds it: the same numbers, to the last digit
    again = plan.run(AnalysisSpec.from_dict(json.loads(json.dumps(first.spec.to_dict()))))
    assert plan.record_of(again.spec, again.result).to_json() == record.to_json()  # type: ignore[union-attr]
    # counts are compared exactly; no number of a calculation is at one participant's resolution
    for part in record.parts:
        for number in part.numbers:
            assert not number.fine
            if isinstance(number.value, int) and not isinstance(number.value, bool):
                assert number.cls == plan.EXACT


def test_a_typed_record_is_never_pooled() -> None:
    with pytest.raises(ValueError, match="never pooled"):
        ResultRecord("proportion", pooled=True, typed=True)
    plain = ResultRecord("proportion")
    assert "typed" not in plain.to_dict() and not ResultRecord.from_dict(plain.to_dict()).typed


# ----------------------------------------------------------------------------- what the reads of the build found


def test_an_option_is_kept_as_the_member_of_its_closed_set_and_a_number_as_one_type() -> None:
    """Options were kept as given: a multiplier of 1000 and one of 1000.0 were two requests, written "1,000" and
    "1,000.0"; a reference share of 1 and of 1.0 likewise."""
    whole = specify("rates", typed=TWO_RATES, options={"per": 1000})
    written = specify("rates", typed=TWO_RATES, options={"per": 1000.0})
    assert whole.request_id == written.request_id and written.options["per"] == 1000
    assert isinstance(written.options["per"], int)
    one = specify("proportion", typed=SHARE, options={"reference": 1})
    assert (
        one.request_id == specify("proportion", typed=SHARE, options={"reference": 1.0}).request_id
    )
    assert isinstance(one.options["reference"], float)
    # a yes or no is no number, and a level other than the one every writer names is a later version's
    for options, named in (
        (
            {"reference": True},
            "the option reference of the proportion calculation is not a finite number",
        ),
        (
            {"reference": math.nan},
            "the option reference of the proportion calculation is not a finite number",
        ),
        ({"reference": "0.2"}, "compared with a reference share"),
        ({"alpha": 0.10}, "does not know this value of its option alpha"),
        ({"alpha": True}, "does not know this value of its option alpha"),
    ):
        with pytest.raises(ValueError, match=named):
            plan.run(specify("proportion", typed=SHARE, options={"reference": 0.2, **options}))
    with pytest.raises(ValueError, match="does not know this value of its option per"):
        specify("rates", typed=TWO_RATES, options={"per": True})


def test_the_reader_gives_a_reason_for_anything_a_file_can_hold() -> None:
    """A count of 400 digits raised past the reader (the whole Word download then failed, every section with it),
    and a calculation of a later version was told to hold a defect, with the name of what it had added."""
    spec = specify("rates", typed=TWO_RATES)
    written = json.loads(json.dumps(spec.to_dict()))
    huge = {**written, "typed": {**written["typed"], "events": 10**400}}
    assert read_spec(huge) == (
        None,
        "the number events of the rates calculation is not a finite number",
    )
    far = {**written, "typed": {**written["typed"], "person_time": 10**400}}
    assert read_spec(far) == (
        None,
        "the number person_time of the rates calculation is not a finite number",
    )
    # a later version first: what it added is not reported as a defect of the file, nor named
    later = {**written, "kind_version": 2, "typed": {**written["typed"], "strata": 3}}
    assert read_spec(later) == (
        None,
        "the rates analysis was specified by a later version of CohortStudio",
    )
    unknown = read_spec({**written, "typed": {**written["typed"], "Jane Doe 1956": 3}})
    assert unknown[0] is None and "Jane Doe" not in unknown[1]
    assert "holds a number this version does not know" in unknown[1]
    with pytest.raises(plan.LaterVersion):
        plan.run(AnalysisSpec("rates", typed=TWO_RATES, kind_version=2))


def test_a_calculation_has_its_own_version_apart_from_the_dataset_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One version served both sources: declaring the share's roles for the dataset, later, would have given every
    typed share another id, and made this version read the next one's typed shares as "not run" for a calculation
    it computes the same way."""
    from dataclasses import replace

    spec = specify("proportion", typed=SHARE, options={"reference": 0.2})
    written = json.loads(json.dumps(spec.to_dict()))
    kind = plan.KINDS["proportion"]
    declared = replace(kind, roles=("variable",), options={"alpha": 0.05, "level": None}, version=2)
    monkeypatch.setitem(plan.KINDS, "proportion", declared)
    after = specify("proportion", typed=SHARE, options={"reference": 0.2})
    assert after.kind_version == 1 and after.request_id == spec.request_id
    assert read_spec(written) == (after, "")
    assert plan.run(AnalysisSpec.from_dict(written)).result.estimate == 0.3
    # the dataset's own spec follows the dataset's version
    assert specify("proportion", variable="smoker").kind_version == 2


def test_a_block_s_words_are_in_the_record_and_a_result_names_its_test() -> None:
    """A rate's block written from its record alone read "12.50 per 1,000 person-years" for 15 events over 1,200
    person-days shown per 100,000, and a standardised incidence ratio was called a mortality ratio: the unit, the
    multiplier and what the events are were options of the spec only."""
    days = plan.run(
        specify(
            "rates",
            typed={"events": 15, "person_time": 1200},
            options={"unit": "person-days", "per": 100000},
        )
    )
    assert days.result.unit == "person-days" and days.result.rates[0].per == 100000
    (rate,) = plan.record_of(days.spec, days.result).parts  # type: ignore[union-attr]
    assert (_numbers(rate)["unit"], _numbers(rate)["per"]) == ("person-days", 100000)
    cases = plan.run(specify("standardised_ratio", typed=EXPECTED, options={"events_are": "cases"}))
    assert cases.result.events_are == "cases" and not cases.result.warnings
    (ratio,) = plan.record_of(cases.spec, cases.result).parts  # type: ignore[union-attr]
    assert _numbers(ratio)["events_are"] == "cases"
    # the test that agrees with the exact interval: twice the smaller tail, where R's poisson.test gives 0.128
    assert cases.result.p_value == pytest.approx(0.165, abs=5e-4)
    assert [ref[:7] for ref in cases.result.references] == ["Ulm K (", "Garwood"]
    # fewer than 1 event expected: said with the result, wherever it is written
    few = plan.run(specify("standardised_ratio", typed={"observed": 1, "expected": 0.05}))
    assert few.result.warnings == [
        "Fewer than 1 event expected: the ratio rests on almost nothing."
    ]
    assert few.result.p_value == pytest.approx(0.098, abs=5e-4)
    assert plan.record_of(few.spec, few.result).warnings == tuple(few.result.warnings)  # type: ignore[union-attr]


# ----------------------------------------------------------------------------- the 2x2 table typed


def test_a_table_typed_is_computed_by_the_function_of_a_table_of_the_dataset() -> None:
    from biostats.diagnostic import diagnostic_accuracy

    done = plan.run(
        specify("diagnostic", typed=TABLE, options={"sampling": "series", "prevalence": 0.02})
    )
    assert done.result == diagnostic_accuracy(90, 30, 10, 270, prevalence=0.02, sampling="series")
    assert (done.result.sensitivity, done.result.specificity) == (0.9, 0.9)
    assert done.result.ppv == 0.75 and done.result.ppv_at_prevalence == pytest.approx(
        0.1552, abs=5e-5
    )
    # not stated: no predictive value at the table's share, and none at a prevalence that was not given
    bare = plan.run(specify("diagnostic", typed=TABLE)).result
    assert bare.sampling == "not_stated" and math.isnan(bare.ppv) and bare.ppv_at_prevalence is None
    # a table asks nothing of a dataset: the options of a column or of a score are not its own
    for option in ("index_form", "positive", "present", "direction", "mode", "cutoff"):
        with pytest.raises(ValueError, match=f"has no option {option}"):
            specify("diagnostic", typed=TABLE, options={option: None})
    for option, value in (("sampling", ""), ("sampling", "cohort"), ("alpha", 0.1)):
        with pytest.raises(ValueError, match=f"does not know this value of its option {option}"):
            specify("diagnostic", typed=TABLE, options={option: value})
    # the dataset's own spec keeps its declaration and its id (tests/biostats_layer/test_plan.py pins it)
    assert set(specify("diagnostic").options) == set(plan.KINDS["diagnostic"].options)


def test_a_table_without_a_side_or_without_a_test_result_is_refused_in_words() -> None:
    """Without participants on a side of the reference standard a measure has no value; a test that calls everyone
    positive, or no one, leaves nothing to assess (the page refuses both). The reason holds no number typed."""
    sides = "participants with the condition and without it"
    results = "positive tests and negative tests"
    for numbers, named in (
        ({"tp": 0, "fp": 31, "fn": 0, "tn": 277}, sides),
        ({"tp": 93, "fp": 0, "fn": 17, "tn": 0}, sides),
        ({"tp": 0, "fp": 0, "fn": 0, "tn": 0}, sides),
        ({"tp": 0, "fp": 0, "fn": 17, "tn": 277}, results),
        ({"tp": 93, "fp": 31, "fn": 0, "tn": 0}, results),
    ):
        with pytest.raises(ValueError, match=named) as refused:
            plan.run(specify("diagnostic", typed=numbers))
        assert not any(character.isdigit() for character in str(refused.value))
    for outside in (0, 1, 2):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            plan.run(specify("diagnostic", typed=TABLE, options={"prevalence": outside}))
    with pytest.raises(
        ValueError, match="the option prevalence of the diagnostic calculation is not a finite"
    ):
        specify("diagnostic", typed=TABLE, options={"prevalence": math.nan})
    with pytest.raises(ValueError, match="the diagnostic calculation lacks its number tn"):
        specify("diagnostic", typed={"tp": 90, "fp": 30, "fn": 10})


def test_the_record_of_a_table_typed_names_no_column_and_no_form() -> None:
    """The record read the page's dict only: a table alone raised, and a mapping without columns wrote the labels
    "None" and the form of a score. A table typed names no column, no value and no form; its measures are those of
    a yes/no test of the dataset with the same four counts, to the last digit."""
    from biostats.diagnostic import diagnostic_accuracy

    done = plan.run(
        specify("diagnostic", typed=TABLE, options={"sampling": "series", "prevalence": 0.02})
    )
    record = plan.record_of(done.spec, done.result)
    assert record is not None and record.typed and record.kind == "diagnostic"
    assert [part.id for part in record.parts] == [
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
    table = record.part("table")
    assert _numbers(table) == {**TABLE, "sampling": "series"}
    assert not any(part.labels for part in record.parts)
    # each count typed is counted as a participant's: the four cells and their sums, as for a table of the dataset
    assert dict(table.support) == {
        **TABLE,
        "rows": 400,
        "with_condition": 100,
        "without_condition": 300,
        "called_positive": 120,
        "called_negative": 280,
    }
    counted = plan.record(
        "diagnostic",
        {
            "acc": diagnostic_accuracy(90, 30, 10, 270, prevalence=0.02, sampling="series"),
            "auc": None,
            "index_form": "yes_no",
            "outcome": "ill",
            "score": "rapid",
            "pos_label": "Yes",
            "positive": "Positive",
        },
    )
    assert counted is not None and not counted.typed
    for typed, of_the_dataset in zip(record.parts, counted.parts, strict=True):
        assert typed.id == of_the_dataset.id and dict(typed.support) == dict(of_the_dataset.support)
        shared = {k: v for k, v in _numbers(of_the_dataset).items() if k != "index_form"}
        assert _numbers(typed) == shared
        assert dict(of_the_dataset.labels) == {
            "reference": "ill",
            "index": "rapid",
            "present": "Yes",
            "positive": "Positive",
        }
    assert _numbers(counted.part("table"))["index_form"] == "yes_no"
    # a result kept before the forms is a score, and a label is written only when its value is there
    kept = {"mode": "Manual", "direction": "higher", "cutoff": 52.0}
    before = plan.record("diagnostic", {"acc": done.result, "auc": None, "outcome": "ill", **kept})
    assert before is not None and _numbers(before.part("table"))["index_form"] == "number"
    assert dict(before.part("table").labels) == {"reference": "ill"}
    # a zero cell: the measures' own warnings go with the record of a table typed
    zero = plan.run(specify("diagnostic", typed={"tp": 12, "fp": 0, "fn": 3, "tn": 40}))
    held = plan.record_of(zero.spec, zero.result)
    assert held is not None and held.warnings == tuple(zero.result.warnings)
    assert len(held.warnings) == 2


def test_the_sampling_of_an_accuracy_study_is_read_for_any_design() -> None:
    from biostats.study_designs import ACCURACY_SAMPLING, BEFORE, accuracy_sampling

    assert (
        accuracy_sampling("case-control") == "two_gate" and accuracy_sampling("cohort") == "series"
    )
    # a result kept before the design was asked, or a design this version does not know: not stated, never an error
    assert BEFORE not in ACCURACY_SAMPLING
    assert accuracy_sampling(BEFORE) == accuracy_sampling("nested") == "not_stated"
