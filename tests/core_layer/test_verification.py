"""
Checking a study package (M8): the outcomes and their order, the catalogue of checks by tier, the tolerance of each
comparison class, a result computed again against its record, the report and its exit code, and the profiles of a
table's columns, which two environments can compare.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from biostats import plan
from biostats.mi import MIData
from biostats.plan import (
    DETERMINISTIC,
    EXACT,
    ITERATIVE,
    ITERATIVE_LOOSE,
    Number,
    Part,
    ResultRecord,
    specify,
)
from biostats.regression import logistic_regression
from biostats.survival import cox_regression
from core import verification as v
from core.profiles import PROFILE_VERSION, column_profile, frame_profile
from core.registry import UnknownIdError
from core.verification import (
    FAILED,
    IDENTICAL,
    NOT_COMPARABLE,
    NOT_RUN,
    WITHIN_MONTE_CARLO,
    WITHIN_TOLERANCE,
    CheckItem,
    CheckResult,
    VerificationReport,
    compare_numbers,
    compare_profiles,
    compare_records,
    monte_carlo_bound,
    place,
    worst,
)

# the first 16 characters of each column's digest in test_the_digests_are_those_of_version_one
PINNED_DIGESTS = {
    "n": "53bda4736010a834",
    "x": "112a4930b4f3a59a",
    "t": "00eb8d93874f65fa",
    "d": "a10334d366f67532",
    "b": "c8fbf601c3a0a09b",
}

# the ids stored in reports: a check may be added, never renamed or removed
CHECK_IDS = (
    "package.files",
    "environment",
    "protocol.checksum",
    "protocol.fingerprint",
    "results.provenance",
    "flow.counts",
    "derived.definitions",
    "data.sha256",
    "trace.chain",
    "steps.shape",
    "population.rows",
    "columns.profiles",
    "results.values",
    "frames.fingerprint",
)
SHAREABLE_IDS = (
    "package.files",
    "environment",
    "protocol.fingerprint",
    "results.provenance",
    "flow.counts",
    "derived.definitions",
    "data.sha256",
)
OUTCOME_IDS = (
    "identical",
    "within_tolerance",
    "within_monte_carlo_error",
    "not_comparable",
    "not_run",
    "failed",
)


# ----------------------------------------------------------------------------- outcomes and checks


def test_a_check_reads_the_worst_of_its_items() -> None:
    assert worst([IDENTICAL, WITHIN_TOLERANCE]) == WITHIN_TOLERANCE
    assert (
        worst([WITHIN_TOLERANCE, WITHIN_MONTE_CARLO, IDENTICAL]) == WITHIN_MONTE_CARLO
    )  # it says less
    assert worst([IDENTICAL, NOT_COMPARABLE, WITHIN_MONTE_CARLO]) == NOT_COMPARABLE
    assert worst([NOT_COMPARABLE, FAILED, IDENTICAL]) == FAILED
    assert (
        worst([IDENTICAL, NOT_RUN]) == IDENTICAL
    )  # an item not run does not count beside one that ran
    assert worst([NOT_RUN]) == NOT_RUN and worst([]) == NOT_RUN
    with pytest.raises(ValueError, match="unknown outcome"):
        worst(["passed"])
    assert v.OUTCOMES == OUTCOME_IDS


def test_the_checks_keep_their_ids_and_their_tiers() -> None:
    assert v.CHECKS.ids() == CHECK_IDS
    assert [k.id for k in v.for_tier("reproduction")] == list(CHECK_IDS)
    # the others read the rows, the trace, or a checksum over what a shareable package does not carry
    assert tuple(k.id for k in v.for_tier("shareable")) == SHAREABLE_IDS
    for partial in ("flow.counts", "derived.definitions", "data.sha256", "results.provenance"):
        assert v.CHECKS.get(
            partial
        ).shareable_part  # less is checked without the rows, and it says what
    assert "no fingerprint of rows" in v.CHECKS.get("results.provenance").shareable_part
    with pytest.raises(ValueError, match="unknown tier"):
        v.for_tier("public")
    with pytest.raises(UnknownIdError):
        CheckResult("results.numbers", IDENTICAL)


def test_a_report_counts_its_outcomes_and_gives_the_exit_code() -> None:
    fine = CheckResult.of("package.files", [CheckItem("protocol.json", IDENTICAL)])
    close = CheckResult.of(
        "results.values",
        [
            CheckItem("cox", IDENTICAL),
            CheckItem(
                "logistic",
                WITHIN_TOLERANCE,
                largest={
                    "quantity": "term 2 of 3 (age): coef",
                    "difference": 2e-12,
                    "allowed": 1e-6,
                    "class": ITERATIVE,
                },
            ),
        ],
    )
    absent = CheckResult.not_run("steps.shape", "the package holds no rows")
    report = VerificationReport(
        {"tier": "shareable"}, {"cohortstudio": "2.5.0"}, (fine, close, absent), "now"
    )
    assert report.exit_code() == v.EXIT_OK == 0
    assert report.counts() == {
        "identical": 1,
        "within_tolerance": 1,
        "within_monte_carlo_error": 0,
        "not_comparable": 0,
        "not_run": 1,
        "failed": 0,
    }
    written = report.to_dict()
    assert written["format"] == "cohortstudio.verification-report" and written["schema"] == 1
    assert written["checks"][1]["items"][1]["largest"]["quantity"] == "term 2 of 3 (age): coef"
    assert written["checks"][2] == {
        "id": "steps.shape",
        "outcome": "not_run",
        "reason": "the package holds no rows",
        "items": [],
    }
    broken = CheckResult.of(
        "trace.chain", [CheckItem("step 4", FAILED, "its hash is not the recorded one")]
    )
    unsure = CheckResult.of(
        "environment", [CheckItem("libraries", NOT_COMPARABLE, "another pandas")]
    )
    redrawn = CheckResult.of("results.values", [CheckItem("cox_mi", WITHIN_MONTE_CARLO)])
    assert (
        VerificationReport({}, {}, (fine, unsure, redrawn)).exit_code() == 0
    )  # neither is a failure
    assert VerificationReport({}, {}, (fine, unsure, broken)).exit_code() == v.EXIT_FAILED == 1
    assert v.EXIT_UNVERIFIABLE == 2


def test_a_check_says_how_many_of_its_items_were_not_run() -> None:
    """Two results identical and one not run read "identical" with no word of the third."""
    partly = CheckResult.of(
        "results.values",
        [
            CheckItem("cox", IDENTICAL),
            CheckItem("logistic", IDENTICAL),
            CheckItem("ancova", NOT_RUN, "no spec"),
        ],
    )
    assert (partly.outcome, partly.reason) == (IDENTICAL, "1 of 3 not run")
    none = CheckResult.of("results.values", [CheckItem("ancova", NOT_RUN, "no spec")])
    assert (none.outcome, none.reason) == (NOT_RUN, "")


# ----------------------------------------------------------------------------- tolerances


def test_a_count_is_equal_or_it_fails() -> None:
    assert compare_numbers(Number("rows", 928, EXACT), Number("rows", 928, EXACT))[0] == IDENTICAL
    assert compare_numbers(Number("rows", 928, EXACT), Number("rows", 927, EXACT))[0] == FAILED
    assert (
        compare_numbers(Number("ties", "efron", EXACT), Number("ties", "breslow", EXACT))[0]
        == FAILED
    )
    assert (
        compare_numbers(Number("m", 5, EXACT), Number("m", 5.0, EXACT))[0] == FAILED
    )  # another type
    nan = Number("median", math.nan, EXACT)
    assert compare_numbers(nan, nan)[0] == IDENTICAL  # two missing numbers are the same
    assert (
        v.difference(928, 927, Number("rows", 928, EXACT)) is None
    )  # nothing to measure: equal or not
    seeded = Number("end", 1.5, plan.SEEDED)
    assert compare_numbers(seeded, Number("end", 1.5000001, plan.SEEDED))[0] == FAILED


def test_closed_form_arithmetic_is_allowed_rounding_error() -> None:
    def two(a: float, b: float, **kwargs) -> tuple[str, float, float]:
        return compare_numbers(
            Number("x", a, DETERMINISTIC, **kwargs), Number("x", b, DETERMINISTIC, **kwargs)
        )

    assert two(6.64, 6.64)[0] == IDENTICAL
    outcome, found, allowed = two(
        6.64, 6.64 * (1 + 2e-12)
    )  # the noise seen when the rows are permuted
    assert outcome == WITHIN_TOLERANCE and found < allowed == pytest.approx(6.64e-8, rel=1e-6)
    assert two(6.64, 6.64 * (1 + 1e-6))[0] == FAILED
    # near zero a relative rule alone would flag noise: the quantity's own scale is the floor
    assert two(4e-12, -4e-12)[0] == FAILED
    assert two(4e-12, -4e-12, scale=0.01)[0] == WITHIN_TOLERANCE


def test_a_fitted_estimate_is_allowed_a_small_part_of_its_standard_error() -> None:
    def estimate(shift: float, cls: str = ITERATIVE) -> str:
        return compare_numbers(
            Number("log_hr", 0.5, cls, scale=0.2), Number("log_hr", 0.5 + shift, cls, scale=0.2)
        )[0]

    assert estimate(0.2 * 4e-11) == WITHIN_TOLERANCE
    assert estimate(0.2 * 0.9e-4) == WITHIN_TOLERANCE and estimate(0.2 * 1e-3) == FAILED
    assert (
        estimate(0.2 * 0.9e-3, ITERATIVE_LOOSE) == WITHIN_TOLERANCE
    )  # an optional library's own fit stops earlier
    assert estimate(0.2 * 1e-2, ITERATIVE_LOOSE) == FAILED
    # the standard error itself, a log-likelihood: relative
    assert compare_numbers(
        Number("llf", -412.3, ITERATIVE), Number("llf", -412.3 * (1 + 5e-7), ITERATIVE)
    )[0] == (WITHIN_TOLERANCE)
    assert compare_numbers(
        Number("llf", -412.3, ITERATIVE), Number("llf", -412.3 * (1 + 5e-5), ITERATIVE)
    )[0] == (FAILED)


def test_what_is_computed_from_an_estimate_and_its_se_passes_when_both_do() -> None:
    """A z and an end of an interval failed while the coefficient and its standard error passed: their rules did not
    agree. Every number of a class is allowed a part of its scale plus a relative part."""
    record = plan.record("logistic", logistic_regression(_cohort(), "died", ["arm", "age", "nyha"]))
    assert record is not None
    tolerance = v.TOLERANCES[ITERATIVE]
    moved = []
    for part in record.parts:
        numbers = {n.name: n for n in part.numbers}
        if not part.id.startswith("term:"):
            moved.append(part)
            continue
        coef = numbers["coef"].value + 0.99 * tolerance.of_scale * numbers["se"].value
        se = numbers["se"].value * (1 + 0.99 * tolerance.relative)
        z = 1.959963984540054
        new = {
            "coef": coef,
            "se": se,
            "odds_ratio": math.exp(coef),
            "ci_low": math.exp(coef - z * se),
            "ci_high": math.exp(coef + z * se),
            "z": coef / se,
            "p": numbers["p"].value,
            "extreme": False,
        }
        moved.append(
            Part(
                part.id,
                tuple(
                    Number(n.name, new[n.name], n.cls, scale=n.scale, rule=n.rule)
                    for n in part.numbers
                ),
                part.support,
                part.labels,
            )
        )
    again = ResultRecord("logistic", parts=tuple(moved), record_version=record.record_version)
    compared = compare_records(record, again)
    assert compared.outcome == WITHIN_TOLERANCE, compared.reason


def test_a_p_value_is_compared_on_the_log_scale() -> None:
    def p(a: float, b: float, cls: str = DETERMINISTIC) -> str:
        return compare_numbers(
            Number("p", a, cls, rule=plan.LOG), Number("p", b, cls, rule=plan.LOG)
        )[0]

    # both below the floor: equal, whatever their digits (not the same 64-bit value, so not "identical")
    assert p(1.48e-43, 1.48e-43 * (1 + 3.5e-14)) == WITHIN_TOLERANCE
    assert p(0.0, 3e-200) == WITHIN_TOLERANCE
    assert p(1.48e-43, 1.48e-43) == IDENTICAL
    assert p(0.0371, 0.0371 * (1 + 5e-7)) == WITHIN_TOLERANCE
    assert p(0.0371, 0.0376) == FAILED
    assert p(0.0371, 0.0371 * (1 + 5e-4), ITERATIVE) == WITHIN_TOLERANCE
    assert p(0.0371, 0.0371 * (1 + 1.5e-2), ITERATIVE_LOOSE) == WITHIN_TOLERANCE
    assert p(0.0371, 0.0) == FAILED


def test_the_degrees_of_freedom_of_a_pooled_term_are_compared_through_their_reciprocal() -> None:
    def df(a: float, b: float) -> str:
        one = Number("df", a, ITERATIVE, rule=plan.RECIPROCAL)
        return compare_numbers(one, Number("df", b, ITERATIVE, rule=plan.RECIPROCAL))[0]

    assert (
        df(math.inf, 1e35) == WITHIN_TOLERANCE
    )  # a fully observed variable: infinite in one run, huge in another
    assert df(math.inf, math.inf) == IDENTICAL
    assert df(41.2, 41.2 * (1 + 1e-5)) == WITHIN_TOLERANCE
    assert df(41.2, 57.0) == FAILED


def test_a_series_is_compared_entry_by_entry() -> None:
    one = Number("survival", [1.0, 0.9, 0.8], DETERMINISTIC, scale=1.0, fine=True)
    assert compare_numbers(
        one, Number("survival", [1.0, 0.9, 0.8 + 1e-13], DETERMINISTIC, scale=1.0)
    )[0] == (WITHIN_TOLERANCE)
    assert (
        compare_numbers(one, Number("survival", [1.0, 0.9], DETERMINISTIC, scale=1.0))[0] == FAILED
    )
    assert (
        compare_numbers(one, Number("survival", [1.0, 0.9, 0.7], DETERMINISTIC, scale=1.0))[0]
        == FAILED
    )
    assert compare_numbers(one, Number("survival", 1.0, DETERMINISTIC, scale=1.0))[0] == FAILED
    assert (
        compare_numbers(one, Number("survival", [1.0, 0.9, 0.8], ITERATIVE, scale=1.0))[0] == FAILED
    )  # another class


# ----------------------------------------------------------------------------- a result against its record


def _cohort(n: int = 300, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.normal(62, 9, n).round(1),
            "arm": rng.choice(["control", "drug"], n),
            "nyha": rng.choice(["I", "II", "III"], n, p=[0.3, 0.5, 0.2]),
        }
    )
    df["died"] = (rng.random(n) < 0.45).astype(int)
    df["days"] = rng.exponential(
        np.exp(6.0 - 0.01 * df["age"] + 0.4 * (df["arm"] == "drug"))
    ).round()
    return df


def _cox(df: pd.DataFrame) -> ResultRecord:
    record = plan.record("cox", cox_regression(df, "days", "died", ["arm", "age", "nyha"]))
    assert record is not None
    return record


def _with(record: ResultRecord, change) -> ResultRecord:
    """The record with each part passed through ``change`` (a part, or None to leave it out)."""
    parts = [changed for part in record.parts if (changed := change(part)) is not None]
    return ResultRecord(
        record.kind,
        record.pooled,
        tuple(parts),
        record.warnings,
        record.complete_case,
        record.record_version,
    )


def _renumbered(part: Part, **values) -> Part:
    numbers = tuple(
        Number(n.name, values[n.name], n.cls, scale=n.scale, rule=n.rule, fine=n.fine)
        if n.name in values
        else n
        for n in part.numbers
    )
    return Part(part.id, numbers, part.support, part.labels)


def test_the_same_rows_in_another_order_read_identical_or_within_tolerance() -> None:
    df = _cohort()
    recorded = _cox(df)
    again = compare_records(recorded, _cox(df), name="cox:1")
    assert (again.name, again.outcome, again.reason, again.largest) == (
        "cox:1",
        IDENTICAL,
        "",
        None,
    )
    shuffled = compare_records(recorded, _cox(df.sample(frac=1, random_state=3)))
    assert shuffled.outcome in (IDENTICAL, WITHIN_TOLERANCE)
    if shuffled.outcome == WITHIN_TOLERANCE:
        assert (
            shuffled.largest is not None
            and shuffled.largest["difference"] <= shuffled.largest["allowed"]
        )


def test_one_row_less_fails_on_the_counts() -> None:
    df = _cohort()
    less = compare_records(_cox(df), _cox(df.iloc[1:]))
    assert less.outcome == FAILED
    assert less.reason.startswith(
        "model: the counts behind it differ, and "
    )  # and the cells it was in


def test_a_failure_is_placed_within_its_kind_with_its_variable_never_its_level() -> None:
    """ "term 7" counted every part of the record: the fourth of four terms. The place is counted among the terms,
    with the variable and the level's own place; the level's name, a value of the data, is never written."""
    recorded = _cox(_cohort())
    assert [p.id for p in recorded.parts if p.id.startswith("term:")] == [
        "term:age",
        "term:arm#1",
        "term:nyha#1",
        "term:nyha#2",
    ]
    assert place(recorded, "term:nyha#2", "log_hr") == "term 4 of 4 (nyha, level 3 of 3): log_hr"
    assert place(recorded, "term:age") == "term 1 of 4 (age)"
    assert place(recorded, "reference:nyha") == "reference 2 of 2 (nyha)"
    assert place(recorded, "model", "rows") == "model: rows"

    def moved(part: Part) -> Part:
        if part.id != "term:nyha#2":
            return part
        number = part.number("log_hr")
        return _renumbered(part, log_hr=number.value + 1e-3 * number.scale)

    off = compare_records(recorded, _with(recorded, moved))
    assert (
        off.outcome == FAILED and off.reason == "term 4 of 4 (nyha, level 3 of 3): log_hr differs"
    )
    assert "III" not in off.reason
    assert off.largest is not None and off.largest["class"] == ITERATIVE
    assert off.largest["difference"] > off.largest["allowed"]

    def two(part: Part) -> Part:
        return (
            _renumbered(part, se=part.number("se").value * 1.01)
            if part.id.startswith("term:nyha")
            else part
        )

    several = compare_records(recorded, _with(recorded, two))
    assert (
        several.reason == "term 3 of 4 (nyha, level 2 of 3): se differs, and 1 other difference(s)"
    )


def test_what_an_absent_library_computes_is_not_comparable() -> None:
    """The concordance and the proportional-hazards test come from lifelines, an optional library: a package made
    with it and checked without it (or the reverse) cannot compare them, and the Cox kind says which they are."""
    base = _cox(_cohort())
    test = Part(
        "ph:age",
        (
            Number("statistic", 1.7, ITERATIVE_LOOSE, scale=1.0),
            Number("p", 0.19, ITERATIVE_LOOSE, rule=plan.LOG),
            Number("flagged", False, EXACT),
        ),
        base.part("model").support,
        {"variable": "age"},
    )

    def model(concordance: float | None):
        return lambda part: (
            None
            if part.id.startswith("ph:")
            else _renumbered(part, concordance=concordance)
            if part.id == "model"
            else part
        )

    with_library = _with(base, model(0.63))
    recorded = ResultRecord(
        "cox", parts=(*with_library.parts, test), record_version=base.record_version
    )
    without = _with(base, model(None))  # computed again where lifelines is not installed
    assert compare_records(recorded, without).outcome == FAILED  # said by nobody: a failure
    said = compare_records(recorded, without, absent=["lifelines"])
    assert (said.outcome, said.reason) == (NOT_COMPARABLE, "lifelines is not installed on one side")
    assert plan.KINDS["cox"].optional == {"lifelines": ("ph:", "model/concordance")}
    assert compare_records(recorded, recorded).outcome == IDENTICAL


def test_another_kind_or_another_version_of_record_is_said() -> None:
    recorded = _cox(_cohort())
    other = ResultRecord("logistic", parts=recorded.parts)
    assert compare_records(recorded, other).reason == "another kind of analysis was computed"
    fewer = _with(recorded, lambda part: None if part.id == "term:nyha#2" else part)
    assert compare_records(recorded, fewer).reason == (
        "the parts of the result differ: 1 recorded only, 0 computed only"
    )
    # a later CohortStudio that adds a number to the record must not fail every package made before it
    later = ResultRecord("cox", parts=recorded.parts, record_version=2)
    version = compare_records(recorded, later)
    assert version.outcome == NOT_COMPARABLE
    assert (
        version.reason == "the result was recorded by another version of the cox record (1, here 2)"
    )


def test_a_fit_without_a_maximum_is_not_compared() -> None:
    """A level held by one row sends its coefficient to -20 with a standard error of 44,000: two runs differ by
    anything, and "within tolerance" there checked nothing. The flags are compared exactly; the estimates are not."""
    df = _cohort()
    df["sex"] = "male"
    df.loc[df.index[0], ["sex", "died"]] = ["F ", 1]  # one row, an event: separation
    record = plan.record("logistic", logistic_regression(df, "died", ["sex", "age"]))
    assert record is not None
    odd = record.part("term:sex#1")
    assert odd.number("extreme").value is True

    def noisy(part: Part) -> Part:
        return (
            _renumbered(part, coef=part.number("coef").value + 3.0)
            if part.id == "term:sex#1"
            else part
        )

    compared = compare_records(record, _with(record, noisy))
    assert compared.outcome == NOT_COMPARABLE
    assert compared.reason == "no maximum of the likelihood: such an estimate cannot be compared"
    # the flag itself is exact: a run that finds a maximum where the record found none is a failure
    flipped = compare_records(
        record,
        _with(record, lambda p: _renumbered(p, extreme=False) if p.id == "term:sex#1" else p),
    )
    assert flipped.outcome == FAILED and flipped.reason.endswith(": extreme differs")
    # a fit that did not converge: none of its terms is compared
    stopped = _with(
        _cox(_cohort()), lambda p: _renumbered(p, converged=False) if p.id == "model" else p
    )
    assert compare_records(stopped, stopped).outcome == NOT_COMPARABLE


def _imputed(seed: int, m: int = 5) -> MIData:
    truth = _cohort()
    rng = np.random.default_rng(seed)
    base = truth.copy()
    gap = (
        np.random.default_rng(99).random(len(base)) < 0.15
    )  # the same rows are missing in both sets of draws
    base.loc[gap, "age"] = np.nan
    completed = []
    for _ in range(m):
        one = base.copy()
        one.loc[gap, "age"] = truth.loc[gap, "age"] + rng.normal(0, 6, gap.sum())
        completed.append(one)
    return MIData.from_completed(base, completed, methods={"age": "pmm"})


def test_imputations_drawn_again_are_held_to_monte_carlo_error() -> None:
    """Another machine draws other imputed values: every number of the pooled result differs, and none is a failure
    of the study. The pooled estimates agree as two draws of one model do, and the outcome says no more than that."""
    spec = specify(
        "cox", pooled=True, time="days", event="died", compare="arm", adjust=["age", "nyha"]
    )
    first = plan.record("cox_mi", plan.run(spec, mi=_imputed(1)).result)
    second = plan.record("cox_mi", plan.run(spec, mi=_imputed(2)).result)
    assert first is not None and second is not None
    assert compare_records(first, second).outcome == FAILED  # as the same draws, they differ
    redrawn = compare_records(first, second, draws="other")
    assert redrawn.outcome == WITHIN_MONTE_CARLO
    assert (
        redrawn.largest["class"] == "monte_carlo"
        and redrawn.largest["difference"] <= redrawn.largest["allowed"]
    )
    assert redrawn.largest["quantity"].endswith(": estimate")

    # a pooled estimate that moved by more than two draws do is a failure
    def far(part: Part) -> Part:
        return (
            _renumbered(part, estimate=part.number("estimate").value + 0.5)
            if part.id == "term:age"
            else part
        )

    assert compare_records(first, _with(second, far), draws="other").outcome == FAILED
    # the bound: the between-imputation variances of both sets, at 0.5% each side shared between the terms
    # (t quantile at 1 - 0.005/4 on 8 degrees of freedom, 4.33, times the root of 2 x 5.2e-4 / 5)
    assert monte_carlo_bound(5.2e-4, 5, 5.2e-4, 5, 4) == pytest.approx(0.0625, abs=5e-4)
    assert (
        monte_carlo_bound(0.0, 5, 0.0, 5, 4) == 0.0
    )  # a fully observed term does not move with the draws
    # the same draws are compared number by number, as any result
    assert compare_records(first, first, draws="same").outcome == IDENTICAL


# ----------------------------------------------------------------------------- column profiles


def _table(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(8)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 90, n),
            "bmi": rng.normal(27, 4, n),
            "sex": rng.choice(["F", "M"], n),
            "seen": pd.to_datetime("2024-01-01")
            + pd.to_timedelta(rng.integers(0, 700, n), unit="D"),
            "smoker": rng.random(n) < 0.3,
        }
    )


def _columns(profile: dict) -> dict[str, dict]:
    return dict(zip(profile["names"], profile["columns"], strict=True))


def test_a_profile_says_what_each_kind_of_column_holds() -> None:
    profile = frame_profile(_table())
    assert (
        profile["format"] == "cohortstudio.column-profile" and profile["version"] == PROFILE_VERSION
    )
    assert profile["rows"] == 400 and profile["names"] == ["age", "bmi", "sex", "seen", "smoker"]
    columns = _columns(profile)
    assert {name: column["kind"] for name, column in columns.items()} == {
        "age": "number",
        "bmi": "number",
        "sex": "text",
        "seen": "date",
        "smoker": "boolean",
    }
    assert columns["age"]["whole"] is True and columns["bmi"]["whole"] is False
    assert set(columns["bmi"]) == {
        "kind", "rows", "missing", "whole", "digest", "plus_inf", "minus_inf",
        "sum", "sum_abs", "sum_sq", "min", "max", "mix", "mix2",
    }  # fmt: skip
    assert columns["sex"]["distinct"] == 2
    assert columns["smoker"]["true"] + columns["smoker"]["false"] == 400
    assert columns["seen"]["min"] <= columns["seen"]["max"]
    assert all(len(column["digest"]) == 64 for column in columns.values())


def test_the_digests_are_those_of_version_one() -> None:
    """Pinned: a refactor, or another numpy or pandas, must not move what a package already holds."""
    table = pd.DataFrame(
        {
            "n": [1, 2, 3],
            "x": [1.5, None, -2.25],
            "t": ["a", None, "é"],
            "d": pd.to_datetime(["2020-01-02", None, "2021-03-04"]),
            "b": [True, False, True],
        },
        index=[2, 0, 1],
    )
    digests = {
        name: column["digest"][:16] for name, column in _columns(frame_profile(table)).items()
    }
    assert digests == PINNED_DIGESTS


def test_a_profile_reads_the_values_not_how_pandas_keeps_them() -> None:
    table = _table()
    profile = frame_profile(table)
    other = table.sample(frac=1, random_state=1)  # another order of the rows
    other["sex"] = other["sex"].astype("category")
    other["seen"] = other["seen"].astype("datetime64[s]")
    other["age"] = other["age"].astype("Int64")
    assert frame_profile(other) == profile
    # whole numbers held as integers or as floats; booleans, dates and numbers held as objects
    held = {
        "whole": (pd.Series([1, 2, 3]), pd.Series([1.0, 2.0, 3.0])),
        "nullable": (pd.Series([1, None, 3], dtype="Int64"), pd.Series([1.0, np.nan, 3.0])),
        "boolean": (
            pd.Series([True, None, False], dtype="boolean"),
            pd.Series([True, None, False], dtype=object),
        ),
        "date": (
            pd.Series(pd.to_datetime(["2020-01-02", "2021-03-04"])),
            pd.Series(
                [pd.Timestamp("2020-01-02").date(), pd.Timestamp("2021-03-04").date()], dtype=object
            ),
        ),
        "numbers as objects": (pd.Series([1.5, 2.5]), pd.Series([1.5, 2.5], dtype=object)),
        "text": (
            pd.Series(["a", None, "b"], dtype=object),
            pd.Series(["a", None, "b"], dtype="string"),
        ),
    }
    for case, (one, two) in held.items():
        assert column_profile(one) == column_profile(two), case
    assert column_profile(pd.Series([1, 2, 3]))["whole"] is True
    # an integer beyond 2**53 is not the float nearest to it
    large = column_profile(pd.Series([2**60 + 1, 2**60 + 2]))
    assert large["digest"] != column_profile(pd.Series([2**60 + 2, 2**60 + 2]))["digest"]
    # a date far from now does not fit in nanoseconds since 1970: the profile reads it all the same
    far = column_profile(pd.Series(np.array(["9999-12-31", "1899-12-30"], dtype="datetime64[s]")))
    assert (far["min"], far["max"]) == ("1899-12-30", "9999-12-31")
    stamp = pd.Series(pd.to_datetime(["2021-03-04T05:06:07.5"], format="ISO8601"))
    assert column_profile(stamp)["max"] == "2021-03-04T05:06:07.5"
    assert (
        column_profile(pd.Series([1.5, -0.0]))["digest"]
        == column_profile(pd.Series([1.5, 0.0]))["digest"]
    )
    assert column_profile(pd.Series([1.0, None]))["missing"] == 1
    assert column_profile(pd.Series([], dtype="float64"))["rows"] == 0
    # two columns of one name keep a profile each
    twice = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["v", "v"])
    assert [c["sum"] for c in frame_profile(twice)["columns"]] == [4.0, 6.0]
    # row labels that are not whole numbers: no fingerprint of them, and the rows as given
    assert frame_profile(pd.DataFrame({"v": [1.0, 2.0]}, index=[1.2, 1.7]))["rows_fp"] is None


def test_rounding_error_is_within_tolerance_and_a_changed_value_fails() -> None:
    table = _table()
    recorded = frame_profile(table)
    assert (
        compare_profiles(recorded, frame_profile(table.copy()), name="step 3").outcome == IDENTICAL
    )

    noisy = table.copy()
    noisy["bmi"] = noisy["bmi"] * (1 + 4e-16)  # another BLAS
    close = compare_profiles(recorded, frame_profile(noisy), name="step 3")
    assert (close.name, close.outcome) == ("step 3", WITHIN_TOLERANCE)
    assert close.reason == "1 numeric column(s) differ within the rounding error of their sums"
    assert (
        close.largest["class"] == "rounding"
        and close.largest["difference"] <= close.largest["allowed"]
    )

    changed = table.copy()
    changed.loc[changed.index[10], "bmi"] += 0.5  # one participant's value
    failed = compare_profiles(recorded, frame_profile(changed))
    assert failed.outcome == FAILED and failed.reason == "column 2 does not hold the same values"

    for column, value in (
        ("age", 30),
        ("sex", "X"),
        ("seen", pd.Timestamp("2030-01-01")),
        ("smoker", True),
    ):
        other = table.copy()
        other.loc[other.index[5], column] = (
            value if other.loc[other.index[5], column] != value else None
        )
        assert compare_profiles(recorded, frame_profile(other)).outcome == FAILED, (
            column
        )  # equal or not


def test_a_change_is_a_failure_however_large_the_table() -> None:
    """The allowance grew with the rows: one value moved by 0.2 in a million, or two values exchanged in a column of
    the tutorial's size, read as "last digits only"."""
    rng = np.random.default_rng(12)
    million = pd.DataFrame({"bmi": rng.normal(27, 4, 1_000_000)})
    recorded = frame_profile(million)
    moved = million.copy()
    moved.iloc[123_456, 0] += 0.2
    assert compare_profiles(recorded, frame_profile(moved)).outcome == FAILED
    close = million * (1 + 2e-16)
    assert compare_profiles(recorded, frame_profile(close)).outcome == WITHIN_TOLERANCE

    tutorial = pd.DataFrame({"bmi": rng.normal(27, 4, 1206).round(1)})
    recorded = frame_profile(tutorial)
    for first, second in (
        (10, 997),
        (10, 11),
        (0, 1205),
        (300, 544),
    ):  # 987 rows apart: close weights in one sum
        swapped = tutorial.copy()
        a, b = swapped.iloc[first, 0], swapped.iloc[second, 0]
        if a == b:
            b = a + 0.1
            tutorial.iloc[second, 0] = b
            recorded = frame_profile(tutorial)
            swapped = tutorial.copy()
        swapped.iloc[first, 0], swapped.iloc[second, 0] = b, a
        assert compare_profiles(recorded, frame_profile(swapped)).outcome == FAILED, (first, second)

    rows = pd.DataFrame({"bmi": rng.normal(27, 4, 100_000).round(1)})
    recorded = frame_profile(rows)
    exchanged = rows.copy()
    exchanged.iloc[5, 0], exchanged.iloc[70_000, 0] = 28.0, 27.9
    rows.iloc[5, 0], rows.iloc[70_000, 0] = 27.9, 28.0
    assert compare_profiles(frame_profile(rows), frame_profile(exchanged)).outcome == FAILED


def test_a_column_with_an_infinite_value_is_still_compared() -> None:
    """One infinite value (a ratio with a zero denominator) left every sum without a number: whatever else changed
    in the column read "within tolerance"."""
    table = _table()
    table["ratio"] = table["bmi"] / 3.0
    table.loc[table.index[4], "ratio"] = np.inf
    recorded = frame_profile(table)
    column = dict(zip(recorded["names"], recorded["columns"], strict=True))["ratio"]
    assert (column["plus_inf"], column["minus_inf"]) == (1, 0) and column["sum"] is not None

    moved = table.copy()
    moved.loc[moved.index[20], "ratio"] += 5.0  # another participant's value
    assert compare_profiles(recorded, frame_profile(moved)).outcome == FAILED
    exchanged = table.copy()
    rows = exchanged.index[30:130]
    exchanged.loc[rows, "ratio"] = exchanged.loc[rows[::-1], "ratio"].to_numpy()
    assert compare_profiles(recorded, frame_profile(exchanged)).outcome == FAILED
    noisy = table.copy()
    noisy["ratio"] = noisy["ratio"] * (1 + 4e-16)
    assert compare_profiles(recorded, frame_profile(noisy)).outcome == WITHIN_TOLERANCE
    other = table.copy()
    other.loc[other.index[4], "ratio"] = -np.inf  # the other infinity: not the same values
    assert compare_profiles(recorded, frame_profile(other)).outcome == FAILED


def test_a_column_an_iterative_step_wrote_is_allowed_its_tolerance_and_says_so() -> None:
    table = _table()
    recorded = frame_profile(table)
    refit = table.copy()
    refit["bmi"] = refit["bmi"] * (1 + 3e-8)  # a fit that stopped a step earlier
    assert compare_profiles(recorded, frame_profile(refit)).outcome == FAILED
    allowed = compare_profiles(recorded, frame_profile(refit), iterative=["bmi"])
    assert allowed.outcome == WITHIN_TOLERANCE and allowed.largest["class"] == "iterative step"
    assert allowed.reason == "1 column(s) written by an iterative step differ within its tolerance"
    far = table.copy()
    far["bmi"] = far["bmi"] * (1 + 1e-4)
    assert compare_profiles(recorded, frame_profile(far), iterative=["bmi"]).outcome == FAILED


def test_other_rows_columns_or_rules_are_said() -> None:
    table = _table()
    recorded = frame_profile(table)
    assert (
        compare_profiles(recorded, frame_profile(table.iloc[1:])).reason
        == "the rows are not the same"
    )
    renamed = frame_profile(table.rename(columns={"bmi": "imc"}))
    assert (
        compare_profiles(recorded, renamed).reason
        == "the columns are not the same, or not in the same order"
    )
    older = {**recorded, "version": 0}
    assert compare_profiles(older, recorded).outcome == NOT_COMPARABLE
