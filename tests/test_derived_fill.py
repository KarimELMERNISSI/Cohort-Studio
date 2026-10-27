"""
Filling the declared derived variables exactly (enrich.derived.plan_exact_fills, fill_steps; core.derived.record):
each formula then its inverses, in the order the definitions depend on each other, a variable's exact formulas first;
rounded as each variable is written; counted as observed only when the data agree, rounding moves the values little and
no variable of the formula was single-imputed; limited to the levels the check saw in enough rows; recorded with the
definition used, so that the replay computes the same values without the declared list.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from core.derived import (
    KINDS,
    NOTE_LENGTH,
    ORIGINS,
    DerivedVariable,
    from_config,
    from_record,
    note_text,
    record,
)
from enrich.derived import (
    AGREES,
    CANNOT,
    DISAGREES,
    LABELS,
    MAX_LEVELS,
    MOSTLY,
    TOO_FEW,
    check_derived,
    check_inverses,
    fill_steps,
    plan_exact_fills,
)

BMI = "weight_kg / (height_cm / 100) ** 2"
HEIGHT = "100 * (weight_kg / bmi) ** 0.5"


def _cohort(n: int = 300, seed: int = 4) -> pd.DataFrame:
    """Weights and heights to 0.1, the BMI computed from them to 0.1, with the three kinds of gaps."""
    rng = np.random.default_rng(seed)
    weight = np.round(rng.normal(78, 13, n), 1)
    height = np.round(rng.normal(171, 9, n), 1)
    frame = pd.DataFrame(
        {"weight_kg": weight, "height_cm": height, "bmi": np.round(weight / (height / 100) ** 2, 1)}
    )
    frame.loc[0:9, "bmi"] = np.nan  # the formula fills them
    frame.loc[10:17, "height_cm"] = np.nan  # the BMI and the weight give the height
    frame.loc[18:21, ["height_cm", "bmi"]] = np.nan  # neither: left missing
    return frame


def _bmi(**kwargs) -> DerivedVariable:
    return DerivedVariable("bmi", BMI, inverses={"height_cm": HEIGHT}, **kwargs)


def _filled(plan, frame: pd.DataFrame, rows: pd.Index | None = None) -> pd.DataFrame:
    """The table once every fill of the plan is applied, as its steps apply them."""
    steps = fill_steps(plan, frame, rows, "r")
    return steps[-1][0] if steps else frame


# --- the record a step keeps ------------------------------------------------------------------------------------------


def test_the_hashes_and_ids_a_trace_holds_never_change() -> None:
    # stored in traces: a change would make every recorded definition read as edited, or unreadable
    tutorial = DerivedVariable(
        "bmi_kg_m2",
        "weight_kg / (height_cm / 100) ** 2",
        inverses={"height_cm": "100 * (weight_kg / bmi_kg_m2) ** 0.5"},
    )
    assert tutorial.semantic_hash == "ded493dea1f61df4"
    assert tutorial.inverse_hash("height_cm") == "109ad1e5cac201f1"
    # what the definition computes only: not its labels, its acceptance or surrounding spaces
    labelled = DerivedVariable(
        "bmi_kg_m2",
        "  weight_kg / (height_cm / 100) ** 2 ",
        note="WHO",
        origin="ai",
        accepted=True,
        kind="ratio",
    )
    assert labelled.semantic_hash == tutorial.semantic_hash
    assert (AGREES, MOSTLY, DISAGREES, TOO_FEW, CANNOT) == (
        "agrees",
        "mostly",
        "disagrees",
        "too_few",
        "cannot",
    )
    assert set(LABELS) == {AGREES, MOSTLY, DISAGREES, TOO_FEW, CANNOT}
    assert ORIGINS == ("user", "ai", "import")
    assert KINDS == ("conversion", "ratio", "total", "index", "equation", "class", "")
    block = record(tutorial, ["bmi_kg_m2"], {}, {})
    assert list(block) == ["definition", "hash", "uses", "check", "run"]


def test_a_record_holds_the_definition_and_its_numbers_only() -> None:
    definition = _bmi(note="WHO", origin="ai")
    check = check_derived(definition, _cohort())
    block = record(
        definition, ["bmi", "height_cm"], check.summary(), {"id": "r1", "fill": 1, "of": 2}
    )
    # the note is free text a sealed step could never lose: the configuration keeps it, the record does not
    assert block["definition"] == {
        "target": "bmi",
        "formula": BMI,
        "inverses": {"height_cm": HEIGHT},
        "origin": "ai",
    }
    assert block["uses"] == {"bmi": None, "height_cm": definition.inverse_hash("height_cm")}
    assert set(block["check"]) == {
        "verdict",
        "compared",
        "agreeing",
        "tolerance",
        "strict",
        "kappa",
    }
    read, uses, problems = from_record(json.loads(json.dumps(block)))
    assert read is not None and read.semantic_hash == definition.semantic_hash
    assert read.inverses == definition.inverses and uses == block["uses"] and problems == []
    # edited after it was recorded: the hashes say so
    block["definition"]["formula"] = "weight_kg / (height_cm / 100) ** 2.1"
    block["definition"]["inverses"]["height_cm"] = "100 * (weight_kg / bmi) ** 0.6"
    assert len(from_record(block)[2]) == 2
    assert from_record({"definition": "bmi"})[2] == ["the record holds no readable definition"]


def test_no_word_of_a_check_reaches_the_record() -> None:
    # a formula wrong for women: the check's reason names the level, "where sex = F"
    rng = np.random.default_rng(7)
    n = 400
    sex = np.where(rng.random(n) < 0.08, "F", "M")  # few women: the others agree
    x = np.round(rng.normal(50, 10, n), 0)
    frame = pd.DataFrame(
        {"x": x, "sex": sex, "y": np.round(x * np.where(sex == "F", 0.85, 1.0), 1)}
    )
    definition = DerivedVariable("y", "x * np.where(sex == 'F', 0.95, 1.0)")
    check = check_derived(definition, frame)
    assert check.verdict == DISAGREES and "sex = F" in (check.reason or "")
    text = json.dumps(record(definition, ["y"], check.summary(), {"id": "r"}))
    assert "sex = F" not in text and "reason" not in text and "evidence" not in text


def test_a_note_is_one_short_line() -> None:
    assert note_text("  WHO;\n weight in kg,\theight in cm  ") == "WHO; weight in kg, height in cm"
    assert len(note_text("x" * 1000)) == NOTE_LENGTH and note_text(None) == ""
    (read,) = from_config({"derived_variables": {"bmi": {"formula": BMI, "note": "a\nb " * 200}}})
    assert "\n" not in read.note and len(read.note) == NOTE_LENGTH


# --- the check ---------------------------------------------------------------------------------------------------------


def test_the_check_leaves_imputed_cells_out() -> None:
    frame = _cohort()
    full = check_derived(_bmi(), frame)
    imputed = {"height_cm": np.array([30, 31, 32]), "bmi": np.array([40])}
    left_out = check_derived(_bmi(), frame, imputed=imputed)
    assert left_out.compared == full.compared - 4 and left_out.verdict == AGREES


def test_rounding_an_input_either_way_is_allowed_for() -> None:
    # weight and height to full precision, the BMI to the unit: the height it gives moves by up to about 1.6 cm, more
    # when the BMI was rounded up than down; allowing the larger move, every row agrees
    rng = np.random.default_rng(11)
    n = 2000
    weight, height = rng.normal(78, 13, n), rng.normal(171, 9, n)
    frame = pd.DataFrame(
        {"weight_kg": weight, "height_cm": height, "bmi": np.round(weight / (height / 100) ** 2, 0)}
    )
    height_check = check_inverses(_bmi(), frame)["height_cm"]
    assert height_check.verdict == AGREES and height_check.agreeing == height_check.compared


def test_a_declared_tolerance_is_not_the_variables_own_rounding() -> None:
    # an eGFR the laboratory computed otherwise, within 4 of the formula: a tolerance of 5 lets it agree, but its
    # values cannot count as observed
    rng = np.random.default_rng(12)
    n = 300
    x = rng.normal(70, 20, n)
    frame = pd.DataFrame({"x": np.round(x, 0), "y": np.round(x + rng.uniform(-4, 4, n), 0)})
    frame.loc[0:19, "y"] = np.nan
    plan = plan_exact_fills([DerivedVariable("y", "x * 1.0", tolerance=5.0)], frame)
    (fill,) = plan.fills
    assert fill.check.verdict == AGREES and not fill.check.strict
    assert fill.filled == 20 and not fill.exact and fill.single == 20


# --- the plan ---------------------------------------------------------------------------------------------------------


def test_each_formula_then_its_inverses_fill_rounded_as_recorded() -> None:
    frame = _cohort()
    plan = plan_exact_fills([_bmi()], frame)
    assert plan.problem is None and not plan.unused
    assert [(f.target, f.inverse, f.filled, f.exact, f.single, f.decimals) for f in plan.fills] == [
        ("bmi", False, 10, True, 0, 1),
        ("height_cm", True, 8, True, 0, 1),
    ]
    filled = _filled(plan, frame)
    rows = frame.index[0:10]
    expected = np.round(frame.loc[rows, "weight_kg"] / (frame.loc[rows, "height_cm"] / 100) ** 2, 1)
    assert filled.loc[rows, "bmi"].tolist() == expected.tolist()
    heights = filled.loc[10:17, "height_cm"]
    assert (heights == heights.round(1)).all() and heights.between(130, 215).all()
    assert filled.loc[18:21, ["height_cm", "bmi"]].isna().all().all()  # neither: left missing
    assert plan.filled == 18 and plan.left == {"bmi": 4, "height_cm": 4}


def test_only_the_analysis_population_is_filled() -> None:
    frame = _cohort()
    rows = frame.index[5:300]
    plan = plan_exact_fills([_bmi()], frame, rows=rows)
    assert [(f.target, f.filled) for f in plan.fills] == [("bmi", 5), ("height_cm", 8)]
    assert _filled(plan, frame, rows).loc[0:4, "bmi"].isna().all()


def test_values_rounding_moves_too_much_are_single_imputed() -> None:
    # y recorded to 0.1 from a and b before they were rounded: the formula moves by about 1.0, more than 5% of the
    # standard deviation of y (about 5.4): the values are estimates, not observations
    rng = np.random.default_rng(9)
    n = 300
    a, b = rng.normal(100, 5, n), rng.normal(1, 0.02, n)
    frame = pd.DataFrame({"a": np.round(a, 0), "b": np.round(b, 2), "y": np.round(a * b, 1)})
    frame.loc[0:9, "y"] = np.nan
    (fill,) = plan_exact_fills([DerivedVariable("y", "a * b")], frame).fills
    assert fill.check.verdict == AGREES and not fill.check.strict
    assert fill.filled == 10 and not fill.exact and fill.decimals == 1


def test_a_value_computed_from_an_imputed_one_is_imputed() -> None:
    frame = _cohort()
    imputed_heights = np.array(
        [0, 1, 2]
    )  # heights a mean imputation filled, where the BMI is missing
    cells = {"height_cm": imputed_heights}
    plan = plan_exact_fills([_bmi()], frame, imputed=cells, single=cells)
    bmi = plan.fills[0]
    assert (bmi.target, bmi.exact, bmi.filled, bmi.single) == ("bmi", True, 10, 3)


def test_a_variables_exact_formula_fills_before_its_others() -> None:
    # the height from the BMI is exact; from a ratio recorded to 0.01, it moves by about 2 cm: the BMI's inverse
    # fills the heights, though the ratio is declared first
    frame = _cohort()
    rng = np.random.default_rng(4)  # as _cohort draws them: the weights, then the heights
    rng.normal(78, 13, 300)
    true_height = np.round(rng.normal(171, 9, 300), 1)
    full = frame.assign(ratio=np.round(frame["weight_kg"] / true_height, 2))
    ratio = DerivedVariable(
        "ratio", "weight_kg / height_cm", inverses={"height_cm": "weight_kg / ratio"}
    )
    alone = plan_exact_fills([ratio], full)  # usable, but not exact
    (from_ratio,) = [f for f in alone.fills if f.target == "height_cm"]
    assert from_ratio.check.verdict == AGREES and not from_ratio.exact and from_ratio.filled == 12
    plan = plan_exact_fills([ratio, _bmi()], full)
    heights = [f for f in plan.fills if f.target == "height_cm"]
    assert [(f.definition.target, f.exact, f.filled) for f in heights] == [
        ("bmi", True, 8),  # where the BMI is recorded
        ("ratio", False, 4),  # where it is not: estimates
    ]
    # the BMI is filled after the heights: in 14 rows, 4 of them from the estimated heights, estimates too
    (bmi,) = [f for f in plan.fills if f.target == "bmi"]
    assert (bmi.exact, bmi.filled, bmi.single) == (True, 14, 4)


def test_only_usable_definitions_fill() -> None:
    frame = _cohort()
    frame.loc[30:39, "bmi"] *= 10_000  # a site recording another unit: the data mostly agree
    mostly = plan_exact_fills([_bmi()], frame)
    assert mostly.fills == () and "accept it first" in mostly.unused["bmi"]
    assert mostly.left == {}  # what no formula may fill is not said to lack a value
    accepted = plan_exact_fills([_bmi(accepted=True)], frame)
    bmi = next(f for f in accepted.fills if f.target == "bmi")
    assert bmi.check.verdict == MOSTLY and not bmi.exact  # accepted, not agreeing: single-imputed
    # the inverse fills only where the data agree with it, as a definition of its own
    assert all(f.check.verdict == AGREES for f in accepted.fills if f.inverse)
    wrong = plan_exact_fills([DerivedVariable("bmi", "weight_kg / height_cm ** 2.1")], frame)
    assert wrong.fills == () and wrong.unused == {"bmi": "the data disagree with it"}


def test_a_date_is_never_filled_and_a_circle_fills_nothing() -> None:
    start = pd.Series(pd.date_range("2024-01-01", periods=120, freq="D"))
    dates = pd.DataFrame({"start": start, "days": np.arange(120.0) % 30 + 1})
    dates["end"] = dates["start"] + pd.to_timedelta(dates["days"], unit="D")
    dates.loc[:4, "end"] = pd.NaT
    plan = plan_exact_fills(
        [DerivedVariable("end", "start + pd.to_timedelta(days, unit='D')")], dates
    )
    assert plan.fills == () and plan.unused == {"end": "a date is never filled"}
    circle = [
        DerivedVariable("bmi", BMI),
        DerivedVariable("weight_kg", "bmi * (height_cm / 100) ** 2"),
    ]
    assert "compute each other" in (plan_exact_fills(circle, _cohort()).problem or "")


def test_a_fill_gives_another_formula_its_value() -> None:
    # the class needs the BMI, which only the weight and the height give: filled in that order
    frame = _cohort()
    frame["obese"] = np.where(frame["bmi"].isna(), None, np.where(frame["bmi"] >= 30, "yes", "no"))
    obese = DerivedVariable("obese", "np.where(bmi >= 30, 'yes', 'no')")
    plan = plan_exact_fills([obese, _bmi()], frame)
    assert [f.target for f in plan.fills][:2] == ["bmi", "height_cm"]
    classes = [f for f in plan.fills if f.target == "obese"]
    assert classes and sum(f.filled for f in classes) >= 10 and classes[0].decimals is None
    assert _filled(plan, frame).loc[0:17, "obese"].notna().all()


def _by_sex(n: int = 200, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sex = rng.choice(["F", "M"], n)
    x = np.round(rng.normal(50, 10, n), 0)
    return pd.DataFrame({"x": x, "sex": sex, "y": np.round(x * np.where(sex == "F", 0.85, 1.0), 1)})


def test_a_level_the_check_saw_too_rarely_is_neither_filled_nor_recorded() -> None:
    frame = _by_sex()
    frame.loc[0:4, "y"] = np.nan
    frame.loc[0:1, "sex"] = "X"  # where y is missing: never compared
    frame.loc[20:22, "sex"] = "Other"  # compared in 3 rows only: the level rule cannot test it
    frame.loc[20:22, "y"] = frame.loc[20:22, "x"]  # as the formula computes them
    frame.loc[3, "sex"] = "Other"
    definition = DerivedVariable("y", "x * np.where(sex == 'F', 0.85, 1.0)")
    plan = plan_exact_fills([definition], frame)
    (fill,) = plan.fills
    assert fill.filled == 2 and fill.levels == {"sex": ("F", "M")}
    filled = _filled(plan, frame)
    assert filled.loc[[0, 1, 3], "y"].isna().all() and plan.left == {"y": 3}
    (_, params, _) = fill_steps(plan, frame, None, "r")[0]
    assert params["only_levels"] == {"sex": ["F", "M"]}  # never "X" or "Other"
    assert "Other" not in json.dumps(params) and "X" not in params["only_levels"]["sex"]


def test_a_text_of_many_values_fills_nothing() -> None:
    rng = np.random.default_rng(5)
    n = 600
    centre = rng.choice(
        [f"centre {i:02d}" for i in range(25)], n
    )  # 25 values, each in about 24 rows
    x = np.round(rng.normal(50, 10, n), 0)
    frame = pd.DataFrame({"x": x, "centre": centre, "y": x + 1})
    frame.loc[0:9, "y"] = np.nan
    plan = plan_exact_fills([DerivedVariable("y", "x + np.where(centre == '', 0, 1)")], frame)
    assert plan.fills == () and f"more than {MAX_LEVELS} values" in plan.unused["y"]


# --- the steps, applied and replayed ----------------------------------------------------------------------------------


def test_the_steps_replay_to_the_same_cells(tmp_path) -> None:
    from core.session import DataSession
    from core.state import ParseOptions
    from ingest.loaders import read_table
    from manage.paths import PathResolver
    from manage.reproduction_manager import run_trace
    from manage.transformation_manager import TransformationManager

    source = tmp_path / "anthropometry.csv"
    _cohort().to_csv(source, index=False)
    tm = TransformationManager(trace_dir=str(tmp_path / "traces"))
    tm.initialize_session(source.name)
    tm.source_dataset = str(source)
    session = DataSession(trace=tm)
    options = ParseOptions()
    session.install(
        read_table(source.read_bytes(), source.name, options).df,
        params={"source_path": str(source), "parse_options": options.to_dict()},
        description="Loaded",
    )
    frame = session.working_frame()
    plan = plan_exact_fills([_bmi()], frame)
    steps = fill_steps(plan, frame, None, "run1")
    for filled, params, description in steps:
        session.stage(filled, "targeted_imputation", params, description)
    first = steps[0][1]
    assert first["round_decimals"] == 1 and first["exact"] and first["finite_only"]
    assert "only_levels" not in first
    assert first["derived"]["uses"] == {"bmi": None} and first["derived"]["run"]["of"] == 2
    assert steps[1][2] == (
        "Imputed 8 values in height_cm using an exact formula (an inverse of the declared definition of bmi); "
        "4 left missing, where bmi has no value"
    )
    session.commit_staging()
    assert {c: len(ids) for c, ids in session.exact_imputed.items()} == {"bmi": 10, "height_cm": 8}

    replay = run_trace(tm.get_trace(), paths=PathResolver.unconfined())
    assert replay.reproducible and not replay.warnings, replay.warnings
    pd.testing.assert_frame_equal(replay.df, session.base)
    assert {c: len(ids) for c, ids in replay.exact_imputed.items()} == {"bmi": 10, "height_cm": 8}
    assert [r.description for r in replay.records.values()] == [s[2] for s in steps]


@pytest.mark.parametrize("decimals", [None, 0, 1])
def test_the_fill_rounds_only_the_values_it_fills(decimals) -> None:
    from enrich.imputation_service import fill_with_formula

    frame = pd.DataFrame({"x": [1.234, 2.345, 3.456], "y": [9.87654, np.nan, np.nan]})
    out, n = fill_with_formula(frame, "y", "x * 3", complete_inputs=True, decimals=decimals)
    assert n == 2 and out.loc[0, "y"] == 9.87654  # a recorded value is never rounded
    expected = (
        [2.345 * 3, 3.456 * 3]
        if decimals is None
        else [round(v, decimals) for v in (7.035, 10.368)]
    )
    assert out.loc[1:2, "y"].tolist() == pytest.approx(expected)


def test_a_half_rounds_away_from_zero_as_records_do() -> None:
    from enrich.imputation_service import fill_with_formula

    frame = pd.DataFrame({"x": [2.5, 3.5, -0.5, 2.675], "y": [np.nan] * 4})
    out, _ = fill_with_formula(frame, "y", "x * 1", complete_inputs=True, decimals=0)
    assert out["y"].tolist() == [3.0, 4.0, -1.0, 3.0]  # not 2, 4, -0 (to the even neighbour)
    out, _ = fill_with_formula(frame, "y", "x * 1", complete_inputs=True, decimals=2)
    assert out.loc[3, "y"] == 2.68  # 2.675 is stored as 2.67499...: rounded as written


def test_an_infinity_is_filled_only_by_a_step_recorded_before_the_rule() -> None:
    from enrich.imputation_service import fill_with_formula

    frame = pd.DataFrame({"w": [70.0, 80.0], "h": [0.0, 1.8], "bmi": [np.nan, np.nan]})
    out, n = fill_with_formula(frame, "bmi", "w / h ** 2", complete_inputs=True, finite_only=True)
    assert n == 1 and pd.isna(out.loc[0, "bmi"])
    # a trace recorded earlier replays as it was computed
    old, n_old = fill_with_formula(frame, "bmi", "w / h ** 2", complete_inputs=True)
    assert n_old == 2 and np.isinf(old.loc[0, "bmi"])


# --- the single imputation that sets them aside -----------------------------------------------------------------------


@pytest.mark.parametrize("method", ["Mean", "KNN"])
def test_a_variable_set_aside_is_neither_imputed_nor_used_to_impute(method: str) -> None:
    from enrich.imputation_service import impute_rows

    frame = _cohort().assign(age=lambda f: np.round(np.linspace(30, 89, len(f)), 0))
    frame.loc[40:45, "age"] = np.nan
    aside = impute_rows(frame, None, numerical=method, set_aside=["bmi"])
    without = impute_rows(frame.drop(columns=["bmi"]), None, numerical=method)
    others = aside.frame.drop(columns=["bmi"]).reindex(columns=without.frame.columns)
    pd.testing.assert_frame_equal(others, without.frame)  # the BMI predicted nothing
    assert aside.frame["bmi"].equals(frame["bmi"]) and aside.params["set_aside"] == ["bmi"]
    assert "bmi" not in aside.params["imputed_values"]
    part = impute_rows(frame, frame.index[100:], numerical=method, set_aside=["bmi"])
    assert part.frame["bmi"].equals(frame["bmi"]) and list(part.frame.columns) == list(
        frame.columns
    )


def test_an_imputation_with_its_derived_variables_recomputed_replays(tmp_path) -> None:
    from core.session import DataSession
    from core.state import ParseOptions
    from enrich.imputation_service import describe_imputation, impute_rows
    from ingest.loaders import read_table
    from manage.paths import PathResolver
    from manage.reproduction_manager import run_trace
    from manage.transformation_manager import TransformationManager

    source = tmp_path / "anthropometry.csv"
    _cohort().to_csv(source, index=False)
    tm = TransformationManager(trace_dir=str(tmp_path / "traces"))
    tm.initialize_session(source.name)
    tm.source_dataset = str(source)
    session = DataSession(trace=tm)
    options = ParseOptions()
    session.install(
        read_table(source.read_bytes(), source.name, options).df,
        params={"source_path": str(source), "parse_options": options.to_dict()},
        description="Loaded",
    )

    def fill(run: str) -> None:
        frame = session.working_frame()
        plan = plan_exact_fills(
            [_bmi()], frame, single=dict(session.draft.single_imputed) if session.draft else None
        )
        for filled, params, description in fill_steps(plan, frame, None, run):
            session.stage(filled, "targeted_imputation", params, description)

    fill("exact")  # 10 BMIs and 8 heights from the recorded values
    outcome = impute_rows(session.working_frame(), None, numerical="Median", set_aside=["bmi"])
    session.stage(
        outcome.frame,
        "imputation",
        outcome.params,
        describe_imputation("Median", "Most_Frequent", None, ["bmi"]),
    )
    fill("after")  # the 4 BMIs of the heights the median gave
    session.commit_staging()
    base = session.base
    assert not base["bmi"].isna().any()
    assert {c: len(ids) for c, ids in session.single_imputed.items()} == {"height_cm": 4, "bmi": 4}
    assert {c: len(ids) for c, ids in session.exact_imputed.items()} == {"bmi": 10, "height_cm": 8}

    replay = run_trace(tm.get_trace(), paths=PathResolver.unconfined())
    assert replay.reproducible and not replay.warnings, replay.warnings
    pd.testing.assert_frame_equal(replay.df, base)
    imputation = next(r for r in replay.records.values() if "set_aside" in r.params)
    assert imputation.description.endswith("set aside, to be computed from the imputed values: bmi")


def test_only_decisions_on_the_data_keep_a_draft_from_being_redone() -> None:
    from core.derived import decided_on_data

    fill = record(_bmi(), ["bmi"], {}, {})
    assert decided_on_data(fill) and decided_on_data({"recompute": True, "definitions": []})
    # an imputation that only records the choice not to compute them decided nothing on the data
    assert not decided_on_data({"recompute": False, "definitions": [fill]})
    assert not decided_on_data(None) and not decided_on_data({})


def test_a_recorded_value_its_imputed_variables_no_longer_match_is_counted() -> None:
    from enrich.derived import contradicted

    frame = _cohort()
    check = check_derived(_bmi(), frame)
    rows = frame.index[30:37]  # recorded BMIs whose heights an imputation then gave
    imputed = frame.copy()
    imputed.loc[rows, "height_cm"] = 150.0
    assert contradicted(_bmi(), check, imputed, {"height_cm": rows.to_numpy()}) == 7
    assert (
        contradicted(_bmi(), check, frame, {"height_cm": rows.to_numpy()}) == 0
    )  # the recorded heights match
