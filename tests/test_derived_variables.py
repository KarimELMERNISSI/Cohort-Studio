"""
Declared derived variables (core.derived) and their check against the data (enrich.derived): a definition is study
data kept with the criteria, and the data say whether its formula gives the recorded values, within the precision they
are recorded with, before it is ever used to fill or impute.
"""

import numpy as np
import pandas as pd
import pytest

from core.derived import (
    CONFIG_KEY,
    DerivedVariable,
    cycles,
    fill_order,
    from_config,
    merge_definitions,
    shape_problems,
    with_definitions,
)
from core.population import configuration_problems
from enrich.derived import (
    AGREES,
    CANNOT,
    DISAGREES,
    MOSTLY,
    TOO_FEW,
    check_derived,
    check_inverses,
    resolution,
)

BMI = "weight_kg / (height_cm / 100) ** 2"
WEIGHT = "bmi * (height_cm / 100) ** 2"


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    """400 patients, the weight to 100 g, the height to 1 cm, the BMI computed from them to 0.1 (30 missing)."""
    rng = np.random.default_rng(1)
    n = 400
    weight = np.round(rng.normal(75, 12, n), 1)
    height = np.round(rng.normal(170, 9, n), 0)
    frame = pd.DataFrame(
        {
            "weight_kg": weight,
            "height_cm": height,
            "bmi": np.round(weight / (height / 100) ** 2, 1),
            "sex": rng.choice(["F", "M"], n),
        }
    )
    frame.loc[rng.choice(n, 30, replace=False), "bmi"] = np.nan
    return frame


# --- the declaration --------------------------------------------------------------------------------------------------


def test_a_definition_is_kept_by_its_variable_and_read_back() -> None:
    bmi = DerivedVariable("bmi", BMI, inverses={"weight_kg": WEIGHT}, kind="ratio")
    config = with_definitions({"mask_families": {}}, [bmi])
    assert config[CONFIG_KEY] == {
        "bmi": {"formula": BMI, "inverses": {"weight_kg": WEIGHT}, "kind": "ratio"}
    }
    assert from_config(config) == (bmi,)
    assert CONFIG_KEY not in with_definitions(config, [])
    # what it computes: not its inverses (each has its own hash), origin, note or acceptance
    same = DerivedVariable("bmi", f" {BMI} ", origin="ai", note="WHO", accepted=True)
    assert bmi.semantic_hash == same.semantic_hash != DerivedVariable("bmi", WEIGHT).semantic_hash
    assert bmi.inverse_hash("weight_kg") != bmi.semantic_hash
    # a kind or an origin this version does not know is only shown: read as the default
    odd = from_config({CONFIG_KEY: {"bmi": {"formula": BMI, "kind": "magic", "origin": "robot"}}})
    assert odd[0].kind == "" and odd[0].origin == "user"


@pytest.mark.parametrize(
    ("entries", "problem"),
    [
        ([{"target": "bmi", "formula": BMI}], "must map each variable"),
        ({"bmi": "weight / height"}, "is not a definition"),
        ({"bmi": {"formula": ""}}, "has no formula"),
        ({"bmi": {"formula": BMI, "inverses": {"weight_kg": ""}}}, "inverses must map"),
        ({"bmi": {"formula": BMI, "tolerance": "small"}}, "the tolerance must be a number"),
        (
            {"bmi": {"formula": BMI, "tolerance": -1}},
            "the tolerance must be a number, zero or more",
        ),
        ({"bmi": {"formula": BMI, "accepted": "yes"}}, "accepted must be true or false"),
    ],
)
def test_an_unreadable_list_is_refused_with_the_configuration(
    entries: object, problem: str
) -> None:
    assert any(problem in p for p in shape_problems(entries))
    # an uploaded config.json with it is refused, as with a criterion of an unknown shape
    assert any(problem in p for p in configuration_problems({CONFIG_KEY: entries}))
    assert from_config({CONFIG_KEY: entries}) == ()


def test_an_uploaded_list_adds_its_definitions_whole() -> None:
    # architecture review: the page's merge skipped a list the session already had, and said it had merged
    session = {"bmi": {"formula": BMI}}
    uploaded = {
        "bmi": {"formula": "weight_kg / height_cm ** 2"},
        "bsa": {"formula": "0.007184 * x"},
    }
    merged, conflicts = merge_definitions(session, uploaded)
    assert merged == {"bmi": {"formula": BMI}, "bsa": {"formula": "0.007184 * x"}}
    assert conflicts == ["bmi"]
    assert merge_definitions(session, {"bmi": {"formula": BMI}}) == (session, [])


def test_definitions_are_filled_in_an_order_and_never_in_a_circle() -> None:
    bmi = DerivedVariable("bmi", BMI)
    bmi_class = DerivedVariable("bmi_class", "np.where(bmi >= 25, 'overweight', 'normal')")
    weight = DerivedVariable("weight_kg", WEIGHT)
    inputs = {
        "bmi": ["weight_kg", "height_cm"],
        "bmi_class": ["bmi"],
        "weight_kg": ["bmi", "height_cm"],
    }
    assert fill_order([bmi_class, bmi], inputs) == ["bmi", "bmi_class"]
    assert cycles([bmi, weight], inputs) and fill_order([bmi, weight], inputs) is None
    assert not cycles([bmi, bmi_class], inputs)


# --- the check against the data ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "unit"),
    [
        ([70.1, 82.5, 64.0, 91.3], 0.1),
        (list(range(20, 70, 2)), 1.0),  # ages in years
        ([1, 2, 3, 1, 2], 0.0),  # codes, and scores of up to twenty values: no rounding
        (list(range(11)) * 3, 0.0),
        ([25.95155709342561, 24.2214532871972], 0.0),  # computed to full precision
    ],
)
def test_the_precision_a_column_is_recorded_with(values: list, unit: float) -> None:
    assert resolution(pd.Series(values)) == unit


def test_a_definition_the_data_agree_with(cohort: pd.DataFrame) -> None:
    check = check_derived(DerivedVariable("bmi", BMI), cohort)
    assert check.verdict == AGREES and check.usable and check.label == "agrees"
    assert (check.compared, check.agreeing, check.fillable) == (370, 370, 30)
    # computed from the recorded weight and height: within the BMI's own rounding
    assert check.strict and check.tolerance is not None and abs(check.tolerance - 0.05) < 1e-6
    assert check.evidence == (
        "gives the recorded value in 370 of 370 rows where bmi, weight_kg and height_cm are recorded (±0.05)"
    )
    inverses = check_inverses(DerivedVariable("bmi", BMI, inverses={"weight_kg": WEIGHT}), cohort)
    # the weight recovered from a BMI rounded to 0.1: within what that rounding moves it
    assert inverses["weight_kg"].verdict == AGREES and not inverses["weight_kg"].strict


def test_an_entry_error_of_two_tenths_is_caught(cohort: pd.DataFrame) -> None:
    # biostatistics review: with the variables' rounding added, +0.2 passed in 7 rows of 10
    errors = cohort.copy()
    wrong = errors.index[errors["bmi"].notna()][:3]
    errors.loc[wrong, "bmi"] += 0.2
    check = check_derived(DerivedVariable("bmi", BMI), errors)
    assert check.verdict == AGREES and check.strict
    assert sorted(check.disagreeing.tolist()) == sorted(wrong.tolist())


def test_a_disagreement_is_said_as_such(cohort: pd.DataFrame) -> None:
    check = check_derived(DerivedVariable("bmi", "weight_kg / (height_cm / 100) ** 2.1"), cohort)
    assert check.verdict == DISAGREES and check.label == "disagrees" and not check.usable
    assert check.evidence.startswith(
        "gives the recorded value in 0 of 370 rows where bmi, weight_kg"
    )
    assert check.unit_hint is None  # a wrong exponent: its ratio is no unit's factor


def test_another_unit_is_named_when_a_constant_factor_explains_the_disagreement(
    cohort: pd.DataFrame,
) -> None:
    check = check_derived(DerivedVariable("bmi", "weight_kg / height_cm ** 2"), cohort)
    assert check.verdict == DISAGREES
    assert check.unit_hint == "the recorded values are 10,000 times the formula's: another unit?"
    mixed = cohort.copy()
    mixed.loc[:39, "bmi"] *= 1e4  # 40 rows from a site recording another unit
    check = check_derived(DerivedVariable("bmi", BMI), mixed)
    assert check.verdict == MOSTLY and check.unit_hint is not None
    # mostly agreeing: used only once the user accepts it
    assert not check.usable
    accepted = DerivedVariable("bmi", BMI, accepted=True)
    assert check_derived(accepted, mixed).usable


def test_a_formula_wrong_for_one_level_disagrees() -> None:
    # biostatistics review: a wrong factor for women (8% of rows) came out "mostly agrees"
    rng = np.random.default_rng(7)
    n = 400
    age = rng.integers(30, 85, n).astype(float)
    weight = np.round(rng.normal(75, 12, n), 1)
    creatinine = np.round(rng.normal(90, 20, n), 0)
    sex = np.where(rng.random(n) < 0.08, "F", "M")
    clearance = (140 - age) * weight / (0.815 * creatinine) * np.where(sex == "F", 0.85, 1.0)
    frame = pd.DataFrame(
        {
            "age": age,
            "weight": weight,
            "creatinine": creatinine,
            "sex": sex,
            "crcl": np.round(clearance, 1),
        }
    )
    wrong = "(140 - age) * weight / (0.815 * creatinine) * np.where(sex == 'F', 0.95, 1.0)"
    check = check_derived(DerivedVariable("crcl", wrong), frame)
    assert check.verdict == DISAGREES and (check.reason or "").startswith(
        "it disagrees where sex = F"
    )
    right = wrong.replace("0.95", "0.85")
    assert check_derived(DerivedVariable("crcl", right), frame).verdict == AGREES


def test_a_total_of_items_is_held_to_its_own_rounding() -> None:
    # biostatistics review: ten items scored 0-10 gave a tolerance of 5.5, and totals off by 4 agreed
    rng = np.random.default_rng(3)
    items = pd.DataFrame(rng.integers(0, 11, (300, 10)), columns=[f"q{i}" for i in range(10)])
    frame = items.assign(total=items.sum(axis=1))
    frame.loc[:5, "total"] += 4
    formula = " + ".join(items.columns)
    check = check_derived(DerivedVariable("total", formula), frame)
    # the six wrong totals are named: 98% agree, which the user must accept before the total is used
    assert check.strict and check.tolerance == pytest.approx(0.5)
    assert check.verdict == MOSTLY and sorted(check.disagreeing.tolist()) == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    ("target", "formula", "reason"),
    [
        ("bmi", "bmi * 1", "uses bmi, the variable it computes"),
        ("bmi", "weight_kg.mean() + 0 * height_cm", "it uses other rows (mean())"),
        ("bmi", "weight_kg / waist_cm", "not in the data: waist_cm"),
        ("waist_cm", BMI, "waist_cm is not in the data"),
        ("bmi", "weight_kg ^ 2", "^ is not a power"),
        ("bmi", "42", "uses no variable"),
    ],
)
def test_a_definition_that_cannot_be_used(
    cohort: pd.DataFrame, target: str, formula: str, reason: str
) -> None:
    check = check_derived(DerivedVariable(target, formula, accepted=True), cohort)
    assert check.verdict == CANNOT and not check.usable and reason in (check.reason or "")


def test_too_few_rows_can_be_accepted_and_a_constant_variable_is_said(cohort: pd.DataFrame) -> None:
    few = check_derived(DerivedVariable("bmi", BMI), cohort.head(40))
    assert few.verdict == TOO_FEW and "the check needs 60" in (few.reason or "") and not few.usable
    assert few.fillable == int(cohort.head(40)["bmi"].isna().sum())
    assert check_derived(DerivedVariable("bmi", BMI, accepted=True), cohort.head(40)).usable
    # one height for everyone: the formula is checked for it only, which the check says
    constant = cohort.assign(height_cm=170.0, bmi=np.round(cohort["weight_kg"] / 1.7**2, 1))
    check = check_derived(DerivedVariable("bmi", BMI), constant)
    assert check.verdict == AGREES
    assert any(w.startswith("height_cm takes one value") for w in check.warnings)


def test_a_class_is_compared_as_text(cohort: pd.DataFrame) -> None:
    classes = cohort.assign(
        bmi_class=np.where(
            cohort["bmi"].isna(), None, np.where(cohort["bmi"] >= 25, "overweight", "normal")
        )
    )
    formula = "np.where(weight_kg / (height_cm / 100) ** 2 >= 25, 'overweight', 'normal')"
    check = check_derived(DerivedVariable("bmi_class", formula), classes)
    # the BMI was rounded before it was classed: two rows on the boundary differ
    assert (
        check.verdict == AGREES and check.tolerance is None and check.compared - check.agreeing == 2
    )


def test_a_date_is_checked_and_never_filled() -> None:
    start = pd.Series(pd.date_range("2024-01-01", periods=120, freq="D"))
    frame = pd.DataFrame({"start": start, "days": np.arange(120.0) % 30 + 1})
    frame["end"] = frame["start"] + pd.to_timedelta(frame["days"], unit="D")
    frame.loc[:4, "end"] = pd.NaT
    check = check_derived(DerivedVariable("end", "start + pd.to_timedelta(days, unit='D')"), frame)
    assert not check.fills and check.fillable == 0
    assert any("is a date" in w for w in check.warnings)


def test_a_copy_of_one_variable_is_flagged(cohort: pd.DataFrame) -> None:
    copied = cohort.assign(weight_copy=cohort["weight_kg"])
    check = check_derived(DerivedVariable("weight_copy", "weight_kg * 1"), copied)
    assert check.verdict == AGREES and any("multiple of weight_kg" in w for w in check.warnings)
    conversion = DerivedVariable("weight_copy", "weight_kg * 1", kind="conversion")
    assert not check_derived(conversion, copied).warnings


def test_a_formula_wrong_everywhere_is_not_blamed_on_one_level() -> None:
    frame = pd.DataFrame(
        {"status": ["Current", "Former", "Never"] * 40, "smoker": ["Yes", "No", "No"] * 40}
    )
    # 'current' for 'Current': "No" everywhere, wrong for the current smokers and right for the others
    check = check_derived(
        DerivedVariable("smoker", "np.where(status == 'current', 'Yes', 'No')"), frame
    )
    assert check.verdict == DISAGREES and not (check.reason or "").startswith("it disagrees where")
    fixed = DerivedVariable("smoker", "np.where(status == 'Current', 'Yes', 'No')")
    assert check_derived(fixed, frame).verdict == AGREES


def test_dates_kept_as_text_are_converted_first() -> None:
    frame = pd.DataFrame(
        {"start": ["01/02/2024", "15/03/2024"] * 40, "end": ["03/02/2024", "20/03/2024"] * 40}
    ).assign(days=[2, 5] * 40)
    check = check_derived(DerivedVariable("days", "end - start"), frame)
    assert check.verdict == CANNOT and (check.reason or "").startswith("the dates of")
    assert "convert them to dates on Validate first" in (check.reason or "")
