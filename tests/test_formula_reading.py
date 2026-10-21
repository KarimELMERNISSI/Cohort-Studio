"""One reading of a formula (enrich.formulas.read_formula): the evaluator, the AI engine's suggestions and their diagrams
read a formula the same way, and a model's notation is converted to the application's before anything uses it."""

import json

import pandas as pd
import pytest

from enrich.formulas import (
    CARET_IS_NOT_A_POWER,
    evaluate_formula_safely,
    formula_from_model,
    read_formula,
    try_formula,
    validate_and_extract_columns_ast,
)
from graph.builders.formula import build_formula_graph
from graph.model import GraphKind
from manage.ai_payload import column_kinds


@pytest.mark.parametrize(
    ("formula", "names"),
    [
        ("weight / (height ** 2)", ("weight", "height")),
        ('""Systolic BP"" - ""Diastolic BP""', ("Systolic BP", "Diastolic BP")),
        (
            "np.log(creatinine) * 1.5 + np.sqrt(bun)",
            ("creatinine", "bun"),
        ),  # functions are not variables
        ("b + a * b", ("b", "a")),  # in the order they first appear, once each
        ("np.sqrt(height * weight / 3600)", ("height", "weight")),
        ("round(bmi, 1) + abs(delta)", ("bmi", "delta")),  # the evaluator's functions, when called
        ("min_value + max", ("min_value", "max")),  # a column named like a function is a column
    ],
)
def test_a_formula_names_the_variables_it_uses(formula: str, names: tuple[str, ...]) -> None:
    reading = read_formula(formula)
    assert reading.names == names and reading.problem is None and not reading.power_caret


@pytest.mark.parametrize(
    "formula", ["weight / (height ** )", "", "bmi = weight / height ** 2", "`Heart Rate` / 60"]
)
def test_a_formula_that_cannot_be_read_says_why_and_names_nothing(formula: str) -> None:
    reading = read_formula(formula)
    assert reading.problem and reading.problem.startswith("Syntax Error") and reading.names == ()


def test_a_function_the_evaluator_lacks_is_refused_with_a_hint() -> None:
    assert read_formula("sqrt(height * weight / 3600)").problem == (
        "sqrt() is not a function of formulas: write np.sqrt(...)"
    )
    assert read_formula("frobnicate(x)").problem == "frobnicate() is not a function of formulas"


def test_a_formula_too_long_or_too_deep_to_read_says_so() -> None:
    assert read_formula("x + " * 1_000 + "x").problem == "a formula longer than 3,000 characters"
    deep = read_formula(
        "-" * 2_900 + "x"
    )  # an answer no model should give: read or refused, never a crash
    assert deep.problem or deep.names == ("x",)
    try_formula(
        "-" * 2_900 + "x", pd.DataFrame({"x": [1.0]})
    )  # computed or refused: never an exception


def test_a_caret_is_flagged_it_is_not_a_power_here() -> None:
    assert read_formula("weight / (height ^ 2)").power_caret
    data = pd.DataFrame({"height": [170, 180]})
    ok, xor, _ = evaluate_formula_safely("height ^ 2", "h2", data, return_series=True)
    assert ok and xor.tolist() == [168, 182]  # what ^ computes: why a new formula refuses it


@pytest.mark.parametrize(
    ("written", "converted"),
    [
        ("weight / (height ^ 2)", "weight / (height ** 2)"),
        ("`Heart Rate` / 60 + `Age (yr)` * 0.1", '""Heart Rate"" / 60 + ""Age (yr)"" * 0.1'),
        (
            "bmi = weight / height ** 2",
            "weight / height ** 2",
        ),  # the variable it computes: the right side
        ("sex = 'F'", "sex = 'F'"),  # SQL, not a formula: still refused, never the constant 'F'
        ('""a^b"" ^ 2', '""a^b"" ** 2'),  # a quoted column keeps its name
        ("grade == 'A^B' or x ^ 2", "grade == 'A^B' or x ** 2"),  # and a text keeps its ^
        ("a == b", "a == b"),  # a comparison is not an assignment
    ],
)
def test_a_models_notation_becomes_the_applications(written: str, converted: str) -> None:
    assert formula_from_model(written, targets=["bmi"]) == converted


def test_a_column_name_inside_a_text_stays_text() -> None:
    # the evaluator renamed the columns in the formula's text: 'Type 2 diabetes' lost its "diabetes", and the formula
    # answered 0 everywhere; columns named var_0 and var_1 were taken for one another
    data = pd.DataFrame(
        {
            "diabetes": ["Type 2 diabetes", "none", "Type 2 diabetes"],
            "var_0": [1.0, 2.0, 3.0],
            "var_1": [10.0, 20.0, 30.0],
        }
    )
    ok, flags, _ = evaluate_formula_safely(
        "np.where(diabetes == 'Type 2 diabetes', 1, 0)", "t2", data, return_series=True
    )
    assert ok and flags.tolist() == [1, 0, 1]
    ok, difference, _ = evaluate_formula_safely("var_1 - var_0", "d", data, return_series=True)
    assert ok and difference.tolist() == [9.0, 18.0, 27.0]
    ok, rounded, _ = evaluate_formula_safely("round(var_1 / 3, 1)", "r", data, return_series=True)
    assert ok and rounded.tolist() == [
        3.3,
        6.7,
        10.0,
    ]  # round is a function, no longer "an unknown variable"


def test_a_converted_suggestion_computes_what_its_diagram_shows() -> None:
    data = pd.DataFrame({"Heart Rate": [60.0, 120.0], "height": [2.0, 3.0]})
    formula = formula_from_model("`Heart Rate` / 60 * height ^ 2")
    reading = read_formula(formula)
    assert reading.names == ("Heart Rate", "height") and not reading.power_caret
    assert validate_and_extract_columns_ast(formula, list(data.columns))[:3] == (
        True,
        None,
        ["Heart Rate", "height"],
    )
    ok, values, _ = evaluate_formula_safely(formula, "x", data, return_series=True)
    assert ok and values.tolist() == [4.0, 18.0]
    spec = build_formula_graph("x", reading.names, data.columns)
    assert [n.category for n in spec.nodes] == ["input", "input", "target"]


def test_the_evaluator_names_the_first_unknown_variable() -> None:
    assert validate_and_extract_columns_ast("weight / height_m ** 2", ["weight"]) == (
        False,
        "Unknown variable: 'height_m'",
        [],
        [],
    )


def test_the_diagram_marks_the_missing_variables_and_needs_no_model_text() -> None:
    spec = build_formula_graph("bmi", ("weight_kg", "height_cm", "weight_kg"), ["weight_kg"])
    assert spec.kind is GraphKind.FORMULA and spec.directed
    assert [(n.id, n.category) for n in spec.nodes] == [
        ("input:weight_kg", "input"),
        ("input:height_cm", "missing"),
        ("target:bmi", "target"),
    ]
    assert {(e.source, e.target, e.category) for e in spec.edges} == {
        ("input:weight_kg", "target:bmi", "input"),
        ("input:height_cm", "target:bmi", "missing"),
    }
    assert spec.meta["missing"] == ["height_cm"]
    alone = build_formula_graph(
        "bmi", (), ["weight_kg"]
    )  # a formula that cannot be read: the variable alone
    assert [n.id for n in alone.nodes] == ["target:bmi"] and not alone.edges


def test_a_variable_named_like_the_target_is_still_drawn() -> None:
    spec = build_formula_graph("x", ("x",), ["x"])  # a corrected value of a column of the same name
    assert {n.id for n in spec.nodes} == {"input:x", "target:x"} and len(spec.edges) == 1


# --- the suggestions of the AI engine, tried on the data before they are offered (owner request) -----------------


def _visits() -> pd.DataFrame:
    """Twelve visits: NYHA classes held by several rows, one free-text note, a missing baseline and HbA1c."""
    return pd.DataFrame(
        {
            "sbp_baseline": [
                120.0,
                135.0,
                None,
                128.0,
                131.0,
                140.0,
                118.0,
                125.0,
                133.0,
                129.0,
                137.0,
                122.0,
            ],
            "nyha_class": ["II"] * 6 + ["III"] * 5 + ["Mrs Dupont, room 12"],
            "hba1c": [7.1, None, 5.4, 6.8, None, 5.9, 6.6, 7.4, 5.1, 6.0, 6.9, 5.5],
            "sbp_6m": [
                None,
                130.0,
                None,
                125.0,
                None,
                138.0,
                120.0,
                None,
                131.0,
                127.0,
                135.0,
                121.0,
            ],
        }
    )


def test_a_suggestion_is_tried_on_the_data_and_says_why_it_cannot_be_computed() -> None:
    data = _visits()
    refused = try_formula("sbp_baseline + np.where(nyha_class > 2, 5.0, -2.0)", data)
    assert not refused.ok and refused.values is None
    # the values held by several rows only: a note held by one row is never shown
    assert (
        "nyha_class holds text (such as 'II', 'III')" in refused.problem
        and "Dupont" not in refused.problem
    )
    fixed = try_formula("sbp_baseline + np.where(nyha_class == 'III', 5.0, -2.0)", data)
    assert fixed.ok and fixed.values.tolist()[:2] == [118.0, 133.0]
    assert try_formula("weight / height ^ 2", data).problem == CARET_IS_NOT_A_POWER
    assert try_formula("sbp_baseline * bsa", data).problem == "not in the data: bsa"
    assert try_formula("sbp_baseline +", data).problem.startswith("Syntax Error")


def test_a_trial_counts_only_values_its_inputs_allow() -> None:
    data = _visits()
    flagged = try_formula("np.where(hba1c >= 6.5, 1, 0)", data)
    # np.where answers "false" for a missing HbA1c: those rows get a value all the same, and are named
    assert int(flagged.given().sum()) == 12 and flagged.from_missing(data) == [("hba1c", 2)]
    assert flagged.from_missing(data, within=data["sbp_6m"].isna()) == [("hba1c", 1)]
    infinite = try_formula("sbp_baseline / 0", data)
    assert int(infinite.given().sum()) == 0  # an infinity fills nothing
    never = try_formula("np.where(nyha_class == 'Class III', 1, 0)", data)
    assert never.ok and never.warnings == ("'Class III' occurs in no row of nyha_class",)
    isin = try_formula("nyha_class.isin(['II', 'IV'])", data)
    assert isin.warnings == ("'IV' occurs in no row of nyha_class",)
    yes_no = try_formula("hba1c >= 6.5", data, target="sbp_6m")
    assert yes_no.problem == "it gives yes/no values, while sbp_6m holds numbers"


def test_a_formula_that_would_keep_the_server_busy_is_refused_at_once() -> None:
    import time

    data = pd.DataFrame({"weight": [70.0, 80.0]})
    started = time.perf_counter()
    for formula in ("weight * 0 * 9 ^ 9 ^ 8", "weight * np.ones(10 ** 9)", "'x' * 10 ** 8"):
        trial = try_formula(formula_from_model(formula), data)
        assert not trial.ok, formula
    assert time.perf_counter() - started < 5  # 9 ** 9 ** 8 alone took 111 s


def test_a_corrected_formula_that_is_too_long_is_dropped_at_once() -> None:
    import time

    from manage.rag_computed_vars import _unwrapped

    started = time.perf_counter()
    assert _unwrapped("```" + "\n" * 3_200 + "x") == ""  # the old pattern took 21 s on this
    assert time.perf_counter() - started < 1


def test_a_prompt_is_told_how_each_column_is_stored_and_no_value() -> None:
    data = pd.DataFrame(
        {
            "age": [61, 70],
            "sex": ["F", "M"],
            "smoker": [True, False],
            "visit": pd.to_datetime(["2024-01-02", "2024-02-03"]),
            "enrol_date": [
                "02/01/2024",
                "03/02/2024",
            ],  # a date kept as text, which formulas compute with
        }
    )
    kinds = column_kinds(
        data, ["age", "sex", "smoker", "visit", "enrol_date", "absent"], date_columns=["enrol_date"]
    )
    assert kinds == {
        "age": "number",
        "sex": "text",
        "smoker": "yes/no",
        "visit": "date",
        "enrol_date": "date",
    }
    assert "02/01/2024" not in str(kinds)  # no value


class _Answer:
    def __init__(self, content: str) -> None:
        self.content = content


class _Model:
    """A language model that answers what it is given, and keeps the prompts."""

    def __init__(self, *answers: str) -> None:
        self.answers, self.prompts = list(answers), []

    def invoke(self, prompt: str) -> _Answer:
        self.prompts.append(prompt)
        return _Answer(self.answers.pop(0))


def _engine(*answers: str):
    from manage.rag_computed_vars import ComputedVarsMixin

    class Engine(ComputedVarsMixin):
        initialized = True

        def __init__(self) -> None:
            self.llm = _Model(*answers)

        @staticmethod
        def _clean_json_response(text: str) -> str:
            return text

    return Engine()


def test_imputation_suggestions_come_in_the_applications_notation() -> None:
    answer = {
        "suggestions": [
            {
                "name": "Imputed_sbp_6m",
                "formula": "`sbp base` + np.where(nyha_class == 'III', 5.0, -2.0)",
            },
            {"name": "Imputed_sbp_6m", "formula": 'sbp_6m = ""sbp base"" * 1.02'},
            "not a suggestion",
        ]
    }
    engine = _engine(json.dumps(answer))
    suggestions, error = engine.suggest_imputation_formulas(
        "sbp_6m",
        ["sbp base", "nyha_class"],
        column_kinds={"sbp base": "number", "nyha_class": "text"},
    )
    assert error is None and len(suggestions) == 2
    assert suggestions[0]["formula"] == '""sbp base"" + np.where(nyha_class == \'III\', 5.0, -2.0)'
    assert suggestions[0]["original_formula"].startswith("`sbp base`")
    assert suggestions[1]["formula"] == '""sbp base"" * 1.02'
    assert "[sbp base (number), nyha_class (text)]" in engine.llm.prompts[0]


def test_a_computed_variable_suggestion_is_checked_as_the_evaluator_reads_it() -> None:
    answer = {
        "suggestions": [
            {"name": "bmi", "formula": "`Weight kg` / (height_m ^ 2)", "missing_variables": None},
            {"name": "broken", "formula": "weight_kg /"},
            {
                "name": "bsa",
                "formula": "np.sqrt(height_cm * weight_kg / 3600)",
                "missing_variables": "height_cm",
            },
        ]
    }
    # the model is then asked to align bsa's variables (it cannot: height_cm stays missing), and writes the notation
    engine = _engine("np.sqrt(height_cm * weight_kg / 3600)", '{"bmi": "BMI"}')
    engine.suggest_computed_variables = lambda *args: (json.dumps(answer), None)
    text, error = engine.suggest_computed_variables_with_validation(
        ["Weight kg", "height_m", "weight_kg"], allow_missing_variables=False
    )
    kept = json.loads(text)["suggestions"]
    assert error is None and [s["name"] for s in kept] == [
        "bmi"
    ]  # unreadable, or missing a column: left out
    assert kept[0]["formula"] == '""Weight kg"" / (height_m ** 2)'
    assert kept[0]["original_formula"] == "`Weight kg` / (height_m ^ 2)"
    assert kept[0]["missing_variables"] == []  # decided by the formula, never the model's own field


@pytest.mark.parametrize(
    ("answer", "formula"),
    [
        ('"weight / height"', "weight / height"),
        ("```python\nweight / height\n```", "weight / height"),
        ('""Weight kg"" / height', '""Weight kg"" / height'),  # the quotes of a column stay
        ("'weight' ", "weight"),
    ],
)
def test_a_corrected_formula_loses_only_its_wrapping(answer: str, formula: str) -> None:
    from manage.rag_computed_vars import _unwrapped

    assert _unwrapped(answer) == formula


def test_a_stopped_engine_sends_no_imputation_prompt() -> None:
    engine = _engine("{}")
    engine.initialized = False  # stopped: the confirmation was withdrawn, say
    suggestions, error = engine.suggest_imputation_formulas(
        "sbp_6m", ["sbp_baseline"], column_kinds={}
    )
    assert (
        suggestions is None and error == "RAG system not initialized." and engine.llm.prompts == []
    )
