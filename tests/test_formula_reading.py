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
        ("sqrt(height * weight / 3600)", ("height", "weight")),
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
        ("bmi = weight / height ** 2", "weight / height ** 2"),  # the formula is the right side
        ('""a^b"" ^ 2', '""a^b"" ** 2'),  # a quoted column keeps its name
        ("a == b", "a == b"),  # a comparison is not an assignment
    ],
)
def test_a_models_notation_becomes_the_applications(written: str, converted: str) -> None:
    assert formula_from_model(written) == converted


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


def test_a_suggestion_is_tried_on_the_data_and_says_why_it_cannot_be_computed() -> None:
    data = pd.DataFrame(
        {
            "sbp_baseline": [120.0, 135.0, None],
            "nyha_class": ["II", "III", "III"],
            "sbp_6m": [None, 130.0, None],
        }
    )
    refused = try_formula("sbp_baseline + np.where(nyha_class > 2, 5.0, -2.0)", data)
    assert not refused.ok and refused.values is None
    assert "nyha_class holds text (such as 'III', 'II')" in refused.problem
    fixed = try_formula("sbp_baseline + np.where(nyha_class == 'III', 5.0, -2.0)", data)
    assert fixed.ok and fixed.values.tolist()[:2] == [118.0, 140.0]
    assert try_formula("weight / height ^ 2", data).problem == CARET_IS_NOT_A_POWER
    assert try_formula("sbp_baseline * bsa", data).problem == "not in the data: bsa"
    assert try_formula("sbp_baseline +", data).problem.startswith("Syntax Error")


def test_a_prompt_is_told_how_each_column_is_stored_and_no_value() -> None:
    data = pd.DataFrame(
        {
            "age": [61, 70],
            "sex": ["F", "M"],
            "smoker": [True, False],
            "visit": pd.to_datetime(["2024-01-02", "2024-02-03"]),
        }
    )
    assert column_kinds(data, ["age", "sex", "smoker", "visit", "absent"]) == {
        "age": "number",
        "sex": "text",
        "smoker": "yes/no",
        "visit": "date",
    }


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
                "formula": "sqrt(height_cm * weight_kg / 3600)",
                "missing_variables": "height_cm",
            },
        ]
    }
    # the model is then asked to align bsa's variables (it cannot: height_cm stays missing), and writes the notation
    engine = _engine("sqrt(height_cm * weight_kg / 3600)", '{"bmi": "BMI"}')
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
