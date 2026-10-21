"""
The AI engine's proposals of derived variables (manage.rag_computed_vars.suggest_derived_variables): one request for
the whole dataset, carrying the column names, how each is stored, the variable map's units and each number column's
typical values (quantiles: never an extreme an outlier would move, never a row); each proposal read here into a
definition (core.derived), and none added by itself.
"""

import json

import numpy as np
import pandas as pd


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


COLUMNS = ["weight_kg", "height_cm", "bmi", "egfr", "creatinine", "age"]


def test_the_proposals_are_read_into_definitions() -> None:
    answer = {
        "candidates": [
            {
                "target": "bmi",
                "formula": "weight_kg / (height_cm / 100) ^ 2",
                "kind": "Ratio",
                "inverses": {
                    "weight_kg": "bmi * (height_cm / 100) ** 2",
                    "waist_cm": "bmi * 2",  # not a variable of the formula
                    "height_cm": "weight_kg * 2",  # does not use the BMI
                },
                "units": "kg and cm",
                "source": "WHO",
                "confidence": "high",  # a key not asked for
            },
            {
                "target": "bmi",
                "formula": "weight_kg / (height_cm / 100) ** 2",
            },  # the same, once read
            {"target": "waist_cm", "formula": "weight_kg * 2"},  # not a column
            {"target": "bmi", "formula": "bmi * 1"},  # its own variable
            {"target": "egfr", "formula": "`creatinine` / 88.4 + age", "kind": "magic"},
            {"target": "age", "formula": "weight_kg +"},  # cannot be read
            "not a candidate",
        ]
    }
    engine = _engine(json.dumps(answer))
    found, error = engine.suggest_derived_variables(COLUMNS)
    assert error is None
    assert [d.target for d in found] == ["bmi", "egfr"]
    bmi = found[0]
    assert (
        bmi.formula == "weight_kg / (height_cm / 100) ** 2"
    )  # ^ read as the power the model meant
    assert bmi.inverses == {"weight_kg": "bmi * (height_cm / 100) ** 2"}
    assert (bmi.origin, bmi.kind, bmi.note) == ("ai", "ratio", "WHO; kg and cm")
    egfr = found[1]
    assert egfr.formula == '""creatinine"" / 88.4 + age' and egfr.kind == ""
    assert not egfr.accepted  # a proposal is never accepted by itself


def test_what_is_sent_is_names_kinds_units_and_quantiles() -> None:
    from manage.ai_payload import column_kinds, typical_ranges

    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            # an outlier: never sent, and it moves no quantile
            "creatinine": np.r_[np.round(rng.normal(90, 15, 199), 0), 2500.0],
            "sex": ["F", "M"] * 99 + ["F", "Xq-only-once"],
        }
    )
    ranges = typical_ranges(frame)
    assert set(ranges) == {"creatinine"} and ranges["creatinine"].startswith("6")
    engine = _engine(json.dumps({"candidates": []}))
    found, error = engine.suggest_derived_variables(
        list(frame.columns),
        column_kinds=column_kinds(frame, frame.columns),
        units={"creatinine": "µmol/L"},
        ranges=ranges,
    )
    assert (found, error) == ([], None)
    prompt = engine.llm.prompts[0]
    assert f"- creatinine (number; unit µmol/L; typical values {ranges['creatinine']})" in prompt
    assert "- sex (text)" in prompt
    assert "2500" not in prompt and "Xq-only-once" not in prompt


def test_an_error_of_the_provider_is_said_not_taken_for_an_answer() -> None:
    engine = _engine("API Error (Gemini): quota exceeded")
    found, error = engine.suggest_derived_variables(COLUMNS)
    assert found == [] and error is not None and error.startswith("API Error")


def test_nothing_is_asked_of_an_engine_that_is_not_running() -> None:
    engine = _engine()
    engine.initialized = False
    assert engine.suggest_derived_variables(COLUMNS) == ([], "RAG system not initialized.")
    assert engine.llm.prompts == []


def test_a_small_column_sends_no_range_and_an_outlier_never_enters_one() -> None:
    # data-protection review: under 21 values, the 5th and 95th percentiles are interpolated with the extremes
    from manage.ai_payload import column_profile, typical_ranges

    small = pd.DataFrame({"x": [60.0 + i for i in range(19)] + [2500.0]})
    assert typical_ranges(small) == {} and "stats" not in column_profile(small)[0]
    enough = pd.DataFrame({"x": [60.0 + i for i in range(20)] + [2500.0]})
    assert typical_ranges(enough) == {"x": "61 to 79 (median 70)"}


def test_a_date_kept_as_a_number_sends_no_range() -> None:
    from manage.ai_payload import typical_ranges

    births = pd.DataFrame({"dob": [19650315 + i for i in range(40)], "age": range(40, 80)})
    assert set(typical_ranges(births, date_columns=["dob"])) == {"age"}


def test_a_unit_on_several_lines_stays_on_its_column_s_line() -> None:
    engine = _engine(json.dumps({"candidates": []}))
    engine.suggest_derived_variables(
        ["creatinine"], units={"creatinine": "µmol/L\n- planted (text)"}
    )
    prompt = engine.llm.prompts[0]
    assert "- creatinine (unit µmol/L - planted (text))" in prompt and "\n- planted" not in prompt
