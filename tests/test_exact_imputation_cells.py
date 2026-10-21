"""
An exact formula gives each row the value its own variables determine (biostatistics and architecture designs of the
derived variables, 2026-10-21): it fills only the rows where every variable of the formula has a value (``np.where``
answers a missing value with its "false" branch), and a value it computes from a single-imputed one is single-imputed,
never kept as observed.
"""

import numpy as np
import pandas as pd


def test_an_exact_formula_fills_only_the_rows_where_every_variable_has_a_value() -> None:
    from enrich.imputation_service import describe_formula_fill, fill_with_formula, left_missing

    frame = pd.DataFrame(
        {"sex": ["F", None, "M"], "a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0], "t": [np.nan] * 3}
    )
    formula = "np.where(sex == 'F', a, b)"
    _, every = fill_with_formula(frame, "t", formula)
    out, exact = fill_with_formula(frame, "t", formula, complete_inputs=True)
    # the missing sex is not taken for "not F"
    assert every == 3 and exact == 2 and pd.isna(out.loc[1, "t"])
    left = left_missing(frame, "t", formula)
    assert left == (1, ("sex",))
    assert describe_formula_fill("t", 2, exact=True, left=left) == (
        "Imputed 2 values in t using an exact formula; 1 left missing, where sex has no value"
    )


def test_a_formula_that_names_the_variable_it_fills_is_not_waiting_for_it() -> None:
    # biostatistics review: the target was taken for one of the formula's variables, and nothing was filled
    from enrich.imputation_service import fill_with_formula, formula_inputs

    frame = pd.DataFrame(
        {
            "weight": [np.nan, 70.0, np.nan],
            "bmi": [20.0, 24.2, np.nan],
            "height": [170.0, 170.0, 160.0],
        }
    )
    formula = "weight.fillna(bmi * (height / 100) ** 2)"
    assert formula_inputs(formula, "weight", frame) == ["bmi", "height"]
    out, filled = fill_with_formula(frame, "weight", formula, complete_inputs=True)
    assert filled == 1 and round(out.loc[0, "weight"], 2) == 57.8 and pd.isna(out.loc[2, "weight"])


def test_a_value_computed_from_a_single_imputed_one_is_single_imputed() -> None:
    from core.session import sort_filled

    before = pd.DataFrame(
        {
            "bmi": [20.0, 22.0, 25.0],
            "height": [170.0, 160.0, 180.0],
            "weight": [np.nan, np.nan, 81.0],
        }
    )
    after = before.assign(weight=[57.8, 56.3, 81.0])
    single = {"height": np.array([0], dtype=np.int64)}  # row 0's height was imputed
    params = {"target_variable": "weight", "exact": True, "inputs": ["bmi", "height"]}
    single_out, exact_out = sort_filled("targeted_imputation", params, before, after, single, {})
    assert single_out["weight"].tolist() == [0] and exact_out["weight"].tolist() == [1]
    # the variables not recorded (an earlier version): every filled cell is exact, as before
    _, legacy = sort_filled("targeted_imputation", {"exact": True}, before, after, single, {})
    assert legacy["weight"].tolist() == [0, 1]


def test_the_replay_sorts_the_cells_of_a_trace_recorded_without_the_variables() -> None:
    from manage.reproduction_manager import apply_steps

    frame = pd.DataFrame(
        {
            "bmi": [20.0, 22.0, 24.0, np.nan],
            "sex": ["F", "M", "F", None],
            "height": [np.nan, 160.0, 180.0, 175.0],
            "weight": [np.nan, np.nan, 77.8, np.nan],
        }
    )
    height = {"target_variable": "height", "formula": "170 + 0 * bmi", "exact": False}
    weight = {
        "target_variable": "weight",
        "formula": "np.where(sex == 'F', 1.0, 1.0) * np.where(bmi > 0, bmi, 21.0) * (height / 100) ** 2",
        "exact": True,
    }
    result = apply_steps(
        frame,
        [
            {"function": "targeted_imputation", "params": height},
            {"function": "targeted_imputation", "params": weight},
        ],
    )
    assert result.error is None
    # row 0 from the imputed height, row 3 where sex and bmi had no value ("false" branches): estimates
    assert result.single_imputed["weight"].tolist() == [0, 3]
    assert result.exact_imputed["weight"].tolist() == [1]
