"""
An exact formula gives each row its value from that row's own variables (biostatistics review, 2026-10-21): a formula of
other rows (an aggregate such as a mean) gives estimates, which "Exact formula" would keep as observed values, and its
values change with the rows it is computed on. enrich.formulas.row_wise tells them apart from the formula itself.
"""

import pandas as pd
import pytest

from core.expressions import safe_eval
from enrich.formulas import _OF_OTHER_ROWS_CALLED, OF_OTHER_ROWS, other_rows, row_wise

# Every function of the evaluator that gives each row a value of its own (the others are in OF_OTHER_ROWS): a function
# added to the evaluator fails this test until it is classified
OF_THE_ROW = frozenset(
    {
        # numpy
        "abs",
        "absolute",
        "add",
        "arccos",
        "arcsin",
        "arctan",
        "array",
        "asarray",
        "cbrt",
        "ceil",
        "clip",
        "cos",
        "deg2rad",
        "divide",
        "e",
        "exp",
        "exp2",
        "expm1",
        "floor",
        "fmod",
        "full",
        "hypot",
        "inf",
        "isfinite",
        "isinf",
        "isnan",
        "log",
        "log10",
        "log1p",
        "log2",
        "logical_and",
        "logical_not",
        "logical_or",
        "logical_xor",
        "maximum",
        "minimum",
        "mod",
        "multiply",
        "nan",
        "ones",
        "pi",
        "power",
        "rad2deg",
        "rint",
        "round",
        "select",
        "sign",
        "sin",
        "sqrt",
        "square",
        "subtract",
        "tan",
        "trunc",
        "true_divide",
        "where",
        "zeros",
        # pandas
        "DateOffset",
        "NA",
        "NaT",
        "Timedelta",
        "Timestamp",
        "cut",  # with its bins written out; a number of bins spreads them over the column's range
        "isna",
        "isnull",
        "notna",
        "notnull",
        "to_datetime",
        "to_numeric",
        "to_timedelta",
        # Series methods
        "astype",
        "between",
        "div",
        "dt",
        "eq",
        "fillna",
        "floordiv",
        "ge",
        "gt",
        "isin",
        "le",
        "lt",
        "mask",
        "mul",
        "ne",
        "pow",
        "replace",
        "str",
        "sub",
        "truediv",
    }
)


def test_every_function_of_the_evaluator_is_classified() -> None:
    # the text and date accessors' methods are each row's own
    offered = safe_eval._NP_FUNCTIONS | safe_eval._PD_FUNCTIONS | safe_eval._SERIES_METHODS
    assert offered - OF_OTHER_ROWS - OF_THE_ROW == set()
    assert OF_OTHER_ROWS.isdisjoint(OF_THE_ROW)
    # called by their name: len(x), max(x) and min(x) are of the whole column
    called = set(safe_eval._BUILTINS) - _OF_OTHER_ROWS_CALLED
    assert called == {"abs", "bool", "float", "int", "round", "str"}


@pytest.mark.parametrize(
    "formula",
    [
        "bmi * (height_cm / 100) ** 2",  # a weight from the BMI and the height
        "np.where(sex == 'F', 0.7, 0.9) * weight_kg",
        "120",
        "visit_date - enrol_date",  # days
        "np.maximum(weight_kg, 60)",  # element by element, not np.max
        'pd.cut(age, [0, 40, 65, 120], labels=["young", "middle", "old"])',
        "first_name.str[0] + last_name.str.upper()",
        "df['weight_kg'] / 2",
        'weight_kg.fillna(0) + ""a column""',
        "np.where(sex.isin(['F', 'f']), 0.7, 0.9)",  # a list written out
        "np.nanmean([sbp_1, sbp_2, sbp_3], axis=0)",  # across the columns, row by row
    ],
)
def test_a_formula_of_each_rows_own_values_is_row_wise(formula: str) -> None:
    assert row_wise(formula)


@pytest.mark.parametrize(
    "formula",
    [
        "bmi.mean() + 0 * weight_kg",
        "weight_kg / weight_kg.max()",
        "weight_kg.rank()",
        "bmi.fillna(bmi.median())",
        "weight_kg.shift(1)",
        "np.where(weight_kg > weight_kg.quantile(0.5), 1, 0)",
        "np.nanmean(bmi) + 0 * bmi",
        "weight_kg / len(weight_kg)",
        "pd.cut(age, 4)",  # four bins over the range of age
        "pd.qcut(age, 4)",
        "weight_kg - weight_kg[0]",  # the first row's value
        "weight_kg.cumsum()",
        "np.where(code.isin(other_code), 1, 0)",  # the codes of every row
        "np.nanmean([sbp_1, sbp_2, sbp_3])",  # one mean of every value
    ],
)
def test_a_formula_of_other_rows_is_not(formula: str) -> None:
    assert not row_wise(formula)


def test_the_refusal_names_what_uses_other_rows() -> None:
    assert other_rows("bmi.fillna(bmi.median())") == "median()"
    assert other_rows("weight_kg - weight_kg[0]") == "the value of another row"
    assert other_rows("pd.cut(age, 4)") == "cut() with a number of bins"
    assert other_rows("weight_kg +") is not None  # a formula that cannot be read


def test_dates_kept_as_text_are_read_from_the_rows() -> None:
    # biostatistics review: the order of day and month of a date kept as text is read from the rows computed on
    text = pd.DataFrame(
        {"visit": ["03/02/2024", "25/12/2024"], "enrol": ["01/01/2024", "01/02/2024"]}
    )
    reason = other_rows("visit - enrol", text)
    assert reason is not None and reason.startswith("the dates of visit are kept as text")
    dates = text.apply(pd.to_datetime, dayfirst=True)
    assert other_rows("visit - enrol", dates) is None
    assert other_rows("np.where(sex == 'F', 1, 0)", pd.DataFrame({"sex": ["F", "M"]})) is None
