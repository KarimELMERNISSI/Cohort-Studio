"""
A criterion's expression in three-valued logic (core.expressions.kleene, RULES_VERSION 2): a missing value never fires
a rule, written with "!=" or "~" too; a test inside a call is read as Python reads it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.criteria import CriterionError, criterion_mask

DF = pd.DataFrame(
    {
        "h": [50.0, np.nan, 150.0, 120.0],
        "s": ["x", None, "y", "withdrawn"],
        "c": pd.Categorical(["a", None, "b", "a"]),
        "d1": ["2021-01-01", None, "2021-03-01", "2021-05-01"],
        "d2": ["2021-02-01", "2021-02-01", None, "2021-04-01"],
        "flag": pd.array([True, None, False, True], dtype="boolean"),
    }
)


def _fires(expression: str) -> list[int]:
    mask = criterion_mask(DF, "rule", {"expression": expression})
    return [int(i) for i in DF.index[mask]]


@pytest.mark.parametrize(
    ("expression", "rows"),
    [
        ("~(df['h'] >= 100)", [0]),  # fired on the missing height too
        ("df['s'] != 'withdrawn'", [0, 2]),  # and on the missing status
        ("~df['s'].isin(['x'])", [2, 3]),
        ("~df['h'].between(100, 200)", [0]),
        ("df['c'] != 'a'", [2]),
        ("~df['s'].str.startswith('w')", [0, 2]),
        ("pd.to_datetime(df['d1']) > pd.to_datetime(df['d2'])", [3]),
        ("~df['flag']", [2]),
        ("(df['h'] > 100) & (df['s'] != 'y')", [3]),
    ],
)
def test_a_missing_value_never_fires_a_rule_written_with_not_equal_or_not(
    expression: str, rows: list[int]
) -> None:
    assert _fires(expression) == rows


def test_unknown_or_true_is_true_and_a_rule_may_ask_for_missing_values() -> None:
    assert _fires("df['h'].isna() | (df['h'] > 130)") == [1, 2]
    assert _fires("df['h'].isna()") == [1]


def test_a_test_inside_a_call_is_read_as_python_reads_it() -> None:
    """np.where, fillna and astype decide what a missing value gives there."""
    assert _fires("np.where(df['h'] > 100, 1, 0) == 1") == [2, 3]
    assert _fires("df['h'].fillna(0) < 100") == [0, 1]
    assert _fires("(df['h'] > 100).astype(int) == 0") == [0, 1]


def test_an_expression_is_still_checked_as_written() -> None:
    with pytest.raises(CriterionError, match="expression rejected"):
        criterion_mask(DF, "rule", {"expression": "known_compare(df['h'], '>', 1)"})
