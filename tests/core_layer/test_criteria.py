"""Strict criterion evaluation: what fires, what never fires, and loud failures."""

import numpy as np
import pandas as pd
import pytest

from core.criteria import CriterionError, criterion_mask

DF = pd.DataFrame(
    {
        "age": [16, 30, 90, np.nan],
        "potassium": [2.0, 4.0, 9.8, np.nan],
        "consent": ["Yes", "No", "Yes", None],
        "text": ["a", "b", "c", "d"],
    }
)


def fires(definition: dict, name: str = "rule") -> list[bool]:
    return criterion_mask(DF, name, definition).tolist()


def test_expressions_fire_where_true_and_never_on_missing_values() -> None:
    assert fires({"expression": "df['age'] > 80"}) == [False, False, True, False]
    assert fires({"expression": "df['consent'] == 'Yes'"}) == [True, False, True, False]
    # missing values fire only when the expression asks for them
    assert fires({"expression": "df['age'].isna()"}) == [False, False, False, True]


def test_numeric_ranges_follow_the_strategy() -> None:
    assert fires(
        {"numeric": True, "lower_bound": 2.5, "upper_bound": 7, "strategy": "exclude"}, "potassium"
    ) == [
        True,
        False,
        True,
        False,
    ]
    assert fires(
        {"numeric": True, "lower_bound": 18, "upper_bound": 85, "strategy": "include"}, "age"
    ) == [
        False,
        True,
        False,
        False,
    ]


def test_a_single_bound_works_with_or_without_a_strategy() -> None:
    """The legacy helper returned an all-False mask for one bound with a strategy (comparison with None)."""
    assert fires({"numeric": True, "lower_bound": 18}, "age") == [False, True, True, False]
    assert fires({"numeric": True, "lower_bound": 18, "strategy": "include"}, "age") == [
        False,
        True,
        True,
        False,
    ]
    assert fires({"numeric": True, "upper_bound": 18, "strategy": "exclude"}, "age") == [
        False,
        True,
        True,
        False,
    ]


def test_frozen_bounds_and_listed_rows() -> None:
    assert fires({"bounds": {"potassium": [2.5, 7.0]}}) == [True, False, True, False]
    assert fires({"row_ids": [[1, 2]]}) == [False, True, True, False]


@pytest.mark.parametrize(
    ("definition", "name", "message"),
    [
        ({"expression": "df['agee'] > 1"}, "rule", "agee"),
        ({"expression": "__import__('os')"}, "rule", "rejected"),
        ({"expression": "df['age'] + 1"}, "rule", "True/False"),
        ({"numeric": True, "lower_bound": 1}, "missing_column", "missing_column"),
        ({"numeric": True, "lower_bound": 1}, "text", "not numeric"),
        ({"numeric": True}, "age", "neither"),
        ({"numeric": True, "lower_bound": 1, "strategy": "between"}, "age", "strategy"),
        ({}, "rule", "neither an expression"),
    ],
)
def test_problems_raise_instead_of_returning_a_silent_mask(
    definition: dict, name: str, message: str
) -> None:
    with pytest.raises(CriterionError, match=message):
        criterion_mask(DF, name, definition)
