"""
An inclusion criterion excludes the rows where it is unknown for want of a value (a missing value meets no condition,
written with != or ~ too): the criterion editor says so before it is added, with the expression that keeps them.
"""

from __future__ import annotations

import pandas as pd

from app_pages.data_monitoring import unknown_advice

FRAME = pd.DataFrame(
    {"pregnant": [None, None, None, "Yes", "No", "No"], "age": [30, 40, 50, 30, 25, None]}
)


def test_a_negation_on_a_variable_recorded_for_a_subgroup_is_said_with_the_way_to_keep_them() -> (
    None
):
    words, kept = unknown_advice(FRAME, "Not pregnant", {"expression": "df['pregnant'] != 'Yes'"})
    assert words.startswith("Unknown on 3 rows (50%), where a value it reads is missing (pregnant)")
    assert "it excludes them. To keep them" in words
    assert kept == "df['pregnant'].isna() | (df['pregnant'] != 'Yes')"
    # the expression it suggests keeps them
    from core.criteria import criterion_mask

    assert criterion_mask(FRAME, "kept", {"expression": kept}).tolist() == [
        True,
        True,
        True,
        False,
        True,
        True,
    ]


def test_a_numeric_range_is_said_without_an_expression_and_a_known_criterion_says_nothing() -> None:
    words, kept = unknown_advice(
        FRAME, "age", {"numeric": True, "lower_bound": 18, "upper_bound": 85}
    )
    assert (
        words.startswith("Unknown on 1 row (17%), where a value it reads is missing (age)")
        and kept is None
    )
    assert "it excludes it." in words
    assert (
        unknown_advice(
            FRAME, "Known", {"expression": "df['pregnant'].isna() | (df['pregnant'] != 'Yes')"}
        )
        is None
    )
