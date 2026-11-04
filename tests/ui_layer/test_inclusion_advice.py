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


def test_a_criterion_of_two_columns_is_kept_as_an_exclusion_of_its_negation() -> None:
    """The advice wrote "isna() | (...)" over each column, which kept the rows a known value excludes: a minor whose
    consent is missing, and a refusal whose age is missing. Added as an exclusion criterion, the negation fires only
    where the criterion is known to fail, so it keeps exactly the rows that meet it and those where it is unknown."""
    from core.criteria import criterion_mask, criterion_unknown

    frame = pd.DataFrame(
        {
            "age": [30, 12, None, 40, None, 15],
            "consent": ["Yes", "Yes", "Yes", None, "No", None],
        }
    )
    definition = {"expression": "(df['age'] >= 18) & (df['consent'] == 'Yes')"}
    words, kept = unknown_advice(frame, "Adult who consented", definition)
    said = "it excludes them. To keep them, add it as an exclusion criterion instead, written this way:"
    assert said in words
    assert kept == "~((df['age'] >= 18) & (df['consent'] == 'Yes'))"
    meets = criterion_mask(frame, "c", definition)
    unknown = criterion_unknown(frame, "c", definition)
    assert meets.tolist() == [True, False, False, False, False, False]
    # a minor, or a refusal, is known to fail whatever the missing value
    assert unknown.tolist() == [False, False, True, True, False, False]
    # an exclusion criterion keeps the rows where it does not fire
    kept_rows = ~criterion_mask(frame, "x", {"expression": kept})
    assert kept_rows.tolist() == (meets | unknown).tolist()


def test_a_few_rows_read_under_1_percent_never_0() -> None:
    """2 rows of 1,206 read "(0%)" beside "Unknown on 2 rows"."""
    frame = pd.DataFrame({"pregnant": [None, None] + ["No"] * 1204})
    words, _ = unknown_advice(frame, "Not pregnant", {"expression": "df['pregnant'] != 'Yes'"})
    assert words.startswith("Unknown on 2 rows (under 1%), where a value it reads is missing")
