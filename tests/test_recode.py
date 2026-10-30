"""
The values of a categorical variable recoded into a copy (enrich.recode, Data > Clean): what is pre-filled, what is
only offered, the copy beside the original, and the step's replay.
"""

from __future__ import annotations

import pandas as pd
import pytest

from enrich.recode import (
    CAPITALS,
    FIRST_LETTER,
    MISSING_CODE,
    SPACES,
    SPELLING,
    Offer,
    level_counts,
    recode_column,
    suggest,
)

# the tutorial cohort's sex, as written
SEX = {
    "M": 709,
    "F": 489,
    "male": 1,
    "Female": 1,
    "f": 1,
    "M ": 1,
    "F ": 1,
    "MALE": 1,
    "female": 1,
    "m": 1,
}


def _listed(values: pd.Series) -> list:
    """The values, a missing one as None (the recoded column holds pandas' missing value)."""
    return [None if pd.isna(v) else v for v in values]


def test_spaces_and_capitals_are_pre_filled_with_the_spelling_most_rows_hold() -> None:
    found = suggest(SEX)
    assert found.spellings == {
        "f": "F",
        "F ": "F",
        "m": "M",
        "M ": "M",
        "male": "MALE",
        "female": "Female",
    }
    assert (found.reasons["F "], found.reasons["f"]) == (SPACES, CAPITALS)


def test_a_one_letter_code_and_the_word_it_begins_are_offered_never_applied() -> None:
    """The rarer joins the more frequent: the 2 rows of Female and female could join the 491 of F."""
    assert suggest(SEX).offers == (
        Offer("Female", "F", FIRST_LETTER),
        Offer("MALE", "M", FIRST_LETTER),
    )


def test_a_rare_near_spelling_is_offered_never_across_a_number_or_between_frequent_values() -> None:
    sites = {"Luxembourg": 397, "Esch-sur-Alzette": 300, "Luxembourgg": 1}
    assert suggest(sites).offers == (Offer("Luxembourgg", "Luxembourg", SPELLING),)
    # one edit apart, two categories
    assert suggest({"HFrEF": 725, "HFmrEF": 321, "HFpEF": 160}).offers == ()
    assert suggest({"NYHA II": 400, "NYHA III": 3}).offers == ()
    assert suggest({"Type 2": 500, "Type 1": 2}).offers == ()


def test_the_codes_that_usually_mean_missing_are_offered_never_m_or_n() -> None:
    """ "M" may mean missing, male or married; "N" No or Never: never guessed."""
    counts = {"Yes": 300, "No": 200, "unknown": 5, "N": 3, "?": 2, "M": 2, "Non renseigné": 1}
    assert suggest(counts).offers == (
        Offer("unknown", None, MISSING_CODE),
        Offer("?", None, MISSING_CODE),
        Offer("Non renseigné", None, MISSING_CODE),
    )


def test_the_values_are_recoded_into_a_copy_beside_the_original() -> None:
    df = pd.DataFrame(
        {"id": [1, 2, 3, 4, 5], "sex": ["F", "f ", None, "unknown", "X"], "age": [5, 6, 7, 8, 9]}
    )
    out, unlisted = recode_column(df, "sex", "sex_clean", {"f ": "F", "unknown": None})
    assert list(out.columns) == ["id", "sex", "sex_clean", "age"]
    assert out["sex"].tolist() == df["sex"].tolist()  # the original is kept
    # X, which the mapping does not list, is kept as written
    assert _listed(out["sex_clean"]) == ["F", "F", None, None, "X"]
    assert unlisted == 2  # F and X


def test_a_copy_is_written_over_a_column_only_when_it_replaces_its_own() -> None:
    df = pd.DataFrame({"sex": ["F", "f"], "sex_clean": ["F", "F"]})
    with pytest.raises(ValueError, match="already a column"):
        recode_column(df, "sex", "sex_clean", {"f": "F"})
    with pytest.raises(ValueError, match="the original is kept"):
        recode_column(df, "sex", "sex", {"f": "F"})
    out, _ = recode_column(df, "sex", "sex_clean", {"f": "female", "F": "female"}, replaces=True)
    assert list(out.columns) == ["sex", "sex_clean"]
    assert out["sex_clean"].tolist() == ["female", "female"]


def test_the_values_as_written_are_listed_most_frequent_first() -> None:
    assert level_counts(pd.Series(["b", "a", "b", None, "a ", "a"])) == {"a": 2, "b": 2, "a ": 1}


def test_the_step_replays_and_says_when_the_column_holds_other_values() -> None:
    from manage.reproduction_manager import apply_steps

    params = {"source": "sex", "target": "sex_clean", "mapping": {"f": "F"}, "values_before": 3}
    step = {"function": "recode_levels", "params": params}
    replayed = apply_steps(pd.DataFrame({"sex": ["F", "f", "M"]}), [step])
    assert replayed.error is None and not replayed.warnings
    assert replayed.df is not None and replayed.df["sex_clean"].tolist() == ["F", "F", "M"]
    other = apply_steps(pd.DataFrame({"sex": ["F", "f", "M", "W"]}), [step])
    assert other.df is not None and other.df["sex_clean"].tolist() == ["F", "F", "M", "W"]
    assert any("4 distinct values, 3 when the step was recorded" in w for w in other.warnings)
