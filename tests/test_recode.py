"""
The values of a categorical variable recoded into a copy (enrich.recode, Data > Clean): the fold rule, what is only
offered, the user's choices over the fold, the copy beside the original, what the step records, and its replay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from enrich.recode import (
    CAPITALS,
    CHOSEN,
    FIRST_LETTER,
    FOLD,
    MISSING_CODE,
    NOT_APPLICABLE,
    SPACES,
    SPACES_AND_CAPITALS,
    SPELLING,
    UNKNOWN,
    Offer,
    candidates,
    changes,
    fold,
    groupings,
    last_record,
    level_counts,
    plan,
    recode,
    recode_column,
    recoded_columns,
    report_params,
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


def _offers(counts: dict[str, int]) -> dict[str, tuple[str | None, str, str | None]]:
    return {o.value: (o.clean, o.why, o.meaning) for o in suggest(counts).offers}


def test_the_fold_takes_the_spelling_most_rows_hold() -> None:
    found = suggest(SEX)
    assert found.spellings == {
        "f": "F",
        "F ": "F",
        "m": "M",
        "M ": "M",
        "MALE": "male",
        "female": "Female",
    }
    assert (found.reasons["F "], found.reasons["f"]) == (SPACES, CAPITALS)
    assert fold({"Luxembourg": 300, " LUXEMBOURG ": 2}) == {" LUXEMBOURG ": "Luxembourg"}
    assert suggest({"Luxembourg": 300, " LUXEMBOURG ": 2}).reasons == {
        " LUXEMBOURG ": SPACES_AND_CAPITALS
    }


def test_capitals_fold_in_words_and_on_a_tie_the_capitalised_or_lower_case_spelling() -> None:
    """In a word of 4 letters or more, capitals carry no meaning: folded whatever their share."""
    assert fold({"Luxembourg": 300, "luxembourg": 200}) == {"luxembourg": "Luxembourg"}
    assert fold({"male": 2, "MALE": 2}) == {"MALE": "male"}
    assert fold({"male": 2, "Male": 2}) == {"male": "Male"}


def test_in_a_code_capitals_fold_only_when_rare_and_never_where_case_is_written_mixed() -> None:
    """Genotypes, Rh haplotypes and pH: case can carry meaning in a code of 3 characters or fewer."""
    assert fold({"M": 709, "m": 1}) == {"m": "M"}
    assert fold({"AA": 300, "Aa": 400, "aa": 100}) == {}
    assert fold({"AA": 900, "Aa": 95, "aa": 5}) == {}  # a rare homozygote is no misspelling
    assert fold({"Cde": 40, "cde": 300}) == {}
    assert fold({"pH": 300, "PH": 2}) == {}
    assert fold({"NR": 300, "nr": 40}) == {}  # 12 %: offered
    assert _offers({"NR": 300, "nr": 40})["nr"][:2] == (None, MISSING_CODE)
    assert _offers({"AA": 300, "Aa": 400, "aa": 100}) == {
        "AA": ("Aa", CAPITALS, None),
        "aa": ("Aa", CAPITALS, None),
    }


def test_a_one_letter_code_and_the_only_value_it_begins_are_offered_never_applied() -> None:
    """The rarer joins the more frequent: the 2 rows of Female and female could join the 491 of F."""
    assert suggest(SEX).offers == (
        Offer("Female", "F", FIRST_LETTER),
        Offer("male", "M", FIRST_LETTER),
    )
    assert _offers({"Yes": 300, "Y": 4, "No": 200, "N": 3}) == {
        "Y": ("Yes", FIRST_LETTER, None)
    }  # No: 2 letters
    assert _offers({"A": 300, "AB": 40, "B": 200, "O": 400}) == {}  # blood groups
    assert (
        _offers({"II": 526, "III": 396, "I": 186, "IV": 98}) == {}
    )  # NYHA I begins II, III and IV
    assert _offers({"III": 396, "I": 186}) == {}  # a Roman numeral is a category
    assert _offers({"Type 2": 300, "T": 4}) == {}  # so is a value holding a digit


@pytest.mark.parametrize(
    ("common", "rare"),
    [
        ("Normal", "Abnormal"),
        ("Stable angina", "Unstable angina"),
        ("STEMI", "NSTEMI"),
        ("Hypertension", "Hypotension"),
        ("Hyperkalaemia", "Hypokalaemia"),
        ("Smoker", "Non-smoker"),
        ("Stage IIIA", "Stage IIIB"),
        ("Stage 3a", "Stage 3b"),
        ("CKD 3a", "CKD 3b"),
        ("S72.001A", "S72.001D"),
        ("HFrEF", "HFmrEF"),
        ("diabetes type II", "diabetes type I"),
        ("NYHA III", "NYHA II"),
        ("Type 2", "Type 1"),
        ("HER2-negative", "HER2+"),
    ],
)
def test_a_near_spelling_is_never_offered_between_two_categories(common: str, rare: str) -> None:
    assert _offers({common: 900, rare: 3}) == {}


def test_a_rare_near_spelling_of_a_frequent_value_is_offered() -> None:
    assert suggest({"Luxembourg": 397, "Esch-sur-Alzette": 300, "Luxembourgg": 1}).offers == (
        Offer("Luxembourgg", "Luxembourg", SPELLING),
    )
    assert _offers({"Hypertension": 900, "Hypertensoin": 2}) == {
        "Hypertensoin": ("Hypertension", SPELLING, None)
    }
    # a hyphen between two letters joins words, as a space does
    assert (
        _offers({"Esch-sur-Alzette": 200, "Esch sur Alzete": 1})["Esch sur Alzete"][0]
        == "Esch-sur-Alzette"
    )
    # rare: at most 5 rows, and at most 1 % of the value it could join
    assert _offers({"Luxembourg": 397, "Luxembourgg": 10}) == {}
    assert _offers({"Luxembourg": 300, "Luxembourgg": 4}) == {}
    # a first difference within a word's first four letters: another word
    assert _offers({"Differdange": 300, "Diferdange": 2}) == {}
    # near spellings are looked for only when asked (the list of variables does not)
    assert suggest({"Luxembourg": 397, "Luxembourgg": 1}, near=False).offers == ()


def test_codes_for_unknown_are_offered_as_unknown_and_the_others_ask_what_they_mean() -> None:
    """ "n.a." may mean not applicable, "none" an absence, "NR" non-reactive, "ND" not detected: the user says.
    "M" (missing, male, married) and "N" (No, Never) are never guessed."""
    counts = {
        "Yes": 300,
        "No": 200,
        "unknown": 5,
        "N": 3,
        "?": 2,
        "M": 2,
        "Non renseigné": 1,
        "Ne sait pas": 1,
    }
    assert _offers(counts) == {
        "unknown": (None, MISSING_CODE, UNKNOWN),
        "?": (None, MISSING_CODE, UNKNOWN),
        "Ne sait pas": (None, MISSING_CODE, UNKNOWN),
        "Non renseigné": (None, MISSING_CODE, UNKNOWN),
    }
    for code in ("n.a.", "NR", "none", "ND", "-", "N/A"):
        assert _offers({"pre": 300, "post": 200, code: 40})[code] == (None, MISSING_CODE, None), (
            code
        )


def test_a_choice_for_a_spelling_applies_to_its_group_and_a_value_s_own_choice_wins() -> None:
    counts = {"F": 489, "F ": 1, "f": 1, "Female": 1, "unknown": 1}
    assert plan(counts, {}) == {"F ": "F", "f": "F"}
    assert plan(counts, {"Female": "F", "unknown": None}) == {
        "F ": "F",
        "f": "F",
        "Female": "F",
        "unknown": None,
    }
    assert plan(counts, {"F": "Female"}) == {"F": "Female", "F ": "Female", "f": "Female"}
    assert plan(counts, {"F": "Female", "f": "F"}) == {"F": "Female", "F ": "Female", "f": "F"}
    assert plan(counts, {}, kept=["f"]) == {"F ": "F"}  # a fold put back as written
    assert (
        plan(counts, {"f ": "F"}, fold_rule=None) == {}
    )  # no fold: the mapping lists every change
    with pytest.raises(ValueError, match="unknown fold rule"):
        plan(counts, {}, fold_rule="other")


def test_the_values_are_recoded_into_a_text_copy_beside_the_original() -> None:
    df = pd.DataFrame(
        {"id": [1, 2, 3, 4, 5], "sex": ["F", "f ", None, "unknown", "X"], "age": [5, 6, 7, 8, 9]}
    )
    out, kept = recode_column(df, "sex", "sex_clean", {"unknown": None})
    assert list(out.columns) == ["id", "sex", "sex_clean", "age"]
    assert out["sex"].tolist() == df["sex"].tolist()  # the original is kept
    # spaces folded; f holds half its group, F and f, so its capital is not (a code: offered)
    assert _listed(out["sex_clean"]) == ["F", "f", None, None, "X"]
    assert kept == 2  # F and X, kept as written


def test_numbers_and_their_text_are_one_value_in_the_copy() -> None:
    """An .xlsx column holding 2 and "2 ": the page merges them, so the copy must too."""
    values = pd.Series([2, 2, 3, "2 ", "III", 1, "3 ", np.nan], dtype=object)
    copy = recode(values, plan(level_counts(values), {}))
    assert _listed(copy) == ["2", "2", "3", "2", "III", "1", "3", None]
    assert copy.dropna().nunique() == 4 and pd.api.types.is_string_dtype(copy)
    codes = pd.Series(pd.Categorical([1, 2, 2, None]))
    assert _listed(recode(codes, {"2": "two"})) == ["1", "two", "two", None]


def test_a_copy_is_written_over_a_column_only_when_it_replaces_its_own() -> None:
    df = pd.DataFrame({"sex": ["F", "f"], "sex_clean": ["F", "F"]})
    with pytest.raises(ValueError, match="already a column"):
        recode_column(df, "sex", "sex_clean", {})
    with pytest.raises(ValueError, match="the original is kept"):
        recode_column(df, "sex", "sex", {})
    out, _ = recode_column(df, "sex", "sex_clean", {"F": "female", "f": "female"}, replaces=True)
    assert list(out.columns) == ["sex", "sex_clean"]
    assert out["sex_clean"].tolist() == ["female", "female"]


def test_the_values_as_written_are_listed_most_frequent_first() -> None:
    assert level_counts(pd.Series(["b", "a", "b", None, "a ", "a"])) == {"a": 2, "b": 2, "a ": 1}


def test_merging_two_values_that_each_hold_5_percent_is_a_grouping() -> None:
    nyha = {"II": 526, "III": 396, "I": 186, "IV": 98}
    assert groupings(nyha, {"III": "III-IV", "IV": "III-IV"}) == {"III-IV": ["III", "IV"]}
    assert (
        groupings(SEX, plan(SEX, {"Female": "F", "male": "M"})) == {}
    )  # 2 rows: a recode, not a grouping


def test_the_changes_are_counted_by_kind_and_meaning_never_named() -> None:
    counts = {"F": 489, "F ": 1, "f": 1, "Female": 1, "n.a.": 3, "?": 2}
    mapping = {"Female": "F", "n.a.": None, "?": None}
    kinds = {"Female": FIRST_LETTER, "n.a.": MISSING_CODE, "?": MISSING_CODE}
    meanings = {"n.a.": NOT_APPLICABLE, "?": UNKNOWN}
    assert changes(counts, mapping, kinds, meanings=meanings) == {
        "by_kind": {SPACES: [1, 1], CAPITALS: [1, 1], FIRST_LETTER: [1, 1], MISSING_CODE: [2, 5]},
        "set_to_missing": {NOT_APPLICABLE: [1, 3], UNKNOWN: [1, 2]},
    }
    assert changes(counts, {"f": "female"}, {})["by_kind"] == {SPACES: [1, 1], CHOSEN: [1, 1]}


def test_the_trace_report_counts_the_choices_and_names_no_value() -> None:
    params = {
        "source": "sex",
        "target": "sex_clean",
        "fold": FOLD,
        "mapping": {"CANARY-1": "F", "CANARY-2": None},
        "kinds": {"CANARY-1": FIRST_LETTER, "CANARY-2": MISSING_CODE},
        "kept": ["CANARY-3"],
        "missing_meaning": {"CANARY-2": UNKNOWN},
        "groupings": {"CANARY-4": "CANARY-5"},
    }
    shown = report_params(params)
    assert "CANARY" not in repr(shown)
    assert shown["choices"] == {FIRST_LETTER: 1, MISSING_CODE: 1}
    assert (shown["kept as written"], shown["groupings"]) == (1, 1)
    # a step recorded before the fold rule listed every change, and why
    older = report_params({"mapping": {"CANARY-1 ": "F"}, "reasons": {"CANARY-1 ": SPACES}})
    assert "CANARY" not in repr(older) and older["choices"] == {SPACES: 1}


def test_the_variables_listed_leave_out_the_copies_and_say_which_were_cleaned() -> None:
    df = pd.DataFrame(
        {
            "site": ["Metz", "Metz ", "Metz", "Luxembourg"],
            "sex": ["F", "f", "M", "M"],
            "sex_clean": ["F", "F", "M", "M"],
            "arm": ["A", "B", "A", "B"],
            "age": [1, 2, 3, 4],
        }
    )
    found = candidates(df, {"sex": "sex_clean"})
    assert [(c.column, c.values, c.to_check, c.cleaned_into) for c in found] == [
        ("site", 3, 1, ""),
        ("sex", 3, 1, "sex_clean"),
        ("arm", 2, 0, ""),
    ]
    assert [c.column for c in candidates(df.drop(columns=["sex_clean"]), {"sex": "sex_clean"})][
        :2
    ] == ["site", "sex"]


def test_the_trace_names_each_column_s_last_copy_and_its_choices() -> None:
    steps = [
        {
            "function": "recode_levels",
            "params": {"source": "sex", "target": "sex_clean", "mapping": {"f": "F"}},
        },
        {"function": "rename_columns", "params": {}},
        {
            "function": "recode_levels",
            "params": {"source": "sex", "target": "sex_clean", "mapping": {"x": "F"}},
        },
    ]
    assert recoded_columns(steps) == {"sex": "sex_clean"}
    assert (last_record(steps, "sex", "sex_clean") or {})["mapping"] == {"x": "F"}
    assert last_record(steps, "sex", "other") is None


def test_the_step_replays_the_fold_and_says_when_the_column_holds_values_it_did_not_see() -> None:
    from manage.reproduction_manager import apply_steps

    params = {
        "source": "sex",
        "target": "sex_clean",
        "fold": FOLD,
        "mapping": {"Female": "F"},
        "kept": [],
        "values_after": 2,
        "kept_as_written": 2,
    }
    step = {"function": "recode_levels", "params": params}
    sex = ["F"] * 30 + ["f", "Female"] + ["M"] * 30  # f: 1 of 31, folded
    replayed = apply_steps(pd.DataFrame({"sex": sex}), [step])
    assert replayed.error is None and not replayed.warnings
    assert replayed.df is not None and set(replayed.df["sex_clean"]) == {"F", "M"}
    other = apply_steps(pd.DataFrame({"sex": [*sex, "W"]}), [step])
    assert other.df is not None and other.df["sex_clean"].tolist()[-1] == "W"
    assert any("1 more value(s) of sex kept as written" in w for w in other.warnings)
    assert any(
        "sex_clean holds 3 distinct values, 2 when it was recorded" in w for w in other.warnings
    )
    assert all("W" not in w.split("sex")[0] for w in other.warnings)


def test_a_step_without_a_fold_rule_replays_its_mapping_alone() -> None:
    from manage.reproduction_manager import apply_steps

    step = {
        "function": "recode_levels",
        "params": {"source": "sex", "target": "sex_clean", "mapping": {"f": "F"}},
    }
    replayed = apply_steps(pd.DataFrame({"sex": ["F", "f", "M "]}), [step])
    assert replayed.df is not None and replayed.df["sex_clean"].tolist() == ["F", "F", "M "]


def test_the_copy_keeps_the_marks_of_the_imputed_values_it_copies() -> None:
    """An imputed value recoded is still imputed; a value set to missing is no value; the copy's old marks go."""
    from core.session import sort_filled

    before = pd.DataFrame({"sex": ["F", "M", "unknown", "F"], "sex_clean": ["x", "x", "x", "x"]})
    after, _ = recode_column(before, "sex", "sex_clean", {"unknown": None}, replaces=True)
    single = {"sex": np.array([1, 2]), "sex_clean": np.array([3])}
    exact = {"sex": np.array([0])}
    params = {"source": "sex", "target": "sex_clean"}
    new_single, new_exact = sort_filled("recode_levels", params, before, after, single, exact)
    assert new_single["sex"].tolist() == [1, 2] and new_single["sex_clean"].tolist() == [1]
    assert new_exact["sex_clean"].tolist() == [0]
