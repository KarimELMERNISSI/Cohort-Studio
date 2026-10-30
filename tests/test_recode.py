"""
The values of a categorical variable recoded into a copy (enrich.recode, Data > Clean): the fold rule, what is only
offered, the user's choices over the fold, what the step records (always the preview), the copy beside the original,
and the replay.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from enrich.recode import (
    CAPITALS,
    CHOSEN,
    CLOSE,
    DECLINED,
    FIRST_LETTER,
    FOLD,
    MEANINGS,
    MISSING_CODE,
    NOT_APPLICABLE,
    SPACES,
    SPACES_AND_CAPITALS,
    SPELLING,
    UNKNOWN,
    VARIANT,
    Offer,
    candidates,
    changes,
    fold,
    give,
    groupings,
    last_record,
    level_counts,
    new_choices,
    outcome,
    plan,
    recode,
    recode_column,
    recoded_columns,
    recorded,
    report_params,
    sort_key,
    suggest,
    tick,
    ticked,
    unset_missing,
    untick,
    values_fp,
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


def test_spaces_always_fold_into_the_spelling_most_rows_hold() -> None:
    found = suggest(SEX)
    assert found.spellings == {"F ": "F", "M ": "M"}
    assert found.reasons == {"F ": SPACES, "M ": SPACES}
    assert fold({"Luxembourg": 300, " LUXEMBOURG ": 2}) == {" LUXEMBOURG ": "Luxembourg"}
    assert suggest({"Luxembourg": 300, " LUXEMBOURG ": 2}).reasons == {
        " LUXEMBOURG ": SPACES_AND_CAPITALS
    }


def test_capitals_fold_only_in_a_word_and_only_in_a_typo_s_number_of_rows() -> None:
    """A word's case variant of 5 rows or fewer and under 5 % of its group folds; the others are offered."""
    assert fold({"Luxembourg": 300, "luxembourg": 2}) == {"luxembourg": "Luxembourg"}
    assert fold({"Luxembourg": 300, "luxembourg": 200}) == {}
    assert _offers({"Luxembourg": 300, "luxembourg": 200})["luxembourg"] == (
        "Luxembourg",
        CAPITALS,
        None,
    )
    assert fold({"male": 2, "MALE": 2}) == {}  # half of its group: offered
    assert _offers({"male": 2, "MALE": 2}) == {
        "MALE": ("male", CAPITALS, None)
    }  # a word's tie: lower case
    assert _offers({"male": 2, "Male": 2}) == {
        "male": ("Male", CAPITALS, None)
    }  # capitalised first


@pytest.mark.parametrize(
    "counts",
    [
        {"M": 709, "m": 1},  # sex, or Kell K and k
        {"k": 91, "Kk": 8, "K": 1},  # Kell
        {"G/L": 300, "g/l": 2},  # giga or grams per litre
        {"AA": 900, "Aa": 95, "aa": 5},  # a rare homozygote is no misspelling
        {"Cde": 40, "cde": 300},
        {"pH": 300, "PH": 2},
        {
            "ccdee": 400,
            "Ccdee": 3,
            "CCDEE": 2,
            "CcDee": 100,
        },  # Rh: capitals inside a word, in the column
    ],
)
def test_in_a_code_or_where_capitals_sit_inside_words_case_never_folds(
    counts: dict[str, int],
) -> None:
    assert fold(counts) == {}
    assert all(
        o.why == CAPITALS for o in suggest(counts).offers if o.clean is not None and o.value != "Kk"
    )


def test_a_code_s_tie_takes_capitals() -> None:
    assert _offers({"ab": 3, "AB": 3, "x": 400})["ab"][:2] == ("AB", CAPITALS)


def test_a_one_letter_code_and_the_only_value_it_begins_are_offered_never_applied() -> None:
    """The rarer joins the more frequent: the 2 rows of Female and female could join the 491 of F."""
    offers = suggest(SEX).offers
    assert (
        Offer("Female", "F", FIRST_LETTER) in offers and Offer("male", "M", FIRST_LETTER) in offers
    )
    assert {o.value: o.why for o in offers if o.why == CAPITALS} == {
        "MALE": CAPITALS,
        "female": CAPITALS,
        "f": CAPITALS,
        "m": CAPITALS,
    }
    assert _offers({"Yes": 300, "Y": 4, "No": 200, "N": 3}) == {
        "Y": ("Yes", FIRST_LETTER, None)
    }  # No: 2 letters
    assert _offers({"A": 300, "AB": 40, "B": 200, "O": 400}) == {}  # blood groups
    assert (
        _offers({"II": 526, "III": 396, "I": 186, "IV": 98}) == {}
    )  # NYHA I begins II, III and IV
    assert _offers({"III": 396, "I": 186}) == {}  # a Roman numeral is a category
    assert _offers({"Type 2": 300, "T": 4}) == {}  # so is a value holding a digit


DISTINCT = [
    ("Normal", "Abnormal"),
    ("Stable angina", "Unstable angina"),
    ("STEMI", "NSTEMI"),
    ("Hypertension", "Hypotension"),
    ("Hyperkalaemia", "Hypokalaemia"),
    ("Hypocalcaemia", "Hypokalaemia"),  # two letters apart
    ("Hypercalcaemia", "Hyperkalaemia"),
    ("Craniotomy", "Craniectomy"),
    ("Hysterectomy", "Hysterotomy"),
    ("Laminectomy", "Laminotomy"),
    ("Nephrectomy", "Nephrotomy"),
    ("Prednisolone", "Prednisone"),
    ("Abduction", "Adduction"),
    ("Macrocytic", "Microcytic"),
    ("Peroneal", "Perineal"),
    ("Typical", "Atypical"),
    ("Febrile", "Afebrile"),
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
    ("HER2", "HER2-"),
]


@pytest.mark.parametrize(("common", "rare"), DISTINCT)
def test_no_spelling_is_offered_between_two_categories(common: str, rare: str) -> None:
    assert _offers({common: 400, rare: 3}) == {}


@pytest.mark.parametrize(
    ("common", "rare"),
    [
        ("Dysphagia", "Dysphasia"),
        ("Aphagia", "Aphasia"),
        ("Colostomy", "Colotomy"),
        ("Tracheostomy", "Tracheotomy"),
    ],
)
def test_two_terms_one_letter_apart_are_offered_only_as_a_spelling_to_check(
    common: str, rare: str
) -> None:
    """One letter can still separate two terms: such an offer is a spelling, which the page says to check."""
    assert _offers({common: 400, rare: 3}) == {rare: (common, SPELLING, None)}


@pytest.mark.parametrize(
    ("common", "rare"),
    [
        ("Pétange", "Petange"),
        ("Hôpital", "Hopital"),
        ("Hépatite", "Hepatite"),
        ("Düdelingen", "Dudelingen"),
        ("Mère", "Mere"),
        ("Anaemia", "Anemia"),
        ("Haemorrhage", "Hemorrhage"),
        ("Oedema", "Edema"),
        ("Paediatric", "Pediatric"),
        ("Oesophageal", "Esophageal"),
        ("Leukaemia", "Leukemia"),
        ("Esch-sur-Alzette", "Esch sur Alzette"),
    ],
)
def test_accents_and_british_spellings_are_offered_as_variants_whatever_their_rows(
    common: str, rare: str
) -> None:
    assert _offers({common: 400, rare: 3}) == {rare: (common, VARIANT, None)}
    assert _offers({common: 300, rare: 200}) == {rare: (common, VARIANT, None)}


def test_a_spelling_one_letter_apart_is_offered_rare_or_to_check_first() -> None:
    assert suggest({"Luxembourg": 397, "Esch-sur-Alzette": 300, "Luxembourgg": 1}).offers == (
        Offer("Luxembourgg", "Luxembourg", SPELLING),
    )
    assert _offers({"Tumour": 400, "Tumor": 3}) == {"Tumor": ("Tumour", SPELLING, None)}
    # more rows than a typo, under 5 % of its target: offered to check first
    assert _offers({"Luxembourg": 400, "Luxembourgg": 15}) == {
        "Luxembourgg": ("Luxembourg", CLOSE, None)
    }
    # 5 % or more: not offered
    assert _offers({"Luxembourg": 300, "Luxembourgg": 20}) == {}
    # a first difference within a word's first four letters: another word
    assert _offers({"Differdange": 300, "Diferdange": 2}) == {}
    # looked for only in the open variable (the list of variables compares no two spellings)
    assert suggest({"Luxembourg": 397, "Luxembourgg": 1}, near=False).offers == ()


def test_codes_for_unknown_are_offered_as_unknown_and_the_others_ask_what_they_mean() -> None:
    """ "n.a." may mean not applicable, "none" an absence, "NR" non-reactive, "ND" not detected, "NK" not known or
    natural killer: the user says. "M" (missing, male, married), "N" (No, Never) and "Na" (sodium) are never
    guessed."""
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
    for code in ("n.a.", "NR", "none", "ND", "NK", "-", "N/A"):
        assert _offers({"pre": 300, "post": 200, code: 40})[code] == (None, MISSING_CODE, None), (
            code
        )
    assert _offers({"K": 300, "Cl": 200, "Na": 40}) == {}


def test_a_choice_for_a_spelling_applies_to_its_group_and_a_value_s_own_choice_wins() -> None:
    counts = {"F": 489, "F ": 1, "f": 1, "Female": 1, "unknown": 1}
    assert plan(counts, {}) == {"F ": "F"}
    assert plan(counts, {"Female": "F", "f": "F", "unknown": None}) == {
        "F ": "F",
        "f": "F",
        "Female": "F",
        "unknown": None,
    }
    assert plan(counts, {"F": "Female"}) == {
        "F": "Female",
        "F ": "Female",
    }  # f: a code's case, apart
    assert plan(counts, {"F": "Female", "F ": "F"}) == {
        "F": "Female",
        "F ": "Female",
    }  # F␣ → F → Female
    assert plan({"F": 400, "F ": 1}, {}, kept=["F "]) == {}  # a fold put back as written
    assert plan(counts, {}, fold_rule=None) == {}  # no fold: the mapping lists every change
    with pytest.raises(ValueError, match="unknown fold rule"):
        plan(counts, {}, fold_rule="other")


def test_a_clean_value_given_another_is_followed() -> None:
    """MALE → male and male → M: MALE becomes M. A cycle takes the first choice only; a step recorded before the fold
    rule chains nothing."""
    assert plan(SEX, {"MALE": "male", "male": "M"})["MALE"] == "M"
    assert plan({"F": 3, "Female": 3}, {"F": "Female", "Female": "F"}) == {
        "F": "Female",
        "Female": "F",
    }
    assert plan({"MALE": 1, "male": 1, "M": 5}, {"MALE": "male", "male": "M"}, fold_rule=None) == {
        "MALE": "male",
        "male": "M",
    }


def test_the_values_are_recoded_into_a_text_copy_beside_the_original() -> None:
    df = pd.DataFrame(
        {"id": [1, 2, 3, 4, 5], "sex": ["F", "f ", None, "unknown", "X"], "age": [5, 6, 7, 8, 9]}
    )
    out, kept = recode_column(df, "sex", "sex_clean", {"unknown": None})
    assert list(out.columns) == ["id", "sex", "sex_clean", "age"]
    assert out["sex"].tolist() == df["sex"].tolist()  # the original is kept
    # spaces folded; f is a code, whose case is offered, never folded
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


def test_the_values_as_written_are_listed_most_frequent_first_and_sort_a_to_z_with_their_spellings() -> (
    None
):
    assert level_counts(pd.Series(["b", "a", "b", None, "a ", "a"])) == {"a": 2, "b": 2, "a ": 1}
    assert sorted(["Petange", "Luxembourg", "Pétange", " Petange", "luxembourg"], key=sort_key) == [
        "Luxembourg",
        "luxembourg",
        " Petange",
        "Petange",
        "Pétange",
    ]


# The choices of the page (give, tick, untick, unset_missing) and what the step records of them


def test_the_record_gives_the_preview_when_a_spelling_is_not_written() -> None:
    """Every row reads "Unknown␣": the offer writes the spelling's choice, an edit on the row the value's own. Both
    made each other redundant, and the record kept neither."""
    counts = {"Unknown ": 3, "Yes": 20, "No": 20}
    choices = new_choices()
    offer = next(o for o in suggest(counts).offers if o.value == "Unknown")
    tick(choices, counts, offer)
    give(choices, counts, "Unknown ", None, meaning=DECLINED)
    mapping, kinds, kept, meanings = recorded(counts, choices)
    assert outcome(counts, mapping, meanings, kept=kept) == outcome(
        counts, choices["mapping"], choices["meanings"], kept=choices["kept"]
    )
    assert outcome(counts, mapping, meanings, kept=kept)["Unknown "] == (None, DECLINED)


def test_a_typed_value_and_its_offer_agree_and_the_offer_reads_ticked() -> None:
    counts = {"Luxembourg": 400, "Luxembourgg ": 1}
    choices = new_choices()
    give(choices, counts, "Luxembourgg ", "Luxembourg")
    (offer,) = suggest(counts).offers
    assert ticked(choices, counts, offer)  # the row already goes where the offer proposes
    tick(choices, counts, offer)
    mapping, _, kept, _ = recorded(counts, choices)
    assert plan(counts, mapping, kept=kept) == {"Luxembourgg ": "Luxembourg"}


def test_unticking_missing_on_a_member_row_keeps_it_as_written() -> None:
    counts = {"unknown": 30, "unknown ": 1, "yes": 400}
    choices = new_choices()
    (offer,) = [o for o in suggest(counts).offers if o.value == "unknown"]
    tick(choices, counts, offer)
    unset_missing(choices, counts, "unknown ")
    assert outcome(counts, choices["mapping"], choices["meanings"], kept=choices["kept"])[
        "unknown "
    ] == (
        "unknown ",
        None,
    )
    untick(choices, offer)
    assert not ticked(choices, counts, offer)


def test_keep_as_written_on_a_spelling_reverts_its_group() -> None:
    counts = {"Female": 3, "female": 1, "F": 400}
    choices = new_choices()
    (offer,) = [o for o in suggest(counts).offers if o.value == "Female"]
    tick(choices, counts, offer)
    give(choices, counts, "Female", "Female")
    assert plan(counts, choices["mapping"], kept=choices["kept"]) == {}


def _actions(rng: random.Random, counts: dict[str, int]):
    values = list(counts)
    offers = suggest(counts).offers
    for _ in range(rng.randint(1, 8)):
        what = rng.random()
        value = rng.choice(values)
        if what < 0.25 and offers:
            yield ("tick", rng.choice(offers))
        elif what < 0.35 and offers:
            yield ("untick", rng.choice(offers))
        elif what < 0.55:
            yield ("give", value, rng.choice([*values, "Other", "F", ""]))
        elif what < 0.75:
            yield ("missing", value, rng.choice([*MEANINGS, None]))
        elif what < 0.85:
            yield ("keep", value)
        else:
            yield ("unset", value)


def test_the_record_always_gives_the_preview() -> None:
    """Random sequences of the page's actions on columns whose spellings are not all written as such."""
    rng = random.Random(20261030)
    pools = [
        {"Unknown ": 3, "unknown": 1, "Yes": 20, "No": 20, "F": 10, "F ": 2, "Female": 1, "f": 1},
        {"Luxembourg": 400, "Luxembourgg ": 1, " Metz": 30, "Metz ": 5, "?": 2, "n.a.": 3},
        {"MALE": 1, "male": 1, "M": 50, "m": 1, "Pétange": 30, "Petange": 2, "ND": 4},
    ]
    for _ in range(600):
        counts = rng.choice(pools)
        choices = new_choices()
        for action in _actions(rng, counts):
            if action[0] == "tick":
                tick(choices, counts, action[1])
            elif action[0] == "untick":
                untick(choices, action[1])
            elif action[0] == "give":
                give(choices, counts, action[1], action[2])
            elif action[0] == "missing":
                give(choices, counts, action[1], None, meaning=action[2])
            elif action[0] == "keep":
                give(choices, counts, action[1], action[1])
            else:
                unset_missing(choices, counts, action[1])
        if any(v == "" for v in choices["mapping"].values()):
            continue  # Apply waits for a clean value
        mapping, kinds, kept, meanings = recorded(counts, choices)
        preview = outcome(
            counts,
            {k: v for k, v in choices["mapping"].items()},
            choices["meanings"],
            kept=choices["kept"],
        )
        assert outcome(counts, mapping, meanings, kept=kept) == preview, (counts, choices)
        assert set(kinds) == set(mapping)


def test_merging_categories_by_hand_is_a_grouping_and_an_offer_is_not() -> None:
    nyha = {"II": 526, "III": 396, "I": 186, "IV": 40}
    assert groupings(nyha, {"IV": "III"}, {"IV": CHOSEN}) == {
        "III": ["III", "IV"]
    }  # 4 %, yet two categories
    assert groupings(nyha, {"III": "III-IV", "IV": "III-IV"}, {"III": CHOSEN, "IV": CHOSEN}) == {
        "III-IV": ["III", "IV"]
    }
    assert groupings({"M": 90, "m": 10}, {"m": "M"}, {"m": CAPITALS}) == {}  # an offer
    assert (
        groupings({"F": 400, "Female": 1}, {"Female": "F"}, {"Female": CHOSEN}) == {}
    )  # a first letter
    assert groupings({"F": 400, "Female": 300}, {"Female": "F"}, {"Female": CHOSEN}) == {
        "F": ["F", "Female"]
    }
    assert (
        groupings({"Male": 300, "MALE": 200}, {"MALE": "Male"}, {"MALE": CHOSEN}) == {}
    )  # one word


def test_the_changes_are_counted_by_kind_and_meaning() -> None:
    counts = {"F": 489, "F ": 1, "f": 1, "Female": 1, "n.a.": 3, "?": 2}
    mapping = {"Female": "F", "f": "F", "n.a.": None, "?": None}
    kinds = {"Female": FIRST_LETTER, "f": CAPITALS, "n.a.": MISSING_CODE, "?": MISSING_CODE}
    meanings = {"n.a.": NOT_APPLICABLE, "?": UNKNOWN}
    assert changes(counts, mapping, kinds, meanings=meanings) == {
        "by_kind": {SPACES: [1, 1], CAPITALS: [1, 1], FIRST_LETTER: [1, 1], MISSING_CODE: [2, 5]},
        "set_to_missing": {NOT_APPLICABLE: [1, 3], UNKNOWN: [1, 2]},
    }


def test_the_trace_report_counts_the_choices_names_no_value_and_says_each_grouping() -> None:
    params = {
        "source": "sex",
        "target": "sex_clean",
        "fold": FOLD,
        "mapping": {"CANARY-1": "F", "CANARY-2": None, "III": "III-IV", "IV": "III-IV"},
        "kinds": {
            "CANARY-1": FIRST_LETTER,
            "CANARY-2": MISSING_CODE,
            "III": "grouping",
            "IV": "grouping",
        },
        "kept": ["CANARY-3"],
        "missing_meaning": {"CANARY-2": UNKNOWN},
        "groupings": {
            "III-IV": {
                "reason": "advanced heart failure",
                "values": ["III", "IV"],
                "each_5_rows": True,
            },
            "CANARY-G": {
                "reason": "rare ones",
                "values": ["CANARY-4", "CANARY-5"],
                "each_5_rows": False,
            },
        },
        "target_fp": "0123456789abcdef",
        "values_fp": "fedcba9876543210",
    }
    shown = report_params(params)
    assert "CANARY" not in repr(shown) and "0123456789abcdef" not in repr(shown)
    assert "fedcba9876543210" not in repr(shown)
    assert shown["choices"] == {FIRST_LETTER: 1, MISSING_CODE: 1, "grouping": 2}
    assert shown["folds put back as written"] == 1
    assert shown["groupings"] == [
        "III, IV grouped as III-IV: advanced heart failure",
        "2 values grouped: rare ones",
    ]
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

    sex = ["F"] * 30 + ["F ", "Female"] + ["M"] * 30
    counts = level_counts(pd.Series(sex))
    params = {
        "source": "sex",
        "target": "sex_clean",
        "fold": FOLD,
        "mapping": {"Female": "F"},
        "kept": [],
        "values_after": 2,
        "kept_as_written": 2,
        "values_fp": values_fp(counts),
    }
    step = {"function": "recode_levels", "params": params}
    replayed = apply_steps(pd.DataFrame({"sex": sex}), [step])
    assert replayed.error is None and not replayed.warnings
    assert replayed.df is not None and set(replayed.df["sex_clean"]) == {"F", "M"}
    other = apply_steps(pd.DataFrame({"sex": [*sex, "W"]}), [step])
    assert other.df is not None and other.df["sex_clean"].tolist()[-1] == "W"
    assert any("1 more value(s) of sex kept as written" in w for w in other.warnings)
    assert any(
        "sex_clean holds 3 distinct values, 2 when it was recorded" in w for w in other.warnings
    )
    # the same counts, another value as written: only the fingerprint tells
    swapped = apply_steps(pd.DataFrame({"sex": [v if v != "F " else "F  " for v in sex]}), [step])
    assert any("differ from those the step saw" in w for w in swapped.warnings)
    assert len(swapped.warnings) == 1 and "F  " not in swapped.warnings[0]


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
