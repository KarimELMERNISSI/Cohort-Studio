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
    CASE,
    CHOSEN,
    CLOSE,
    DECLINED,
    FIRST_LETTER,
    FOLD,
    MEANINGS,
    MISSING_CODE,
    NOT_APPLICABLE,
    SPACES,
    SPELLING,
    UNKNOWN,
    VARIANT,
    Offer,
    candidates,
    changes,
    documented,
    fold,
    give,
    grouped_apart,
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
    swaps,
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


def test_spaces_fold_into_the_value_with_its_spaces_collapsed() -> None:
    found = suggest(SEX)
    assert found.spellings == {"F ": "F", "M ": "M"}
    assert fold({"Luxembourg": 300, " LUXEMBOURG ": 2}) == {" LUXEMBOURG ": "LUXEMBOURG"}
    # its capitals, offered
    assert _offers({"Luxembourg": 300, " LUXEMBOURG ": 2}) == {
        "LUXEMBOURG": ("Luxembourg", CAPITALS, None)
    }
    assert plan({"Luxembourg": 300, " LUXEMBOURG ": 2}, {"LUXEMBOURG": "Luxembourg"}) == {
        " LUXEMBOURG ": "Luxembourg"
    }


def test_capitals_never_fold_they_are_offered() -> None:
    """Case can name a category (Kell K and k, anti-k and anti-K, Rh Ccdee and ccdee): a case variant is offered at
    any row count, never folded. A word's variant of a typo's size is the same word written another way; two sites
    writing a word two ways in many rows, a case to check."""
    assert fold({"Luxembourg": 300, "luxembourg": 2}) == {}
    assert _offers({"Luxembourg": 300, "luxembourg": 2})["luxembourg"] == (
        "Luxembourg",
        CAPITALS,
        None,
    )
    assert _offers({"Luxembourg": 300, "luxembourg": 200})["luxembourg"] == (
        "Luxembourg",
        CASE,
        None,
    )
    assert (
        fold({"anti-K": 40, "anti-k": 2}) == {}
        and fold({"Male": 300, "MALE": 200, "male": 3}) == {}
    )
    assert fold({"male": 2, "MALE": 2}) == {}
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
def test_in_a_code_or_where_capitals_sit_inside_words_case_is_offered_to_check(
    counts: dict[str, int],
) -> None:
    """Never folded, and never in the group that "Tick all" ticks: one click turned 9 Rh phenotypes, 153 D-negative
    rows among them, into CcDee."""
    assert fold(counts) == {}
    assert all(
        o.why == CASE for o in suggest(counts).offers if o.clean is not None and o.value != "Kk"
    )


@pytest.mark.parametrize(
    "counts",
    [
        {"anti-K": 40, "anti-k": 2},  # a one-letter word: the antibody
        {"K+": 300, "k+": 3},  # Kell
        {"Luxembourg": 300, "luxembourg": 2, "CcDee": 5},  # a column that writes case codes
        {"Female": 300, "FEMALE": 40},  # many rows: two sites, or two categories
    ],
)
def test_a_case_variant_is_one_word_only_in_plain_words_of_a_typo_s_size(
    counts: dict[str, int],
) -> None:
    variant = min(
        (v for v in counts if v.casefold() == max(counts, key=counts.get).casefold()),
        key=counts.get,
    )
    assert _offers(counts)[variant][1] == CASE


def test_a_code_s_tie_takes_capitals() -> None:
    assert _offers({"ab": 3, "AB": 3, "x": 400})["ab"][:2] == ("AB", CASE)


def test_a_one_letter_code_and_the_only_value_it_begins_are_offered_never_applied() -> None:
    """The rarer joins the more frequent: the 2 rows of Female and female could join the 491 of F."""
    offers = suggest(SEX).offers
    assert (
        Offer("Female", "F", FIRST_LETTER) in offers and Offer("male", "M", FIRST_LETTER) in offers
    )
    assert {o.value: o.why for o in offers if o.why in (CAPITALS, CASE)} == {
        "MALE": CAPITALS,
        "female": CAPITALS,
        "f": CASE,  # one letter: Kell k and K are two antigens
        "m": CASE,
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
    ("Abduction", "Adduction"),
    ("Macrocytic", "Microcytic"),
    ("Colostomy", "Colotomy"),  # another ending on one root
    ("Tracheostomy", "Tracheotomy"),
    ("Unilateral", "Bilateral"),
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
        ("Prednisolone", "Prednisone"),  # two edits in a word of 10 letters: a typo's size
        ("Peroneal", "Perineal"),
        ("Ilium", "Ileum"),
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
    # a typo early in a word: the prefixes that negate or oppose are guarded, not the first four letters
    assert _offers({"Differdange": 300, "Diferdange": 2}) == {
        "Diferdange": ("Differdange", SPELLING, None)
    }
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
    for code in ("n.a.", "NR", "none", "ND", "NK", "-", "N/A", "NC", "SO", "Refus", "Not tested",
                 "Not recorded", "Not assessed", "Pending", "Awaiting result", "Inconclusive", "Don’t know",
                 "Prefer not to say", "Non précisé", "Manquant", "N/K", "Not performed", "Not obtained",
                 "No data", "Missing data", "Results pending", "Unable to assess", "Insufficient sample", "QNS",
                 "N.D.", "DK", "TBC"):  # fmt: skip
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


def test_a_clean_value_given_another_is_followed() -> None:
    """MALE → male and male → M: MALE becomes M. A cycle takes the first choice only."""
    assert plan(SEX, {"MALE": "male", "male": "M"})["MALE"] == "M"
    assert plan({"F": 3, "Female": 3}, {"F": "Female", "Female": "F"}) == {
        "F": "Female",
        "Female": "F",
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
    untick(choices, counts, offer)
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
                untick(choices, counts, action[1])
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


def test_a_merge_no_offer_relation_joins_is_a_grouping_whoever_proposed_it() -> None:
    """One test, whoever proposed the merge: the user, an offer or the AI engine. A merge the AI proposed was never a
    grouping, and the same merge typed by hand was one; a hand merge of unrelated words under 5 % of rows was none."""
    nyha = {"II": 526, "III": 396, "I": 186, "IV": 40}
    assert groupings(nyha, {"IV": "III"}) == {"III": ["III", "IV"]}  # 4 %, yet two categories
    assert groupings(nyha, {"III": "III-IV", "IV": "III-IV"}) == {"III-IV": ["III", "IV"]}
    assert groupings({"F": 400, "Female": 1}, {"Female": "F"}) == {}  # a first letter
    assert (
        groupings({"F": 400, "Female": 300}, {"Female": "F"}) == {}
    )  # the code and its word, however many rows
    assert groupings({"Male": 300, "MALE": 2}, {"MALE": "Male"}) == {}  # one word, a typo's rows
    assert groupings({"Male": 300, "MALE": 200}, {"MALE": "Male"}) == {
        "Male": ["MALE", "Male"]
    }  # two sites?
    assert groupings({"M": 90, "m": 10}, {"m": "M"}) == {
        "M": ["M", "m"]
    }  # Kell K and k are two antigens
    rh = {"ccdee": 140, "CcDee": 400, "CCDee": 250}
    assert groupings(rh, {"CCDee": "CcDee"}) == {
        "CcDee": ["CCDee", "CcDee"]
    }  # a column of case codes
    assert (
        groupings({"Luxembourg": 400, "Luxembourgg": 2}, {"Luxembourgg": "Luxembourg"}) == {}
    )  # one letter
    assert groupings({"Pétange": 300, "Petange": 200}, {"Petange": "Pétange"}) == {}  # accents
    stable = {"Stable angina": 300, "Unstable angina": 4}
    assert groupings(stable, {"Unstable angina": "Stable angina"}) == {
        "Stable angina": ["Stable angina", "Unstable angina"]
    }
    for small in ({"Negative": 400, "Pending": 3}, {"Ibuprofen": 300, "Naproxen": 4}):
        rare, common = sorted(small, key=small.get)
        assert groupings(small, {rare: common}) == {common: sorted(small)}, small
    # a typo's fix is an offer relation: no reason asked; the same edit in many rows is a value of its own
    typo = {"Diabetes mellitus": 300, "Diabetes melitus": 2}
    assert groupings(typo, {"Diabetes melitus": "Diabetes mellitus"}) == {}
    many = {"Diabetes mellitus": 300, "Diabetes melitus": 40}
    assert groupings(many, {"Diabetes melitus": "Diabetes mellitus"}) == {
        "Diabetes mellitus": ["Diabetes melitus", "Diabetes mellitus"]
    }


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
        "2 values grouped (the reason is in the trace)",
    ]
    assert "rare ones" not in repr(shown)  # free text, perhaps naming the rare values
    # a key another version writes is never printed raw
    assert "CANARY" not in repr(report_params({**params, "reasons": {"CANARY-6": "chosen"}}))


# the records of 30 October, in the shapes their pages wrote: no fold and "reasons" (baba1b7, 3ea6198), groupings as
# text (747ccc3), groupings as mappings (191a550), the proposer under "ai" (751ad38)
_COMMON = {"source": "sex", "target": "sex_clean", "replaces": False, "values_before": 6}
_EARLIER = {
    "baba1b7": {
        **_COMMON,
        "mapping": {"MALE": "Male", "male ": "Male", "Feemale": "Female", "ND": None},
        "reasons": {
            "MALE": "spaces and capitals",
            "male ": "spaces and capitals",
            "Feemale": "chosen",
        },
        "values_after": 2,
    },
    "747ccc3": {
        **_COMMON,
        "fold": "spaces_and_capitals",
        "mapping": {"Feemale": "Female", "ND": None},
        "kinds": {"Feemale": "spelling", "ND": "missing code"},
        "groupings": {"Female": "Feemale is a typo"},
    },
    "191a550": {
        **_COMMON,
        "fold": "spaces_and_capitals",
        "mapping": {"Feemale": "Female", "ND": None},
        "groupings": {"Female": {"reason": "typo", "values": ["Female", "Feemale"]}},
        "values_fp": "0" * 16,
    },
}
_EARLIER["751ad38"] = {**_EARLIER["191a550"], "ai": {"provider": "Anthropic", "model": "claude"}}


@pytest.mark.parametrize("shape", sorted(_EARLIER))
def test_a_recode_of_an_earlier_version_is_reported_by_its_columns_and_counts(shape: str) -> None:
    """The replay refuses it; the trace report still builds, and names no value of it."""
    import io

    from docx import Document

    from manage.trace_documenter import TraceDocumenter

    params = _EARLIER[shape]
    shown = report_params(params)
    assert shown["source"] == "sex" and shown["values recoded"] == len(params["mapping"])
    assert not any(v in repr(shown) for v in ("Feemale", "MALE", "male ", "typo"))
    step = {
        "function": "recode_levels",
        "params": params,
        "description": "Recoded sex into sex_clean",
    }
    trace = {"session_id": f"s_{shape}", "source_dataset": "cohort.csv", "steps": [step]}
    document = Document(io.BytesIO(TraceDocumenter(trace).generate_report().getvalue()))
    text = "\n".join(p.text for p in document.paragraphs) + "\n".join(
        c.text for table in document.tables for row in table.rows for c in row.cells
    )
    assert "another version of Data > Clean" in text
    assert not any(v in text for v in ("Feemale", "MALE", "male "))


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
    # extra spaces are counted apart: the fold collapses them, with nothing to check
    assert [(c.column, c.values, c.to_check, c.spaced, c.cleaned_into) for c in found] == [
        ("site", 3, 0, 1, ""),
        ("sex", 3, 1, 0, "sex_clean"),
        ("arm", 2, 0, 0, ""),
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


def test_a_corrected_record_is_no_change_of_the_data_and_the_replay_checks_it() -> None:
    """The choices corrected while the copy keeps its values (what missing means): the step is recorded with the data's
    fingerprint in and out, so the workflow counts no new revision and marks no result out of date, and the replay
    redoes it and finds the table it recorded."""
    from core.progress import changes_data, data_revision
    from core.session import DataSession
    from manage.reproduction_manager import apply_steps

    class Trace:
        def __init__(self) -> None:
            self.steps: list[dict] = []

        def add_step(self, function_name, params, description, output_dataset_path=None, **extra):
            self.steps.append(
                {"function": function_name, "params": params, "description": description, **extra}
            )

    raw = pd.DataFrame({"sex": ["F"] * 30 + ["Female", None] + ["M"] * 30})
    session = DataSession(trace=Trace())
    session.install(raw, params={"source": "test"})
    assert session.base is not None
    new, kept = recode_column(session.base, "sex", "sex_clean", {"Female": "F"})
    params = {
        "source": "sex",
        "target": "sex_clean",
        "fold": FOLD,
        "mapping": {"Female": "F"},
        "kept": [],
        "values_after": 2,
        "kept_as_written": kept,
        "values_fp": values_fp(level_counts(raw["sex"])),
    }
    assert session.commit(new, "recode_levels", params, "Recoded sex into sex_clean") is not None
    again = {**params, "replaces": True, "missing_meaning": {"": "not asked"}}
    same, _ = recode_column(session.base, "sex", "sex_clean", {"Female": "F"}, replaces=True)
    assert session.commit(same, "recode_levels", again, "Recoded sex into sex_clean") is None
    session.record("recode_levels", again, "Recoded sex into sex_clean again, its values unchanged")

    steps = session.trace.steps  # type: ignore[union-attr]
    assert [changes_data(step) for step in steps] == [False, True, False]
    assert data_revision(steps) == 1 and session.revision == 1
    assert steps[-1]["data"]["fp_in"] == steps[-1]["data"]["fp_out"] == session.data_fp
    replayed = apply_steps(raw, steps[1:])
    assert replayed.error is None and not replayed.warnings
    assert replayed.verified_steps == 2 and replayed.divergence is None


@pytest.mark.parametrize("fold_id", ["spaces_and_capitals", None])
def test_a_recode_of_an_earlier_version_is_refused_by_the_replay(fold_id: str | None) -> None:
    """Only the fold rule of today is kept: a recode recorded under another (30 October 2026), or before the fold rule,
    is refused with a word, never replayed under a rule it did not use."""
    from manage.reproduction_manager import apply_steps

    params = {"source": "sex", "target": "sex_clean", "mapping": {}, "kept": [], "fold": fold_id}
    step = {
        "function": "recode_levels",
        "params": {k: v for k, v in params.items() if v is not None},
    }
    replayed = apply_steps(pd.DataFrame({"sex": ["Male", "MALE", "Female"]}), [step])
    assert replayed.df is None and "another version of Data > Clean (fold rule: " in (
        replayed.error or ""
    )
    assert "clean it again" in (replayed.error or "")


def test_a_recode_recorded_without_what_the_replay_checks_is_refused_with_a_word() -> None:
    """A record of today's fold rule that lacks its counts (written by hand, or cut) is refused by name, never with a
    bare key name for an error."""
    from manage.reproduction_manager import apply_steps

    params = {"source": "sex", "target": "sex_clean", "mapping": {}, "kept": [], "fold": FOLD}
    replayed = apply_steps(
        pd.DataFrame({"sex": ["Male", "MALE"]}), [{"function": "recode_levels", "params": params}]
    )
    assert replayed.df is None
    assert "recorded without kept_as_written, values_after, values_fp, which the replay checks" in (
        replayed.error or ""
    )


@pytest.mark.parametrize(
    "counts",
    [
        {"IgG κ": 300, "IgG λ": 40},
        {"α-blocker": 300, "β-blocker": 40},
        {"α-thalassaemia trait": 300, "β-thalassaemia trait": 3},
        {"µg/L": 300, "g/L": 40},
        {"≤ 2 cm": 300, "≥ 2 cm": 40},
        {"HER2": 300, "HER2−": 40},
        {"Normal": 300, "β": 4},
    ],
)
def test_greek_letters_units_and_signs_name_categories(counts: dict[str, int]) -> None:
    """Dropped by an ASCII form, they made two categories one "variant" (and β the first letter of Normal)."""
    assert _offers(counts) == {}


def test_ae_and_oe_read_as_e_inside_a_word_only() -> None:
    assert _offers({"Œdème": 300, "Oedème": 3}) == {"Oedème": ("Œdème", VARIANT, None)}
    assert _offers({"Diarrhoea": 300, "Diarrhea": 3}) == {"Diarrhea": ("Diarrhoea", VARIANT, None)}
    assert _offers({"Michel": 300, "Michael": 3}) == {
        "Michael": ("Michel", SPELLING, None)
    }  # to check


def test_values_that_take_each_other_s_name_are_a_swap() -> None:
    counts = {"Oui": 40, "Yes": 300, "No": 200}
    choices = new_choices()
    give(choices, counts, "Oui", "Yes")
    assert swaps(counts, choices) == []
    give(choices, counts, "Yes", "Oui")
    assert sorted(swaps(counts, choices)) == ["Oui", "Yes"]
    chain = new_choices()
    give(chain, SEX, "MALE", "male")
    give(chain, SEX, "male", "M")
    assert swaps(SEX, chain) == []


def test_an_offer_read_ticked_from_a_typed_row_unticks() -> None:
    """The typed row's own choice made the offer read ticked; unticking dropped only the spelling's choice, so the
    pill came back ticked."""
    counts = {"Luxembourg": 400, "Metz": 40, "Luxembourgg ": 1}
    choices = new_choices()
    give(choices, counts, "Luxembourgg ", "Luxembourg")
    (offer,) = [o for o in suggest(counts).offers if o.value == "Luxembourgg"]
    assert ticked(choices, counts, offer)
    untick(choices, counts, offer)
    assert not ticked(choices, counts, offer)
    assert plan(counts, choices["mapping"], kept=choices["kept"]) == {"Luxembourgg ": "Luxembourgg"}


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


def test_a_corrected_meaning_or_reason_documents_another_record() -> None:
    """The copy is the same, the record is not: "declined" in place of "not applicable", another reason."""
    first = {
        "mapping": {"n.a.": None, "IV": "III"},
        "kinds": {"n.a.": CHOSEN, "IV": "grouping"},
        "kept": [],
        "missing_meaning": {"n.a.": NOT_APPLICABLE},
        "groupings": {"III": {"reason": "advanced", "values": ["III", "IV"], "each_5_rows": True}},
    }
    assert documented(first) == documented(dict(first))
    assert documented(first) != documented({**first, "missing_meaning": {"n.a.": DECLINED}})
    other = {**first, "groupings": {"III": {"reason": "pre-specified", "values": ["III", "IV"]}}}
    assert documented(first) != documented(other)
    assert documented({**first, "kept": ["b", "a"]}) == documented({**first, "kept": ["a", "b"]})


def test_the_closest_then_most_frequent_value_is_the_reference_of_a_rarer_one() -> None:
    """A rarer value within a typo's edits of a more frequent one is probably its typo: offered towards the value the
    fewest edits away, then the most frequent; two edits apart in a value of 10 letters or more, only in a typo's rows
    and past a word's first four letters; never on a word's first letter."""
    counts = {"Luxembourg": 400, "Luxemburg": 50, "Luxemburgg": 2, "Esch": 300}
    assert _offers(counts) == {
        "Luxemburgg": ("Luxemburg", CLOSE, None)
    }  # one edit from Luxemburg, two from Luxembourg
    assert _offers({"Luxembourg": 400, "Luxenbourgg": 2}) == {
        "Luxenbourgg": ("Luxembourg", SPELLING, None)
    }
    assert _offers({"Luxembourg": 400, "Luxenbourgg": 15}) == {}  # two edits: a typo's rows only
    assert _offers({"Diabetes": 300, "Diabtes": 10}) == {
        "Diabtes": ("Diabetes", CLOSE, None)
    }  # to check first
    assert _offers({"Ettelbruck": 12, "Ettelbruk": 1}) == {
        "Ettelbruk": ("Ettelbruck", SPELLING, None)
    }  # small column
    assert _offers({"Hypokalaemia": 400, "Hypocalcaemia": 3}) == {}  # one ending, two roots
    assert _offers({"Abduction": 400, "Adduction": 3}) == {}  # ab- and ad-
    for common, rare in (("Progression", "Regression"), ("Neuropathy", "Nephropathy"), ("Sitagliptin", "Saxagliptin"),
                         ("Zambian", "Gambian"), ("Heisdorf", "Reisdorf"), ("Sailor", "Tailor")):  # fmt: skip
        assert _offers({common: 320, rare: 3}) == {}, (
            rare
        )  # two edits early in a word; a word's first letter
    assert _offers({"Sacubitril/valsartan": 300, "Sacubitril/valsartn": 2}) == {
        "Sacubitril/valsartn": ("Sacubitril/valsartan", SPELLING, None)
    }


def test_each_spelling_is_tested_against_the_clean_value_it_ends_at() -> None:
    """Two of the page's offers for one word (capitals, a close spelling) are one category, merged in one step or two:
    each spelling is tested against the clean value, never against the other. A neighbouring category merged with
    them is a grouping, and its question names it apart; a new name for one category is a relabel."""
    counts = {
        "Hypertension": 300,
        "hypertension": 4,
        "Hypertenson": 1,
        "Hypotension": 25,
        "Diabetes": 200,
    }
    spellings = {"hypertension": "Hypertension", "Hypertenson": "Hypertension"}
    assert groupings(counts, spellings) == {}
    assert (
        groupings(counts, {"Hypertenson": "hypertension", "hypertension": "Hypertension"}) == {}
    )  # a chain
    with_hypotension = {**spellings, "Hypotension": "Hypertension"}
    assert groupings(counts, with_hypotension) == {
        "Hypertension": ["Hypertension", "Hypertenson", "Hypotension", "hypertension"]
    }
    assert grouped_apart(counts, with_hypotension) == {"Hypertension": ["Hypotension"]}
    relabel = {v: "Arterial hypertension" for v in ("Hypertension", "hypertension", "Hypertenson")}
    assert groupings(counts, relabel) == {}
    two = {**relabel, "Hypotension": "Arterial hypertension"}
    assert grouped_apart(counts, two) == {"Arterial hypertension": ["Hypotension"]}
