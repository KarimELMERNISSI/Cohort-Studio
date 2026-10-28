"""
The criterion kinds catalogue (core.criteria.CRITERION_KINDS) agrees with itself and with what is stored.

Every kind has examples, keeps the definitions it reads as they were written, says in words the rows it selects (the
words, read back, select the rows of the mask), treats a missing value as it declares, reads only the columns it
lists, and is refused when loaded exactly when it cannot be evaluated. A kind registered later (M6 date_order, M9
concept) fails these tests until it brings its examples and its read-back.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from core.attrition import participant_flow
from core.criteria import (
    ASKED,
    CRITERION_KINDS,
    MEMBERS,
    NEVER,
    NOT_READ,
    CriterionError,
    CriterionKind,
    criterion_columns,
    criterion_mask,
    criterion_problems,
    describe_definition,
    holds_row_list,
    rule_kind,
)
from core.fingerprint import ensure_row_ids
from core.population import (
    catalogue,
    criteria_from_trace,
    evaluate_population,
    without_row_lists,
)
from tests.core_layer.test_criteria_golden import EDGE

TUTORIAL = Path(__file__).resolve().parents[2] / "assets" / "tutorial"

# the ids that traces, configurations and dossiers store: never removed, never renamed (a kind that nothing writes any
# more is registered legacy=True); the five kinds of M5, read from their keys in this order
STORED = ("threshold", "row_ids", "bounds", "numeric", "expression")

OLD = {"numeric": False, "expression": "df['age'] > 80"}
HIGH_SBP = {"numeric": True, "upper_bound": 180, "strategy": "exclude"}  # a rule named "sbp"

# (name, definition): a numeric rule reads the column named like it
EXAMPLES: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "threshold": [
        (
            "two signs",
            {
                "threshold": 2,
                "members": {"old": OLD, "sbp": HIGH_SBP, "ldl": {"bounds": {"ldl": [0.5, 9.0]}}},
            },
        ),
        (
            "any sign",  # a threshold inside a threshold, and listed rows among the members
            {
                "threshold": 1,
                "members": {
                    "two signs": {"threshold": 2, "members": {"old": OLD, "sbp": HIGH_SBP}},
                    "listed": {"row_ids": [[0, 1]], "method": "zscore"},
                },
            },
        ),
    ],
    "row_ids": [
        (
            "flagged",
            {
                "row_ids": [[2, 3], [9, 9]],
                "description": "Rows an outlier method flagged.",
                "method": "iqr",
                "params": {"k": 1.5},
                "columns": ["ldl"],
            },
        ),
        ("listed", {"row_ids": [[0, 0]]}),
    ],
    "bounds": [
        ("frozen", {"bounds": {"ldl": [0.5, 9.0], "sbp": [None, 180.0]}}),
        ("one side", {"bounds": {"age": [18, None], "ldl": [None, None]}}),
    ],
    "numeric": [
        ("age", {"numeric": True, "lower_bound": 18, "upper_bound": 85, "strategy": "include"}),
        ("age", {"numeric": True, "lower_bound": 18, "upper_bound": 85, "strategy": "exclude"}),
        ("adult", {"numeric": True, "column": "age", "lower_bound": 18}),
        ("young", {"numeric": True, "column": "age", "lower_bound": 18, "strategy": "exclude"}),
        ("sbp", HIGH_SBP),
        ("ldl", {"numeric": True, "upper_bound": 9.0}),
    ],
    "expression": [
        ("old", OLD),
        ("male", {"expression": "df['sex'] == 'M'"}),
        ("young or high", {"expression": "(df['age'] < 18) | (df['sbp'] >= 180)"}),
        ("age unknown", {"expression": "df['age'].isna()"}),  # asks for missing values
    ],
}

# definitions of a kind that fail on any table: refused when loaded, and never evaluated (an expression is checked
# when it is evaluated)
MALFORMED: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "threshold": [
        ("a threshold in words", {"threshold": "two", "members": {"old": OLD}}),
        ("rules as a list", {"threshold": 1, "members": ["old"]}),
        ("no rules", {"threshold": 2, "members": {}}),
        (
            "a malformed rule inside",
            {"threshold": 1, "members": {"young": {"numeric": True, "upper_bound": "ten"}}},
        ),
    ],
    "row_ids": [
        ("a range of three", {"row_ids": [[1, 2, 3]]}),
        ("a range in words", {"row_ids": "1-3"}),
    ],
    "bounds": [
        ("bounds of one value", {"bounds": {"age": [1]}}),
        ("bounds as a list", {"bounds": ["age", 1, 2]}),
    ],
    "numeric": [
        ("age", {"numeric": True, "lower_bound": "18 years"}),
        ("age", {"numeric": True}),
        ("age", {"numeric": True, "lower_bound": 18, "strategy": "keep"}),
    ],
}


def _frame() -> pd.DataFrame:
    nan = np.nan  # rows 2, 3, 7 and 8 have a missing value
    return ensure_row_ids(
        pd.DataFrame(
            {
                "age": [10.0, 17.0, 18.0, 45.0, 80.0, 86.0, 95.0, nan, 30.0, 60.0],
                "sbp": [100.0, 185.0, nan, 120.0, 190.0, 140.0, 200.0, 130.0, nan, 175.0],
                "ldl": [0.4, 2.0, 3.1, nan, 9.5, 4.0, 1.0, 5.0, 12.0, 2.2],
                "sex": ["F", "M", None, "F", "M", "F", "M", "F", None, "M"],
            }
        )
    )


def _every(examples: Mapping[str, list[tuple[str, dict[str, Any]]]]) -> list[Any]:
    return [
        pytest.param(kind, name, definition, id=f"{kind}: {name} {i}")
        for kind, cases in examples.items()
        for i, (name, definition) in enumerate(cases)
    ]


def _rows(df: pd.DataFrame, mask: pd.Series) -> list[int]:
    return [int(i) for i in df.index[mask.to_numpy(dtype=bool)]]


# ---------------------------------------------------------------------------------------------------------- read back

_RANGE = re.compile(
    r"(?P<column>.+?) (?P<where>within|outside) \[(?P<low>[^,\]]+), (?P<high>[^,\]]+)\]"
)
_COMPARISON = re.compile(r"(?P<column>.+?) (?P<op>>=|<=|>|<) (?P<value>-?\d+(?:\.\d+)?)")
_OPERATORS = {">": "gt", ">=": "ge", "<": "lt", "<=": "le"}


def _read_condition(df: pd.DataFrame, text: str) -> pd.Series:
    """One condition in words ("age within [18, 85]", "sbp > 180"), as the rows it names: a missing value meets none."""
    if match := _RANGE.fullmatch(text):
        series = df[match["column"]]
        inside = series.between(float(match["low"]), float(match["high"]))
        rows = inside if match["where"] == "within" else ~inside
    elif match := _COMPARISON.fullmatch(text):
        series = df[match["column"]]
        rows = getattr(series, _OPERATORS[match["op"]])(float(match["value"]))
    else:
        raise AssertionError(f"not a condition in words: {text!r}")
    return rows & series.notna()


def _read_bounds(
    df: pd.DataFrame, name: str, definition: Mapping[str, Any], text: str
) -> pd.Series:
    rows = pd.Series(False, index=df.index)
    for part in text.split("; "):
        if not part.endswith(": no bound"):
            rows |= _read_condition(df, part)
    return rows


def _read_listed(df: pd.DataFrame, name: str, definition: Mapping[str, Any], text: str) -> int:
    """Listed rows give how many rows they list, and the method that flagged them: a count, not the rows."""
    match = re.fullmatch(r"(?P<n>[\d,]+) listed rows( \(flagged by (?P<method>[^)]+)\))?", text)
    assert match, text
    assert bool(match["method"]) == bool(definition.get("method")), text
    return int(match["n"].replace(",", ""))


def _read_threshold(
    df: pd.DataFrame, name: str, definition: Mapping[str, Any], text: str
) -> pd.Series:
    """At least k of the rules, each read back from its own words."""
    match = re.match(r"at least (?P<k>\d+) of: ", text)
    assert match, text
    members = definition["members"]
    words = {member: describe_definition(member, spec) for member, spec in members.items()}
    assert text[match.end() :] == "; ".join(f"{m} ({w})" for m, w in words.items()), text
    fired = []
    for member, spec in members.items():
        rows = _read_back(df, member, spec, words[member])
        if isinstance(rows, int):  # listed rows: their count read back, their rows by their ids
            mask = criterion_mask(df, member, spec)
            assert rows == int(mask.sum())
            rows = mask
        fired.append(rows)
    return pd.concat(fired, axis=1).sum(axis=1) >= int(match["k"])


READ_BACK: dict[str, Callable[[pd.DataFrame, str, Mapping[str, Any], str], pd.Series | int]] = {
    "threshold": _read_threshold,
    "row_ids": _read_listed,
    "bounds": _read_bounds,
    "numeric": lambda df, name, definition, text: _read_condition(df, text),
    # the words are the expression
    "expression": lambda df, name, definition, text: criterion_mask(df, name, {"expression": text}),
}


def _read_back(df: pd.DataFrame, name: str, definition: Mapping[str, Any], text: str):
    return READ_BACK[rule_kind(name, definition).id](df, name, definition, text)


# ------------------------------------------------------------------------------------------------------------- tests


def test_no_stored_kind_is_ever_removed() -> None:
    assert set(STORED) <= set(CRITERION_KINDS.ids())
    # none is legacy today: a kind that nothing writes any more is marked so, never deleted
    assert [i for i in CRITERION_KINDS.ids() if CRITERION_KINDS.is_legacy(i)] == []


def test_the_five_kinds_of_m5_are_read_by_their_keys_in_a_fixed_order() -> None:
    """The order decides a rule that holds the keys of two kinds ("threshold" and "row_ids" is a threshold rule)."""
    assert [k.id for k in CRITERION_KINDS if k.recognise is not None] == list(STORED)
    assert (
        rule_kind("both", {"threshold": 1, "members": {"old": OLD}, "row_ids": [[0, 0]]}).id
        == "threshold"
    )


def test_every_kind_has_a_label_examples_a_read_back_and_a_way_with_missing_values() -> None:
    ids = set(CRITERION_KINDS.ids())
    assert set(EXAMPLES) == ids and set(READ_BACK) == ids
    assert all(EXAMPLES[i] for i in ids)
    labels = [k.label for k in CRITERION_KINDS]
    assert all(labels) and len(set(labels)) == len(labels)
    assert {k.missing for k in CRITERION_KINDS} <= {NEVER, ASKED, NOT_READ, MEMBERS}
    assert set(MALFORMED) <= ids


@pytest.mark.parametrize(("kind", "name", "definition"), _every(EXAMPLES))
def test_a_definition_reads_as_its_kind_and_is_kept_as_written(
    kind: str, name: str, definition: dict
) -> None:
    written = copy.deepcopy(definition)
    assert rule_kind(name, definition).id == kind
    config = {"mask_families": {"checks": {"operator": "any", name: definition}}}
    (criterion,) = catalogue(config)
    assert criterion.definition == written and criterion.snapshot()["definition"] == written
    block = json.loads(json.dumps(evaluate_population(_frame(), [criterion]).to_trace()))
    (recorded,) = criteria_from_trace(block)
    assert recorded.definition == written and recorded.fingerprint == criterion.fingerprint
    # nothing that reads a definition changes it
    criterion_mask(_frame(), name, definition)
    criterion_columns(name, definition)
    describe_definition(name, definition)
    criterion_problems(name, definition)
    holds_row_list(definition)
    without_row_lists(config)
    assert definition == written


@pytest.mark.parametrize(("kind", "name", "definition"), _every(EXAMPLES))
def test_the_words_read_back_select_the_rows_of_the_mask(
    kind: str, name: str, definition: dict
) -> None:
    df = _frame()
    mask = criterion_mask(df, name, definition)
    expected = _read_back(df, name, definition, describe_definition(name, definition))
    if isinstance(expected, int):
        assert expected == int(mask.sum())
    else:
        assert _rows(df, mask) == _rows(df, expected)


@pytest.mark.parametrize(("kind", "name", "definition"), _every(EXAMPLES))
def test_each_kind_treats_a_missing_value_as_it_declares(
    kind: str, name: str, definition: dict
) -> None:
    df = _frame()
    declared = rule_kind(name, definition)
    before = criterion_mask(df, name, definition)
    columns = list(criterion_columns(name, definition))
    blank = df.assign(**dict.fromkeys(columns, np.nan))
    after = criterion_mask(blank, name, definition)
    if declared.missing == NOT_READ:  # the listed rows fire, whatever their values
        everything_blank = df.assign(**dict.fromkeys(df.columns, np.nan))
        assert after.equals(before)
        assert criterion_mask(everything_blank, name, definition).equals(before)
    elif declared.missing == MEMBERS:  # as the rules it combines
        members = declared.members(definition)
        fired = pd.concat([criterion_mask(blank, m, spec) for m, spec in members.items()], axis=1)
        assert after.equals(fired.sum(axis=1) >= int(definition["threshold"]))
    elif declared.missing == ASKED and ".isna()" in definition["expression"]:
        assert after.all()  # it asks for missing values
    else:  # NEVER, and an expression that does not ask: a row with every column blank never fires
        assert declared.missing in (NEVER, ASKED) and columns
        assert not after.any()
        # blanking a value can only turn a firing off
        partly = df.copy()
        partly.loc[df.index[::2], columns] = np.nan
        assert not (criterion_mask(partly, name, definition) & ~before).any()


@pytest.mark.parametrize(("kind", "name", "definition"), _every(EXAMPLES))
def test_a_rule_reads_only_the_columns_it_lists(kind: str, name: str, definition: dict) -> None:
    df = _frame()
    columns = list(criterion_columns(name, definition))
    assert set(columns) <= set(df.columns)
    narrowed = criterion_mask(df.filter(items=columns), name, definition)
    assert _rows(df, narrowed) == _rows(df, criterion_mask(df, name, definition))


@pytest.mark.parametrize(("kind", "name", "definition"), _every(EXAMPLES))
def test_a_rule_that_evaluates_is_loaded(kind: str, name: str, definition: dict) -> None:
    assert criterion_problems(name, definition) == []
    assert criterion_mask(_frame(), name, definition).dtype == bool


@pytest.mark.parametrize(("kind", "name", "definition"), _every(MALFORMED))
def test_a_rule_that_cannot_be_evaluated_is_refused_when_loaded(
    kind: str, name: str, definition: dict
) -> None:
    assert rule_kind(name, definition).id == kind
    assert criterion_problems(name, definition)
    with pytest.raises(CriterionError):
        criterion_mask(_frame(), name, definition)
    assert describe_definition(name, definition)  # never an error


def test_every_stored_definition_reads_as_a_registered_kind() -> None:
    """The tutorial's configuration and the blocks pinned before the catalogue (test_criteria_golden.EDGE)."""
    tutorial = json.loads((TUTORIAL / "tutorial_config.json").read_text(encoding="utf-8"))
    for config in (tutorial, EDGE):
        for criterion in catalogue(config):
            kind = rule_kind(criterion.name, criterion.definition)
            assert kind.id in STORED
            for member, spec in kind.members(criterion.definition).items():
                assert rule_kind(member, spec).id in STORED


def test_a_row_list_is_found_by_its_kind_or_its_key() -> None:
    """holds_row_list reads the kinds (listed rows, and the rules a threshold combines); a "row_ids" or "members" key
    counts whatever the rule reads as, so the row ids of a rule that cannot be read never reach a shared file."""
    assert [k.id for k in CRITERION_KINDS if k.lists_rows] == ["row_ids"]
    for _, name, definition in (p.values for p in _every(EXAMPLES)):
        kind = rule_kind(name, definition)
        expected = kind.lists_rows or any(
            holds_row_list(spec) for spec in kind.members(definition).values()
        )
        assert holds_row_list(definition) == expected, name
    assert holds_row_list({"threshold": 1, "members": {"old": OLD}, "row_ids": [[0, 0]]})
    assert holds_row_list({"expression": "df['x'] > 1", "members": {"a": {"row_ids": [[0, 0]]}}})
    unreadable = {
        "numeric": True,
        "expression": "df['x'] > 1",
        "members": {"a": {"row_ids": [[0, 0]]}},
    }
    with pytest.raises(CriterionError):
        rule_kind("both", unreadable)
    assert holds_row_list(unreadable)
    assert not holds_row_list({"members": {"a": OLD}}) and not holds_row_list(["row_ids"])


def test_a_kind_declares_its_row_lists_and_the_rules_it_combines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A kind registered later is read for row lists by what it declares, not by the keys of the five."""
    monkeypatch.setattr(CRITERION_KINDS, "_kinds", dict(CRITERION_KINDS._kinds))

    def nothing(df: pd.DataFrame, name: str, definition: Mapping[str, Any]) -> pd.Series:
        return pd.Series(False, index=df.index)

    def kind(id: str, **declared: Any) -> CriterionKind:
        return CriterionKind(
            id,
            id.title(),
            mask=nothing,
            columns=lambda name, definition: (),
            describe=lambda name, definition: id,
            recognise=lambda name, definition: id in definition,
            **declared,
        )

    CRITERION_KINDS.register(kind("sample", lists_rows=True))
    CRITERION_KINDS.register(kind("group", members=lambda definition: definition["rules"]))
    assert holds_row_list({"sample": "rows 4 to 9"})
    assert holds_row_list({"group": True, "rules": {"a": {"sample": "rows 4 to 9"}}})
    assert not holds_row_list({"group": True, "rules": {"a": OLD}})


def test_the_flow_and_the_dossier_give_the_words() -> None:
    from app_pages.study_dossier import _criteria_table

    cases = [(name, definition) for cases in EXAMPLES.values() for name, definition in cases]
    config = {
        "mask_families": {
            f"family_{i}": {"operator": "any", name: definition}
            for i, (name, definition) in enumerate(cases)
        }
    }
    population = evaluate_population(_frame(), catalogue(config))
    assert population.ok, population.errors
    words = [f"{c.name}: {describe_definition(c.name, c.definition)}" for c in population.criteria]
    details = [d for exit_ in participant_flow(population).exits for d in exit_.details]
    assert details == words
    assert _criteria_table(population)["Definition"].tolist() == [
        describe_definition(c.name, c.definition) for c in population.criteria
    ]
