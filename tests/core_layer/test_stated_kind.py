"""
A definition may state its kind (M5b step 6, decision 1). A kind registered without a recogniser, as M6's date order
and M9's concepts will be, is read by the "kind" a definition states. The five kinds of M5 are read by their keys,
whatever "kind" says, so that no definition that reads today reads otherwise, or is refused.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pandas as pd
import pytest

from core.attrition import participant_flow
from core.criteria import (
    CRITERION_KINDS,
    CriterionError,
    CriterionKind,
    criterion_columns,
    criterion_mask,
    criterion_problems,
    describe_definition,
    holds_row_list,
    is_text,
    rule_kind,
    shared_definition,
)
from core.fingerprint import ensure_row_ids
from core.population import (
    Criterion,
    catalogue,
    configuration_for_sharing,
    configuration_problems,
    criteria_from_trace,
    evaluate_population,
    family_masks,
)

LATER = {"kind": "date_order", "first": "admission", "then": "discharge"}  # as M6 will write one


def _later_mask(df: pd.DataFrame, name: str, definition: Mapping[str, Any]) -> pd.Series:
    """Fires where the first date follows the second; a missing date never fires."""
    first, then = (pd.to_datetime(df[definition[key]]) for key in ("first", "then"))
    return first > then


def _later_problems(name: str, definition: Mapping[str, Any]) -> list[str]:
    if all(isinstance(definition.get(key), str) for key in ("first", "then")):
        return []
    return [f"rule '{name}': a date order names two columns"]


@pytest.fixture
def date_order(monkeypatch: pytest.MonkeyPatch) -> CriterionKind:
    """A kind that definitions state, registered for one test in a copy of the catalogue."""
    return _register(monkeypatch)


def _register(monkeypatch: pytest.MonkeyPatch) -> CriterionKind:
    import core.criteria as criteria

    kinds = CRITERION_KINDS.copy()
    monkeypatch.setattr(criteria, "CRITERION_KINDS", kinds)
    # "kind" is the catalogue's own key: it is shared without being listed among the kind's keys
    return kinds.register(
        CriterionKind(
            "date_order",
            "Date order",
            mask=_later_mask,
            columns=lambda name, definition: (definition["first"], definition["then"]),
            describe=lambda name, definition: f"{definition['first']} after {definition['then']}",
            problems=_later_problems,
            keys={"first": is_text, "then": is_text, "description": is_text},
        )
    )


def _frame() -> pd.DataFrame:
    return ensure_row_ids(
        pd.DataFrame(
            {
                "admission": ["2024-01-05", "2024-02-10", None, "2024-03-01"],
                "discharge": ["2024-01-09", "2024-02-01", "2024-02-20", None],
                "age": [40.0, 70.0, 55.0, 90.0],
            }
        )
    )


def _config(definition: Mapping[str, Any]) -> dict[str, Any]:
    return {"mask_families": {"clinical_anomalies": {"operator": "any", "late": definition}}}


def test_a_kind_that_definitions_state_is_read_everywhere(date_order: CriterionKind) -> None:
    config = _config(LATER)
    assert rule_kind("late", LATER) is date_order
    assert configuration_problems(config) == []
    df = _frame()
    masks = family_masks(df, config["mask_families"]["clinical_anomalies"])
    assert masks.flag is not None and list(df.index[masks.flag]) == [
        1
    ]  # a missing date never fires
    (criterion,) = catalogue(config)
    population = evaluate_population(df, [criterion])
    assert population.ok and list(population.included) == [0, 2, 3]
    details = [d for exit_ in participant_flow(population).exits for d in exit_.details]
    assert details == ["late: excluded when admission after discharge"]
    assert criterion_columns("late", LATER) == ("admission", "discharge")
    # what states the kind is kept, shared and hashed with the rest of the definition
    assert not holds_row_list(LATER) and shared_definition("late", LATER) == LATER
    shared = configuration_for_sharing(config).config
    assert shared["mask_families"]["clinical_anomalies"]["late"] == LATER
    (recorded,) = criteria_from_trace(json.loads(json.dumps(population.to_trace())))
    assert recorded.definition == LATER and recorded.fingerprint == criterion.fingerprint
    unstated = {key: value for key, value in LATER.items() if key != "kind"}
    other = Criterion(criterion.id, criterion.family, criterion.kind, criterion.name, unstated)
    assert other.fingerprint != criterion.fingerprint
    # its own checks refuse what it cannot read, by name
    assert configuration_problems(_config({"kind": "date_order", "first": "admission"})) == [
        "family 'clinical_anomalies': rule 'late': a date order names two columns"
    ]


def test_without_its_registration_a_stated_kind_is_refused_and_named() -> None:
    """An older version reading a newer file refuses its rule, and says which kind it does not know."""
    (problem,) = configuration_problems(_config(LATER))
    assert problem == (
        "family 'clinical_anomalies': criterion 'late': unknown kind 'date_order'; the kinds are threshold, row_ids, "
        "bounds, numeric, expression"
    )
    with pytest.raises(CriterionError, match="unknown kind 'date_order'"):
        criterion_mask(_frame(), "late", LATER)
    assert describe_definition("late", LATER).startswith(
        "cannot be evaluated: criterion 'late': unknown kind 'date_order'"
    )
    assert configuration_for_sharing(_config(LATER)).unreadable == ("clinical_anomalies: late",)


@pytest.mark.parametrize(
    ("definition", "read_as"),
    [
        ({"kind": "numeric", "numeric": False, "expression": "df['age'] > 80"}, "expression"),
        ({"kind": "lab", "numeric": True, "lower_bound": 18}, "numeric"),
        ({"kind": "expression", "bounds": {"age": [18, 90]}}, "bounds"),
        ({"kind": ["date_order"], "expression": "df['age'] > 80"}, "expression"),
        # a kind registered later, named by a definition that holds the keys of the five: read by its keys
        ({"kind": "date_order", "numeric": False, "expression": "df['age'] > 80"}, "expression"),
    ],
)
def test_the_five_kinds_of_m5_are_read_by_their_keys_whatever_kind_says(
    definition: dict, read_as: str, date_order: CriterionKind
) -> None:
    assert rule_kind("age", definition).id == read_as
    assert criterion_problems("age", definition) == []


@pytest.mark.parametrize("stated", [["date_order"], {"id": "date_order"}, 3])
def test_a_kind_that_is_not_a_text_is_refused_without_a_type_error(stated: Any) -> None:
    with pytest.raises(CriterionError, match="unknown kind"):
        rule_kind("x", {"kind": stated})
    assert criterion_problems("x", {"kind": stated})


def test_a_recorded_population_replays_once_the_kind_it_names_is_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No definition that reads today reads otherwise later (decision 1): one holding the keys of the five and naming
    a kind that is not registered yet is recorded as one of the five, and replays as recorded once the kind is
    registered. Reading the stated kind first made it a date order, and its block no longer replayed."""
    definition = {"kind": "date_order", "numeric": False, "expression": "df['age'] > 80"}
    df = _frame()
    (criterion,) = catalogue(_config(definition))
    recorded = evaluate_population(df, [criterion])
    block = json.loads(json.dumps(recorded.to_trace()))
    _register(monkeypatch)  # a later version registers the kind
    (replayed,) = criteria_from_trace(block)
    again = evaluate_population(df, [replayed])
    assert again.ok and list(again.included) == list(recorded.included) == [0, 1, 2]
    assert again.rows_fingerprint == block["rows_fp"]
