"""
Today's criteria, pinned before the criterion kinds catalogue (M5b step 6): the fingerprint of every definition, of the
population's specification and rows, the recorded blocks and their replay, and the words of every shape of rule.

A stored definition is never rewritten: its fingerprint keys the population memos and is written in every population
block of a trace, which the trace chain hashes. These values must not change when the code that reads a definition
does; only the three texts the catalogue's commit 5 corrects may.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.criteria import describe_definition
from core.fingerprint import ensure_row_ids
from core.lineage import canonical_json
from core.population import active_criteria, catalogue, evaluate_population
from manage.reproduction_manager import apply_steps

TUTORIAL = Path(__file__).resolve().parents[2] / "assets" / "tutorial"

# every shape of rule, one family each way of combining them: listed rows with what an outlier method records, frozen
# bounds on two columns (one side open), and a family of rules that fire together (a numeric rule with a side open)
EDGE = {
    "mask_families": {
        "outliers": {
            "operator": "any",
            "Flagged rows": {
                "row_ids": [[2, 3], [9, 9]],
                "description": "Rows an outlier method flagged.",
                "method": "isolation_forest",
                "params": {"contamination": 0.05},
                "columns": ["x", "y"],
            },
            "Frozen bounds": {"bounds": {"x": [0.0, 100.0], "y": [None, 50.0]}},
        },
        "two_or_more": {
            "operator": {"threshold": 2},
            "x high": {"numeric": False, "expression": "df['x'] > 70"},
            "y": {"numeric": True, "lower_bound": 10.0, "upper_bound": None, "strategy": "exclude"},
            "z top": {"numeric": False, "expression": "df['z'] >= 4"},
        },
    }
}


@pytest.fixture(scope="module")
def tutorial() -> tuple[pd.DataFrame, dict]:
    from core.state import ParseOptions
    from ingest.loaders import read_table

    data = (TUTORIAL / "tutorial_cohort.csv").read_bytes()
    cohort = ensure_row_ids(read_table(data, "tutorial_cohort.csv", ParseOptions(dayfirst=True)).df)
    return cohort, json.loads((TUTORIAL / "tutorial_config.json").read_text(encoding="utf-8"))


def _edge_frame() -> pd.DataFrame:
    nan = np.nan  # a missing value never fires a criterion
    return ensure_row_ids(
        pd.DataFrame(
            {
                "x": [10.0, 55.0, 120.0, 75.0, nan, 30.0, 80.0, 95.0, -5.0, 60.0, 72.0, 40.0],
                "y": [5.0, 20.0, 45.0, 8.0, 30.0, nan, 60.0, 12.0, 25.0, 9.0, 35.0, 15.0],
                "z": [0.0, 4.0, 1.0, 4.0, 2.0, 4.0, 0.0, 3.0, 4.0, 1.0, 4.0, 2.0],
            }
        )
    )


def _pinned(frame: pd.DataFrame, config: dict) -> tuple[list[str], dict, list[int]]:
    criteria = catalogue(config)
    population = evaluate_population(
        frame, active_criteria(config, [c.id for c in criteria]), data_fp="d"
    )
    assert population.ok, population.errors
    return [c.fingerprint for c in criteria], population.to_trace(), list(population.included)


def _criteria_hash(block: dict) -> str:
    return hashlib.sha256(canonical_json(block["criteria"]).encode()).hexdigest()[:16]


def _replayed(frame: pd.DataFrame, block: dict) -> tuple[list[str], list[int]]:
    steps = [
        {
            "function": "population_definition",
            "params": {},
            "description": "population",
            "population": block,
        },
        {
            "function": "freeze_population",
            "params": {},
            "description": "freeze",
            "population": block,
        },
    ]
    result = apply_steps(frame, steps)
    assert result.error is None and result.df is not None
    return result.warnings, list(result.df.index)


def test_the_tutorial_criteria_keep_their_fingerprints(tutorial: tuple[pd.DataFrame, dict]) -> None:
    cohort, config = tutorial
    fingerprints, block, included = _pinned(cohort, config)
    assert fingerprints == [
        "5c38974be4baeff3",
        "a09de96799b18210",
        "3c91acca2b6890aa",
        "b109da32aedbfb4a",
        "d55300b467074837",
        "b1ac7b57a1fcb03c",
        "183bc75379a977a1",
        "b39092a25ad614aa",
        "1cf945a99f432ae3",
        "83e0b87c7ccd7ff8",
        "2646ef1a5083d274",
        "9884229e59a2c8ac",
    ]
    assert (block["spec_fp"], block["rows_fp"]) == ("c280584cb53a91b3", "713336d28c6e7391")
    assert (len(included), len(cohort)) == (928, 1206)
    assert _criteria_hash(block) == "3f64ee7a93326ada"  # the definitions as the block records them
    warnings, rows = _replayed(cohort, block)
    assert warnings == [] and rows == included


def test_every_shape_of_rule_keeps_its_fingerprint() -> None:
    frame = _edge_frame()
    fingerprints, block, included = _pinned(frame, EDGE)
    assert fingerprints == ["2d92b76ad9ad38d5", "a75bdd632bb2710b", "4daf2e515c84c865"]
    assert (block["spec_fp"], block["rows_fp"]) == ("efc22f9600b4d80b", "f7748196adba7b48")
    assert included == [0, 1, 4, 5, 7, 11]
    assert _criteria_hash(block) == "785d835323144858"
    warnings, rows = _replayed(frame, block)
    assert warnings == [] and rows == included


@pytest.mark.parametrize(
    ("definition", "words"),
    [
        ({"numeric": False, "expression": "df['x'] > 70"}, "df['x'] > 70"),
        (
            {"numeric": True, "lower_bound": 18, "upper_bound": 85, "strategy": "include"},
            "age within [18, 85]",
        ),
        (
            {"numeric": True, "lower_bound": 18, "upper_bound": None, "strategy": "exclude"},
            "age < 18",
        ),
        (
            {"numeric": True, "lower_bound": None, "upper_bound": 85, "strategy": "exclude"},
            "age > 85",
        ),
        ({"bounds": {"age": [18, 90]}}, "age outside [18, 90]"),
        # corrected by the catalogue's commit 5: listed rows name their method, a bound open on one side reads as
        # one, and a family of rules that fire together gives each rule's definition
        (
            {"row_ids": [[2, 4]], "method": "zscore", "columns": ["x"]},
            "3 listed rows (flagged by z-score)",
        ),
        ({"row_ids": [[2, 4]]}, "3 listed rows"),
        (
            {"row_ids": [[1, 1]], "method": "Isolation Forest"},
            "1 listed rows (flagged by Isolation Forest)",
        ),
        ({"bounds": {"age": [None, 85]}}, "age > 85"),
        ({"bounds": {"age": [18, None], "ldl": [None, None]}}, "age < 18; ldl: no bound"),
        (
            {
                "threshold": 2,
                "members": {
                    "old": {"numeric": False, "expression": "df['age'] > 80"},
                    "age": {"numeric": True, "lower_bound": 18, "upper_bound": 85},
                },
            },
            "at least 2 of: old (df['age'] > 80); age (age within [18, 85])",
        ),
    ],
)
def test_todays_words_for_every_shape_of_rule(definition: dict, words: str) -> None:
    assert describe_definition("age", definition) == words
