"""
A malformed rule is refused when a configuration is loaded, and never stops a page (M5b step 6).

Each rule below passed the upload check, then raised outside CriterionError when evaluated. The sidebar counts every
criterion of the configuration before each page, so every page failed until the session ended. A listed range was
turned into every row id it spans: [[0, 1000000000]] made the server allocate about 8 GB.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.attrition import describe_definition
from core.criteria import CriterionError, criterion_columns, criterion_mask, criterion_problems
from core.fingerprint import ensure_row_ids
from core.population import active_criteria, catalogue, configuration_problems, evaluate_population

OLD = {"numeric": False, "expression": "df['age'] > 80"}
MALFORMED = {
    "a bound in words": {"numeric": True, "column": "age", "lower_bound": "18 years"},
    "bounds of one value": {"bounds": {"age": [1]}},
    "bounds as a list": {"bounds": ["age", 1, 2]},
    "a range of three": {"row_ids": [[1, 2, 3]]},
    "a range in words": {"row_ids": "1-3"},
    "a threshold in words": {"threshold": "two", "members": {"old": OLD}},
    "rules as a list": {"threshold": 1, "members": ["old"]},
    "a malformed rule inside": {
        "threshold": 1,
        "members": {"young": {"numeric": True, "column": "age", "upper_bound": "ten"}},
    },
}


@pytest.fixture
def frame() -> pd.DataFrame:
    return ensure_row_ids(pd.DataFrame({"age": [10.0, 50.0, 90.0]}))


@pytest.mark.parametrize("name", MALFORMED)
def test_a_malformed_rule_is_refused_by_name_when_loaded(name: str) -> None:
    config = {"mask_families": {"checks": {"operator": "any", name: MALFORMED[name]}}}
    problems = configuration_problems(config)
    assert problems and all(f"'{name}'" in problem for problem in problems), problems
    assert criterion_problems(name, MALFORMED[name])


@pytest.mark.parametrize("name", MALFORMED)
def test_a_malformed_rule_raises_only_a_criterion_error(name: str, frame: pd.DataFrame) -> None:
    with pytest.raises(CriterionError):
        criterion_mask(frame, name, MALFORMED[name])
    assert isinstance(criterion_columns(name, MALFORMED[name]), tuple)  # never an error
    assert describe_definition(name, MALFORMED[name])  # never an error


def test_the_population_records_a_malformed_rule_instead_of_stopping(frame: pd.DataFrame) -> None:
    config = {"mask_families": {"checks": {"operator": "any", **MALFORMED}}}
    criteria = catalogue(config)
    population = evaluate_population(frame, active_criteria(config, [c.id for c in criteria]))
    assert not population.ok and len(population.errors) == len(MALFORMED)


def test_whatever_evaluates_today_still_loads(frame: pd.DataFrame) -> None:
    sound = {
        # a bound written as text
        "age": {"numeric": True, "lower_bound": "18", "upper_bound": 85, "strategy": "include"},
        "two and a half": {
            "threshold": 2.5,
            "members": {"old": OLD, "age": {"numeric": True, "lower_bound": 18}},
        },
        "listed as text": {"row_ids": [["0", "1"]]},
        # what float() and int() read as the mask does, and rules that fire on no row
        "true as a bound": {"numeric": True, "column": "age", "lower_bound": True},
        "bounds from false": {"bounds": {"age": [False, 60]}},
        "listed from true": {"row_ids": [[True, 2]]},
        "a threshold of true": {"threshold": True, "members": {"old": OLD}},
        "no bounds": {"bounds": None},
        "empty bounds": {"bounds": {}},
        "no rows": {"row_ids": None},
        "empty rows": {"row_ids": []},
    }
    assert configuration_problems({"mask_families": {"checks": {"operator": "any", **sound}}}) == []
    for name, definition in sound.items():
        assert criterion_mask(frame, name, definition).dtype == bool


def test_a_listed_range_larger_than_the_table_gives_the_same_rows(frame: pd.DataFrame) -> None:
    assert criterion_mask(frame, "huge", {"row_ids": [[0, 1_000_000_000]]}).all()
    assert criterion_mask(frame, "around", {"row_ids": [[-5, 1], [2, 2]]}).all()
    assert not criterion_mask(frame, "outside", {"row_ids": [[100, 200]]}).any()
    assert describe_definition("huge", {"row_ids": [[0, 1_000_000_000]]}).startswith(
        "1,000,000,001 listed rows"
    )


def test_listed_rows_are_found_without_listing_the_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """The work follows the table and the number of ranges: never the ranges' lengths (5,000 ranges of a billion ids
    on 1,000 rows listed five million ids), on sparse ids, an empty table or labels of another kind."""
    import core.criteria as criteria
    import core.fingerprint as fingerprint

    def never(*args: object) -> None:
        raise AssertionError("no range is turned into the ids it spans")

    monkeypatch.setattr(fingerprint, "ranges_to_ids", never)
    big = pd.DataFrame({"x": range(1000)})
    assert criteria.criterion_mask(big, "many", {"row_ids": [[0, 10**9]] * 5000}).all()
    sparse = pd.DataFrame({"x": [1, 2]}, index=[5, 10**8])
    assert criteria.criterion_mask(sparse, "sparse", {"row_ids": [[0, 10**9]]}).tolist() == [
        True,
        True,
    ]
    assert (
        criteria.criterion_mask(
            pd.DataFrame({"x": []}), "empty", {"row_ids": [[0, 10**9]]}
        ).tolist()
        == []
    )
    labels = pd.DataFrame({"x": [1, 2, 3]}, index=pd.Index([1, "two", 3.0], dtype=object))
    assert criteria.criterion_mask(labels, "labels", {"row_ids": [[0, 5]]}).tolist() == [
        True,
        False,
        False,
    ]


@pytest.mark.parametrize(
    "definition",
    [
        {"numeric": True, "column": "age", "lower_bound": 10**400},  # overflows a float
        {"row_ids": [[0, float("inf")]]},  # JSON allows Infinity: no whole number
        {"threshold": float("inf"), "members": {"old": OLD}},
    ],
)
def test_numbers_beyond_reach_are_refused_and_never_stop_the_check(definition: dict) -> None:
    assert criterion_problems("huge", definition)
    assert configuration_problems(
        {"mask_families": {"checks": {"operator": "any", "huge": definition}}}
    )


def test_every_page_opens_with_a_malformed_rule_in_the_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rule loaded before this check, or written by hand into the session: the sidebar shows it cannot be
    evaluated, and every data page still opens."""
    import os

    from streamlit.testing.v1 import AppTest

    import utils.data_paths as dp
    from ui import nav

    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    monkeypatch.setitem(os.environ, "CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(
        str(Path(__file__).resolve().parents[2] / "main.py"), default_timeout=300
    )
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "malformed_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_tutorial").click().run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    at.session_state["config"]["mask_families"]["clinical_anomalies"]["a bound in words"] = (
        MALFORMED["a bound in words"]
    )
    for page in ("validate", "explore", "enrich", "visualize"):
        at.switch_page(nav.page_path(page)).run()
        assert not at.exception, (page, [str(e.value)[:300] for e in at.exception])
    labels = [box.label for box in at.sidebar.checkbox]
    assert any(
        "a bound in words" in label and "cannot be evaluated" in label for label in labels
    ), labels
