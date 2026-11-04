"""
A criterion's expression in three-valued logic (core.expressions.kleene, RULES_VERSION 2): a missing value never fires
a rule, written with "!=" or "~" too; a test inside a call is read as Python reads it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.criteria import CriterionError, criterion_mask, criterion_unknown

DF = pd.DataFrame(
    {
        "h": [50.0, np.nan, 150.0, 120.0],
        "s": ["x", None, "y", "withdrawn"],
        "c": pd.Categorical(["a", None, "b", "a"]),
        "d1": ["2021-01-01", None, "2021-03-01", "2021-05-01"],
        "d2": ["2021-02-01", "2021-02-01", None, "2021-04-01"],
        "flag": pd.array([True, None, False, True], dtype="boolean"),
    }
)


def _fires(expression: str) -> list[int]:
    mask = criterion_mask(DF, "rule", {"expression": expression})
    return [int(i) for i in DF.index[mask]]


@pytest.mark.parametrize(
    ("expression", "rows"),
    [
        ("~(df['h'] >= 100)", [0]),  # fired on the missing height too
        ("df['s'] != 'withdrawn'", [0, 2]),  # and on the missing status
        ("~df['s'].isin(['x'])", [2, 3]),
        ("~df['h'].between(100, 200)", [0]),
        ("df['c'] != 'a'", [2]),
        ("~df['s'].str.startswith('w')", [0, 2]),
        ("pd.to_datetime(df['d1']) > pd.to_datetime(df['d2'])", [3]),
        ("~df['flag']", [2]),
        ("(df['h'] > 100) & (df['s'] != 'y')", [3]),
    ],
)
def test_a_missing_value_never_fires_a_rule_written_with_not_equal_or_not(
    expression: str, rows: list[int]
) -> None:
    assert _fires(expression) == rows


def test_a_comparison_of_two_tests_is_unknown_where_either_is() -> None:
    """Read as pandas reads them, the missing height and the missing status both gave False, so the two tests were
    "equal" on that row and the rule fired there."""
    assert _fires("(df['h'] > 100) == (df['s'] == 'x')") == []
    assert _fires("(df['h'] > 100) != (df['s'] == 'x')") == [0, 2, 3]


def test_a_rule_of_at_least_k_lacks_a_value_only_where_the_missing_rules_could_decide() -> None:
    """A row with one rule missing and the two others false cannot reach 2 of 3: it was counted as lacking a value,
    and the flow said so of a row that the known rules exclude."""
    frame = pd.DataFrame({"a": [np.nan, np.nan, 5.0, 5.0], "b": [0, 9, 9, 0], "c": [0, 0, 9, 0]})
    two = {
        "threshold": 2,
        "members": {
            "A": {"expression": "df['a'] > 3"},
            "B": {"expression": "df['b'] > 3"},
            "C": {"expression": "df['c'] > 3"},
        },
    }
    assert criterion_mask(frame, "two", two).tolist() == [False, False, True, False]
    assert criterion_unknown(frame, "two", two).tolist() == [False, True, False, False]
    numeric = {"numeric": True, "column": "a", "lower_bound": 3, "strategy": "include"}
    assert criterion_unknown(frame, "a", numeric).tolist() == [True, True, False, False]


def test_unknown_or_true_is_true_and_a_rule_may_ask_for_missing_values() -> None:
    assert _fires("df['h'].isna() | (df['h'] > 130)") == [1, 2]
    assert _fires("df['h'].isna()") == [1]


def test_a_test_inside_a_call_is_read_as_python_reads_it() -> None:
    """np.where, fillna and astype decide what a missing value gives there, and so does a comparison's fill_value."""
    assert _fires("np.where(df['h'] > 100, 1, 0) == 1") == [2, 3]
    assert _fires("df['h'].fillna(0) < 100") == [0, 1]
    assert _fires("(df['h'] > 100).astype(int) == 0") == [0, 1]
    assert _fires("df['h'].lt(100, fill_value=0)") == [0, 1]


def test_an_expression_is_still_checked_as_written() -> None:
    with pytest.raises(CriterionError, match="expression rejected"):
        criterion_mask(DF, "rule", {"expression": "known_compare(df['h'], '>', 1)"})
    with pytest.raises(CriterionError, match="expression rejected"):
        criterion_mask(DF, "rule", {"expression": "known_between(df['h'], 1, 2)"})


@pytest.mark.parametrize(
    ("written", "spelt"),
    [
        ("~(df['h'] >= 100)", "~df['h'].ge(100)"),
        ("~(df['h'] >= 100)", "np.logical_not(df['h'] >= 100)"),
        ("~(df['h'] >= 100)", "(df['h'] >= 100) == False"),
        ("~(df['h'] >= 100)", "(df['h'] >= 100) != True"),
        ("df['h'] >= 100", "(df['h'] >= 100) == True"),
        ("~df['h'].between(100, 200)", "df['h'].between(100, 200) == False"),
        ("df['h'] != 150", "df['h'].ne(150)"),
        ("df['h'] < 100", "df['h'].lt(100)"),
        ("df['s'] != 'withdrawn'", "df['s'].ne('withdrawn')"),
        ("df['c'] != 'a'", "df['c'].ne('a')"),
        ("~df['s'].isin(['x'])", "df['s'].isin(['x']) != True"),
        ("df['s'].str.lower() != 'y'", "df['s'].str.lower().ne('y')"),
        ("~df['s'].str.startswith('w')", "df['s'].str.startswith('w') == False"),
        ("(df['h'] > 100) & (df['s'] != 'y')", "np.logical_and(df['h'] > 100, df['s'].ne('y'))"),
        ("(df['h'] > 100) | (df['s'] == 'x')", "np.logical_or(df['h'].gt(100), df['s'].eq('x'))"),
        ("(df['h'] > 100) ^ (df['s'] == 'x')", "np.logical_xor(df['h'] > 100, df['s'] == 'x')"),
        # a test compared with True or False by its method: they fired on the missing height
        ("~(df['h'] >= 100)", "(df['h'] >= 100).eq(False)"),
        ("~(df['h'] >= 100)", "(df['h'] >= 100).ne(True)"),
        ("df['h'] >= 100", "(df['h'] >= 100).eq(True)"),
        # by keyword too: they fired on the missing height
        ("~(df['h'] >= 100)", "~df['h'].ge(other=100)"),
        ("~(df['h'] >= 100)", "~df['h'].ge(100, axis=0)"),
        ("~df['h'].between(100, 200)", "~df['h'].between(left=100, right=200)"),
        ("~df['h'].between(100, 200)", "~df['h'].between(100, right=200, inclusive='both')"),
    ],
)
def test_one_reading_however_the_test_is_spelt(written: str, spelt: str) -> None:
    """Methods, numpy's logic and a test compared with True or False read as the operators they stand for: two
    spellings of one rule select the same rows (and give the same Domain Validity)."""
    assert _fires(spelt) == _fires(written)


def test_one_domain_validity_however_the_rule_is_spelt() -> None:
    from core.criteria import rule_checks

    def counted(expression: str) -> tuple[int, int]:
        (check,) = rule_checks(DF, "rule", {"expression": expression})
        return int(check.checked.sum()), int(check.fired.sum())

    assert counted("~df['h'].ge(100)") == counted("df['h'] < 100") == (3, 1)


def test_between_is_its_two_comparisons_and_a_missing_bound_is_unknown() -> None:
    """Unknown where the value or a bound is missing, unless the other comparison is False: a visit before the start
    of a follow-up still open is outside it, a visit after its start is not known to be."""
    df = pd.DataFrame(
        {
            "x": [5.0, 5.0, 5.0, 50.0, np.nan],
            "lo": [1.0, np.nan, 10.0, 1.0, 1.0],
            "hi": [10.0, 10.0, np.nan, np.nan, 10.0],
        }
    )

    def fires(expression: str) -> list[int]:
        return [int(i) for i in df.index[criterion_mask(df, "rule", {"expression": expression})]]

    assert fires("~df['x'].between(df['lo'], df['hi'])") == [2]
    assert fires("(df['x'] < df['lo']) | (df['x'] > df['hi'])") == [2]
    assert fires("df['x'].between(df['lo'], df['hi'])") == [0]
    assert fires("~df['x'].between(5, 10, inclusive='neither')") == [0, 1, 2, 3]
    assert fires("~df['x'].between(5, 10, 'left')") == [3]


def test_a_population_recorded_under_the_earlier_rules_replays_with_the_rows_it_kept() -> None:
    """Recorded under version 1 (a block without its version), "!=" kept the missing status: the replay keeps those
    rows, and says that the rules changed, not the data. Recorded under today's rules, its criteria are read again."""
    from core.fingerprint import ensure_row_ids, ids_to_ranges
    from core.population import active_criteria, catalogue, evaluate_population
    from manage.reproduction_manager import apply_steps

    frame = ensure_row_ids(DF.copy())
    config = {
        "mask_families": {
            "study_inclusion": {
                "operator": "all",
                "Not withdrawn": {"expression": "df['s'] != 'withdrawn'"},
            }
        }
    }
    criteria = active_criteria(config, [c.id for c in catalogue(config)])
    block = evaluate_population(frame, criteria, data_fp="d").to_trace()
    assert block["rules_version"] == 2 and list(evaluate_population(frame, criteria).included) == [
        0,
        2,
    ]

    def replayed(population: dict) -> tuple[list[str], list[int]]:
        steps = [
            {
                "function": "freeze_population",
                "params": {},
                "description": "freeze",
                "population": population,
            }
        ]
        result = apply_steps(frame, steps)
        assert result.error is None and result.df is not None
        return result.warnings, list(result.df.index)

    assert replayed(block) == ([], [0, 2])
    earlier = {k: v for k, v in block.items() if k != "rules_version"} | {
        "included": ids_to_ranges([0, 1, 2])
    }
    warnings, rows = replayed(earlier)
    assert rows == [0, 1, 2]
    assert len(warnings) == 1 and "recorded under rules version 1" in warnings[0]
    assert "the rows it kept then are used" in warnings[0]
    # more than 20,000 rows kept: the step holds those its criteria excluded, and the table's other rows are used
    large = {k: v for k, v in block.items() if k not in ("rules_version", "included")} | {
        "excluded_by": {criteria[0].id: ids_to_ranges([3])}
    }
    warnings, rows = replayed(large)
    assert rows == [0, 1, 2] and "the rows it kept then are used" in warnings[0]
    # neither: a step that changes the data on it is refused; a step that only checks it says what it reads
    neither = {k: v for k, v in large.items() if k != "excluded_by"}
    freeze = {
        "function": "freeze_population",
        "params": {},
        "description": "",
        "population": neither,
    }
    refused = apply_steps(frame, [freeze])
    assert refused.df is None and "holds neither the rows it kept nor those it excluded" in (
        refused.error or ""
    )
    check = {
        "function": "population_definition",
        "params": {},
        "description": "",
        "population": neither,
    }
    checked = apply_steps(frame, [check])
    assert checked.error is None and "which may keep other rows" in checked.warnings[0]


def test_a_population_read_otherwise_on_the_table_it_was_recorded_on_replays_with_the_rows_it_kept() -> (
    None
):
    """A study recorded under version 2 before a spelling's reading changed (a test's fillna(True) on a float column:
    1,110 rows of the tutorial cohort then, 1,171 since): the same definitions on the same table, so the reading
    changed, not the data. The replay keeps the rows the step holds and says why; on other data, it reads the criteria
    again and says that the data differ."""
    from core.fingerprint import ensure_row_ids, frame_fingerprint, ids_fingerprint, ids_to_ranges
    from core.population import active_criteria, catalogue, evaluate_population
    from manage.reproduction_manager import apply_steps

    frame = ensure_row_ids(DF.copy())
    config = {
        "mask_families": {
            "study_inclusion": {
                "operator": "all",
                "Not withdrawn": {"expression": "df['s'] != 'withdrawn'"},
            }
        }
    }
    criteria = active_criteria(config, [c.id for c in catalogue(config)])
    block = evaluate_population(frame, criteria, data_fp=frame_fingerprint(frame)).to_trace()
    assert block["rules_version"] == 2
    then = block | {"included": ids_to_ranges([0, 1, 2]), "rows_fp": ids_fingerprint([0, 1, 2])}

    def freeze(population: dict):
        step = {
            "function": "freeze_population",
            "params": {},
            "description": "freeze",
            "population": population,
        }
        result = apply_steps(frame, [step])
        assert result.error is None and result.df is not None
        return result.warnings, list(result.df.index)

    warnings, rows = freeze(then)
    assert rows == [0, 1, 2] and len(warnings) == 1
    assert (
        "the same definitions are read otherwise now. The rows it kept then are used" in warnings[0]
    )
    warnings, rows = freeze(then | {"data_fp": "another table"})
    assert rows == [0, 2] and "re-evaluated on the replayed data keeps other rows" in warnings[0]
    assert freeze(block) == ([], [0, 2])  # read as recorded: nothing to say


def test_an_expression_that_says_what_a_missing_value_gives_is_read_as_written() -> None:
    """A text test's na= and a missing value in an isin list are the expression deciding, as fillna is."""
    assert _fires("~df['s'].str.contains('with', na=False)") == [0, 1, 2]
    assert _fires("df['s'].str.contains('with', na=True)") == [1, 3]
    assert _fires("df['s'].isin(['x', None])") == [0, 1]
    assert _fires("df['s'].isin(['x', np.nan])") == [0, 1]
    assert _fires("~df['s'].isin(['x', None])") == [2, 3]


@pytest.mark.parametrize(
    "missing",
    [np.nan, pd.NA],
    ids=["float64", "Int64"],
)
def test_a_test_s_fillna_and_a_missing_value_in_an_isin_list_decide_on_every_dtype(missing) -> None:
    """pandas' comparison of a missing float gives False, so (x >= 30).fillna(True) filled nothing on float64 and kept
    the missing values on Int64; np.nan in an isin list found a float's missing values, not an Int64's. Read in
    three-valued logic, each says the same on every dtype."""
    values = [50, missing, 20, 2]
    df = pd.DataFrame({"x": pd.array(values, dtype="Int64") if missing is pd.NA else values})

    def fires(expression: str) -> list[int]:
        return [int(i) for i in df.index[criterion_mask(df, "rule", {"expression": expression})]]

    assert fires("(df['x'] >= 30).fillna(True)") == [0, 1]
    assert fires("(df['x'] >= 30).fillna(value=False)") == [0]
    assert fires("~(df['x'] >= 30).fillna(True)") == [2, 3]
    for marker in ("None", "np.nan", "pd.NA", "float('nan')"):
        assert fires(f"df['x'].isin([2, {marker}])") == [1, 3], marker
        assert fires(f"~df['x'].isin([2, {marker}])") == [0, 2], marker
    # what the expression decides is known: no row lacks a value it reads
    assert not criterion_unknown(df, "rule", {"expression": "(df['x'] >= 30).fillna(True)"}).any()


@pytest.mark.parametrize("dtype", ["str", "object", "category"])
def test_a_text_test_s_fillna_decides_on_every_text_dtype(dtype: str) -> None:
    """(status != 'withdrawn').fillna(False) fired on the missing statuses of a str or object column: pandas' "!=" is
    True there, so the fill saw nothing."""
    df = pd.DataFrame({"s": pd.Series(["a", None, "withdrawn"], dtype=dtype)})

    def fires(expression: str) -> list[int]:
        return [int(i) for i in df.index[criterion_mask(df, "rule", {"expression": expression})]]

    assert fires("(df['s'] != 'withdrawn').fillna(False)") == [0]
    assert fires("(df['s'] != 'withdrawn').fillna(True)") == [0, 1]
    assert fires("df['s'].isin(['a', None])") == [0, 1]


def test_a_study_written_under_the_earlier_rules_says_what_reads_otherwise() -> None:
    """Under version 1, "!=" kept the missing status (an inclusion criterion met) and "~" flagged the missing height (an
    exclusion criterion fired): read now, the first excludes one row more, the second one fewer."""
    from core.population import rules_changes

    config = {
        "mask_families": {
            "study_inclusion": {
                "operator": "all",
                "Not withdrawn": {"expression": "df['s'] != 'withdrawn'"},
            },
            "clinical_anomalies": {
                "operator": "any",
                "Short": {"expression": "~(df['h'] >= 100)"},
                "Isna": {"expression": "df['h'].isna()"},  # read alike: never listed
            },
        }
    }
    found = {c.criterion.name: (c.more, c.fewer) for c in rules_changes(DF, config)}
    assert found == {"Not withdrawn": (1, 0), "Short": (0, 1)}


def test_the_study_s_population_is_counted_now_and_as_written() -> None:
    """A criterion's own count is not the study's: "not pregnant" excludes 3 more rows than before, the two men whose
    pregnancy is not recorded among them, but "women" excluded the men already: the population loses one row."""
    from core.population import rules_changes, rules_population

    frame = pd.DataFrame(
        {
            "sex": ["F", "F", "M", "M", "F"],
            "pregnant": ["No", "Yes", None, None, None],
        }
    )
    config = {
        "mask_families": {
            "study_inclusion": {
                "operator": "all",
                "Women": {"expression": "df['sex'] == 'F'"},
                "Not pregnant": {"expression": "df['pregnant'] != 'Yes'"},
            }
        }
    }
    found = {c.criterion.name: (c.more, c.fewer) for c in rules_changes(frame, config)}
    assert found == {"Not pregnant": (3, 0)}
    # the woman whose pregnancy is missing (row 4) is the one row the study keeps no more
    assert rules_population(frame, config) == (1, 2)


def test_a_family_counting_its_rules_is_read_as_then_too() -> None:
    """A family read "at least 2 of" its rules is one threshold rule, whose members are criteria: an expression member
    was read as Python reads it then. Two measurements missing together fired "~(... >= ...)" twice under version 1,
    and fire neither now: the notice said nothing."""
    from core.population import rules_changes

    frame = pd.DataFrame(
        {
            "height_cm": [170.0, np.nan, 80.0, np.nan, 175.0],
            "weight_kg": [70.0, np.nan, 10.0, 60.0, np.nan],
            "age": [40, 50, 60, 130, 70],
        }
    )
    rules = {
        "Short": {"expression": "~(df['height_cm'] >= 100)"},
        "Light": {"expression": "~(df['weight_kg'] >= 20)"},
        "Old": {"expression": "df['age'] > 120"},
    }
    family = {"mask_families": {"clinical_anomalies": {"operator": {"threshold": 2}, **rules}}}
    (change,) = rules_changes(frame, family)
    # then, a missing value made "~(... >= ...)" fire: row 1 (both missing) and row 3 (a missing height and an age
    # over 120) reached 2 of 3; row 2 (both low) fires now as then; row 4 (one missing, nothing else) never did
    assert (change.more, change.fewer) == (0, 2) and change.criterion.family == "clinical_anomalies"
    by_hand = {
        "mask_families": {
            "clinical_anomalies": {
                "operator": "any",
                "Two of three": {"threshold": 2, "members": rules},
            }
        }
    }
    assert [(c.criterion.name, c.more, c.fewer) for c in rules_changes(frame, by_hand)] == [
        ("Two of three", 0, 2)
    ]
    # no expression at any depth: read alike, never listed
    from core.criteria import criterion_mask, earlier_mask

    ranges = {
        "Tall": {"numeric": True, "column": "height_cm", "lower_bound": 172, "strategy": "include"},
        "Heavy": {"numeric": True, "column": "weight_kg", "lower_bound": 65, "strategy": "include"},
    }
    two = {"threshold": 1, "members": ranges}
    assert criterion_mask(frame, "Two", two).tolist() == [True, False, False, False, True]
    assert earlier_mask(frame, "Two", two) is None
    numeric = {"mask_families": {"clinical_anomalies": {"operator": "any", "Two": two}}}
    assert rules_changes(frame, numeric) == []


@pytest.mark.parametrize("dtype", ["Int64", "float64", "object"])
def test_a_keyword_spelling_reads_as_its_positional_one(dtype: str) -> None:
    """isin(values=[..., None]) left the missing row unknown where isin([..., None]) finds it, and a test's
    fillna(True, inplace=False) or fillna(value=True, axis=0) filled nothing on a float column, where fillna(True)
    fills: one reading for one meaning."""
    values = ["150", None, "90", "200"] if dtype == "object" else [150, None, 90, 200]
    df = pd.DataFrame({"h": pd.array(values, dtype=dtype)})
    listed = "['150', None]" if dtype == "object" else "[150, None]"

    def fires(expression: str) -> list[int]:
        return [int(i) for i in df.index[criterion_mask(df, "rule", {"expression": expression})]]

    assert fires(f"df['h'].isin(values={listed})") == fires(f"df['h'].isin({listed})") == [0, 1]
    assert fires(f"~df['h'].isin(values={listed})") == [2, 3]
    if dtype != "object":
        test = "(df['h'] >= 100)"
        assert fires(f"{test}.fillna(True)") == [0, 1, 3]
        assert (
            fires(f"{test}.fillna(True, inplace=False)")
            == fires(f"{test}.fillna(value=True, axis=0)")
            == [0, 1, 3]
        )


@pytest.mark.parametrize(
    "spelling",
    [
        "df['pregnant'] != 'Yes'",
        "df['pregnant'] == 'No'",
        "df['pregnant'].isin(['No'])",
        "(df['pregnant'] != 'Yes').fillna(False)",
        "(df['pregnant'] == 'No').fillna(False)",
        "df['pregnant'].notna() & (df['pregnant'] != 'Yes')",
        "~df['pregnant'].isin(['Yes', None])",
        "df['pregnant'].fillna('unknown') == 'No'",
        "~df['pregnant'].str.contains('Yes', na=True)",
    ],
)
def test_the_flow_counts_the_rows_lacking_a_value_however_the_criterion_is_written(
    spelling: str,
) -> None:
    """Pregnancy recorded for the 500 women only: every spelling of "Not pregnant" excludes the same 508 rows, and the
    flow said 500 of them lack a value for three spellings, 0 for those that say what a missing value gives, though
    the 500 lack it all the same (STROBE item 13b). The criterion editor's advice still honours what they say."""
    from core.criteria import criterion_lacking
    from core.population import active_criteria, catalogue, evaluate_population

    n = 1000
    pregnant = np.where(np.arange(n) < 8, "Yes", np.where(np.arange(n) < 500, "No", None))
    frame = pd.DataFrame({"pregnant": pregnant})
    config = {
        "mask_families": {
            "study_inclusion": {"operator": "all", "Not pregnant": {"expression": spelling}}
        }
    }
    criteria = active_criteria(config, [c.id for c in catalogue(config)])
    (count,) = evaluate_population(frame, criteria).counts
    assert (count.sequential, count.missing) == (508, 500)
    said = criterion_unknown(frame, "Not pregnant", {"expression": spelling})
    assert int(said.sum()) == (
        0
        if "fillna" in spelling or "notna" in spelling or "None" in spelling or "na=" in spelling
        else 500
    )
    known_false = {"expression": "(df['age'] >= 18) & (df['consent'] == 'Yes')"}
    adults = pd.DataFrame({"age": [10, 30, np.nan, 30], "consent": [None, None, "Yes", "No"]})
    assert criterion_lacking(adults, "c", known_false).tolist() == [
        False,
        True,
        True,
        False,
    ]  # a minor is settled


def test_a_rule_of_at_least_k_reads_each_rule_by_its_value() -> None:
    """A rule counting "at least 1 of" its rules read an expression rule as unsettled wherever a column it reads is
    missing, even where its value is known: (a > 1) | (b > 2) is true where b is 5, whatever a."""
    from core.criteria import criterion_lacking

    frame = pd.DataFrame(
        {"a": [np.nan, np.nan, 0.0], "b": [5.0, 0.0, 0.0], "w": [10.0, 10.0, 10.0]}
    )
    either = {"expression": "(df['a'] > 1) | (df['b'] > 2)"}
    one = {"threshold": 1, "members": {"A": either, "B": {"expression": "df['w'] >= 20"}}}
    assert criterion_mask(frame, "one", one).tolist() == [True, False, False]
    assert criterion_lacking(frame, "one", one).tolist() == [False, True, False]
    filled = {
        "threshold": 1,
        "members": {"A": {"expression": "(df['a'] > 1).fillna(False)"}, "B": one["members"]["B"]},
    }
    assert criterion_lacking(frame, "filled", filled).tolist() == [
        True,
        True,
        False,
    ]  # as A alone reads it
