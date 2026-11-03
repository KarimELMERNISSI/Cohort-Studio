"""
A criterion's expression in three-valued logic (core.expressions.kleene, RULES_VERSION 2): a missing value never fires
a rule, written with "!=" or "~" too; a test inside a call is read as Python reads it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.criteria import CriterionError, criterion_mask

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


def test_unknown_or_true_is_true_and_a_rule_may_ask_for_missing_values() -> None:
    assert _fires("df['h'].isna() | (df['h'] > 130)") == [1, 2]
    assert _fires("df['h'].isna()") == [1]


def test_a_test_inside_a_call_is_read_as_python_reads_it() -> None:
    """np.where, fillna and astype decide what a missing value gives there."""
    assert _fires("np.where(df['h'] > 100, 1, 0) == 1") == [2, 3]
    assert _fires("df['h'].fillna(0) < 100") == [0, 1]
    assert _fires("(df['h'] > 100).astype(int) == 0") == [0, 1]


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


def test_an_expression_that_says_what_a_missing_value_gives_is_read_as_written() -> None:
    """A text test's na= and a missing value in an isin list are the expression deciding, as fillna is."""
    assert _fires("~df['s'].str.contains('with', na=False)") == [0, 1, 2]
    assert _fires("df['s'].str.contains('with', na=True)") == [1, 3]
    assert _fires("df['s'].isin(['x', None])") == [0, 1]
    assert _fires("df['s'].isin(['x', np.nan])") == [0, 1]
    assert _fires("~df['s'].isin(['x', None])") == [2, 3]


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
