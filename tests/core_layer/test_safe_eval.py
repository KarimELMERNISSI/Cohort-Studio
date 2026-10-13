"""The expression evaluator accepts the real corpus (configs, prompts, UI examples) and rejects escapes."""

import ast

import numpy as np
import pandas as pd
import pytest

from core.expressions import (
    UnsafeExpressionError,
    column_references,
    evaluate,
    evaluate_tree,
    safe_format,
    validate,
)


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 25, 70, np.nan],
            "sex": ["male", "female", "male", "female"],
            "status": ["Active", "Pending", "Closed", "Active"],
            "systolic": [120, 150, 135, 90],
            "diastolic": [80, 95, 85, 60],
            "Weight": [50.0, 80.0, 90.0, 60.0],
            "Height": [160.0, 180.0, 170.0, 150.0],
            "visit": ["01/02/2020", "15/06/2021", "31/12/2019", "05/05/2022"],
            "dob": ["01/02/1990", "15/06/1950", "31/12/2010", "05/05/2000"],
            "Lean mass": [40.0, 60.0, 65.0, 45.0],
        }
    )


ALLOW_CORPUS = [
    "df['age'] < 18",
    "(df['systolic'] > 140) | (df['diastolic'] > 90)",
    "df['status'].isin(['Active', 'Pending'])",
    "df['age'].isna()",
    "df['sex']!='male'",
    "(df['Weight'] / ((df['Height']/100) ** 2)) > 60",
    "abs(df['Weight'] - (df['Lean mass'] + df['Lean mass'])) > 0.10 * df['Weight']",
    "(pd.to_datetime(df['visit'], format='%d/%m/%Y') - pd.to_datetime(df['dob'], format='%d/%m/%Y')) > pd.Timedelta(days=365*18)",
    "((pd.to_datetime(df['visit'], dayfirst=True) - pd.to_datetime(df['dob'], dayfirst=True)).dt.days / 365.25).round()",
    "(pd.to_datetime(df['visit'], dayfirst=True, errors='coerce') - pd.to_datetime(df['dob'], dayfirst=True)) < pd.Timedelta(days=0)",
    "np.where(df['sex'] == 'male', df['Weight'] * 0.387, df['Weight'])",
    "np.sqrt(df['Weight']) + np.log(df['Height']) - np.exp(0)",
    "df['status'].str.contains('Act') & ~df['age'].isna()",
    "df['age'].between(18, 65)",
    "df['age'].fillna(0).astype('int64') + 1",
    "df[['age', 'systolic']].max(axis=1) if True else df['age']",
    "pd.to_numeric(df['age'], errors='coerce') > np.nanmedian(df['age'])",
    "df['age'].clip(lower=0, upper=100).round(1)",
]

DENY_CORPUS = [
    "__import__('os').system('id')",
    "().__class__.__mro__[1].__subclasses__()",
    "df.__class__",
    "getattr(df, 'to_pickle')",
    "open('x.txt')",
    "(lambda: 1)()",
    "[x for x in df['age']]",
    "f'{df}'",
    "df['age'].apply(lambda v: v)",
    "df.eval('age + 1')",
    "df.query('age > 1')",
    "df.to_pickle('x.pkl')",
    "pd.read_pickle('x.pkl')",
    "np.load('x.npy')",
    "os.system('id')",
    "df['age'].values",
    "df.iloc[0]",
    "exec('1')",
    "x := 1",
    "b'bytes'",
    "pd.DataFrame.__dict__",
    "df['age'].str._validate",
]


@pytest.mark.parametrize("expression", ALLOW_CORPUS)
def test_allow_corpus_evaluates(df: pd.DataFrame, expression: str) -> None:
    result = evaluate(expression, df)
    assert result is not None


@pytest.mark.parametrize("expression", DENY_CORPUS)
def test_deny_corpus_rejected(df: pd.DataFrame, expression: str) -> None:
    with pytest.raises(UnsafeExpressionError):
        evaluate(expression, df)


def test_results_match_pandas(df: pd.DataFrame) -> None:
    mask = evaluate("(df['systolic'] > 140) | (df['diastolic'] > 90)", df)
    assert mask.tolist() == [False, True, False, False]
    bmi = evaluate("df['Weight'] / ((df['Height']/100) ** 2)", df)
    assert bmi.iloc[0] == pytest.approx(50 / 1.6**2)
    age = evaluate(
        "((pd.to_datetime(df['visit'], dayfirst=True) - pd.to_datetime(df['dob'], dayfirst=True)).dt.days / 365.25).round()",
        df,
    )
    assert age.tolist() == [30.0, 71.0, 9.0, 22.0]


def test_builtins_are_restricted(df: pd.DataFrame) -> None:
    assert evaluate("abs(-3) + round(2.6) + max(1, 2)", df) == 8
    with pytest.raises(UnsafeExpressionError):
        evaluate("print('x')", df)
    with pytest.raises(UnsafeExpressionError):
        evaluate("__builtins__['open']", df)


def test_extra_locals_and_tree_api(df: pd.DataFrame) -> None:
    def safe_sub(a, b):
        return a - b

    tree = ast.parse("__safe_sub__(var_0, var_1) * 2", mode="eval")
    result = evaluate_tree(
        tree,
        df,
        extra_locals={"var_0": df["Weight"], "var_1": df["Lean mass"], "__safe_sub__": safe_sub},
    )
    assert result.tolist() == [20.0, 40.0, 50.0, 30.0]
    with pytest.raises(UnsafeExpressionError, match="Unknown name"):
        evaluate("var_9 + 1", df, extra_locals={"var_0": df["Weight"]})


def test_validate_reports_syntax_and_length() -> None:
    with pytest.raises(UnsafeExpressionError, match="Syntax error"):
        validate("df['age'] >")
    with pytest.raises(UnsafeExpressionError, match="empty"):
        validate("   ")
    with pytest.raises(UnsafeExpressionError, match="longer"):
        validate("1 + " * 2000 + "1")


def test_column_references() -> None:
    assert column_references("(df['a'] > 1) & df[['b', 'c']].sum(axis=1).gt(df['a'])") == {
        "a",
        "b",
        "c",
    }
    assert column_references("not an expression (") == set()


class TestSafeFormat:
    def test_substitutes_known_fields(self) -> None:
        assert (
            safe_format(
                "{initial_variable}_{method_applied}", initial_variable="age", method_applied="Z"
            )
            == "age_Z"
        )
        assert safe_format("plain_name") == "plain_name"

    def test_rejects_attribute_access_and_unknown_fields(self) -> None:
        with pytest.raises(ValueError, match="simple"):
            safe_format("{initial_variable.__class__}", initial_variable="age")
        with pytest.raises(ValueError, match="simple"):
            safe_format("{initial_variable!r}", initial_variable="age")
        with pytest.raises(ValueError, match="Unknown placeholder"):
            safe_format("{category}", initial_variable="age")
        with pytest.raises(ValueError):
            safe_format(None)  # type: ignore[arg-type]
