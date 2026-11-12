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


# --- the sizes an expression can ask for (a formula written by a language model is tried before anyone reads it) ---


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("9 ** 9 ** 8", "A number this large"),  # minutes of a busy server before
        ("df['age'] * 0 * 9 ** 9 ** 8", "A number this large"),
        ("df['age'] ** 2000", "An exponent larger than 1,000"),
        ("df['age'] ** (10 ** 6)", "An exponent larger than 1,000"),
        ("np.ones(10 ** 9)", "An array of more than 10,000,000 values"),
        ("np.full(2 * 10 ** 8, 1)", "An array of more than 10,000,000 values"),
        ("df['sex'].str.pad(100000)", "A width larger than 10,000 characters"),
        ("'x' * 10 ** 6", "A text repeated more than 10,000 times"),
        # a size given by its name, a shape of several dimensions, a text repeated in two steps
        ("np.ones(shape=10 ** 9)", "An array of more than 10,000,000 values"),
        ("np.full(shape=10 ** 12, fill_value=1)", "An array of more than 10,000,000 values"),
        ("np.zeros((10 ** 4, 10 ** 4))", "An array of more than 10,000,000 values"),
        ("df['sex'].str.pad(width=100000000)", "A width larger than 10,000 characters"),
        ("('x' * 10000) * 10000", "A text repeated more than 10,000 times"),
        ("('x' * 200 + 'y') * 200 * 2", "A text repeated more than 10,000 times"),
    ],
)
def test_an_expression_cannot_ask_for_huge_sizes(expression: str, message: str) -> None:
    with pytest.raises(UnsafeExpressionError, match=message):
        validate(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "df['age'] / (df['age'] / 100) ** 2",
        "71.3 * df['age'] ** 0.425 * df['age'] ** 0.725 / 10000",  # Du Bois
        "df['age'] ** (1 / 3)",
        "2 ** (df['age'] / 10)",
        "10 ** 6 * df['age']",
        "np.where(df['age'] > 2, 5.0, -2.0)",
        "df['sex'].str.pad(12)",
        "'-' * 20",
    ],
)
def test_clinical_formulas_keep_their_sizes(expression: str) -> None:
    validate(expression)


# --- an expression reads the table and never changes it (every change of the data goes through the trace) ---


@pytest.mark.parametrize(
    "expression",
    [
        "df['age'] == df.replace(70, 60, inplace=True)",
        "df['age'].isna() & (df.fillna(0, inplace=True) is None)",
        "df.where(df['age'] > 20, 0, inplace=True)",
        "df['age'].clip(0, 20, inplace=True)",
        "df['age'].fillna(0, inplace=1)",  # anything but False
        "df['age'].fillna(0, inplace=df['age'].notna().all())",
    ],
)
def test_an_expression_cannot_change_the_table_it_reads(df: pd.DataFrame, expression: str) -> None:
    """``df.replace(70, 60, inplace=True)`` inside a criterion changed the table the criterion was evaluated on,
    without a step of the trace, even when the criterion was then refused."""
    before = df.copy(deep=True)
    with pytest.raises(UnsafeExpressionError, match="never changes it"):
        evaluate(expression, df)
    assert df.equals(before)


def test_an_expression_gets_a_table_of_its_own(df: pd.DataFrame) -> None:
    """Whatever a method does in place, it does to the expression's own table, which shares the caller's values
    until one is written."""
    before = df.copy(deep=True)
    seen: list[pd.DataFrame] = []

    def keep(frame: pd.DataFrame) -> bool:
        seen.append(frame)
        frame.fillna({"age": 0}, inplace=True)
        frame.loc[frame.index[0], "sex"] = "changed"
        return True

    tree = ast.parse("keep(df)", mode="eval")
    assert evaluate_tree(tree, df, extra_locals={"keep": keep}) is True
    assert seen[0] is not df and seen[0]["sex"].iloc[0] == "changed"
    assert df.equals(before)
    # a new column, as ever
    assert evaluate("df['age'].fillna(0)", df).tolist() == [10.0, 25.0, 70.0, 0.0]
    # inplace=False changes nothing, and stored criteria are written with it: read as before
    assert evaluate("df['age'].fillna(0, inplace=False)", df).tolist() == [10.0, 25.0, 70.0, 0.0]
    assert df.equals(before)


# --- a size that is not written as a plain number, and a size known only when the expression is computed ---


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("df['sex'].str.pad(int(1e8))", "A width larger than 10,000 characters"),
        ("df['sex'].str.pad(width=abs(-100000000))", "A width larger than 10,000 characters"),
        ("df['sex'].str.zfill(max(100000000, 1))", "A width larger than 10,000 characters"),
        ("np.ones(int(1e12))", "An array of more than 10,000,000 values"),
        ("np.zeros((int(1e6), int(1e6)))", "An array of more than 10,000,000 values"),
        ("np.full(shape=round(1e12), fill_value=1)", "An array of more than 10,000,000 values"),
        ("np.ones(min(10 ** 9, 10 ** 10))", "An array of more than 10,000,000 values"),
        ("'x' * int(5e9)", "A text repeated more than 10,000 times"),
        ("'x' * float(20000)", "A text repeated more than 10,000 times"),
    ],
)
def test_a_size_passed_through_a_function_is_a_size(expression: str, message: str) -> None:
    """``int(1e8)`` is a call, not a number written in the expression: a width of a hundred million characters was
    accepted, and ``str.pad(int(2e5))`` built 12 million characters on 60 rows."""
    with pytest.raises(UnsafeExpressionError, match=message):
        validate(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "df['sex'].str.pad(int(12.0))",
        "df['sex'].str.zfill(max(4, 8))",
        "np.ones(int(4))",
        "np.full(shape=round(3.7), fill_value=1)",
        "df['age'].round(int(2))",
        "'-' * int(20)",
        "np.where(df['age'] > max(18, 21), 1, 0)",
    ],
)
def test_a_size_of_ordinary_use_passes_through_a_function(
    df: pd.DataFrame, expression: str
) -> None:
    evaluate(expression, df)


@pytest.mark.parametrize(
    "expression",
    [
        "df['sex'] * int(1e8)",
        "df['sex'] * 100000",
        "100000 * df['sex']",
        "df['sex'] * (df['systolic'] * 1000)",
        "df['sex'].str.upper() * 20000",
    ],
)
def test_a_column_of_texts_is_not_repeated_without_bound(df: pd.DataFrame, expression: str) -> None:
    """``df['site'] * 100000000`` builds a hundred million copies of each site's name. A text written in the
    expression is counted when the expression is read; a column of texts is known only when it is computed."""
    with pytest.raises(UnsafeExpressionError, match="A text repeated more than 10,000 times"):
        evaluate(expression, df)


def test_a_product_computes_as_it_did(df: pd.DataFrame) -> None:
    assert evaluate("df['systolic'] * 2", df).tolist() == [240, 300, 270, 180]
    assert evaluate("2 * df['Weight'] * 3", df).tolist() == [300.0, 480.0, 540.0, 360.0]
    assert evaluate("df['Weight'] * df['Height']", df).tolist() == [
        8000.0,
        14400.0,
        15300.0,
        9000.0,
    ]
    # numbers, however large; and a text repeated a few times
    assert evaluate("df['systolic'] * 100000", df).tolist() == [
        12000000,
        15000000,
        13500000,
        9000000,
    ]
    assert evaluate("(df['Weight'] * 1000000).round()", df).tolist() == [5e7, 8e7, 9e7, 6e7]
    assert evaluate("df['sex'] * 2", df).tolist() == [
        "malemale",
        "femalefemale",
        "malemale",
        "femalefemale",
    ]
    assert evaluate("3 * 4", df) == 12 and evaluate("'ab' * 3", df) == "ababab"
    # the caller's tree is evaluated as it is: it is not changed, and can be evaluated again
    tree = ast.parse("df['Weight'] * df['systolic']", mode="eval")
    as_given = ast.dump(tree)
    first, second = evaluate_tree(tree, df), evaluate_tree(tree, df)
    assert first.tolist() == second.tolist() == [6000.0, 12000.0, 12150.0, 5400.0]
    assert ast.dump(tree) == as_given


def test_an_expression_nested_too_deep_is_refused(df: pd.DataFrame) -> None:
    """240 methods called one after the other are fewer nodes than the policy allows and a tree too deep to walk:
    RecursionError came out of the evaluator, a failure of the application where a refusal was due."""
    chained = "df['age']" + ".abs()" * 240
    with pytest.raises(UnsafeExpressionError, match="too complex"):
        validate(chained)
    with pytest.raises(UnsafeExpressionError, match="too complex"):
        evaluate(chained, df)
    assert evaluate("df['age']" + ".abs()" * 20, df).tolist()[:3] == [10.0, 25.0, 70.0]
