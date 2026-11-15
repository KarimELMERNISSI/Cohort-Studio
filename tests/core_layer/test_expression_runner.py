"""
One place computes an expression. Every computation goes through ``core.expressions.evaluate_tree``, which hands the
tree to the runner installed in the process: none means this process, as always. A tree travels with the names it
calls, so every callable the application passes among an expression's names is a registered helper; a worker can be
given only the columns an expression reads; and since a worker serves every user in turn, the evaluator offers no
function that reads memory it was not given, or a file.
"""

from __future__ import annotations

import ast
import pickle
from typing import Any

import numpy as np
import pandas as pd
import pytest

from core.expressions import (
    ExpressionStopped,
    UnsafeExpressionError,
    WorkerUnavailable,
    compute_here,
    evaluate,
    evaluate_tree,
)
from core.expressions.runner import HELPERS, SAFE_SUB, columns_read, install, installed, running


def _tree(expression: str) -> ast.Expression:
    return ast.parse(expression, mode="eval")


@pytest.mark.parametrize(
    ("expression", "read"),
    [
        ("df['age'] > 60", ["age"]),
        ("df['age'] + df['bmi'] * 2", ["age", "bmi"]),
        ("df[['age', 'sex']]", ["age", "sex"]),
        ("df['age'].where(df['sex'] == 'F', 0)", ["age", "sex"]),
        ("np.where(df['age'] > 60, 'older', 'younger')", ["age"]),
        ("df['age'] + df['age']", ["age"]),
        ("1 + 1", []),
        # the table read in any other way: sent whole
        ("df.isna()", None),
        ("len(df)", None),
        ("df[df['age'] > 60]", None),
        ("df[['age'] * 2]", None),
        ("df['age'][df['age'] > 60]", ["age"]),
    ],
)
def test_the_columns_an_expression_reads(expression: str, read: list[str] | None) -> None:
    assert columns_read(_tree(expression), frozenset({"df"})) == read


def test_nothing_installed_computes_here_and_running_restores_the_runner() -> None:
    df = pd.DataFrame({"age": [40, 70]})
    with running(None):
        _nothing_installed_computes_here(df)


def _nothing_installed_computes_here(df: pd.DataFrame) -> None:
    assert installed() is None
    assert evaluate("df['age'] > 60", df).tolist() == [False, True]
    calls: list[str] = []

    def counting(tree: ast.Expression, frame: Any, **kwargs: Any) -> Any:
        calls.append(ast.unparse(tree))
        return compute_here(tree, frame, **kwargs)

    with running(counting):
        assert installed() is counting
        assert evaluate("df['age'] * 2", df).tolist() == [80, 140]
        with running(None):  # an inner block computes here, and gives the runner back
            assert evaluate("df['age'] + 1", df).tolist() == [41, 71]
        assert installed() is counting
    assert installed() is None and calls == ["df['age'] * 2"]
    previous = install(counting)
    assert previous is None and install(previous) is counting


def _strict(tree: ast.Expression, frame: Any, *, policy: Any, extra_locals: Any) -> Any:
    """A runner that takes no callable but a registered helper: what a worker can be sent."""
    for name, value in extra_locals.items():
        if callable(value):
            assert HELPERS.get(name) is value, f"{name} is not a registered helper"
    return compute_here(tree, frame, policy=policy, extra_locals=extra_locals)


def test_the_callables_the_application_passes_are_registered_helpers() -> None:
    """The criteria's Kleene comparisons and the formulas' days travel by their names."""
    from core.criteria import criterion_mask
    from enrich.formulas import evaluate_formula_safely

    df = pd.DataFrame(
        {
            "age": [40.0, 70.0, np.nan],
            "admitted": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-02-01"]),
            "discharged": pd.to_datetime(["2024-01-11", "2024-01-06", None]),
        }
    )
    with running(_strict):
        mask = criterion_mask(df, "older", {"expression": "df['age'] > 60"})
        ok, _, out = evaluate_formula_safely("discharged - admitted", "stay", df)
    assert mask.tolist() == [False, True, False]
    assert ok and out["stay"].tolist()[:2] == [10, 1] and pd.isna(out["stay"].iloc[2])
    assert SAFE_SUB in HELPERS


def test_a_formula_s_subtraction_goes_through_the_evaluator_s_checks() -> None:
    """The days of a subtraction are read by a helper on the evaluator's difference: a helper that subtracted
    computed out of sight of the checks (two columns aligned by repeated labels)."""
    from enrich.formulas import evaluate_formula_safely

    df = pd.DataFrame({"x": [5.0, 7.0], "y": [1.0, 2.0]})
    ok, _, out = evaluate_formula_safely("x - y - 1", "z", df)
    assert ok and out["z"].tolist() == [3.0, 4.0]
    seen: list[str] = []

    def looking(tree: ast.Expression, frame: Any, **kwargs: Any) -> Any:
        seen.append(ast.unparse(tree))
        return compute_here(tree, frame, **kwargs)

    with running(looking):
        evaluate_formula_safely("x - y", "z", df)
    assert seen and f"{SAFE_SUB}(" in seen[0] and " - " in seen[0]


def test_what_a_worker_raises_reads_as_the_callers_know() -> None:
    stopped = ExpressionStopped("It took longer than 30 seconds.", reason="time", limit=30)
    assert isinstance(stopped, UnsafeExpressionError) and isinstance(stopped, ValueError)
    again = pickle.loads(pickle.dumps(stopped))
    assert (str(again), again.reason, again.limit) == (
        "It took longer than 30 seconds.",
        "time",
        30,
    )
    assert not isinstance(WorkerUnavailable("busy"), ValueError)

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise WorkerUnavailable("no worker")

    with running(unavailable), pytest.raises(WorkerUnavailable):
        evaluate_tree(_tree("1 + 1"))


# what would let an expression read memory it was not given (what an earlier job left in a worker that serves every
# user in turn), read or write a file, or call a function of its choosing
_NUMPY_NEVER = frozenset(
    {
        "ctypeslib",
        "DataSource",
        "empty",
        "empty_like",
        "frombuffer",
        "fromfile",
        "frompyfunc",
        "fromregex",
        "fromstring",
        "genfromtxt",
        "lib",
        "load",
        "loadtxt",
        "memmap",
        "ndarray",
        "save",
        "savetxt",
        "savez",
        "savez_compressed",
        "vectorize",
    }
)
_PANDAS_NEVER = frozenset(
    {
        "api",
        "core",
        "eval",
        "ExcelFile",
        "ExcelWriter",
        "HDFStore",
        "io",
        "options",
        "set_option",
        "testing",
    }
)
_METHODS_NEVER = frozenset(
    {
        "agg",
        "aggregate",
        "apply",
        "eval",
        "map",
        "pipe",
        "query",
        "to_clipboard",
        "to_csv",
        "to_excel",
        "to_feather",
        "to_hdf",
        "to_html",
        "to_json",
        "to_latex",
        "to_markdown",
        "to_orc",
        "to_parquet",
        "to_pickle",
        "to_sql",
        "to_stata",
        "to_xml",
        "tofile",
        "transform",
        "view",
    }
)
_BUILTINS_NEVER = frozenset(
    {
        "__import__",
        "breakpoint",
        "bytearray",
        "compile",
        "delattr",
        "eval",
        "exec",
        "getattr",
        "globals",
        "input",
        "locals",
        "memoryview",
        "open",
        "setattr",
        "vars",
    }
)


def test_no_expression_reads_memory_it_was_not_given_or_a_file() -> None:
    """A worker serves the expressions of every user in turn: what one job leaves in its memory is out of the next
    job's reach only because the evaluator offers no function that reads memory it was not given, reads or writes a
    file, or calls a function of the expression's choosing. Kept for every policy the application computes with."""
    from core.criteria import _THREE_VALUED
    from core.expressions import DEFAULT_POLICY

    for policy in (DEFAULT_POLICY, _THREE_VALUED):
        assert not policy.np_functions & _NUMPY_NEVER, policy.np_functions & _NUMPY_NEVER
        assert not policy.pd_functions & _PANDAS_NEVER, policy.pd_functions & _PANDAS_NEVER
        assert not [name for name in policy.pd_functions if name.startswith(("read_", "to_pickle"))]
        for methods in (policy.series_methods, policy.str_methods, policy.dt_attributes):
            assert not methods & _METHODS_NEVER, methods & _METHODS_NEVER
        assert not set(policy.builtins) & _BUILTINS_NEVER, set(policy.builtins) & _BUILTINS_NEVER
