"""
Where an expression is computed. Every computation of an expression goes through
:func:`core.expressions.evaluate_tree`, which hands the validated tree to the runner installed here: with none
installed (the test suite, a notebook, the worker itself) it is computed in this process, as it always was; the
application and the replay's command line install a pool of worker processes that compute it under a time limit and a
memory limit (``manage.workers``). No caller changes its call.

A tree travels to a worker with the names it calls: the functions an expression may call by name are registered in
:data:`HELPERS` (Kleene's comparisons of the criteria, the formulas' date subtraction), so that the other side finds
them by the same names. A callable that is not registered cannot travel.
"""

from __future__ import annotations

import ast
import contextlib
import datetime as dt
import threading
from collections.abc import Callable, Collection, Iterator, Mapping
from typing import Any, Protocol

import pandas as pd

from core.expressions import kleene
from core.expressions.errors import ExpressionStopped, UnsafeExpressionError, WorkerUnavailable

__all__ = [
    "HELPERS",
    "SAFE_SUB",
    "ExpressionStopped",
    "Runner",
    "UnsafeExpressionError",
    "WorkerUnavailable",
    "columns_read",
    "in_days",
    "install",
    "installed",
    "running",
]


class Runner(Protocol):
    """Computes a tree that the evaluator validated, as :func:`core.expressions.safe_eval.compute_here` does."""

    def __call__(
        self,
        tree: ast.Expression,
        df: pd.DataFrame | None,
        *,
        policy: Any,
        extra_locals: Mapping[str, Any],
    ) -> Any: ...


_lock = threading.Lock()
_runner: Runner | None = None


def install(runner: Runner | None) -> Runner | None:
    """Install ``runner`` for every expression of this process (None: compute here); returns the one it replaces."""
    global _runner
    with _lock:
        previous, _runner = _runner, runner
    return previous


def installed() -> Runner | None:
    """The runner installed in this process, or None when expressions are computed here."""
    return _runner


@contextlib.contextmanager
def running(runner: Runner | None) -> Iterator[None]:
    """Compute every expression with ``runner`` inside the block, then the previous runner again (for tests)."""
    previous = install(runner)
    try:
        yield
    finally:
        install(previous)


SAFE_SUB = (
    "__safe_sub__"  # the name a formula's subtraction is read in days under (enrich.formulas)
)


def in_days(difference: Any) -> Any:
    """A formula's subtraction, read in days when it is one of two dates: a column of whole days, or one number of
    days; anything else as it is. The subtraction itself is the evaluator's (``__safe_sub__(a - b)``), with its
    checks: a helper that subtracted would compute out of their sight."""
    try:
        if hasattr(difference, "dtype") and pd.api.types.is_timedelta64_dtype(difference.dtype):
            return difference.dt.days
        if isinstance(difference, pd.Timedelta | dt.timedelta):
            return difference.days
    except Exception:  # a difference that cannot be read in days stays as it is
        return difference
    return difference


# the functions an expression may call by name: Kleene's three of the criteria, and the formulas' days
HELPERS: dict[str, Callable[..., Any]] = {**kleene.HELPERS, SAFE_SUB: in_days}


def columns_read(tree: ast.AST, frame_names: Collection[str]) -> list[str] | None:
    """
    The columns an expression reads, when it reads its table only by name (``df['age']``, ``df[['age', 'sex']]``):
    a worker can then be given those columns only. None when it reads the table in any other way (``df.isna()``,
    ``len(df)``, ``df[mask]``, ``df[['age'] * 2]``): the table is then sent whole.
    """
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    read: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Name) and node.id in frame_names):
            continue
        parent = parents.get(node)
        if not (isinstance(parent, ast.Subscript) and parent.value is node):
            return None
        key = parent.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            read.append(key.value)
        elif (
            isinstance(key, ast.List | ast.Tuple)
            and key.elts
            and all(
                isinstance(one, ast.Constant) and isinstance(one.value, str) for one in key.elts
            )
        ):
            read.extend(str(one.value) for one in key.elts)  # type: ignore[attr-defined]
        else:
            return None
    return list(dict.fromkeys(read))
