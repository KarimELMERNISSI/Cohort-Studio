"""
What a worker that computes an expression is sent, and what it gives back.

- **A message** is a pickle (protocol 5) after its length in 8 bytes.
- **A job:** the tree, the table cut to the columns the expression reads (whole when it reads the table in another
  way), the policy, the values among the expression's names, and the helpers by their names: a callable that is not a
  registered helper (:data:`core.expressions.runner.HELPERS`) cannot be sent.
- **An answer:** the value, or the expression's error, with the warnings it raised. It is read by a reader that knows
  the names values and errors need and nothing else (an answer that names ``os.system`` is refused), its size capped
  before it is read: code that got past the evaluator in a worker cannot run in the server through its answer.

:func:`round_trip` passes a job and its answer through the wire in memory, without a process: the test suite computes
every expression through it, so that what a worker computes is shown to be what this process computes.
"""

from __future__ import annotations

import ast
import logging
import pickle
import re
import struct
import sys
import warnings
from collections.abc import Mapping
from typing import IO, Any

import numpy as np
import pandas as pd

from core.expressions.errors import ExpressionError, ExpressionStopped, WorkerUnavailable
from core.expressions.runner import HELPERS, columns_read
from core.expressions.safe_eval import compute_here

logger = logging.getLogger(__name__)

HEADER = struct.Struct("<Q")  # the length of a message, before it
ANSWER_FLOOR = (
    64 * 2**20
)  # an answer is read up to the largest of this, twice what was sent, and 256 bytes a row
BYTES_A_ROW = 256

# the names an answer may hold: their rebuilders, and nothing else. Found in the answers of every dtype, scalar and
# index a formula or a criterion can return (the design's probe P7; one run of the whole test suite, which computes
# every expression through the wire; a probe of the kinds the suite does not return, 15 November 2026), and kept by
# tests/test_worker_wire.py
ANSWER_NAMES = frozenset(
    {
        ("builtins", "slice"),
        ("datetime", "date"),
        ("datetime", "datetime"),
        ("datetime", "time"),
        ("datetime", "timedelta"),
        ("datetime", "timezone"),
        ("dateutil.relativedelta", "relativedelta"),  # inside a pd.DateOffset
        ("decimal", "Decimal"),  # a column read from a file that holds decimals
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy._core.numeric", "_frombuffer"),
        ("pandas", "ArrowDtype"),
        ("pandas", "BooleanDtype"),
        ("pandas", "Categorical"),
        ("pandas", "CategoricalDtype"),
        ("pandas", "CategoricalIndex"),
        ("pandas", "DataFrame"),
        ("pandas", "DateOffset"),
        ("pandas", "DatetimeIndex"),
        ("pandas", "DatetimeTZDtype"),
        ("pandas", "Float32Dtype"),
        ("pandas", "Float64Dtype"),
        ("pandas", "Index"),
        ("pandas", "Int8Dtype"),
        ("pandas", "Int16Dtype"),
        ("pandas", "Int32Dtype"),
        ("pandas", "Int64Dtype"),
        ("pandas", "Interval"),
        ("pandas", "IntervalDtype"),
        ("pandas", "IntervalIndex"),
        ("pandas", "MultiIndex"),
        ("pandas", "NA"),
        ("pandas", "Period"),
        ("pandas", "PeriodDtype"),
        ("pandas", "PeriodIndex"),
        ("pandas", "RangeIndex"),
        ("pandas", "Series"),
        ("pandas", "StringDtype"),
        ("pandas", "TimedeltaIndex"),
        ("pandas", "UInt8Dtype"),
        ("pandas", "UInt16Dtype"),
        ("pandas", "UInt32Dtype"),
        ("pandas", "UInt64Dtype"),
        ("pandas._libs.arrays", "__pyx_unpickle_NDArrayBacked"),
        ("pandas._libs.internals", "_unpickle_block"),
        ("pandas._libs.interval", "__pyx_unpickle_IntervalMixin"),
        ("pandas._libs.tslibs.nattype", "_nat_unpickle"),
        ("pandas._libs.tslibs.timedeltas", "_timedelta_unpickle"),
        ("pandas._libs.tslibs.timestamps", "_unpickle_timestamp"),
        ("pandas.arrays", "ArrowExtensionArray"),
        ("pandas.arrays", "ArrowStringArray"),
        ("pandas.arrays", "BooleanArray"),
        ("pandas.arrays", "DatetimeArray"),
        ("pandas.arrays", "FloatingArray"),
        ("pandas.arrays", "IntegerArray"),
        ("pandas.arrays", "IntervalArray"),
        ("pandas.arrays", "PeriodArray"),
        ("pandas.arrays", "StringArray"),
        ("pandas.arrays", "TimedeltaArray"),
        ("pandas.core.indexes.base", "_new_Index"),
        ("pandas.core.indexes.datetimes", "_new_DatetimeIndex"),
        ("pandas.core.indexes.interval", "_new_IntervalIndex"),
        ("pandas.core.indexes.timedeltas", "_new_TimedeltaIndex"),
        ("pandas.core.internals.managers", "BlockManager"),
        ("pandas.core.internals.managers", "SingleBlockManager"),
        ("pyarrow.lib", "_restore_array"),
        ("pyarrow.lib", "decimal128"),
        ("pyarrow.lib", "dictionary"),
        ("pyarrow.lib", "field"),
        ("pyarrow.lib", "list_"),
        ("pyarrow.lib", "py_buffer"),
        ("pyarrow.lib", "timestamp"),
        ("pyarrow.lib", "type_for_alias"),
        ("zoneinfo", "ZoneInfo"),  # a date in a named time zone
        ("core.expressions.errors", "_stopped"),
    }
)
# pandas' date offsets (a range's frequency, a period's, pd.DateOffset), read as classes of this module
OFFSETS_MODULE = "pandas._libs.tslibs.offsets"
# the one attribute an answer may read by name: a named time zone is rebuilt by ZoneInfo._unpickle, which pickle
# writes as getattr(ZoneInfo, "_unpickle"); getattr itself is never given (it would reach any method of a class)
READ_BY_NAME = frozenset({("zoneinfo.ZoneInfo", "_unpickle")})
# an error or a warning comes back as its class, when that class is an exception of one of these packages
ERROR_PACKAGES = (
    "builtins",
    "numpy",
    "pandas",
    "pyarrow",
    "core.expressions",
    "warnings",
    "re",  # a pattern that does not compile
    "dateutil",  # a date pandas cannot read
    "zoneinfo",
    "decimal",
)


def pack(obj: Any) -> bytes:
    return pickle.dumps(obj, protocol=5)


def write_message(stream: IO[bytes], payload: bytes) -> None:
    stream.write(HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def read_length(stream: IO[bytes]) -> int:
    head = stream.read(HEADER.size)
    if len(head) < HEADER.size:
        raise EOFError("the channel closed")
    return int(HEADER.unpack(head)[0])


def read_exactly(stream: IO[bytes], size: int) -> bytes:
    data = stream.read(size)
    if len(data) < size:
        raise EOFError("the channel closed")
    return data


# --- the job --------------------------------------------------------------------------------------------------------


def job_for(
    tree: ast.Expression, df: pd.DataFrame | None, *, policy: Any, extra_locals: Mapping[str, Any]
) -> dict[str, Any]:
    """The job of an expression: its table cut to the columns it reads (the index kept, so the rows it counts and its
    result's labels are the same), its values, and its helpers by name."""
    values: dict[str, Any] = {}
    helpers: list[str] = []
    for name, value in extra_locals.items():
        held = isinstance(value, pd.Series | pd.DataFrame | pd.Index | np.ndarray)
        if callable(value) and not held:
            if HELPERS.get(name) is not value:
                raise TypeError(f"{name} is not a registered helper: it cannot be sent to a worker")
            helpers.append(name)
        else:
            values[name] = value
    table = df
    if df is not None:
        read = columns_read(tree, policy.frame_names)
        if read is not None:
            table = df.loc[:, df.columns.isin(read)]
    return {"tree": tree, "table": table, "policy": policy, "values": values, "helpers": helpers}


def rows_of(job: Mapping[str, Any]) -> int:
    table = job.get("table")
    return 0 if table is None else len(table)


# --- what a worker computes -----------------------------------------------------------------------------------------


def compute_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """The answer to a job, computed with the same evaluator as the server: the value and the warnings raised on the
    way, or the expression's error (whole when it can travel, else its message). A MemoryError is the memory limit,
    the answer ``stopped``."""
    caught: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            names = dict(job["values"])
            names.update({name: HELPERS[name] for name in job["helpers"]})
            value = compute_here(
                job["tree"], job["table"], policy=job["policy"], extra_locals=names
            )
        return {"value": value, "warnings": [(one.category, str(one.message)) for one in caught]}
    except MemoryError:
        return {"stopped": ("memory", None), "warnings": []}
    except Exception as exc:
        return {
            "error": _portable(exc),
            "warnings": [(one.category, str(one.message)) for one in caught],
        }


def _portable(exc: Exception) -> Any:
    """The error itself when it can be sent and read back; its message otherwise (read as the expression's own
    error, :class:`core.expressions.errors.ExpressionError`)."""
    try:
        pickle.loads(pickle.dumps(exc, protocol=5))
    except Exception:
        return str(exc)
    return exc


# --- the answer -----------------------------------------------------------------------------------------------------


def answer_limit(sent: int, rows: int) -> int:
    """The largest answer read for a job of ``sent`` bytes over ``rows`` rows."""
    return max(ANSWER_FLOOR, 2 * sent, BYTES_A_ROW * rows)


class _Refused(Exception):
    pass


_DOTTED = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){1,8}")


def _refused(written: str) -> _Refused:
    """What was refused, for the log: a name a worker could fill with anything is not written."""
    plain = len(written) <= 120 and _DOTTED.fullmatch(written)
    return _Refused(written if plain else "a name that is not a Python name")


def _read_by_name(owner: Any, attribute: Any) -> Any:
    """getattr, as an answer may use it: only the attributes of :data:`READ_BY_NAME`."""
    if isinstance(owner, type) and isinstance(attribute, str):
        if (f"{owner.__module__}.{owner.__qualname__}", attribute) in READ_BY_NAME:
            return getattr(owner, attribute)
        raise _refused(f"{owner.__module__}.{owner.__qualname__}.{attribute}")
    raise _refused("builtins.getattr")


class _AnswerReader(pickle.Unpickler):
    """Reads an answer, knowing the names its values and errors need and nothing else."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in ANSWER_NAMES:
            return super().find_class(module, name)
        if (module, name) == ("builtins", "getattr"):
            return _read_by_name
        # a class of the packages named (never another module's attribute: its module could compute it on demand)
        if module == OFFSETS_MODULE or any(
            module == p or module.startswith(p + ".") for p in ERROR_PACKAGES
        ):
            found = getattr(sys.modules.get(module), name, None)
            if isinstance(found, type):
                if module == OFFSETS_MODULE and issubclass(found, pd.offsets.BaseOffset):
                    return found
                if issubclass(found, Exception):  # a worker sends no SystemExit
                    return found
        raise _refused(f"{module}.{name}")


UNREAD = (
    "The result could not be read back from the process that computed it, and was not used. If it happens again, "
    "tell the person who runs this server."
)


def read_answer(payload: bytes) -> dict[str, Any]:
    """An answer, read with the restricted reader. One that names anything else, or cannot be read, means the worker
    cannot be trusted: :class:`WorkerUnavailable`, the name refused in the log (a name, never data)."""
    import io

    try:
        answer = _AnswerReader(io.BytesIO(payload)).load()
    except _Refused as refused:
        # its text, not the exception: a name _refused has checked, which the log's cleaning of exceptions
        # (utils.logging_config.records_without_values) would otherwise write as "_Refused"
        logger.warning(
            "A worker's answer named %s, which the server does not read; it was not used",
            str(refused),
        )
        raise WorkerUnavailable(UNREAD) from None
    except Exception as exc:
        logger.warning(
            "A worker's answer could not be read (%s); it was not used", type(exc).__name__
        )
        raise WorkerUnavailable(UNREAD) from None
    if not isinstance(answer, dict):
        logger.warning("A worker's answer was not a dictionary; it was not used")
        raise WorkerUnavailable(UNREAD)
    return answer


def value_of(answer: Mapping[str, Any]) -> Any:
    """The value of an answer, its warnings raised again here (the caller's filters apply), or its error raised."""
    for category, message in answer.get("warnings") or ():
        if isinstance(category, type) and issubclass(category, Warning):
            warnings.warn(str(message), category, stacklevel=3)
    if "stopped" in answer:
        reason, limit = answer["stopped"]
        raise stopped(str(reason), limit)
    if "error" in answer:
        error = answer["error"]
        if isinstance(error, BaseException):
            raise error
        raise ExpressionError(str(error))
    return answer.get("value")


# --- why an expression was stopped ----------------------------------------------------------------------------------

_PATTERN = (
    " If its pattern repeats a repetition, as (a+)+ does, write it without one: such a pattern can take hours on one "
    "long value."
)


def stopped(
    reason: str,
    limit: float | None = None,
    *,
    pattern: bool = False,
    again: bool = False,
    whole: bool = True,
) -> ExpressionStopped:
    """The sentence a page shows when a worker stopped an expression at one of its limits; ``again``: the same job,
    stopped earlier, refused now without being computed; ``whole``: the job held the whole table (``input``)."""
    shown = f"{limit:,.0f}" if isinstance(limit, int | float) else ""
    seconds = "second" if limit == 1 else "seconds"
    if again:
        sentence = {
            "time": f"It was stopped earlier on the same data after {shown} {seconds}, the longest this server gives "
            "one formula or criterion, and is not computed again." + (_PATTERN if pattern else ""),
            "memory": "It was stopped earlier on the same data for needing more memory than this server gives one "
            f"formula or criterion ({shown} MB), and is not computed again.",
            "result": "Its result on the same data was larger than this server takes back from one formula or "
            "criterion, and it is not computed again.",
        }.get(reason, "It was stopped earlier on the same data, and is not computed again.")
        return ExpressionStopped(sentence, reason=reason, limit=limit)
    sentence = {
        "time": f"It took longer than {shown} {seconds}, the longest this server gives one formula or criterion, "
        "and was stopped." + (_PATTERN if pattern else ""),
        "memory": f"It needed more memory than this server gives one formula or criterion ({shown} MB), and was "
        "stopped.",
        "input": (
            f"It reads the whole table, larger than this server sends to one formula or criterion ({shown} MB at "
            "most), and was not computed: name the columns it reads, as df['age']."
            if whole
            else f"The columns it reads are larger than this server sends to one formula or criterion ({shown} MB "
            "at most), and it was not computed."
        ),
        "result": "Its result is larger than this server takes back from one formula or criterion, and was not kept.",
        "ended": "The process computing it ended before it finished, perhaps for want of memory, and nothing was "
        "changed.",
    }.get(reason, "It was stopped.")
    return ExpressionStopped(sentence, reason=reason, limit=limit)


# --- in memory ------------------------------------------------------------------------------------------------------


def round_trip(
    tree: ast.Expression, df: pd.DataFrame | None, *, policy: Any, extra_locals: Mapping[str, Any]
) -> Any:
    """
    A runner that passes the job and its answer through the wire in memory, as a worker would: the job pickled and
    read back, computed by :func:`compute_job`, the answer pickled and read by the restricted reader under its cap. No
    process: the test suite computes every expression through it.
    """
    job = job_for(tree, df, policy=policy, extra_locals=extra_locals)
    sent = pack(job)
    answer = pack(compute_job(pickle.loads(sent)))
    if len(answer) > answer_limit(len(sent), rows_of(job)):
        raise stopped("result")
    return value_of(read_answer(answer))
