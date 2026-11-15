"""
The worker's wire (manage.workers.wire), in memory: what a worker computes is what this process computes. The whole
test suite computes its expressions through the round trip (tests/conftest.py); this file holds what the round trip
alone shows: values of every dtype the application holds compared exactly, an answer of every kind a formula can
return read back, errors and warnings coming back as they were raised, an answer that names anything else refused (and
reads no attribute by name but a time zone's rebuilder), an answer over its cap not kept.
"""

from __future__ import annotations

import ast
import decimal
import gc
import os
import pickle
import re
import weakref
import zoneinfo
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from core.expressions import ExpressionStopped, WorkerUnavailable, compute_here
from core.expressions.runner import running
from core.expressions.safe_eval import DEFAULT_POLICY
from manage.workers import wire
from manage.workers.wire import compute_job

ROOT = Path(__file__).resolve().parents[1]


def _table() -> pd.DataFrame:
    """Every dtype the application holds, with its missing values."""
    return pd.DataFrame(
        {
            "count": pd.array([1, 2, None, 4, 5, 6], dtype="Int64"),
            "weight": [70.5, np.nan, 81.0, 64.2, 90.1, 77.7],
            "site": pd.array(["Alpha", "Beta", None, "Alpha", "Gamma", "Beta"], dtype="string"),
            "grade": pd.Categorical(["I", "II", "I", "III", None, "II"]),
            "admitted": pd.to_datetime(
                ["2024-01-01", "2024-02-01", None, "2024-03-05", "2024-04-01", "2024-05-01"]
            ),
            "seen": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", None, "2024-01-05", "2024-01-06"]
            ).tz_localize("UTC"),
            "flag": pd.array([True, False, None, True, False, True], dtype="boolean"),
            "code": pd.Series(["A1", 2, None, "B3", 4.5, "C"], dtype=object),
        },
        index=pd.Index([10, 11, 12, 13, 14, 15]),
    )


CORPUS = [
    "df['count'] + 1",
    "df['count'] > 2",
    "df['weight'].round(1)",
    "df['weight'] / df['count']",
    "df['site'].str.upper()",
    "df['site'] == 'Alpha'",
    "df['site'].str.contains('a', regex=False)",
    "df['grade'] == 'I'",
    "df['admitted'].dt.year",
    "df['seen'].dt.day",
    "df['flag'] & (df['weight'] > 70)",
    "df['code'].isna()",
    "df.isna().sum()",
    "len(df)",
    "df[df['weight'] > 70]",
    "np.where(df['weight'] > 75, 'heavy', 'light')",
    "pd.cut(df['weight'], [0, 70, 80, 100])",
    "df['weight'].mean()",
    # the expression's own errors come back as they were raised
    "df['nope'] + 1",
    "df['site'] + 1",
    "df['weight'].str.upper()",
]


def _outcome(expression: str, frame: pd.DataFrame, *, through_wire: bool) -> tuple[str, object]:
    tree = ast.parse(expression, mode="eval")
    try:
        with running(None):
            if through_wire:
                return "value", wire.round_trip(tree, frame, policy=DEFAULT_POLICY, extra_locals={})
            return "value", compute_here(tree, frame)
    except Exception as exc:
        return "error", (type(exc), str(exc))


@pytest.mark.parametrize("expression", CORPUS)
def test_what_a_worker_computes_is_what_this_process_computes(expression: str) -> None:
    frame = _table()
    here = _outcome(expression, frame, through_wire=False)
    there = _outcome(expression, frame, through_wire=True)
    assert here[0] == there[0], (expression, here, there)
    if here[0] == "error":
        assert here[1] == there[1]
        return
    mine, theirs = here[1], there[1]
    if isinstance(mine, pd.DataFrame):
        pd.testing.assert_frame_equal(mine, theirs, check_exact=True)
    elif isinstance(mine, pd.Series):
        pd.testing.assert_series_equal(mine, theirs, check_exact=True)
    elif isinstance(mine, np.ndarray):
        assert (
            np.array_equal(mine, theirs, equal_nan=mine.dtype.kind == "f")
            and mine.dtype == theirs.dtype
        )
    else:
        assert mine == theirs and type(mine) is type(theirs)


def test_a_table_is_cut_to_the_columns_an_expression_reads() -> None:
    job = wire.job_for(
        ast.parse("df['weight'] > 70", mode="eval"),
        _table(),
        policy=DEFAULT_POLICY,
        extra_locals={},
    )
    assert list(job["table"].columns) == ["weight"] and list(job["table"].index) == [
        10,
        11,
        12,
        13,
        14,
        15,
    ]
    whole = wire.job_for(
        ast.parse("df.isna()", mode="eval"), _table(), policy=DEFAULT_POLICY, extra_locals={}
    )
    assert whole["table"].shape == (6, 8)
    with pytest.raises(TypeError, match="not a registered helper"):
        wire.job_for(
            ast.parse("f(1)", mode="eval"), None, policy=DEFAULT_POLICY, extra_locals={"f": print}
        )


def test_a_warning_comes_back_and_meets_the_callers_filters() -> None:
    job = wire.job_for(
        ast.parse("np.log(df['weight'] - 80)", mode="eval"),
        _table(),
        policy=DEFAULT_POLICY,
        extra_locals={},
    )
    answer = compute_job(pickle.loads(wire.pack(job)))
    assert answer["warnings"] and issubclass(answer["warnings"][0][0], RuntimeWarning)
    with pytest.warns(RuntimeWarning):
        wire.value_of(wire.read_answer(wire.pack(answer)))


def test_an_answer_that_names_anything_else_is_not_read(caplog: pytest.LogCaptureFixture) -> None:
    """Code that got past the evaluator in a worker cannot run in the server through its answer. The page says the
    result was not used; the log names what was refused (a name, never data)."""
    forged_answers = (
        {"value": os.system},
        {"value": pickle.loads},
        {"value": __import__},
        {"error": SystemExit(0)},  # an exception, but not one an expression raises
    )
    for forged in forged_answers:
        with (
            caplog.at_level("WARNING", logger="manage.workers.wire"),
            pytest.raises(WorkerUnavailable) as caught,
        ):
            wire.read_answer(pickle.dumps(forged, protocol=5))
        assert str(caught.value) == wire.UNREAD
    assert "builtins.SystemExit" in caplog.text and "which the server does not read" in caplog.text
    with (
        caplog.at_level("WARNING", logger="manage.workers.wire"),
        pytest.raises(WorkerUnavailable, match="could not be read back"),
    ):
        wire.read_answer(pickle.dumps(["a list"], protocol=5))
    assert "was not a dictionary" in caplog.text
    # an error of the expression travels; a stop too, with its reason and limit
    error = wire.read_answer(pickle.dumps({"error": KeyError("nope")}, protocol=5))
    with pytest.raises(KeyError):
        wire.value_of(error)
    stop = wire.stopped("memory", 2048)
    with pytest.raises(ExpressionStopped) as caught:
        wire.value_of(wire.read_answer(pickle.dumps({"error": stop}, protocol=5)))
    assert (caught.value.reason, caught.value.limit) == ("memory", 2048)


def _kinds() -> dict[str, object]:
    """What a formula or a criterion can return beyond what the corpus returns: scalars, every nullable dtype, Arrow
    columns, intervals, dates in a named time zone, indexes with a frequency, a date offset."""
    dates = pd.to_datetime(["2024-01-02", None, "2024-03-01"])
    return {
        "NaT": pd.NaT,
        "a Period": pd.Period("2024-01", "M"),
        "an Interval": pd.Interval(0, 1),
        "a DateOffset": pd.DateOffset(months=3),
        "Int32": pd.Series([1, None, 3], dtype="Int32"),
        "UInt8": pd.Series([1, None, 3], dtype="UInt8"),
        "Float32": pd.Series([1.5, None, 3.0], dtype="Float32"),
        "python strings": pd.Series(["a", None, "b"], dtype="string[python]"),
        "an Arrow column": pd.Series([1, None, 3], dtype=pd.ArrowDtype(pa.int64())),
        "Arrow timestamps": pd.Series(dates).astype(pd.ArrowDtype(pa.timestamp("ns"))),
        "Arrow decimals": pd.Series(
            [decimal.Decimal("1.5"), None], dtype=pd.ArrowDtype(pa.decimal128(5, 2))
        ),
        "decimals": pd.Series([decimal.Decimal("1.5"), None], dtype=object),
        "intervals": pd.Series(pd.arrays.IntervalArray.from_breaks([0, 1, 2])),
        "qcut": pd.qcut(pd.Series([1.0, 5.0, 9.0, 12.0]), 2),
        "a named time zone": pd.Series(dates).dt.tz_localize("Europe/Luxembourg"),
        "periods by quarter": pd.Series(pd.period_range("2024Q1", periods=2, freq="Q")),
        "a weekly range": pd.Series(
            [1, 2, 3], index=pd.date_range("2024-01-01", periods=3, freq="W-MON")
        ),
        "durations": pd.Series(
            pd.to_timedelta([1, None], unit="D"), index=pd.timedelta_range("1D", periods=2)
        ),
        "a category index": pd.Series([1, 2], index=pd.CategoricalIndex(["a", "b"])),
        "a two-level index": pd.Series(
            [1, 2], index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)])
        ),
        "a period index": pd.Series([1, 2], index=pd.period_range("2024-01", periods=2, freq="M")),
    }


@pytest.mark.parametrize("kind", list(_kinds()))
def test_an_answer_of_every_kind_a_formula_can_return_is_read(kind: str) -> None:
    """The reader knows the names of every scalar, dtype and index a formula or a criterion can return, not only
    those the test suite returns: a name missing here would fail on a user's page, not in the tests."""
    value = _kinds()[kind]
    back = wire.value_of(wire.read_answer(wire.pack({"value": value, "warnings": []})))
    if isinstance(value, pd.Series):
        pd.testing.assert_series_equal(value, back, check_exact=True)
    elif value is pd.NaT:
        assert back is pd.NaT
    else:
        assert back == value and type(back) is type(value)


class _ReadByName:
    """A forged answer that reads an attribute of a class by its name, as pickle writes a bound method."""

    def __init__(self, owner: object, attribute: str) -> None:
        self.owner, self.attribute = owner, attribute

    def __reduce__(self) -> tuple[object, tuple[object, str]]:
        return (getattr, (self.owner, self.attribute))


def test_an_answer_reads_no_attribute_by_name_but_a_time_zones_rebuilder(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """getattr would reach any method of a class the reader knows (Series.to_pickle writes a file): only
    ZoneInfo._unpickle, which rebuilds a named time zone, is read."""
    forged = (
        (pd.Series, "to_pickle"),
        (pd.DataFrame, "to_csv"),
        (zoneinfo.ZoneInfo, "no_cache"),
        (zoneinfo.ZoneInfo, "__subclasses__"),
        (pd.NA, "__class__"),
    )
    for owner, attribute in forged:
        with (
            caplog.at_level("WARNING", logger="manage.workers.wire"),
            pytest.raises(WorkerUnavailable),
        ):
            wire.read_answer(pickle.dumps({"value": _ReadByName(owner, attribute)}, protocol=5))
    assert "pandas.Series.to_pickle" in caplog.text and "zoneinfo.ZoneInfo.no_cache" in caplog.text
    rebuilt = wire.read_answer(
        pickle.dumps({"value": zoneinfo.ZoneInfo("Europe/Luxembourg")}, protocol=5)
    )
    assert rebuilt["value"] == zoneinfo.ZoneInfo("Europe/Luxembourg")


def test_an_answer_over_its_cap_is_not_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wire, "ANSWER_FLOOR", 1024)
    monkeypatch.setattr(wire, "BYTES_A_ROW", 0)
    with pytest.raises(ExpressionStopped) as caught:
        wire.round_trip(
            ast.parse("np.zeros(100000)", mode="eval"), None, policy=DEFAULT_POLICY, extra_locals={}
        )
    assert caught.value.reason == "result" and "not kept" in str(caught.value)
    assert wire.answer_limit(100, 0) == 1024 and wire.answer_limit(10_000, 0) == 20_000


def test_a_worker_keeps_nothing_of_a_table_after_its_job() -> None:
    frame = _table()
    job = pickle.loads(
        wire.pack(
            wire.job_for(
                ast.parse("df['weight'] * 2", mode="eval"),
                frame,
                policy=DEFAULT_POLICY,
                extra_locals={},
            )
        )
    )
    held = weakref.ref(job["table"])
    answer = compute_job(job)
    del job, answer
    gc.collect()
    assert held() is None


def test_the_server_sets_no_option_that_changes_a_computation() -> None:
    """A worker starts from pandas' and numpy's defaults: an option the server set would compute otherwise there.
    Only options of how a table is shown are set (a styled table's size on Validate)."""
    found = []
    for folder in (
        "app_pages",
        "ui",
        "manage",
        "core",
        "enrich",
        "explore",
        "monitor",
        "biostats",
        "utils",
        "ingest",
    ):
        for path in (ROOT / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for call in re.findall(r"(?:pd|pandas)\.set_option\(\s*['\"]([^'\"]+)['\"]", text):
                if not call.startswith(("display.", "styler.")):
                    found.append(f"{path.name}: {call}")
            if re.search(r"np\.seterr\(|numpy\.seterr\(", text):
                found.append(f"{path.name}: seterr")
    assert not found, found


def test_the_sentences_of_a_stop() -> None:
    """Each stop says what happened in the server's numbers; a stop said again says it was not computed again; the
    advice on patterns is a condition, not an order."""
    assert str(wire.stopped("time", 1)).startswith("It took longer than 1 second,")
    assert str(wire.stopped("time", 30)).startswith("It took longer than 30 seconds,")
    pattern = str(wire.stopped("time", 30, pattern=True))
    assert "If its pattern repeats a repetition" in pattern
    again = str(wire.stopped("time", 30, again=True))
    assert (
        again.startswith("It was stopped earlier on the same data after 30 seconds")
        and "not computed again" in again
    )
    assert "(2,048 MB), and is not computed again" in str(wire.stopped("memory", 2048, again=True))
    assert "reads the whole table" in str(
        wire.stopped("input", 1024)
    ) and "(1,024 MB at most)" in str(wire.stopped("input", 1024))
    assert str(wire.stopped("input", 1024, whole=False)).startswith(
        "The columns it reads are larger"
    )
    assert "perhaps for want of memory" in str(wire.stopped("ended"))
