"""
The workers that compute expressions, as processes (manage.workers.pool): a pattern that backtracks is stopped by the
time limit, an expression that needs too much memory by the memory limit (Windows: a Job Object; Linux: RLIMIT_DATA),
the results are this process's, an interrupted call leaves no process, and a worker holds no setting or key. Each test
ends its workers.
"""

from __future__ import annotations

import ast
import os
import re
import statistics
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.expressions import ExpressionStopped, WorkerUnavailable, compute_here
from core.expressions.safe_eval import DEFAULT_POLICY
from manage.workers import limits
from manage.workers.pool import NOT_STARTED, ExpressionWorkers, waiting


def _tree(expression: str) -> ast.Expression:
    return ast.parse(expression, mode="eval")


def _gone(pids: list[int], seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not any(_alive(pid) for pid in pids):
            return True
        time.sleep(0.1)
    return False


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel.CloseHandle(handle)
        return code.value == 259  # still active
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@pytest.fixture
def pool() -> Iterator[ExpressionWorkers]:
    workers = ExpressionWorkers(workers=1, seconds=3, memory_mb=600)
    try:
        yield workers
    finally:
        pids = workers.processes()
        workers.close()
        assert _gone(pids), "a worker was left running"


def _run(pool: ExpressionWorkers, expression: str, frame: pd.DataFrame | None) -> object:
    return pool(_tree(expression), frame, policy=DEFAULT_POLICY, extra_locals={})


def _warm_replacement(pool: ExpressionWorkers, before: list[int], seconds: float = 60.0) -> None:
    """Waits until a worker that is not one of ``before`` is idle in the pool: the replacement the pool starts in the
    background after a worker is ended."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        with pool._changed:
            if any(w.process.pid not in before and w.alive() for w in pool._idle):
                return
        time.sleep(0.1)
    raise AssertionError("no replacement worker was started")


def test_a_worker_computes_what_this_process_computes(pool: ExpressionWorkers) -> None:
    frame = pd.DataFrame(
        {
            "age": pd.array([40, 70, None], dtype="Int64"),
            "site": pd.array(["Alpha", None, "Beta"], dtype="string"),
            "admitted": pd.to_datetime(["2024-01-01", None, "2024-03-01"]),
        }
    )
    for expression in (
        "df['age'] > 60",
        "df['site'].str.lower()",
        "df['admitted'].dt.month",
        "len(df)",
    ):
        mine = compute_here(_tree(expression), frame)
        theirs = _run(pool, expression, frame)
        if isinstance(mine, pd.Series):
            pd.testing.assert_series_equal(mine, theirs, check_exact=True)
        else:
            assert mine == theirs
    with pytest.raises(KeyError):
        _run(pool, "df['nope']", frame)


def test_a_pattern_that_backtracks_is_stopped_by_the_time_limit(pool: ExpressionWorkers) -> None:
    """(a+)+$ on one value of 40 letters a and a b: Python's pattern engine would search for hours, holding its
    interpreter. The worker is ended at the limit, a replacement starts at once in the background, and the same job
    is refused at once, in words that say so."""
    frame = pd.DataFrame({"code": pd.Series(["a" * 40 + "b", "x"], dtype=object)})
    _run(pool, "df['code'].str.len()", frame)
    before = pool.processes()
    started = time.monotonic()
    with pytest.raises(ExpressionStopped) as caught:
        _run(pool, "df['code'].str.contains('(a+)+$')", frame)
    assert time.monotonic() - started < 6
    assert caught.value.reason == "time" and "repeats a repetition" in str(caught.value)
    assert _gone(before)
    _warm_replacement(pool, before)  # no call made: the pool starts the replacement itself
    started = time.monotonic()
    with pytest.raises(ExpressionStopped) as again:
        _run(pool, "df['code'].str.contains('(a+)+$')", frame)
    assert time.monotonic() - started < 1  # remembered
    assert "stopped earlier on the same data" in str(again.value) and "not computed again" in str(
        again.value
    )
    assert _run(pool, "df['code'].str.len()", frame).tolist() == [41, 1]


@pytest.mark.skipif(not limits.memory_limited(), reason="macOS: no memory limit to start under")
def test_a_worker_that_cannot_start_is_said_in_fixed_words(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker that cannot start (here: 8 MB, too little to load its libraries; it ends, or hangs until the start
    limit): the page's sentence, and the log's cause in fixed words and numbers, never a message that could hold a
    path."""
    monkeypatch.setattr("manage.workers.pool.START_SECONDS", 5.0)
    pool = ExpressionWorkers(workers=1, seconds=3, memory_mb=8)
    try:
        with (
            caplog.at_level("WARNING", logger="manage.workers.pool"),
            pytest.raises(WorkerUnavailable) as caught,
        ):
            _run(pool, "1 + 1", None)
        assert str(caught.value) == NOT_STARTED
        said = re.search(r"did not start: (.*)", caplog.text)
        assert said and re.fullmatch(
            r"it ended before it was ready \(exit code -?\d+\)|it was not ready after 5 seconds",
            said.group(1).strip(),
        ), caplog.text
    finally:
        pids = pool.processes()
        pool.close()
        assert _gone(pids)


def test_a_table_larger_than_half_the_limit_is_not_sent() -> None:
    """The input limit is half the memory limit, said with its size, and whether the table went whole: no worker is
    started for it."""
    pool = ExpressionWorkers(workers=1, seconds=3, memory_mb=2)  # 1 MB sent at most
    frame = pd.DataFrame({"site": [f"site number {i}" for i in range(100_000)]})
    try:
        with pytest.raises(ExpressionStopped) as whole:
            _run(pool, "len(df)", frame)
        assert whole.value.reason == "input" and "reads the whole table" in str(whole.value)
        assert "(1 MB at most)" in str(whole.value)
        with pytest.raises(ExpressionStopped) as named:
            _run(pool, "df['site'].str.len()", frame)
        assert "The columns it reads are larger" in str(named.value)
        assert pool.processes() == []
    finally:
        pool.close()


@pytest.mark.skipif(not limits.memory_limited(), reason="macOS: the time limit only")
def test_an_expression_beyond_the_memory_limit_is_stopped(pool: ExpressionWorkers) -> None:
    """Each step within the evaluator's budget, together more than a gigabyte on a million rows."""
    frame = pd.DataFrame({"site": np.where(np.arange(1_000_000) % 2 == 0, "Alpha", "Beta")})
    pool.seconds = 120  # the memory, not the time
    with pytest.raises(ExpressionStopped) as caught:
        _run(
            pool,
            "np.where(df['site'] == 'Alpha', df['site'].str.pad(95), df['site'].str.pad(96))",
            frame,
        )
    assert caught.value.reason in ("memory", "ended"), caught.value
    assert _run(pool, "df['site'].str.len().max()", frame.head(10)) == 5


def test_an_interrupted_call_leaves_no_process(pool: ExpressionWorkers) -> None:
    """A rerun of the page raises in the run's hook while a worker computes: the worker is ended."""

    class Rerun(BaseException):
        pass

    def hook(waits: bool) -> None:
        if waits:
            raise Rerun

    frame = pd.DataFrame({"code": pd.Series(["a" * 40 + "b"], dtype=object)})
    _run(pool, "len(df)", frame)
    before = pool.processes()
    pool.seconds = 60
    with waiting(hook), pytest.raises(Rerun):
        _run(pool, "df['code'].str.contains('(aa+)+$')", frame)
    assert _gone(before)
    assert _run(pool, "len(df)", frame) == 1  # the pool starts a worker again


def test_a_call_that_waited_tells_the_page_once_when_it_ends(pool: ExpressionWorkers) -> None:
    """The page's line says why it waits while the call waits, and goes when the call ends (stopped here)."""
    told: list[bool] = []
    frame = pd.DataFrame({"code": pd.Series(["a" * 40 + "b"], dtype=object)})
    _run(pool, "len(df)", frame)
    before = pool.processes()
    with waiting(told.append), pytest.raises(ExpressionStopped):
        _run(pool, "df['code'].str.contains('(aa+)+$')", frame)
    assert told and all(told[:-1]) and told[-1] is False and told.count(False) == 1
    # the stop ended the worker and its replacement starts in the background: whether the next call waits for it
    # depends on how fast a process starts here (under a second on Linux), so it is waited for first
    _warm_replacement(pool, before)
    told.clear()
    with waiting(told.append):
        _run(pool, "len(df)", frame)  # a quick call on a warm worker says nothing
    assert told == []


def test_a_call_costs_a_few_milliseconds_on_a_warm_worker(pool: ExpressionWorkers) -> None:
    frame = pd.DataFrame({"age": np.arange(100_000, dtype=float), "bmi": np.ones(100_000)})
    _run(pool, "df['age'] > 1", frame)
    spent = []
    for _ in range(50):
        started = time.perf_counter()
        _run(pool, "df['age'] + df['bmi']", frame)
        spent.append(time.perf_counter() - started)
    median = statistics.median(spent)
    print(f"median of 50 calls on 100,000 rows x 2 columns: {median * 1000:.1f} ms")
    assert median < 0.25


def test_a_worker_holds_no_setting_or_key_and_a_folder_of_its_own(
    pool: ExpressionWorkers, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CMP_TEST_API_KEY", "sk-not-a-real-key-000000000000")
    _run(pool, "1 + 1", None)
    (worker,) = pool._idle  # the worker itself is what is checked
    environment = ExpressionWorkers._environment({})
    assert not [name for name in environment if name.startswith("CMP_") or "KEY" in name]
    assert set(environment) <= {
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "ARROW_DEFAULT_MEMORY_POOL",
        "SYSTEMROOT",
        "MALLOC_ARENA_MAX",
    }
    folder = Path(worker.folder)
    assert folder.is_dir() and not list(folder.iterdir())
    if sys.platform.startswith("linux"):
        held = Path(f"/proc/{worker.process.pid}/environ").read_bytes()
        assert b"CMP_TEST_API_KEY" not in held
        assert "Max data size" in Path(f"/proc/{worker.process.pid}/limits").read_text()
        assert Path(f"/proc/{worker.process.pid}/oom_score_adj").read_text().strip() == "1000"
    if sys.platform == "win32":
        assert worker.job is not None
        held_limits = limits.job_limits(worker.job)
        assert (
            held_limits["memory_mb"] == 600
            and held_limits["active"] == 1
            and held_limits["kill_on_close"] == 1
        )
    pool.close()
    assert _gone([worker.process.pid]) and not folder.exists()
