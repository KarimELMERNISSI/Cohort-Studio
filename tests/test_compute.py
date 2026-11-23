"""The process-wide budget of worker processes for the multiple imputation (manage/compute.py, M5b)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from manage.compute import SETTING, WorkerBudget, default_total, pays_to_parallel


def test_a_lease_grants_what_is_free_and_gives_it_back() -> None:
    budget = WorkerBudget(4)
    with budget.lease(3) as first:
        assert first == 3 and budget.free == 1
        with budget.lease(3) as second:  # one free: the run takes place in its own process
            assert second == 1 and budget.free == 1
    assert budget.free == 4
    with budget.lease(1) as one:  # a sequential run takes none of the budget
        assert one == 1 and budget.free == 4


def test_a_failed_or_interrupted_run_gives_its_workers_back() -> None:
    class Rerun(BaseException):  # a click reruns the page: Streamlit raises a BaseException
        pass

    def fail(error: type[BaseException]) -> None:
        raise error

    budget = WorkerBudget(4)
    for error in (RuntimeError, Rerun):
        with pytest.raises(error), budget.lease(4):
            fail(error)
        assert budget.free == 4


def test_concurrent_sessions_never_exceed_the_budget() -> None:
    budget = WorkerBudget(5)
    granted: list[int] = []
    together = threading.Barrier(8, timeout=10)  # every session holds its lease at the same time

    def session() -> None:
        with budget.lease(3) as workers:
            granted.append(workers)
            together.wait()

    threads = [threading.Thread(target=session) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(granted, reverse=True) == [3, 2, 1, 1, 1, 1, 1, 1] and budget.free == 5


def test_the_budget_comes_from_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    """CMP_MI_MAX_WORKERS is validated with the other settings (tests/test_settings.py); unset, processors - 1."""
    from manage import compute

    monkeypatch.setattr(compute, "_budget", None)
    monkeypatch.setenv(SETTING, "3")
    assert compute.mi_workers().total == 3
    monkeypatch.setattr(compute, "_budget", None)
    monkeypatch.delenv(SETTING)
    assert compute.mi_workers().total == default_total() >= 1
    with pytest.raises(ValueError, match="at least 1"):
        WorkerBudget(0)


def test_a_draw_pays_for_worker_processes_above_the_threshold() -> None:
    """The Enrich page's default for its switch, and what a replay asks before it leases."""
    assert not pays_to_parallel(1_206, 12, 20, 10)  # the tutorial cohort
    assert not pays_to_parallel(50_000, 10, 10, 10) and pays_to_parallel(50_001, 10, 10, 10)


FRESH = """
import json
from threadpoolctl import threadpool_info, threadpool_limits

def blas():
    return [[str(p.get("internal_api")), int(p.get("num_threads") or 0)] for p in threadpool_info()
            if p.get("user_api") == "blas"]

seen = {"at start": blas()}
from manage.compute import one_blas_thread_for_this_process
one_blas_thread_for_this_process()
seen["once set"] = blas()
import numpy, scipy.linalg
scipy.linalg.svd(numpy.eye(3)); numpy.linalg.inv(numpy.eye(3))
seen["after a later computation"] = blas()
with threadpool_limits(limits=1, user_api="blas"):  # as scikit-learn's KMeans holds and gives back
    pass
seen["after another code's limit"] = blas()
print(json.dumps(seen))
"""


def test_one_blas_thread_reaches_every_library_of_a_fresh_process() -> None:
    """Set when a server starts, before any page has computed: SciPy brings a BLAS of its own, which a setting made
    before it is loaded would not reach (it would start on every thread). In a process that has imported neither,
    every BLAS library is on one thread once the setting is made, after a later computation through each, and after
    another code's limit is set and given back."""
    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if not k.upper().endswith("_NUM_THREADS")}
    env["PYTHONPATH"] = os.pathsep.join([str(root), env.get("PYTHONPATH", "")])
    done = subprocess.run(
        [sys.executable, "-B", "-c", FRESH],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    seen = json.loads(done.stdout.strip().splitlines()[-1])
    for moment in ("once set", "after a later computation", "after another code's limit"):
        assert seen[moment] and all(threads == 1 for _, threads in seen[moment]), (moment, seen)
    # no library appeared after the setting: each was loaded, and set, by it
    assert len(seen["after a later computation"]) == len(seen["once set"])
