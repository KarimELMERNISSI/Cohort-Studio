"""The process-wide budget of worker processes for the multiple imputation (manage/compute.py, M5b)."""

from __future__ import annotations

import threading

import pytest

from manage.compute import SETTING, WorkerBudget, default_total


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
