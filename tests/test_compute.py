"""The process-wide budget of worker processes for the multiple imputation (manage/compute.py, M5b)."""

from __future__ import annotations

import threading
import time

import pytest

from manage.compute import SETTING, WorkerBudget, default_total, total_from


def test_a_lease_grants_what_is_free_and_gives_it_back() -> None:
    budget = WorkerBudget(4)
    with budget.lease(3) as first:
        assert first == 3 and budget.free == 1
        with budget.lease(3) as second:  # one free: the run takes place in its own process
            assert second == 1 and budget.free == 1
    assert budget.free == 4
    with budget.lease(1) as one:  # a sequential run takes none of the budget
        assert one == 1 and budget.free == 4


def test_a_failed_run_gives_its_workers_back() -> None:
    budget = WorkerBudget(4)
    with pytest.raises(RuntimeError), budget.lease(4):
        raise RuntimeError("the imputation failed")
    assert budget.free == 4


def test_concurrent_sessions_never_exceed_the_budget() -> None:
    budget = WorkerBudget(5)
    in_use: list[int] = []
    lock = threading.Lock()
    current = [0]

    def session() -> None:
        with budget.lease(3) as granted:
            taken = granted if granted > 1 else 0
            with lock:
                current[0] += taken
                in_use.append(current[0])
            time.sleep(0.01)
            with lock:
                current[0] -= taken

    threads = [threading.Thread(target=session) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert max(in_use) <= 5 and budget.free == 5


def test_the_setting_names_itself_when_wrong() -> None:
    assert total_from("") == default_total() >= 1
    assert total_from(" 3 ") == 3
    for wrong in ("0", "-2", "two", "1.5"):
        with pytest.raises(ValueError, match=SETTING):
            total_from(wrong)
    with pytest.raises(ValueError, match="at least 1"):
        WorkerBudget(0)
