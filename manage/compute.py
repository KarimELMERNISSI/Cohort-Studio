"""
The process-wide budget of worker processes for the multiple imputation (M5b).

Each session asked for cpu_count - 1 workers, so two people imputing at once started twice the processor's cores.
The budget is shared by every session of the server process: a lease grants what is free, never waits, and gives
it back when the run ends, however it ends. When fewer than two are free, the run takes place in the session's
own process (one worker), which uses none of the budget.

``CMP_MI_MAX_WORKERS`` (utils.settings) sets the total. The default is the processors of this process minus one; a
container's CPU quota is not visible in that count, so set the variable there.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from utils.settings import get_settings

SETTING = "CMP_MI_MAX_WORKERS"


class WorkerBudget:
    """Worker processes shared by the sessions of this server process."""

    def __init__(self, total: int) -> None:
        if total < 1:
            raise ValueError(f"the worker budget is at least 1, not {total}")
        self.total = total
        self._used = 0
        self._lock = threading.Lock()

    @property
    def free(self) -> int:
        with self._lock:
            return self.total - self._used

    @contextmanager
    def lease(self, wanted: int) -> Iterator[int]:
        """
        Grants ``min(wanted, free)`` worker processes, or 1 (a run in this process, outside the budget) when one or
        none would be granted; returns them when the run ends.
        """
        with self._lock:
            free = self.total - self._used
            granted = min(wanted, free) if wanted > 1 and free > 1 else 1
            taken = granted if granted > 1 else 0
            self._used += taken
        try:
            yield granted
        finally:
            with self._lock:
                self._used -= taken


def default_total() -> int:
    return max(1, (os.process_cpu_count() or 2) - 1)


_budget: WorkerBudget | None = None
_create = threading.Lock()


def mi_workers() -> WorkerBudget:
    """The budget of this server process, from the ``CMP_MI_MAX_WORKERS`` setting, read once."""
    global _budget
    with _create:
        if _budget is None:
            _budget = WorkerBudget(get_settings().statistics.mi_max_workers or default_total())
        return _budget


__all__ = ["SETTING", "WorkerBudget", "default_total", "mi_workers"]
