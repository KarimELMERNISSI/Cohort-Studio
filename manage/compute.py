"""
The process-wide budget of worker processes for the multiple imputation (M5b).

Each session asked for cpu_count - 1 workers, so two people imputing at once started twice the processor's cores.
The budget is shared by every session of the server process: a lease grants what is free, never waits, and gives
it back when the run ends, however it ends. When fewer than two are free, the run takes place in the session's
own process (one worker), which uses none of the budget.

``CMP_MI_MAX_WORKERS`` (utils.settings) sets the total. The default is the processors of this process minus one; a
container's CPU quota is not visible in that count, so set the variable there.

The cells of an imputation are the same in one process and in several: every chain is drawn on one BLAS thread
(``biostats.mi.engine.one_blas_thread``), and so is everything a replay compares: the server's own process runs
its linear algebra on one thread from its start, and a replay from its first step
(:func:`one_blas_thread_for_this_process`). The budget, and whether several processes pay for their start
(:func:`pays_to_parallel`), decide the time a draw takes, on the page and in a replay alike.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from utils.settings import get_settings

SETTING = "CMP_MI_MAX_WORKERS"
# rows x variables x m x iterations above which several processes pay for their start-up (several seconds)
PARALLEL_WORK = 5e7


def pays_to_parallel(rows: int, variables: int, m: int, iterations: int) -> bool:
    """Whether an imputation of this size is worth several processes: the page's default for its switch, and what a
    replay asks before it leases workers for a draw."""
    return rows * variables * m * iterations > PARALLEL_WORK


def one_blas_thread_for_this_process() -> None:
    """
    The linear algebra of this process on one thread, set once and never given back: by the server when it starts
    (``main.py``), by a replay as it begins (``manage.replay``), and by the test suite.

    The rule is one: what a fingerprint is taken of is computed on one BLAS thread, when it is recorded and when it
    is replayed. A BLAS adds a long product up in another order with several threads than with one, and the last
    digits follow: an imputation's cells (a ``norm`` variable observed on more than 10,000 rows), the scores of a
    PCA or a FAMD of a large table (50,000 rows by 13 columns differed; 20,000 by 6 did not), and the values of a
    single imputation by chained equations (MICE; 50,000 by 13 differed).

    The setting is the process's, and other code moves it: scikit-learn's KMeans sets one thread for the time of
    each of its runs and restores what it found, from whichever session's thread runs it. What another session
    computed meanwhile ran partly on one thread and partly on all; and a limit that the imputation set for the time
    of a draw, and restored, was undone the same way, then left wrong for good. Set once and kept, nothing moves it:
    what a session computes never depends on what another one is doing, nor on the number of cores of the machine.
    The sessions of a server are threads of one process and shared one pool of BLAS threads; at a cohort's
    sizes one thread costs little (measured on 14 cores: nothing for a PCA, a correlation matrix or K-Means, up to
    half as long again for a regression of 50,000 rows and more). The loops other libraries run on OpenMP keep
    their threads.
    """
    from biostats.mi.engine import one_blas_thread

    one_blas_thread()


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


__all__ = [
    "PARALLEL_WORK",
    "SETTING",
    "WorkerBudget",
    "default_total",
    "mi_workers",
    "one_blas_thread_for_this_process",
    "pays_to_parallel",
]
