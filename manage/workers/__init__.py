"""
The workers that compute expressions (formulas and criteria) under a time limit and a memory limit. Headless: nothing
here reaches Streamlit or the AI packages.

- :mod:`manage.workers.wire`: what a worker is sent and gives back, and the reader that takes nothing else back;
- :mod:`manage.workers.serve`: what a worker computes, and its loop;
- :mod:`manage.workers.limits`: the limits on each platform (a Job Object on Windows, the worker's own on Linux);
- :mod:`manage.workers.pool`: the pool shared by every session of a server process;
- :func:`install_for`: the pool installed for every expression of the process, from its settings.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from manage.workers.pool import ExpressionWorkers

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_pool: ExpressionWorkers | None = None


def expression_workers(settings: Any) -> ExpressionWorkers:
    """The pool of this server process, made once from its settings (``CMP_FORMULA_*``), its first worker started in
    the background; the log says its limits, and when they may need more memory than the server has."""
    global _pool
    from manage.workers import limits
    from manage.workers.pool import ExpressionWorkers

    with _lock:
        if _pool is None:
            formulas = settings.formulas
            _pool = ExpressionWorkers(
                workers=formulas.workers, seconds=formulas.seconds, memory_mb=formulas.memory_mb
            )
            if limits.memory_limited():
                logger.info(
                    "Formulas and criteria are computed in %d worker processes, each stopped after %d seconds or %d "
                    "MB (memory limited by %s)",
                    formulas.workers,
                    formulas.seconds,
                    formulas.memory_mb,
                    limits.mechanism(),
                )
            else:
                logger.info(
                    "Formulas and criteria are computed in %d worker processes, each stopped after %d seconds (the "
                    "memory is not limited on this system)",
                    formulas.workers,
                    formulas.seconds,
                )
            machine = limits.machine_memory_mb()
            needed = formulas.workers * formulas.memory_mb
            if machine and needed > machine / 2:
                logger.warning(
                    "The workers for formulas and criteria may use up to %d MB, more than half of this server's "
                    "memory (%d MB): lower CMP_FORMULA_WORKERS or CMP_FORMULA_MEMORY_MB",
                    needed,
                    machine,
                )
            _pool.warm()
        return _pool


def install_for(settings: Any) -> bool:
    """The pool installed for every expression of this process (:mod:`core.expressions.runner`); False when the
    development switch ``CMP_FORMULA_IN_PROCESS`` keeps them in this process, without a limit, which the log warns."""
    from core.expressions.runner import install

    if settings.development.formula_in_process:
        logger.warning(
            "CMP_FORMULA_IN_PROCESS is on: formulas and criteria are computed in the server's own process, without a "
            "time limit or a memory limit"
        )
        return False
    install(expression_workers(settings))
    return True


__all__ = ["expression_workers", "install_for"]
