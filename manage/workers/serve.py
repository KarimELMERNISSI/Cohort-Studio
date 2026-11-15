"""
A worker's loop: the jobs of :mod:`manage.workers.wire`, computed one at a time with the same evaluator as the server
(:func:`manage.workers.wire.compute_job`). A worker imports ``core.expressions`` only, never ``enrich``, the AI packages
or Streamlit.
"""

from __future__ import annotations

import gc
import os
import pickle
import sys
from typing import IO

from manage.workers.limits import cpu_deadline
from manage.workers.wire import compute_job, pack, read_exactly, read_length, write_message

# after a job of more bytes than this, the memory it held is given back to the system
RELEASE_AFTER = 16 * 2**20


def main(seconds: float) -> None:
    """
    The worker's loop: one job at a time, read from its input, computed, answered, until its input closes (the pool
    closed it, or the server ended). Its standard output is moved aside first, so that a library that prints cannot
    break the channel: the answers go to the handle saved before. Before each job its CPU time is limited (POSIX);
    after a large one it gives its memory back (the collector, and pyarrow's pool).
    """
    channel = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    quiet = os.open(os.devnull, os.O_WRONLY)
    os.dup2(quiet, sys.stdout.fileno())
    os.close(quiet)  # the standard output now holds its copy
    with open(os.devnull, "w", encoding="utf-8") as silent:  # for the life of the worker
        sys.stdout = silent
        try:
            _serve(channel, seconds)
        finally:
            sys.stdout = sys.__stdout__


def _serve(channel: IO[bytes], seconds: float) -> None:
    """The jobs, one at a time, until the worker's input closes."""
    source = sys.stdin.buffer
    write_message(channel, pack({"ready": os.getpid()}))
    while True:
        try:
            size = read_length(source)
            payload = read_exactly(source, size)
        except EOFError:
            return
        cpu_deadline(seconds)
        try:
            answer = pack(compute_job(pickle.loads(payload)))
        except MemoryError:
            answer = pack({"stopped": ("memory", None), "warnings": []})
        except Exception as exc:  # an answer that cannot be sent: its message
            answer = pack({"error": str(exc), "warnings": []})
        del payload
        write_message(channel, answer)
        del answer
        if size > RELEASE_AFTER:
            gc.collect()
            arrow = sys.modules.get("pyarrow")
            if arrow is not None:
                arrow.default_memory_pool().release_unused()
