"""
The pool of workers that compute expressions, shared by every session of a server process (:class:`ExpressionWorkers`,
a runner of :mod:`core.expressions.runner`). Each expression leases a warm worker, which computes it under the time
limit and the memory limit, and gives it back.

- **A worker** is the real interpreter (``biostats.processes.interpreter``), started isolated (``-I``), writing no
  bytecode (``-B``), in an empty folder of its own, with no setting or key in its environment and one thread for the
  linear algebra (its memory: about 70 MB rather than 480). It imports ``core.expressions`` only.
- **The limits:** the memory's is the worker's own (:mod:`manage.workers.limits`); the time's is the pool's, which ends
  the worker at the limit (the only way to stop Python's pattern engine, which holds its interpreter while it
  searches). A worker is replaced after a stop, after a MemoryError, and after 500 jobs.
- **Waiting:** a call that waits more than a second calls the run's hook (:func:`waiting`) every tenth of a second, so
  that a page can say why it waits, and a rerun can interrupt it: the worker computing is then ended. When that call
  ends, the hook is told once, so that the page's line goes with it.
- **A stop is remembered:** the same job (its bytes' digest, no data) is refused at once, with the same sentence.
- **Fail closed:** when no worker can start, an expression is not computed (:class:`WorkerUnavailable`); it is never
  computed in the server's process instead.
"""

from __future__ import annotations

import ast
import atexit
import contextlib
import contextvars
import hashlib
import logging
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from biostats.processes import interpreter
from core.expressions.errors import ExpressionStopped, WorkerUnavailable
from core.expressions.runner import columns_read
from core.expressions.safe_eval import validate_tree
from manage.workers import limits
from manage.workers.wire import (
    answer_limit,
    job_for,
    pack,
    read_answer,
    read_exactly,
    read_length,
    rows_of,
    stopped,
    value_of,
    write_message,
)

logger = logging.getLogger(__name__)

START_SECONDS = 60.0  # a worker that has not said it is ready by then did not start (2-4 s; 17-29 s on a busy machine)
IDLE_SECONDS = 600.0  # a worker idle this long is retired, one staying warm
JOBS_BEFORE_REPLACEMENT = 500
SLICE = 0.1  # a waiting call looks at its answer, and at the run's hook, this often
HOOK_AFTER = 1.0  # a call that has waited this long calls the run's hook
REMEMBERED = 256  # the stopped jobs remembered (their digests)
ENDED_WAIT = 5.0  # a closed worker that has not returned by then is ended
FOLDER_TRIES = 10  # an ended worker's folder, removed again every 0.2 s until it is gone (Windows)
ROOT = str(Path(__file__).resolve().parents[2])

BUSY = "The server is computing other formulas and criteria: try again in a moment."
NOT_STARTED = (
    "Formulas and criteria cannot be computed on this server at the moment: the process that computes them did not "
    "start. Try again in a moment; if it lasts, tell the person who runs this server."
)


class _NotReady(RuntimeError):
    """A worker that did not say it was ready: why, in fixed words and numbers (for the log)."""


_BOOT = (
    "import sys; sys.path.insert(0, sys.argv[1]); "
    "from manage.workers.limits import limit_self; limit_self(int(sys.argv[2])); "
    "from manage.workers.serve import main; main(float(sys.argv[3]))"
)
_PATTERN_METHODS = frozenset(
    {"contains", "match", "fullmatch", "replace", "split", "count", "findall", "extract"}
)

_waiting: contextvars.ContextVar[Callable[[bool], None] | None] = contextvars.ContextVar(
    "expression_waiting", default=None
)


@contextlib.contextmanager
def waiting(hook: Callable[[bool], None] | None) -> Iterator[None]:
    """Inside the block, a call that waits more than a second calls ``hook(True)`` every tenth of a second (where a
    page says why it waits, and where Streamlit can stop the run), then ``hook(False)`` once when it ends."""
    token = _waiting.set(hook)
    try:
        yield
    finally:
        _waiting.reset(token)


class _Wait:
    """The run's hook for one call: called while the call has waited a second or more, told once when it ends."""

    def __init__(self) -> None:
        self.hook = _waiting.get()
        self.started = time.monotonic()
        self.told = False

    def tick(self) -> None:
        if self.hook is not None and time.monotonic() - self.started >= HOOK_AFTER:
            self.told = True
            self.hook(True)  # where a rerun interrupts the wait: the worker is ended by the caller

    def end(self) -> None:
        if self.told and self.hook is not None:
            self.told = False
            self.hook(False)


def _holds_pattern(tree: ast.AST) -> bool:
    """Whether an expression searches a text by a pattern (``.str.contains`` and its kind)."""
    return any(
        isinstance(node, ast.Attribute)
        and node.attr in _PATTERN_METHODS
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "str"
        for node in ast.walk(tree)
    )


class _Worker:
    """One worker process, its job (Windows), its folder, and the thread that reads its answers."""

    def __init__(self, process: subprocess.Popen[bytes], job: int | None, folder: str) -> None:
        self.process, self.job, self.folder = process, job, folder
        self._finishing = threading.Lock()
        self.answers: queue.Queue[tuple[str, bytes | None]] = queue.Queue()
        self.limit = (
            0  # the largest answer the job being computed may give (0: any, for the ready message)
        )
        self.jobs = 0
        self.last_used = time.monotonic()
        threading.Thread(target=self._read, daemon=True, name="expression-worker-reader").start()

    def _read(self) -> None:
        source = self.process.stdout
        assert source is not None
        while True:
            try:
                size = read_length(source)
                if self.limit and size > self.limit:
                    self.answers.put(("too large", None))
                    return
                self.answers.put(("answer", read_exactly(source, size)))
            except EOFError, OSError, ValueError:
                self.answers.put(("ended", None))
                return

    def alive(self) -> bool:
        return self.process.poll() is None

    def end(self) -> None:
        """Ends the process now, and gives back what it held."""
        with contextlib.suppress(OSError):
            self.process.kill()
        self._finish(wait=ENDED_WAIT)

    def close(self) -> None:
        """Closes its input: an idle worker returns; one that does not is ended."""
        with contextlib.suppress(OSError, ValueError):
            if self.process.stdin is not None:
                self.process.stdin.close()
        try:
            self.process.wait(ENDED_WAIT)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                self.process.kill()
        self._finish(wait=ENDED_WAIT)

    def _finish(self, *, wait: float) -> None:
        with contextlib.suppress(subprocess.TimeoutExpired, OSError):
            self.process.wait(wait)
        for stream in (self.process.stdin, self.process.stdout):
            with contextlib.suppress(OSError, ValueError):
                if stream is not None:
                    stream.close()
        with (
            self._finishing
        ):  # the job's handle closed once: closed twice, its number could be another handle's
            job, self.job = self.job, None
        if job is not None and sys.platform == "win32":
            limits.close_job(job)
        # Windows keeps a process's folder until it has wholly ended: removed again for a moment if need be
        for _ in range(FOLDER_TRIES):
            shutil.rmtree(self.folder, ignore_errors=True)
            if not os.path.exists(self.folder):
                break
            time.sleep(0.2)


class ExpressionWorkers:
    """
    The runner that computes each expression in a warm worker, under ``seconds`` and ``memory_mb``; at most
    ``workers`` of them at once, shared by every session of the process. Validated here first, so that an expression
    the evaluator refuses costs no round trip; validated again in the worker.
    """

    def __init__(self, *, workers: int = 2, seconds: float = 30.0, memory_mb: int = 2048) -> None:
        self.workers = max(int(workers), 1)
        self.seconds = float(seconds)
        self.memory_mb = int(memory_mb)
        self._idle: list[_Worker] = []
        self._busy: set[_Worker] = set()
        self._starting = 0
        self._changed = threading.Condition()
        self._stopped: OrderedDict[bytes, tuple[str, float | None, bool]] = OrderedDict()
        self._closed = False
        atexit.register(self.close)

    # --- the runner -------------------------------------------------------------------------------------------------

    def __call__(
        self,
        tree: ast.Expression,
        df: pd.DataFrame | None,
        *,
        policy: Any,
        extra_locals: Mapping[str, Any],
    ) -> Any:
        validate_tree(tree, policy=policy, extra_names=frozenset(extra_locals))
        job = job_for(tree, df, policy=policy, extra_locals=extra_locals)
        payload = pack(job)
        if (
            len(payload) > self.memory_mb * 2**20 // 2
        ):  # half the memory limit: the worker needs room to compute
            whole = df is not None and columns_read(tree, policy.frame_names) is None
            raise stopped("input", self.memory_mb // 2, whole=whole)
        digest = hashlib.blake2b(payload, digest_size=16).digest()
        with self._changed:
            remembered = self._stopped.get(digest)
        if remembered is not None:
            reason, limit, pattern = remembered
            raise stopped(reason, limit, pattern=pattern, again=True)
        wait = _Wait()
        try:
            worker = self._lease(wait)
            healthy = False
            try:
                answer = self._compute(worker, payload, rows_of(job), wait)
                healthy = "stopped" not in answer
            except ExpressionStopped as stop:
                pattern = stop.reason == "time" and _holds_pattern(tree)
                self._remember(digest, stop.reason, stop.limit, pattern)
                if pattern:  # with the advice on patterns that repeat a repetition
                    raise stopped(stop.reason, stop.limit, pattern=True) from None
                raise
            finally:
                self._give_back(worker, healthy=healthy)
        finally:
            wait.end()
        if "stopped" in answer:
            reason, _ = answer["stopped"]
            self._remember(digest, str(reason), self.memory_mb, False)
            raise stopped(str(reason), self.memory_mb)
        return value_of(answer)

    def _compute(self, worker: _Worker, payload: bytes, rows: int, wait: _Wait) -> dict[str, Any]:
        """Sends a job and waits for its answer, up to the time limit; ends the worker on a stop or an interruption
        (the caller then does not give it back healthy)."""
        worker.limit = answer_limit(len(payload), rows)
        worker.jobs += 1
        try:
            assert worker.process.stdin is not None
            write_message(worker.process.stdin, payload)
        except OSError, ValueError:
            raise stopped("ended") from None
        started = time.monotonic()
        while True:
            try:
                kind, data = worker.answers.get(timeout=SLICE)
                break
            except queue.Empty:
                waited = time.monotonic() - started
                if waited >= self.seconds:
                    logger.warning(
                        "An expression was stopped after %.0f seconds (its limit); %d rows were sent, the worker is "
                        "replaced",
                        waited,
                        rows,
                    )
                    raise stopped("time", self.seconds) from None
                wait.tick()
        if kind == "too large":
            logger.warning(
                "An expression's answer was larger than the server takes; the worker is replaced"
            )
            raise stopped("result")
        if kind != "answer" or data is None:
            logger.warning("A worker ended while it computed an expression; it is replaced")
            raise stopped("ended")
        answer = read_answer(data)
        if "stopped" in answer:
            logger.warning(
                "An expression was stopped by the memory limit (%d MB); %d rows were sent, the worker is replaced",
                self.memory_mb,
                rows,
            )
        return answer

    def _remember(self, digest: bytes, reason: str, limit: float | None, pattern: bool) -> None:
        if reason in ("time", "memory", "result"):
            with self._changed:
                self._stopped[digest] = (reason, limit, pattern)
                while len(self._stopped) > REMEMBERED:
                    self._stopped.popitem(last=False)

    # --- the workers ------------------------------------------------------------------------------------------------

    def _lease(self, wait: _Wait | None = None) -> _Worker:
        """A worker for one job: an idle one, else a new one while fewer than ``workers`` run; else waits, up to the
        time limit and the start limit, for one to be given back."""
        give_up = time.monotonic() + self.seconds + START_SECONDS
        while True:
            with self._changed:
                if self._closed:
                    raise WorkerUnavailable(NOT_STARTED)
                self._retire_idle()
                while self._idle:
                    worker = self._idle.pop()
                    if worker.alive():
                        self._busy.add(worker)
                        return worker
                    threading.Thread(target=worker.end, daemon=True).start()
                if len(self._busy) + self._starting < self.workers:
                    self._starting += 1
                    break
                remaining = give_up - time.monotonic()
                if remaining <= 0:
                    raise WorkerUnavailable(BUSY)
                self._changed.wait(timeout=min(SLICE, remaining))
            if wait is not None:
                wait.tick()
        try:
            worker = self._start(wait)
        except BaseException as exc:
            with self._changed:
                self._starting -= 1
                self._changed.notify_all()
            if not isinstance(exc, Exception):
                raise  # a rerun of the page while the worker started: ended by _start
            if isinstance(exc, _NotReady):  # its own words: fixed, with numbers only
                logger.warning("A worker for formulas and criteria did not start: %s", str(exc))
            else:  # another error's type and code, never its message (a path names the server's account)
                code = getattr(exc, "winerror", None)
                logger.warning(
                    "A worker for formulas and criteria did not start (%s%s)",
                    type(exc).__name__,
                    f", code {code}" if isinstance(code, int) else "",
                )
            raise WorkerUnavailable(NOT_STARTED) from None
        with self._changed:
            self._starting -= 1
            self._busy.add(worker)
        return worker

    def _give_back(self, worker: _Worker, *, healthy: bool) -> None:
        with self._changed:
            self._busy.discard(worker)
            keep = (
                healthy
                and not self._closed
                and worker.alive()
                and worker.jobs < JOBS_BEFORE_REPLACEMENT
            )
            if keep:
                worker.last_used = time.monotonic()
                self._idle.append(worker)
            self._changed.notify_all()
            replace = not keep and not self._closed and not self._idle
        if not keep:
            threading.Thread(target=worker.end, daemon=True, name="expression-worker-end").start()
        if (
            replace
        ):  # a worker ended (a stop, an interruption, its 500 jobs): the next call finds one warm
            self.warm()

    def _retire_idle(self) -> None:
        """Ends the workers idle for more than ten minutes, one staying warm (called under the lock)."""
        now = time.monotonic()
        old = [w for w in self._idle if now - w.last_used > IDLE_SECONDS]
        for worker in old[: max(len(self._idle) - 1, 0)]:
            self._idle.remove(worker)
            threading.Thread(target=worker.close, daemon=True).start()

    def _start(self, wait: _Wait | None = None) -> _Worker:
        executable, needs = interpreter()
        folder = tempfile.mkdtemp(prefix="cohortstudio-worker-")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(
                [executable, "-I", "-B", "-c", _BOOT, ROOT, str(self.memory_mb), str(self.seconds)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # a library's message can quote a cell
                env=self._environment(needs),
                cwd=folder,
                close_fds=True,
                creationflags=flags,
            )
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        job = None
        if sys.platform == "win32":
            try:
                job = limits.job_for_worker(self.memory_mb)
                limits.join(job, process)
            except Exception:
                process.kill()
                if job is not None:
                    limits.close_job(job)
                shutil.rmtree(folder, ignore_errors=True)
                raise
        worker = _Worker(process, job, folder)
        deadline = time.monotonic() + START_SECONDS
        try:
            while True:
                try:
                    kind, data = worker.answers.get(timeout=SLICE)
                    break
                except queue.Empty:
                    if time.monotonic() >= deadline:
                        kind, data = "late", None
                        break
                    if wait is not None:
                        wait.tick()
        except BaseException:  # a rerun of the page while it starts
            worker.end()
            raise
        ready = read_answer(data) if kind == "answer" and data is not None else {}
        if "ready" not in ready:
            worker.end()  # waits for the process: its exit code is known after
            if kind == "late":
                raise _NotReady(f"it was not ready after {START_SECONDS:.0f} seconds")
            raise _NotReady(f"it ended before it was ready (exit code {worker.process.returncode})")
        return worker

    @staticmethod
    def _environment(needs: Mapping[str, str]) -> dict[str, str]:
        """What a worker's environment holds: what the interpreter needs, one thread for the linear algebra, the system
        memory pool for pyarrow; never a setting or a key of the server's."""
        env = dict(needs)
        env.update(
            {
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "ARROW_DEFAULT_MEMORY_POOL": "system",
            }
        )
        if sys.platform == "win32":
            env["SYSTEMROOT"] = limits.windows_directory()
        else:
            env["MALLOC_ARENA_MAX"] = "2"
        return env

    def warm(self) -> None:
        """Starts one worker in the background, so that the first expression does not wait for it."""

        def start() -> None:
            try:
                worker = self._lease()
            except Exception:  # said by the first expression that needs one
                return
            self._give_back(worker, healthy=True)

        threading.Thread(target=start, daemon=True, name="expression-worker-warm").start()

    def processes(self) -> list[int]:
        """The ids of the workers running (for the tests)."""
        with self._changed:
            return [w.process.pid for w in [*self._idle, *self._busy] if w.alive()]

    def close(self) -> None:
        """Ends every worker: the idle ones return at the end of their input; any still running after 5 s is ended."""
        with self._changed:
            self._closed = True
            workers, self._idle = [*self._idle, *self._busy], []
            self._busy = set()
            self._changed.notify_all()
        for worker in workers:
            worker.close()


__all__ = ["BUSY", "NOT_STARTED", "ExpressionWorkers", "waiting"]
