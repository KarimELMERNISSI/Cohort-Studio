"""
Parallel imputation in a separate Python process.

Worker processes started with ``spawn`` re-import the parent's ``__main__`` module. Inside Streamlit (and its
AppTest) that module is the page script, which must not run again in the workers. ``impute(workers > 1)``
therefore runs ``python -m biostats.mi.parallel`` with the data and the specification in a temporary folder:
there, ``__main__`` is this module and the process pool is safe. Progress is written to a small file that
the caller polls; the result comes back the same way. Each chain has its own random stream, so the result is
the same as a sequential run.

An interrupted run (a click reruns the page: Streamlit raises a BaseException in the progress callback) stops the
whole run: the interpreter started here, and its pool workers, which exit with it (``engine._init_worker``). The
temporary folder holds a copy of the data: it is deleted however the run ends.
"""

from __future__ import annotations

import contextlib
import logging
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

IN_SUBPROCESS = "CMP_MI_SUBPROCESS"


def in_subprocess() -> bool:
    """Whether this process is the worker of a parallel imputation (a process variable, not a setting)."""
    return bool(os.environ.get(IN_SUBPROCESS))


def interpreter() -> tuple[str, dict[str, str]]:
    """
    The interpreter to start, and what its environment needs. On Windows, the python.exe of a virtual environment
    is a launcher that starts the real interpreter as its own process: stopping the launcher would leave the
    interpreter running. The base interpreter is started instead, told which environment it serves, as
    multiprocessing does (bpo-35797).
    """
    base = getattr(sys, "_base_executable", None) or sys.executable
    if sys.platform == "win32" and os.path.normcase(base) != os.path.normcase(sys.executable):
        return base, {"__PYVENV_LAUNCHER__": sys.executable}
    return sys.executable, {}


def _remove(folder: str) -> None:
    """
    Delete the job folder, which holds a copy of the data. A file still held by a process that is stopping is
    tried again for a few seconds; a folder left behind is logged, never raised over the run's own outcome (an
    interruption must reach Streamlit as it is).
    """
    for _ in range(50):
        try:
            shutil.rmtree(folder)
            return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(0.1)
    logger.warning(
        "The temporary folder %s of a parallel imputation could not be deleted",
        os.path.basename(folder),
    )


def run_in_subprocess(
    df: Any, spec: Any, workers: int, progress: Callable[[float], None] | None = None
) -> Any:
    """Run ``impute(df, spec, workers=workers)`` in a fresh interpreter and return its ImputationRun."""
    folder = tempfile.mkdtemp(prefix="cmp_mi_")
    try:
        return _run(Path(folder), df, spec, workers, progress)
    finally:
        _remove(folder)


def _run(
    tmp: Path, df: Any, spec: Any, workers: int, progress: Callable[[float], None] | None
) -> Any:
    root = Path(__file__).resolve().parents[2]
    job, result, ticks = tmp / "job.pkl", tmp / "run.pkl", tmp / "progress.txt"
    with job.open("wb") as handle:
        pickle.dump((df, spec, workers), handle, protocol=pickle.HIGHEST_PROTOCOL)
    executable, needs = interpreter()
    env = {
        **os.environ,
        **needs,
        IN_SUBPROCESS: "1",
        "PYTHONPATH": os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")]),
    }
    errors = tmp / "stderr.txt"
    with errors.open("wb") as err:  # a file, not a pipe: many warnings cannot block the child
        process = subprocess.Popen(
            [executable, "-m", "biostats.mi.parallel", str(job), str(result), str(ticks)],
            cwd=root,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=err,
        )
        try:
            while process.poll() is None:
                if progress is not None and ticks.exists():
                    with contextlib.suppress(ValueError, OSError):
                        progress(float(ticks.read_text(encoding="utf-8") or 0))
                time.sleep(0.3)
        finally:
            # interrupted (a click reruns the page: Streamlit raises a BaseException in the progress callback):
            # the run must not go on, unseen; its pool workers exit with it
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    stderr = errors.read_text(encoding="utf-8", errors="replace")
    if process.returncode != 0 or not result.exists():
        last = (
            stderr.strip().splitlines()[-1] if stderr.strip() else f"exit code {process.returncode}"
        )
        raise RuntimeError(f"the parallel imputation failed: {last}")
    with result.open("rb") as handle:
        run = pickle.load(handle)
    if progress is not None:
        progress(1.0)
    return run


def _main(job: str, result: str, ticks: str) -> None:
    from biostats.mi.engine import impute

    with open(job, "rb") as handle:
        df, spec, workers = pickle.load(handle)

    def report(fraction: float) -> None:
        tmp = f"{ticks}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(f"{fraction:.4f}")
        os.replace(tmp, ticks)

    run = impute(df, spec, workers=workers, progress=report)
    with open(result, "wb") as handle:
        pickle.dump(run, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    _main(*sys.argv[1:4])
