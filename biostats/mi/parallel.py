"""
Parallel imputation in a separate Python process.

Worker processes started with ``spawn`` re-import the parent's ``__main__`` module. Inside Streamlit (and its
AppTest) that module is the page script, which must not run again in the workers. ``impute(workers > 1)``
therefore runs ``python -m biostats.mi.parallel`` with the data and the specification in a temporary file:
there, ``__main__`` is this module and the process pool is safe. Progress is written to a small file that
the caller polls; the result comes back the same way. Each chain has its own random stream, so the result is
the same as a sequential run.
"""

from __future__ import annotations

import contextlib
import os
import pickle
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

IN_SUBPROCESS = "CMP_MI_SUBPROCESS"


def run_in_subprocess(
    df: Any, spec: Any, workers: int, progress: Callable[[float], None] | None = None
) -> Any:
    """Run ``impute(df, spec, workers=workers)`` in a fresh interpreter and return its ImputationRun."""
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="cmp_mi_") as tmp:
        job, result, ticks = Path(tmp, "job.pkl"), Path(tmp, "run.pkl"), Path(tmp, "progress.txt")
        with job.open("wb") as handle:
            pickle.dump((df, spec, workers), handle, protocol=pickle.HIGHEST_PROTOCOL)
        env = {
            **os.environ,
            IN_SUBPROCESS: "1",
            "PYTHONPATH": os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")]),
        }
        errors = Path(tmp, "stderr.txt")
        with errors.open("wb") as err:  # a file, not a pipe: many warnings cannot block the child
            process = subprocess.Popen(
                [sys.executable, "-m", "biostats.mi.parallel", str(job), str(result), str(ticks)],
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
                # the worker must not run on, unseen, and must release the files of the temporary folder
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
                stderr.strip().splitlines()[-1]
                if stderr.strip()
                else f"exit code {process.returncode}"
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
