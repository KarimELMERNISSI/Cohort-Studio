"""
The worker process of a parallel imputation (``python -m biostats.mi.worker job result ticks``, started by
:mod:`biostats.mi.parallel`): here ``__main__`` is this module, so the process pool it starts never re-runs the
Streamlit page script. It reads the data and the specification, imputes with the workers asked, writes its progress
to a small file and the result to another. It imports the engine; the engine starts it by its name only.
"""

from __future__ import annotations

import os
import pickle
import sys

from biostats.mi.engine import impute


def main(job: str, result: str, ticks: str) -> None:
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
    main(*sys.argv[1:4])
