"""
Starting a child interpreter: the multiple-imputation engine's parallel runs (biostats.mi.parallel) and the search of
values by a pattern (enrich.patterns). A module of its own, light to import: the search needs it without loading the
imputation engine (several hundred modules and seconds on a fresh server).
"""

from __future__ import annotations

import os
import sys


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


__all__ = ["interpreter"]
