"""
The values a search picks (Data > Clean > Categories): text found anywhere in each value as written, ``*`` standing for
any text, or a regular expression (Python's syntax: ``^`` and ``$`` anchor it), case ignored unless asked. A dot, a
bracket or a question mark typed as text is that character: "n.a." finds n.a. only, never "infa" in Myocardial
infarction.

The search runs in a separate Python process with a time limit: a pattern that backtracks without end (``(a+)+$`` on a
long value) stops there, and never holds the server, which every user shares (Python's matcher keeps the interpreter
while it runs). The limit counts from the moment the process is ready, so a busy server's slow start is no fault of the
pattern; the start has a limit of its own. The values travel to that process and back on its own pipes, the process
gets no environment (the server's settings stay out of it); nothing is written.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
from collections.abc import Sequence

MAX_LENGTH = 200  # characters of a pattern
TIMEOUT = 2.0  # seconds the search may take, once its process is ready
START = 30.0  # seconds its process may take to start (a busy server)
# the process: ready, then the pattern and the values on its input (ASCII JSON), the indices of the values found
_MATCHER = (
    "import json, re, sys\n"
    "sys.stdout.write('r'); sys.stdout.flush()\n"
    "job = json.loads(sys.stdin.buffer.read().decode('ascii'))\n"
    "found = re.compile(job['pattern'], re.IGNORECASE if job['ignore_case'] else 0)\n"
    "sys.stdout.write(json.dumps([i for i, v in enumerate(job['values']) if found.search(v)]))\n"
)
# a repetition inside a repeated group, as (a+)+ or (a*b?)*: what backtracks without end
_NESTED = re.compile(r"\([^()]*[+*}][^()]*\)[+*{]")


class PatternError(ValueError):
    """A search that picks nothing it could: why, in plain words."""


def _compiled(text: str, regex: bool, ignore_case: bool) -> str:
    """The pattern the process compiles: text escaped, its ``*`` any text; a regular expression as typed."""
    pattern = text if regex else re.escape(text).replace(r"\*", ".*")
    try:
        re.compile(
            pattern, re.IGNORECASE if ignore_case else 0
        )  # compiling takes its length, no more
    except re.error as exc:
        at = f" (at character {exc.pos + 1})" if exc.pos is not None else ""
        hint = (
            ": * repeats what comes before it; to find values holding kidney, type kidney"
            if pattern.lstrip("^").startswith(("*", "+", "?"))
            else ""
        )
        raise PatternError(f"Not a regular expression: {exc.msg}{at}{hint}.") from exc
    return pattern


def matching(
    pattern: str,
    values: Sequence[str],
    *,
    ignore_case: bool = True,
    regex: bool = False,
    timeout: float = TIMEOUT,
    start: float = START,
) -> list[str]:
    """
    The values the search finds, in their order: ``pattern`` as text found anywhere in a value (``*`` any text), or
    with ``regex`` a regular expression. Raises :class:`PatternError` for an empty, overlong or invalid pattern, one
    that takes longer than ``timeout`` seconds once its process is ready, or a process that cannot start.
    """
    # the base interpreter: a venv's launcher starts slowly when the server is busy; biostats.processes, light to
    # import, never the imputation engine (biostats.mi: hundreds of modules on the first search)
    from biostats.processes import interpreter

    text = str(
        pattern or ""
    )  # as typed: a space can be what is looked for (" $", a value ending in one)
    if not text.strip():
        raise PatternError("Type what to look for.")
    if len(text) > MAX_LENGTH:
        raise PatternError(f"What to look for is longer than {MAX_LENGTH} characters.")
    compiled = _compiled(text, regex, ignore_case)
    if not values:
        return []
    job = json.dumps({"pattern": compiled, "values": list(values), "ignore_case": ignore_case})
    executable, needs = interpreter()
    try:
        child = subprocess.Popen(
            [executable, "-I", "-S", "-c", _MATCHER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=dict(needs),  # nothing of the server's environment: its settings and keys stay out
        )
    except OSError as exc:
        raise PatternError("The search could not start on this server: try again.") from exc
    with child:
        ready, out = threading.Event(), bytearray()
        started: list[bytes] = []  # the ready byte, or nothing when the process ended first

        def read() -> None:
            first = child.stdout.read(1) if child.stdout else b""
            started.append(first)
            ready.set()
            if first:
                out.extend(child.stdout.read() if child.stdout else b"")

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        if not ready.wait(start):
            child.kill()
            raise PatternError("The server is busy and the search could not start: try again.")
        try:
            if child.stdin:
                child.stdin.write(job.encode("ascii"))
                child.stdin.close()
        except OSError:  # the process ended before reading: said below
            pass
        reader.join(timeout)
        if reader.is_alive():
            child.kill()
            reader.join()
            nested = bool(regex and _NESTED.search(compiled))
            raise PatternError(
                f"The search took more than {timeout:g} seconds on these values"
                + (
                    ": write it without a repetition inside another, as (a+)+."
                    if nested
                    else ": try again."
                )
            )
        child.wait()
    if not any(started):  # it ended before it was ready: the server's fault, never the values'
        raise PatternError("The search could not start on this server: try again.")
    if child.returncode != 0 or not out:
        raise PatternError("The search could not be made on these values.")
    return [values[i] for i in json.loads(out.decode("ascii"))]


__all__ = ["MAX_LENGTH", "START", "TIMEOUT", "PatternError", "matching"]
