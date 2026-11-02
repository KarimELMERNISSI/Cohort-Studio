"""
The values a regular expression picks (Data > Clean > Categories), Python's own syntax, searched anywhere in each value
as written (``^`` and ``$`` anchor it), case ignored unless asked. The pattern is tried in a separate Python process
with a time limit: a pattern that backtracks without end (``(a+)+$`` on a long value) stops there, and never holds the
server, which every user shares (Python's matcher keeps the interpreter while it runs). The values travel to that
process and back on its own pipes; nothing is written.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Sequence

MAX_LENGTH = 200  # characters of a pattern
TIMEOUT = 2.0  # seconds, the process's start included
# the process: the pattern and the values on its input (ASCII JSON), the indices of the values it matches on its output
_MATCHER = (
    "import json, re, sys\n"
    "job = json.loads(sys.stdin.buffer.read().decode('ascii'))\n"
    "found = re.compile(job['pattern'], re.IGNORECASE if job['ignore_case'] else 0)\n"
    "sys.stdout.write(json.dumps([i for i, v in enumerate(job['values']) if found.search(v)]))\n"
)


class PatternError(ValueError):
    """A pattern that picks nothing it could: why, in plain words."""


def matching(
    pattern: str, values: Sequence[str], *, ignore_case: bool = True, timeout: float = TIMEOUT
) -> list[str]:
    """The values ``pattern`` finds, in their order. Raises :class:`PatternError` for an empty, overlong or invalid
    pattern, or one that takes longer than ``timeout`` seconds on these values."""
    text = str(
        pattern or ""
    )  # as typed: a space can be what is looked for (" $", a value ending in one)
    if not text.strip():
        raise PatternError("Type a pattern.")
    if len(text) > MAX_LENGTH:
        raise PatternError(f"The pattern is longer than {MAX_LENGTH} characters.")
    try:
        re.compile(
            text, re.IGNORECASE if ignore_case else 0
        )  # compiling takes the pattern's length, no more
    except re.error as exc:
        raise PatternError(
            f"Not a regular expression: {exc.msg} (at character {exc.pos})."
        ) from exc
    if not values:
        return []
    job = json.dumps(
        {"pattern": text, "values": list(values), "ignore_case": ignore_case}
    )  # ASCII only
    try:
        done = subprocess.run(
            [sys.executable, "-I", "-S", "-c", _MATCHER],
            input=job.encode("ascii"),
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PatternError(
            f"The pattern took more than {timeout:g} seconds on these values: write it without a repetition inside "
            "another, as (a+)+."
        ) from exc
    if done.returncode != 0:
        raise PatternError("The pattern could not be tried on these values.")
    return [values[i] for i in json.loads(done.stdout.decode("ascii"))]


__all__ = ["MAX_LENGTH", "TIMEOUT", "PatternError", "matching"]
