"""
What a page shows while a formula or a criterion waits on a worker (``manage.workers``): after a second, one line
says why the page waits, until the call returns. Each time the worker's pool looks again (every tenth of a second),
the line is written again: that is where Streamlit can stop the run, when the user clicks elsewhere or on Stop, and
the worker computing is then ended.

``main.py`` opens one for the whole run, at the top of the page; a page opens its own where the user starts a
computation (under Compute & Preview), so that the line is where the user looks: the innermost line is the one
written.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from manage.workers.pool import waiting

WAITING = "Computing a formula or criterion... Stop, at the top right, ends it."


@contextlib.contextmanager
def computing(place: DeltaGenerator | None = None) -> Iterator[DeltaGenerator]:
    """Inside the block, a call that waits on a worker writes :data:`WAITING` in a line of its own, here (in
    ``place`` when it is given), until it returns; the line is empty otherwise. Yields the line, where a page may say
    why nothing was computed."""
    line = (place if place is not None else st).empty()

    def hook(waits: bool) -> None:
        if waits:
            line.caption(f":material/hourglass_top: {WAITING}")
        else:
            line.empty()

    try:
        with waiting(hook):
            yield line
    finally:  # also when no worker could compute it: the page's error then stands alone
        line.empty()


__all__ = ["WAITING", "computing"]
