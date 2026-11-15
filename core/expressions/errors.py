"""
What computing an expression can raise, apart from the errors of the expression itself (a ``TypeError``, a missing
column): the evaluator refusing a construct or a size, a worker stopping an expression at one of its limits, and no
worker to compute it. A module of its own, so that the evaluator and the runner that computes through it import one
another in one direction only.
"""

from __future__ import annotations


class UnsafeExpressionError(ValueError):
    """The expression uses a construct that is not allowed."""


class ExpressionStopped(UnsafeExpressionError):
    """
    A worker stopped the expression at one of its limits: it took too long (``time``), needed too much memory
    (``memory``), was given a table larger than a worker takes (``input``), gave back a result larger than the
    server takes (``result``), or the process computing it ended (``ended``). The message is the sentence a page
    shows; ``limit`` is the limit's number (seconds, megabytes), when there is one. The same expression on the same
    table is stopped again, so a caller may remember it as the expression's own failure.
    """

    def __init__(self, sentence: str, *, reason: str, limit: float | None = None) -> None:
        super().__init__(sentence)
        self.reason = reason
        self.limit = limit

    def __reduce__(self):  # an exception with keyword arguments travels whole
        return (_stopped, (str(self), self.reason, self.limit))


def _stopped(sentence: str, reason: str, limit: float | None) -> ExpressionStopped:
    return ExpressionStopped(sentence, reason=reason, limit=limit)


class ExpressionError(ValueError):
    """An error of the expression that a worker could not send whole: its message, as the expression raised it."""


class WorkerUnavailable(RuntimeError):
    """No worker could compute the expression: all are busy, or none started. Nothing is known of the expression
    itself, so a caller never remembers it as the expression's failure: the next run asks again."""
