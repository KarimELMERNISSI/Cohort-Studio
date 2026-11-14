"""
How a number is written in a plan, in a sentence and in a result of the analysis pages: one rule, so that a number
is not written two ways on one screen. No Streamlit: the plans (ui.guidance.measures), the readings
(ui.guidance.interpretation) and the screens (app_pages.analysis_layout) all write through it.

- A number alone: three significant figures for its decimals, thousands grouped, the whole part kept, no exponent.
- An estimate with its interval: the same decimals for the three, those the bounds need, never more than one
  beyond what the estimate needs.
- A half is rounded away from zero, as a reader of the unrounded number expects.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

MOST = 6  # decimals at most for numbers written together


def known(value: Any) -> bool:
    """Whether ``value`` is a number with a value: not None, not missing, not infinite."""
    try:
        return value is not None and math.isfinite(float(value))
    except TypeError, ValueError:
        return False


def decimals(value: float, digits: int) -> int:
    """The decimals that give ``value`` ``digits`` significant figures."""
    if value == 0 or not math.isfinite(value):
        return 0
    return max(0, digits - 1 - math.floor(math.log10(abs(value))))


def fixed(value: Any, places: int) -> str:
    """
    ``value`` with ``places`` decimals: thousands grouped, a half rounded away from zero (the built-in rounding goes
    to the even digit: 60,962.5 read "60,962" and 1,235.5 "1,236"), and no sign on a zero ("-0.00"). "n/a" without
    a value.
    """
    if not known(value):
        return "n/a"
    number_ = float(value)
    if abs(number_) >= 1e15:  # no decimal is left to round
        return f"{number_:,.{places}f}"
    rounded = Decimal(repr(number_)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    text = f"{rounded:,.{places}f}"
    return text.lstrip("-") if not rounded else text


def number(value: Any, digits: int = 3) -> str:
    """
    A number as a plan, a sentence or a result writes it: ``digits`` significant figures for its decimals, "n/a"
    when it has no value. Thousands are grouped at any size, the whole part is never rounded (60,962.5 read
    "61,000", and two means 1,200 apart read "243,700 to 244,900" beside "a change of 1,230"), and no exponent is
    written (0.00001 read "0", 93,656 "+9.37e+04", and a plan said that a score "runs from 174 to 2.802e+04").
    """
    if not known(value):
        return "n/a"
    amount = float(value)
    text = fixed(amount, min(12, decimals(amount, digits)))
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def together(estimates: Sequence[Any], bounds: Sequence[Any], digits: int = 3) -> int:
    """
    The decimals of numbers written together, an estimate and the bounds of its interval: those the bounds need for
    ``digits`` significant figures, never more than one beyond what the estimate needs. An estimate of 0, or none,
    leaves the bounds to decide; without bounds the estimate decides.

    With the decimals of the number nearest 0 for all, a result got more digits the nearer it was to "no difference"
    or to p = 0.05, which the data do not carry: "-0.0711, 95% CI -0.6941 to 0.5518" (now "-0.071, -0.694 to
    0.552"), "0.50000, 95% CI 0.00847 to 9.60453" (now "0.5000, 0.0085 to 9.6045"). Each rounded on its own, a
    rate ratio read "3.12, 95% CI 0.996 to 12.935", which stays "3.125".
    """
    of_estimates = [decimals(float(v), digits) for v in estimates if known(v) and float(v) != 0]
    of_bounds = [decimals(float(v), digits) for v in bounds if known(v)]
    if not of_bounds:
        return min(MOST, max(of_estimates, default=0))
    if not of_estimates:
        return min(MOST, max(of_bounds))
    return min(MOST, max(of_bounds), max(of_estimates) + 1)


def span(lower: Any, upper: Any, digits: int = 3, *, unit: str = "") -> str:
    """ "0.902 to 0.953": two numbers written with the same decimals, those the one nearer 0 needs for ``digits``
    significant figures (each rounded on its own, an interval read "-9 to -7.38"); a percentage with one decimal."""
    if not (known(lower) and known(upper)):
        return f"{number(lower, digits)}{unit} to {number(upper, digits)}{unit}"
    places = 1 if unit == "%" else together((), (lower, upper), digits)
    return f"{fixed(lower, places)}{unit} to {fixed(upper, places)}{unit}"


def interval(ci: Any, digits: int = 3, *, scale: float = 1.0, unit: str = "") -> str:
    """ "95% CI 0.902 to 0.953" from an interval object; "" without one, or without bounds ("95% CI n/a% to n/a%"
    stood beside a predictive value that has no value)."""
    if ci is None or not (known(ci.lower) and known(ci.upper)):
        return ""
    return f"95% CI {span(ci.lower * scale, ci.upper * scale, digits, unit=unit)}"


def with_interval(
    estimate: Any, ci: Any, digits: int = 3, *, places: int | None = None
) -> tuple[str, str]:
    """
    An estimate and its interval written with the same decimals: ("3.125", "95% CI 0.996 to 12.935"). The decimals
    are those of :func:`together`, or ``places`` when the three are of a kind that has its own (three for an AUC, a
    kappa, an ICC). An estimate or an interval without a value is said for itself.
    """
    lower, upper = getattr(ci, "lower", None), getattr(ci, "upper", None)
    if not (known(estimate) and known(lower) and known(upper)):
        alone = fixed(estimate, places) if places is not None else number(estimate, digits)
        return alone, interval(ci, digits)
    shared = places if places is not None else together((estimate,), (lower, upper), digits)
    return fixed(estimate, shared), f"95% CI {fixed(lower, shared)} to {fixed(upper, shared)}"


def alike(*values: Any, digits: int = 3, estimates: int = 1) -> list[str]:
    """
    Numbers that are read together, written with the same decimals (:func:`together`): the first ``estimates`` of
    them are estimates, the others the bounds of their intervals. Two limits of agreement read "-8.19 to 20.17" over
    "95% CI 19.4 to 21.0", and "20.2" in the figure's legend. A number without a value reads "n/a".
    """
    places = together(values[:estimates], values[estimates:], digits)
    return [fixed(value, places) for value in values]


def percent(value: Any, digits: int = 1) -> str:
    """A share as a percentage: "86.1%"; "n/a" without a value."""
    return f"{fixed(100 * float(value), digits)}%" if known(value) else "n/a"


def p_text(p: Any) -> str:
    """
    A p-value as a sentence writes it after "p": "< 0.001", "= 0.017"; "n/a" when it was not computed. Next to 0.05
    it takes the decimals that tell it from 0.05: 0.0499 and 0.0501 both read "= 0.050", and 0.04996 "= 0.0500".
    A half is rounded away from zero, as every number here: McNemar's exact p of 0.0625 read "= 0.062".
    """
    if not known(p):
        return "n/a"
    value = float(p)
    if value < 0.001:
        return "< 0.001"
    if not 0.0495 <= value < 0.0505:
        return f"= {fixed(value, 3)}"
    text = fixed(value, 4)
    for places in (4, 5, 6):
        text = fixed(value, places)
        if value == 0.05 or float(text) != 0.05:
            break
    return f"= {text}"


__all__ = [
    "alike",
    "decimals",
    "fixed",
    "interval",
    "known",
    "number",
    "p_text",
    "percent",
    "span",
    "together",
    "with_interval",
]
