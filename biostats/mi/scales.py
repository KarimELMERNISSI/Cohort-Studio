"""
Scales on which estimates are pooled.

Rubin's rules assume each completed-data estimate is approximately normal. Ratios, proportions,
correlations and survival probabilities are not, so they are pooled on a transformed scale and the
pooled estimate and interval are transformed back (Marshall et al. 2009; van Buuren 2018, 5.2):

=========== ===================== ===========================================================
scale       transform             typical estimates
=========== ===================== ===========================================================
identity    x                     differences in means, regression coefficients
log         log(x)                odds, risk, hazard and rate ratios
logit       log(x / (1 - x))      proportions, AUC, sensitivity, specificity, predictive values
fisher_z    atanh(x)              correlation coefficients
cloglog     log(-log(x))          survival probabilities S(t)
=========== ===================== ===========================================================

Variances are carried to the new scale with the delta method: Var(g(X)) = Var(X) * g'(X)^2.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from biostats._types import CI


@dataclass(frozen=True)
class Scale:
    """A monotone transformation, its inverse and its derivative."""

    name: str
    forward: Callable[[float], float]
    inverse: Callable[[float], float]
    derivative: Callable[[float], float]
    lower: float  # open domain bounds of the natural scale
    upper: float

    def check(self, value: float) -> None:
        if not (self.lower < value < self.upper):
            raise ValueError(
                f"{value!r} is outside the domain of the {self.name} scale ({self.lower}, {self.upper})"
            )


def _expit(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


SCALES: dict[str, Scale] = {
    "identity": Scale("identity", float, float, lambda x: 1.0, -np.inf, np.inf),
    "log": Scale(
        "log",
        lambda x: float(np.log(x)),
        lambda x: float(np.exp(x)),
        lambda x: 1.0 / x,
        0.0,
        np.inf,
    ),
    "logit": Scale(
        "logit",
        lambda x: float(np.log(x / (1.0 - x))),
        _expit,
        lambda x: 1.0 / (x * (1.0 - x)),
        0.0,
        1.0,
    ),
    "fisher_z": Scale(
        "fisher_z",
        lambda x: float(np.arctanh(x)),
        lambda x: float(np.tanh(x)),
        lambda x: 1.0 / (1.0 - x * x),
        -1.0,
        1.0,
    ),
    "cloglog": Scale(
        "cloglog",
        lambda x: float(np.log(-np.log(x))),
        lambda x: float(np.exp(-np.exp(x))),
        lambda x: 1.0 / (x * np.log(x)),
        0.0,
        1.0,
    ),
}


def get_scale(name: str) -> Scale:
    try:
        return SCALES[name]
    except KeyError:
        raise ValueError(
            f"unknown pooling scale {name!r}; expected one of {sorted(SCALES)}"
        ) from None


def to_scale(estimate: float, variance: float, scale: str) -> tuple[float, float]:
    """An estimate and its variance on the pooling scale (delta method for the variance)."""
    s = get_scale(scale)
    s.check(estimate)
    return s.forward(estimate), float(variance) * s.derivative(estimate) ** 2


def from_scale(estimate: float, ci: CI, scale: str) -> tuple[float, CI]:
    """A pooled estimate and interval back on the natural scale (bounds reordered for decreasing transforms)."""
    s = get_scale(scale)
    low, high = s.inverse(ci.lower), s.inverse(ci.upper)
    return s.inverse(estimate), CI(
        lower=min(low, high),
        upper=max(low, high),
        level=ci.level,
        method=f"{ci.method}, pooled on the {scale} scale" if scale != "identity" else ci.method,
    )


__all__ = ["SCALES", "Scale", "from_scale", "get_scale", "to_scale"]
