"""Shared result types for the biostatistics engine."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RANDOM_STATE: int = int(os.getenv("CMP_RANDOM_SEED", "20260101"))


class CI(BaseModel):
    """A confidence interval with the method that produced it."""

    model_config = ConfigDict(frozen=True)

    lower: float
    upper: float
    level: float = Field(default=0.95, gt=0.0, lt=1.0)
    method: str = ""

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def as_text(self, digits: int = 2, pct: bool = False) -> str:
        scale = 100.0 if pct else 1.0
        unit = "%" if pct else ""
        return f"{self.lower * scale:.{digits}f}{unit} to {self.upper * scale:.{digits}f}{unit}"


# Cohen (1988) conventions, used only for the magnitude *label*; the CI carries the evidence.
_MAGNITUDE_THRESHOLDS: dict[str, tuple[float, float, float]] = {
    "d": (0.2, 0.5, 0.8),
    "g": (0.2, 0.5, 0.8),
    "r": (0.1, 0.3, 0.5),
    "eta2": (0.01, 0.06, 0.14),
    "omega2": (0.01, 0.06, 0.14),
    "eta2_h": (0.01, 0.06, 0.14),
    "v": (0.1, 0.3, 0.5),
}


class EffectSize(BaseModel):
    """A standardised effect estimate with its interval and provenance."""

    model_config = ConfigDict(frozen=True)

    name: str  # "Hedges' g", "rank-biserial r", "eta squared", ...
    kind: str  # key of _MAGNITUDE_THRESHOLDS ("d", "g", "r", "eta2", "omega2", "eta2_h", "v") or "other"
    estimate: float
    ci: CI | None = None
    n: int | None = None
    reference: str = ""  # method citation
    note: str = ""
    se: float | None = (
        None  # standard error of the estimate, when it has one (pooling across imputations)
    )

    def magnitude(self) -> str:
        bounds = _MAGNITUDE_THRESHOLDS.get(self.kind)
        if bounds is None:
            return ""
        value = abs(self.estimate)
        small, medium, large = bounds
        if value < small:
            return "negligible"
        if value < medium:
            return "small"
        if value < large:
            return "medium"
        return "large"


class MIInfo(BaseModel):
    """How a result was pooled across multiply imputed datasets (Rubin's rules)."""

    model_config = ConfigDict(frozen=True)

    m: int  # number of imputed datasets analysed
    fmi: float  # fraction of missing information
    riv: float  # relative increase in variance due to the missing data
    lam: float  # proportion of the total variance due to the missing data
    df: float  # Barnard-Rubin degrees of freedom
    dfcom: float  # complete-data degrees of freedom
    scale: str = "identity"  # scale on which the estimate was pooled
    method: str = "Rubin's rules, Barnard-Rubin df"
    note: str = ""


class AssumptionCheck(BaseModel):
    """A diagnostic, never a gate: the test runs regardless, the status informs interpretation."""

    model_config = ConfigDict(frozen=True)

    name: str
    status: Literal["ok", "caution", "violated", "na"]
    detail: str = ""
    p_value: float | None = None


class TestResult(BaseModel):
    """Result of a group comparison: estimate and CI first, then the test, then diagnostics."""

    __test__ = False  # not a pytest test class despite its name
    model_config = ConfigDict(frozen=True)

    name: str
    family: Literal["parametric", "nonparametric", "exact"] = "parametric"
    statistic: float | None = None
    statistic_label: str = ""
    df: list[float] = Field(default_factory=list)
    p_value: float | None = None
    p_adjusted: float | None = None
    adjustment: str | None = None
    estimate: float | None = None
    estimate_label: str = ""
    estimate_ci: CI | None = None
    se: float | None = None  # standard error of ``estimate`` (pooling across imputations)
    mi: MIInfo | None = None  # set when the result pools multiply imputed datasets
    effect_sizes: list[EffectSize] = Field(default_factory=list)
    assumptions: list[AssumptionCheck] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)
    n_per_group: list[int] = Field(default_factory=list)
    reference_group: str | None = None
    comparison_group: str | None = None
    alpha: float = 0.05

    @property
    def n_total(self) -> int:
        return int(sum(self.n_per_group))

    @property
    def primary_effect(self) -> EffectSize | None:
        return self.effect_sizes[0] if self.effect_sizes else None

    @property
    def significant(self) -> bool | None:
        p = self.p_adjusted if self.p_adjusted is not None else self.p_value
        return None if p is None else bool(p < self.alpha)


__all__ = ["CI", "RANDOM_STATE", "AssumptionCheck", "EffectSize", "MIInfo", "TestResult"]
