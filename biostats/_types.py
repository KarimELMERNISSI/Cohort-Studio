"""Shared result types for the biostatistics engine."""

from __future__ import annotations

import os

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


__all__ = ["CI", "RANDOM_STATE"]
