"""
The global data-quality score as a weighted mean of the audit dimensions, with its decomposition.

Pure Python (no Streamlit, no plotting): the dashboard draws the decomposition, the tests check it.
Each dimension scores 0-100; its weight decides its share of the global score; the points a
dimension "loses" are ``(100 - score) * share``, so the global score is ``100 - sum(lost)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

# Display order and the default weights (the plausibility rules count double).
DIMENSIONS: tuple[str, ...] = (
    "Completeness",
    "Statistical Validity",
    "Domain Validity",
    "Consistency",
    "Uniformity",
    "Uniqueness",
)
DEFAULT_WEIGHTS: dict[str, float] = {name: 1.0 for name in DIMENSIONS} | {"Domain Validity": 2.0}

DIMENSION_HELP: dict[str, str] = {
    "Completeness": "Share of cells that are not empty.",
    "Statistical Validity": "Share of numeric values not flagged as outliers by the method chosen in Handle Outliers.",
    "Domain Validity": "Share of the plausibility checks that pass: each rule is checked on the rows holding the "
    "values it reads, never on a missing or imputed value (Completeness counts those). Each check counts once, so a "
    "rule that never fails raises the score: the Domain Validity tab lists the rows. A rule written with != or ~ "
    "also fires on a missing value.",
    "Consistency": "Share of text columns whose values do not look like numbers or dates stored as text.",
    "Uniformity": "Share of text columns without stray spaces or mixed capitalisation.",
    "Uniqueness": "Share of rows that are not an exact duplicate of another row.",
}

# Bands used by the colours and the interpretation sentence.
EXCELLENT, GOOD, FAIR = 90.0, 80.0, 50.0


@dataclass(frozen=True)
class DimensionScore:
    name: str
    score: float  # 0-100
    weight: float
    share: float  # weight / total weight (0-1)

    @property
    def earned(self) -> float:
        """Points this dimension contributes to the global score."""
        return self.score * self.share

    @property
    def lost(self) -> float:
        """Points of the global score this dimension costs."""
        return (100.0 - self.score) * self.share


@dataclass(frozen=True)
class QualityScore:
    total: float
    dimensions: tuple[DimensionScore, ...]

    def by_loss(self) -> list[DimensionScore]:
        """Dimensions from the most to the least costly (ties in display order)."""
        order = {d.name: i for i, d in enumerate(self.dimensions)}
        return sorted(self.dimensions, key=lambda d: (-d.lost, order[d.name]))

    @property
    def lost(self) -> float:
        return sum(d.lost for d in self.dimensions)


def ordered(names) -> list[str]:
    """Known dimensions first, in display order, then any other metric."""
    names = list(names)
    return [n for n in DIMENSIONS if n in names] + [n for n in names if n not in DIMENSIONS]


def weighted_score(
    metrics: Mapping[str, float | None], weights: Mapping[str, float] | None = None
) -> QualityScore:
    """
    Weighted mean of the available dimensions (a ``None`` metric, e.g. domain validity without rules, is left out).

    With every weight at zero the global score is 0, as before.
    """
    weights = weights or DEFAULT_WEIGHTS
    present = [n for n in ordered(metrics) if metrics[n] is not None]
    total_weight = sum(max(float(weights.get(n, 1.0)), 0.0) for n in present)
    dims = tuple(
        DimensionScore(
            name=n,
            score=float(metrics[n]),  # type: ignore[arg-type]
            weight=max(float(weights.get(n, 1.0)), 0.0),
            share=(max(float(weights.get(n, 1.0)), 0.0) / total_weight)
            if total_weight > 0
            else 0.0,
        )
        for n in present
    )
    total = sum(d.earned for d in dims) if total_weight > 0 else 0.0
    return QualityScore(total=total, dimensions=dims)


def band(score: float) -> str:
    """``excellent`` (>= 90), ``good`` (>= 80), ``fair`` (>= 50) or ``poor``."""
    if score >= EXCELLENT:
        return "excellent"
    if score >= GOOD:
        return "good"
    if score >= FAIR:
        return "fair"
    return "poor"


def interpretation(score: float, *, high_advice: bool = False) -> str:
    """The sentence under the global score; never "ready for analysis" beside a high-priority advice."""
    if high_advice and band(score) in ("excellent", "good"):
        return f"{band(score).capitalize()} overall: review the high-priority advice below before analysis."
    return {
        "excellent": "Excellent: ready for analysis.",
        "good": "Good: review the dimensions that lose points before inference.",
        "fair": "Fair: fix the main issues before reliable analysis.",
        "poor": "Poor: the data need cleaning before any analysis.",
    }[band(score)]


def score_text(score: float) -> str:
    """A score to one decimal, as the dashboard prints it; a score under 100 never reads 100.0 (99.99 does)."""
    text = f"{score:.1f}"
    return f"{min(score, 99.99):.2f}" if score < 100 and text == "100.0" else text


def lost_text(points: float) -> str:
    """Points or a share lost, to one decimal; a loss that rounds to nothing still shows (0.01)."""
    return f"{points:.1f}" if points >= 0.05 or points <= 0 else f"{max(points, 0.01):.2f}"


def points_text(points: float) -> str:
    """Points of the global score, to two decimals, as the chart of points lost labels them: a loss that rounds to
    nothing still shows (0.01), never 0.00."""
    return f"{points:.2f}" if points >= 0.005 or points <= 0 else "0.01"


__all__ = [
    "DEFAULT_WEIGHTS",
    "DIMENSIONS",
    "DIMENSION_HELP",
    "DimensionScore",
    "QualityScore",
    "band",
    "interpretation",
    "lost_text",
    "points_text",
    "score_text",
    "ordered",
    "weighted_score",
]
