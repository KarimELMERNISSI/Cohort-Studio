"""
Correlation strength categories and thresholds (single source of truth).

The absolute-value cut-offs for Pearson correlations follow Schober, Boer & Schwarte (2018),
*Correlation Coefficients: Appropriate Use and Interpretation*, Anesthesia & Analgesia 126(5):
negligible < 0.10, weak < 0.40, moderate < 0.70, strong < 0.90, very strong >= 0.90.
Spearman and Kendall cut-offs are the values historically used by this application
(Kendall's tau is converted approximately with tau ~ (2/pi) * asin(r)).

This module has no UI dependency; the correlation graph builder, the Excel export and the
legend all read the same table.
"""

from __future__ import annotations

from enum import StrEnum

import pandas as pd


class CorrelationMethod(StrEnum):
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class StrengthCategory(StrEnum):
    NEGLIGIBLE = "Negligible"
    WEAK = "Weak"
    MODERATE = "Moderate"
    STRONG = "Strong"
    VERY_STRONG = "Very Strong"


STRENGTH_ORDER: tuple[StrengthCategory, ...] = tuple(StrengthCategory)

STRENGTH_THRESHOLDS: dict[CorrelationMethod, tuple[float, float, float, float, float]] = {
    CorrelationMethod.PEARSON: (0.00, 0.10, 0.40, 0.70, 0.90),
    CorrelationMethod.SPEARMAN: (0.00, 0.10, 0.38, 0.68, 0.89),
    CorrelationMethod.KENDALL: (0.00, 0.06, 0.26, 0.49, 0.71),
}

REFERENCE = "Schober, Boer & Schwarte (2018), Anesth Analg 126(5):1763-1768"


def coerce_method(method: CorrelationMethod | str | None) -> CorrelationMethod:
    """Accept enum members or strings (case-insensitive); unknown values fall back to Pearson."""
    if isinstance(method, CorrelationMethod):
        return method
    try:
        return CorrelationMethod(str(method or "pearson").lower())
    except ValueError:
        return CorrelationMethod.PEARSON


def coerce_category(category: StrengthCategory | str | None) -> StrengthCategory:
    """Accept enum members, display labels ('Very Strong') or slugs ('very_strong')."""
    if isinstance(category, StrengthCategory):
        return category
    raw = str(category or StrengthCategory.WEAK).strip()
    for member in StrengthCategory:
        if raw.lower() in (member.value.lower(), member.name.lower(), member.value.lower().replace(" ", "_")):
            return member
    return StrengthCategory.WEAK


def thresholds_for(method: CorrelationMethod | str | None) -> tuple[float, float, float, float, float]:
    return STRENGTH_THRESHOLDS[coerce_method(method)]


def categorize_strength(abs_value: float, method: CorrelationMethod | str | None) -> StrengthCategory:
    """Map |r| to its strength category for the given method."""
    cuts = thresholds_for(method)
    category = StrengthCategory.NEGLIGIBLE
    for member, cut in zip(STRENGTH_ORDER, cuts, strict=True):
        if abs_value >= cut:
            category = member
    return category


def min_abs_for(category: StrengthCategory | str | None, method: CorrelationMethod | str | None) -> float:
    """Smallest |r| that belongs to ``category`` (or stronger) for the given method."""
    cuts = thresholds_for(method)
    return dict(zip(STRENGTH_ORDER, cuts, strict=True))[coerce_category(category)]


def strength_table(method: CorrelationMethod | str | None = None) -> pd.DataFrame:
    """Table of categories and lower bounds (all methods, or one) for exports and legends."""
    rows = []
    methods = [coerce_method(method)] if method is not None else list(CorrelationMethod)
    for m in methods:
        cuts = STRENGTH_THRESHOLDS[m]
        for i, member in enumerate(STRENGTH_ORDER):
            upper = cuts[i + 1] if i + 1 < len(cuts) else 1.0
            rows.append(
                {
                    "Method": m.value,
                    "Strength": member.value,
                    "Min |r|": cuts[i],
                    "Max |r|": upper,
                    "Reference": REFERENCE if m is CorrelationMethod.PEARSON else "application default",
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "REFERENCE",
    "STRENGTH_ORDER",
    "STRENGTH_THRESHOLDS",
    "CorrelationMethod",
    "StrengthCategory",
    "categorize_strength",
    "coerce_category",
    "coerce_method",
    "min_abs_for",
    "strength_table",
    "thresholds_for",
]
