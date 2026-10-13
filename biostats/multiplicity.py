"""Multiplicity adjustment with explicit families (audit S6/S9)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from statsmodels.stats.multitest import multipletests

# UI labels and aliases -> statsmodels method names
_ALIASES: dict[str, str] = {
    "holm": "holm",
    "holm (recommended)": "holm",
    "holm-bonferroni": "holm",
    "bonferroni": "bonferroni",
    "fdr_bh": "fdr_bh",
    "bh": "fdr_bh",
    "benjamini-hochberg": "fdr_bh",
    "benjamini-hochberg (fdr)": "fdr_bh",
    "fdr": "fdr_bh",
    "none": "none",
    "": "none",
}

LABELS: dict[str, str] = {
    "holm": "Holm",
    "bonferroni": "Bonferroni",
    "fdr_bh": "Benjamini-Hochberg (FDR)",
    "none": "none",
}
UI_OPTIONS: tuple[str, ...] = (
    "Holm (recommended)",
    "Benjamini-Hochberg (FDR)",
    "Bonferroni",
    "None",
)


def normalize_method(label: str | None) -> str:
    key = (label or "none").strip().lower()
    if key not in _ALIASES:
        raise ValueError(f"Unknown multiplicity method {label!r}. Use one of {UI_OPTIONS}.")
    return _ALIASES[key]


def adjust(
    pvals: Sequence[float | None], method: str | None = "holm", alpha: float = 0.05
) -> np.ndarray:
    """Adjusted p-values over the whole family; ``None``/NaN entries stay NaN and do not count."""
    name = normalize_method(method)
    raw = np.array([np.nan if p is None else float(p) for p in pvals], dtype=float)
    out = np.full(raw.shape, np.nan)
    mask = ~np.isnan(raw)
    if not mask.any():
        return out
    if name == "none":
        out[mask] = raw[mask]
        return out
    _, adjusted, _, _ = multipletests(raw[mask], alpha=alpha, method=name)
    out[mask] = adjusted
    return out


__all__ = ["LABELS", "UI_OPTIONS", "adjust", "normalize_method"]
