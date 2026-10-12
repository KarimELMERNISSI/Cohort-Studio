"""
UI-free biostatistics engine.

Modules are pure functions over pandas/numpy/scipy/statsmodels returning typed results; pages only
render them. Legacy entry points in ``utils.epidemiology_utils`` and ``utils.analysis_utils`` are
thin shims over this package.

Tier A (October 2026 audit): :mod:`biostats.contingency` (2x2 orientation and measures),
:mod:`biostats.power` (sample-size semantics), :mod:`biostats.missing` (full-family missingness
scan), :mod:`biostats.zscores` (reference-population Z-scores).
"""

from biostats._types import CI, RANDOM_STATE

__all__ = ["CI", "RANDOM_STATE"]
