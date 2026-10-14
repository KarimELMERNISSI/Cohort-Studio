"""
UI-free biostatistics engine.

Modules are pure functions over pandas/numpy/scipy/statsmodels returning typed results; pages only
render them. Legacy entry points in ``utils.epidemiology_utils`` and ``utils.analysis_utils`` are
thin shims over this package.

Tier A (October 2026 audit): :mod:`biostats.contingency` (2x2 orientation and measures),
:mod:`biostats.power` (sample-size semantics), :mod:`biostats.missing` (full-family missingness
scan), :mod:`biostats.zscores` (reference-population Z-scores).

Tier B: :mod:`biostats.effect_sizes` (Hedges' g, rank-biserial, eta/omega squared, bias-corrected
Cramer's V, all with CIs), :mod:`biostats.inference` (Welch-first group comparisons),
:mod:`biostats.posthoc` (Games-Howell, Tukey, Dunn, pairwise tests reporting every pair),
:mod:`biostats.multiplicity` (explicit families), :mod:`biostats.normality` (Lilliefors,
Anderson-Darling), :mod:`biostats.mcar` (EM-based Little's test), :mod:`biostats.diagnostic`
(sensitivity/specificity, likelihood ratios, DeLong AUC, Youden), :mod:`biostats.agreement`
(Bland-Altman with CIs), :mod:`biostats.correlation` (mixed-type association matrix),
:mod:`biostats.references` (citations).
"""

from biostats._types import CI, RANDOM_STATE, AssumptionCheck, EffectSize, MIInfo, TestResult
from biostats.references import cite

__all__ = ["CI", "RANDOM_STATE", "AssumptionCheck", "EffectSize", "MIInfo", "TestResult", "cite"]
