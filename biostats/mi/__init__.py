"""
Multiple imputation (follow-up programme M2).

:mod:`biostats.mi.data` stores a multiply imputed dataset compactly (``MIData``, mice long import and
export); :mod:`biostats.mi.analyze` runs one analysis on every imputed dataset and pools it
(``analyze_mi``) through the adapters of :mod:`biostats.mi.adapters`. :mod:`biostats.mi.engine` imputes by
chained equations (``impute`` with an ``ImputationSpec``; methods in :mod:`biostats.mi.methods`).

:mod:`biostats.mi.pooling` combines the analyses of m imputed datasets (Rubin's rules as in R mice,
Barnard-Rubin df, D1 and D2 tests, Harel's R^2); :mod:`biostats.mi.scales` holds the scales on which
ratios, proportions, correlations and survival probabilities are pooled; :mod:`biostats.mi.not_poolable`
lists the analyses that cannot be pooled and what to do instead.
"""

from biostats.mi.adapters import NotPoolableError, adapter_for, pool_km_curves
from biostats.mi.analyze import MIAnalysisError, MIResult, analyze_mi
from biostats.mi.data import MIData
from biostats.mi.engine import ImputationRun, impute
from biostats.mi.not_poolable import NOT_POOLABLE, NotPoolable, not_poolable
from biostats.mi.pooling import (
    MultiParameterTest,
    PooledRSquared,
    PooledTable,
    PooledTerm,
    TermMismatchError,
    barnard_rubin_df,
    d1_wald,
    d2_chisq,
    pool_r_squared,
    pool_scalar,
    pool_terms,
)
from biostats.mi.scales import SCALES, from_scale, to_scale
from biostats.mi.spec import ImputationSpec, default_method, suggest_m

__all__ = [
    "NOT_POOLABLE",
    "SCALES",
    "MIAnalysisError",
    "MIData",
    "MIResult",
    "ImputationRun",
    "ImputationSpec",
    "MultiParameterTest",
    "NotPoolable",
    "NotPoolableError",
    "PooledRSquared",
    "PooledTable",
    "PooledTerm",
    "TermMismatchError",
    "adapter_for",
    "analyze_mi",
    "barnard_rubin_df",
    "d1_wald",
    "d2_chisq",
    "default_method",
    "from_scale",
    "impute",
    "not_poolable",
    "pool_km_curves",
    "pool_r_squared",
    "pool_scalar",
    "pool_terms",
    "suggest_m",
    "to_scale",
]
