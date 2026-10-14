"""
Multiple imputation: pooling layer (follow-up programme M2).

:mod:`biostats.mi.pooling` combines the analyses of m imputed datasets (Rubin's rules as in R mice,
Barnard-Rubin df, D1 and D2 tests, Harel's R^2); :mod:`biostats.mi.scales` holds the scales on which
ratios, proportions, correlations and survival probabilities are pooled; :mod:`biostats.mi.not_poolable`
lists the analyses that cannot be pooled and what to do instead.
"""

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

__all__ = [
    "NOT_POOLABLE",
    "SCALES",
    "MultiParameterTest",
    "NotPoolable",
    "PooledRSquared",
    "PooledTable",
    "PooledTerm",
    "TermMismatchError",
    "barnard_rubin_df",
    "d1_wald",
    "d2_chisq",
    "from_scale",
    "not_poolable",
    "pool_r_squared",
    "pool_scalar",
    "pool_terms",
    "to_scale",
]
