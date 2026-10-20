"""
Analyses whose results cannot be combined with Rubin's rules, with the reason and what to do instead.

Rubin's rules need, from each completed dataset, an estimate that is approximately normal and its
variance. Rank tests, bootstrap intervals, percentile-based limits and statistics computed on
data-dependent scales do not provide that (van Buuren 2018, sections 5.2 and 5.3). The pages show the
entry instead of a pooled number.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from biostats.references import REFERENCES

REFERENCES.setdefault(
    "vanbuuren2018",
    "van Buuren S (2018). Flexible Imputation of Missing Data, 2nd ed. Chapman & Hall/CRC.",
)


class NotPoolable(BaseModel):
    model_config = ConfigDict(frozen=True)

    analysis: str
    reason: str
    alternative: str


NOT_POOLABLE: dict[str, NotPoolable] = {
    entry.analysis: entry
    for entry in (
        NotPoolable(
            analysis="mann_whitney",
            reason="A rank-sum statistic and its p-value are not an estimate with a variance, so Rubin's rules do "
            "not apply.",
            alternative="Compare means with a Welch t-test or a linear regression (on the log scale if skewed) and "
            "pool the difference; the D2 combination of the test's chi-square form is only an approximation.",
        ),
        NotPoolable(
            analysis="wilcoxon_signed_rank",
            reason="A signed-rank statistic is not an estimate with a variance.",
            alternative="Pool the mean paired difference (paired t-test) instead.",
        ),
        NotPoolable(
            analysis="kruskal_wallis",
            reason="A rank statistic is not an estimate with a variance.",
            alternative="Fit a linear regression on the groups, with robust (HC3) variances when the groups' "
            "variances may differ, and test the group terms jointly with D1 (on the log scale if skewed); the D2 "
            "combination of the Kruskal-Wallis chi-square is only an approximation.",
        ),
        NotPoolable(
            analysis="fisher_exact",
            reason="Fisher's exact p-value is conditional on the margins of one table; there is no rule to combine "
            "exact p-values across imputed datasets.",
            alternative="Pool the odds ratio, risk ratio and risk difference with their intervals; the D2 combination "
            "of the chi-square statistic is approximate, all the more when expected counts are small.",
        ),
        NotPoolable(
            analysis="hodges_lehmann",
            reason="The Hodges-Lehmann shift has a rank-based interval but no variance to combine.",
            alternative="Pool the difference in means, or a median difference from quantile regression with its "
            "standard error.",
        ),
        NotPoolable(
            analysis="bootstrap_ci",
            reason="A bootstrap percentile interval is not a variance on a scale where Rubin's rules hold.",
            alternative="Use the bootstrap standard error as the within-imputation variance, or bootstrap the whole "
            "imputation and analysis.",
        ),
        NotPoolable(
            analysis="icc",
            reason="The intraclass correlation is bounded and skewed, and no pooling rule has been validated for it.",
            alternative="Report the ICC on the complete cases and its range across the imputed datasets.",
        ),
        NotPoolable(
            analysis="bland_altman",
            reason="Limits of agreement are percentiles of the differences, not estimates with a variance.",
            alternative="Pool the mean difference (bias); report the limits of agreement on the complete pairs.",
        ),
        NotPoolable(
            analysis="twobytwo_zero_cell",
            reason="A cell of the 2x2 table is zero in some imputed datasets. Their odds and risk ratios carry the "
            "+0.5 correction and the others do not, so the estimates measure different things and Rubin's rules "
            "cannot combine them.",
            alternative="The risk difference is still pooled. For the ratios, run the analysis again with 0.5 added "
            "to every cell of every imputed dataset, and report it as a sensitivity analysis.",
        ),
        NotPoolable(
            analysis="standardised_variables",
            reason="z-scores or min-max scaling use the mean and spread of each completed dataset, so each "
            "imputation is on a different scale.",
            alternative="Standardise with fixed reference values before the analysis, or pool on the original scale.",
        ),
    )
}


def not_poolable(analysis: str) -> NotPoolable | None:
    """The entry for ``analysis`` when its result cannot be pooled, else None."""
    return NOT_POOLABLE.get(analysis)


__all__ = ["NOT_POOLABLE", "NotPoolable", "not_poolable"]
