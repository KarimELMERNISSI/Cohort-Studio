"""Plain-language definitions of the terms used on the analysis pages."""

from __future__ import annotations

GLOSSARY: dict[str, str] = {
    "p-value": "The probability of seeing a result at least as extreme as the observed one if the null hypothesis were true. "
               "It is not the probability that the null hypothesis is true, nor a measure of the size of an effect.",
    "Confidence interval (CI)": "The range of true values compatible with the data at the chosen level (usually 95%). "
                                "Values inside the interval are plausible; the width shows the precision.",
    "Effect size": "A standardised or clinically meaningful measure of how large a difference or association is, independent of the sample size.",
    "Hedges' g": "The standardised mean difference (difference divided by the pooled standard deviation) with a small-sample correction. "
                 "Reported with a confidence interval; 0.2, 0.5 and 0.8 are conventional small, medium and large benchmarks.",
    "Cohen's d": "The uncorrected standardised mean difference; Hedges' g is the same quantity with the small-sample bias removed.",
    "Rank-biserial r": "For rank tests: the probability that a value from the comparison group exceeds one from the reference group, "
                       "rescaled to -1..1 (0 = no tendency).",
    "Hodges-Lehmann shift": "The median of all pairwise differences between two groups: the location shift that goes with the Mann-Whitney test.",
    "Eta squared": "Share of the total variance explained by the grouping in a one-way design (0.01 small, 0.06 medium, 0.14 large).",
    "Omega squared": "A less biased version of eta squared for the population.",
    "Cramer's V": "Strength of association between two categorical variables (0 = none, 1 = perfect); the bias-corrected version avoids spurious positive values.",
    "Odds ratio (OR)": "Ratio of the odds of the event in the exposed group to the odds in the reference group. Approximates the relative risk only when the event is rare.",
    "Relative risk (RR)": "Ratio of the event risk in the exposed group to the risk in the reference group (cohort or trial designs).",
    "Risk difference (RD)": "Absolute difference in event risk between the groups; the basis of the number needed to treat.",
    "Number needed to treat (NNT)": "1 / risk difference: how many people must receive the exposure for one additional event to be prevented (NNTB) or caused (NNTH).",
    "Welch's t-test": "Compares two means without assuming equal variances; the default two-group test in this application.",
    "Student's t-test": "Compares two means assuming equal variances; kept for compatibility with older analyses.",
    "Mann-Whitney U test": "Rank-based comparison of two groups; tests whether one group tends to have larger values.",
    "ANOVA": "Analysis of variance: compares the means of three or more groups assuming equal variances.",
    "Welch ANOVA": "ANOVA without the equal-variance assumption; the default for three or more groups here.",
    "Kruskal-Wallis test": "Rank-based comparison of three or more groups.",
    "Games-Howell": "Pairwise comparisons after Welch ANOVA; controls the family-wise error without assuming equal variances.",
    "Tukey HSD": "Pairwise comparisons after classic ANOVA, controlling the family-wise error rate.",
    "Dunn's test": "Pairwise rank comparisons after Kruskal-Wallis, with a multiplicity adjustment (Holm here).",
    "Holm adjustment": "A step-down correction that controls the family-wise error rate and is uniformly more powerful than Bonferroni.",
    "Bonferroni adjustment": "Multiplies each p-value by the number of tests; simple and conservative.",
    "Benjamini-Hochberg (FDR)": "Controls the expected proportion of false discoveries among the significant results; suited to exploratory screens.",
    "Family-wise error rate": "The probability of at least one false positive across a family of tests.",
    "False discovery rate": "The expected proportion of false positives among the results declared significant.",
    "Alpha (significance level)": "The threshold for calling a p-value significant (conventionally 0.05); the accepted type I error rate.",
    "Type I error": "Concluding there is an effect when there is none (a false positive).",
    "Type II error": "Missing a real effect (a false negative); its complement is the power.",
    "Statistical power": "The probability of detecting an effect of a given size with a given sample size and alpha.",
    "Normality": "Whether values follow a bell-shaped distribution; matters most for small samples. Tests become over-sensitive in large samples.",
    "Lilliefors test": "A Kolmogorov-Smirnov test of normality that accounts for the mean and SD being estimated from the data.",
    "Shapiro-Wilk test": "A powerful normality test for small and moderate samples.",
    "Levene's test": "Tests whether groups have equal variances; a diagnostic, not a gate, in this application.",
    "MCAR / MAR / MNAR": "Missing completely at random (missingness unrelated to any value), missing at random (related to observed values), "
                         "missing not at random (related to the missing values themselves).",
    "Little's MCAR test": "Tests whether the observed means differ between missingness patterns; rejection argues against MCAR, non-rejection does not prove it.",
    "Imputation": "Filling in missing values with estimates; single imputation understates uncertainty, multiple imputation propagates it.",
    "Sensitivity": "Proportion of true cases that the test identifies as positive.",
    "Specificity": "Proportion of true non-cases that the test identifies as negative.",
    "Positive / negative predictive value": "Probability of disease given a positive test, and of no disease given a negative test; both depend on the prevalence.",
    "Likelihood ratio": "How much a positive (LR+) or negative (LR-) result changes the odds of disease; independent of the prevalence.",
    "ROC curve and AUC": "Sensitivity against 1 - specificity across all cut-offs; the area under the curve (0.5 = chance, 1 = perfect) summarises discrimination.",
    "Youden index": "Sensitivity + specificity - 1; the cut-off maximising it is optimistic when chosen on the same data.",
    "Bland-Altman plot": "Differences between two measurement methods against their means, with the bias and the limits of agreement.",
    "Limits of agreement": "Bias +/- 1.96 SD of the differences: the range within which most differences between two methods fall.",
    "Chi-square test": "Tests the association between two categorical variables when expected counts are adequate.",
    "Fisher's exact test": "An exact test of association for 2x2 tables, used when expected counts are small.",
    "ANCOVA": "Compares group means after adjusting for continuous covariates.",
    "Z-score": "Number of standard deviations a value lies from the mean of a reference population.",
    "Trace": "The recorded sequence of every operation applied to a dataset, used to reproduce an analysis.",
}


def define(term: str) -> str | None:
    """Case-insensitive lookup."""
    lowered = term.strip().lower()
    for key, text in GLOSSARY.items():
        if key.lower() == lowered:
            return text
    return None


__all__ = ["GLOSSARY", "define"]
