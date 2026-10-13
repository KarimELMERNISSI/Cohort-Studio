"""
Declarative test chooser ("Which test?"): four questions -> a recommendation that names what the
application runs, why, the effect size to report, and whether the route exists in the app yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

OUTCOMES = (
    "Continuous (measurement)",
    "Binary (yes/no)",
    "Categorical (3+ levels)",
    "Ordinal (ranked levels)",
    "Time to event",
)
GROUPS = ("Two groups", "Three or more groups", "One group vs a reference value")
DESIGNS = ("Independent groups", "Paired / repeated measures")
DISTRIBUTIONS = (
    "Roughly symmetric, at least 15 per group",
    "Skewed, outliers or fewer than 15 per group",
)


@dataclass(frozen=True)
class Recommendation:
    test: str
    why: str
    effect_size: str
    page: str  # navigation key
    where: str  # where on that page
    available: bool = True
    alternatives: tuple[str, ...] = ()
    caveats: tuple[str, ...] = field(default_factory=tuple)


_PLANNED = "Not available in this release (planned with the survival/paired-test module); the handbook lists the method."


def recommend(outcome: str, groups: str, design: str, distribution: str) -> Recommendation:
    if (
        outcome not in OUTCOMES
        or groups not in GROUPS
        or design not in DESIGNS
        or distribution not in DISTRIBUTIONS
    ):
        raise ValueError("unknown answer")
    symmetric = distribution == DISTRIBUTIONS[0]
    paired = design == DESIGNS[1]
    two, many, one = (groups == g for g in GROUPS)

    if outcome == "Time to event":
        return Recommendation(
            test="Kaplan-Meier curves with the log-rank test; Cox regression to adjust for covariates",
            why="Follow-up times are censored: ordinary group tests on durations are biased.",
            effect_size="hazard ratio with 95% CI; median survival per group",
            page="analyze",
            where="Survival (planned)",
            available=False,
            caveats=(
                _PLANNED,
                "Check the proportional-hazards assumption before reporting a single hazard ratio.",
            ),
        )

    if outcome in ("Continuous (measurement)", "Ordinal (ranked levels)"):
        rank_only = outcome == "Ordinal (ranked levels)"
        if one:
            return Recommendation(
                test="Wilcoxon signed-rank test (or one-sample t-test for symmetric measurements)",
                why="A single group is compared with a fixed reference value.",
                effect_size="difference from the reference with 95% CI",
                page="analyze",
                where="Z-score & reference tab (standardisation only)",
                available=False,
                caveats=(
                    _PLANNED,
                    "Use the Z-score tab to express values relative to a reference population meanwhile.",
                ),
            )
        if paired:
            return Recommendation(
                test="Wilcoxon signed-rank test"
                if (rank_only or not symmetric)
                else "Paired t-test",
                why="Each subject contributes two measurements; the pairing must be respected.",
                effect_size="mean (or median) within-pair difference with 95% CI",
                page="analyze",
                where="Paired tests (planned)",
                available=False,
                caveats=(_PLANNED,),
            )
        if two:
            if rank_only or not symmetric:
                return Recommendation(
                    test="Mann-Whitney U test",
                    why="Ranks are robust to skew and outliers and are the natural scale for ordinal outcomes.",
                    effect_size="rank-biserial r and the Hodges-Lehmann shift, both with 95% CI",
                    page="analyze",
                    where="Analysis tab > Univariate, test strategy 'Force non-parametric'",
                    alternatives=(
                        "Welch's t-test if the measurement is symmetric and groups are not tiny",
                    ),
                )
            return Recommendation(
                test="Welch's t-test",
                why="Welch does not assume equal variances and loses almost nothing when they are equal (Delacre 2017).",
                effect_size="mean difference with 95% CI and Hedges' g with 95% CI",
                page="analyze",
                where="Analysis tab > Univariate (default strategy)",
                alternatives=("Mann-Whitney U when the distribution is skewed",),
            )
        if rank_only or not symmetric:
            return Recommendation(
                test="Kruskal-Wallis test, then Dunn's pairwise comparisons with Holm adjustment",
                why="A rank-based omnibus test followed by all pairwise comparisons with the family controlled.",
                effect_size="eta squared (H); rank-biserial r per pair",
                page="analyze",
                where="Analysis tab > Univariate, test strategy 'Force non-parametric'; pairwise table in the detail view",
            )
        return Recommendation(
            test="Welch ANOVA, then Games-Howell pairwise comparisons",
            why="Welch ANOVA tolerates unequal variances; Games-Howell reports every pair with the family-wise error controlled.",
            effect_size="eta squared and omega squared with 95% CI; Hedges' g per pair",
            page="analyze",
            where="Analysis tab > Univariate (default strategy); pairwise table in the detail view",
            alternatives=("Classic ANOVA + Tukey HSD when variances are clearly equal",),
        )

    if outcome == "Binary (yes/no)":
        if one:
            return Recommendation(
                test="Exact binomial test with a Wilson confidence interval for the proportion",
                why="A single proportion is compared with a reference value.",
                effect_size="proportion with 95% Wilson CI",
                page="analyze",
                where="planned",
                available=False,
                caveats=(_PLANNED,),
            )
        if paired:
            return Recommendation(
                test="McNemar's test (exact)",
                why="Paired binary outcomes: only the discordant pairs carry information.",
                effect_size="difference in proportions with 95% CI",
                page="analyze",
                where="Paired tests (planned)",
                available=False,
                caveats=(_PLANNED,),
            )
        if two:
            return Recommendation(
                test="Pearson chi-square test (Fisher's exact test when an expected count is below 5) with the 2x2 measures",
                why="Two groups and a binary outcome form a 2x2 table; the measures of association answer the clinical question.",
                effect_size="relative risk or odds ratio, risk difference and NNT, all with 95% CI (choose by study design)",
                page="analyze",
                where="Analysis tab > Univariate > detail view > 2x2 panel",
                caveats=(
                    "State the reference group and the event level explicitly; the app asks for both.",
                ),
            )
        return Recommendation(
            test="Pearson chi-square test on the r x 2 table, then pairwise 2x2 comparisons with Holm adjustment",
            why="Several groups and a binary outcome; the omnibus test is followed by adjusted pairwise contrasts.",
            effect_size="bias-corrected Cramer's V; risk differences per pair with 95% CI",
            page="analyze",
            where="Analysis tab > Univariate",
        )

    # categorical 3+ levels
    if paired or one:
        return Recommendation(
            test="Bowker's symmetry test" if paired else "Chi-square goodness-of-fit",
            why="Paired or single-sample categorical outcomes need symmetry or goodness-of-fit tests.",
            effect_size="Cramer's V",
            page="analyze",
            where="planned",
            available=False,
            caveats=(_PLANNED,),
        )
    return Recommendation(
        test="Pearson chi-square test of independence (r x c)",
        why="Both variables are categorical; the test asks whether the distribution differs between groups.",
        effect_size="bias-corrected Cramer's V",
        page="analyze",
        where="Analysis tab > Univariate",
        caveats=(
            "Expected counts below 5 in many cells make the chi-square approximation unreliable; merge sparse levels.",
        ),
    )


def all_combinations() -> list[tuple[str, str, str, str]]:
    return [(o, g, d, s) for o in OUTCOMES for g in GROUPS for d in DESIGNS for s in DISTRIBUTIONS]


__all__ = [
    "DESIGNS",
    "DISTRIBUTIONS",
    "GROUPS",
    "OUTCOMES",
    "Recommendation",
    "all_combinations",
    "recommend",
]
