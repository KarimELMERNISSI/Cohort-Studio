"""
Declarative test chooser ("Which test?"): four questions -> a recommendation that names what the
application runs, why, the effect size to report, and whether the route exists in the app yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ui import nav

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
    tab: str | None = None  # the tab of that page its link opens (ui.nav.PageDef.tabs)


_PLANNED = "Not available in this release; the handbook lists the method."
_ANALYSIS = "analysis"  # the Analysis tab of the page "analyze": group comparisons
# the questions the four above cannot reach, and the analysis that answers each (a tab of the page "analyze")
OTHER_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("How well does a test identify a condition?", "accuracy"),
    ("Can two methods replace each other? How reliable is a measurement?", "measurements"),
    ("Do two raters agree?", "raters"),
    ("Is an association the same within the groups of a third variable?", "strata"),
    ("How many events per person-time, in one group or two?", "rates"),
    ("More events than expected from reference rates?", "expected"),
)


def _on(tab: str) -> dict[str, str]:
    """A route to an analysis of Hypothesis tests & epidemiology: its page, the tab its link opens, and that place in
    the words of the navigation (renamed with it, never left behind)."""
    return {"page": "analyze", "tab": tab, "where": nav.where("analyze", tab)}


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
            page="survival",
            where="Survival & regression page (Kaplan-Meier, log-rank, Cox)",
            available=not (paired or one),
            caveats=(
                "Check the proportional-hazards assumption (Grambsch-Therneau test on the page) before reporting a single hazard ratio.",
            )
            + ((_PLANNED,) if (paired or one) else ()),
        )

    if outcome in ("Continuous (measurement)", "Ordinal (ranked levels)"):
        rank_only = outcome == "Ordinal (ranked levels)"
        if one:
            return Recommendation(
                test="Wilcoxon signed-rank test (or one-sample t-test for symmetric measurements)",
                why="A single group is compared with a fixed reference value.",
                effect_size="difference from the reference with 95% CI",
                **_on("one_mean"),
            )
        if paired:
            return Recommendation(
                test="Wilcoxon signed-rank test"
                if (rank_only or not symmetric)
                else "Paired t-test",
                why="Each participant contributes two measurements; the pairing must be respected.",
                effect_size="mean (or pseudo-median) within-pair difference with 95% CI; Cohen's d_z",
                **_on("paired_number"),
            )
        if two:
            if rank_only or not symmetric:
                return Recommendation(
                    test="Mann-Whitney U test",
                    why="Ranks are robust to skew and outliers and are the natural scale for ordinal outcomes.",
                    effect_size="rank-biserial r and the Hodges-Lehmann shift, both with 95% CI",
                    page="analyze",
                    where="Analysis tab > Univariate, test strategy 'Force non-parametric'",
                    tab=_ANALYSIS,
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
                tab=_ANALYSIS,
                alternatives=("Mann-Whitney U when the distribution is skewed",),
            )
        if rank_only or not symmetric:
            return Recommendation(
                test="Kruskal-Wallis test, then Dunn's pairwise comparisons with Holm adjustment",
                why="A rank-based omnibus test followed by all pairwise comparisons with the family controlled.",
                effect_size="eta squared (H); rank-biserial r per pair",
                page="analyze",
                where="Analysis tab > Univariate, test strategy 'Force non-parametric'; pairwise table in the detail view",
                tab=_ANALYSIS,
            )
        return Recommendation(
            test="Welch ANOVA, then Games-Howell pairwise comparisons",
            why="Welch ANOVA tolerates unequal variances; Games-Howell reports every pair with the family-wise error controlled.",
            effect_size="eta squared and omega squared with 95% CI; Hedges' g per pair",
            page="analyze",
            where="Analysis tab > Univariate (default strategy); pairwise table in the detail view",
            tab=_ANALYSIS,
            alternatives=("Classic ANOVA + Tukey HSD when variances are clearly equal",),
        )

    if outcome == "Binary (yes/no)":
        if one:
            return Recommendation(
                test="Exact binomial test with a Wilson confidence interval for the proportion",
                why="A single proportion is compared with a reference value.",
                effect_size="proportion with 95% Wilson CI",
                **_on("one_share"),
            )
        if paired and two:
            return Recommendation(
                test="McNemar's test (exact)",
                why="Paired binary outcomes: only the discordant pairs carry information.",
                effect_size="difference in proportions with 95% CI",
                **_on("paired_yes_no"),
            )
        if paired:
            return Recommendation(
                test="Cochran's Q test",
                why="Three or more related binary measurements per participant.",
                effect_size="pairwise McNemar differences with Holm adjustment",
                page="analyze",
                where="planned",
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
                tab=_ANALYSIS,
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
            tab=_ANALYSIS,
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
        tab=_ANALYSIS,
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
    "OTHER_QUESTIONS",
    "OUTCOMES",
    "Recommendation",
    "all_combinations",
    "recommend",
]
