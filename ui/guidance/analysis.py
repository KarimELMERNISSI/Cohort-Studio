"""
Guidance for Analyze > Hypothesis tests & epidemiology: help texts, the "which tab" orientation,
the plan shown before a group comparison runs, ANCOVA capacity checks, effect-size helpers for the
power calculator and plain-language result sentences.

Pure Python (no Streamlit): the page renders these, the tests check them. The predicted tests use the
same rules as the engine (``biostats.inference.recommend`` for numeric variables, expected counts for
categorical ones), so the plan says exactly what Run will do.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from biostats.inference import DISPLAY_NAMES, groups_from_frame
from biostats.inference import recommend as recommend_kind

# ----------------------------------------------------------------------------- help texts

HELP: dict[str, str] = {
    # page
    "guidance": "Shows the plan before each run, checks on your data and plain-language results. "
    "Turn it off once you know the page.",
    "transform": "Applied to numeric variables for the Analysis tab only. Effect sizes and means are then on the "
    "transformed scale. Keep the raw data unless a positive, right-skewed variable needs a log scale.\n"
    "- Log: reduces right skew (values >= 0).\n- Z-score: mean 0, SD 1.\n- Min-max: scales to [0, 1].",
    # group comparison
    "group_col": "The variable that defines the groups to compare (treatment arm, sex, exposure). Variables with "
    "fewer than 20 distinct values are offered.",
    "groups": "Groups kept in the comparison. Groups with fewer than 5 observations give unstable estimates: "
    "deselect them or merge categories first (Data > Enrich).",
    "targets": "Outcomes compared across the groups, one test per variable: numeric variables get a Welch-type "
    "or rank-based test, categorical ones a chi-square or Fisher's exact test.",
    "strategy": "Auto-detect (recommended): Welch's test, or a rank-based test when a group has fewer than 15 values "
    "and the variable is strongly skewed. Force a family only when your protocol pre-specified it.",
    "correction": "The family is the set of variables tested in this run. Holm controls the family-wise error rate "
    "(confirmatory comparisons); Benjamini-Hochberg controls the false discovery rate (exploratory screens of many "
    "variables); Bonferroni is the conservative classic. Raw and adjusted p-values are both shown.",
    "detail_variable": "Opens the full result for one variable: estimate with its confidence interval, effect size, "
    "pairwise comparisons, the 2x2 measures for a binary outcome, plots and assumption checks.",
    "detail_test": "Auto-Detect repeats the choice made above. Pick another test only to check robustness (for "
    "example a rank test next to Welch's test) and report the pre-specified one.",
    "ref_value": "Clinical threshold or normal-range limit drawn on the plots, in the variable's unit.",
    "ref_label": "Text shown next to the reference line.",
    "ref_show": "Draws the reference line on the distribution and means charts.",
    # ANCOVA
    "ancova_target": "The continuous outcome compared across groups after adjustment.",
    "ancova_group": "The categorical variable defining the groups (treatment, exposure). Fewer than 10 levels.",
    "ancova_covariates": "Variables measured before group assignment that predict the outcome (age, baseline value). "
    "Do not adjust for consequences of the group (mediators). Rule of thumb: at least 10 to 20 complete "
    "observations per model parameter.",
    # z-scores
    "z_mean": "Published reference mean, in the variable's unit (population norm tables, growth charts).",
    "z_sd": "Published reference standard deviation, in the same unit; must be positive.",
    "z_group_by": "Splits the Z-score comparison by this variable.",
    "z_sort": "Order of the participants (rows) in the heatmap.",
    "z_id_col": "Column that identifies each participant in the profile view.",
    "z_row": "Row number of the participant to display.",
    "z_sample": "Participant to display.",
    # power
    "power_import": "A result from the Analysis tab pre-fills the effect size and the group sizes. Use it to size the "
    "next study: power computed from an observed effect says nothing new about the current result.",
    "power_autofill_group": "Uses the group sizes of this variable to pre-fill the size ratio and the number of groups.",
    "power_test": "The test you will use in the final analysis: t-test for a numeric outcome in 2 groups, ANOVA for 3 "
    "or more, Mann-Whitney or Kruskal-Wallis if you plan rank-based tests, chi-square for a categorical outcome.",
    "power_goal": "Sample size: how many participants are needed to detect the effect. Power: the chance that a "
    "planned or available sample detects it.",
    "alpha": "Two-sided false-positive rate; 0.05 is the convention. Use a smaller value (0.025, or alpha divided by "
    "the number of primary comparisons) when several primary comparisons are planned.",
    "power": "Chance of detecting the effect if it exists. 0.80 is the usual minimum; 0.90 for pivotal studies.",
    "ratio": "Size of group 2 divided by group 1 (1 = equal groups). Unequal allocation needs more participants in "
    "total for the same power.",
    "n1": "Planned or available number of participants in group 1 (group 2 = ratio x this number).",
    "n_per_group": "Planned or available number of participants in each group (equal groups).",
    "n_total": "Planned or available total number of participants (all groups together).",
    "k_groups": "Number of groups that will be compared.",
    "es_mode": "Give the standardised effect size directly, or describe the effect you expect in the outcome's own "
    "terms (means and SD, or event proportions) and let the app convert it.",
    "es_d": "The smallest difference worth detecting, in standard deviations: (mean 1 - mean 2) / SD. Base it on a "
    "minimal clinically important difference or pilot data, not on the effect you hope for. "
    "Conventions: 0.2 small, 0.5 medium, 0.8 large.",
    "es_mwu_scale": "P(X < Y): probability that a participant of group 1 has a lower value than one of group 2 "
    "(0.5 = no effect). Rank-biserial r = 2P - 1. Cohen's d is converted assuming normal data.",
    "es_mwu": "The smallest effect worth detecting, on the chosen scale.",
    "es_anova_scale": "Cohen's f (0.10 small, 0.25 medium, 0.40 large) or eta squared, the share of the outcome's "
    "variance explained by the groups (0.01, 0.06, 0.14).",
    "es_anova": "The smallest effect worth detecting, on the chosen scale.",
    "es_kw_scale": "Cohen's f equivalent (0.10 small, 0.25 medium, 0.40 large) or epsilon squared, the rank-based "
    "share of variance explained (0.01, 0.06, 0.14).",
    "es_chi2_scale": "Cohen's w (0.1 small, 0.3 medium, 0.5 large) or Cramer's V, converted with the table "
    "dimension.",
    "es_chi2": "The smallest association worth detecting, on the chosen scale.",
    "min_dim": "min(rows - 1, columns - 1) of the planned table; 1 for any table with a binary variable. Used to "
    "convert Cramer's V into Cohen's w.",
    "mean1": "Expected mean of the outcome in group 1, in the outcome's unit.",
    "mean2": "Expected mean of the outcome in group 2, in the outcome's unit.",
    "sd": "Expected standard deviation of the outcome within groups (pilot data or published studies).",
    "group_means": "Expected mean of the outcome in each group, separated by commas (for example 120, 125, 132).",
    "proportions": "Expected proportion of participants with the event in each group, separated by commas "
    "(for example 0.20, 0.35). Equal group sizes are assumed.",
}

# ----------------------------------------------------------------------------- orientation


@dataclass(frozen=True)
class TabGuide:
    tab: str
    question: str
    examples: str


TAB_GUIDE: tuple[TabGuide, ...] = (
    TabGuide(
        "Analysis",
        "Do groups differ on one or more variables?",
        "Treatment vs placebo on blood pressure; event rate by arm (risk ratio, odds ratio, NNT). "
        "Choose Multivariate above to adjust for confounders (ANCOVA).",
    ),
    TabGuide(
        "Diagnostic accuracy & agreement",
        "How well does a test identify a condition? Do two methods or raters agree?",
        "Sensitivity, specificity, ROC and AUC; Bland-Altman limits of agreement; kappa and ICC.",
    ),
    TabGuide(
        "Paired, rates & stratified",
        "Same participants measured twice? Events per person-time? An association adjusted for one stratum?",
        "Paired t-test, Wilcoxon, McNemar; incidence rates and rate ratios, SMR; Mantel-Haenszel.",
    ),
    TabGuide(
        "Power & Sample Size",
        "How many participants do I need, or how much power does my sample have?",
        "Planning a study, or judging whether a non-significant result was informative.",
    ),
    TabGuide(
        "Z-Score & Reference",
        "How far are values from a reference population?",
        "Lab or growth values against controls or published norms.",
    ),
    TabGuide(
        "Test Guide & Education",
        "Which test fits my question, and how do I read the results?",
        "Decision tree, effect-size conventions, p-values and errors.",
    ),
)

# Notes shown when a test is chosen by hand in the detailed view.
TEST_NOTES: dict[str, str] = {
    "Auto-Detect": "Same choice as the summary table.",
    "Student's t-test": "Assumes equal variances in both groups; prefer Welch's test unless they are known to be equal.",
    "Welch's t-test": "Compares means without assuming equal variances; the default for two groups.",
    "Mann-Whitney U": "Rank-based: compares the distributions, robust to outliers and skew; reports the "
    "Hodges-Lehmann shift.",
    "ANOVA": "Assumes equal variances across groups; Welch ANOVA is safer when group variances differ.",
    "Kruskal-Wallis": "Rank-based comparison of three or more groups; Dunn's tests give the pairs.",
    "Chi-Square": "Needs expected counts of at least 5 in nearly every cell.",
    "Fisher's Exact": "Exact test for small 2x2 tables (2x2 only).",
}

# ----------------------------------------------------------------------------- plan of a group comparison

STRATEGY_PARAMETRIC = "Force Parametric (T-test/ANOVA)"
STRATEGY_NONPARAMETRIC = "Force Non-Parametric (Mann-Whitney/Kruskal)"


@dataclass(frozen=True)
class Advice:
    level: str  # "ok" | "info" | "caution" | "violated" (as status chips)
    text: str


@dataclass(frozen=True)
class PlannedTest:
    variable: str
    kind: str  # numeric | categorical | skipped
    test: str
    reason: str
    n: int = 0
    missing_share: float = 0.0


@dataclass(frozen=True)
class CorrectionAdvice:
    option: str  # one of biostats.multiplicity.UI_OPTIONS
    reason: str


@dataclass(frozen=True)
class AnalysisPlan:
    group_sizes: dict[str, int]
    tests: tuple[PlannedTest, ...]
    advice: tuple[Advice, ...] = ()
    correction: CorrectionAdvice | None = None
    expected_false_positives: float = 0.0
    pairs: int = 0  # pairwise comparisons per numeric variable when there are 3 or more groups
    notes: tuple[str, ...] = field(default_factory=tuple)


def recommend_correction(n_tests: int) -> CorrectionAdvice:
    if n_tests <= 1:
        return CorrectionAdvice("None", "One test: there is no family of comparisons to correct.")
    if n_tests < 10:
        return CorrectionAdvice(
            "Holm (recommended)",
            f"{n_tests} tests: Holm keeps the chance of any false positive at 5% and is more powerful than Bonferroni.",
        )
    return CorrectionAdvice(
        "Benjamini-Hochberg (FDR)",
        f"{n_tests} tests: for a screen of many variables, Benjamini-Hochberg keeps the share of false discoveries at "
        "5%. Use Holm instead if these are pre-specified confirmatory comparisons.",
    )


def _numeric_test(
    df: pd.DataFrame, group_col: str, target: str, strategy: str, k: int
) -> tuple[str, str]:
    if strategy == STRATEGY_PARAMETRIC:
        kind = "welch" if k == 2 else "welch_anova"
        return DISPLAY_NAMES[kind], "parametric family forced"
    if strategy == STRATEGY_NONPARAMETRIC:
        kind = "mannwhitney" if k == 2 else "kruskal"
        return DISPLAY_NAMES[kind], "rank-based family forced"
    groups = {g: v for g, v in groups_from_frame(df, group_col, target).items() if len(v) > 0}
    if len(groups) < 2 or min(len(v) for v in groups.values()) < 2:
        return "not computed", "fewer than two groups with two or more values"
    kind, reasons = recommend_kind(groups)
    if kind in ("welch", "student", "mannwhitney") and len(groups) > 2:
        kind = {"welch": "welch_anova", "student": "anova", "mannwhitney": "kruskal"}[kind]
    reason = reasons[0] if reasons else ""
    if kind in ("welch", "welch_anova"):
        reason = "unequal variances allowed (Welch)"
    return DISPLAY_NAMES[kind], reason


def _categorical_test(df: pd.DataFrame, group_col: str, target: str) -> tuple[str, str]:
    table = pd.crosstab(df[group_col], df[target])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return "not computed", "the outcome takes a single value in these groups"
    expected = stats.contingency.expected_freq(table.to_numpy())
    small = int((expected < 5).sum())
    if small and table.shape == (2, 2):
        return (
            "Fisher's exact test",
            f"{small} expected count(s) below 5; risk ratio, odds ratio and NNT in the detail",
        )
    if small:
        return (
            "Chi-square",
            f"{small} of {expected.size} expected counts below 5: sparse table, merge categories if you can",
        )
    if table.shape == (2, 2):
        return (
            "Chi-square",
            "2x2 table; risk ratio, odds ratio, risk difference and NNT in the detail",
        )
    return "Chi-square", f"{table.shape[0]}x{table.shape[1]} table, all expected counts at least 5"


def plan_group_comparison(
    df: pd.DataFrame,
    group_col: str,
    groups: Sequence,
    targets: Sequence[str],
    *,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    strategy: str = "Auto-Detect (Recommended)",
    correction: str = "None",
    transformed: bool = False,
) -> AnalysisPlan:
    """What Run will do with this configuration, and what to watch out for."""
    data = df[df[group_col].isin(list(groups))]
    sizes = {str(g): int(n) for g, n in data[group_col].value_counts().sort_index().items()}
    k = len(sizes)
    advice: list[Advice] = []
    tiny = [g for g, n in sizes.items() if n < 2]
    small = [g for g, n in sizes.items() if 2 <= n < 5]
    if tiny:
        advice.append(
            Advice(
                "violated",
                f"{', '.join(tiny)}: fewer than 2 observations, cannot be compared. "
                "Deselect in Groups to include.",
            )
        )
    if small:
        advice.append(
            Advice(
                "caution",
                f"{', '.join(small)}: fewer than 5 observations, estimates will be unstable. "
                "Merge categories or deselect them.",
            )
        )
    if sizes and min(sizes.values()) >= 5 and min(sizes.values()) < 15:
        advice.append(
            Advice(
                "info",
                "Small groups (fewer than 15): strongly skewed variables will get a rank-based test.",
            )
        )
    if sizes and min(sizes.values()) > 0:
        imbalance = max(sizes.values()) / min(sizes.values())
        if imbalance >= 3:
            advice.append(
                Advice(
                    "info",
                    f"Unbalanced groups (largest {imbalance:.1f}x the smallest): Welch tests cope "
                    "with unequal variances, but power is driven by the smallest group.",
                )
            )
    if k > 6:
        advice.append(
            Advice(
                "caution",
                f"{k} groups: the omnibus test only says that some groups differ; consider "
                "merging levels or a pre-specified contrast.",
            )
        )
    if transformed:
        advice.append(
            Advice(
                "info",
                "Numeric variables are analysed on the transformed scale: means and effect "
                "sizes refer to the transformed values.",
            )
        )

    numeric = set(numeric_cols)
    categorical = set(categorical_cols)
    planned: list[PlannedTest] = []
    for target in targets:
        values = data[target]
        n = int(values.notna().sum())
        missing = 1 - n / len(values) if len(values) else 0.0
        if int(values.nunique(dropna=True)) < 2:
            planned.append(
                PlannedTest(
                    target,
                    "skipped",
                    "not computed",
                    "a single value: nothing to compare",
                    n,
                    missing,
                )
            )
            continue
        subset = data[[group_col, target]].dropna()
        if target in numeric:
            test, reason = _numeric_test(subset, group_col, target, strategy, k)
            planned.append(PlannedTest(target, "numeric", test, reason, n, missing))
        elif target in categorical:
            levels = int(values.nunique(dropna=True))
            test, reason = _categorical_test(subset, group_col, target)
            if levels > 10:
                reason = (
                    f"{levels} categories: very sparse table, consider grouping them; " + reason
                )
            planned.append(PlannedTest(target, "categorical", test, reason, n, missing))
        else:
            planned.append(
                PlannedTest(
                    target,
                    "skipped",
                    "not tested",
                    "neither numeric nor categorical (text, date)",
                    n,
                    missing,
                )
            )
        if missing >= 0.2:
            advice.append(
                Advice(
                    "caution",
                    f"{target}: {missing:.0%} missing in these groups. The comparison uses "
                    "complete cases only and may be biased; check the pattern on Validate.",
                )
            )

    n_tests = sum(1 for p in planned if p.kind != "skipped")
    rec = recommend_correction(n_tests)
    if n_tests >= 2 and correction == "None":
        advice.append(
            Advice(
                "caution",
                f"{n_tests} tests without correction: about {n_tests * 0.05:.1f} "
                "false positive(s) expected by chance alone.",
            )
        )
    pairs = k * (k - 1) // 2 if k > 2 else 0
    notes = []
    if pairs:
        notes.append(
            f"With {k} groups the summary test asks whether any group differs; the detailed view lists all "
            f"{pairs} pairs with adjusted p-values."
        )
    return AnalysisPlan(
        group_sizes=sizes,
        tests=tuple(planned),
        advice=tuple(advice),
        correction=rec,
        expected_false_positives=n_tests * 0.05,
        pairs=pairs,
        notes=tuple(notes),
    )


# ----------------------------------------------------------------------------- ANCOVA


@dataclass(frozen=True)
class AncovaCheck:
    n_complete: int
    n_excluded: int
    parameters: int
    per_parameter: float
    advice: tuple[Advice, ...]


def ancova_check(
    df: pd.DataFrame, target: str, group: str, covariates: Sequence[str]
) -> AncovaCheck:
    """Observations per model parameter and common pitfalls of the covariate set."""
    cols = [target, group, *covariates]
    complete = df[cols].dropna()
    n = len(complete)
    k = int(complete[group].nunique()) if n else 0
    parameters = 1 + max(k - 1, 0) + len(covariates)
    per = n / parameters if parameters else 0.0
    advice: list[Advice] = []
    if n and k < 2:
        advice.append(
            Advice(
                "violated",
                f"Only one group of {group} has complete data in these variables: there is "
                "nothing to compare. Check the missing values of the covariates.",
            )
        )
    if per < 10:
        advice.append(
            Advice(
                "violated",
                f"{per:.1f} complete observations per model parameter (fewer than 10): the "
                "model is unstable. Remove covariates or collect more data.",
            )
        )
    elif per < 20:
        advice.append(
            Advice(
                "caution",
                f"{per:.1f} observations per parameter: acceptable, but keep the covariate set "
                "small and pre-specified.",
            )
        )
    else:
        advice.append(
            Advice("ok", f"{per:.0f} observations per parameter: enough for a stable model.")
        )
    excluded = len(df) - n
    if excluded:
        share = excluded / len(df) if len(df) else 0
        level = "caution" if share >= 0.2 else "info"
        advice.append(
            Advice(
                level,
                f"{excluded} rows ({share:.0%}) have a missing value in the model variables and "
                "are excluded.",
            )
        )
    for cov in covariates:
        if n and int(complete[cov].nunique()) <= 2:
            advice.append(
                Advice(
                    "info", f"{cov} has two values only: it acts as a second factor, not a slope."
                )
            )
        if n and k >= 2:
            # covariates should be balanced-ish or at least not determined by the group
            r2 = _eta_squared(complete[cov], complete[group])
            if r2 >= 0.5:
                advice.append(
                    Advice(
                        "caution",
                        f"{cov} is largely explained by {group} (eta squared {r2:.2f}): it may be "
                        "a consequence of the group (mediator) or nearly collinear with it.",
                    )
                )
    if not covariates:
        advice.append(Advice("info", "No covariate: the model is a one-way ANOVA."))
    return AncovaCheck(n, excluded, parameters, per, tuple(advice))


def _eta_squared(values: pd.Series, groups: pd.Series) -> float:
    y = pd.to_numeric(values, errors="coerce")
    frame = pd.DataFrame({"y": y, "g": groups}).dropna()
    if frame.empty or float(frame["y"].var(ddof=0)) == 0:
        return 0.0
    grand = frame["y"].mean()
    between = frame.groupby("g")["y"].agg(["count", "mean"])
    ss_between = float((between["count"] * (between["mean"] - grand) ** 2).sum())
    ss_total = float(((frame["y"] - grand) ** 2).sum())
    return ss_between / ss_total if ss_total else 0.0


# ----------------------------------------------------------------------------- effect sizes for planning


def parse_numbers(text: str) -> list[float]:
    """Numbers separated by commas, semicolons or spaces ('120, 125; 132')."""
    values = []
    for part in text.replace(";", ",").replace(" ", ",").split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values


def cohens_d_from_means(mean1: float, mean2: float, sd: float) -> float:
    if sd <= 0:
        raise ValueError("The standard deviation must be positive.")
    return abs(mean1 - mean2) / sd


def cohens_f_from_means(means: Sequence[float], sd: float) -> float:
    """Cohen's f for equal group sizes: SD of the group means / within-group SD."""
    if sd <= 0:
        raise ValueError("The standard deviation must be positive.")
    if len(means) < 2:
        raise ValueError("Give at least two group means.")
    m = np.asarray(means, dtype=float)
    return float(np.sqrt(np.mean((m - m.mean()) ** 2)) / sd)


def cohens_w_from_proportions(
    proportions: Sequence[float], shares: Sequence[float] | None = None
) -> float:
    """Cohen's w for a k x 2 table (event yes/no by group), equal group sizes unless ``shares`` are given."""
    p = np.asarray(proportions, dtype=float)
    if len(p) < 2:
        raise ValueError("Give the expected proportion in at least two groups.")
    if ((p <= 0) | (p >= 1)).any():
        raise ValueError("Proportions must be strictly between 0 and 1.")
    s = (
        np.full(len(p), 1 / len(p))
        if shares is None
        else np.asarray(shares, dtype=float) / np.sum(shares)
    )
    observed = np.column_stack([s * p, s * (1 - p)])
    pbar = float(np.sum(s * p))
    expected = np.column_stack([s * pbar, s * (1 - pbar)])
    return float(np.sqrt(np.sum((observed - expected) ** 2 / expected)))


def p_superiority_from_d(d: float) -> float:
    """P(X < Y) for two normal groups separated by d standard deviations."""
    return float(stats.norm.cdf(d / math.sqrt(2)))


# ----------------------------------------------------------------------------- plain-language results


def sample_size_sentence(
    effect: str,
    *,
    power: float,
    alpha: float,
    n_groups: Sequence[int] | None = None,
    n_total: float | None = None,
) -> str:
    if n_groups:
        sizes = [int(math.ceil(n)) for n in n_groups]
        total = sum(sizes)
        if len(set(sizes)) == 1:
            who = f"{sizes[0]} participants per group ({total} in total)"
        else:
            who = (
                " and ".join(f"{n} in group {i + 1}" for i, n in enumerate(sizes))
                + f" ({total} in total)"
            )
    else:
        total = int(math.ceil(n_total or 0))
        who = f"{total} participants in total"
    return (
        f"To detect {effect} with {power:.0%} power at a two-sided alpha of {alpha:g}, you need {who}. "
        "Add a margin for dropouts and missing data (often 10 to 20%)."
    )


def power_sentence(effect: str, power: float, *, sample: str) -> str:
    text = f"With {sample}, the chance of detecting {effect} is {power:.0%}."
    if power < 0.5:
        text += " A non-significant result would say little: most real effects of this size would be missed."
    elif power < 0.8:
        text += " This is below the usual 80% target: a non-significant result would be hard to interpret."
    else:
        text += " This meets the usual 80% target."
    return text


__all__ = [
    "HELP",
    "TAB_GUIDE",
    "TEST_NOTES",
    "Advice",
    "AnalysisPlan",
    "AncovaCheck",
    "CorrectionAdvice",
    "PlannedTest",
    "TabGuide",
    "ancova_check",
    "cohens_d_from_means",
    "cohens_f_from_means",
    "cohens_w_from_proportions",
    "p_superiority_from_d",
    "parse_numbers",
    "plan_group_comparison",
    "power_sentence",
    "recommend_correction",
    "sample_size_sentence",
]
