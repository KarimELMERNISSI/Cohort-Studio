import logging

import numpy as np
import pandas as pd
from scipy import stats

from biostats.inference import LEGACY_NAMES, compare_groups, groups_from_frame
from biostats.inference import recommend as _recommend_kind
from biostats.posthoc import default_method, pairwise_comparisons

logger = logging.getLogger(__name__)


def render_analysis_configuration(*args, **kwargs):
    """Compatibility shim: the widgets live in :mod:`app_pages.analysis_config`."""
    from app_pages.analysis_config import render_analysis_configuration as _render

    return _render(*args, **kwargs)


def analyze_variable(df, group_col, target, numeric_cols, categorical_cols, binary_cols, manual_test="Auto-Detect",
                     *, exposure_ref=None, event_level=None, design="unknown"):
    """
    Analyzes a single variable against the group column, selecting the appropriate test.

    For binary outcomes the 2x2 measures use an explicit orientation (audit S1): ``exposure_ref`` is the
    reference level of ``group_col`` and ``event_level`` the outcome level counted as the event; both are
    auto-detected (0/No/control vs 1/Yes/case) when omitted.
    """
    # Prepare data
    clean_df = df[[group_col, target]].dropna()

    is_numeric = target in numeric_cols
    is_categorical = target in categorical_cols or target in binary_cols
    
    test_name = "Unknown"
    p_value = np.nan
    statistic = np.nan
    assumption_notes = []
    epi_2x2_data = None  # the 2x2 measures, for a binary outcome only
    reasoning = []

    # Effect Size Calculation
    effect_size = np.nan
    effect_size_type = "N/A"
    sample_info = {}

    test_result = None
    estimate = np.nan
    estimate_ci_text = ""
    effect_ci_text = ""
    if is_numeric:
        # Welch-first policy; assumptions are diagnostics, not gates (audit S7/S8). Engine: biostats.inference
        groups = groups_from_frame(df, group_col, target)
        kind = LEGACY_NAMES.get(manual_test, "auto")
        reference = str(exposure_ref) if exposure_ref is not None and str(exposure_ref) in groups else None
        test_result = compare_groups(groups, kind=kind, reference=reference, n_boot=200)
        test_name = test_result.name
        statistic = test_result.statistic if test_result.statistic is not None else np.nan
        p_value = test_result.p_value if test_result.p_value is not None else np.nan
        primary = test_result.primary_effect
        if primary is not None:
            effect_size = primary.estimate
            effect_size_type = primary.name
            effect_ci_text = primary.ci.as_text(3) if primary.ci else ""
        if test_result.estimate is not None:
            estimate = test_result.estimate
            estimate_ci_text = test_result.estimate_ci.as_text(3) if test_result.estimate_ci else ""
        reasoning = list(test_result.warnings) + [f"{c.name}: {c.detail}" for c in test_result.assumptions]
        assumption_notes = reasoning
        sizes = test_result.n_per_group
        if len(sizes) == 2:
            n1, n2 = sizes
            sample_info = {"N1": n1, "N2": n2, "Ratio": n2 / n1 if n1 > 0 else 0}
        else:
            n_total = sum(sizes)
            k = len(sizes)
            sample_info = {"N_Total": n_total, "k_Groups": k, "N_Per_Group": n_total / k if k else 0}

    elif is_categorical:
        # Chi-Square or Fisher
        contingency_table = pd.crosstab(clean_df[group_col], clean_df[target])
        n_total = contingency_table.sum().sum()
        r, c = contingency_table.shape
        min_dim = min(r-1, c-1)
        sample_info = {"N_Total": n_total, "N_Cats": (r-1)*(c-1) + 1, "Min_Dim": min_dim}
        
        # Check expected frequencies for Fisher's Exact
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        
        if manual_test == "Auto-Detect":
            if (expected < 5).any():
                use_test = "Fisher's Exact"
                reasoning.append("Expected frequencies < 5 found. Using Fisher's Exact.")
            else:
                use_test = "Chi-Square"
                reasoning.append("All expected frequencies >= 5.")
        else:
            use_test = manual_test

        if contingency_table.shape == (2, 2):
            from biostats.contingency import TwoByTwo, analyze_twobytwo
            try:
                table_2x2 = TwoByTwo.from_crosstab(
                    clean_df, group_col, target, exposure_ref=exposure_ref, event_level=event_level
                )
                epi_2x2_data = analyze_twobytwo(table_2x2, design=design).legacy_dict()
            except Exception as e_epi:
                logger.debug(f"2x2 epidemiology calculation notice: {e_epi}")

        if use_test == "Fisher's Exact":
            test_name = "Fisher's Exact"
            if contingency_table.shape == (2, 2):
                statistic, p_value = stats.fisher_exact(contingency_table)
                effect_size = epi_2x2_data["odds_ratio"] if epi_2x2_data else statistic
                effect_size_type = "Odds Ratio"
            else:
                test_name = "Chi-Square (Fisher N/A for >2x2)"
                statistic, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                effect_size = compute_cramers_v(contingency_table)
                effect_size_type = "Cramer's V"
                reasoning.append("Fisher's Exact not available for >2x2 tables. Used Chi-Square.")

        else:
            test_name = "Chi-Square"
            statistic, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            if contingency_table.shape == (2, 2) and epi_2x2_data:
                effect_size = epi_2x2_data["odds_ratio"]
                effect_size_type = "Odds Ratio"
            else:
                effect_size = compute_cramers_v(contingency_table)
                effect_size_type = "Cramer's V"
            
        assumption_notes = reasoning

    ret_dict = {
        "Variable": target,
        "Test Used": test_name,
        "Statistic": statistic,
        "P-Value": p_value,
        "Effect Size": effect_size,
        "Effect Type": effect_size_type,
        "Effect CI": effect_ci_text,
        "Estimate": estimate,
        "Estimate CI": estimate_ci_text,
        "Assumptions": "; ".join(assumption_notes),
        "Sample Info": sample_info
    }
    if epi_2x2_data:
        ret_dict["epi_2x2"] = epi_2x2_data
    return ret_dict


def recommend_statistical_test(df, group_col, target_col, numeric_cols, categorical_cols, binary_cols):
    """Recommended test and reasoning: Welch-first for numeric outcomes (audit S7); chi-square family otherwise."""
    from biostats.inference import DISPLAY_NAMES

    if target_col not in numeric_cols:
        return "Chi-Square", ["Target is categorical: chi-square family (Fisher's exact test when expected counts are small)."]
    groups = groups_from_frame(df, group_col, target_col)
    groups = {k: v for k, v in groups.items() if len(v) > 0}
    if len(groups) < 2:
        return "Unknown", ["Fewer than two groups with data."]
    kind, reasons = _recommend_kind(groups)
    sizes = {k: len(v) for k, v in groups.items()}
    reasons = [f"Group sizes: {sizes}."] + reasons
    return DISPLAY_NAMES[kind], reasons


def compare_variable_result(df, group_col, target, manual_test="Auto-Detect", reference=None):
    """Full :class:`biostats.TestResult` for a numeric variable (used by the result card)."""
    groups = groups_from_frame(df, group_col, target)
    if len([v for v in groups.values() if len(v) > 0]) < 2:
        return None
    return compare_groups(groups, kind=LEGACY_NAMES.get(manual_test, "auto"), reference=reference, n_boot=300)


def perform_post_hoc(df, group_col, target_col, test_type, adjust_method="holm"):
    """Every pairwise comparison with raw and adjusted p-values; the method follows the omnibus test (audit S7)."""
    groups = groups_from_frame(df, group_col, target_col)
    try:
        return pairwise_comparisons(groups, method=default_method(str(test_type)), adjust_method=adjust_method)
    except Exception as exc:
        logger.warning("Post-hoc comparison failed: %s", exc)
        return None


def compute_cohens_d(group1, group2):
    """Compute Cohen's d for two independent groups."""
    from utils.epidemiology_utils import compute_cohens_d_with_ci
    return compute_cohens_d_with_ci(group1, group2).estimate

def compute_eta_squared(groups_data):
    """Compute Eta Squared for ANOVA."""
    from utils.epidemiology_utils import compute_eta_and_omega_squared
    return compute_eta_and_omega_squared(groups_data)["eta_squared"]


def compute_cramers_v(confusion_matrix):
    """Bias-corrected Cramer's V (Bergsma 2013) from the Pearson chi-square without Yates correction."""
    from biostats.effect_sizes import cramers_v

    return cramers_v(np.asarray(confusion_matrix, dtype=float), n_boot=0).estimate


def littles_mcar_test(df):
    """Little's MCAR test on EM estimates of the mean and covariance (audit S15); see :mod:`biostats.mcar`."""
    from biostats.mcar import littles_mcar_test as _littles

    return _littles(df)


def scan_missingness_dependencies(df, target_cols=None, predictor_cols=None, test_preference="Auto-Detect",
                                  progress_callback=None, correction="none", alpha=0.05):
    """Deprecated shim over :func:`biostats.missing.scan_missingness_dependencies`.

    Returns EVERY performed test (not only p < 0.05, audit S3) so that a multiplicity correction applied by
    the caller controls the real family. With ``correction`` set, adjusted p-values are included.
    """
    from biostats.missing import scan_missingness_dependencies as _scan

    result = _scan(df, target_cols, predictor_cols, test_preference=test_preference, correction=correction,
                   alpha=alpha, progress_callback=progress_callback)
    table = result.table.copy()
    table.attrs["n_tests"] = result.n_tests
    table.attrs["n_skipped"] = result.n_skipped
    if str(correction).lower() in ("none", ""):
        table = table.drop(columns=["P-Value (Adj)", "Significant"])
    return table
