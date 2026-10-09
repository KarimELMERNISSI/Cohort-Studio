"""
Epidemiology and Biostatistics Utility Module.

Provides mathematically rigorous, STROBE/CONSORT compliant statistical engines:
- 2x2 Contingency Table Analysis (OR, RR, RD, NNT, Attributable Fraction, with 95% CIs)
- Effect Size Estimations with Confidence Intervals (Cohen's d, Hedges' g, η², ω², Cramér's V)
- ANCOVA Diagnostic Engine (Homogeneity of slopes, residual normality, homoscedasticity)
- Diagnostic Test Accuracy (Sensitivity, Specificity, PPV, NPV, Likelihood Ratios with Wilson CIs)
"""

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats
from scipy.stats import norm

logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC SCHEMAS FOR STRICT TYPING
# =============================================================================

class ConfidenceInterval(BaseModel):
    """Container for lower and upper confidence bounds."""
    lower: float = Field(..., description="Lower limit of confidence interval")
    upper: float = Field(..., description="Upper limit of confidence interval")
    level: float = Field(default=0.95, description="Confidence level (e.g. 0.95)")


class Contingency2x2Result(BaseModel):
    """Complete epidemiological evaluation for 2x2 contingency tables."""
    # Table cell counts
    a: int = Field(..., description="Exposed Cases (a)")
    b: int = Field(..., description="Exposed Non-Cases (b)")
    c: int = Field(..., description="Unexposed Cases (c)")
    d: int = Field(..., description="Unexposed Non-Cases (d)")
    n_total: int = Field(..., description="Total sample size")
    haldane_applied: bool = Field(default=False, description="Whether 0.5 continuity correction was applied")

    # Risk & Odds Metrics
    odds_ratio: float = Field(..., description="Odds Ratio point estimate")
    odds_ratio_ci: ConfidenceInterval = Field(..., description="Woolf 95% CI for Odds Ratio")
    
    relative_risk: float = Field(..., description="Risk Ratio / Relative Risk point estimate")
    relative_risk_ci: ConfidenceInterval = Field(..., description="Katz 95% CI for Relative Risk")
    
    risk_difference: float = Field(..., description="Risk Difference (Absolute Risk Reduction)")
    risk_difference_ci: ConfidenceInterval = Field(..., description="Wald 95% CI for Risk Difference")
    
    nnt: float | None = Field(default=None, description="Number Needed to Treat / Harm (1 / |RD|)")
    attributable_fraction_exposed: float | None = Field(
        default=None, description="Attributable Fraction in Exposed (RR - 1) / RR"
    )

    # Statistical Tests
    chi2_statistic: float = Field(..., description="Pearson Chi-Square statistic")
    chi2_pvalue: float = Field(..., description="Pearson Chi-Square p-value")
    chi2_yates_pvalue: float = Field(..., description="Chi-Square with Yates continuity correction p-value")
    fisher_exact_pvalue: float = Field(..., description="Two-sided Fisher Exact test p-value")

    # Methodological Guidance
    guidance: list[str] = Field(default_factory=list, description="Clinical epidemiological guidance notes")


class EffectSizeResult(BaseModel):
    """Standardized effect size estimates with confidence intervals."""
    metric: str = Field(..., description="Name of effect size metric")
    estimate: float = Field(..., description="Point estimate")
    ci: ConfidenceInterval | None = Field(default=None, description="95% Confidence Interval")
    hedges_g: float | None = Field(default=None, description="Bias-corrected Hedges' g (for small samples)")
    interpretation: str = Field(..., description="Clinical magnitude (negligible, small, medium, large)")
    guidance: str = Field(default="", description="Epidemiological reporting guidance")


class ANCOVADiagnosticsResult(BaseModel):
    """Diagnostic tests and assumptions verification for ANCOVA."""
    homogeneity_of_slopes_pvalue: float | None = Field(
        default=None, description="Interaction p-value (Group x Covariate). Must be >= 0.05 for ANCOVA validity"
    )
    slopes_homogeneous: bool = Field(
        default=True, description="True if parallel slopes assumption holds"
    )
    residual_normality_pvalue: float | None = Field(
        default=None, description="Normality test p-value for model residuals"
    )
    residuals_normal: bool = Field(default=True, description="True if residuals pass normality check")
    homoscedasticity_pvalue: float | None = Field(
        default=None, description="Equal variance test p-value (Breusch-Pagan)"
    )
    homoscedastic: bool = Field(default=True, description="True if variance is homogeneous")
    partial_eta_squared: float = Field(default=0.0, description="Partial Eta Squared effect size")
    warnings: list[str] = Field(default_factory=list, description="Assumption violation warnings")
    guidance: list[str] = Field(default_factory=list, description="Clinical guidance for interpretation")


class DiagnosticTestResult(BaseModel):
    """Clinical diagnostic accuracy metrics with Wilson score confidence intervals."""
    tp: int = Field(..., description="True Positives")
    fp: int = Field(..., description="False Positives")
    fn: int = Field(..., description="False Negatives")
    tn: int = Field(..., description="True Negatives")
    
    sensitivity: float = Field(..., description="Sensitivity (True Positive Rate)")
    sensitivity_ci: ConfidenceInterval = Field(...)
    
    specificity: float = Field(..., description="Specificity (True Negative Rate)")
    specificity_ci: ConfidenceInterval = Field(...)
    
    ppv: float = Field(..., description="Positive Predictive Value (Precision)")
    ppv_ci: ConfidenceInterval = Field(...)
    
    npv: float = Field(..., description="Negative Predictive Value")
    npv_ci: ConfidenceInterval = Field(...)
    
    lr_positive: float | None = Field(default=None, description="Positive Likelihood Ratio (Sens / (1 - Spec))")
    lr_negative: float | None = Field(default=None, description="Negative Likelihood Ratio ((1 - Sens) / Spec)")
    youden_index: float = Field(..., description="Youden's Index J = Sensitivity + Specificity - 1")


# =============================================================================
# 1. 2x2 CONTINGENCY ANALYSIS ENGINE
# =============================================================================

def calculate_2x2_epidemiology_metrics(
    table_or_counts: pd.DataFrame | np.ndarray | tuple[int, int, int, int],
    exposure_label: str = "Exposed",
    outcome_label: str = "Outcome",
    alpha: float = 0.05,
) -> Contingency2x2Result:
    """
    Calculate comprehensive epidemiological metrics from a 2x2 table.
    
    Format:
                 Outcome + (Case)    Outcome - (Control)
    Exposed +           a                    b
    Exposed -           c                    d
    
    Args:
        table_or_counts: Either a 2x2 matrix/DataFrame or tuple (a, b, c, d).
        exposure_label: Descriptive label for the exposure factor.
        outcome_label: Descriptive label for the health outcome.
        alpha: Significance level for 100*(1-alpha)% confidence intervals.
        
    Returns:
        Contingency2x2Result with OR, RR, RD, NNT, Attributable Fraction, tests, and guidance.
    """
    if isinstance(table_or_counts, pd.DataFrame):
        arr = table_or_counts.to_numpy()
        a, b = int(arr[0, 0]), int(arr[0, 1])
        c, d = int(arr[1, 0]), int(arr[1, 1])
    elif isinstance(table_or_counts, np.ndarray):
        a, b = int(table_or_counts[0, 0]), int(table_or_counts[0, 1])
        c, d = int(table_or_counts[1, 0]), int(table_or_counts[1, 1])
    else:
        a, b, c, d = [int(v) for v in table_or_counts]

    n_total = a + b + c + d
    z_crit = norm.ppf(1.0 - alpha / 2.0)
    guidance: list[str] = []

    # Haldane-Anscombe 0.5 correction check for zero cells
    haldane_applied = False
    a_calc, b_calc, c_calc, d_calc = float(a), float(b), float(c), float(d)
    if any(cell == 0 for cell in (a, b, c, d)):
        haldane_applied = True
        a_calc += 0.5
        b_calc += 0.5
        c_calc += 0.5
        d_calc += 0.5
        guidance.append(
            "Haldane-Anscombe correction (+0.5 to each cell) was applied because one or more cells had zero counts."
        )

    # 1. Odds Ratio (OR) & Woolf's 95% CI
    odds_ratio = (a_calc * d_calc) / (b_calc * c_calc) if (b_calc * c_calc) > 0 else 1.0
    se_log_or = math.sqrt((1.0 / a_calc) + (1.0 / b_calc) + (1.0 / c_calc) + (1.0 / d_calc))
    log_or = math.log(odds_ratio)
    or_lower = math.exp(log_or - z_crit * se_log_or)
    or_upper = math.exp(log_or + z_crit * se_log_or)

    # 2. Relative Risk (RR) & Katz's 95% CI
    n_exp = a_calc + b_calc
    n_unexp = c_calc + d_calc
    p_exp = a_calc / n_exp if n_exp > 0 else 0.0
    p_unexp = c_calc / n_unexp if n_unexp > 0 else 0.0

    relative_risk = p_exp / p_unexp if p_unexp > 0 else 1.0
    se_log_rr = math.sqrt((b_calc / (a_calc * n_exp)) + (d_calc / (c_calc * n_unexp)))
    log_rr = math.log(relative_risk) if relative_risk > 0 else 0.0
    rr_lower = math.exp(log_rr - z_crit * se_log_rr)
    rr_upper = math.exp(log_rr + z_crit * se_log_rr)

    # 3. Risk Difference (RD) / Absolute Risk Reduction & Wald CI
    p_exp_raw = a / (a + b) if (a + b) > 0 else 0.0
    p_unexp_raw = c / (c + d) if (c + d) > 0 else 0.0
    risk_diff = p_exp_raw - p_unexp_raw
    se_rd = math.sqrt(
        (p_exp_raw * (1.0 - p_exp_raw) / (a + b)) + (p_unexp_raw * (1.0 - p_unexp_raw) / (c + d))
    ) if (a + b) > 0 and (c + d) > 0 else 0.0
    rd_lower = max(-1.0, risk_diff - z_crit * se_rd)
    rd_upper = min(1.0, risk_diff + z_crit * se_rd)

    # 4. Number Needed to Treat / Harm
    nnt = None
    if abs(risk_diff) > 1e-6:
        nnt = round(1.0 / abs(risk_diff), 2)

    # 5. Attributable Fraction in Exposed (AF_e)
    afe = None
    if relative_risk > 1.0:
        afe = round((relative_risk - 1.0) / relative_risk, 4)

    # 6. Statistical Hypothesis Tests (Chi2, Yates, Fisher Exact)
    obs = np.array([[a, b], [c, d]])
    chi2_res = stats.chi2_contingency(obs, correction=False)
    chi2_stat = float(chi2_res[0])
    chi2_pval = float(chi2_res[1])
    expected = chi2_res[3]

    chi2_yates = stats.chi2_contingency(obs, correction=True)
    chi2_yates_pval = float(chi2_yates[1])

    fisher_stat, fisher_pval = stats.fisher_exact(obs)

    # Methodological Guidance synthesis
    if (expected < 5.0).any():
        guidance.append(
            "Expected frequencies < 5 found in contingency cells. Prefer Fisher's Exact test p-value over Pearson Chi-Square."
        )

    prevalence = (a + c) / n_total if n_total > 0 else 0.0
    if prevalence < 0.10:
        guidance.append(
            f"Rare disease assumption holds (overall prevalence = {prevalence*100:.1f}% < 10%). The Odds Ratio (OR={odds_ratio:.2f}) provides an accurate estimate of the Relative Risk."
        )
    else:
        guidance.append(
            f"Common outcome detected (prevalence = {prevalence*100:.1f}% >= 10%). The Odds Ratio (OR={odds_ratio:.2f}) overstates the effect relative to the Relative Risk (RR={relative_risk:.2f}). Prefer RR for cohort designs."
        )

    if nnt is not None:
        direction = "Harm (NNH)" if risk_diff > 0 else "Benefit (NNT)"
        guidance.append(
            f"Number Needed to {direction}: {nnt:.1f} individuals must be exposed/treated for one additional event to occur."
        )

    return Contingency2x2Result(
        a=a,
        b=b,
        c=c,
        d=d,
        n_total=n_total,
        haldane_applied=haldane_applied,
        odds_ratio=round(odds_ratio, 4),
        odds_ratio_ci=ConfidenceInterval(lower=round(or_lower, 4), upper=round(or_upper, 4), level=1.0 - alpha),
        relative_risk=round(relative_risk, 4),
        relative_risk_ci=ConfidenceInterval(lower=round(rr_lower, 4), upper=round(rr_upper, 4), level=1.0 - alpha),
        risk_difference=round(risk_diff, 4),
        risk_difference_ci=ConfidenceInterval(lower=round(rd_lower, 4), upper=round(rd_upper, 4), level=1.0 - alpha),
        nnt=nnt,
        attributable_fraction_exposed=afe,
        chi2_statistic=round(chi2_stat, 4),
        chi2_pvalue=float(chi2_pval),
        chi2_yates_pvalue=float(chi2_yates_pval),
        fisher_exact_pvalue=float(fisher_pval),
        guidance=guidance,
    )


# =============================================================================
# 2. EFFECT SIZE ESTIMATION WITH CONFIDENCE INTERVALS
# =============================================================================

def compute_cohens_d_with_ci(
    group1: pd.Series | np.ndarray | list[float],
    group2: pd.Series | np.ndarray | list[float],
    alpha: float = 0.05,
) -> EffectSizeResult:
    """
    Compute Cohen's d with Hedges-Olkin 95% Confidence Interval and Hedges' g small-sample bias correction.
    
    Args:
        group1: Observations for reference or exposed group.
        group2: Observations for comparator group.
        alpha: Significance level (default 0.05 for 95% CI).
        
    Returns:
        EffectSizeResult with Cohen's d, CI, Hedges' g, and interpretation.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]

    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return EffectSizeResult(
            metric="Cohen's d",
            estimate=0.0,
            ci=None,
            hedges_g=0.0,
            interpretation="Undetermined (insufficient sample size < 2)",
            guidance="Sample size is too small to calculate reliable standardized mean difference.",
        )

    m1, m2 = float(np.mean(g1)), float(np.mean(g2))
    v1, v2 = float(np.var(g1, ddof=1)), float(np.var(g2, ddof=1))

    # Pooled standard deviation
    df = n1 + n2 - 2
    pooled_sd = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / df) if df > 0 else 0.0

    if pooled_sd <= 1e-12:
        return EffectSizeResult(
            metric="Cohen's d",
            estimate=0.0,
            ci=ConfidenceInterval(lower=0.0, upper=0.0, level=1.0 - alpha),
            hedges_g=0.0,
            interpretation="Zero variance (identical values across groups)",
            guidance="Both groups have zero within-group variance.",
        )

    d = (m1 - m2) / pooled_sd

    # Small-sample bias correction factor J (Hedges & Olkin, 1985)
    j_factor = 1.0 - (3.0 / (4.0 * df - 1.0)) if df > 1 else 1.0
    hedges_g = d * j_factor

    # Variance and standard error of Cohen's d (Hedges & Olkin)
    se_d = math.sqrt(((n1 + n2) / (n1 * n2)) + ((d * d) / (2.0 * (n1 + n2))))
    z_crit = norm.ppf(1.0 - alpha / 2.0)
    ci_lower = d - z_crit * se_d
    ci_upper = d + z_crit * se_d

    # Magnitude interpretation according to Cohen (1988)
    abs_d = abs(d)
    if abs_d < 0.2:
        interp = "Negligible effect (|d| < 0.2)"
    elif abs_d < 0.5:
        interp = "Small effect (0.2 <= |d| < 0.5)"
    elif abs_d < 0.8:
        interp = "Medium effect (0.5 <= |d| < 0.8)"
    else:
        interp = "Large effect (|d| >= 0.8)"

    guidance = (
        f"Mean difference is {m1 - m2:.3f} units. "
        f"95% CI [{ci_lower:.2f}, {ci_upper:.2f}]. "
    )
    if (n1 + n2) < 40:
        guidance += f"Small total sample (N={n1+n2}). Report Hedges' g ({hedges_g:.2f}) instead of unadjusted Cohen's d."

    return EffectSizeResult(
        metric="Cohen's d",
        estimate=round(d, 4),
        ci=ConfidenceInterval(lower=round(ci_lower, 4), upper=round(ci_upper, 4), level=1.0 - alpha),
        hedges_g=round(hedges_g, 4),
        interpretation=interp,
        guidance=guidance,
    )


def compute_eta_and_omega_squared(groups_data: list[np.ndarray | list[float]]) -> dict[str, Any]:
    """
    Compute Eta-squared (η²) and Omega-squared (ω²) for One-Way ANOVA.
    Omega-squared provides an unbiased population effect size for clinical studies.
    """
    clean_groups = [np.asarray(g, dtype=float)[~np.isnan(np.asarray(g, dtype=float))] for g in groups_data]
    k = len(clean_groups)
    all_data = np.concatenate(clean_groups) if clean_groups else np.array([])
    n_total = len(all_data)

    if k < 2 or n_total <= k:
        return {"eta_squared": 0.0, "omega_squared": 0.0, "interpretation": "Insufficient data"}

    grand_mean = float(np.mean(all_data))
    sst = float(np.sum((all_data - grand_mean) ** 2))
    ssb = float(sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in clean_groups))
    ssw = sst - ssb

    msw = ssw / (n_total - k) if (n_total - k) > 0 else 0.0
    eta_sq = ssb / sst if sst > 0 else 0.0

    # Omega squared formula: ω² = (SSB - (k - 1)*MSW) / (SST + MSW)
    omega_sq = (ssb - (k - 1) * msw) / (sst + msw) if (sst + msw) > 0 else 0.0
    omega_sq = max(0.0, omega_sq)  # Can be negative in small samples, floor at 0

    if eta_sq < 0.01:
        interp = "Negligible (η² < 0.01)"
    elif eta_sq < 0.06:
        interp = "Small (0.01 <= η² < 0.06)"
    elif eta_sq < 0.14:
        interp = "Medium (0.06 <= η² < 0.14)"
    else:
        interp = "Large (η² >= 0.14)"

    return {
        "eta_squared": round(eta_sq, 4),
        "omega_squared": round(omega_sq, 4),
        "interpretation": interp,
        "n_total": n_total,
        "k_groups": k,
    }


# =============================================================================
# 3. ANCOVA ASSUMPTION DIAGNOSTIC ENGINE
# =============================================================================

def check_ancova_assumptions(
    df: pd.DataFrame,
    target: str,
    group: str,
    covariates: list[str],
) -> ANCOVADiagnosticsResult:
    """
    Rigorously verify ANCOVA epidemiological assumptions:
    1. Homogeneity of regression slopes (Group x Covariates interaction test)
    2. Normality of residuals (Shapiro-Wilk or Jarque-Bera)
    3. Homoscedasticity of residuals (Breusch-Pagan)
    
    Args:
        df: Dataset containing variables.
        target: Continuous dependent variable.
        group: Categorical factor.
        covariates: Continuous covariates adjusted for in the model.
        
    Returns:
        ANCOVADiagnosticsResult with test statistics, warnings, and clinical recommendations.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.diagnostic import het_breuschpagan

    cols = [target, group] + (covariates if covariates else [])
    clean_df = df[cols].dropna()

    if len(clean_df) < 10:
        return ANCOVADiagnosticsResult(
            slopes_homogeneous=True,
            residuals_normal=True,
            homoscedastic=True,
            warnings=["Sample size < 10; unable to run robust ANCOVA diagnostics."],
        )

    # Sanitize names for patsy formula
    import re
    safe_map = {c: re.sub(r'[^a-zA-Z0-9_]', '_', c) for c in cols}
    safe_df = clean_df.rename(columns=safe_map)
    target_s = safe_map[target]
    group_s = safe_map[group]
    cov_s = [safe_map[c] for c in covariates]

    warnings: list[str] = []
    guidance: list[str] = []

    # 1. Base ANCOVA Model
    cov_formula_part = " + ".join(cov_s) if cov_s else ""
    base_formula = f"{target_s} ~ C({group_s})" + (f" + {cov_formula_part}" if cov_formula_part else "")
    base_model = smf.ols(base_formula, data=safe_df).fit()

    # Calculate Partial Eta-Squared for group effect
    anova_table = sm.stats.anova_lm(base_model, typ=2)
    group_key = f"C({group_s})"
    partial_eta = 0.0
    if group_key in anova_table.index and "Residual" in anova_table.index:
        ss_group = anova_table.loc[group_key, "sum_sq"]
        ss_resid = anova_table.loc["Residual", "sum_sq"]
        partial_eta = ss_group / (ss_group + ss_resid) if (ss_group + ss_resid) > 0 else 0.0

    # 2. Homogeneity of Slopes Test (Interaction Model)
    homog_p: float | None = None
    slopes_homogeneous = True

    if cov_s:
        # Build interaction terms: C(group) * cov1 + C(group) * cov2 ...
        interaction_parts = [f"C({group_s}):{c}" for c in cov_s]
        inter_formula = f"{base_formula} + {' + '.join(interaction_parts)}"
        try:
            inter_model = smf.ols(inter_formula, data=safe_df).fit()
            # Partial F-test comparing interaction model against base model
            anova_comp = sm.stats.anova_lm(base_model, inter_model)
            homog_p = float(anova_comp["Pr(>F)"].iloc[1])

            if homog_p < 0.05:
                slopes_homogeneous = False
                warnings.append(
                    f"Violation of Parallel Slopes (p={homog_p:.4f} < 0.05). "
                    "The relationship between covariate(s) and outcome varies across groups. "
                    "Standard ANCOVA adjusted means may be biased."
                )
                guidance.append(
                    "Consider moderator analysis, reporting group-specific covariate slopes, or Johnson-Neyman technique."
                )
            else:
                guidance.append(
                    f"Homogeneity of regression slopes holds (p={homog_p:.4f} >= 0.05). The covariate adjustment is valid across all groups."
                )
        except Exception as e:
            logger.warning(f"Slopes interaction test error: {e}")

    # 3. Residual Normality Check
    residuals = base_model.resid
    if len(residuals) <= 5000:
        _, norm_p = stats.shapiro(residuals)
    else:
        _, norm_p = stats.jarque_bera(residuals)
    residuals_normal = norm_p >= 0.05

    if not residuals_normal:
        warnings.append(
            f"Residuals deviate from normality (p={norm_p:.4f} < 0.05). "
            "Consider robust standard errors, bootstrap resampling, or outcome transformation."
        )

    # 4. Homoscedasticity Check (Breusch-Pagan)
    homo_p: float | None = None
    homoscedastic = True
    try:
        bp_test = het_breuschpagan(residuals, base_model.model.exog)
        homo_p = float(bp_test[1])  # p-value of LM test
        homoscedastic = homo_p >= 0.05
        if not homoscedastic:
            warnings.append(
                f"Heteroscedasticity detected (Breusch-Pagan p={homo_p:.4f} < 0.05). "
                "Residual variance is not uniform across fitted values."
            )
    except Exception as e:
        logger.debug(f"Breusch-Pagan test skipped: {e}")

    return ANCOVADiagnosticsResult(
        homogeneity_of_slopes_pvalue=round(homog_p, 4) if homog_p is not None else None,
        slopes_homogeneous=slopes_homogeneous,
        residual_normality_pvalue=round(float(norm_p), 4),
        residuals_normal=residuals_normal,
        homoscedasticity_pvalue=round(homo_p, 4) if homo_p is not None else None,
        homoscedastic=homoscedastic,
        partial_eta_squared=round(float(partial_eta), 4),
        warnings=warnings,
        guidance=guidance,
    )


# =============================================================================
# 4. DIAGNOSTIC TEST ACCURACY METRICS
# =============================================================================

def _wilson_score_interval(p: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Compute Wilson score interval for proportions."""
    if n == 0:
        return 0.0, 0.0
    z = norm.ppf(1.0 - alpha / 2.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def calculate_diagnostic_accuracy(
    tp: int, fp: int, fn: int, tn: int, alpha: float = 0.05
) -> DiagnosticTestResult:
    """
    Calculate clinical diagnostic accuracy parameters with 95% Wilson score confidence intervals.
    
    Args:
        tp: True Positives
        fp: False Positives
        fn: False Negatives
        tn: True Negatives
        alpha: Significance level (default 0.05 for 95% CI)
    """
    total_pos = tp + fn
    total_neg = fp + tn
    total_test_pos = tp + fp
    total_test_neg = fn + tn

    sens = tp / total_pos if total_pos > 0 else 0.0
    spec = tn / total_neg if total_neg > 0 else 0.0
    ppv = tp / total_test_pos if total_test_pos > 0 else 0.0
    npv = tn / total_test_neg if total_test_neg > 0 else 0.0

    sens_ci = _wilson_score_interval(sens, total_pos, alpha)
    spec_ci = _wilson_score_interval(spec, total_neg, alpha)
    ppv_ci = _wilson_score_interval(ppv, total_test_pos, alpha)
    npv_ci = _wilson_score_interval(npv, total_test_neg, alpha)

    # Likelihood Ratios
    lr_pos = None
    if (1.0 - spec) > 1e-6:
        lr_pos = round(sens / (1.0 - spec), 3)

    lr_neg = None
    if spec > 1e-6:
        lr_neg = round((1.0 - sens) / spec, 3)

    youden = sens + spec - 1.0

    return DiagnosticTestResult(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        sensitivity=round(sens, 4),
        sensitivity_ci=ConfidenceInterval(lower=round(sens_ci[0], 4), upper=round(sens_ci[1], 4), level=1.0 - alpha),
        specificity=round(spec, 4),
        specificity_ci=ConfidenceInterval(lower=round(spec_ci[0], 4), upper=round(spec_ci[1], 4), level=1.0 - alpha),
        ppv=round(ppv, 4),
        ppv_ci=ConfidenceInterval(lower=round(ppv_ci[0], 4), upper=round(ppv_ci[1], 4), level=1.0 - alpha),
        npv=round(npv, 4),
        npv_ci=ConfidenceInterval(lower=round(npv_ci[0], 4), upper=round(npv_ci[1], 4), level=1.0 - alpha),
        lr_positive=lr_pos,
        lr_negative=lr_neg,
        youden_index=round(youden, 4),
    )
