import numpy as np
import pandas as pd
import statsmodels.stats.multitest as smt
import streamlit as st
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def render_analysis_configuration(df, numeric_cols, categorical_cols, binary_cols, group_col_options=None, default_group_col="None", target_col_options=None, key_prefix=""):
    """
    Renders the Analysis Configuration expander.
    Returns a dictionary with the selected configuration.
    """
    with st.expander("📊 Analysis Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Select Grouping Variable
            if group_col_options is None:
                potential_groupers = [col for col in categorical_cols + binary_cols + numeric_cols 
                                      if df[col].nunique() < 20]
                group_col_options = ["None"] + sorted(potential_groupers)
            
            group_col = st.selectbox(
                "Select Grouping Variable (Factor)", 
                options=group_col_options,
                index=group_col_options.index(default_group_col) if default_group_col in group_col_options else 0,
                help="Select a categorical variable to define groups (e.g., Treatment vs Control).",
                key=f"{key_prefix}group_col"
            )
            
            # Group Filtering
            selected_groups = []
            if group_col != "None" and group_col in df.columns:
                unique_groups = sorted(df[group_col].dropna().unique())
                selected_groups = st.multiselect(
                    "Groups to Include",
                    options=unique_groups,
                    default=unique_groups,
                    help="Deselect groups with small sample sizes (e.g., < 5) to exclude them from analysis.",
                    key=f"{key_prefix}selected_groups"
                )
            elif group_col != "None":
                 # Handle case where group_col is not in df (e.g. "Missingness Status" created on the fly)
                 # In that case, we might not be able to filter groups easily here without the derived df.
                 # For MAR check, we usually just have 2 groups (Missing, Observed).
                 pass

        with col2:
            # Select Target Variables
            if target_col_options is None:
                target_col_options = [c for c in df.columns if c != group_col]
            
            # Default selection logic
            default_targets = []
            if numeric_cols:
                default_targets = [c for c in numeric_cols if c != group_col][:5]
            
            target_cols = st.multiselect(
                "Select Variables to Test",
                options=target_col_options,
                default=default_targets,
                help="Select variables to compare across groups.",
                key=f"{key_prefix}target_cols"
            )
            
        # Global Test Preference
        col3, col4 = st.columns(2)
        with col3:
            test_preference = st.selectbox(
                "Test Selection Strategy",
                ["Auto-Detect (Recommended)", "Force Parametric (T-test/ANOVA)", "Force Non-Parametric (Mann-Whitney/Kruskal)"],
                help="Auto-Detect checks normality and variance assumptions. You can force specific test families if needed.",
                key=f"{key_prefix}test_pref"
            )
        with col4:
            correction_method = st.selectbox(
                "Multiple Testing Correction",
                ["None", "Bonferroni", "Benjamini-Hochberg (FDR)"],
                help="Adjust P-values to control for Type I errors when testing multiple variables.",
                key=f"{key_prefix}correction"
            )
            
    return {
        "group_col": group_col,
        "selected_groups": selected_groups,
        "target_cols": target_cols,
        "test_preference": test_preference,
        "correction_method": correction_method
    }

def analyze_variable(df, group_col, target, numeric_cols, categorical_cols, binary_cols, manual_test="Auto-Detect"):
    """
    Analyzes a single variable against the group column, selecting the appropriate test.
    """
    # Prepare data
    clean_df = df[[group_col, target]].dropna()
    groups_data = [clean_df[clean_df[group_col] == g][target] for g in clean_df[group_col].unique()]
    
    is_numeric = target in numeric_cols
    is_categorical = target in categorical_cols or target in binary_cols
    
    test_name = "Unknown"
    p_value = np.nan
    statistic = np.nan
    assumption_notes = []
    reasoning = []

    # Effect Size Calculation
    effect_size = np.nan
    effect_size_type = "N/A"
    sample_info = {}

    if is_numeric:
        # Use Recommendation Engine
        rec_test, rec_reasoning = recommend_statistical_test(df, group_col, target, numeric_cols, categorical_cols, binary_cols)
        reasoning = rec_reasoning
        
        # Determine Test to Run
        if manual_test == "Auto-Detect":
            use_test = rec_test
        else:
            use_test = manual_test
            reasoning.append(f"User manually selected {manual_test}.")

        # Collect Assumption Notes for Display
        assumption_notes = reasoning

        if len(groups_data) == 2:
            n1, n2 = len(groups_data[0]), len(groups_data[1])
            sample_info = {"N1": n1, "N2": n2, "Ratio": n2/n1 if n1 > 0 else 0}
            
            # Execution
            from utils.epidemiology_utils import compute_cohens_d_with_ci
            d_res = compute_cohens_d_with_ci(groups_data[0], groups_data[1])
            
            if use_test == "Student's t-test":
                test_name = "Student's t-test"
                statistic, p_value = stats.ttest_ind(*groups_data)
                effect_size = d_res.estimate
                effect_size_type = "Cohen's d"
            elif use_test == "Welch's t-test":
                test_name = "Welch's t-test"
                statistic, p_value = stats.ttest_ind(*groups_data, equal_var=False)
                effect_size = d_res.estimate
                effect_size_type = "Cohen's d"
            elif use_test == "Mann-Whitney U":
                test_name = "Mann-Whitney U"
                statistic, p_value = stats.mannwhitneyu(*groups_data)
                # Rank-Biserial Correlation r = 1 - (2U)/(n1*n2)
                effect_size = 1 - (2 * statistic) / (n1 * n2) if (n1 * n2) > 0 else 0
                effect_size_type = "Rank-Biserial r"

        else: # > 2 groups
            n_total = sum(len(g) for g in groups_data)
            k = len(groups_data)
            sample_info = {"N_Total": n_total, "k_Groups": k, "N_Per_Group": n_total/k}

            from utils.epidemiology_utils import compute_eta_and_omega_squared
            eta_omega = compute_eta_and_omega_squared(groups_data)

            if use_test == "ANOVA":
                test_name = "ANOVA"
                statistic, p_value = stats.f_oneway(*groups_data)
                effect_size = eta_omega["eta_squared"]
                effect_size_type = "Eta Squared"
            elif use_test == "Kruskal-Wallis":
                test_name = "Kruskal-Wallis"
                statistic, p_value = stats.kruskal(*groups_data)
                # Eta-Squared H (η²H) for Kruskal-Wallis (Tomczak & Tomczak, 2014)
                effect_size = (statistic - k + 1) / (n_total - k) if (n_total - k) > 0 else 0
                effect_size_type = "η²H"

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

        epi_2x2_data = None
        if contingency_table.shape == (2, 2):
            from utils.epidemiology_utils import calculate_2x2_epidemiology_metrics
            try:
                epi_res = calculate_2x2_epidemiology_metrics(contingency_table)
                epi_2x2_data = epi_res.model_dump()
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
        "Assumptions": "; ".join(assumption_notes),
        "Sample Info": sample_info
    }
    if 'epi_2x2_data' in locals() and epi_2x2_data:
        ret_dict["epi_2x2"] = epi_2x2_data
    return ret_dict


def recommend_statistical_test(df, group_col, target_col, numeric_cols, categorical_cols, binary_cols):
    # Returns a recommended test and the reasoning based on data properties.
    reasoning = []
    
    # 1. Check Data Types
    is_numeric = target_col in numeric_cols
    is_binary = target_col in binary_cols
    groups = df[group_col].dropna().unique()
    n_groups = len(groups)
    
    if not is_numeric and not is_binary:
        return "Chi-Square", ["Target is categorical with >2 levels."]
        
    # 2. Check Sample Size & Balance
    group_sizes = df[group_col].value_counts()
    min_group_size = group_sizes.min()
    reasoning.append(f"Smallest group size is {min_group_size}.")
    
    check_normality = True
    if min_group_size < 30:
        reasoning.append("Small sample size detected (<30). Central Limit Theorem may not apply.")
    else:
        reasoning.append("Large sample size (>30). Parametric tests are generally robust (CLT).")
        # We still check normality but might be more lenient or prioritize variance

    # 3. Normality Check (Only if numeric)
    if is_numeric:
        clean_df = df[[group_col, target_col]].dropna()
        groups_data = [clean_df[clean_df[group_col] == g][target_col] for g in groups]
        
        # Shapiro-Wilk or D'Agostino
        normality_p_values = []
        for g_data in groups_data:
            if len(g_data) >= 3:
                try:
                    if len(g_data) > 5000:
                        stat, p = stats.normaltest(g_data)
                    else:
                        stat, p = stats.shapiro(g_data)
                    normality_p_values.append(p)
                except:
                    normality_p_values.append(0)
        
        is_normal = all(p > 0.05 for p in normality_p_values)
        
        if is_normal:
            reasoning.append("Data appears Normally Distributed (Shapiro-Wilk p > 0.05).")
        else:
            reasoning.append("Data deviates from Normality (Shapiro-Wilk p < 0.05).")

        # Homogeneity of Variance
        try:
            l_stat, l_p = stats.levene(*groups_data)
            equal_var = l_p > 0.05
            if equal_var:
                reasoning.append("Variances are equal (Levene p > 0.05).")
            else:
                reasoning.append("Variances are unequal (Levene p < 0.05).")
        except:
            equal_var = False
            reasoning.append("Could not test for equal variances.")

        if n_groups == 2:
            if is_normal:
                if equal_var:
                    return "Student's t-test", reasoning
                else:
                    return "Welch's t-test", reasoning
            else:
                return "Mann-Whitney U", reasoning
        elif n_groups > 2:
            if is_normal and equal_var:
                return "ANOVA", reasoning
            else:
                return "Kruskal-Wallis", reasoning

    return "Unknown", ["Could not determine appropriate test."]

def perform_post_hoc(df, group_col, target_col, test_type):
    # Runs post-hoc analysis based on the global test type.
    clean_df = df[[group_col, target_col]].dropna()
    
    if test_type == "ANOVA":
        # Tukey HSD
        try:
            tukey = pairwise_tukeyhsd(endog=clean_df[target_col], groups=clean_df[group_col], alpha=0.05)
            # Convert to DataFrame for display
            results = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
            return results[results['reject'] == True] # Return only significant pairs
        except Exception as e:
            st.warning(f"Post-hoc failed: {e}")
            return None
        
    elif test_type == "Kruskal-Wallis":
        # Fallback to pairwise Mann-Whitney with Bonferroni correction
        import itertools
        groups = clean_df[group_col].unique()
        pairs = []
        p_vals = []
        
        for g1, g2 in itertools.combinations(groups, 2):
            data1 = clean_df[clean_df[group_col] == g1][target_col]
            data2 = clean_df[clean_df[group_col] == g2][target_col]
            try:
                _, p = stats.mannwhitneyu(data1, data2)
                pairs.append(f"{g1} vs {g2}")
                p_vals.append(p)
            except:
                pass
            
        if not p_vals:
            return None

        # Apply correction
        _, p_adj, _, _ = smt.multipletests(p_vals, method='bonferroni')
        
        results = pd.DataFrame({'Pair': pairs, 'P-adj': p_adj})
        return results[results['P-adj'] < 0.05]
        
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
    # Compute Cramer's V for categorical association.
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1)) if min(k-1, r-1) > 0 else 0

def littles_mcar_test(df):
    # Performs Littles MCAR (Missing Completely At Random) test.
    # H0: Data is MCAR
    # H1: Data is not MCAR (MAR or MNAR)
    # Returns: dict with 'statistic', 'df', 'p_value', 'result'
    # Ensure only numeric data
    df = df.select_dtypes(include=[np.number])
    
    if df.empty:
        return {"error": "No numeric data available for MCAR test."}
    
    n, k = df.shape
    
    # 1. Estimate Mean and Covariance using EM Algorithm (Simplified)
    # For Little's test, we need the MLE of mean and covariance under H0 (MCAR).
    # We will use a basic iterative approach or available estimates.
    
    # Initial estimates (ignoring missing values)
    mu_mle = df.mean().values
    cov_mle = df.cov().values
    
    # 2. Compute Statistic
    d2_total = 0
    df_total = 0
    
    # Group by pattern
    mask = df.isnull()
    patterns = mask.apply(lambda x: tuple(x), axis=1)
    grouped = df.groupby(patterns)
    
    for pattern, group in grouped:
        # pattern is tuple of bools (True=Missing, False=Observed)
        is_missing = np.array(pattern)
        is_observed = ~is_missing
        
        # Number of observed variables in this pattern
        p_j = np.sum(is_observed)
        
        if p_j == 0:
            continue
            
        # Number of samples in this pattern
        m_j = len(group)
        
        # Observed values in this pattern
        obs_indices = np.where(is_observed)[0]
        
        # Observed Mean vector for this pattern (d_j)
        y_bar_j = group.iloc[:, obs_indices].mean().values
        
        # Expected Mean vector (mu_j) - subset of global MLE
        mu_j = mu_mle[obs_indices]
        
        # Difference
        diff = y_bar_j - mu_j
        
        # Covariance for this pattern (Sigma_j) - subset of global MLE
        sigma_jj = cov_mle[np.ix_(obs_indices, obs_indices)]
        
        # Calculate contribution: m_j * (diff.T * inv(sigma_jj) * diff)
        try:
            # Use pseudo-inverse for stability
            inv_sigma_jj = np.linalg.pinv(sigma_jj)
            d2_j = m_j * (diff.T @ inv_sigma_jj @ diff)
            
            d2_total += d2_j
            df_total += p_j
        except np.linalg.LinAlgError:
            continue
            
    # Degrees of Freedom: sum(p_j) - k
    final_df = df_total - k
    
    # P-Value
    # If final_df <= 0, something is wrong (e.g. only complete cases or single pattern)
    if final_df <= 0:
        return {"error": "Insufficient degrees of freedom for MCAR test (df <= 0)."}
        
    p_value = 1 - stats.chi2.cdf(d2_total, final_df)
    
    return {
        "statistic": d2_total,
        "df": final_df,
        "p_value": p_value,
        "result": "Likely Not MCAR" if p_value < 0.05 else "Likely MCAR"
    }



def scan_missingness_dependencies(df, target_cols=None, predictor_cols=None, test_preference="Auto-Detect", progress_callback=None):
    """
    Systematically tests for dependencies between missingness of targets and values/missingness of predictors.
    
    Args:
        test_preference: "Auto-Detect", "Force Parametric", "Force Non-Parametric"
    
    Returns:
    - pd.DataFrame with significant results.
    """
    results = []
    
    # Defaults
    if target_cols is None:
        # Targets must have missing values
        target_cols = [c for c in df.columns if df[c].isnull().any()]
    
    if predictor_cols is None:
        predictor_cols = df.columns.tolist()
        
    # Pre-calculate missingness masks to speed up
    missing_masks = {col: df[col].isnull() for col in df.columns}
    
    total_tests = len(target_cols) * len(predictor_cols)
    processed = 0
    
    # Identify column types
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = set(df.select_dtypes(include=['object', 'category']).columns)
    binary_cols = set([c for c in df.columns if df[c].nunique() == 2])
    
    for target in target_cols:
        target_missing = missing_masks[target]
        
        # If no missing or all missing, skip
        if target_missing.sum() == 0 or target_missing.sum() == len(df):
            processed += len(predictor_cols)
            if progress_callback: progress_callback(processed / total_tests)
            continue
            
        # Create binary target for analysis
        # We want to see if Predictor differs between Target=Missing vs Target=Observed
        
        for pred in predictor_cols:
            if pred == target:
                processed += 1
                continue
                
            # --- Test 1: Value Impact (Predictor Value -> Target Missingness) ---
            # Does the value of 'pred' differ when 'target' is missing?
            # Only consider rows where 'pred' is observed
            
            valid_rows = ~missing_masks[pred]
            
            if valid_rows.sum() < 10: # Skip if too few observed values
                processed += 1
                continue
                
            y_vals = df.loc[valid_rows, pred]
            groups = target_missing.loc[valid_rows] # True=Missing, False=Observed
            
            # Check group sizes
            n_missing = groups.sum()
            n_observed = (~groups).sum()
            
            if n_missing < 5 or n_observed < 5:
                # Too few samples in one group to test
                pass
            else:
                try:
                    p_val = np.nan
                    test_name = "Unknown"
                    
                    if pred in numeric_cols:
                        g1 = y_vals[groups] # Target Missing
                        g2 = y_vals[~groups] # Target Observed
                        
                        use_parametric = False
                        
                        if test_preference == "Force Parametric":
                            use_parametric = True
                        elif test_preference == "Force Non-Parametric":
                            use_parametric = False
                        else: # Auto-Detect
                            # Check Normality (Shapiro/Normaltest)
                            # Only check if N is reasonable, otherwise assume non-normal for small N, normal for huge N?
                            # Actually, for "Auto", we should be robust.
                            is_normal = True
                            for g in [g1, g2]:
                                if len(g) < 3: is_normal = False; break
                                try:
                                    if len(g) > 5000:
                                        _, p = stats.normaltest(g)
                                    else:
                                        _, p = stats.shapiro(g)
                                    if p < 0.05:
                                        is_normal = False
                                        break
                                except:
                                    is_normal = False
                                    break
                            
                            # Check Homogeneity of Variance (Levene)
                            equal_var = True
                            try:
                                _, p = stats.levene(g1, g2)
                                if p < 0.05: equal_var = False
                            except:
                                pass
                                
                            if is_normal:
                                use_parametric = True
                                # We will handle Welch vs Student below
                            else:
                                use_parametric = False
                        
                        if use_parametric:
                            # Check variance for Welch vs Student if not already checked
                            if test_preference == "Force Parametric":
                                try:
                                    _, p = stats.levene(g1, g2)
                                    equal_var = p > 0.05
                                except:
                                    equal_var = False
                            
                            if equal_var:
                                stat, p_val = stats.ttest_ind(g1, g2, nan_policy='omit')
                                test_name = "Student's t-test"
                            else:
                                stat, p_val = stats.ttest_ind(g1, g2, equal_var=False, nan_policy='omit')
                                test_name = "Welch's t-test"
                        else:
                            stat, p_val = stats.mannwhitneyu(g1, g2)
                            test_name = "Mann-Whitney U"
                            
                    elif pred in categorical_cols or pred in binary_cols:
                        # Chi-Square
                        contingency = pd.crosstab(y_vals, groups)
                        if contingency.size > 0:
                            # Check expected freqs for Fisher?
                            chi2, p, dof, expected = stats.chi2_contingency(contingency)
                            
                            if (expected < 5).any() and contingency.shape == (2,2):
                                # Fisher's Exact
                                _, p_val = stats.fisher_exact(contingency)
                                test_name = "Fisher's Exact"
                            else:
                                p_val = p
                                test_name = "Chi-Square"
                            
                    if not np.isnan(p_val) and p_val < 0.05:
                        results.append({
                            "Target Variable": target,
                            "Predictor Variable": pred,
                            "Dependency Type": "Value Impact",
                            "Test Used": test_name,
                            "P-Value": p_val,
                            "Conclusion": f"{pred} value affects missingness of {target}"
                        })
                except Exception:
                    pass

            # --- Test 2: Pattern Impact (Predictor Missingness -> Target Missingness) ---
            # Is missingness of 'pred' associated with missingness of 'target'?
            # Chi-Square on two boolean vectors
            
            pred_missing = missing_masks[pred]
            
            # If pred has no missing values, skip
            if pred_missing.sum() == 0 or pred_missing.sum() == len(df):
                pass
            else:
                try:
                    contingency = pd.crosstab(target_missing, pred_missing)
                    if contingency.shape == (2, 2):
                        # Always Chi-Square or Fisher for 2x2 boolean
                        chi2, p, dof, expected = stats.chi2_contingency(contingency)
                        
                        if (expected < 5).any():
                             _, p_val = stats.fisher_exact(contingency)
                             test_name = "Fisher's Exact"
                        else:
                             p_val = p
                             test_name = "Chi-Square"
                        
                        if p_val < 0.05:
                             results.append({
                                "Target Variable": target,
                                "Predictor Variable": pred,
                                "Dependency Type": "Missingness Correlation",
                                "Test Used": test_name,
                                "P-Value": p_val,
                                "Conclusion": f"Missingness of {pred} is linked to missingness of {target}"
                            })
                except Exception:
                    pass
            
            processed += 1
            if progress_callback and processed % 10 == 0:
                progress_callback(processed / total_tests)
                
    if progress_callback: progress_callback(1.0)
    
    if not results:
        return pd.DataFrame(columns=["Target Variable", "Predictor Variable", "Dependency Type", "Test Used", "P-Value", "Conclusion"])
        
    return pd.DataFrame(results).sort_values("P-Value")
