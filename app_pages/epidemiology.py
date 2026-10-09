import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import t as t_dist
import plotly.express as px
import plotly.graph_objects as go
from utils.data_analyzer import DataAnalyzer
import utils.visualization_utils as vu
import utils.analysis_utils as au
from io import BytesIO
import statsmodels.stats.multitest as smt
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower, GofChisquarePower
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import re
import statsmodels.api as sm
import itertools

def to_excel(df):
    """
    Convert a DataFrame to an Excel file in binary format.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to convert.

    Returns:
    --------
    bytes
        The Excel file content as bytes.
    """
    output = BytesIO()
    with pd.ExcelWriter(output) as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
    return output.getvalue()

def app():
    """
    Main application function for the Epidemiology Analysis page.

    Handles:
    1. Univariate Group Comparisons (T-tests, ANOVA, Chi-Square, etc.).
    2. Multivariate Analysis (ANCOVA) to control for confounders.
    3. Power Analysis & Sample Size Calculation.
    4. Z-Score Calculation (Reference Standardization).
    5. Educational Content display.
    """
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.warning("Please upload a dataset in the Main View first.")
        return

    df = st.session_state['data']
    analyzer = DataAnalyzer(df)

    # Sidebar for Analysis Mode
    with st.sidebar:
        st.header("Analysis Configuration")
        analysis_mode = st.radio(
            "Select Analysis Type",
            ["Univariate (Group Comparison)", "Multivariate (ANCOVA/Regression)"],
            help="Choose 'Univariate' to compare groups one variable at a time. Choose 'Multivariate' to adjust for confounding variables (ANCOVA)."
        )
        
        st.divider()
        st.subheader("Data Preprocessing")
        transformation = st.selectbox(
            "Apply Transformation",
            ["None (Raw Data)", "Log Transformation (Log1p)", "Z-Score Standardization", "Min-Max Normalization"],
            help="Transform numeric variables before analysis. \n- Log: Reduces skewness (good for normality).\n- Z-Score: Centers data (Mean=0, SD=1).\n- Min-Max: Scales to [0, 1]."
        )

    # Apply Transformation
    df_analysis = df.copy()
    if transformation != "None (Raw Data)":
        numeric_cols = analyzer.numeric_cols
        if numeric_cols:
            if transformation == "Log Transformation (Log1p)":
                # Check for negative values
                cols_to_transform = []
                skipped_cols = []
                for col in numeric_cols:
                    if pd.api.types.is_numeric_dtype(df_analysis[col]):
                        if df_analysis[col].min() >= 0:
                            df_analysis[col] = np.log1p(df_analysis[col])
                            cols_to_transform.append(col)
                        else:
                            skipped_cols.append(col)
                
                if skipped_cols:
                    st.warning(f"Log transformation skipped for columns with negative values: {', '.join(skipped_cols)}")
                    
            elif transformation == "Z-Score Standardization":
                for col in numeric_cols:
                    if pd.api.types.is_numeric_dtype(df_analysis[col]) and df_analysis[col].std() > 0:
                        df_analysis[col] = (df_analysis[col] - df_analysis[col].mean()) / df_analysis[col].std()
                        
            elif transformation == "Min-Max Normalization":
                for col in numeric_cols:
                    if pd.api.types.is_numeric_dtype(df_analysis[col]):
                        min_val = df_analysis[col].min()
                        max_val = df_analysis[col].max()
                        if max_val > min_val:
                            df_analysis[col] = (df_analysis[col] - min_val) / (max_val - min_val)
        
        st.info(f"**{transformation}** applied to numeric variables for the analysis below.")

    tabs = st.tabs(["Analysis", "Power & Sample Size", "Z-Score & Reference", "Test Guide & Education"])

    with tabs[0]:
        if analysis_mode == "Univariate (Group Comparison)":
            run_univariate_analysis(df_analysis, analyzer)
        else:
            run_multivariate_analysis(df_analysis, analyzer)

    with tabs[1]:
        run_power_analysis(df, analyzer)

    with tabs[2]:
        # Use raw data for Z-Score tab as it has its own standardization logic
        run_zscore_analysis(df, analyzer)

    with tabs[3]:
        show_educational_content()

def run_univariate_analysis(df, analyzer):
    st.header("Univariate Group Comparison")
    st.markdown("Compare variables between groups using statistical tests (T-test, ANOVA, Chi-Square, etc.).")
    
    # --- Configuration ---
    config = au.render_analysis_configuration(
        df, 
        analyzer.numeric_cols, 
        analyzer.categorical_cols, 
        analyzer.binary_cols
    )
    group_col = config["group_col"]
    selected_groups = config["selected_groups"]
    target_cols = config["target_cols"]
    test_preference = config["test_preference"]
    correction_method = config["correction_method"]

    if group_col == "None" or not target_cols:
        st.info("Please select a grouping variable and at least one target variable to proceed.")
        return
        
    if len(selected_groups) < 2:
        st.warning("Please select at least 2 groups to perform a comparison.")
        return

    # Filter Dataframe based on selected groups
    df_analysis_filtered = df[df[group_col].isin(selected_groups)].copy()

    # Prepare results container
    results = []
    
    # Get groups
    groups = df_analysis_filtered[group_col].dropna().unique()
    n_groups = len(groups)
    
    if n_groups < 2:
        st.error("Grouping variable must have at least 2 levels in the filtered data.")
        return

    st.write(f"### Comparing groups defined by **{group_col}**: {', '.join(map(str, groups))}")

    # Progress bar for many variables
    progress_bar = st.progress(0)
    
    for i, target in enumerate(target_cols):
        # Determine manual test override based on preference
        manual_override = "Auto-Detect"
        if test_preference == "Force Parametric (T-test/ANOVA)":
            manual_override = "Student's t-test" if n_groups == 2 else "ANOVA"
        elif test_preference == "Force Non-Parametric (Mann-Whitney/Kruskal)":
            manual_override = "Mann-Whitney U" if n_groups == 2 else "Kruskal-Wallis"
            
        res = au.analyze_variable(
            df_analysis_filtered, 
            group_col, 
            target, 
            analyzer.numeric_cols,
            analyzer.categorical_cols,
            analyzer.binary_cols,
            manual_test=manual_override
        )
        results.append(res)
        progress_bar.progress((i + 1) / len(target_cols))
    
    progress_bar.empty()

    # Display Results Table
    results_df = pd.DataFrame(results)
    
    # Store results in session state for Power Analysis
    st.session_state['epidemiology_results'] = results
    st.session_state['epidemiology_group_col'] = group_col
    
    # Apply Multiple Testing Correction
    if correction_method != "None" and not results_df.empty:
        p_values = results_df["P-Value"].fillna(1.0).values
        if correction_method == "Bonferroni":
            reject, pvals_corrected, _, _ = smt.multipletests(p_values, method='bonferroni')
        elif correction_method == "Benjamini-Hochberg (FDR)":
            reject, pvals_corrected, _, _ = smt.multipletests(p_values, method='fdr_bh')
        
        results_df["P-Value (Adj)"] = pvals_corrected
        results_df["Significant (Adj)"] = ["Yes" if p < 0.05 else "No" for p in pvals_corrected]
        
        # Reorder columns to show adjusted p-value next to original
        cols = list(results_df.columns)
        # Move Adj columns
        cols.remove("P-Value (Adj)")
        cols.remove("Significant (Adj)")
        p_idx = cols.index("P-Value")
        cols.insert(p_idx + 1, "P-Value (Adj)")
        cols.insert(p_idx + 2, "Significant (Adj)")
        results_df = results_df[cols]

    # Formatting for display
    st.subheader("Summary Results")
    
    # Improve readability of columns for display and download
    if not results_df.empty:
        # Format Sample Info
        def fmt_sample(x):
            if isinstance(x, dict):
                parts = []
                if 'N1' in x: parts.append(f"N1={x['N1']}, N2={x['N2']}")
                if 'Ratio' in x: parts.append(f"Ratio={x['Ratio']:.2f}")
                if 'k_Groups' in x: parts.append(f"k={x['k_Groups']}")
                if 'N_Total' in x: parts.append(f"N={x['N_Total']}")
                if 'N_Cats' in x: parts.append(f"Cells={x['N_Cats']}")
                return " | ".join(parts)
            return str(x)
        
        if 'Sample Info' in results_df.columns:
            results_df['Sample Info'] = results_df['Sample Info'].apply(fmt_sample)

    if correction_method != "None":
        st.info("""
        **Multiple Testing Correction Applied**
        
        When testing many variables, the chance of finding a "significant" result by random chance increases.
        *   **Bonferroni**: Very conservative. Controls the Family-Wise Error Rate.
        *   **Benjamini-Hochberg**: Controls the False Discovery Rate (FDR). Recommended for exploratory analysis.
        """)
    
    # Download Button
    st.download_button(
        label="Download Summary Table (Excel)",
        data=to_excel(results_df),
        file_name=f"epidemiology_results_{group_col}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Prepare display dataframe (Split Assumptions for ListColumn)
    display_df = results_df.copy()
    if 'Assumptions' in display_df.columns:
        display_df['Assumptions'] = display_df['Assumptions'].str.split('; ')

    # Style the dataframe
    format_dict = {"P-Value": "{:.4f}", "Statistic": "{:.2f}", "Effect Size": "{:.3f}"}
    subset_cols = ['P-Value']
    
    if "P-Value (Adj)" in results_df.columns:
        format_dict["P-Value (Adj)"] = "{:.4f}"
        subset_cols.append("P-Value (Adj)")
        
    st.dataframe(
        display_df.style.map(lambda x: 'background-color: #d4edda' if isinstance(x, (float, int)) and x < 0.05 else '', subset=subset_cols)
                        .format(format_dict),
        column_config={
            "Assumptions": st.column_config.ListColumn("Assumptions & Warnings", width="large"),
            "Sample Info": st.column_config.TextColumn("Sample Size Info", width="medium"),
            "Variable": st.column_config.TextColumn("Variable", width="medium"),
            "Test Used": st.column_config.TextColumn("Test", width="small"),
            "Effect Size": st.column_config.NumberColumn("Effect Size", format="%.3f"),
            "P-Value": st.column_config.NumberColumn("P-Value", format="%.4f"),
            "P-Value (Adj)": st.column_config.NumberColumn("P-Value (Adj)", format="%.4f"),
        },
        width="stretch"
    )
    
    # ============ NEW: Epidemiological Interpretation Summary ============
    st.subheader("Results Interpretation")
    
    # Categorize results by significance and effect size
    def classify_result(row):
        """Classify result into 4 quadrants based on p-value and effect size."""
        p_col = "P-Value (Adj)" if "P-Value (Adj)" in row.index else "P-Value"
        p_val = row.get(p_col, 1.0)
        es = abs(row.get('Effect Size', 0)) if not pd.isna(row.get('Effect Size', np.nan)) else 0
        es_type = row.get('Effect Type', '')
        
        # Determine if effect size is small/medium/large based on effect type
        if es_type in ["Cohen's d", "Rank-Biserial r"]:
            is_large = es >= 0.5
        elif es_type in ["Eta Squared", "η²H", "Epsilon Squared"]:
            is_large = es >= 0.06
        elif es_type == "Cramer's V":
            is_large = es >= 0.3
        elif es_type == "Odds Ratio":
            is_large = es >= 1.5 or es <= 0.67  # Moderate effect
        else:
            is_large = es >= 0.3  # Default
        
        is_sig = p_val < 0.05
        
        if is_sig and is_large:
            return "sig_large"
        elif is_sig and not is_large:
            return "sig_small"
        elif not is_sig and is_large:
            return "nonsig_large"
        else:
            return "nonsig_small"
    
    # Apply classification
    if not results_df.empty:
        classifications = results_df.apply(classify_result, axis=1)
        
        sig_large = results_df[classifications == "sig_large"]['Variable'].tolist()
        sig_small = results_df[classifications == "sig_small"]['Variable'].tolist()
        nonsig_large = results_df[classifications == "nonsig_large"]['Variable'].tolist()
        nonsig_small = results_df[classifications == "nonsig_small"]['Variable'].tolist()
        
        col_int1, col_int2 = st.columns(2)
        
        with col_int1:
            # Significant + Large Effect
            if sig_large:
                st.markdown(f"""
                <div style="background-color: #f1f8e9; padding: 14px; border-radius: 8px; margin: 5px 0; border-left: 4px solid #43a047;">
                    <strong style="color: #2e7d32;">Statistically and Clinically Significant ({len(sig_large)})</strong><br/>
                    <span style="font-size: 0.9em;">{', '.join(sig_large[:5])}{' ...' if len(sig_large) > 5 else ''}</span><br/>
                    <span style="font-size: 0.8em; color: #558b2f;">Strong statistical evidence paired with meaningful effect magnitude. Prioritize for clinical translation.</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Non-Significant + Small Effect
            if nonsig_small:
                st.markdown(f"""
                <div style="background-color: #f9fbe7; padding: 14px; border-radius: 8px; margin: 5px 0; border-left: 4px solid #9e9d24;">
                    <strong style="color: #558b2f;">True Negatives / No Effect ({len(nonsig_small)})</strong><br/>
                    <span style="font-size: 0.9em;">{', '.join(nonsig_small[:5])}{' ...' if len(nonsig_small) > 5 else ''}</span><br/>
                    <span style="font-size: 0.8em; color: #689f38;">Negligible effect size and non-significant p-value, consistent with the null hypothesis.</span>
                </div>
                """, unsafe_allow_html=True)
        
        with col_int2:
            # Significant + Small Effect
            if sig_small:
                st.markdown(f"""
                <div style="background-color: #fff8e1; padding: 14px; border-radius: 8px; margin: 5px 0; border-left: 4px solid #ffb300;">
                    <strong style="color: #f57f17;">Statistically Significant with Small Effect ({len(sig_small)})</strong><br/>
                    <span style="font-size: 0.9em;">{', '.join(sig_small[:5])}{' ...' if len(sig_small) > 5 else ''}</span><br/>
                    <span style="font-size: 0.8em; color: #b78103;">Likely detected due to sample size. Verify whether this magnitude meets minimum clinically important difference (MCID).</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Non-Significant + Large Effect
            if nonsig_large:
                st.markdown(f"""
                <div style="background-color: #ffebee; padding: 14px; border-radius: 8px; margin: 5px 0; border-left: 4px solid #e53935;">
                    <strong style="color: #c62828;">Potentially Underpowered / Indeterminate ({len(nonsig_large)})</strong><br/>
                    <span style="font-size: 0.9em;">{', '.join(nonsig_large[:5])}{' ...' if len(nonsig_large) > 5 else ''}</span><br/>
                    <span style="font-size: 0.8em; color: #b71c1c;">Substantial effect size without reaching alpha=0.05. High risk of Type II error; inspect power analysis tab.</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Summary stats
        n_sig = len(sig_large) + len(sig_small)
        n_total = len(results_df)
        st.caption(f"**Summary**: {n_sig}/{n_total} variables show p < 0.05 | {len(sig_large)} with meaningful effect size")
    
    with st.expander("Methodological Guidance"):
        st.markdown("""
        **Effect Size Standards** (Magnitude of difference):
        
        | Test | Effect Size | Small | Medium | Large |
        |:-----|:------------|:-----:|:------:|:-----:|
        | T-test | Cohen's d / Hedges' g | 0.2 | 0.5 | 0.8 |

        | Mann-Whitney | Rank-Biserial r | 0.1 | 0.3 | 0.5 |
        | ANOVA | η² | 0.01 | 0.06 | 0.14 |
        | Kruskal-Wallis | η²H | 0.01 | 0.06 | 0.14 |
        | Chi-Square | Cramér's V | 0.1 | 0.3 | 0.5 |
        | Fisher's Exact | Odds Ratio | 1 = No effect | >1 Increased | <1 Decreased |
        
        **🔬 Multiple Testing Correction**:
        - **Bonferroni**: Very conservative. Controls Family-Wise Error Rate (FWER). Best for confirmatory analysis.
        - **Benjamini-Hochberg (FDR)**: Controls False Discovery Rate. Recommended for exploratory analysis.
        """)

    # Detailed View
    st.subheader("Detailed Analysis & Visualization")
    selected_detail = st.selectbox("Select variable for detailed view", target_cols)
    
    if selected_detail:
        # Allow specific manual test selection for this variable
        is_numeric = selected_detail in analyzer.numeric_cols
        is_categorical = selected_detail in analyzer.categorical_cols or selected_detail in analyzer.binary_cols
        
        available_tests = ["Auto-Detect"]
        if is_numeric:
            if n_groups == 2:
                available_tests.extend(["Student's t-test", "Welch's t-test", "Mann-Whitney U"])
            else:
                available_tests.extend(["ANOVA", "Kruskal-Wallis"])
        elif is_categorical:
            available_tests.extend(["Chi-Square", "Fisher's Exact"])
            
        col_det1, col_det2 = st.columns([1, 3])
        with col_det1:
            selected_test_method = st.selectbox("Test Method (Detail)", available_tests, index=0)
        
        # Re-run analysis for this specific variable with the selected method
        detail_res = au.analyze_variable(
            df_analysis_filtered, 
            group_col, 
            selected_detail, 
            analyzer.numeric_cols,
            analyzer.categorical_cols,
            analyzer.binary_cols,
            manual_test=selected_test_method
        )
        
        with col_det2:
            st.info(f"""
            **{detail_res['Test Used']} Result**:
            Statistic: **{detail_res['Statistic']:.4f}** | P-Value: **{detail_res['P-Value']:.4e}**
            
            *Assumptions/Reasoning*: {detail_res['Assumptions']}
            """)

        # Post-Hoc Analysis
        if detail_res['P-Value'] < 0.05 and detail_res['Test Used'] in ["ANOVA", "Kruskal-Wallis"]:
            st.subheader("Post-Hoc Analysis")
            st.markdown(f"Since the global test is significant, we perform post-hoc tests to see which groups differ.")
            ph_res = au.perform_post_hoc(df_analysis_filtered, group_col, selected_detail, detail_res['Test Used'])
            if ph_res is not None and not ph_res.empty:
                st.dataframe(ph_res, width="stretch")
            else:
                st.write("No significant pairwise differences found after correction.")

        # Epidemiological 2x2 Risk & Association Panel
        if is_categorical and "epi_2x2" in detail_res:
            epi = detail_res["epi_2x2"]
            st.subheader("Epidemiological Risk & Association Analysis (2x2)")
            
            col_tbl, col_metrics = st.columns([1, 1])
            with col_tbl:
                st.markdown("##### 2x2 Contingency Table")
                tbl_df = pd.DataFrame(
                    [
                        [epi["a"], epi["b"], epi["a"] + epi["b"]],
                        [epi["c"], epi["d"], epi["c"] + epi["d"]],
                        [epi["a"] + epi["c"], epi["b"] + epi["d"], epi["n_total"]],
                    ],
                    index=["Exposed / Group 1", "Unexposed / Group 2", "Total"],
                    columns=["Event +", "Event -", "Total"],
                )
                st.table(tbl_df)
                if epi.get("haldane_applied"):
                    st.caption("Note: Haldane-Anscombe continuity correction (+0.5) applied due to zero cell count.")

            with col_metrics:
                st.markdown("##### Risk & Association Estimates")
                c_or, c_rr = st.columns(2)
                with c_or:
                    or_ci = epi["odds_ratio_ci"]
                    st.metric("Odds Ratio (OR)", f"{epi['odds_ratio']:.2f}")
                    st.caption(f"95% CI: **[{or_ci['lower']:.2f}, {or_ci['upper']:.2f}]**")
                with c_rr:
                    rr_ci = epi["relative_risk_ci"]
                    st.metric("Risk Ratio (RR)", f"{epi['relative_risk']:.2f}")
                    st.caption(f"95% CI: **[{rr_ci['lower']:.2f}, {rr_ci['upper']:.2f}]**")

                c_rd, c_nnt = st.columns(2)
                with c_rd:
                    rd_ci = epi["risk_difference_ci"]
                    st.metric("Risk Difference (RD)", f"{epi['risk_difference']*100:.1f}%")
                    st.caption(f"95% CI: **[{rd_ci['lower']*100:.1f}%, {rd_ci['upper']*100:.1f}%]**")
                with c_nnt:
                    nnt_val = epi.get("nnt")
                    st.metric("NNT / NNH", f"{nnt_val:.1f}" if nnt_val is not None else "N/A")
                    st.caption("Number Needed to Treat / Harm")

            # Methodological & Clinical Guidance
            if epi.get("guidance"):
                st.markdown("##### Clinical Epidemiological Guidance")
                for note in epi["guidance"]:
                    st.info(note)

        # Visualizations
        visualize_result(df_analysis_filtered, group_col, selected_detail, analyzer)

        
        # Diagnostic Plots (Q-Q Plot)
        if is_numeric:
            st.subheader("Assumption Diagnostics")
            col_diag1, col_diag2 = st.columns(2)
            with col_diag1:
                st.markdown("**Normality Check (Q-Q Plot)**")
                # Plot Q-Q for residuals or raw data? Usually residuals for ANOVA, but per group for T-test.
                # Let's plot for the whole dataset or per group? Per group is better but cluttered.
                # Let's plot residuals of the model: Value - GroupMean
                
                clean_df = df_analysis_filtered[[group_col, selected_detail]].dropna()
                group_means = clean_df.groupby(group_col)[selected_detail].transform('mean')
                residuals = clean_df[selected_detail] - group_means
                
                fig_qq = vu.create_qq_plot(residuals, title=f"Q-Q Plot of Residuals ({selected_detail})")
                st.plotly_chart(fig_qq, width='stretch')
                st.caption("Points should fall along the red line if data is Normally distributed.")

def run_multivariate_analysis(df, analyzer):
    st.header("Multivariate Analysis (ANCOVA)")
    
    # Educational Introduction
    with st.expander("What is ANCOVA?", expanded=False):
        st.markdown("""
        ### Analysis of Covariance (ANCOVA)
        
        ANCOVA combines ANOVA and regression to compare group means while **controlling for confounding variables** (covariates).
        
        **When to use ANCOVA:**
        - You want to compare group means on an outcome variable
        - There are continuous variables that may influence the outcome (confounders)
        - You need to "adjust" for baseline differences between groups
        
        **Model:** `Outcome ~ Group + Covariate₁ + Covariate₂ + ...`
        
        | Term | Description |
        |:-----|:------------|
        | **Target (Y)** | Continuous outcome variable (e.g., blood pressure, score) |
        | **Factor (Group)** | Categorical grouping variable (e.g., treatment vs control) |
        | **Covariate (X)** | Continuous confounder to adjust for (e.g., age, baseline value) |
        """)
        
        st.markdown("""
        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #4caf50;">
            <strong>💡 Key Interpretation:</strong><br/>
            ANCOVA answers: "Is there a group difference <em>after adjusting for</em> the covariates?"
        </div>
        """, unsafe_allow_html=True)
        
        st.warning("""
        **ANCOVA Assumptions:**
        1. **Independence** of observations
        2. **Normality** of residuals (check Q-Q plot)
        3. **Homogeneity of variance** across groups
        4. **Linearity** between covariate and outcome
        5. **Homogeneity of regression slopes** (same covariate effect in all groups)
        """)
    
    # Model Configuration
    st.subheader("Model Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_col = st.selectbox(
            "🎯 Target Variable (Y)",
            options=analyzer.numeric_cols,
            help="The continuous dependent variable you want to compare across groups."
        )
        
    with col2:
        group_col = st.selectbox(
            "👥 Grouping Variable (Factor)",
            options=[c for c in analyzer.categorical_cols + analyzer.binary_cols if df[c].nunique() < 10],
            help="The main categorical variable defining your groups (e.g., Treatment, Sex)."
        )
        
    with col3:
        covariates = st.multiselect(
            "📐 Covariates (Confounders)",
            options=[c for c in analyzer.numeric_cols if c != target_col],
            help="Continuous variables to adjust for (e.g., Age, Baseline score). These are potential confounders."
        )
    
    # Show preview of group structure
    if target_col and group_col:
        col_preview1, col_preview2 = st.columns(2)
        with col_preview1:
            group_counts = df[group_col].value_counts()
            st.caption(f"**Groups:** {', '.join([f'{k} (n={v})' for k, v in group_counts.items()])}")
        with col_preview2:
            if covariates:
                st.caption(f"**Adjusting for:** {', '.join(covariates)}")
            else:
                st.caption("⚠️ No covariates selected (equivalent to One-way ANOVA)")
        
    if not target_col or not group_col:
        st.info("Please select a Target and a Grouping variable to proceed.")
        return
    
    # Run Button
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        run_button = st.button("Run ANCOVA", type="primary", width="stretch")
    with col_btn2:
        st.caption("This will fit an OLS regression model and compute Type II ANOVA table.")
        
    if run_button:
        perform_ancova(df, target_col, group_col, covariates)

def perform_ancova(df, target, group, covariates):
    # Prepare data
    cols = [target, group] + (covariates if covariates else [])
    data = df[cols].dropna()
    
    if data.empty:
        st.error("No valid data after dropping missing values.")
        return
    
    n_missing = len(df) - len(data)
    if n_missing > 0:
        st.warning(f"{n_missing} rows with missing values were excluded from analysis.")

    try:
        # Sanitize column names for statsmodels formula
        safe_cols = {col: re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in cols}
        data_safe = data.rename(columns=safe_cols)
        
        target_safe = safe_cols[target]
        group_safe = safe_cols[group]
        covariates_safe = [safe_cols[c] for c in covariates]
        
        # Construct Formula
        if covariates_safe:
            formula = f"{target_safe} ~ C({group_safe}) + {' + '.join(covariates_safe)}"
        else:
            formula = f"{target_safe} ~ C({group_safe})"
        
        st.info(f"**Model Formula:** `{formula}`")
        
        # Fit OLS Model
        model = smf.ols(formula, data=data_safe).fit()
        
        # ============ Results Section ============
        st.subheader("ANCOVA Results")
        
        # ANOVA Table
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        # Calculate Partial Eta-Squared for effect size
        anova_table['Partial η²'] = anova_table['sum_sq'] / (anova_table['sum_sq'] + anova_table.loc['Residual', 'sum_sq'])
        
        # Display ANOVA Table with styling
        st.markdown("#### Type II ANOVA Table")
        
        formatted_anova = anova_table.copy()
        formatted_anova = formatted_anova.rename(columns={
            'sum_sq': 'Sum of Squares',
            'df': 'DF',
            'F': 'F-statistic',
            'PR(>F)': 'P-value'
        })
        
        st.dataframe(
            formatted_anova.style.format({
                'Sum of Squares': '{:.2f}',
                'F-statistic': '{:.3f}',
                'P-value': '{:.4f}',
                'Partial η²': '{:.3f}'
            }).map(
                lambda x: 'background-color: #c8e6c9' if isinstance(x, float) and x < 0.05 else '',
                subset=['P-value']
            ),
            width="stretch"
        )
        
        # Extract and interpret group effect
        group_row = anova_table.loc[f"C({group_safe})"]
        p_value = group_row["PR(>F)"]
        f_stat = group_row["F"]
        eta_sq = group_row["Partial η²"]
        
        # Interpretation box
        st.markdown("#### Results Interpretation")
        
        # Effect size interpretation
        if eta_sq < 0.01:
            eta_interp = "negligible"
            eta_color = "#e0e0e0"
        elif eta_sq < 0.06:
            eta_interp = "small"
            eta_color = "#fff3e0"
        elif eta_sq < 0.14:
            eta_interp = "medium"
            eta_color = "#ffe0b2"
        else:
            eta_interp = "large"
            eta_color = "#ffcc80"
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if p_value < 0.05:
                st.markdown(f"""
                <div style="background-color: #c8e6c9; padding: 18px; border-radius: 10px; border-left: 4px solid #4caf50;">
                    <h4 style="margin:0; color: #2e7d32;">Significant Group Effect</h4>
                    <p style="margin: 10px 0 5px 0;">
                        <strong>F({int(group_row['df'])}, {int(anova_table.loc['Residual', 'df'])}) = {f_stat:.2f}</strong>, 
                        <strong>p = {p_value:.4f}</strong>
                    </p>
                    <p style="margin: 0; font-size: 0.9em;">
                        Groups differ significantly on {target} after adjusting for covariates.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #fff3e0; padding: 18px; border-radius: 10px; border-left: 4px solid #ff9800;">
                    <h4 style="margin:0; color: #e65100;">No Significant Group Effect</h4>
                    <p style="margin: 10px 0 5px 0;">
                        <strong>F({int(group_row['df'])}, {int(anova_table.loc['Residual', 'df'])}) = {f_stat:.2f}</strong>, 
                        <strong>p = {p_value:.4f}</strong>
                    </p>
                    <p style="margin: 0; font-size: 0.9em;">
                        No significant group difference after adjusting for covariates.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_res2:
            st.markdown(f"""
            <div style="background-color: {eta_color}; padding: 18px; border-radius: 10px; border-left: 4px solid #ff9800;">
                <h4 style="margin:0; color: #5d4037;">Effect Size</h4>
                <p style="margin: 10px 0 5px 0;">
                    <strong>Partial η² = {eta_sq:.3f}</strong> ({eta_interp} effect)
                </p>
                <p style="margin: 0; font-size: 0.85em;">
                    {eta_sq*100:.1f}% of variance in {target} is explained by {group} (after controlling for covariates).
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Covariate effects
        if covariates:
            st.markdown("#### Covariate Effects")
            cov_data = []
            for cov in covariates_safe:
                if cov in anova_table.index:
                    cov_row = anova_table.loc[cov]
                    cov_p = cov_row["PR(>F)"]
                    cov_eta = cov_row["Partial η²"]
                    cov_beta = model.params.get(cov, 0)
                    cov_data.append({
                        "Covariate": cov,
                        "β (coefficient)": f"{cov_beta:.3f}",
                        "P-value": f"{cov_p:.4f}",
                        "Partial η²": f"{cov_eta:.3f}",
                        "Significant?": "Yes (p < 0.05)" if cov_p < 0.05 else "No"
                    })
            
            if cov_data:
                st.dataframe(pd.DataFrame(cov_data), width="stretch", hide_index=True)
                st.caption("**β coefficient**: Change in target per 1-unit increase in covariate, holding other variables constant.")
        
        # Full model summary (collapsible)
        with st.expander("Full Model Summary (OLS Regression)"):
            st.write(model.summary())
            st.caption("**R²** = proportion of variance explained by model. **Adj. R²** = penalized for number of predictors.")
            
        # ============ Visualizations & Diagnostics ============
        st.subheader("Visualizations & Diagnostics")
        
        viz_tabs = st.tabs(["Adjusted Means", "Raw vs Adjusted", "Model Diagnostics"])
        
        with viz_tabs[0]:
            # Calculate adjusted values
            data_adj = data.copy()
            y_adj = data_safe[target_safe].copy()
            
            for cov in covariates_safe:
                beta = model.params.get(cov, 0)
                mean_val = data_safe[cov].mean()
                y_adj -= beta * (data_safe[cov] - mean_val)
                
            data_adj['Adjusted ' + target] = y_adj
            
            fig = px.box(data_adj, x=group, y='Adjusted ' + target, color=group, 
                         title=f"Adjusted {target} by {group} (Covariates evaluated at group means)",
                         points="outliers")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")
            st.caption("Adjusted values = Original values minus covariate effects evaluated at covariate means.")
        
        with viz_tabs[1]:
            col_viz1, col_viz2 = st.columns(2)
            with col_viz1:
                fig_raw = px.box(data, x=group, y=target, color=group, title=f"Raw {target} by {group}")
                fig_raw.update_layout(showlegend=False)
                st.plotly_chart(fig_raw, width="stretch")
                st.caption("Raw (unadjusted) observed distributions")
            with col_viz2:
                fig_adj = px.box(data_adj, x=group, y='Adjusted ' + target, color=group, title=f"Adjusted {target} by {group}")
                fig_adj.update_layout(showlegend=False)
                st.plotly_chart(fig_adj, width="stretch")
                st.caption("Adjusted distributions with covariate confounding controlled")
        
        with viz_tabs[2]:
            st.markdown("#### Methodological Assumptions Verification")
            
            # Formally check ANCOVA assumptions using epidemiology engine
            from utils.epidemiology_utils import check_ancova_assumptions
            ancova_diag = check_ancova_assumptions(data, target=target, group=group, covariates=covariates)

            # 1. Homogeneity of Regression Slopes
            if covariates and ancova_diag.homogeneity_of_slopes_pvalue is not None:
                st.markdown("##### 1. Homogeneity of Regression Slopes (Parallel Slopes)")
                if ancova_diag.slopes_homogeneous:
                    st.success(
                        f"Parallel slopes assumption validated (Interaction test p = {ancova_diag.homogeneity_of_slopes_pvalue:.4f} >= 0.05). "
                        "The covariate effect is uniform across groups; standard ANCOVA adjusted means are statistically valid."
                    )
                else:
                    st.warning(
                        f"Violation of parallel slopes detected (Interaction test p = {ancova_diag.homogeneity_of_slopes_pvalue:.4f} < 0.05). "
                        "The covariate effect differs significantly by group. Standard ANCOVA adjusted means may be misleading. "
                        "Consider moderation analysis or reporting group-specific slopes."
                    )

            # 2. Residual Normality and Homoscedasticity Plots
            st.markdown("##### 2. Residual Distribution & Homoscedasticity")
            col_diag1, col_diag2 = st.columns(2)
            
            with col_diag1:
                fig_qq = vu.create_qq_plot(model.resid, title="Q-Q Plot of Residuals")
                st.plotly_chart(fig_qq, width="stretch")
                norm_status = "Residual normality confirmed" if ancova_diag.residuals_normal else "Residuals deviate from normality"
                st.caption(f"{norm_status} (p = {ancova_diag.residual_normality_pvalue:.4f})")
            
            with col_diag2:
                fig_rvf = px.scatter(x=model.fittedvalues, y=model.resid, 
                                     labels={'x': 'Fitted Values', 'y': 'Residuals'},
                                     title="Residuals vs Fitted Values")
                fig_rvf.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_rvf, width="stretch")
                homo_status = "Homoscedasticity confirmed" if ancova_diag.homoscedastic else "Heteroscedasticity detected"
                homo_p_str = f"p = {ancova_diag.homoscedasticity_pvalue:.4f}" if ancova_diag.homoscedasticity_pvalue is not None else "N/A"
                st.caption(f"{homo_status} ({homo_p_str})")

        
    except Exception as e:
        st.error(f"An error occurred during ANCOVA computation: {e}")
        st.write("**Tip:** Ensure categorical variables are properly encoded. Check for multicollinearity among covariates.")

def visualize_result(df, group_col, target, analyzer):
    is_numeric = target in analyzer.numeric_cols
    
    if is_numeric:
        # Reference Lines Configuration
        with st.expander("Reference Lines & Thresholds"):
            st.caption("Add horizontal reference lines (e.g., clinical thresholds, normal range limits).")
            col_ref1, col_ref2, col_ref3 = st.columns([2, 2, 1])
            with col_ref1:
                ref_val = st.number_input("Value", value=0.0, key=f"ref_val_{target}")
            with col_ref2:
                ref_label = st.text_input("Label", value="Threshold", key=f"ref_lbl_{target}")
            with col_ref3:
                add_ref = st.checkbox("Show Line", key=f"add_ref_{target}")
        
        # 1. Boxplot with Significance
        st.subheader("Distribution & Significance")
        
        groups = df[group_col].dropna().unique()
        if len(groups) >= 2:
            data_list = [df[df[group_col] == g][target].dropna().values for g in groups]
            significant_pairs = vu.compute_significant_pairs(data_list)
            
            fig = vu.create_boxplot_with_significance_streamlit(
                df=df,
                group_col=group_col,
                value_col=target,
                significant_pairs=significant_pairs,
                title=f"{target} by {group_col} (with Significance)",
                y_label=target
            )
            
            if add_ref:
                fig.add_hline(y=ref_val, line_dash="dash", line_color="red", annotation_text=ref_label, annotation_position="bottom right")
                
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("Not enough groups to display significance plot.")

        # 2. Confidence Interval Plot
        st.subheader("Means & 95% Confidence Intervals")
        
        # Calculate stats
        stats_df = df.groupby(group_col)[target].agg(['mean', 'count', 'std']).reset_index()
        # Calculate 95% CI
        # CI = 1.96 * (std / sqrt(n))
        # Use Student's t-distribution for accurate CI (important for small samples)
        stats_df['ci'] = stats_df.apply(
            lambda row: t_dist.ppf(0.975, row['count'] - 1) * (row['std'] / np.sqrt(row['count'])) 
            if row['count'] > 1 else 0, axis=1
        )
        
        fig_ci = go.Figure()
        
        fig_ci.add_trace(go.Scatter(
            x=stats_df[group_col],
            y=stats_df['mean'],
            error_y=dict(
                type='data',
                array=stats_df['ci'],
                visible=True,
                width=5,
                thickness=1.5
            ),
            mode='markers',
            marker=dict(size=12, color='royalblue', symbol='diamond'),
            name='Mean (95% CI)'
        ))
        
        if add_ref:
            fig_ci.add_hline(y=ref_val, line_dash="dash", line_color="red", annotation_text=ref_label)
        
        fig_ci.update_layout(
            title=f"Mean {target} by {group_col} (95% CI)",
            xaxis_title=group_col,
            yaxis_title=f"Mean {target}",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig_ci, width='stretch')
        
        # 3. Histogram
        st.subheader("Distribution Histogram")
        fig2 = px.histogram(df, x=target, color=group_col, barmode="overlay", 
                            title=f"Histogram of {target} by {group_col}",
                            opacity=0.7)
        if add_ref:
            fig2.add_vline(x=ref_val, line_dash="dash", line_color="red", annotation_text=ref_label)
            
        st.plotly_chart(fig2, width='stretch')
        
    else:
        # Bar chart (Stacked or Grouped)
        # Calculate percentages
        counts = df.groupby([group_col, target]).size().reset_index(name='count')
        totals = df.groupby([group_col]).size().reset_index(name='total')
        counts = counts.merge(totals, on=group_col)
        counts['percentage'] = counts['count'] / counts['total'] * 100
        
        fig = px.bar(counts, x=group_col, y='percentage', color=target, 
                     title=f"Distribution of {target} by {group_col} (%)",
                     text=counts['percentage'].apply(lambda x: '{0:1.2f}%'.format(x)))
        st.plotly_chart(fig, width='stretch')

def run_zscore_analysis(df, analyzer):
    st.header("Z-Score Standardization & Reference Analysis")
    
    # Educational Introduction
    with st.expander("What is Z-Score Standardization?", expanded=False):
        st.markdown("""
        ### Understanding Z-Scores in Epidemiology
        
        A **Z-score** expresses how many standard deviations a value is from the reference mean.
        
        $$Z = \\frac{X - \\mu_{ref}}{\\sigma_{ref}}$$
        
        | Term | Meaning |
        |:-----|:--------|
        | **X** | Individual observation |
        | **μ_ref** | Reference population mean |
        | **σ_ref** | Reference population standard deviation |
        | **Z** | Number of standard deviations from reference mean |
        """)
        
        st.markdown("### Z-Score Interpretation")
        st.markdown("""
        <table style="width:100%; border-collapse: collapse; text-align: center;">
            <tr style="background-color: #f5f5f5;">
                <th style="border: 1px solid #ddd; padding: 10px;">Z-Score</th>
                <th style="border: 1px solid #ddd; padding: 10px;">Interpretation</th>
                <th style="border: 1px solid #ddd; padding: 10px;">Percentile</th>
                <th style="border: 1px solid #ddd; padding: 10px;">Clinical Significance</th>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;"><strong>Z > +2</strong></td>
                <td style="border: 1px solid #ddd; padding: 8px; background-color: #ffcdd2;">Abnormally High</td>
                <td style="border: 1px solid #ddd; padding: 8px;">> 97.7%</td>
                <td style="border: 1px solid #ddd; padding: 8px; color: #c62828;">Outside normal range</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;"><strong>+1 < Z ≤ +2</strong></td>
                <td style="border: 1px solid #ddd; padding: 8px; background-color: #fff3e0;">Mildly Elevated</td>
                <td style="border: 1px solid #ddd; padding: 8px;">84.1% - 97.7%</td>
                <td style="border: 1px solid #ddd; padding: 8px; color: #ef6c00;">Borderline high</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;"><strong>-1 ≤ Z ≤ +1</strong></td>
                <td style="border: 1px solid #ddd; padding: 8px; background-color: #c8e6c9;">Normal Range</td>
                <td style="border: 1px solid #ddd; padding: 8px;">15.9% - 84.1%</td>
                <td style="border: 1px solid #ddd; padding: 8px; color: #2e7d32;">Within normal limits</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;"><strong>-2 ≤ Z < -1</strong></td>
                <td style="border: 1px solid #ddd; padding: 8px; background-color: #fff3e0;">Mildly Low</td>
                <td style="border: 1px solid #ddd; padding: 8px;">2.3% - 15.9%</td>
                <td style="border: 1px solid #ddd; padding: 8px; color: #ef6c00;">Borderline low</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;"><strong>Z < -2</strong></td>
                <td style="border: 1px solid #ddd; padding: 8px; background-color: #ffcdd2;">Abnormally Low</td>
                <td style="border: 1px solid #ddd; padding: 8px;">< 2.3%</td>
                <td style="border: 1px solid #ddd; padding: 8px; color: #c62828;">Outside normal range</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)
        
        st.info("""
        **Common Applications:**
        - **Growth charts** (height-for-age, weight-for-age Z-scores)
        - **Laboratory values** (comparing to reference ranges)
        - **Cognitive/behavioral assessments** (IQ, developmental scores)
        - **Multi-variable comparison** (making different measures comparable)
        """)
        
        st.warning("""
        **Assumptions:**
        - Reference population is representative
        - Data is approximately normally distributed (for meaningful percentile interpretation)
        - Same measurement method was used
        """)
    
    # Configuration Section
    st.subheader("Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_vars = st.multiselect(
            "Select Variables to Standardize",
            options=analyzer.numeric_cols,
            default=analyzer.numeric_cols[:3] if len(analyzer.numeric_cols) >= 3 else analyzer.numeric_cols,
            help="Select continuous variables to calculate Z-scores for."
        )
        
    with col2:
        ref_method = st.selectbox(
            "Reference Population Method",
            ["Internal Control Group", "Whole Cohort (Standardization)", "Manual Reference Values"],
            help="Define the mean and standard deviation used for Z-score calculation."
        )
        
    ref_group_col = None
    ref_group_val = None
    manual_refs = {}
    
    # Reference method specific configuration
    if ref_method == "Internal Control Group":
        col_ref1, col_ref2 = st.columns(2)
        with col_ref1:
            potential_groupers = [c for c in analyzer.categorical_cols + analyzer.binary_cols if df[c].nunique() < 20]
            ref_group_col = st.selectbox(
                "Select Grouping Variable",
                options=potential_groupers,
                help="Variable that identifies groups (e.g., Case/Control, Treatment/Placebo)"
            )
        with col_ref2:
            if ref_group_col:
                ref_group_val = st.selectbox(
                    "Select Reference Group (Control)",
                    options=df[ref_group_col].unique(),
                    help="The group to use as reference (mean=0, SD=1 for this group)"
                )
                
        if ref_group_col and ref_group_val:
            ref_n = len(df[df[ref_group_col] == ref_group_val])
            st.caption(f"Reference group: **{ref_group_val}** (n = {ref_n})")
            
    elif ref_method == "Whole Cohort (Standardization)":
        st.info("Using the entire cohort mean and SD as reference. All Z-scores will have mean ≈ 0 and SD ≈ 1.")
        
    elif ref_method == "Manual Reference Values":

        st.info("Enter published or population-based reference values below.")

    if not target_vars:
        st.warning("Please select at least one variable to standardize.")
        return
    
    # Create a configuration key to detect changes
    config_key = f"{','.join(sorted(target_vars))}_{ref_method}_{ref_group_col}_{ref_group_val}"
    
    # Check if we need to recalculate (config changed or button pressed)
    need_recalc = False
    
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        run_button = st.button("Calculate Z-Scores", type="primary", width="stretch")
    with col_btn2:
        st.caption("This will standardize selected variables against the reference population.")
    
    # Determine if we need to calculate
    if run_button:
        need_recalc = True
    elif 'zscore_config_key' in st.session_state and st.session_state['zscore_config_key'] != config_key:
        # Config changed, need recalc
        need_recalc = True
    
    # Check if we have cached results
    has_cached_results = 'zscore_results' in st.session_state and st.session_state.get('zscore_config_key') == config_key
    
    if not has_cached_results and not run_button:
        st.info("Click 'Calculate Z-Scores' to compute standardized values.")
        return
    
    # Calculate Z-Scores (only if needed)
    if need_recalc or not has_cached_results:
        z_df = df.copy()
        z_cols = []
        ref_stats = {}

        for var in target_vars:
            col_name = f"{var}_Z"
            z_cols.append(col_name)
            
            if ref_method == "Whole Cohort (Standardization)":
                mu = df[var].mean()
                sigma = df[var].std()
            elif ref_method == "Internal Control Group":
                if ref_group_col and ref_group_val:
                    control_data = df[df[ref_group_col] == ref_group_val][var]
                    mu = control_data.mean()
                    sigma = control_data.std()
                else:
                    mu, sigma = 0, 1
            else:  # Manual - use session state values
                mu = st.session_state.get(f"mu_{var}", 0.0)
                sigma = st.session_state.get(f"sigma_{var}", 1.0)
            
            ref_stats[var] = {"mean": round(mu, 3), "std": round(sigma, 3)}
            z_df[col_name] = (df[var] - mu) / sigma if sigma > 0 else 0
        
        # Cache results in session state
        st.session_state['zscore_results'] = {
            'z_df': z_df,
            'z_cols': z_cols,
            'ref_stats': ref_stats,
            'target_vars': target_vars
        }
        st.session_state['zscore_config_key'] = config_key
    else:
        # Use cached results
        cached = st.session_state['zscore_results']
        z_df = cached['z_df']
        z_cols = cached['z_cols']
        ref_stats = cached['ref_stats']
        target_vars = cached['target_vars']

    # ============ Results Section ============
    st.subheader("Results")
    
    # Summary Statistics
    st.markdown("#### Summary of Z-Scores")
    
    summary_data = []
    for z_col in z_cols:
        orig_var = z_col.replace("_Z", "")
        vals = z_df[z_col].dropna()
        n_abnormal_high = (vals > 2).sum()
        n_abnormal_low = (vals < -2).sum()
        n_abnormal = n_abnormal_high + n_abnormal_low
        pct_abnormal = (n_abnormal / len(vals) * 100) if len(vals) > 0 else 0
        
        summary_data.append({
            "Variable": orig_var,
            "Ref Mean": ref_stats[orig_var]["mean"],
            "Ref SD": ref_stats[orig_var]["std"],
            "Mean Z": f"{vals.mean():.2f}",
            "SD": f"{vals.std():.2f}",
            "Min": f"{vals.min():.2f}",
            "Max": f"{vals.max():.2f}",
            "n Abnormal (|Z|>2)": n_abnormal,
            "% Abnormal": f"{pct_abnormal:.1f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, width="stretch", hide_index=True)
    
    # Interpretation helper
    total_abnormal = sum([int(row["n Abnormal (|Z|>2)"]) for row in summary_data])
    if total_abnormal > 0:
        st.warning(f"**{total_abnormal} observations** have Z-scores outside ±2 SD (potentially abnormal values)")
    else:
        st.success("All observations are within ±2 SD of the reference mean")
    
    # Download buttons
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="📥 Download Z-Score Data",
            data=to_excel(z_df[list(df.columns) + z_cols]),
            file_name="z_score_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with col_dl2:
        st.download_button(
            label="📥 Download Summary Table",
            data=to_excel(summary_df),
            file_name="z_score_summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # ============ Visualization Section ============
    st.subheader("Visualizations")
    
    viz_tabs = st.tabs(["Group Comparison", "Heatmap", "Individual Profile"])
    
    with viz_tabs[0]:
        st.markdown("#### Forest Plot: Mean Z-Scores by Group")
        
        col_grp1, col_grp2 = st.columns([2, 1])
        with col_grp1:
            group_col = st.selectbox(
                "Group by", 
                options=[c for c in analyzer.categorical_cols + analyzer.binary_cols if df[c].nunique() < 20],
                key="zscore_group_col"
            )
        with col_grp2:
            error_type = st.radio(
                "Error bars",
                ["95% CI", "±1 SD", "±2 SD"],
                horizontal=True,
                help="CI = Confidence Interval of the mean (smaller). SD = Standard Deviation of values (larger)."
            )
        
        if group_col:
            plot_data = []
            groups = z_df[group_col].dropna().unique()
            
            for g in groups:
                g_data = z_df[z_df[group_col] == g]
                for z_col in z_cols:
                    orig_var = z_col.replace("_Z", "")
                    vals = g_data[z_col].dropna()
                    if len(vals) > 1:
                        mean = vals.mean()
                        sd = vals.std()
                        sem = stats.sem(vals)
                        ci = t_dist.ppf(0.975, len(vals)-1) * sem if len(vals) > 1 else 0
                        
                        # Select error bar based on user choice
                        if error_type == "95% CI":
                            error = ci
                        elif error_type == "±1 SD":
                            error = sd
                        else:  # ±2 SD
                            error = 2 * sd
                        
                        plot_data.append({
                            "Group": g,
                            "Variable": orig_var,
                            "Mean Z": mean,
                            "SD": sd,
                            "SEM": sem,
                            "CI": ci,
                            "Error": error,
                            "n": len(vals)
                        })
            
            plot_df = pd.DataFrame(plot_data)
            
            if not plot_df.empty:
                fig = go.Figure()
                colors = px.colors.qualitative.Plotly
                
                for i, g in enumerate(groups):
                    g_plot = plot_df[plot_df["Group"] == g]
                    color = colors[i % len(colors)]
                    
                    fig.add_trace(go.Scatter(
                        x=g_plot["Mean Z"],
                        y=g_plot["Variable"],
                        error_x=dict(
                            type='data',
                            array=g_plot["Error"],
                            visible=True,
                            thickness=2,
                            width=4
                        ),
                        mode='markers',
                        name=str(g),
                        marker=dict(color=color, size=12, line=dict(width=1, color='black')),
                        hovertemplate=f"<b>{g}</b><br>Mean: %{{x:.2f}}<br>SD: %{{customdata[0]:.2f}}<br>n: %{{customdata[1]}}<extra></extra>",
                        customdata=g_plot[["SD", "n"]].values
                    ))
                
                # Add reference lines
                fig.add_vline(x=0, line_width=2, line_color="black")
                fig.add_vrect(x0=-2, x1=2, fillcolor="green", opacity=0.08, line_width=0)
                fig.add_vline(x=2, line_width=1, line_dash="dash", line_color="red")
                fig.add_vline(x=-2, line_width=1, line_dash="dash", line_color="red")
                
                fig.update_layout(
                    title=f"Mean Z-Scores by {group_col} ({error_type})",
                    xaxis_title="Z-Score (SD from Reference Mean)",
                    yaxis_title="Variable",
                    height=max(450, len(target_vars) * 80),
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    xaxis=dict(range=[-4, 4], dtick=1),
                    margin=dict(l=150)  # More space for variable names
                )
                st.plotly_chart(fig, width="stretch")
                
                # Explanatory note
                # Scientific interpretation
                with st.expander("How to Interpret this Forest Plot", expanded=False):
                    st.markdown("""
                    ### Forest Plot Interpretation Guide
                    
                    **What this shows:** Mean Z-scores for each variable, stratified by group, relative to the reference population.
                    
                    | Pattern | Interpretation | Action |
                    |:--------|:---------------|:-------|
                    | Mean at 0 | Group is similar to reference | No deviation |
                    | Mean > +2 or < -2 | Group is **abnormally** different | Investigate clinically |
                    | Small CI | Precise estimate (large n or low variance) | High confidence |
                    | Large CI | Imprecise estimate (small n or high variance) | Interpret cautiously |
                    | Groups overlap | No significant difference between groups | Similar profiles |
                    | Groups don't overlap | Likely significant difference | Compare further |
                    
                    **Epidemiological Use:**
                    - Compare disease vs control groups on biomarkers
                    - Assess treatment effects across clinical measures  
                    - Identify variables where groups differ most
                    
                    > **Note:** When using **95% CI**, non-overlapping bars suggest p < 0.05.
                    > When using **SD**, bars show individual variability, not statistical precision.
                    """)

    with viz_tabs[1]:
        st.markdown("#### Heatmap: Z-Score Distribution Across Samples")
        
        if len(z_df) > 500:
            st.info(f"Downsampling from {len(z_df)} to 500 rows for performance.")
            heatmap_df = z_df.sample(500, random_state=42)
        else:
            heatmap_df = z_df
            
        sort_col = st.selectbox("Sort samples by", ["None"] + z_cols, key="heatmap_sort")
        if sort_col != "None":
            heatmap_df = heatmap_df.sort_values(sort_col)
            
        fig = px.imshow(
            heatmap_df[z_cols].values,
            x=[c.replace("_Z", "") for c in z_cols],
            labels=dict(x="Variable", y="Sample", color="Z-Score"),
            color_continuous_scale="RdBu_r",
            zmin=-3, zmax=3,
            aspect="auto"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, width="stretch")
        
        # Scientific interpretation
        with st.expander("How to Interpret this Heatmap", expanded=False):
            st.markdown("""
            ### Heatmap Interpretation Guide
            
            **What this shows:** Z-scores for all samples (rows) across all variables (columns).
            
            | Shading | Z-Score | Meaning |
            |:--------|:--------|:--------|
            | Dark Blue | < -2 | Abnormally LOW (< 2.3 percentile) |
            | Light Blue | -2 to -1 | Mildly low |
            | White | -1 to +1 | Normal range |
            | Light Red | +1 to +2 | Mildly elevated |
            | Dark Red | > +2 | Abnormally HIGH (> 97.7 percentile) |
            
            **Patterns to Look For:**
            - **Horizontal bands (rows):** Individual with multiple abnormal values -> systemic issue
            - **Vertical bands (columns):** Variable abnormal across many samples -> measurement issue or population shift
            - **Clusters:** Groups of similar profiles -> potential subpopulations
            
            **Epidemiological Use:**
            - Identify outlier individuals for case review
            - Detect variables with high population-level deviation
            - Screen for systematic measurement biases
            """)

    with viz_tabs[2]:
        st.markdown("#### Individual Z-Score Profile")
        
        sample_id_col = st.selectbox("Select Sample ID Column", ["Row Index"] + analyzer.categorical_cols, key="profile_id_col")
        
        if sample_id_col == "Row Index":
            sample_idx = st.number_input("Row Index", min_value=0, max_value=len(z_df)-1, value=0, step=1)
            selected_row = z_df.iloc[sample_idx]
            label = f"Sample #{sample_idx}"
        else:
            sample_val = st.selectbox("Select Sample", z_df[sample_id_col].unique())
            selected_row = z_df[z_df[sample_id_col] == sample_val].iloc[0]
            label = str(sample_val)
            
        values = [selected_row[c] for c in z_cols]
        var_names = [c.replace("_Z", "") for c in z_cols]
        
        # Color by abnormality
        colors_bar = []
        for v in values:
            if abs(v) > 2:
                colors_bar.append('#f44336')  # Red - abnormal
            elif abs(v) > 1:
                colors_bar.append('#ff9800')  # Orange - borderline
            else:
                colors_bar.append('#4caf50')  # Green - normal
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=var_names,
            y=values,
            marker_color=colors_bar,
            text=[f"{v:.2f}" for v in values],
            textposition='outside'
        ))
        
        # Reference lines
        fig.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="+2 SD (Abnormal)")
        fig.add_hline(y=-2, line_dash="dash", line_color="red", annotation_text="-2 SD (Abnormal)")
        fig.add_hline(y=0, line_color="black", line_width=2)
        fig.add_hrect(y0=-2, y1=2, fillcolor="green", opacity=0.1, line_width=0)
        
        fig.update_layout(
            title=f"Z-Score Profile: {label}",
            yaxis_title="Z-Score",
            yaxis_range=[min(min(values) - 0.5, -3), max(max(values) + 0.5, 3)],
            template="plotly_white"
        )
        st.plotly_chart(fig, width="stretch")
        
        # Interpretation table for this individual
        n_abnormal = sum(1 for v in values if abs(v) > 2)
        n_borderline = sum(1 for v in values if 1 < abs(v) <= 2)
        
        col_interp1, col_interp2, col_interp3 = st.columns(3)
        with col_interp1:
            st.metric("Abnormal (|Z| > 2)", n_abnormal)
        with col_interp2:
            st.metric("Borderline (1 < |Z| ≤ 2)", n_borderline)
        with col_interp3:
            st.metric("Normal (|Z| ≤ 1)", len(values) - n_abnormal - n_borderline)
        
        # Scientific interpretation
        with st.expander("How to Interpret Individual Profiles", expanded=False):
            st.markdown("""
            ### Individual Profile Interpretation Guide
            
            **What this shows:** One individual's Z-scores across all measured variables.
            
            | Zone | Clinical Interpretation |
            |:-----|:------------------------|
            | 🟢 Green bars | Values within normal limits |
            | 🟠 Orange bars | Borderline values - monitor |
            | 🔴 Red bars | Abnormal values - investigate |
            
            **Clinical Decision Rules:**
            - **All green:** Normal profile, routine follow-up
            - **1-2 orange:** Monitor at next visit
            - **Any red:** Requires clinical review
            - **Multiple red:** Consider comprehensive workup
            
            **Things to Consider:**
            - Is the abnormality **clinically meaningful** (not just statistically)?
            - Is it **isolated** or part of a **pattern** (e.g., all lipid markers elevated)?
            - What is the **direction** of deviation (high vs low)?
            - Does the **reference population** match this individual's demographics?
            
            **Epidemiological Context:**
            - ~5% of healthy individuals will have at least one |Z| > 2 by chance alone
            - More variables = higher false positive rate (multiple testing)
            """)

def show_educational_content():
    st.header("Statistical Test Guide")
    
    with st.expander("Choosing the Right Test - Decision Tree", expanded=True):
        st.markdown("### Step-by-Step Test Selection Guide")
        st.caption("Follow this decision tree to select the appropriate statistical test for your analysis.")
        
        # ============ NODE 1: Number of Groups ============
        st.markdown("""
        <div style="background-color: #e3f2fd; padding: 20px; border-radius: 12px; border: 2px solid #1976d2; margin: 10px 0;">
            <h4 style="margin:0; color: #1976d2;">Node 1: How many groups are you comparing?</h4>
            <p style="margin: 10px 0 5px 0;"><strong>Decision Criterion:</strong> Count the number of distinct categories in your grouping variable.</p>
            <p style="margin: 0; font-size: 0.9em; color: #555;">
                <em>Example: Treatment vs Control = 2 groups | Low/Medium/High dose = 3 groups</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col_2g, col_3g = st.columns(2)
        with col_2g:
            st.markdown("##### **2 Groups** → Go to Node 2A")
        with col_3g:
            st.markdown("##### **3+ Groups** → Go to Node 2B")
        
        st.markdown("---")
        
        # ============ NODE 2: Variable Type ============
        col_2a, col_2b = st.columns(2)
        
        with col_2a:
            st.markdown("""
            <div style="background-color: #fff3e0; padding: 18px; border-radius: 12px; border: 2px solid #f57c00; margin: 5px 0;">
                <h4 style="margin:0; color: #f57c00;">Node 2A: What type of variable is your outcome? (2 Groups)</h4>
                <p style="margin: 10px 0 5px 0;"><strong>Decision Criterion:</strong></p>
                <ul style="margin: 5px 0; font-size: 0.9em;">
                    <li><strong>Numeric (Continuous)</strong>: Age, blood pressure, BMI, lab values</li>
                    <li><strong>Categorical</strong>: Disease status, sex, treatment response (Yes/No)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("##### **Numeric** → Go to Node 3A")
            st.markdown("##### **Categorical** → Go to Node 3C")
        
        with col_2b:
            st.markdown("""
            <div style="background-color: #fff3e0; padding: 18px; border-radius: 12px; border: 2px solid #f57c00; margin: 5px 0;">
                <h4 style="margin:0; color: #f57c00;">Node 2B: What type of variable is your outcome? (3+ Groups)</h4>
                <p style="margin: 10px 0 5px 0;"><strong>Decision Criterion:</strong></p>
                <ul style="margin: 5px 0; font-size: 0.9em;">
                    <li><strong>Numeric (Continuous)</strong>: Age, blood pressure, BMI, lab values</li>
                    <li><strong>Categorical</strong>: Disease status, sex, treatment response (Yes/No)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("##### **Numeric** → Go to Node 3B")
            st.markdown("##### **Categorical** → Use **Chi-Square Test** ")
        
        st.markdown("---")
        
        # ============ NODE 3: Normality Check ============
        st.markdown("""
        <div style="background-color: #e8f5e9; padding: 20px; border-radius: 12px; border: 2px solid #4caf50; margin: 10px 0;">
            <h4 style="margin:0; color: #388e3c;">Node 3: Is your data normally distributed?</h4>
            <p style="margin: 10px 0 5px 0;"><strong>Decision Criteria:</strong></p>
            <table style="width:100%; font-size: 0.9em; margin-top: 10px;">
                <tr>
                    <td style="padding: 5px;"><strong>Test</strong></td>
                    <td style="padding: 5px;"><strong>How to Check</strong></td>
                    <td style="padding: 5px;"><strong>Interpretation</strong></td>
                </tr>
                <tr>
                    <td style="padding: 5px;">Shapiro-Wilk Test</td>
                    <td style="padding: 5px;">Run on each group separately</td>
                    <td style="padding: 5px;">p > 0.05 → Normal ✅ | p < 0.05 → Non-Normal ❌</td>
                </tr>
                <tr>
                    <td style="padding: 5px;">Q-Q Plot</td>
                    <td style="padding: 5px;">Visual inspection</td>
                    <td style="padding: 5px;">Points on diagonal → Normal ✅ | Curved → Non-Normal ❌</td>
                </tr>
                <tr>
                    <td style="padding: 5px;">Sample Size Rule</td>
                    <td style="padding: 5px;">Check n per group</td>
                    <td style="padding: 5px;">n ≥ 30 → CLT applies (parametric OK) | n < 30 → Check normality carefully</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        col_3a, col_3b = st.columns(2)
        
        with col_3a:
            st.markdown("##### Node 3A (2 Groups, Numeric):")
            st.markdown("➡️ **Normal** → Go to Node 4")
            st.markdown("➡️ **Non-Normal** → Use **Mann-Whitney U Test** ✅")
            st.caption("_Mann-Whitney compares ranks, not means. Robust to outliers._")
        
        with col_3b:
            st.markdown("##### Node 3B (3+ Groups, Numeric):")
            st.markdown("➡️ **Normal** → Go to Node 5")
            st.markdown("➡️ **Non-Normal** → Use **Kruskal-Wallis Test** ✅")
            st.caption("_Kruskal-Wallis is the non-parametric alternative to ANOVA._")
        
        st.markdown("---")
        
        # ============ NODE 3C: Categorical Decision ============
        st.markdown("""
        <div style="background-color: #f3e5f5; padding: 18px; border-radius: 12px; border: 2px solid #9c27b0; margin: 10px 0;">
            <h4 style="margin:0; color: #7b1fa2;">Node 3C: Categorical Variable - Check Expected Frequencies</h4>
            <p style="margin: 10px 0 5px 0;"><strong>Decision Criterion:</strong> Calculate expected cell counts in the contingency table.</p>
            <ul style="margin: 5px 0; font-size: 0.9em;">
                <li><strong>All expected counts ≥ 5</strong> → Use <strong>Chi-Square Test</strong> ✅</li>
                <li><strong>Any expected count < 5</strong> → Use <strong>Fisher's Exact Test</strong> ✅</li>
            </ul>
            <p style="margin: 5px 0 0 0; font-size: 0.85em; color: #555;">
                <em>Expected count = (Row Total × Column Total) / Grand Total</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ============ NODE 4: Equal Variances (2 Groups) ============
        st.markdown("""
        <div style="background-color: #fce4ec; padding: 18px; border-radius: 12px; border: 2px solid #e91e63; margin: 10px 0;">
            <h4 style="margin:0; color: #c2185b;">Node 4: Are variances equal between groups? (2 Groups)</h4>
            <p style="margin: 10px 0 5px 0;"><strong>Decision Criterion:</strong> Levene's Test for Equality of Variances</p>
            <ul style="margin: 5px 0; font-size: 0.9em;">
                <li><strong>Levene p > 0.05</strong> → Variances are equal → Use <strong>Student's t-test</strong> ✅</li>
                <li><strong>Levene p < 0.05</strong> → Variances are unequal → Use <strong>Welch's t-test</strong> ✅</li>
            </ul>
            <p style="margin: 5px 0 0 0; font-size: 0.85em; color: #555;">
                <em>Welch's t-test is robust to unequal variances (heteroscedasticity) and is often recommended as the default choice.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ============ NODE 5: Equal Variances (3+ Groups) ============
        st.markdown("""
        <div style="background-color: #fce4ec; padding: 18px; border-radius: 12px; border: 2px solid #e91e63; margin: 10px 0;">
            <h4 style="margin:0; color: #c2185b;">Node 5: Are variances equal between groups? (3+ Groups)</h4>
            <p style="margin: 10px 0 5px 0;"><strong>Decision Criterion:</strong> Levene's Test for Equality of Variances</p>
            <ul style="margin: 5px 0; font-size: 0.9em;">
                <li><strong>Levene p > 0.05</strong> → Variances are equal → Use <strong>One-way ANOVA</strong> ✅</li>
                <li><strong>Levene p < 0.05</strong> → Variances are unequal → Use <strong>Welch's ANOVA</strong> or <strong>Kruskal-Wallis</strong> ✅</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ============ FINAL: Post-Hoc Tests ============
        st.markdown("""
        <div style="background-color: #e0f7fa; padding: 18px; border-radius: 12px; border: 2px solid #00bcd4; margin: 10px 0;">
            <h4 style="margin:0; color: #0097a7;">After the Test: Post-Hoc Analysis (for 3+ Groups)</h4>
            <p style="margin: 10px 0 5px 0;"><strong>If the global test is significant (p < 0.05):</strong></p>
            <ul style="margin: 5px 0; font-size: 0.9em;">
                <li><strong>After ANOVA</strong> → Run <strong>Tukey's HSD</strong> (Honestly Significant Difference)</li>
                <li><strong>After Kruskal-Wallis</strong> → Run <strong>Dunn's Test</strong> with Bonferroni correction</li>
            </ul>
            <p style="margin: 5px 0 0 0; font-size: 0.85em; color: #555;">
                <em>Post-hoc tests identify WHICH groups differ from each other.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Legend
        st.markdown("---")
        st.markdown("#### Summary Table")
        st.markdown("""
        | Scenario | Normal Data | Non-Normal Data |
        |:---------|:-----------:|:---------------:|
        | **2 Groups, Numeric** | Student's / Welch's t-test | Mann-Whitney U |
        | **3+ Groups, Numeric** | One-way ANOVA | Kruskal-Wallis |
        | **2 Groups, Categorical** | Chi-Square / Fisher's Exact | Chi-Square / Fisher's Exact |
        | **3+ Groups, Categorical** | Chi-Square | Chi-Square |
        """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parametric Tests")
        st.caption("Assume data follows a specific distribution (usually Normal). More powerful if assumptions are met.")
        
        st.markdown(r"""
        **1. Student's t-test**
        *   **Use**: Compare means of 2 independent groups.
        *   **Assumptions**: Normality, Equal Variances.
        *   **Effect Size**: Cohen's d.
        
        **2. Welch's t-test**
        *   **Use**: Compare means of 2 independent groups (Robust).
        *   **Assumptions**: Normality (Robust to unequal variances).
        *   **Effect Size**: Cohen's d.
        
        **3. ANOVA (Analysis of Variance)**
        *   **Use**: Compare means of 3+ independent groups.
        *   **Assumptions**: Normality, Equal Variances.
        *   **Effect Size**: Eta Squared ($\eta^2$).
        """)
        
    with col2:
        st.subheader("Non-Parametric Tests")
        st.caption("Do not assume a specific distribution. Use ranks instead of raw values. Robust to outliers.")
        
        st.markdown(r"""
        **1. Mann-Whitney U Test**
        *   **Use**: Compare distributions of 2 independent groups.
        *   **Assumptions**: Independent samples, similar distribution shapes.
        *   **Effect Size**: Rank-Biserial Correlation ($r$).
        
        **2. Kruskal-Wallis Test**
        *   **Use**: Compare distributions of 3+ independent groups.
        *   **Assumptions**: Independent samples, similar distribution shapes.
        *   **Effect Size**: Eta-Squared H ($\eta^2_H$).
        
        **3. Chi-Square Test of Independence**
        *   **Use**: Test association between two categorical variables.
        *   **Assumptions**: Expected cell counts ≥ 5 (use Fisher's Exact otherwise).
        *   **Effect Size**: Cramér's V.
        """)
        
    st.divider()
    st.subheader("Effect Size Interpretation")
    st.markdown(r"""
    **P-value** tells you *if* there is a difference. **Effect Size** tells you *how big* the difference is.
    
    | Test | Effect Size Metric | Small | Medium | Large |
    | :--- | :--- | :---: | :---: | :---: |
    | **T-test** | Cohen's d | 0.2 | 0.5 | 0.8 |
    | **Mann-Whitney** | Rank-Biserial r | 0.1 | 0.3 | 0.5 |
    | **ANOVA** | Eta Squared ($\eta^2$) | 0.01 | 0.06 | 0.14 |
    | **Kruskal-Wallis** | Eta-Squared H ($\eta^2_H$) | 0.01 | 0.06 | 0.14 |
    | **Chi-Square** | Cramér's V | 0.1 | 0.3 | 0.5 |
    | **Fisher's Exact** | Odds Ratio | - | - | - |
    
    > 💡 **Tip**: An Odds Ratio of 1 means no association. OR > 1 indicates increased odds, OR < 1 indicates decreased odds.
    """)
    
    # Key Assumptions Section
    st.divider()
    st.subheader("Key Assumptions to Check")
    st.markdown("""
    | Assumption | Why It Matters | How to Check |
    | :--- | :--- | :--- |
    | **Independence** | Observations must not influence each other | Study design review |
    | **Normality** | Required for parametric tests | Shapiro-Wilk test, Q-Q plot |
    | **Homogeneity of Variance** | Groups should have similar variances | Levene's test |
    | **Sample Size** | n ≥ 30 per group for CLT to apply | Check group sizes |
    
    > ⚠️ **Small samples (n < 30)**: Prefer non-parametric tests or verify normality carefully.
    """)
    
    # NEW: Epidemiological Interpretation Tables
    st.divider()
    st.subheader("Epidemiological Interpretation Guide")
    
    # Tab system for different tables
    epi_tabs = st.tabs(["P-value vs Effect Size", "Type I & II Errors", "Odds Ratio Guide", "Sample Size Rules"])
    
    with epi_tabs[0]:
        st.markdown("### Statistical Significance vs Clinical Significance")
        st.markdown("""
        <table style="width:100%; border-collapse: collapse; text-align: center;">
            <tr style="background-color: #f5f5f5;">
                <th style="border: 1px solid #ddd; padding: 12px;"></th>
                <th style="border: 1px solid #ddd; padding: 12px;" colspan="2">Effect Size</th>
            </tr>
            <tr style="background-color: #f5f5f5;">
                <th style="border: 1px solid #ddd; padding: 12px;">P-value</th>
                <th style="border: 1px solid #ddd; padding: 12px;">Small / Negligible</th>
                <th style="border: 1px solid #ddd; padding: 12px;">Medium / Large</th>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 12px; font-weight: bold;">p < 0.05<br/>(Significant)</td>
                <td style="border: 1px solid #ddd; padding: 12px; background-color: #fff3e0;">
                    <strong>⚠️ Statistically significant</strong><br/>
                    but likely NOT clinically meaningful<br/>
                    <em>→ May be due to large sample size</em>
                </td>
                <td style="border: 1px solid #ddd; padding: 12px; background-color: #c8e6c9;">
                    <strong>✅ Both statistically AND clinically significant</strong><br/>
                    <em>→ Strong evidence for real effect</em>
                </td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 12px; font-weight: bold;">p ≥ 0.05<br/>(Not Significant)</td>
                <td style="border: 1px solid #ddd; padding: 12px; background-color: #e8f5e9;">
                    <strong>✅ True negative</strong><br/>
                    No effect detected, none exists<br/>
                    <em>→ Consistent with null hypothesis</em>
                </td>
                <td style="border: 1px solid #ddd; padding: 12px; background-color: #ffcdd2;">
                    <strong>❌ Possibly underpowered</strong><br/>
                    Effect may exist but study too small<br/>
                    <em>→ Check power, consider larger study</em>
                </td>
            </tr>
        </table>
        """, unsafe_allow_html=True)
        
        st.info("**Key insight**: A small p-value does NOT mean a large effect. Always report both p-value AND effect size!")
    
    with epi_tabs[1]:
        st.markdown("### Type I and Type II Errors (α and β)")
        st.markdown("""
        <table style="width:100%; border-collapse: collapse; text-align: center;">
            <tr style="background-color: #f5f5f5;">
                <th style="border: 1px solid #ddd; padding: 12px;"></th>
                <th style="border: 1px solid #ddd; padding: 12px;" colspan="2">Reality (Truth)</th>
            </tr>
            <tr style="background-color: #f5f5f5;">
                <th style="border: 1px solid #ddd; padding: 12px;">Test Result</th>
                <th style="border: 1px solid #ddd; padding: 12px;">H₀ True (No Effect)</th>
                <th style="border: 1px solid #ddd; padding: 12px;">H₁ True (Effect Exists)</th>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 12px; font-weight: bold;">Reject H₀<br/>(p < α)</td>
                <td style="border: 1px solid #ddd; padding: 12px; background-color: #ffcdd2;">
                    <strong>❌ Type I Error (α)</strong><br/>
                    False Positive<br/>
                    <em>Probability = α (usually 0.05)</em>
                </td>
                <td style="border: 1px solid #ddd; padding: 12px; background-color: #c8e6c9;">
                    <strong>✅ Correct Decision</strong><br/>
                    True Positive (Power = 1-β)<br/>
                    <em>Goal: Power ≥ 0.80</em>
                </td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 12px; font-weight: bold;">Fail to Reject H₀<br/>(p ≥ α)</td>
                <td style="border: 1px solid #ddd; padding: 12px; background-color: #c8e6c9;">
                    <strong>✅ Correct Decision</strong><br/>
                    True Negative<br/>
                    <em>Probability = 1-α</em>
                </td>
                <td style="border: 1px solid #ddd; padding: 12px; background-color: #fff3e0;">
                    <strong>⚠️ Type II Error (β)</strong><br/>
                    False Negative<br/>
                    <em>Missed real effect</em>
                </td>
            </tr>
        </table>
        """, unsafe_allow_html=True)
        
        col_err1, col_err2 = st.columns(2)
        with col_err1:
            st.warning("**Type I (α = 0.05)**: 5% chance of finding an effect when none exists")
        with col_err2:
            st.warning("**Type II (β = 0.20)**: 20% chance of missing a real effect (if Power = 80%)")
    
    with epi_tabs[2]:
        st.markdown("### Odds Ratio (OR) Interpretation Guide")
        st.markdown("""
        | OR Value | Interpretation | Clinical Meaning |
        |:--------:|:---------------|:-----------------|
        | **OR = 1** | No association | Exposure does not affect outcome |
        | **OR > 1** | Positive association | Exposure **increases** odds of outcome |
        | **OR < 1** | Negative association | Exposure **decreases** odds of outcome (protective) |
        | **OR = 2** | 2x higher odds | Exposed have **double** the odds vs unexposed |
        | **OR = 0.5** | 50% lower odds | Exposed have **half** the odds vs unexposed |
        """)
        
        st.markdown("#### Strength of Association (for OR):")
        st.markdown("""
        | OR Range | Strength |
        |:--------:|:---------|
        | 1.0 - 1.5 | Weak |
        | 1.5 - 3.0 | Moderate |
        | 3.0 - 10.0 | Strong |
        | > 10.0 | Very Strong |
        
        > 💡 **Note**: Same thresholds apply for OR < 1 (use 1/OR for comparison)
        """)
        
        st.info("**Confidence Interval**: If 95% CI includes 1, the OR is NOT statistically significant")
    
    with epi_tabs[3]:
        st.markdown("### Minimum Sample Size Recommendations")
        st.markdown("""
        | Analysis Type | Minimum n per group | Notes |
        |:--------------|:-------------------:|:------|
        | **T-test / ANOVA** | 30 | For CLT to apply |
        | **Mann-Whitney / Kruskal** | 15-20 | Non-parametric, more robust |
        | **Chi-Square** | 5 per cell (expected) | Use Fisher's if < 5 |
        | **Correlation** | 30 | For stable estimates |
        | **Regression** | 10-20 per predictor | Rule of thumb |
        """)
        
        st.markdown("#### Effect Size → Required Sample Size (Power = 80%, α = 0.05)")
        st.markdown("""
        | Effect Size (Cohen's d) | n per group (approx) |
        |:-----------------------:|:--------------------:|
        | Small (d = 0.2) | ~400 |
        | Medium (d = 0.5) | ~65 |
        | Large (d = 0.8) | ~25 |
        
        > 📌 **Use the Power & Sample Size tab** for precise calculations based on your data!
        """)

def run_power_analysis(df, analyzer):
    st.header("Power Analysis & Sample Size Calculator")
    st.markdown("Calculate sample size or power for your study design.")
    
    # ============ Epidemiological Context ============
    with st.expander("Understanding Power Analysis", expanded=False):
        st.markdown("""
        ### Why Power Matters in Epidemiology
        
        **Statistical Power** is the probability of detecting a TRUE effect when it exists.
        
        | Term | Symbol | Definition | Goal |
        |:-----|:------:|:-----------|:----:|
        | **Significance Level** | α | Probability of Type I error (false positive) | 0.05 |
        | **Power** | 1-β | Probability of detecting true effect | ≥ 0.80 |
        | **Type II Error** | β | Probability of missing true effect | ≤ 0.20 |
        """)
        
        st.markdown("""
        <table style="width:100%; border-collapse: collapse; text-align: center; margin: 15px 0;">
            <tr style="background-color: #f5f5f5;">
                <th style="border: 1px solid #ddd; padding: 10px;"></th>
                <th style="border: 1px solid #ddd; padding: 10px;">H₀ True (No Effect)</th>
                <th style="border: 1px solid #ddd; padding: 10px;">H₁ True (Effect Exists)</th>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 10px; font-weight: bold;">Reject H₀</td>
                <td style="border: 1px solid #ddd; padding: 10px; background-color: #ffcdd2;">
                    <strong>Type I Error (α=0.05)</strong><br/>False Positive
                </td>
                <td style="border: 1px solid #ddd; padding: 10px; background-color: #c8e6c9;">
                    <strong>True Positive (Power)</strong><br/>Correct Detection
                </td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 10px; font-weight: bold;">Fail to Reject H₀</td>
                <td style="border: 1px solid #ddd; padding: 10px; background-color: #c8e6c9;">
                    <strong>True Negative</strong><br/>Correct Non-Detection
                </td>
                <td style="border: 1px solid #ddd; padding: 10px; background-color: #fff3e0;">
                    <strong>Type II Error (β)</strong><br/>Missed Real Effect
                </td>
            </tr>
        </table>
        """, unsafe_allow_html=True)
        
        st.info("""
        **Key Insights:**
        - **Underpowered study (Power < 80%)**: High risk of missing real effects (Type II error)
        - **Power ≥ 80%**: Standard threshold for adequate studies
        - **Power ≥ 90%**: Recommended for confirmatory/clinical trials
        - **Larger effect size or sample size → Higher power**
        """)

    auto_fill_data = None

    # 1. Import from Analysis Results
    with st.expander("Import from Analysis Results", expanded=False):
        st.caption("Select a result from the 'Univariate Analysis' tab to auto-fill parameters.")
        
        if 'epidemiology_results' in st.session_state and st.session_state['epidemiology_results']:
            results_list = st.session_state['epidemiology_results']
            # Create options
            options = {f"{r['Variable']} ({r['Test Used']})": r for r in results_list}
            selected_import = st.selectbox("Select Result to Import", ["None"] + list(options.keys()))
            
            if selected_import != "None":
                res = options[selected_import]
                st.write(f"Selected: **{selected_import}**")
                if 'epidemiology_group_col' in st.session_state:
                    st.write(f"Grouping Variable: **{st.session_state['epidemiology_group_col']}**")
                st.write(f"Effect Size: **{res.get('Effect Size', 'N/A')} ({res.get('Effect Type', 'N/A')})**")
                
                if st.button("Load Parameters into Calculator"):
                    auto_fill_data = res
                    target_test = res['Test Used']
                    try:
                        es_val = float(res.get('Effect Size', 0.5))
                        if np.isnan(es_val): es_val = 0.5
                    except:
                        es_val = 0.5
                    
                    es_val = abs(es_val) # Power analysis uses absolute effect size
                    es_type_res = res.get('Effect Type', 'Unknown')
                    
                    # Extract Sample Info
                    sample_info = res.get('Sample Info', {})
                    n1_val = int(sample_info.get('N1', 30))
                    ratio_val = float(sample_info.get('Ratio', 1.0))
                    k_val = int(sample_info.get('k_Groups', 3))
                    n_cat_val = int(sample_info.get('N_Cats', 2))
                    n_total_val = int(sample_info.get('N_Total', 100))
                    min_dim_val = int(sample_info.get('Min_Dim', 1))
                    
                    if target_test == "Student's t-test" or target_test == "Welch's t-test":
                        # Map Welch to T-test for power approx
                        target_test = "T-Test (Independent)"
                        st.session_state['es_ttest'] = es_val
                        st.session_state['ratio_ttest'] = ratio_val
                        st.session_state['n1_ttest'] = n1_val
                        
                    elif target_test == "Mann-Whitney U":
                        target_test = "Mann-Whitney U (Non-parametric)"
                        if es_type_res == "Rank-Biserial r":
                            st.session_state['es_type_mwu'] = "Rank-Biserial r"
                            st.session_state['mwu_p_xy'] = (es_val + 1) / 2
                            st.session_state['es_input_Rank-Biserial r'] = es_val
                        elif es_type_res == "Cohen's d":
                            st.session_state['es_type_mwu'] = "Cohen's d (approx)"
                            st.session_state['mwu_p_xy'] = stats.norm.cdf(es_val / np.sqrt(2))
                            st.session_state["es_input_Cohen's d (approx)"] = es_val
                        else:
                            st.session_state['es_type_mwu'] = "P(X < Y)" # Default
                            st.session_state['mwu_p_xy'] = es_val
                            st.session_state['es_input_P(X < Y)'] = es_val
                            
                        st.session_state['ratio_mwu'] = ratio_val
                        st.session_state['n1_mwu'] = n1_val
                        
                    elif target_test == "ANOVA":
                        target_test = "ANOVA (One-way)"
                        if es_type_res == "Eta Squared":
                            st.session_state['es_type_anova'] = "Eta Squared"
                        else:
                            st.session_state['es_type_anova'] = "Cohen's f"
                        st.session_state['es_anova'] = es_val
                        st.session_state['k_anova'] = k_val
                        st.session_state['n_per_group_anova'] = int(n_total_val / k_val) if k_val else 20
                        
                    elif target_test == "Kruskal-Wallis":
                        target_test = "Kruskal-Wallis (Non-parametric)"
                        if es_type_res == "η²H" or es_type_res == "Epsilon Squared":
                            st.session_state['es_type_kw'] = "Epsilon Squared"
                        else:
                            st.session_state['es_type_kw'] = "Cohen's f equivalent"
                        st.session_state['es_kw'] = es_val
                        st.session_state['k_kw'] = k_val
                        st.session_state['n_per_group_kw'] = int(n_total_val / k_val) if k_val else 20
                        
                    elif target_test == "Chi-Square" or target_test == "Fisher's Exact":
                        target_test = "Chi-Square"
                        if es_type_res == "Cramer's V":
                            st.session_state['es_type_chi2'] = "Cramer's V"
                            st.session_state['chi2_w'] = es_val * np.sqrt(min_dim_val)
                        elif es_type_res == "Odds Ratio":
                            st.session_state['es_type_chi2'] = "Cohen's w"
                            st.session_state['chi2_w'] = 0.3
                            st.toast("Odds Ratio cannot be directly converted to Cohen's w. Defaulted to Medium effect (0.3). Please adjust manually.")
                        else:
                            st.session_state['es_type_chi2'] = "Cohen's w"
                            st.session_state['chi2_w'] = es_val
                        
                        st.session_state['n_cat_chi2'] = n_cat_val
                        st.session_state['n_total_chi2'] = n_total_val
                        st.session_state['min_dim_chi2'] = min_dim_val
                    
                    # Set the test type dropdown - use the selectbox key directly
                    st.session_state['power_test_type'] = target_test
                    st.session_state['power_test_type_selector'] = target_test  # This is the actual selectbox key
                    
                    # Auto-fill Grouping Variable (Factor)
                    if 'epidemiology_group_col' in st.session_state:
                         g_col = st.session_state['epidemiology_group_col']
                         # Basic validation to ensure it matches the options logic downstream (< 20 unique values)
                         if g_col in df.columns and df[g_col].nunique() < 20:
                             st.session_state['power_group_col'] = g_col
                             # Clear selected groups to force a refresh/reset based on the new group column
                             st.session_state.pop('power_selected_groups', None)

                    st.success("Parameters loaded! The calculator fields below have been updated.")
                    st.rerun()  # Refresh UI to show updated values

    # 2. Auto-fill from Dataset (Group Structure)
    with st.expander("Auto-fill Group Structure from Dataset", expanded=True):
        st.caption("Select a grouping variable from your current dataset to auto-fill sample sizes and group counts.")
        
        # Filter for categorical variables suitable for grouping
        potential_groupers = [c for c in analyzer.categorical_cols + analyzer.binary_cols + analyzer.low_cardinality_numeric_cols if df[c].nunique() < 20]
        potential_groupers = sorted(list(set(potential_groupers))) # Ensure unique and sorted
        
        # Try to pre-select the one from session state if available
        prev_group = st.session_state.get('epidemiology_group_col', None)
        default_idx = potential_groupers.index(prev_group) if prev_group in potential_groupers else 0
        
        # Check for change manually to trigger updates
        if 'last_power_group_col' not in st.session_state:
            st.session_state['last_power_group_col'] = None

        group_col_power = st.selectbox(
            "Select Grouping Variable", 
            options=["None"] + potential_groupers,
            index=default_idx + 1 if potential_groupers else 0,
            key="power_group_col"
        )
        
        # Group Filtering (New)
        selected_groups_power = []
        if group_col_power != "None":
            unique_groups = sorted(df[group_col_power].dropna().unique())
            selected_groups_power = st.multiselect(
                "Groups to Include (Power Analysis)",
                options=unique_groups,
                default=unique_groups,
                help="Deselect groups to exclude them from the structure calculation.",
                key="power_selected_groups"
            )
        
        # Track changes to trigger updates
        if 'last_power_selected_groups' not in st.session_state:
            st.session_state['last_power_selected_groups'] = []

        has_group_changed = group_col_power != st.session_state['last_power_group_col']
        has_selection_changed = selected_groups_power != st.session_state['last_power_selected_groups']

        # Trigger update if changed
        if has_group_changed or has_selection_changed:
            st.session_state['last_power_group_col'] = group_col_power
            st.session_state['last_power_selected_groups'] = selected_groups_power
            
            if group_col_power != "None" and len(selected_groups_power) >= 2:
                # Filter data
                df_filtered = df[df[group_col_power].isin(selected_groups_power)]
                counts = df_filtered[group_col_power].value_counts().sort_index()
                
                k_groups_data = len(counts)
                n_total_valid = counts.sum()
                
                # Update Session State Keys for Inputs
                if k_groups_data == 2:
                    n1 = counts.iloc[0]
                    n2 = counts.iloc[1]
                    ratio = n2/n1
                    
                    st.session_state['n1_ttest'] = int(n1)
                    st.session_state['ratio_ttest'] = float(ratio)
                    st.session_state['n1_mwu'] = int(n1)
                    st.session_state['ratio_mwu'] = float(ratio)
                    
                elif k_groups_data > 2:
                    n_per_group = int(n_total_valid / k_groups_data)
                    st.session_state['k_anova'] = int(k_groups_data)
                    st.session_state['n_per_group_anova'] = n_per_group
                    st.session_state['k_kw'] = int(k_groups_data)
                    st.session_state['n_per_group_kw'] = n_per_group
                
                # Chi-Square defaults
                st.session_state['n_total_chi2'] = int(n_total_valid)
                st.session_state['n_cat_chi2'] = max(2, k_groups_data) 

        dataset_params = {}
        if group_col_power != "None" and len(selected_groups_power) >= 2:
            df_filtered = df[df[group_col_power].isin(selected_groups_power)]
            counts = df_filtered[group_col_power].value_counts().sort_index()
            k_groups_data = len(counts)
            n_total_valid = counts.sum()
            
            st.markdown(f"**Structure of '{group_col_power}' (Filtered):**")
            st.write(counts.to_frame("Count").T)
            st.caption(f"Total Valid N: {n_total_valid} | Groups: {k_groups_data}")
            
            dataset_params['k_groups'] = k_groups_data
            dataset_params['n_total'] = n_total_valid
            
            if k_groups_data == 2:
                n1 = counts.iloc[0]
                n2 = counts.iloc[1]
                dataset_params['n1'] = n1
                dataset_params['ratio'] = n2/n1
            elif k_groups_data > 2:
                    dataset_params['n_per_group'] = n_total_valid / k_groups_data
        elif group_col_power != "None":
            st.warning("Select at least 2 groups to calculate structure.")

    # Determine defaults based on auto_fill_data AND dataset_params
    default_effect_size = 0.5
    default_ratio = 1.0
    default_nobs1 = 30
    default_k_groups = 3
    default_n_cat = 2
    default_n_total = 100
    
    # Priority 1: Dataset Params for Structure (N, Ratio, k)
    if dataset_params:
        if 'n1' in dataset_params:
            default_nobs1 = int(dataset_params['n1'])
            default_ratio = dataset_params['ratio']
        if 'k_groups' in dataset_params:
            default_k_groups = int(dataset_params['k_groups'])
            default_n_total = int(dataset_params['n_total'])
        
    # Priority 2: Analysis Results (Overrides Structure if specific test/sample info provided, and sets Effect Size)
    if auto_fill_data:
        if not np.isnan(auto_fill_data.get('Effect Size', np.nan)):
            default_effect_size = abs(auto_fill_data['Effect Size'])
        
        sample_info = auto_fill_data.get('Sample Info', {})
        if 'N1' in sample_info:
            default_nobs1 = int(sample_info['N1'])
            default_ratio = sample_info['Ratio']
        if 'k_Groups' in sample_info:
            default_k_groups = int(sample_info['k_Groups'])
            default_n_total = int(sample_info['N_Total'])
        if 'N_Cats' in sample_info:
            default_n_cat = int(sample_info['N_Cats'])
            default_n_total = int(sample_info['N_Total'])

    col1, col2 = st.columns(2)
    with col1:
        # Use session state for test type if set by import
        test_type_idx = 0
        if 'power_test_type' in st.session_state:
            try:
                test_type_idx = ["T-Test (Independent)", "Mann-Whitney U (Non-parametric)", "ANOVA (One-way)", "Kruskal-Wallis (Non-parametric)", "Chi-Square"].index(st.session_state['power_test_type'])
            except:
                pass
                
        test_type = st.selectbox(
            "Statistical Test", 
            ["T-Test (Independent)", "Mann-Whitney U (Non-parametric)", "ANOVA (One-way)", "Kruskal-Wallis (Non-parametric)", "Chi-Square"], 
            index=test_type_idx,
            key='power_test_type_selector' # Use different key to avoid conflict or handle sync manually
        )
        # Sync selector with session state if needed, but simpler to just let user select
    with col2:
        calc_mode = st.selectbox("Calculation Goal", ["Calculate Required Sample Size", "Calculate Achieved Power"])
        
    st.divider()
    
    # Common Inputs
    c1, c2 = st.columns(2)
    with c1:
        alpha = st.number_input("Significance Level (Alpha)", value=0.05, step=0.01, format="%.2f", help="Probability of Type I error (False Positive). Usually 0.05.")
    
    if test_type == "T-Test (Independent)":
        st.subheader("T-Test Parameters")
        st.caption("Effect Size: Cohen's d (Small=0.2, Medium=0.5, Large=0.8)")
        
        with c2:
            effect_size = st.number_input("Effect Size (Cohen's d)", value=float(default_effect_size), step=0.001, format="%.3f", key="es_ttest")
        
        ratio = st.number_input("Ratio of Sample Sizes (n2/n1)", value=float(default_ratio), step=0.1, help="1.0 for equal group sizes.", key="ratio_ttest")
        
        analysis = TTestIndPower()
        
        if calc_mode == "Calculate Required Sample Size":
            power = st.slider("Desired Power", 0.5, 0.99, 0.80)
            if st.button("Calculate Sample Size"):
                nobs = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=ratio)
                st.success(f"Required Sample Size (Group 1): **{np.ceil(nobs):.0f}**")
                st.info(f"Total Sample Size: **{np.ceil(nobs) + np.ceil(nobs*ratio):.0f}**")
                st.caption("Calculation based on Cohen's d and non-central t-distribution (Cohen, 1988).")
                
                # Plot Power Curve
                st.subheader("Power Curve")
                max_n1 = max(100, int(nobs * 2))
                n1_range = np.linspace(max(10, int(nobs*0.1)), max_n1, 50)
                powers = [analysis.solve_power(effect_size=effect_size, nobs1=n, alpha=alpha, ratio=ratio) for n in n1_range]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(n1_range, powers, linewidth=2, label='Group 1 (n1)', color='blue')
                ax.plot(nobs, power, 'b*', markersize=15, label=f'Required n1={nobs:.0f}')
                ax.axhline(power, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlabel("Sample Size (Group 1)")
                ax.set_ylabel("Power")
                ax.set_title(f"Power Curve (d={effect_size})")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
        else: # Calculate Power
            nobs1 = st.number_input("Sample Size (Group 1)", value=int(default_nobs1), step=1, key="n1_ttest", help="Number of observations in the first group.")
            if st.button("Calculate Power"):
                power_val = analysis.solve_power(effect_size=effect_size, nobs1=nobs1, alpha=alpha, ratio=ratio)
                st.success(f"Achieved Power: **{power_val:.4f}** ({power_val*100:.1f}%)")

    elif test_type == "Mann-Whitney U (Non-parametric)":
        st.subheader("Mann-Whitney U Parameters")
        st.info("ℹ️ Uses Noether's formula (1987) based on the probability P(X < Y).")
        
        with c2:
            if 'mwu_p_xy' not in st.session_state: st.session_state['mwu_p_xy'] = 0.64
            es_type = st.selectbox("Effect Size Input", ["P(X < Y)", "Rank-Biserial r", "Cohen's d (approx)"], key="es_type_mwu")
            
            if es_type == "P(X < Y)":
                display_val = st.session_state['mwu_p_xy']
                min_v, max_v = 0.0, 1.0
            elif es_type == "Rank-Biserial r":
                display_val = 2 * st.session_state['mwu_p_xy'] - 1
                min_v, max_v = -1.0, 1.0
            else:
                p_safe = np.clip(st.session_state['mwu_p_xy'], 0.001, 0.999)
                display_val = np.sqrt(2) * stats.norm.ppf(p_safe)
                min_v, max_v = -10.0, 10.0

            val_input = st.number_input(f"Effect Size ({es_type})", value=float(display_val), step=0.001, format="%.3f", min_value=min_v, max_value=max_v, key=f"es_input_{es_type}")
            
            if es_type == "P(X < Y)": st.session_state['mwu_p_xy'] = val_input
            elif es_type == "Rank-Biserial r": st.session_state['mwu_p_xy'] = (val_input + 1) / 2
            else: st.session_state['mwu_p_xy'] = stats.norm.cdf(val_input / np.sqrt(2))
            
            p_xy = st.session_state['mwu_p_xy']
        
        ratio = st.number_input("Ratio of Sample Sizes (n2/n1)", value=float(default_ratio), step=0.1, key="ratio_mwu")
        
        if calc_mode == "Calculate Required Sample Size":
            power = st.slider("Desired Power", 0.5, 0.99, 0.80)
            if st.button("Calculate Sample Size"):
                if abs(p_xy - 0.5) < 0.001:
                    st.error("Effect size is too close to 0.5 (No Effect).")
                else:
                    z_alpha = stats.norm.ppf(1 - alpha/2)
                    z_beta = stats.norm.ppf(power)
                    c = 1 / (1 + ratio)
                    numerator = (z_alpha + z_beta)**2
                    denominator = 12 * c * (1-c) * (p_xy - 0.5)**2
                    N_total = numerator / denominator
                    n1 = N_total * c
                    st.success(f"Required Sample Size (Group 1): **{np.ceil(n1):.0f}**")
                    st.info(f"Total Sample Size: **{np.ceil(N_total):.0f}**")
        else:
            nobs1 = st.number_input("Sample Size (Group 1)", value=int(default_nobs1), step=1, key="n1_mwu")
            if st.button("Calculate Power"):
                c = 1 / (1 + ratio)
                N_total = nobs1 * (1 + ratio)
                z_alpha = stats.norm.ppf(1 - alpha/2)
                z_beta = np.sqrt(N_total * 12 * c * (1-c) * (p_xy - 0.5)**2) - z_alpha
                power_val = stats.norm.cdf(z_beta)
                st.success(f"Achieved Power: **{power_val:.4f}** ({power_val*100:.1f}%)")

    elif test_type == "ANOVA (One-way)":
        st.subheader("ANOVA Parameters")
        st.caption("Effect Size: Cohen's f (Small=0.1, Medium=0.25, Large=0.4)")
        with c2:
            es_type = st.selectbox("Effect Size Input", ["Cohen's f", "Eta Squared"], key="es_type_anova")
            effect_size_input = st.number_input(f"Effect Size ({es_type})", value=float(default_effect_size), step=0.001, format="%.3f", key="es_anova")
            if es_type == "Eta Squared":
                effect_size = np.sqrt(effect_size_input / (1 - effect_size_input)) if effect_size_input < 1 else 0.1
            else:
                effect_size = effect_size_input
        
        k_groups = st.number_input("Number of Groups", value=int(default_k_groups), step=1, min_value=2, key="k_anova")
        analysis = FTestAnovaPower()
        
        if calc_mode == "Calculate Required Sample Size":
            power = st.slider("Desired Power", 0.5, 0.99, 0.80)
            if st.button("Calculate Sample Size"):
                nobs = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, k_groups=k_groups)
                st.success(f"Required Sample Size (per group): **{np.ceil(nobs):.0f}**")
                st.info(f"Total Sample Size: **{np.ceil(nobs) * k_groups:.0f}**")
        else:
            def_per_group = int(default_n_total / default_k_groups) if default_n_total else 20
            nobs_per_group = st.number_input("Sample Size (per group)", value=def_per_group, step=1, key="n_per_group_anova")
            if st.button("Calculate Power"):
                power_val = analysis.solve_power(effect_size=effect_size, nobs=nobs_per_group, alpha=alpha, k_groups=k_groups)
                st.success(f"Achieved Power: **{power_val:.4f}** ({power_val*100:.1f}%)")

    elif test_type == "Kruskal-Wallis (Non-parametric)":
        st.subheader("Kruskal-Wallis Parameters")
        with c2:
            es_type = st.selectbox("Effect Size Input", ["Cohen's f equivalent", "Epsilon Squared"], key="es_type_kw")
            effect_size_input = st.number_input(f"Effect Size ({es_type})", value=float(default_effect_size), step=0.001, format="%.3f", key="es_kw")
            if es_type == "Epsilon Squared":
                effect_size = np.sqrt(effect_size_input / (1 - effect_size_input)) if effect_size_input < 1 else 0.1
            else:
                effect_size = effect_size_input
        
        k_groups = st.number_input("Number of Groups", value=int(default_k_groups), step=1, min_value=2, key="k_kw")
        dist_assumption = st.radio("Underlying Distribution Assumption", ["Normal Distribution (Optimistic)", "Unknown / Heavy-Tailed (Conservative)"])
        adjustment_factor = 1.0 / 0.955 if dist_assumption == "Normal Distribution (Optimistic)" else 1.0 / 0.864
        analysis = FTestAnovaPower()
        
        if calc_mode == "Calculate Required Sample Size":
            power = st.slider("Desired Power", 0.5, 0.99, 0.80)
            if st.button("Calculate Sample Size"):
                nobs_parametric = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, k_groups=k_groups)
                nobs = nobs_parametric * adjustment_factor
                st.success(f"Required Sample Size (per group): **{np.ceil(nobs):.0f}** (Adjusted)")
                st.info(f"Total Sample Size: **{np.ceil(nobs) * k_groups:.0f}**")
        else:
            def_per_group = int(default_n_total / default_k_groups) if default_n_total else 20
            nobs_per_group = st.number_input("Sample Size (per group)", value=def_per_group, step=1, key="n_per_group_kw")
            if st.button("Calculate Power"):
                effective_nobs = nobs_per_group / adjustment_factor
                power_val = analysis.solve_power(effect_size=effect_size, nobs=effective_nobs, alpha=alpha, k_groups=k_groups)
                st.success(f"Achieved Power (Estimated): **{power_val:.4f}** ({power_val*100:.1f}%)")

    elif test_type == "Chi-Square":
        st.subheader("Chi-Square Parameters")
        st.caption("Effect Size: Cohen's w (Small=0.1, Medium=0.3, Large=0.5)")
        with c2:
            if 'chi2_w' not in st.session_state: st.session_state['chi2_w'] = 0.3
            es_type = st.selectbox("Effect Size Input", ["Cohen's w", "Cramer's V"], key="es_type_chi2")
            if 'min_dim_chi2' not in st.session_state: st.session_state['min_dim_chi2'] = 1
            min_dim_input = st.number_input("Min Dimension (min(r-1, c-1))", value=int(st.session_state['min_dim_chi2']), min_value=1, step=1)
            
            if es_type == "Cramer's V": display_val = st.session_state['chi2_w'] / np.sqrt(min_dim_input)
            else: display_val = st.session_state['chi2_w']
            
            effect_size_input = st.number_input(f"Effect Size ({es_type})", value=float(display_val), step=0.001, format="%.3f", key="es_chi2_input")
            
            if es_type == "Cramer's V": st.session_state['chi2_w'] = effect_size_input * np.sqrt(min_dim_input)
            else: st.session_state['chi2_w'] = effect_size_input
            effect_size = st.session_state['chi2_w']
            
        n_cat = st.number_input("Number of Categories (or Cells - 1)", value=int(default_n_cat), step=1, min_value=2, key="n_cat_chi2")
        analysis = GofChisquarePower()
        
        if calc_mode == "Calculate Required Sample Size":
            power = st.slider("Desired Power", 0.5, 0.99, 0.80)
            if st.button("Calculate Sample Size"):
                nobs = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, n_bins=n_cat)
                st.success(f"Required Total Sample Size: **{np.ceil(nobs):.0f}**")
        else:
            nobs = st.number_input("Total Sample Size", value=int(default_n_total), step=1, key="n_total_chi2")
            if st.button("Calculate Power"):
                power_val = analysis.solve_power(effect_size=effect_size, nobs=nobs, alpha=alpha, n_bins=n_cat)
                st.success(f"Achieved Power: **{power_val:.4f}** ({power_val*100:.1f}%)")

    with st.expander("Methodology & References"):
        st.markdown("""
        **1. T-Test (Independent)**: Exact power calculation based on the non-central t-distribution. (Cohen, 1988)
        **2. Mann-Whitney U Test**: Noether's Formula (1987).
        **3. ANOVA (One-Way)**: Exact power calculation based on the non-central F-distribution. (Cohen, 1988)
        **4. Kruskal-Wallis Test**: F-Test approximation with Asymptotic Relative Efficiency (ARE) adjustment. (Lehmann, 1975)
        **5. Chi-Square Test**: Power calculation based on the non-central Chi-Square distribution. (Cohen, 1988)
        """)

