import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from app_pages.home import DataAnalyzer
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
    output = BytesIO()
    with pd.ExcelWriter(output) as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
    return output.getvalue()

def app():
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
                    st.warning(f"⚠️ Log transformation skipped for columns with negative values: {', '.join(skipped_cols)}")
                    
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
        
        st.info(f"ℹ️ **{transformation}** applied to numeric variables for the analysis below.")

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
        st.warning("⚠️ Please select at least 2 groups to perform a comparison.")
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
        label="📥 Download Summary Table (Excel)",
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
        display_df.style.applymap(lambda x: 'background-color: #d4edda' if isinstance(x, (float, int)) and x < 0.05 else '', subset=subset_cols)
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
    
    with st.expander("ℹ️ Methodological Guidance"):
        st.markdown("""
        *   **Effect Size**: Indicates the magnitude of the difference.
            *   *Cohen's d* (T-test): 0.2 (Small), 0.5 (Medium), 0.8 (Large).
            *   *Rank-Biserial r* (Mann-Whitney): 0.1 (Small), 0.3 (Medium), 0.5 (Large).
            *   *Eta Squared* (ANOVA): 0.01 (Small), 0.06 (Medium), 0.14 (Large).
            *   *Epsilon Squared* (Kruskal-Wallis): 0.01 (Small), 0.06 (Medium), 0.14 (Large).
            *   *Cramer's V* (Chi-Square): 0.1 (Small), 0.3 (Medium), 0.5 (Large).
            *   *Odds Ratio* (Fisher): 1 (No association), >1 (Increased odds), <1 (Decreased odds).
        *   **Multiple Testing**: When testing many variables, the chance of finding a "significant" result by random chance increases.
            *   *Bonferroni**: Very conservative. Controls the Family-Wise Error Rate.
            *   *Benjamini-Hochberg**: Controls the False Discovery Rate (FDR). Recommended for exploratory analysis.
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
    st.markdown("""
    **Analysis of Covariance (ANCOVA)** allows you to compare group means while adjusting for the effect of other continuous variables (covariates).
    
    *Model: Target ~ Group + Covariates*
    """)
    
    with st.expander("📊 Model Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            target_col = st.selectbox(
                "Target Variable (Y)",
                options=analyzer.numeric_cols,
                help="The continuous dependent variable."
            )
            
        with col2:
            group_col = st.selectbox(
                "Grouping Variable (Factor)",
                options=[c for c in analyzer.categorical_cols + analyzer.binary_cols if df[c].nunique() < 10],
                help="The main categorical independent variable."
            )
            
        with col3:
            covariates = st.multiselect(
                "Covariates (X)",
                options=[c for c in analyzer.numeric_cols if c != target_col],
                help="Continuous variables to adjust for."
            )
            
    if not target_col or not group_col:
        st.info("Please select a Target and a Grouping variable.")
        return
        
    if st.button("Run ANCOVA"):
        perform_ancova(df, target_col, group_col, covariates)

def perform_ancova(df, target, group, covariates):
    # Prepare data
    cols = [target, group] + (covariates if covariates else [])
    data = df[cols].dropna()
    
    if data.empty:
        st.error("No valid data after dropping missing values.")
        return

    try:
        # Sanitize column names for statsmodels formula (remove spaces, special chars)
        # We create a mapping to rename columns temporarily
        safe_cols = {col: re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in cols}
        data_safe = data.rename(columns=safe_cols)
        
        target_safe = safe_cols[target]
        group_safe = safe_cols[group]
        covariates_safe = [safe_cols[c] for c in covariates]
        
        # Construct Formula
        # Target ~ C(Group) + Covariate1 + Covariate2 ...
        formula = f"{target_safe} ~ C({group_safe}) + {' + '.join(covariates_safe)}"
        
        st.write(f"**Model Formula:** `{formula}`")
        
        # Fit OLS Model
        model = smf.ols(formula, data=data_safe).fit()
        
        # Display Summary
        st.subheader("ANCOVA Results (OLS Regression)")
        st.write(model.summary())
        
        # Extract Group Effect P-value
        # The group effect is represented by the C(Group)[T.level] coefficients
        # To get a single p-value for the Factor "Group", we run an ANOVA on the model
        anova_table = sm.stats.anova_lm(model, typ=2)
        st.subheader("ANOVA Table (Type II SS)")
        st.dataframe(anova_table.style.format("{:.4f}"))
        
        # Check significance of the Group variable
        group_row = anova_table.loc[f"C({group_safe})"]
        p_value = group_row["PR(>F)"]
        
        if p_value < 0.05:
            st.success(f"The group effect is **significant** (p = {p_value:.4e}) after adjusting for covariates.")
        else:
            st.info(f"The group effect is **not significant** (p = {p_value:.4f}) after adjusting for covariates.")
            
        # Visualization: Adjusted Means (Partial Residual Plot)
        st.subheader("Visualization: Adjusted Distributions")
        
        # Calculate adjusted values: Y_adj = Y - (Covariate_Effect)
        # Or simply plot the residuals + mean + Group_Effect?
        # Standard approach: Plot the fitted values holding covariates at their mean.
        
        # Create a prediction dataset
        # We want to see the effect of Group, so we vary Group and hold Covariates constant (at mean)
        
        # Actually, for boxplots, we often want "Adjusted Y" for each individual.
        # Y_adj_i = Y_i - (Beta_cov * (X_cov_i - Mean_X_cov))
        # This removes the variability due to covariates deviating from their mean.
        
        data_adj = data.copy()
        y_adj = data_safe[target_safe].copy()
        
        for cov in covariates_safe:
            beta = model.params.get(cov, 0)
            mean_val = data_safe[cov].mean()
            y_adj -= beta * (data_safe[cov] - mean_val)
            
        data_adj['Adjusted Target'] = y_adj
        
        fig = px.box(data_adj, x=group, y='Adjusted Target', color=group, 
                     title=f"Distribution of {target} by {group} (Adjusted for Covariates)")
        st.plotly_chart(fig, width='stretch')
        
        # Residual Diagnostics
        st.subheader("Model Diagnostics")
        col1, col2 = st.columns(2)
        with col1:
            # Q-Q Plot of Residuals
            fig_qq = vu.create_qq_plot(model.resid, title="Q-Q Plot of Model Residuals")
            st.plotly_chart(fig_qq, width='stretch')
        with col2:
            # Residuals vs Fitted
            fig_rvf = px.scatter(x=model.fittedvalues, y=model.resid, 
                                 labels={'x': 'Fitted Values', 'y': 'Residuals'},
                                 title="Residuals vs Fitted")
            fig_rvf.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_rvf, width='stretch')
        
    except Exception as e:
        st.error(f"An error occurred during ANCOVA computation: {e}")
        st.write("Tip: Ensure categorical variables are not encoded as numbers without being declared as factors.")

def visualize_result(df, group_col, target, analyzer):
    is_numeric = target in analyzer.numeric_cols
    
    if is_numeric:
        # Reference Lines Configuration
        with st.expander("📏 Reference Lines & Thresholds"):
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
        stats_df['ci'] = 1.96 * (stats_df['std'] / np.sqrt(stats_df['count']))
        
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
    st.markdown("""
    Standardize your data against a reference population to compare deviations across different variables.
    
    $Z = \\frac{X - \mu_{ref}}{\sigma_{ref}}$
    """)
    
    with st.expander("⚙️ Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Select Variables
            target_vars = st.multiselect(
                "Select Variables to Standardize",
                options=analyzer.numeric_cols,
                default=analyzer.numeric_cols[:5] if analyzer.numeric_cols else [],
                help="Select continuous variables to calculate Z-scores for."
            )
            
        with col2:
            # Reference Method
            ref_method = st.selectbox(
                "Reference Population Method",
                ["Internal Control Group", "Whole Cohort (Standardization)", "Manual Reference Values"],
                help="Define the mean and standard deviation used for Z-score calculation."
            )
            
            ref_group_col = None
            ref_group_val = None
            manual_refs = {}
            
            if ref_method == "Internal Control Group":
                ref_group_col = st.selectbox(
                    "Select Grouping Variable",
                    options=[c for c in analyzer.categorical_cols + analyzer.binary_cols if df[c].nunique() < 20]
                )
                if ref_group_col:
                    ref_group_val = st.selectbox(
                        "Select Control Group Value",
                        options=df[ref_group_col].unique()
                    )
            elif ref_method == "Manual Reference Values":
                st.info("Define Mean and Std Dev for each variable below.")

    if not target_vars:
        st.warning("Please select at least one variable.")
        return

    # Calculate Z-Scores
    z_df = df.copy()
    z_cols = []
    
    ref_stats = {} # Store used mean/std for reporting

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
                mu, sigma = 0, 1 # Fallback
        else: # Manual
            c1, c2 = st.columns(2)
            with c1:
                mu = st.number_input(f"Mean for {var}", value=0.0, key=f"mu_{var}")
            with c2:
                sigma = st.number_input(f"Std Dev for {var}", value=1.0, key=f"sigma_{var}")
        
        ref_stats[var] = {"mean": mu, "std": sigma}
        z_df[col_name] = (df[var] - mu) / sigma

    # Display Reference Stats
    with st.expander("View Reference Statistics Used"):
        st.json(ref_stats)

    # Download Z-Score Data
    st.download_button(
        label="📥 Download Z-Score Data",
        data=to_excel(z_df[list(df.columns) + z_cols]),
        file_name="z_score_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Visualization Section
    st.subheader("Visualization")
    
    viz_type = st.selectbox("Select Visualization", ["Group Comparison (Forest Plot)", "Heatmap", "Individual Profile"])
    
    if viz_type == "Group Comparison (Forest Plot)":
        group_col = st.selectbox(
            "Group by", 
            options=[c for c in analyzer.categorical_cols + analyzer.binary_cols if df[c].nunique() < 20],
            index=0
        )
        
        if group_col:
            # Calculate Mean Z-Score and CI per group
            plot_data = []
            
            groups = z_df[group_col].dropna().unique()
            
            for g in groups:
                g_data = z_df[z_df[group_col] == g]
                for z_col in z_cols:
                    orig_var = z_col.replace("_Z", "")
                    vals = g_data[z_col].dropna()
                    if len(vals) > 1:
                        mean = vals.mean()
                        sem = stats.sem(vals)
                        ci = 1.96 * sem
                        plot_data.append({
                            "Group": g,
                            "Variable": orig_var,
                            "Mean Z": mean,
                            "Lower CI": mean - ci,
                            "Upper CI": mean + ci
                        })
            
            plot_df = pd.DataFrame(plot_data)
            
            if not plot_df.empty:
                st.download_button(
                    label="📥 Download Group Summary (Excel)",
                    data=to_excel(plot_df),
                    file_name="zscore_group_summary.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                fig = go.Figure()
                
                # Color by Group
                colors = px.colors.qualitative.Plotly
                
                for i, g in enumerate(groups):
                    g_plot = plot_df[plot_df["Group"] == g]
                    color = colors[i % len(colors)]
                    
                    fig.add_trace(go.Scatter(
                        x=g_plot["Mean Z"],
                        y=g_plot["Variable"],
                        error_x=dict(
                            type='data',
                            symmetric=False,
                            array=g_plot["Upper CI"] - g_plot["Mean Z"],
                            arrayminus=g_plot["Mean Z"] - g_plot["Lower CI"],
                            visible=True
                        ),
                        mode='markers',
                        name=str(g),
                        marker=dict(color=color, size=10)
                    ))
                
                # Add Reference Lines
                fig.add_vline(x=0, line_width=2, line_dash="solid", line_color="black", annotation_text="Ref Mean")
                fig.add_vline(x=1.96, line_width=1, line_dash="dash", line_color="gray", annotation_text="+1.96 SD")
                fig.add_vline(x=-1.96, line_width=1, line_dash="dash", line_color="gray", annotation_text="-1.96 SD")
                
                fig.update_layout(
                    title=f"Mean Z-Scores by {group_col} (Relative to Reference)",
                    xaxis_title="Z-Score (SD from Reference Mean)",
                    yaxis_title="Variable",
                    height=max(400, len(target_vars) * 50),
                    template="plotly_white"
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.warning("Not enough data to plot.")

    elif viz_type == "Heatmap":
        st.write("Heatmap of Z-Scores (Rows: Samples, Columns: Variables)")
        
        # Limit rows for performance
        if len(z_df) > 1000:
            st.info("Downsampling to 1000 rows for heatmap performance.")
            heatmap_df = z_df.sample(1000)
        else:
            heatmap_df = z_df
            
        # Sort option
        sort_col = st.selectbox("Sort by", ["None"] + z_cols)
        if sort_col != "None":
            heatmap_df = heatmap_df.sort_values(sort_col)
            
        fig = px.imshow(
            heatmap_df[z_cols],
            labels=dict(x="Variable", y="Sample", color="Z-Score"),
            color_continuous_scale="RdBu_r",
            zmin=-3, zmax=3,
            aspect="auto"
        )
        st.plotly_chart(fig, width='stretch')

    elif viz_type == "Individual Profile":
        # Select a sample (row)
        # Assuming there is some ID column or just use index
        sample_id_col = st.selectbox("Select Sample ID Column (Optional)", ["Index"] + analyzer.categorical_cols)
        
        if sample_id_col == "Index":
            sample_idx = st.number_input("Row Index", min_value=0, max_value=len(z_df)-1, value=0, step=1)
            selected_row = z_df.iloc[sample_idx]
            label = f"Row {sample_idx}"
        else:
            sample_val = st.selectbox("Select Sample", z_df[sample_id_col].unique())
            selected_row = z_df[z_df[sample_id_col] == sample_val].iloc[0]
            label = str(sample_val)
            
        # Radar Chart or Bar Chart
        values = [selected_row[c] for c in z_cols]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=target_vars,
            y=values,
            marker_color=['red' if abs(v) > 1.96 else 'blue' for v in values]
        ))
        
        fig.add_hline(y=1.96, line_dash="dash", line_color="red")
        fig.add_hline(y=-1.96, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title=f"Z-Score Profile for {label}",
            yaxis_title="Z-Score",
            yaxis_range=[min(min(values), -3), max(max(values), 3)]
        )
        st.plotly_chart(fig, width='stretch')

def show_educational_content():
    st.header("📚 Statistical Test Guide")
    
    with st.expander("Choosing the Right Test", expanded=True):
        st.markdown("""
        ### Flowchart for Group Comparisons
        
        1.  **Are you comparing 2 Groups or >2 Groups?**
            *   **2 Groups**: Go to Step 2.
            *   **>2 Groups**: Go to Step 3.
            
        2.  **Comparing 2 Groups (e.g., Treatment vs Control)**
            *   *Is the Target Variable Numeric?*
                *   **Yes**: Check Normality.
                    *   **Normal**: Use **Student's t-test** (or Welch's if variances unequal).
                    *   **Not Normal**: Use **Mann-Whitney U Test** (Non-parametric).
            *   *Is the Target Variable Categorical?*
                *   **Yes**: Use **Chi-Square Test** (or Fisher's Exact for small samples).
                
        3.  **Comparing >2 Groups (e.g., Low vs Med vs High Dose)**
            *   *Is the Target Variable Numeric?*
                *   **Yes**: Check Normality.
                    *   **Normal**: Use **ANOVA (One-way)**.
                    *   **Not Normal**: Use **Kruskal-Wallis Test** (Non-parametric).
            *   *Is the Target Variable Categorical?*
                *   **Yes**: Use **Chi-Square Test**.
        """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parametric Tests")
        st.caption("Assume data follows a specific distribution (usually Normal). More powerful if assumptions are met.")
        
        st.markdown("""
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
        
        st.markdown("""
        **1. Mann-Whitney U Test**
        *   **Use**: Compare distributions of 2 independent groups.
        *   **Assumptions**: Independent samples.
        *   **Effect Size**: Rank-Biserial Correlation ($r$).
        
        **2. Kruskal-Wallis Test**
        *   **Use**: Compare distributions of 3+ independent groups.
        *   **Assumptions**: Independent samples.
        *   **Effect Size**: Epsilon Squared ($\epsilon^2$).
        
        **3. Chi-Square Test of Independence**
        *   **Use**: Test association between two categorical variables.
        *   **Assumptions**: Expected cell counts > 5.
        *   **Effect Size**: Cramer's V.
        """)
        
    st.divider()
    st.subheader("Effect Size Interpretation")
    st.markdown("""
    **P-value** tells you *if* there is a difference. **Effect Size** tells you *how big* the difference is.
    
    | Test | Effect Size Metric | Small | Medium | Large |
    | :--- | :--- | :--- | :--- | :--- |
    | **T-test** | Cohen's d | 0.2 | 0.5 | 0.8 |
    | **Mann-Whitney** | Rank-Biserial r | 0.1 | 0.3 | 0.5 |
    | **ANOVA** | Eta Squared ($\eta^2$) | 0.01 | 0.06 | 0.14 |
    | **Kruskal-Wallis** | Epsilon Squared ($\epsilon^2$) | 0.01 | 0.06 | 0.14 |
    | **Chi-Square** | Cramer's V | 0.1 | 0.3 | 0.5 |
    """)

def run_power_analysis(df, analyzer):
    st.header("Power Analysis & Sample Size Calculator")
    st.markdown("Calculate sample size or power for your study design.")

    auto_fill_data = None

    # 1. Import from Analysis Results
    with st.expander("📥 Import from Analysis Results", expanded=False):
        st.caption("Select a result from the 'Univariate Analysis' tab to auto-fill parameters.")
        
        if 'epidemiology_results' in st.session_state and st.session_state['epidemiology_results']:
            results_list = st.session_state['epidemiology_results']
            # Create options
            options = {f"{r['Variable']} ({r['Test Used']})": r for r in results_list}
            selected_import = st.selectbox("Select Result to Import", ["None"] + list(options.keys()))
            
            if selected_import != "None":
                res = options[selected_import]
                st.write(f"**Selected:** {selected_import}")
                st.write(f"Effect Size: {res.get('Effect Size', 'N/A')} ({res.get('Effect Type', 'N/A')})")
                
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
                        if es_type_res == "Epsilon Squared":
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
                            st.toast("⚠️ Odds Ratio cannot be directly converted to Cohen's w. Defaulted to Medium effect (0.3). Please adjust manually.")
                        else:
                            st.session_state['es_type_chi2'] = "Cohen's w"
                            st.session_state['chi2_w'] = es_val
                        
                        st.session_state['n_cat_chi2'] = n_cat_val
                        st.session_state['n_total_chi2'] = n_total_val
                        st.session_state['min_dim_chi2'] = min_dim_val
                    
                    # Set the test type dropdown
                    st.session_state['power_test_type'] = target_test
                    
                    st.success("Values loaded! Check the calculator below.")

    # 2. Auto-fill from Dataset (Group Structure)
    with st.expander("📊 Auto-fill Group Structure from Dataset", expanded=True):
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

