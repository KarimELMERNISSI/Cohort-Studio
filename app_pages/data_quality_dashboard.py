import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
import statsmodels.stats.multitest as smt
import streamlit as st

import utils.analysis_utils as au
import utils.visualization_utils as vu
from explore.data_quality_auditor import DataQualityAuditor


def render_dashboard(df, config):
    """
    Renders the Data Quality Dashboard.
    """
    st.markdown("Comprehensive assessment of your dataset's health across key dimensions.")

    # Initialize session state defaults if not set (first render)
    if 'validity_outlier_method' not in st.session_state:
        st.session_state['validity_outlier_method'] = 'iqr'
    if 'validity_outlier_params' not in st.session_state:
        st.session_state['validity_outlier_params'] = {'multiplier': 1.5}

    # Get selected outlier method from session state (set in Tab 2)
    validity_method = st.session_state['validity_outlier_method']
    validity_params = st.session_state['validity_outlier_params']
    
    print(f"[DEBUG render_dashboard] validity_method={validity_method}, validity_params={validity_params}")

    # Initialize Auditor
    auditor = DataQualityAuditor(df, config=config)
    metrics = auditor.run_audit(validity_method=validity_method, validity_params=validity_params)
    advice_list = auditor.generate_advice()
    
    print(f"[DEBUG render_dashboard] metrics={metrics}")

    # --- Global Score Calculation ---
    with st.expander("⚙️ Score Weights Configuration", expanded=False):
        st.write("Adjust the importance of each metric for the Global Score:")
        weights = {}
        total_weight = 0
        
        # Define preferred order for weights display too
        preferred_order = ["Completeness", "Statistical Validity", "Clinical Validity", "Consistency", "Uniformity", "Uniqueness"]
        ordered_keys = [k for k in preferred_order if k in metrics] + [k for k in metrics if k not in preferred_order]
        
        cols = st.columns(3)
        for i, metric_name in enumerate(ordered_keys):
            # Default weights
            default = 1.0
            if metric_name == "Clinical Validity": default = 2.0 # Higher importance
            
            with cols[i % 3]:
                weights[metric_name] = st.slider(f"{metric_name} Weight", 0.0, 5.0, default, 0.5)
                total_weight += weights[metric_name]
    
    # Calculate Weighted Average
    global_score = 0
    if total_weight > 0:
        for name, score in metrics.items():
            global_score += score * weights[name]
        global_score /= total_weight
    
    # --- Global Score Gauge ---
    col_gauge, col_kpi = st.columns([1, 2])
    
    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = global_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Global Quality Score"},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#ff4b4b'},
                    {'range': [50, 80], 'color': '#ffa421'},
                    {'range': [80, 100], 'color': '#21c354'}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': global_score}}))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, width='stretch')

    with col_kpi:
        st.subheader("Health Scorecard")
        
        # Reorder metrics for display
        # Desired order: Completeness, Statistical Validity, Clinical Validity, Consistency, Uniqueness
        ordered_metrics = {k: metrics[k] for k in preferred_order if k in metrics}
        # Add any others not in preferred list
        for k, v in metrics.items():
            if k not in ordered_metrics:
                ordered_metrics[k] = v
        
        # Display metrics in a grid with containers for better look
        cols = st.columns(3) # 3 columns grid
        for i, (name, value) in enumerate(ordered_metrics.items()):
            with cols[i % 3]:
                with st.container(border=True):
                    # Custom Metric with Red/Orange/Green scale
                    if value >= 80:
                        color = "#21c354" # Green
                    elif value >= 50:
                        color = "#ffa421" # Orange
                    else:
                        color = "#ff4b4b" # Red
                    
                    delta_val = value - 100
                    delta_str = f"{delta_val:.1f}%" if value < 100 else "Perfect"
                    arrow = "↓" if value < 100 else ""
                    
                    st.markdown(f"""
                        <div style="display: flex; flex-direction: column;">
                            <span style="font-size: 0.875rem; opacity: 0.8;">{name}</span>
                            <span style="font-size: 2rem; font-weight: 600; line-height: 1.2;">{value}%</span>
                            <span style="font-size: 0.875rem; color: {color};">
                                {arrow} {delta_str}
                            </span>
                        </div>
                    """, unsafe_allow_html=True)
        
        st.info(f"**Global Score Interpretation:** Your dataset has a quality score of **{global_score:.1f}/100** based on your custom weights. "
                f"{'Excellent! Ready for analysis.' if global_score >= 90 else 'Needs improvement before reliable analysis.'}")

    st.divider()

    # --- 2. Actionable Advice ---
    if advice_list:
        with st.expander("💡 AI-Driven Recommendations & Best Practices", expanded=True):
            
            # Split into two columns: Specific Advice vs General Best Practices
            col_advice, col_best_practices = st.columns([1.5, 1])
            
            with col_advice:
                st.markdown("### 🛠️ Recommended Actions")
                for item in advice_list:
                    severity_icon = {
                        "high": "🔴",
                        "medium": "🟠",
                        "low": "🟡",
                        "success": "🟢"
                    }.get(item['severity'], "⚪")
                    
                    with st.container(border=True):
                        st.markdown(f"**{severity_icon} {item['category']}**")
                        st.markdown(item['message'])
            
            with col_best_practices:
                st.markdown("### 🏆 Data Quality Best Practices")
                st.info("""
                **1. Completeness**
                Aim for >95%. Use *MICE* or *MissForest* imputation for critical variables instead of dropping rows.
                
                **2. Validity**
                Investigate all outliers. Are they errors or rare biological events? Use the *Outlier Detection* module to filter or cap them.
                
                **3. Consistency**
                Ensure dates are actual Date objects and numbers are Numeric. This prevents analysis errors later.
                
                **4. Clinical Validity**
                This is your most important metric. Ensure your data respects the biological rules defined in *Data Validation*.
                """)

    st.divider()

    # --- 3. Detailed Analysis Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧩 Completeness (Missing)", 
        "📉 Validity (Outliers)", 
        "🏥 Clinical Validity",
        "🔍 Conformity"
    ])

    # --- Tab 1: Completeness ---
    with tab1:
        st.subheader("Missing Data Analysis")
        
        # 1. Basic Counts (Existing)
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
        
        if not missing_counts.empty:
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = px.bar(
                    x=missing_counts.index, 
                    y=missing_counts.values,
                    labels={'x': 'Column', 'y': 'Missing Count'},
                    title="Missing Values per Column",
                    color=missing_counts.values,
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                st.dataframe(
                    missing_counts.rename("Count").to_frame().assign(
                        Percentage=lambda x: (x['Count'] / len(df) * 100).round(2)
                    ),
                    width='stretch'
                )
                
            st.divider()
            
            # --- Advanced Analysis ---
            st.markdown("### 🕵️ Advanced Diagnosis")
            
            diag_tabs = st.tabs(["👁️ Visual Diagnosis", "🧪 Statistical Diagnosis (MCAR & MAR)"])
            
            # A. Visual Diagnosis
            with diag_tabs[0]:
                st.markdown("**1. Nullity Matrix** (Visualizing the pattern of missing data)")
                st.info("This matrix visualizes missing values (black) across rows (samples) and columns (variables).\n"
                        "- **Clustering**: Groups similar samples and variables together to reveal patterns (e.g., blocks of missing data).\n"
                        "- **Interpretation**: Vertical bands indicate variables with high missingness. Horizontal bands indicate incomplete samples.")

                # Options
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                with c1:
                    # Allow user to select a column to use as row identifier
                    id_options = ["Index"] + df.columns.tolist()
                    row_id_col = st.selectbox("Row Label (ID)", options=id_options, index=0, help="Select a column to use as the row label.")
                with c2:
                    cluster_cols_matrix = st.checkbox("Cluster Variables (Cols)", value=True, key="nm_cluster_cols")
                with c3:
                    cluster_rows_matrix = st.checkbox("Cluster Samples (Rows)", value=False, key="nm_cluster_rows")
                with c4:
                    show_dendro_matrix = st.checkbox("Show Dendrograms", value=False, key="nm_show_dendro")
                with c5:
                    show_row_labels = st.checkbox("Show Row Labels", value=True, key="nm_show_row_labels")
                with c6:
                    show_col_labels = st.checkbox("Show Column Labels", value=True, key="nm_show_col_labels")
                
                # Row Filtering
                st.markdown("##### 🔍 Filter Rows by Missingness")
                row_missing_pct = df.isnull().mean(axis=1) * 100
                min_miss, max_miss = st.slider(
                    "Filter Rows by % Missing",
                    min_value=0.0,
                    max_value=100.0,
                    value=(0.0, 100.0),
                    step=1.0,
                    help="Show only rows where the percentage of missing values falls within this range."
                )
                
                # Filter Data
                mask = (row_missing_pct >= min_miss) & (row_missing_pct <= max_miss)
                filtered_df = df[mask]
                
                if filtered_df.empty:
                    st.warning("No rows match the selected missingness range.")
                else:
                    st.caption(f"Showing {len(filtered_df)} out of {len(df)} rows ({len(filtered_df)/len(df)*100:.1f}%).")

                    # --- PLOTTING ---
                    # Use modular function from visualization_utils
                    try:
                        fig_matrix = vu.plot_nullity_matrix(
                            df=filtered_df,
                            row_id_col=row_id_col,
                            cluster_cols=cluster_cols_matrix,
                            cluster_rows=cluster_rows_matrix,
                            show_dendrograms=show_dendro_matrix,
                            show_row_labels=show_row_labels,
                            show_col_labels=show_col_labels,
                            figsize=(18, 12)
                        )
                        st.plotly_chart(fig_matrix, width='stretch')
                    except Exception as e:
                        st.error(f"Error generating Nullity Matrix: {e}")
                        
                    # --- Distribution Plot ---
                    st.markdown("##### 📊 Distribution of Row Missingness")
                    # Create a histogram of row missingness
                    # We use the FULL dataframe for context, or filtered? 
                    # User asked for "complement", usually helpful to see where the data lies.
                    # Let's show the filtered distribution but maybe with a reference?
                    # Simple histogram of the filtered data seems most appropriate for "what am I looking at".
                    
                    fig_dist = px.histogram(
                        row_missing_pct[mask],
                        x=row_missing_pct[mask],
                        nbins=20,
                        labels={'x': '% Missing per Row', 'y': 'Count of Rows'},
                        title=f"Distribution of Missingness (Filtered Rows: {min_miss}% - {max_miss}%)",
                        color_discrete_sequence=['#636EFA']
                    )
                    fig_dist.update_layout(bargap=0.1)
                    st.plotly_chart(fig_dist, width='stretch')
                # Compute correlation from the boolean matrix directly
                # This is much faster and more robust
                # Re-calculate nullity_df as it is needed here (was local to plot_nullity_matrix)
                nullity_df = df.isnull().astype(int)
                corr_matrix = nullity_df.corr()
                
                # Filter out columns with no missing values (variance is 0)
                # In the boolean matrix, if a col is all 0s (no missing) or all 1s (all missing), std is 0.
                # corr() returns NaN for these.
                
                # Options
                c1, c2, c3 = st.columns(3)
                with c1:
                    filter_complete = st.checkbox("Filter Complete Columns", value=True, help="Exclude columns with no missing values.")
                with c2:
                    enable_clustering = st.checkbox("Cluster Variables", value=True, help="Group variables with similar missingness patterns.")
                with c3:
                    show_dendrogram = st.checkbox("Show Dendrogram", value=False, disabled=not enable_clustering, help="Display the hierarchical clustering dendrogram.")

                # Filter logic
                if filter_complete:
                    # Drop columns that are all 0 (no missing values)
                    # We can check the original nullity_df sum
                    cols_with_missing = nullity_df.columns[nullity_df.sum() > 0]
                    if len(cols_with_missing) > 0:
                        corr_matrix = corr_matrix.loc[cols_with_missing, cols_with_missing]
                
                if not corr_matrix.empty and not corr_matrix.isna().all().all():
                    # 1. Clustering for Ordering
                    if enable_clustering and len(corr_matrix) > 2:
                        try:
                            # Use 1 - correlation as distance
                            dist_matrix = 1 - corr_matrix.fillna(0).abs()
                            # Hierarchical clustering
                            linkage = sch.linkage(sch.distance.squareform(dist_matrix), method='average')
                            
                            # Show Dendrogram if requested
                            if show_dendrogram:
                                fig_dendro = ff.create_dendrogram(corr_matrix, linkagefun=lambda x: linkage, labels=corr_matrix.columns)
                                fig_dendro.update_layout(height=400, title="Hierarchical Clustering Dendrogram", margin=dict(b=100))
                                st.plotly_chart(fig_dendro, width='stretch')

                            # Get sorted indices
                            new_order_idx = sch.leaves_list(linkage)
                            new_order = corr_matrix.columns[new_order_idx]
                            # Reorder matrix
                            corr_matrix = corr_matrix.loc[new_order, new_order]
                        except Exception as e:
                            st.warning(f"Could not cluster variables: {e}")

                    # 2. Triangular Mask
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    masked_corr = corr_matrix.where(~mask, np.nan)
                    
                    # 3. Visualization
                    # Dynamic height based on number of variables
                    heatmap_height = max(400, len(corr_matrix) * 20)
                    
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=masked_corr.values,
                        x=masked_corr.columns,
                        y=masked_corr.index,
                        colorscale='RdBu_r',
                        zmin=-1, zmax=1,
                        xgap=1, ygap=1,
                        hoverongaps=False,
                        hovertemplate='Variable X: %{x}<br>Variable Y: %{y}<br>Correlation: %{z:.2f}<extra></extra>'
                    ))
                    
                    fig_corr.update_layout(
                        title="Nullity Correlation Matrix" + (" (Clustered)" if enable_clustering else ""),
                        height=heatmap_height,
                        xaxis_showgrid=False,
                        yaxis_showgrid=False,
                        yaxis_autorange='reversed', # Match matrix convention
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_corr, width='stretch')
                else:
                    st.info("Not enough missing data variation to compute correlations.")


            # B. Statistical Diagnosis (MCAR)
            with diag_tabs[1]:
                st.markdown("#### Little's MCAR Test")
                st.markdown("""
                **Little's Missing Completely At Random (MCAR) Test** checks if the missingness pattern is random.
                
                - **H0 (Null Hypothesis):** Data is Missing Completely at Random (MCAR).
                - **H1 (Alternative Hypothesis):** Data is Not MCAR (likely MAR or MNAR).
                
                *Note: This test requires numerical data and assumes multivariate normality.*
                """)
                
                # Variable Selection
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if not numeric_cols:
                    st.warning("No numerical variables found in the dataset. Little's MCAR test requires numerical data.")
                else:
                    with st.expander("⚙️ Test Configuration", expanded=True):
                        mcar_vars = st.multiselect(
                            "Select Variables for MCAR Test",
                            options=numeric_cols,
                            default=numeric_cols[:min(10, len(numeric_cols))], # Default to top 10 to avoid performance issues
                            help="Select numerical variables to include in the test. Including too many variables may reduce performance."
                        )
                    
                    if st.button("Run Little's MCAR Test"):
                        if not mcar_vars:
                            st.error("Please select at least one variable.")
                        else:
                            with st.spinner("Running Little's MCAR Test..."):
                                # Run Test
                                mcar_result = au.littles_mcar_test(df[mcar_vars])
                                
                                if "error" in mcar_result:
                                    st.error(mcar_result["error"])
                                else:
                                    # Display Results
                                    st.markdown("### Test Results")
                                    
                                    c1, c2, c3 = st.columns(3)
                                    with c1:
                                        st.metric("Chi-Square Statistic", f"{mcar_result['statistic']:.2f}")
                                    with c2:
                                        p_val = mcar_result['p_value']
                                        st.metric("P-Value", f"{p_val:.4f}", delta="Significant" if p_val < 0.05 else "Not Significant", delta_color="inverse")
                                    with c3:
                                        st.metric("Conclusion", mcar_result['result'])
                                    
                                    # Detailed Interpretation
                                    if mcar_result['p_value'] < 0.05:
                                        st.error("""
                                        **Result: Likely Not MCAR** (p < 0.05)
                                        
                                        The test rejects the null hypothesis. The missingness pattern appears to be systematic (MAR or MNAR).
                                        You should investigate potential dependencies using the **MAR Indicator** tab.
                                        """)
                                    else:
                                        st.success("""
                                        **Result: Likely MCAR** (p >= 0.05)
                                        
                                        The test fails to reject the null hypothesis. There is no strong evidence against the data being Missing Completely at Random.
                                        """)
                st.markdown("#### Missing At Random (MAR) Indicator")
                st.markdown("Checks if the missingness of a specific variable is influenced by the values of other variables.")
                
                mar_mode = st.radio("Analysis Mode", ["Single Variable Analysis", "Global Dependency Scan"], horizontal=True)

                if mar_mode == "Single Variable Analysis":
                    target_col = st.selectbox("Select Variable to Analyze (Target)", options=missing_counts.index, key="mar_target")
                    
                    if target_col:
                        # Create temporary dataframe for analysis
                        temp_df = df.copy()
                        temp_df['Missingness_Status'] = temp_df[target_col].isnull().map({True: 'Missing', False: 'Observed'})
                        
                        # Configuration
                        with st.expander("⚙️ Analysis Configuration", expanded=True):
                            # Predictors
                            all_cols = [c for c in df.columns if c != target_col and c != 'Missingness_Status']
                            predictors = st.multiselect("Select Predictor Variables", options=all_cols, default=all_cols[:5] if len(all_cols) > 5 else all_cols)
                            
                            # Test Preference
                            c1, c2 = st.columns(2)
                            with c1:
                                test_pref = st.selectbox("Test Preference", ["Auto-Detect", "Force Parametric", "Force Non-Parametric"])
                            with c2:
                                correction = st.selectbox("Multiple Testing Correction", ["None", "Bonferroni", "Benjamini-Hochberg (FDR)"], index=2)
    
                        if st.button("Run Dependency Check"):
                            results = []
                            progress = st.progress(0)
                            
                            # Identify col types for analyzer
                            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                            binary_cols = [c for c in df.columns if df[c].nunique() == 2]
                            
                            for i, pred in enumerate(predictors):
                                # Determine manual test
                                manual_test = "Auto-Detect"
                                if test_pref == "Force Parametric":
                                    manual_test = "Student's t-test" if pred in numeric_cols else "Chi-Square"
                                elif test_pref == "Force Non-Parametric":
                                    manual_test = "Mann-Whitney U" if pred in numeric_cols else "Chi-Square"
                                    
                                # Call analyze_variable
                                try:
                                    res = au.analyze_variable(
                                        temp_df, 
                                        'Missingness_Status', 
                                        pred, 
                                        numeric_cols, 
                                        categorical_cols, 
                                        binary_cols, 
                                        manual_test=manual_test
                                    )
                                    results.append(res)
                                except Exception as e:
                                    st.error(f"Error analyzing {pred}: {e}")
                                    
                                progress.progress((i + 1) / len(predictors))
                                
                            progress.empty()
                            
                            # Display Results
                            if results:
                                res_df = pd.DataFrame(results)
                                
                                # Correction
                                if correction != "None":
                                    p_values = res_df["P-Value"].fillna(1.0).values
                                    if correction == "Bonferroni":
                                        reject, pvals_corrected, _, _ = smt.multipletests(p_values, method='bonferroni')
                                    elif correction == "Benjamini-Hochberg (FDR)":
                                        reject, pvals_corrected, _, _ = smt.multipletests(p_values, method='fdr_bh')
                                    
                                    res_df["P-Value (Adj)"] = pvals_corrected
                                    res_df["Significant"] = ["Yes" if p < 0.05 else "No" for p in pvals_corrected]
                                else:
                                    res_df["Significant"] = ["Yes" if p < 0.05 else "No" for p in res_df["P-Value"]]
                                
                                # Sort by significance
                                sort_col = "P-Value (Adj)" if correction != "None" else "P-Value"
                                res_df = res_df.sort_values(sort_col)
                                
                                st.dataframe(
                                    res_df.style.format({"P-Value": "{:.4f}", "P-Value (Adj)": "{:.4f}", "Statistic": "{:.2f}", "Effect Size": "{:.3f}"}),
                                    width='stretch'
                                )
                                
                                # Visualization of Top Result
                                if not res_df.empty:
                                    top_res = res_df.iloc[0]
                                    is_sig = top_res['P-Value (Adj)'] < 0.05 if correction != "None" else top_res['P-Value'] < 0.05
                                    
                                    if is_sig:
                                        st.markdown(f"### 🔍 Top Influencer: {top_res['Variable']}")
                                        st.caption(f"Missingness in **{target_col}** is most strongly associated with **{top_res['Variable']}**.")
                                        
                                        if top_res['Variable'] in numeric_cols:
                                            fig = px.box(
                                                temp_df, 
                                                x='Missingness_Status', 
                                                y=top_res['Variable'], 
                                                color='Missingness_Status', 
                                                title=f"{top_res['Variable']} Distribution by Missingness of {target_col}"
                                            )
                                            st.plotly_chart(fig, width='stretch')
                                        else:
                                            # Bar chart
                                            counts = temp_df.groupby(['Missingness_Status', top_res['Variable']]).size().reset_index(name='count')
                                            fig = px.bar(
                                                counts, 
                                                x='Missingness_Status', 
                                                y='count', 
                                                color=top_res['Variable'], 
                                                barmode='group', 
                                                title=f"{top_res['Variable']} Distribution by Missingness of {target_col}"
                                            )
                                            st.plotly_chart(fig, width='stretch')
                                    else:
                                        st.success(f"No strong statistical evidence that missingness in **{target_col}** depends on the selected predictors.")

                else: # Global Dependency Scan
                    st.markdown("""
                    **Global Missingness Dependency Scan**
                    
                    This tool systematically tests all pairs of variables to identify:
                    1.  **Value Impacts**: Does the *value* of a predictor affect the missingness of a target?
                    2.  **Pattern Correlations**: Is the *missingness* of a predictor linked to the missingness of a target?
                    """)
                    
                    with st.expander("⚙️ Scan Configuration", expanded=True):
                        # Targets: Variables with missing data
                        scan_targets = st.multiselect(
                            "Select Target Variables (Missing Data)",
                            options=missing_counts.index,
                            default=missing_counts.index.tolist(),
                            help="Variables whose missingness you want to explain."
                        )
                        
                        # Predictors: All variables
                        scan_predictors = st.multiselect(
                            "Select Predictor Variables",
                            options=df.columns,
                            default=df.columns.tolist()[:min(20, len(df.columns))], # Default to top 20 to avoid overwhelming
                            help="Variables that might explain the missingness."
                        )
                        
                        # Test Preference
                        c1, c2 = st.columns(2)
                        with c1:
                            scan_test_pref = st.selectbox("Test Preference", ["Auto-Detect", "Force Parametric", "Force Non-Parametric"], key="scan_test_pref")
                        with c2:
                            scan_correction = st.selectbox("Multiple Testing Correction", ["None", "Bonferroni", "Benjamini-Hochberg (FDR)"], index=2, key="scan_correction")
                        
                    if st.button("Run Global Scan"):
                        if not scan_targets or not scan_predictors:
                            st.error("Please select at least one target and one predictor.")
                        else:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            def update_progress(p):
                                progress_bar.progress(p)
                                status_text.text(f"Scanning... {int(p*100)}%")
                                
                            with st.spinner("Running Global Dependency Scan... (This may take a moment)"):
                                scan_results = au.scan_missingness_dependencies(
                                    df, 
                                    target_cols=scan_targets, 
                                    predictor_cols=scan_predictors,
                                    test_preference=scan_test_pref,
                                    progress_callback=update_progress
                                )
                                
                            status_text.empty()
                            progress_bar.empty()
                            
                            if scan_results.empty:
                                st.info("No significant dependencies found (p < 0.05).")
                            else:
                                # Apply Correction
                                if scan_correction != "None":
                                    p_values = scan_results["P-Value"].fillna(1.0).values
                                    if scan_correction == "Bonferroni":
                                        reject, pvals_corrected, _, _ = smt.multipletests(p_values, method='bonferroni')
                                    elif scan_correction == "Benjamini-Hochberg (FDR)":
                                        reject, pvals_corrected, _, _ = smt.multipletests(p_values, method='fdr_bh')
                                    
                                    scan_results["P-Value (Adj)"] = pvals_corrected
                                    scan_results["Significant"] = ["Yes" if p < 0.05 else "No" for p in pvals_corrected]
                                    
                                    # Filter for significant after correction?
                                    # Or just show all? The user asked for "big table of significant associations".
                                    # Let's show all but sort by adjusted p-value.
                                    scan_results = scan_results.sort_values("P-Value (Adj)")
                                else:
                                    scan_results["Significant"] = ["Yes" if p < 0.05 else "No" for p in scan_results["P-Value"]]
                                    scan_results = scan_results.sort_values("P-Value")

                                st.success(f"Found {len(scan_results[scan_results['Significant'] == 'Yes'])} significant dependencies (Adjusted p < 0.05) out of {len(scan_results)} tests.")
                                
                                # Display Table
                                st.dataframe(
                                    scan_results.style.format({"P-Value": "{:.4f}", "P-Value (Adj)": "{:.4f}"}),
                                    width='stretch'
                                )
                                
                                # Download
                                csv = scan_results.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    "Download Results CSV",
                                    csv,
                                    "missingness_dependencies.csv",
                                    "text/csv",
                                    key='download-scan'
                                )

        else:
            st.success("No missing values detected in the dataset!")

    # --- Tab 2: Validity (Outliers) ---
    with tab2:
        st.subheader("Numerical Outlier Detection")
        st.markdown("Detect outliers using various statistical methods. The selected method will be used to calculate the **Statistical Validity** score.")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            # Callback to trigger rerun when method/params change
            def on_method_change():
                """Callback when method or params change - updates session state."""
                # Session state is already updated by widget, rerun happens automatically
            
            # --- Method Selection UI ---
            col1, col2 = st.columns(2)
            with col1:
                detection_method = st.selectbox(
                    "Detection Method",
                    ["iqr", "zscore", "quantile", "Local Outlier Factor", "Isolation Forest", "DBSCAN"],
                    index=["iqr", "zscore", "quantile", "Local Outlier Factor", "Isolation Forest", "DBSCAN"].index(
                        st.session_state.get('validity_outlier_method', 'iqr')
                    ),
                    help="**IQR**: Uses interquartile range (default)\n"
                         "**Z-score**: Uses standard deviations from mean\n"
                         "**Quantile**: Uses percentile thresholds\n"
                         "**LOF/IsoForest/DBSCAN**: Multivariate machine learning methods",
                    key="validity_detection_method"
                )
            
            # Store selected method in session state for dashboard recalculation
            st.session_state['validity_outlier_method'] = detection_method
            
            # --- Method-Specific Parameters ---
            with col2:
                if detection_method == "iqr":
                    # Get default from session state or use 1.5
                    default_mult = st.session_state.get('validity_outlier_params', {}).get('multiplier', 1.5)
                    multiplier = st.slider("IQR Multiplier", 0.5, 3.0, float(default_mult), 0.1, key="validity_iqr_mult",
                                          help="Higher values = fewer outliers detected")
                    params = {"multiplier": multiplier}
                elif detection_method == "zscore":
                    default_thresh = st.session_state.get('validity_outlier_params', {}).get('threshold', 3.0)
                    threshold = st.slider("Z-score Threshold", 1.0, 5.0, float(default_thresh), 0.1, key="validity_zscore_thresh",
                                         help="Number of standard deviations from mean")
                    params = {"threshold": threshold}
                elif detection_method == "quantile":
                    c1, c2 = st.columns(2)
                    with c1:
                        lower = st.number_input("Lower Percentile", 0.0, 0.5, 0.01, 0.01, key="validity_q_lower")
                    with c2:
                        upper = st.number_input("Upper Percentile", 0.5, 1.0, 0.99, 0.01, key="validity_q_upper")
                    params = {"lower": lower, "upper": upper}
                else:
                    # ML methods - show info
                    st.info("ℹ️ Multivariate methods analyze patterns across all selected columns.")
                    params = {}
            
            # Check if params changed - if so, trigger rerun to update scorecard
            old_params = st.session_state.get('validity_outlier_params', {})
            st.session_state['validity_outlier_params'] = params
            
            # Show guidance about score synchronization
            st.caption("💡 **Tip:** The health scorecard above will update automatically when you change detection settings.")
            
            # --- Column Selection (for ML methods) ---
            if detection_method in ["Local Outlier Factor", "Isolation Forest", "DBSCAN"]:
                selected_cols = st.multiselect(
                    "Select Columns for Analysis",
                    numeric_cols,
                    default=numeric_cols[:min(5, len(numeric_cols))],
                    key="validity_ml_cols"
                )
            else:
                selected_cols = numeric_cols
            
            # --- Run Detection ---
            outlier_summary = []
            outlier_mask_dict = {}
            total_outliers = 0
            total_values = 0
            
            if detection_method in ["iqr", "zscore", "quantile"]:
                # Univariate methods - per column
                for col in selected_cols:
                    data = df[col].dropna()
                    if len(data) == 0:
                        continue
                    
                    if detection_method == "iqr":
                        Q1 = data.quantile(0.25)
                        Q3 = data.quantile(0.75)
                        IQR = Q3 - Q1
                        mask = (data < (Q1 - params["multiplier"] * IQR)) | (data > (Q3 + params["multiplier"] * IQR))
                    elif detection_method == "zscore":
                        z_scores = np.abs((data - data.mean()) / data.std())
                        mask = z_scores > params["threshold"]
                    elif detection_method == "quantile":
                        lower_bound = data.quantile(params["lower"])
                        upper_bound = data.quantile(params["upper"])
                        mask = (data < lower_bound) | (data > upper_bound)
                    
                    outliers = mask.sum()
                    total_outliers += outliers
                    total_values += len(data)
                    outlier_mask_dict[col] = mask
                    
                    if outliers > 0:
                        outlier_summary.append({
                            "Column": col, 
                            "Outliers": outliers, 
                            "Percentage": round(outliers/len(data)*100, 2)
                        })
                
            else:
                # Multivariate ML methods
                if len(selected_cols) < 2:
                    st.warning("Please select at least 2 columns for multivariate analysis.")
                else:
                    try:
                        from sklearn.cluster import DBSCAN
                        from sklearn.ensemble import IsolationForest
                        from sklearn.neighbors import LocalOutlierFactor
                        from sklearn.preprocessing import StandardScaler
                        
                        # Prepare data
                        ml_data = df[selected_cols].dropna()
                        if len(ml_data) > 10:
                            scaler = StandardScaler()
                            scaled_data = scaler.fit_transform(ml_data)
                            
                            if detection_method == "Isolation Forest":
                                model = IsolationForest(contamination='auto', random_state=42)
                                predictions = model.fit_predict(scaled_data)
                                mask = predictions == -1
                            elif detection_method == "Local Outlier Factor":
                                model = LocalOutlierFactor(contamination='auto')
                                predictions = model.fit_predict(scaled_data)
                                mask = predictions == -1
                            elif detection_method == "DBSCAN":
                                model = DBSCAN(eps=0.5, min_samples=5)
                                predictions = model.fit_predict(scaled_data)
                                mask = predictions == -1  # Noise points
                            
                            outliers = mask.sum()
                            total_outliers = outliers
                            total_values = len(ml_data)
                            outlier_mask_dict['overall'] = pd.Series(mask, index=ml_data.index)
                            
                            outlier_summary.append({
                                "Column": "Overall (Multivariate)",
                                "Outliers": outliers,
                                "Percentage": round(outliers/len(ml_data)*100, 2)
                            })
                        else:
                            st.warning("Not enough data points for ML-based detection (need > 10).")
                    except ImportError as e:
                        st.error(f"Missing dependency: {e}")
            
            # --- Display Results ---
            if outlier_summary:
                outlier_df = pd.DataFrame(outlier_summary).sort_values("Outliers", ascending=False)
                
                # Display the scorecard value (same as Health Scorecard above)
                # This ensures alignment between Tab 2 and the scorecard
                scorecard_validity = metrics.get("Statistical Validity", 100.0)
                st.metric(
                    "Statistical Validity Score (from Health Scorecard)", 
                    f"{scorecard_validity:.2f}%",
                    help=f"Based on {detection_method.upper()} method. This value matches the Health Scorecard above."
                )
                
                # Summary chart and table
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig = px.bar(
                        outlier_df, 
                        x='Column', 
                        y='Outliers',
                        title=f"Outliers Count ({detection_method.upper()})",
                        color='Outliers',
                        color_continuous_scale='Oranges'
                    )
                    st.plotly_chart(fig, width="stretch")
                with col2:
                    st.dataframe(outlier_df, width="stretch")
                
                # --- Visualization Options ---
                st.divider()
                st.markdown("### Visual Inspection")
                
                viz_col1, viz_col2 = st.columns(2)
                with viz_col1:
                    viz_type = st.selectbox(
                        "Visualization Type",
                        ["Box Plot", "Violin Plot", "Scatter (with outliers)"],
                        key="validity_viz_type"
                    )
                with viz_col2:
                    if detection_method in ["iqr", "zscore", "quantile"]:
                        # Multi-select for columns with outliers
                        available_cols = [s["Column"] for s in outlier_summary]
                        viz_columns = st.multiselect(
                            "Select Columns",
                            available_cols,
                            default=available_cols[:min(3, len(available_cols))],
                            key="validity_viz_cols",
                            help="Select multiple columns to compare"
                        )
                    else:
                        # ML methods - automatically use the analysis columns
                        viz_columns = selected_cols
                        st.info(f"📊 Displaying columns from analysis: {', '.join(selected_cols[:5])}{'...' if len(selected_cols) > 5 else ''}")
                
                # Generate visualization
                if viz_type == "Box Plot" and viz_columns:
                    valid_cols = [c for c in viz_columns if c in df.columns]
                    if valid_cols:
                        # Create faceted subplots - each column with its own y-axis range
                        from plotly.subplots import make_subplots
                        
                        n_cols = len(valid_cols)
                        fig = make_subplots(rows=1, cols=n_cols, subplot_titles=valid_cols)
                        
                        colors = px.colors.qualitative.Plotly
                        for i, col in enumerate(valid_cols):
                            fig.add_trace(
                                go.Box(
                                    y=df[col].dropna(), 
                                    name=col, 
                                    boxpoints='outliers',
                                    marker_color=colors[i % len(colors)]
                                ),
                                row=1, col=i+1
                            )
                        
                        fig.update_layout(
                            title=f"Box Plot Comparison ({n_cols} columns)",
                            showlegend=False,
                            height=450
                        )
                        st.plotly_chart(fig, width="stretch")
                    
                elif viz_type == "Violin Plot" and viz_columns:
                    valid_cols = [c for c in viz_columns if c in df.columns]
                    if valid_cols:
                        # Create faceted subplots - each column with its own y-axis range
                        from plotly.subplots import make_subplots
                        
                        n_cols = len(valid_cols)
                        fig = make_subplots(rows=1, cols=n_cols, subplot_titles=valid_cols)
                        
                        colors = px.colors.qualitative.Plotly
                        for i, col in enumerate(valid_cols):
                            fig.add_trace(
                                go.Violin(
                                    y=df[col].dropna(), 
                                    name=col, 
                                    box_visible=True,
                                    points='outliers',
                                    marker_color=colors[i % len(colors)]
                                ),
                                row=1, col=i+1
                            )
                        
                        fig.update_layout(
                            title=f"Violin Plot Comparison ({n_cols} columns)",
                            showlegend=False,
                            height=450
                        )
                        st.plotly_chart(fig, width="stretch")
                    
                elif viz_type == "Scatter (with outliers)" and viz_columns:
                    valid_cols = [c for c in viz_columns if c in df.columns]
                    if valid_cols:
                        from plotly.subplots import make_subplots
                        
                        n_cols = len(valid_cols)
                        fig = make_subplots(rows=1, cols=n_cols, subplot_titles=valid_cols)
                        
                        # Get outlier mask - either per-column (univariate) or overall (ML methods)
                        if 'overall' in outlier_mask_dict:
                            # ML methods: use same mask for all columns
                            overall_mask = outlier_mask_dict['overall']
                        else:
                            overall_mask = None
                        
                        for i, col in enumerate(valid_cols):
                            col_data = df[[col]].dropna().copy()
                            col_data['Index'] = range(len(col_data))
                            
                            # Get appropriate outlier mask
                            if overall_mask is not None:
                                # ML methods - use overall mask
                                outlier_mask = overall_mask.reindex(col_data.index, fill_value=False)
                            else:
                                # Univariate methods - use per-column mask
                                outlier_mask = outlier_mask_dict.get(col, pd.Series(False, index=col_data.index))
                                outlier_mask = outlier_mask.reindex(col_data.index, fill_value=False)
                            
                            # Non-outliers (blue)
                            non_outliers = col_data[~outlier_mask]
                            fig.add_trace(
                                go.Scatter(
                                    x=non_outliers['Index'], 
                                    y=non_outliers[col],
                                    mode='markers',
                                    marker=dict(color='blue', size=5),
                                    name='Normal',
                                    showlegend=(i == 0)
                                ),
                                row=1, col=i+1
                            )
                            
                            # Outliers (red)
                            outliers_data = col_data[outlier_mask]
                            fig.add_trace(
                                go.Scatter(
                                    x=outliers_data['Index'], 
                                    y=outliers_data[col],
                                    mode='markers',
                                    marker=dict(color='red', size=7),
                                    name='Outlier',
                                    showlegend=(i == 0)
                                ),
                                row=1, col=i+1
                            )
                        
                        method_label = "Multivariate" if overall_mask is not None else "Univariate"
                        fig.update_layout(
                            title=f"Scatter Plot ({n_cols} columns) - {method_label} Outliers in Red",
                            height=450
                        )
                        st.plotly_chart(fig, width="stretch")
            else:
                st.success("No statistical outliers detected with the selected method.")
        else:
            st.info("No numerical columns to analyze.")

    # --- Tab 3: Clinical Validity ---
    with tab3:
        st.subheader("Compliance with Declared Anomaly Criteria")
        if "Clinical Validity" in metrics:
            st.metric("Clinical Validity Score", f"{metrics['Clinical Validity']}%")
            if metrics['Clinical Validity'] < 100:
                st.warning("Some rows violate the expert-defined anomaly rules.")
                
                if hasattr(auditor, 'clinical_anomalies_booleans') and auditor.clinical_anomalies_booleans is not None:
                    # --- Summary Report ---
                    st.markdown("### 📊 Violations Summary")
                    
                    # Calculate stats
                    summary_stats = []
                    total_rows = len(df)
                    
                    # Iterate over columns (rules) in the boolean dataframe
                    for col in auditor.clinical_anomalies_booleans.columns:
                        count = auditor.clinical_anomalies_booleans[col].sum()
                        if count > 0:
                            summary_stats.append({
                                "Criteria": col,
                                "Violations Count": count,
                                "Percentage": round((count / total_rows) * 100, 2)
                            })
                    
                    if summary_stats:
                        summary_df = pd.DataFrame(summary_stats).sort_values("Violations Count", ascending=False)
                        
                        # Reorder columns
                        summary_df = summary_df[["Criteria", "Violations Count", "Percentage"]]
                        
                        # Prepare Table DataFrame with Total Row
                        total_violations = summary_df["Violations Count"].sum()
                        total_percentage = summary_df["Percentage"].sum()
                        
                        total_row = pd.DataFrame([{
                            "Criteria": "TOTAL", 
                            "Violations Count": total_violations, 
                            "Percentage": total_percentage
                        }])
                        table_df = pd.concat([summary_df, total_row], ignore_index=True)
                        
                        # Use vertical_alignment="center" for better layout
                        col1, col2 = st.columns([2, 1], vertical_alignment="center")
                        
                        with col1:
                            # Horizontal Bar Chart for better label readability
                            fig = px.bar(
                                summary_df, 
                                y='Criteria', 
                                x='Violations Count',
                                orientation='h',
                                title="Violations by Criteria",
                                text='Violations Count',
                                color='Violations Count',
                                color_continuous_scale='Reds'
                            )
                            fig.update_layout(yaxis={'categoryorder':'total ascending'}) # Sort bars
                            st.plotly_chart(fig, width='stretch')
                        
                        with col2:
                            st.dataframe(
                                table_df,
                                width='stretch',
                                hide_index=True,
                                column_config={
                                    "Percentage": st.column_config.NumberColumn(
                                        "Percentage",
                                        format="%.2f%%"
                                    )
                                }
                            )
                            st.caption("Note: *Total Percentage* is the sum of all violations. It may be higher than the *Clinical Validity Score* gap because a single row can violate multiple criteria.")
                    
                    st.divider()

                    st.markdown("### 📋 Detailed Anomaly Report")
                    st.markdown("The table below shows rows that triggered at least one anomaly. Cells causing the anomaly are highlighted (where applicable).")
                    
                    # Define styling function
                    def highlight_anomalies(row):
                        styles = [''] * len(row)
                        # row.name is the index
                        if hasattr(auditor, 'clinical_anomalies_booleans') and auditor.clinical_anomalies_booleans is not None:
                            if row.name in auditor.clinical_anomalies_booleans.index:
                                bool_row = auditor.clinical_anomalies_booleans.loc[row.name]
                                
                                # Handle potential duplicate indices safely
                                if isinstance(bool_row, pd.DataFrame):
                                    bool_row = bool_row.iloc[0] # Take first match if duplicates exist

                                for i, col_name in enumerate(row.index):
                                    # Check if col_name is a mask that is True for this row
                                    if col_name in bool_row.index and bool_row[col_name]:
                                        styles[i] = 'background-color: #ff4b4b; color: white'
                        return styles

                    # Display with style
                    st.dataframe(auditor.clinical_anomalies_df.style.apply(highlight_anomalies, axis=1), width='stretch')
                    
                    # Download button
                    csv = auditor.clinical_anomalies_df.to_csv(index=True).encode('utf-8')
                    st.download_button(
                        label="📥 Download Anomaly Report",
                        data=csv,
                        file_name='clinical_anomalies.csv',
                        mime='text/csv',
                    )
                
                # Display Criteria List
                if auditor.config and "mask_families" in auditor.config:
                    clinical_config = auditor.config["mask_families"].get("clinical_anomalies", {})
                    if clinical_config:
                        st.divider()
                        st.markdown("### 📜 Applied Clinical Criteria")
                        st.markdown("The following rules were used to detect anomalies:")
                        
                        # Use card-based layout similar to Data Monitoring page
                        cols_per_row = 3
                        grid = st.columns(cols_per_row)
                        
                        # Filter out 'operator' key
                        rules = {k: v for k, v in clinical_config.items() if k != "operator"}
                        
                        for i, (name, rule) in enumerate(rules.items()):
                            is_numeric = rule.get('numeric', False)
                            color = '#3498db' if is_numeric else '#2ecc71'
                            
                            with grid[i % cols_per_row]:
                                if is_numeric:
                                    with st.expander(f"🔢 {name}", expanded=False):
                                        st.markdown(f"**Type:** 🔢 <span style='color:{color};'>Numeric</span>", unsafe_allow_html=True)
                                        
                                        c1, c2 = st.columns(2)
                                        with c1:
                                            if rule.get('lower_bound') is not None:
                                                st.metric("Min", rule['lower_bound'])
                                        with c2:
                                            if rule.get('upper_bound') is not None:
                                                st.metric("Max", rule['upper_bound'])
                                                
                                        if rule.get('strategy'):
                                            st.markdown(f"**Strategy:** {rule['strategy']}")
                                else:
                                    with st.expander(f"📝 {name}", expanded=False):
                                        st.markdown(f"**Type:** 📝 <span style='color:{color};'>Expression</span>", unsafe_allow_html=True)
                                        st.code(rule.get('expression', 'N/A'), language='python')

                else:
                    st.info("👉 Go to the **'Data Validation & Monitoring'** page to inspect specific rows and rules.")
            else:
                st.success("All rows comply with the declared clinical rules.")
        else:
            st.info("No clinical anomaly rules defined yet. Go to **'Data Validation & Monitoring'** to set them up.")

    # --- Tab 4: Conformity ---
    with tab4:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Uniqueness")
            duplicates = df.duplicated().sum()
            st.metric("Duplicate Rows", duplicates)
            if duplicates > 0:
                st.warning(f"Found {duplicates} duplicate rows.")
                with st.expander("Show Duplicates"):
                    st.dataframe(df[df.duplicated()].head(50), width="stretch")
            else:
                st.success("No duplicate rows found.")

        with c2:
            st.subheader("Type Consistency")
            object_cols = df.select_dtypes(include=['object']).columns
            potential_issues = []
            for col in object_cols:
                # Check if numeric
                try:
                    pd.to_numeric(df[col].dropna())
                    potential_issues.append(f"Column **'{col}'** is type 'Object' but looks Numeric.")
                    continue
                except: pass
                # Check if date
                try:
                    pd.to_datetime(df[col].dropna())
                    potential_issues.append(f"Column **'{col}'** is type 'Object' but looks like a Date.")
                    continue
                except: pass
            
            if potential_issues:
                for issue in potential_issues:
                    st.warning(issue)
            else:
                st.success("Data types appear consistent.")

        with c3:
            st.subheader("Uniformity")
            if "Uniformity" in metrics:
                st.metric("Uniformity Score", f"{metrics['Uniformity']}%")
                if metrics['Uniformity'] < 100:
                    st.warning("Inconsistent text formatting detected.")
                    
                    if hasattr(auditor, 'uniformity_details') and auditor.uniformity_details:
                        st.markdown("### 📝 Uniformity Issues")
                        uniformity_df = pd.DataFrame(auditor.uniformity_details)
                        st.dataframe(uniformity_df, width='stretch', hide_index=True)
                    else:
                        st.markdown("""
                        **Common Issues:**
                        - Trailing/Leading whitespace
                        - Inconsistent capitalization (e.g., 'Male' vs 'male')
                        """)
                else:
                    st.success("Text data is uniform.")
