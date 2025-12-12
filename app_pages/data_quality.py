import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from explore.data_quality import DataQualityAuditor

def render_dashboard(df, config):
    """
    Renders the Data Quality Dashboard.
    """
    st.markdown("Comprehensive assessment of your dataset's health across key dimensions.")

    # Initialize Auditor
    auditor = DataQualityAuditor(df, config=config)
    metrics = auditor.run_audit()
    advice_list = auditor.generate_advice()

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
                    delta_color = "normal"
                    if value < 80: delta_color = "inverse"
                    elif value < 95: delta_color = "off"
                    
                    st.metric(
                        label=name, 
                        value=f"{value}%", 
                        delta=f"{value - 100:.1f}%" if value < 100 else "Perfect",
                        delta_color=delta_color
                    )
        
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
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
        
        if not missing_counts.empty:
            col1, col2 = st.columns([2, 1])
            with col1:
                # Plotly Bar Chart
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
        else:
            st.success("No missing values detected in the dataset!")

    # --- Tab 2: Validity (Outliers) ---
    with tab2:
        st.subheader("Numerical Outlier Detection (IQR Method)")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            outlier_summary = []
            for col in numeric_cols:
                data = df[col].dropna()
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
                if outliers > 0:
                    outlier_summary.append({"Column": col, "Outliers": outliers, "Percentage": round(outliers/len(data)*100, 2)})
            
            if outlier_summary:
                outlier_df = pd.DataFrame(outlier_summary).sort_values("Outliers", ascending=False)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig = px.bar(
                        outlier_df, 
                        x='Column', 
                        y='Outliers',
                        title="Outliers Count per Column",
                        color='Outliers',
                        color_continuous_scale='Oranges'
                    )
                    st.plotly_chart(fig, width='stretch')
                with col2:
                    st.dataframe(outlier_df, width='stretch')
            else:
                st.success("No statistical outliers detected.")
        else:
            st.info("No numerical columns to analyze.")

    # --- Tab 3: Clinical Validity ---
    with tab3:
        st.subheader("Compliance with Declared Anomaly Criteria")
        if "Clinical Validity" in metrics:
            st.metric("Clinical Validity Score", f"{metrics['Clinical Validity']}%")
            if metrics['Clinical Validity'] < 100:
                st.warning("Some rows violate the expert-defined anomaly rules.")
                
                if hasattr(auditor, 'clinical_anomalies_df') and auditor.clinical_anomalies_df is not None:
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
                    st.dataframe(df[df.duplicated()].head(50))
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
