import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats
from plotly.subplots import make_subplots
from app_pages import data_monitoring as dm
import logging
from manage.db_manager import DBManager
import utils.visualization_utils as vu
from utils.visualization_utils import DataAnalyzer
import plotly.figure_factory as ff

logging.basicConfig(level=logging.DEBUG)

def app():
    # Initialize DB Manager
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DBManager()

    # Dataset Management Sidebar (Automatic Versioning)
    with st.sidebar.expander("Dataset History", expanded=False):
        st.caption("Manage dataset versions")
        
        datasets = st.session_state.db_manager.get_available_datasets()
        if datasets:
            selected_dataset = st.selectbox("Select Dataset Version", datasets, index=0)
            if st.button("Load Selected Version"):
                df_loaded, msg = st.session_state.db_manager.load_dataset(selected_dataset)
                if df_loaded is not None:
                    st.session_state['data'] = df_loaded
                    st.session_state['working_df'] = df_loaded.copy()
                    st.session_state['current_dataset_name'] = selected_dataset
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        else:
            st.info("No saved datasets found.")

    # Implicitly load latest dataset if none is loaded
    if st.session_state.get("data") is None:
        datasets = st.session_state.db_manager.get_available_datasets()
        if datasets:
            latest_dataset = datasets[0]
            df_loaded, msg = st.session_state.db_manager.load_dataset(latest_dataset)
            if df_loaded is not None:
                st.session_state['data'] = df_loaded
                st.session_state['working_df'] = df_loaded.copy()
                st.session_state['current_dataset_name'] = latest_dataset
                st.toast(f"Auto-loaded latest dataset: {latest_dataset}")
            else:
                st.warning("Failed to auto-load latest dataset.")
        else:
            st.warning("Please upload data first!")
            return
    
    df_full = st.session_state["data"]
    analyzer = DataAnalyzer(df_full)
    # Apply pd.to_datetime with infer_datetime_format=True for each date column
    df_full[analyzer.date_cols] = df_full[analyzer.date_cols].apply(pd.to_datetime, errors='coerce')


    with st.expander("Apply Inclusion and Anomaly Criteria",expanded=False):
        col_inc, col_ano = st.columns(2)
        # Checkbox for filtering out anomalies
        with col_ano:
            filter_out_anomaly = st.checkbox(
                label="Exclude Rows with Anomalies.",
                value=False,
                help="Check this box to exclude rows flagged as anomalies from the dataset."
            )

        # Checkbox for limiting rows based on inclusion criteria
        with col_inc:
            filter_on_inclusion = st.checkbox(
                label="Restrict to Inclusion Criteria.",
                value=False,
                help="Check this box to limit the displayed rows to those meeting the inclusion criteria."
            )


        # Base DataFrame
        df = df_full.copy()

        # Apply filters based on checkbox states
        if filter_on_inclusion:
            if "study_inclusion_all" not in df.columns:
                st.warning(f"The inclusion criteria are missing and need to be defined first.")
            else:
                df = df[df['study_inclusion_all'] == True]

        if filter_out_anomaly:
            if "clinical_anomalies_any" not in df.columns:
                st.warning(f"The anomaly criteria are missing and need to be defined first.")
            else:
                df = df[df['clinical_anomalies_any'] == False]

        st.dataframe(df.head(), width='stretch', hide_index=True)
        st.write(f"📊 **Filtered Data Overview:** {df.shape[0]:,} rows and {df.shape[1]:,} columns selected.")

    # Outliers Handler
    with st.expander("Outliers Handler"):
        df = dm.add_outlier_handling_ui(df, analyzer.numeric_cols)
    
    # Create tabs for different visualization aspects
    tabs = st.tabs(["Basic Visualizations", "..."])
    
    with tabs[0]:
        st.subheader("Basic Data Visualization")
        
        # Plot categories with requirements
        plot_categories = {
            "Distribution": {
                "plots": ["Histogram", "Box Plot", "Violin Plot"],
                "requirements": {"numeric": 1}
            },
            "Categorical": {
                "plots": ["Bar Plot", "Pie Chart", "Sunburst Chart", "Icicle Chart", "Treemap"],
                "requirements": {"categorical": 1}
            },
            "Time Series": {
                "plots": ["Line Plot"],
                "requirements": {"date": 1,"numeric": 1}
            },
            "Relationships": {
                "plots": ["Scatter Plot", "Correlation Matrix", "Clustermap"],
                "requirements": {"numeric": 1}
            },
            "Medical Research": {
                "plots": ["Bland-Altman Plot", "ROC Curve"],
                "requirements": {"numeric": 2}
            }
        }
        
        # Filter available categories based on data
        available_categories = []
        for category, info in plot_categories.items():
            requirements_met = True
            for req_type, req_count in info["requirements"].items():
                if req_type == "numeric" and len(analyzer.numeric_cols) < req_count:
                    requirements_met = False
                elif req_type == "categorical" and len(analyzer.categorical_cols) < req_count:
                    requirements_met = False
                elif req_type == "date" and len(analyzer.date_cols) < req_count:
                    requirements_met = False
            if requirements_met:
                available_categories.append(category)
        
        if not available_categories:
            st.error("Your data doesn't contain enough suitable columns for visualization.")
            return
        
        # Sidebar configurations
        st.sidebar.subheader("Visualization Options")
        plot_category = st.sidebar.selectbox("Plot Category", available_categories)
        plot_type = st.sidebar.selectbox("Plot Type", plot_categories[plot_category]["plots"])
        
        # Get suitable columns for the selected plot type
        suitable_cols = analyzer.get_suitable_columns(plot_type)
        
        # Visual customization options
        with st.sidebar.expander("Visual Settings"):
            # Theme selection
            theme = st.selectbox(
                "Theme",
                ["plotly", "plotly_dark", "plotly_white", "seaborn", "ggplot2"]
            )

            # Discrete color palettes
            color_sequences = {
                "Set1 - Default qualitative color palette": px.colors.qualitative.Set1,
                "Pastel1 - Soft pastel color palette": px.colors.qualitative.Pastel1,
                "Safe - Colorblind-friendly color palette": px.colors.qualitative.Safe,
                "Vivid - Vibrant color palette": px.colors.qualitative.Vivid,
                "Set2 - Alternative qualitative color palette": px.colors.qualitative.Set2,
                "Set3 - Another qualitative color palette": px.colors.qualitative.Set3,
                "Dark24 - Dark color palette with 24 distinct colors": px.colors.qualitative.Dark24,
                "Light24 - Light color palette with 24 distinct colors": px.colors.qualitative.Light24,
                "Alphabet - Wide range of colors for categorical data": px.colors.qualitative.Alphabet,
                "T10 - 10 distinct colors for smaller datasets": px.colors.qualitative.T10,
            }

            # Continuous color palettes
            continuous_color_sequences = {
                "Viridis - Perceptually uniform, colorblind-friendly": px.colors.sequential.Viridis,
                "Cividis - Perceptually uniform, colorblind-friendly": px.colors.sequential.Cividis,
                "Plasma - Smooth transition from dark to light": px.colors.sequential.Plasma,
                "Inferno - Smooth transition from dark to light": px.colors.sequential.Inferno,
                "RdBu - Diverging color map": px.colors.sequential.RdBu,
                "Blues - Sequential color map with shades of blue": px.colors.sequential.Blues,
                "Greens - Sequential color map with shades of green": px.colors.sequential.Greens,
                "Greys - Sequential color map with shades of grey": px.colors.sequential.Greys,
                "YlOrBr - Sequential color map with shades of yellow, orange, and brown": px.colors.sequential.YlOrBr,
                "YlOrRd - Sequential color map with shades of yellow, orange, and red": px.colors.sequential.YlOrRd,
            }

            # Determine available color schemes based on plot type
            if plot_type in ["Histogram", "Box Plot", "Violin Plot", "Scatter Plot", "Line Plot", "Bar Plot", "Pie Chart", "Clustermap", "Bland-Altman Plot", "ROC Curve"]:
                color_scheme_type = "Discrete"
            else:
                color_scheme_type = "Continuous"

            # Unified color scheme selection
            if color_scheme_type == "Discrete":
                color_scheme = st.selectbox("Color Scheme", list(color_sequences.keys()))
                selected_color = color_sequences[color_scheme]  # Use discrete color scheme
            else:
                color_scheme = st.selectbox("Color Scheme", list(continuous_color_sequences.keys()))
                selected_color = continuous_color_sequences[color_scheme]  # Use continuous color scheme

        
        # Main visualization area
        # Plot Descriptions
        plot_info = {
            "Histogram": {
                "title": "Histogram",
                "desc": "A graphical representation of the distribution of numerical data, where data is grouped into continuous bins.",
                "usage": "Analyze the underlying frequency distribution, central tendency, and dispersion of continuous variables.",
                "insight": "Identify skewness, kurtosis, modality (unimodal/bimodal), and potential outliers.",
                "inputs": "1 Numeric variable. Optional: 1 Categorical variable for grouping."
            },
            "Box Plot": {
                "title": "Box Plot (Box-and-Whisker)",
                "desc": "A standardized visualization of data distribution based on a five-number summary (minimum, Q1, median, Q3, maximum), integrated with comparative statistical analysis.",
                "usage": "Compare distributions across groups, detect outliers, and validate statistical differences using parametric or non-parametric tests.",
                "insight": "Assess the interquartile range (IQR), median values, and dispersion. Statistically significant pairwise differences are annotated.",
                "inputs": "1 or more Numeric variables (Y). Optional: 1 Categorical variable (X) for grouping."
            },
            "Violin Plot": {
                "title": "Violin Plot",
                "desc": "A hybrid visualization combining a box plot with a kernel density estimation to depict the probability density of the data at different values.",
                "usage": "Visualize the full distribution shape and summary statistics simultaneously, facilitating detailed group comparisons.",
                "insight": "Observe the distribution modality and density. Wider sections indicate higher frequency. Includes statistical significance annotations for group comparisons.",
                "inputs": "1 or more Numeric variables (Y). Optional: 1 Categorical variable (X) for grouping."
            },
            "Bar Plot": {
                "title": "Bar Plot",
                "desc": "A chart that presents categorical data with rectangular bars with heights or lengths proportional to the values they represent.",
                "usage": "Compare aggregate values (counts, means, sums) across discrete categories.",
                "insight": "Evaluate relative magnitude differences between categories.",
                "inputs": "1 Categorical variable (X). Optional: 1 Categorical variable for grouping."
            },
            "Pie Chart": {
                "title": "Pie Chart",
                "desc": "A circular statistical graphic divided into slices to illustrate numerical proportions of a whole.",
                "usage": "Display relative composition or percentage distribution of a categorical variable.",
                "insight": "Identify the dominant and minor constituents of the dataset.",
                "inputs": "1 Categorical variable (Labels)."
            },
            "Sunburst Chart": {
                "title": "Sunburst Chart",
                "desc": "A hierarchical chart used to visualize the proportion of different categories within a hierarchy.",
                "usage": "Visualize hierarchical data structures and the distribution of categories at each level.",
                "insight": "Understand relationships between parent and child categories and their relative sizes.",
                "inputs": "1 or more Categorical variables (Hierarchy)."
            },
            "Icicle Chart": {
                "title": "Icicle Chart",
                "desc": "A hierarchical chart where the hierarchy is defined by the nesting of rectangles.",
                "usage": "Visualize hierarchical data and the proportion of categories. Good for visualizing deep hierarchies.",
                "insight": "Analyze the hierarchical structure and value contribution of each segment.",
                "inputs": "1 or more Categorical variables (Hierarchy)."
            },
            "Treemap": {
                "title": "Treemap",
                "desc": "A hierarchical chart using nested rectangles to represent data values.",
                "usage": "Compare proportions within a hierarchy and spot patterns across categories.",
                "insight": "Efficiently displays large amounts of hierarchical data in a compact space.",
                "inputs": "1 or more Categorical variables (Hierarchy)."
            },
            "Line Plot": {
                "title": "Line Plot",
                "desc": "A chart displaying information as a series of data points connected by straight line segments, primarily used for temporal data.",
                "usage": "Analyze trends, patterns, and fluctuations over a continuous interval or time period.",
                "insight": "Detect seasonality, trends (upward/downward), and structural breaks in the time series.",
                "inputs": "1 Date/Time variable (X), 1 Numeric variable (Y). Optional: 1 Categorical variable for grouping."
            },
            "Scatter Plot": {
                "title": "Scatter Plot",
                "desc": "A diagram using Cartesian coordinates to display values for typically two variables for a set of data.",
                "usage": "Investigate the relationship, correlation, or association between two continuous variables.",
                "insight": "Determine the nature (linear/non-linear), direction, and strength of the relationship between variables.",
                "inputs": "2 Numeric variables (X, Y). Optional: 1 Categorical variable for color/size coding."
            },
            "Correlation Matrix": {
                "title": "Correlation Matrix",
                "desc": "A tabular visualization displaying the correlation coefficients between multiple variables.",
                "usage": "Quantify the degree of linear relationship between multiple pairs of numeric variables.",
                "insight": "Identify strong positive (near +1) or negative (near -1) correlations and potential multicollinearity.",
                "inputs": "2 or more Numeric variables."
            },
            "Clustermap": {
                "title": "Clustermap (Hierarchical Clustering)",
                "desc": "A matrix-based visualization that applies hierarchical clustering to reorganize rows and columns based on similarity.",
                "usage": "Uncover hidden structures, patterns, and relationships in complex, high-dimensional datasets.",
                "insight": "Identify clusters of similar samples or features, indicated by dendrogram structures and color intensity.",
                "inputs": "2 or more Numeric variables."
            },
            "Bland-Altman Plot": {
                "title": "Bland-Altman Plot (Difference Plot)",
                "desc": "A difference plot used to analyze the agreement between two quantitative measurements.",
                "usage": "Assess the concordance between two methods of measurement or clinical instruments.",
                "insight": "Evaluate bias (mean difference) and limits of agreement (±1.96 SD). Detect systematic differences or proportional bias.",
                "inputs": "2 Numeric variables (paired measurements)."
            },
            "ROC Curve": {
                "title": "ROC Curve (Receiver Operating Characteristic)",
                "desc": "A graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.",
                "usage": "Evaluate and compare the performance of diagnostic tests or classification models.",
                "insight": "The Area Under the Curve (AUC) quantifies performance (1.0 is perfect). The curve visualizes the trade-off between Sensitivity and Specificity.",
                "inputs": "1 Numeric variable (Score/Probability), 1 Binary variable (Ground Truth)."
            }
        }

        if plot_type in plot_info:
            info = plot_info[plot_type]
            with st.expander(f"ℹ️ Guide: {info['title']}", expanded=True):
                st.markdown(f"**Description:** {info['desc']}")
                st.markdown(f"**Inputs:** {info['inputs']}")
                st.markdown(f"**When to use:** {info['usage']}")
                st.markdown(f"**What to look for:** {info['insight']}")

        try:
            fig = None  # Initialize figure variable
            
            if plot_type == "Histogram":
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_col = st.selectbox("Select Variable", suitable_cols["x"])
                with col2:
                    color_col = st.selectbox("Group by (optional)", 
                                           ["None"] + suitable_cols.get("color", []))
                with col3:
                    hist_mode = st.selectbox("Representation", ["Bars", "Lines (Frequency Polygon)", "Smooth Density (KDE)"])
                
                nbins = st.slider("Number of Bins", 5, 100, 20)
                
                if hist_mode == "Bars":
                    fig = px.histogram(
                        df,
                        x=x_col,
                        color=None if color_col == "None" else color_col,
                        nbins=nbins,
                        color_discrete_sequence=selected_color,
                        barmode='overlay' if color_col != "None" else 'relative',
                        opacity=0.7 if color_col != "None" else 1.0
                    )
                
                elif hist_mode == "Lines (Frequency Polygon)":
                    fig = go.Figure()
                    if color_col == "None":
                        counts, bin_edges = np.histogram(df[x_col].dropna(), bins=nbins)
                        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                        fig.add_trace(go.Scatter(x=bin_centers, y=counts, mode='lines+markers', name=x_col, line=dict(color=selected_color[0])))
                    else:
                        groups = df[color_col].dropna().unique()
                        # Compute common bins for comparison
                        data_min = df[x_col].min()
                        data_max = df[x_col].max()
                        bins = np.linspace(data_min, data_max, nbins + 1)
                        bin_centers = 0.5 * (bins[:-1] + bins[1:])
                        
                        for i, group in enumerate(groups):
                            group_data = df[df[color_col] == group][x_col].dropna()
                            counts, _ = np.histogram(group_data, bins=bins)
                            fig.add_trace(go.Scatter(
                                x=bin_centers, 
                                y=counts, 
                                mode='lines+markers', 
                                name=str(group),
                                line=dict(color=selected_color[i % len(selected_color)])
                            ))
                    fig.update_layout(title=f"Frequency Polygon of {x_col}", xaxis_title=x_col, yaxis_title="Count")

                elif hist_mode == "Smooth Density (KDE)":
                    hist_data = []
                    group_labels = []
                    colors = []
                    
                    if color_col == "None":
                        data = df[x_col].dropna()
                        if len(data) > 1:
                            hist_data = [data]
                            group_labels = [x_col]
                            colors = [selected_color[0]]
                    else:
                        groups = df[color_col].dropna().unique()
                        for i, group in enumerate(groups):
                            group_data = df[df[color_col] == group][x_col].dropna()
                            if len(group_data) > 1:
                                hist_data.append(group_data)
                                group_labels.append(str(group))
                                colors.append(selected_color[i % len(selected_color)])
                    
                    if hist_data:
                        # Create distplot with no histogram, only curve
                        fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False, colors=colors)
                        fig.update_layout(title=f"Density Plot of {x_col}", xaxis_title=x_col, yaxis_title="Density")
                    else:
                        st.warning("Not enough data to generate KDE.")
                        fig = None
           
            elif plot_type == "Box Plot":
                col1, col2 = st.columns(2)

                with col1:
                    y_cols = st.multiselect("Select Numeric Variables", suitable_cols["y"], default=[suitable_cols["y"][0]])

                with col2:
                    x_col = st.selectbox("Group by", ["None"] + suitable_cols["x"])

                orientation = st.radio("Orientation", ["Vertical", "Horizontal"], index=0)
                
                show_significance = st.checkbox("Show Statistical Significance", value=False, key="box_sig")
                test_type = "Auto-Detect"
                correction_method = "None"

                if show_significance:
                    if x_col == "None":
                        st.warning("Statistical significance requires a grouping variable.")
                        show_significance = False
                    elif orientation == "Horizontal":
                        st.warning("Statistical significance annotations are currently only supported for Vertical orientation.")
                        show_significance = False
                    else:
                        col_test, col_corr = st.columns(2)
                        with col_test:
                            test_type = st.selectbox(
                                "Test Selection Strategy",
                                ["Auto-Detect", "Force Parametric (T-test/ANOVA)", "Force Non-Parametric (Mann-Whitney/Kruskal)"],
                                key="box_test_type"
                            )
                        with col_corr:
                            correction_method = st.selectbox(
                                "Multiple Testing Correction",
                                ["None", "Bonferroni", "Benjamini-Hochberg (FDR)"],
                                key="box_corr_method"
                            )

                selected_colors = color_sequences[color_scheme]

                if len(y_cols) > 0:
                    fig = make_subplots(rows=1, cols=len(y_cols), subplot_titles=y_cols)

                    for i, y_col in enumerate(y_cols):
                        if x_col == "None":
                            fig.add_trace(
                                go.Box(
                                    y=df[y_col] if orientation == "Vertical" else None,
                                    x=None if orientation == "Vertical" else df[y_col],
                                    name=y_col,
                                    orientation='v' if orientation == "Vertical" else 'h',
                                    marker=dict(color=selected_colors[i % len(selected_colors)])
                                ),
                                row=1,
                                col=i+1
                            )
                        else:
                            unique_categories = df[x_col].dropna().unique()
                            try:
                                unique_categories = sorted(unique_categories)
                            except:
                                pass

                            for j, category in enumerate(unique_categories):
                                show_legend = i == 0
                                fig.add_trace(
                                    go.Box(
                                        y=df[df[x_col] == category][y_col] if orientation == "Vertical" else None,
                                        x=None if orientation == "Vertical" else df[df[x_col] == category][y_col],
                                        name=str(category),
                                        legendgroup=str(category),
                                        orientation='v' if orientation == "Vertical" else 'h',
                                        marker=dict(color=selected_colors[j % len(selected_colors)]),
                                        showlegend=show_legend
                                    ),
                                    row=1,
                                    col=i+1
                                )
                            
                            if show_significance:
                                data_list = [df[df[x_col] == g][y_col].dropna().values for g in unique_categories]
                                group_labels = [str(g) for g in unique_categories]
                                
                                # Compute stats
                                stats_df = vu.compute_pairwise_stats(data_list, group_labels, test_type=test_type, correction_method=correction_method)
                                
                                # Extract significant pairs for plot
                                sig_pairs = []
                                if not stats_df.empty:
                                    for _, row in stats_df.iterrows():
                                        if row["Significant"]:
                                            sig_pairs.append((row["g1_index"], row["g2_index"], row["P-Value (Adj)"]))
                                    sig_pairs.sort(key=lambda x: x[2])
                                
                                vu.add_significance_annotations(fig, data_list, sig_pairs, row=1, col=i+1)
                                
                                # Display Table
                                if sig_pairs:
                                    expander_title = f"**:green[📊 Statistical Results for {y_col} (Significant)]**"
                                else:
                                    expander_title = f"📊 Statistical Results for {y_col}"

                                with st.expander(expander_title, expanded=False):
                                    cols_to_drop = ["g1_index", "g2_index"]
                                    if correction_method == "None":
                                        cols_to_drop.append("P-Value (Adj)")
                                    
                                    display_df = stats_df.drop(columns=cols_to_drop, errors='ignore')
                                    
                                    st.dataframe(
                                        display_df.style.apply(lambda x: ['background-color: #d4edda' if x['Significant'] else '' for _ in x], axis=1)
                                        .format({"P-Value": "{:.4f}", "P-Value (Adj)": "{:.4f}", "Statistic": "{:.2f}"}, na_rep="-"),
                                        width='stretch'
                                    )

                    fig.update_layout(title_text="Box Plots")
                    fig.update_layout(title_text="Box Plots")
                    for i in range(1, len(y_cols) + 1):
                        fig.update_xaxes(showgrid=True, gridcolor='lightgray', row=1, col=i)
                        fig.update_yaxes(showgrid=True, gridcolor='lightgray', row=1, col=i)
                        
                        # Ensure consistent tick angle for all subplots
                        if orientation == "Vertical" and x_col != "None":
                            fig.update_xaxes(tickangle=-45, row=1, col=i)


            elif plot_type == "Violin Plot":
                col1, col2 = st.columns(2)

                with col1:
                    y_cols = st.multiselect("Select Numeric Variables", suitable_cols["y"], default=[suitable_cols["y"][0]])

                with col2:
                    x_col = st.selectbox("Group by", ["None"] + suitable_cols["x"])

                orientation = st.radio("Orientation", ["Vertical", "Horizontal"], index=0)
                
                show_significance = st.checkbox("Show Statistical Significance", value=False, key="violin_sig")
                test_type = "Auto-Detect"
                correction_method = "None"

                if show_significance:
                    if x_col == "None":
                        st.warning("Statistical significance requires a grouping variable.")
                        show_significance = False
                    elif orientation == "Horizontal":
                        st.warning("Statistical significance annotations are currently only supported for Vertical orientation.")
                        show_significance = False
                    else:
                        col_test, col_corr = st.columns(2)
                        with col_test:
                            test_type = st.selectbox(
                                "Test Selection Strategy",
                                ["Auto-Detect", "Force Parametric (T-test/ANOVA)", "Force Non-Parametric (Mann-Whitney/Kruskal)"],
                                key="violin_test_type"
                            )
                        with col_corr:
                            correction_method = st.selectbox(
                                "Multiple Testing Correction",
                                ["None", "Bonferroni", "Benjamini-Hochberg (FDR)"],
                                key="violin_corr_method"
                            )

                selected_colors = color_sequences[color_scheme]

                if len(y_cols) > 0:
                    fig = make_subplots(rows=1, cols=len(y_cols), subplot_titles=y_cols)

                    for i, y_col in enumerate(y_cols):
                        if x_col == "None":
                            fig.add_trace(
                                go.Violin(
                                    y=df[y_col] if orientation == "Vertical" else None,
                                    x=None if orientation == "Vertical" else df[y_col],
                                    name=y_col,
                                    orientation='v' if orientation == "Vertical" else 'h',
                                    marker=dict(color=selected_colors[i % len(selected_colors)])
                                ),
                                row=1,
                                col=i+1
                            )
                        else:
                            unique_categories = df[x_col].dropna().unique()
                            try:
                                unique_categories = sorted(unique_categories)
                            except:
                                pass

                            for j, category in enumerate(unique_categories):
                                show_legend = i == 0
                                fig.add_trace(
                                    go.Violin(
                                        y=df[df[x_col] == category][y_col] if orientation == "Vertical" else None,
                                        x=None if orientation == "Vertical" else df[df[x_col] == category][y_col],
                                        name=str(category),
                                        legendgroup=str(category),
                                        orientation='v' if orientation == "Vertical" else 'h',
                                        marker=dict(color=selected_colors[j % len(selected_colors)]),
                                        showlegend=show_legend
                                    ),
                                    row=1,
                                    col=i+1
                                )
                            
                            if show_significance:
                                data_list = [df[df[x_col] == g][y_col].dropna().values for g in unique_categories]
                                group_labels = [str(g) for g in unique_categories]
                                
                                # Compute stats
                                stats_df = vu.compute_pairwise_stats(data_list, group_labels, test_type=test_type, correction_method=correction_method)
                                
                                # Extract significant pairs for plot
                                sig_pairs = []
                                if not stats_df.empty:
                                    for _, row in stats_df.iterrows():
                                        if row["Significant"]:
                                            sig_pairs.append((row["g1_index"], row["g2_index"], row["P-Value (Adj)"]))
                                    sig_pairs.sort(key=lambda x: x[2])
                                
                                vu.add_significance_annotations(fig, data_list, sig_pairs, row=1, col=i+1)
                                
                                # Display Table
                                if sig_pairs:
                                    expander_title = f"**:green[📊 Statistical Results for {y_col} (Significant)]**"
                                else:
                                    expander_title = f"📊 Statistical Results for {y_col}"

                                with st.expander(expander_title, expanded=False):
                                    cols_to_drop = ["g1_index", "g2_index"]
                                    if correction_method == "None":
                                        cols_to_drop.append("P-Value (Adj)")
                                    
                                    display_df = stats_df.drop(columns=cols_to_drop, errors='ignore')
                                    
                                    st.dataframe(
                                        display_df.style.apply(lambda x: ['background-color: #d4edda' if x['Significant'] else '' for _ in x], axis=1)
                                        .format({"P-Value": "{:.4f}", "P-Value (Adj)": "{:.4f}", "Statistic": "{:.2f}"}, na_rep="-"),
                                        width='stretch'
                                    )

                    fig.update_layout(title_text="Violin Plots")
                    fig.update_layout(title_text="Violin Plots")
                    for i in range(1, len(y_cols) + 1):
                        fig.update_xaxes(showgrid=True, gridcolor='lightgray', row=1, col=i)
                        fig.update_yaxes(showgrid=True, gridcolor='lightgray', row=1, col=i)

                        # Ensure consistent tick angle for all subplots
                        if orientation == "Vertical" and x_col != "None":
                            fig.update_xaxes(tickangle=-45, row=1, col=i)
            
            
            elif plot_type == "Scatter Plot":
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_col = st.selectbox("X Variable", suitable_cols["x"])
                with col2:
                    y_col = st.selectbox("Y Variable", suitable_cols["y"])
                with col3:
                    color_col = st.selectbox("Group by (optional)", 
                                           ["None"] + suitable_cols["color"])
                
                c1, c2 = st.columns(2)
                with c1:
                    scatter_mode = st.selectbox("Plot Mode", ["Points", "Regression (Linear)", "Density Contour (KDE)", "Density Contour (Filled)", "Density Heatmap (2D Hist)"])
                with c2:
                    marginal_mode = st.selectbox("Marginal Distribution (Joint Plot)", ["None", "Histogram", "Box", "Violin", "Rug"])
                
                marginal_x = marginal_mode.lower() if marginal_mode != "None" else None
                marginal_y = marginal_mode.lower() if marginal_mode != "None" else None
                
                if scatter_mode == "Points":
                    fig = px.scatter(
                        df, x=x_col, y=y_col,
                        color=None if color_col == "None" else color_col,
                        marginal_x=marginal_x, marginal_y=marginal_y,
                        color_discrete_sequence=selected_color,
                        title=f"Scatter Plot of {y_col} vs {x_col}"
                    )
                elif scatter_mode == "Regression (Linear)":
                    fig = px.scatter(
                        df, x=x_col, y=y_col,
                        color=None if color_col == "None" else color_col,
                        trendline="ols",
                        marginal_x=marginal_x, marginal_y=marginal_y,
                        color_discrete_sequence=selected_color,
                        title=f"Linear Regression: {y_col} vs {x_col}"
                    )
                elif scatter_mode == "Density Contour (KDE)":
                    fig = px.density_contour(
                        df, x=x_col, y=y_col,
                        color=None if color_col == "None" else color_col,
                        marginal_x=marginal_x, marginal_y=marginal_y,
                        color_discrete_sequence=selected_color,
                        title=f"Density Contour of {y_col} vs {x_col}"
                    )
                elif scatter_mode == "Density Contour (Filled)":
                    fig = px.density_contour(
                        df, x=x_col, y=y_col,
                        color=None if color_col == "None" else color_col,
                        marginal_x=marginal_x, marginal_y=marginal_y,
                        color_discrete_sequence=selected_color,
                        title=f"Filled Density Contour of {y_col} vs {x_col}"
                    )
                    
                    for trace in fig.data:
                        if trace.type == 'histogram2dcontour':
                            color = trace.line.color
                            if color:
                                try:
                                    # Create gradient from transparent to opaque color
                                    c_start = vu.adjust_color_to_rgba(color, transparency=0.0)
                                    c_end = vu.adjust_color_to_rgba(color, transparency=0.6)
                                    trace.colorscale = [[0, c_start], [1, c_end]]
                                    trace.contours.coloring = 'fill'
                                    trace.showscale = False
                                except Exception:
                                    pass
                elif scatter_mode == "Density Heatmap (2D Hist)":
                    fig = px.density_heatmap(
                        df, x=x_col, y=y_col,
                        marginal_x=marginal_x, marginal_y=marginal_y,
                        color_continuous_scale=px.colors.sequential.Viridis,
                        title=f"Density Heatmap of {y_col} vs {x_col}"
                    )
            
            elif plot_type == "Line Plot":
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    x_col = st.selectbox("Time Variable", suitable_cols["x"])
                with col2:
                    y_col = st.selectbox("Quantitative Variable", suitable_cols["y"])
                with col3:
                    agg = st.selectbox("Aggregate", ["None", "Mean", "Median", "Sum", "Count"], index=0)
                with col4:
                    color_col = st.selectbox("Group by (optional)", ["None"] + suitable_cols["color"])

                time_granularity = st.selectbox("Time Granularity", ["Hour", "Day", "Month", "Year"], index=1)

                df = vu.convert_to_datetime_or_timedelta(df, x_col)

                agg_mapping = {"None": None, "Mean": "mean", "Median": "median", "Sum": "sum", "Count": "count"}

                if agg_mapping[agg] is None:
                    agg_result = df
                else:
                    agg_result = vu.resample_data(df, x_col, y_col, color_col, time_granularity, agg)

                show_ci = st.checkbox("Add Confidence Interval", False)

                if color_col != "None":
                    if show_ci:
                        fig = go.Figure()
                    else:
                        fig = px.line(
                            agg_result,
                            x=x_col,
                            y=y_col,
                            color=color_col,
                            color_discrete_sequence=selected_color,
                        )
                else:
                    if show_ci:
                        fig = go.Figure()
                    else:
                        fig = px.line(
                            agg_result,
                            x=x_col,
                            y=y_col,
                            color_discrete_sequence=selected_color,
                        )

                if show_ci:
                    confidence_level = st.slider("Confidence Level (%)", min_value=80, max_value=99, value=95, step=1)
                    alpha = 1 - (confidence_level / 100)
                    z_score = stats.norm.ppf(1 - alpha / 2)

                    if color_col != "None":
                        agg_result["y_upper"] = agg_result.groupby(color_col)[y_col].transform(
                            lambda y: y + z_score * (np.std(y) / np.sqrt(len(y)))
                        )
                        agg_result["y_lower"] = agg_result.groupby(color_col)[y_col].transform(
                            lambda y: y - z_score * (np.std(y) / np.sqrt(len(y)))
                        )
                    else:
                        std_err = np.std(agg_result[y_col]) / np.sqrt(len(agg_result[y_col]))
                        agg_result["y_upper"] = agg_result[y_col] + z_score * std_err
                        agg_result["y_lower"] = agg_result[y_col] - z_score * std_err

                    unique_groups = agg_result[color_col].unique() if color_col != "None" else [None]
                    color_map = {group: selected_color[i % len(selected_color)] for i, group in enumerate(unique_groups)}

                    for i, (group_name, group_df) in enumerate(agg_result.groupby(color_col) if color_col != "None" else [(None, agg_result)]):
                        line_color = color_map[group_name]
                        fillcolor = vu.adjust_color_to_rgba(line_color, transparency=0.2)

                        fig.add_trace(go.Scatter(
                            x=group_df[x_col],
                            y=group_df[y_col],
                            mode="lines",
                            name=str(group_name) if group_name else y_col,
                            line=dict(color=line_color, width=2)
                        ))

                        fig.add_trace(go.Scatter(
                            x=pd.concat([group_df[x_col], group_df[x_col][::-1]]),
                            y=pd.concat([group_df["y_upper"], group_df["y_lower"][::-1]]),
                            fill="toself",
                            fillcolor=fillcolor,
                            line=dict(width=0),
                            name=f"{group_name} Confidence Interval" if group_name else "Confidence Interval"
                        ))

                    fig.update_layout(
                        title="Line Plot with Confidence Interval",
                        xaxis_title=x_col,
                        yaxis_title=y_col,
                        template="plotly_white"
                    )

            elif plot_type == "Bar Plot":
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("Category", suitable_cols["x"])
                with col2:
                    color_col = st.selectbox("Group by (optional)",
                                           ["None"] + suitable_cols["color"])
                
                orientation = st.radio("Orientation", ["Vertical", "Horizontal"])
                
                # Count plot
                df_plot = df.groupby([x_col] + ([color_col] if color_col != "None" else [])).size().reset_index(name='Count')
                y_plot = 'Count'
                
                fig = px.bar(
                    df_plot,
                    x=x_col if orientation == "Vertical" else y_plot,
                    y=y_plot if orientation == "Vertical" else x_col,
                    color=None if color_col == "None" else color_col,
                    color_discrete_sequence=selected_color,
                    orientation="v" if orientation == "Vertical" else "h",
                    title=f"Count of {x_col}"
                )
            
            elif plot_type == "Pie Chart":
                names_col = st.selectbox("Categories", suitable_cols["names"])
                
                # Count plot
                df_plot = df[names_col].value_counts().reset_index()
                df_plot.columns = [names_col, 'Count']
                
                fig = px.pie(
                    df_plot,
                    names=names_col,
                    values='Count',
                    color_discrete_sequence=selected_color,
                    title=f"Distribution of {names_col}"
                )

            elif plot_type == "Sunburst Chart":
                st.info("Select multiple categorical variables to define the hierarchy.")
                path_cols = st.multiselect("Select Hierarchy (in order)", suitable_cols["categorical"], default=suitable_cols["categorical"][:2] if len(suitable_cols["categorical"]) >= 2 else suitable_cols["categorical"])
                
                if not path_cols:
                    st.warning("Please select at least one variable for the hierarchy.")
                else:
                    # Missing values handling
                    df_plot = df.copy()
                    for col in path_cols:
                        df_plot[col] = df_plot[col].fillna('Unknown').astype(str)
                    
                    fig = px.sunburst(
                        df_plot,
                        path=path_cols,
                        color_discrete_sequence=selected_color,
                        title=f"Sunburst Chart: {' > '.join(path_cols)}"
                    )

            elif plot_type == "Icicle Chart":
                st.info("Select multiple categorical variables to define the hierarchy.")
                path_cols = st.multiselect("Select Hierarchy (in order)", suitable_cols["categorical"], default=suitable_cols["categorical"][:2] if len(suitable_cols["categorical"]) >= 2 else suitable_cols["categorical"])
                
                if not path_cols:
                    st.warning("Please select at least one variable for the hierarchy.")
                else:
                    # Missing values handling
                    df_plot = df.copy()
                    for col in path_cols:
                        df_plot[col] = df_plot[col].fillna('Unknown').astype(str)
                    
                    fig = px.icicle(
                        df_plot,
                        path=path_cols,
                        color_discrete_sequence=selected_color,
                        title=f"Icicle Chart: {' > '.join(path_cols)}"
                    )
                    fig.update_traces(root_color="lightgrey")
                    fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))

            elif plot_type == "Treemap":
                st.info("Select multiple categorical variables to define the hierarchy.")
                path_cols = st.multiselect("Select Hierarchy (in order)", suitable_cols["categorical"], default=suitable_cols["categorical"][:2] if len(suitable_cols["categorical"]) >= 2 else suitable_cols["categorical"])
                
                if not path_cols:
                    st.warning("Please select at least one variable for the hierarchy.")
                else:
                    # Missing values handling
                    df_plot = df.copy()
                    for col in path_cols:
                        df_plot[col] = df_plot[col].fillna('Unknown').astype(str)
                    
                    fig = px.treemap(
                        df_plot,
                        path=path_cols,
                        color_discrete_sequence=selected_color,
                        title=f"Treemap: {' > '.join(path_cols)}"
                    )
                    fig.update_traces(root_color="lightgrey")
                    fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
            
            elif plot_type == "Correlation Matrix":
                if len(suitable_cols["numeric"]) < 2:
                    st.error("Need at least 2 numeric columns for correlation matrix!")
                    return

                col1, col2 = st.columns(2)
                with col1:
                    include_binary_and_categorical = st.checkbox(
                        "Include Binary/Categorical", value=False
                    )
                
                if include_binary_and_categorical:
                    all_numeric_options = (suitable_cols["numeric"] + suitable_cols["categorical"])
                    corr_methods = ["kendall", "spearman"]
                else:
                    all_numeric_options = suitable_cols["numeric"]
                    corr_methods = ["pearson", "kendall", "spearman"]

                selected_cols = st.multiselect(
                    "Select Columns for Correlation",
                    all_numeric_options,
                    default=all_numeric_options[:min(10, len(all_numeric_options))]
                )

                if len(selected_cols) < 2:
                    st.warning("Please select at least 2 columns")
                    return

                with st.expander("Advanced Options", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        corr_method = st.selectbox("Method", options=corr_methods, index=0)
                        triangle = st.selectbox("Triangle", ["lower", "upper", "full"], index=0)
                    with c2:
                        cluster_rows = st.checkbox("Cluster Rows", value=False)
                        cluster_cols = st.checkbox("Cluster Cols", value=False)
                    with c3:
                        show_row_dendrogram = st.checkbox("Show Row Dendrogram", value=False)
                        show_col_dendrogram = st.checkbox("Show Col Dendrogram", value=False)
                    
                    cmap = st.selectbox("Color Map", ["Viridis", "RdBu", "Plasma", "Inferno", "Magma", "Cividis"], index=1)

                with st.spinner("Generating Correlation Matrix..."):
                    fig = vu.create_correlation_matrix_streamlit(
                        df,
                        selected_cols,
                        method=corr_method,
                        cmap=cmap,
                        triangle=triangle,
                        cluster_rows=cluster_rows,
                        cluster_cols=cluster_cols,
                        show_row_dendrogram=show_row_dendrogram,
                        show_col_dendrogram=show_col_dendrogram
                    )
            
            elif plot_type == "Clustermap":
                st.info("Clustermap: Select features (numeric) to cluster and a sample identifier.")
                
                # Data Format Selection
                data_format = st.radio("Data Format", ["Wide (Samples x Features)", "Long (Sample, Feature, Value)"], horizontal=True)
                
                if "Wide" in data_format:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        value_cols = st.multiselect("Select Features (Numeric Values)", suitable_cols["numeric"])
                    
                    with col2:
                        sample_cols = st.multiselect("Sample Identifier (Columns)", suitable_cols["identifiers"])
                        group_col = st.selectbox("Group Samples by (Optional Color)", ["None"] + suitable_cols["categorical"])
                    
                    # Feature Grouping Options for Wide Format
                    row_group_map = None
                    with st.expander("Advanced Options"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            transformation = st.selectbox("Transformation", ["None", "log", "zscore", "log_zscore"], index=3)
                            cmap = st.selectbox("Color Map", ["vlag", "coolwarm", "viridis", "plasma", "inferno", "magma", "cividis", "RdBu_r"], index=0)
                            
                            st.markdown("---")
                            st.markdown("**Feature Grouping**")
                            row_group_option = st.selectbox("Group Features by", ["None", "Name Pattern (Separator)", "Correlation with Column", "P-value (Sample Group)", "Manual Selection"])
                            
                            row_group_sep = "_"
                            corr_target_col = None
                            corr_threshold = 0.5
                            pval_threshold = 0.05
                            manual_groups = {}
                            
                            if row_group_option == "Name Pattern (Separator)":
                                row_group_sep = st.text_input("Separator", value="_")
                            
                            elif row_group_option == "Correlation with Column":
                                corr_target_col = st.selectbox("Select Target Column", suitable_cols["numeric"])
                                corr_threshold = st.slider("Correlation Threshold", 0.0, 1.0, 0.5, 0.05)
                            
                            elif row_group_option == "P-value (Sample Group)":
                                if group_col == "None":
                                    st.warning("Please select a 'Group Samples by' column first.")
                                else:
                                    pval_threshold = st.number_input("P-value Threshold", 0.001, 0.5, 0.05, 0.001, format="%.3f")
                                    test_method = st.selectbox(
                                        "Statistical Test Method",
                                        ["Auto-Detect (Robust)", "Parametric (ANOVA/T-test)", "Non-Parametric (Kruskal/Mann-Whitney)"],
                                        help="Choose the statistical test to determine feature significance.\n- Auto: Uses Non-Parametric methods (Mann-Whitney/Kruskal-Wallis) which are robust to outliers.\n- Parametric: Uses T-test (2 groups) or ANOVA (>2 groups).\n- Non-Parametric: Uses Mann-Whitney (2 groups) or Kruskal-Wallis (>2 groups)."
                                    )
                                    
                            elif row_group_option == "Manual Selection":
                                n_groups = st.number_input("Number of Groups", 1, 10, 2)
                                remaining_cols = list(value_cols)
                                for i in range(n_groups):
                                    g_name = st.text_input(f"Group {i+1} Name", f"Group {i+1}", key=f"g_name_{i}")
                                    g_cols = st.multiselect(f"Features for {g_name}", remaining_cols, key=f"g_cols_{i}")
                                    if g_cols:
                                        for c in g_cols:
                                            manual_groups[c] = g_name
                                            if c in remaining_cols: remaining_cols.remove(c)
                                # Assign remaining to "Other"
                                for c in remaining_cols:
                                    manual_groups[c] = "Other"

                        with col_b:
                            cluster_cols = st.checkbox("Cluster Columns (Samples)", value=True)
                            cluster_rows = st.checkbox("Cluster Rows (Features)", value=True)
                            show_col_dendrogram = st.checkbox("Show Column Dendrogram", value=False)
                            show_row_dendrogram = st.checkbox("Show Row Dendrogram", value=False)
                            show_sample_labels = st.checkbox("Show Sample Labels", value=True)
                            show_feature_labels = st.checkbox("Show Feature Labels", value=True)
                            font_scale = st.slider("Font Scale", 0.5, 3.0, 1.0, 0.1)

                    if st.button("Generate Clustermap"):
                        if not value_cols:
                            st.warning("Please select at least one feature column.")
                        else:
                            # Prepare Row Group Map
                            if row_group_option == "Name Pattern (Separator)" and row_group_sep:
                                row_group_map = {col: col.split(row_group_sep)[0] for col in value_cols}
                            
                            elif row_group_option == "Correlation with Column" and corr_target_col:
                                # Calculate correlation
                                corrs = df[value_cols].corrwith(df[corr_target_col])
                                row_group_map = {}
                                for col, val in corrs.items():
                                    if val >= corr_threshold:
                                        row_group_map[col] = f"Pos Corr (>{corr_threshold})"
                                    elif val <= -corr_threshold:
                                        row_group_map[col] = f"Neg Corr (<-{corr_threshold})"
                                    else:
                                        row_group_map[col] = "No Corr"
                            
                            elif row_group_option == "P-value (Sample Group)" and group_col != "None":
                                row_group_map = {}
                                # Calculate p-values for each feature against the sample group
                                for col in value_cols:
                                    try:
                                        # Drop NaNs for this specific column and group
                                        temp_df = df[[col, group_col]].dropna()
                                        groups_data = [group[col].values for name, group in temp_df.groupby(group_col)]
                                        
                                        n_groups = len(groups_data)
                                        if n_groups < 2:
                                            row_group_map[col] = "Not Calculated (1 Group)"
                                            continue
                                            
                                        p = 1.0
                                        test_name = ""
                                        
                                        # Determine Test
                                        use_parametric = test_method == "Parametric (ANOVA/T-test)"
                                        
                                        if use_parametric:
                                            if n_groups == 2:
                                                stat, p = stats.ttest_ind(*groups_data, equal_var=False) # Welch's t-test
                                                test_name = "T-test"
                                            else:
                                                stat, p = stats.f_oneway(*groups_data)
                                                test_name = "ANOVA"
                                        else: # Non-Parametric or Auto (Default to robust)
                                            if n_groups == 2:
                                                stat, p = stats.mannwhitneyu(*groups_data)
                                                test_name = "MWU"
                                            else:
                                                stat, p = stats.kruskal(*groups_data)
                                                test_name = "KW"
                                        
                                        if p < pval_threshold:
                                            row_group_map[col] = f"Sig ({test_name}, p<{pval_threshold})"
                                        else:
                                            row_group_map[col] = f"Not Sig (p>={pval_threshold})"
                                    except Exception as e:
                                        row_group_map[col] = "Error"
                                        
                            elif row_group_option == "Manual Selection" and manual_groups:
                                row_group_map = manual_groups

                            with st.spinner("Generating Clustermap..."):
                                fig_clustermap, error = vu.create_clustermap_streamlit(
                                    df, 
                                    value_cols=value_cols, 
                                    sample_col=sample_cols, 
                                    group_col=None if group_col == "None" else group_col,
                                    row_group_map=row_group_map,
                                    transformation=transformation,
                                    cluster_cols=cluster_cols,
                                    cluster_rows=cluster_rows,
                                    show_row_dendrogram=show_row_dendrogram,
                                    show_col_dendrogram=show_col_dendrogram,
                                    show_row_labels=show_feature_labels,
                                    show_col_labels=show_sample_labels,
                                    figsize=None,
                                    font_scale=font_scale,
                                    cmap=cmap
                                )
                                
                                if fig_clustermap:
                                    st.session_state['fig_clustermap'] = fig_clustermap
                                    st.session_state['fig_clustermap_error'] = None
                                else:
                                    st.session_state['fig_clustermap'] = None
                                    st.session_state['fig_clustermap_error'] = error

                else: # Long Format
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        val_col = st.selectbox("Value Column", suitable_cols["numeric"])
                    with col2:
                        feat_col = st.selectbox("Feature Name Column", suitable_cols["categorical"])
                    with col3:
                        samp_col = st.selectbox("Sample ID Column", suitable_cols["identifiers"])
                    
                    col4, col5 = st.columns(2)
                    with col4:
                        samp_group_col = st.selectbox("Group Samples by (Optional)", ["None"] + suitable_cols["categorical"])
                    with col5:
                        feat_group_col = st.selectbox("Group Features by (Optional)", ["None"] + suitable_cols["categorical"])

                    with st.expander("Advanced Options"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            transformation = st.selectbox("Transformation", ["None", "log", "zscore", "log_zscore"], index=3)
                            cmap = st.selectbox("Color Map", ["vlag", "coolwarm", "viridis", "plasma", "inferno", "magma", "cividis", "RdBu_r"], index=0)
                        with col_b:
                            cluster_cols = st.checkbox("Cluster Columns (Samples)", value=True)
                            cluster_rows = st.checkbox("Cluster Rows (Features)", value=True)
                            show_col_dendrogram = st.checkbox("Show Column Dendrogram", value=False)
                            show_row_dendrogram = st.checkbox("Show Row Dendrogram", value=False)
                            show_sample_labels = st.checkbox("Show Sample Labels", value=True)
                            show_feature_labels = st.checkbox("Show Feature Labels", value=True)
                            font_scale = st.slider("Font Scale", 0.5, 3.0, 1.0, 0.1)

                    if st.button("Generate Clustermap"):
                        if not val_col or not feat_col or not samp_col:
                            st.warning("Please select Value, Feature, and Sample columns.")
                        else:
                            with st.spinner("Processing Long Format & Generating Clustermap..."):
                                try:
                                    # Pivot Data
                                    df_pivot = df.pivot_table(index=samp_col, columns=feat_col, values=val_col, aggfunc='mean')
                                    
                                    # Handle Sample Groups
                                    if samp_group_col != "None":
                                        # Create mapping: SampleID -> Group
                                        samp_group_map = df.groupby(samp_col)[samp_group_col].first()
                                        df_pivot[samp_group_col] = df_pivot.index.map(samp_group_map)
                                    
                                    # Handle Feature Groups
                                    row_group_map = None
                                    if feat_group_col != "None":
                                        # Create mapping: FeatureName -> Group
                                        row_group_map = df.groupby(feat_col)[feat_group_col].first().to_dict()

                                    fig_clustermap, error = vu.create_clustermap_streamlit(
                                        df_pivot, 
                                        value_cols=df_pivot.columns.tolist() if samp_group_col == "None" else [c for c in df_pivot.columns if c != samp_group_col], 
                                        sample_col=None, # Index is already set
                                        group_col=None if samp_group_col == "None" else samp_group_col,
                                        row_group_map=row_group_map,
                                        transformation=transformation,
                                        cluster_cols=cluster_cols,
                                        cluster_rows=cluster_rows,
                                        show_row_dendrogram=show_row_dendrogram,
                                        show_col_dendrogram=show_col_dendrogram,
                                        show_row_labels=show_feature_labels,
                                        show_col_labels=show_sample_labels,
                                        figsize=None,
                                        font_scale=font_scale,
                                        cmap=cmap
                                    )
                                    
                                    if fig_clustermap:
                                        st.session_state['fig_clustermap'] = fig_clustermap
                                        st.session_state['fig_clustermap_error'] = None
                                    else:
                                        st.session_state['fig_clustermap'] = None
                                        st.session_state['fig_clustermap_error'] = error
                                except Exception as e:
                                    st.error(f"Error processing data: {e}")

                if 'fig_clustermap' in st.session_state and st.session_state['fig_clustermap']:
                    fig = st.session_state['fig_clustermap']
                elif 'fig_clustermap_error' in st.session_state and st.session_state['fig_clustermap_error']:
                    st.error(f"Failed to generate clustermap: {st.session_state['fig_clustermap_error']}")

            elif plot_type == "Bland-Altman Plot":
                col1, col2 = st.columns(2)
                with col1:
                    method1 = st.selectbox("Method 1 (Reference)", suitable_cols["method1"])
                with col2:
                    method2 = st.selectbox("Method 2 (New)", [c for c in suitable_cols["method2"] if c != method1])
                
                if method1 and method2:
                    fig = vu.create_bland_altman_plot(df, method1, method2)
            
            elif plot_type == "ROC Curve":
                col1, col2 = st.columns(2)
                with col1:
                    true_col = st.selectbox("True Class (Binary)", suitable_cols["true_class"])
                with col2:
                    score_col = st.selectbox("Predicted Score/Probability", suitable_cols["score"])
                
                pos_label = None
                if true_col:
                    uniques = sorted(df[true_col].dropna().unique())
                    if len(uniques) == 2:
                        pos_label = st.selectbox("Positive Label", uniques, index=1)
                    else:
                        st.warning("Selected True Class column must be binary.")
                
                if true_col and score_col and pos_label is not None:
                    fig, error = vu.create_roc_curve(df, true_col, score_col, pos_label)
                    if error:
                        st.error(error)

            if fig is not None:
                with st.expander("Customize Plot"):
                    col1, col2 = st.columns(2)
                    with col1:
                        title = st.text_input("Plot Title", "")
                        title_size = st.slider("Title Font Size", 10, 50, 20)
                        show_grid = st.checkbox("Show Grid", True)
                    with col2:
                        height = st.slider("Plot Height", 400, 2000, 700)
                        width = st.slider("Plot Width", 400, 2000, 1000)
                    
                    fig.update_layout(
                        title=dict(text=title, font=dict(size=title_size), x=0.5),
                        height=height,
                        width=width
                    )

                    if 'y_cols' in locals() and len(y_cols) > 1:
                        for i in range(1, len(y_cols) + 1):
                            fig.update_xaxes(showgrid=show_grid, gridcolor='lightgray', row=1, col=i)
                            fig.update_yaxes(showgrid=show_grid, gridcolor='lightgray', row=1, col=i)
                    else:
                        fig.update_xaxes(showgrid=show_grid, gridcolor='lightgray')
                        fig.update_yaxes(showgrid=show_grid, gridcolor='lightgray')
                

                fig.update_layout(template=theme)

                if theme == "plotly":
                    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="black", size=14))
                elif theme == "plotly_dark":
                    fig.update_layout(paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e", font=dict(color="white", size=14))
                elif theme == "plotly_white":
                    fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(color="black", size=14))
                elif theme == "seaborn":
                    fig.update_layout(paper_bgcolor="#eaf2f8", plot_bgcolor="#eaf2f8", font=dict(color="#333333", size=14))
                elif theme == "ggplot2":
                    fig.update_layout(paper_bgcolor="#ebebeb", plot_bgcolor="#ebebeb", font=dict(color="black", size=14))

                if plot_type not in ["Clustermap", "Correlation Matrix"]:
                    fig = vu.add_custom_hovertemplate(fig, df=df)
                
                st.plotly_chart(fig, width='content')

                with st.expander("Statistical Insights"):
                    if plot_type == "Histogram":
                        summary_statistics = []
                        if color_col != 'None':
                            has_categorical = df[color_col].nunique() > 1
                        else:
                            has_categorical = False

                        if df[x_col].dtype in ['int64', 'float64']:
                            global_stats = {
                                'Category': 'All',
                                'Variable': x_col,
                                'Count': df[x_col].count(),
                                'Mean': df[x_col].mean(),
                                'Std Dev': df[x_col].std(),
                                'Min': df[x_col].min(),
                                '25th Percentile': df[x_col].quantile(0.25),
                                'Median': df[x_col].median(),
                                '75th Percentile': df[x_col].quantile(0.75),
                                'Max': df[x_col].max(),
                                'Skewness': df[x_col].skew(),
                                'Kurtosis': df[x_col].kurtosis(),
                                }
                            summary_statistics.append(global_stats)
                        
                            if has_categorical:
                                for category in df[color_col].unique():
                                    category_data = df[df[color_col] == category][x_col]
                                    sum_stats = {
                                        'Category': category,
                                        'Variable': x_col,
                                        'Count': category_data.count(),
                                        'Mean': category_data.mean(),
                                        'Std Dev': category_data.std(),
                                        'Min': category_data.min(),
                                        '25th Percentile': category_data.quantile(0.25),
                                        'Median': category_data.median(),
                                        '75th Percentile': category_data.quantile(0.75),
                                        'Max': category_data.max(),
                                        'Skewness': category_data.skew(),
                                        'Kurtosis': category_data.kurtosis(),
                                        }
                                    summary_statistics.append(sum_stats)
                        else:
                            st.error(f"{x_col} is None or not in DataFrame columns.")                  
                                    
                        summary_df = pd.DataFrame(summary_statistics)
                        pivot_df = summary_df.pivot_table(
                            index='Variable',
                            columns='Category',
                            values=['Count', 'Mean', 'Std Dev', 'Min', '25th Percentile', 'Median', '75th Percentile', 'Max', 'Skewness', 'Kurtosis'],
                            aggfunc='first'
                        )

                        if color_col != 'None':
                            summary_df = pivot_df[['Count', 'Mean', 'Std Dev', 'Min', '25th Percentile', 'Median', '75th Percentile', 'Max', 'Skewness', 'Kurtosis']]
                        else:
                            summary_df = summary_df.drop(columns='Category').set_index('Variable')

                        summary_df = (
                            summary_df.style
                            .format("{:.2f}")
                            .set_table_attributes('style="font-size: 14px; border-collapse: collapse; width: 100%;"')
                            .set_caption("Summary Statistics by Variable and Category")
                            .apply(lambda x: ['background: lightgrey' if i % 2 == 0 else '' for i in range(len(x))], axis=0)
                        )

                        st.write(f"Summary Statistics for {x_col}:")
                        st.dataframe(summary_df, width='stretch')
                        
                    elif plot_type in ["Box Plot", "Violin Plot"]:
                        summary_statistics = []
                        if x_col != 'None':
                            has_categorical = df[x_col].nunique() > 1
                        else:
                            has_categorical = False

                        for y_col in y_cols:
                            if df[y_col].dtype in ['int64', 'float64']:
                                global_stats = {
                                    'Category': 'All',
                                    'Variable': y_col,
                                    'Count': df[y_col].count(),
                                    'Mean': df[y_col].mean(),
                                    'Std Dev': df[y_col].std(),
                                    'Min': df[y_col].min(),
                                    '25th Percentile': df[y_col].quantile(0.25),
                                    'Median': df[y_col].median(),
                                    '75th Percentile': df[y_col].quantile(0.75),
                                    'Max': df[y_col].max(),
                                    'Skewness': df[y_col].skew(),
                                    'Kurtosis': df[y_col].kurtosis(),
                                }
                                summary_statistics.append(global_stats)
                            else:
                                st.error(f"{y_col} is None or not in DataFrame columns.")
                                
                        if has_categorical:
                            for y_col in y_cols:
                                if df[y_col].dtype in ['int64', 'float64']:
                                    for category in df[x_col].unique():
                                        category_data = df[df[x_col] == category][y_col]
                                        sum_stats = {
                                            'Category': category,
                                            'Variable': y_col,
                                            'Count': category_data.count(),
                                            'Mean': category_data.mean(),
                                            'Std Dev': category_data.std(),
                                            'Min': category_data.min(),
                                            '25th Percentile': category_data.quantile(0.25),
                                            'Median': category_data.median(),
                                            '75th Percentile': category_data.quantile(0.75),
                                            'Max': category_data.max(),
                                            'Skewness': category_data.skew(),
                                            'Kurtosis': category_data.kurtosis(),
                                        }
                                        summary_statistics.append(sum_stats)

                        summary_df = pd.DataFrame(summary_statistics)
                        pivot_df = summary_df.pivot_table(
                            index='Variable',
                            columns='Category',
                            values=['Count', 'Mean', 'Std Dev', 'Min', '25th Percentile', 'Median', '75th Percentile', 'Max', 'Skewness', 'Kurtosis'],
                            aggfunc='first'
                        )

                        if x_col != 'None':
                            summary_df = pivot_df[['Count', 'Mean', 'Std Dev', 'Min', '25th Percentile', 'Median', '75th Percentile', 'Max', 'Skewness', 'Kurtosis']]
                        else:
                            summary_df = summary_df.drop(columns='Category').set_index('Variable')

                        summary_df = (
                            summary_df.style
                            .format("{:.2f}")
                            .set_table_attributes('style="font-size: 14px; border-collapse: collapse; width: 100%;"')
                            .set_caption("Summary Statistics by Variable and Category")
                            .apply(lambda x: ['background: lightgrey' if i % 2 == 0 else '' for i in range(len(x))], axis=0)
                        )

                        st.write("Summary Statistics (Cross-Tabulated):")
                        st.dataframe(summary_df, width='stretch')
                    
                    elif plot_type == "Scatter Plot":
                        if df[x_col].dtype in ['int64', 'float64'] and df[y_col].dtype in ['int64', 'float64']:
                            correlation = df[x_col].corr(df[y_col])
                            st.write(f"Correlation coefficient: {correlation:.2f}")
                            
                            slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_col], df[y_col])
                            st.write(f"R-squared: {r_value**2:.2f}")
                            st.write(f"P-value: {p_value:.4f}")

                    elif plot_type == "Bland-Altman Plot":
                        diff = df[method1] - df[method2]
                        md = diff.mean()
                        sd = diff.std()
                        st.write(f"**Mean Difference (Bias):** {md:.4f}")
                        st.write(f"**Standard Deviation of Difference:** {sd:.4f}")
                        st.write(f"**Limits of Agreement (95%):** [{md - 1.96*sd:.4f}, {md + 1.96*sd:.4f}]")
                        
                        # T-test for bias
                        t_stat, p_val = stats.ttest_1samp(diff, 0)
                        st.write(f"**T-test for Bias (H0: Mean Diff = 0):** p-value = {p_val:.4f}")
                        if p_val < 0.05:
                            st.write("⚠️ Significant bias detected.")
                        else:
                            st.write("✅ No significant bias detected.")

                    elif plot_type == "ROC Curve":
                        from sklearn.metrics import roc_curve, auc
                        y_true = df[true_col].dropna()
                        y_score = df[score_col].dropna()
                        # Align indices
                        common_idx = y_true.index.intersection(y_score.index)
                        y_true = y_true.loc[common_idx]
                        y_score = y_score.loc[common_idx]
                        
                        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label)
                        roc_auc = auc(fpr, tpr)
                        
                        st.write(f"**Area Under the Curve (AUC):** {roc_auc:.4f}")
                        
                        # Youden's J statistic
                        J = tpr - fpr
                        ix = np.argmax(J)
                        best_thresh = thresholds[ix]
                        st.write(f"**Best Threshold (Youden's J):** {best_thresh:.4f}")
                        st.write(f"**Sensitivity at Best Threshold:** {tpr[ix]:.4f}")
                        st.write(f"**Specificity at Best Threshold:** {1-fpr[ix]:.4f}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try different variables or plot settings.")