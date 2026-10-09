import re
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.figure_factory as ff
import plotly.graph_objects as go
import statsmodels.stats.multitest as smt
import streamlit as st
from matplotlib.colors import to_rgba
from scipy import stats
from scipy.stats import zscore


def compute_pairwise_stats(data_list, group_labels, test_type="Auto-Detect", correction_method="None", alpha=0.05):
    """
    Compute pairwise comparisons and return a detailed DataFrame.
    """
    n_groups = len(data_list)
    results = []
    
    # Helper to get p-value and stat
    def calculate_stat_p(x, y):
        try:
            if test_type == "Force Parametric (T-test/ANOVA)":
                stat, p = stats.ttest_ind(x, y, nan_policy='omit')
                test_name = "T-test"
            elif test_type == "Force Non-Parametric (Mann-Whitney/Kruskal)":
                stat, p = stats.mannwhitneyu(x, y, alternative='two-sided')
                test_name = "Mann-Whitney"
            else:
                # Auto-Detect: Default to Mann-Whitney U
                stat, p = stats.mannwhitneyu(x, y, alternative='two-sided')
                test_name = "Mann-Whitney"
            return stat, p, test_name
        except Exception:
            return np.nan, 1.0, "Error"

    # 1. Compute all stats
    raw_p_values = []
    temp_results = []

    for i, j in combinations(range(n_groups), 2):
        x, y = data_list[i], data_list[j]
        if len(x) > 0 and len(y) > 0:
            stat, p, test_name = calculate_stat_p(x, y)
            raw_p_values.append(p)
            temp_results.append({
                "Group 1": group_labels[i],
                "Group 2": group_labels[j],
                "Test": test_name,
                "Statistic": stat,
                "P-Value": p,
                "g1_index": i,
                "g2_index": j
            })

    if not temp_results:
        return pd.DataFrame()

    # 2. Apply Correction
    if correction_method == "Bonferroni":
        _, p_corrected, _, _ = smt.multipletests(raw_p_values, alpha=alpha, method='bonferroni')
    elif correction_method == "Benjamini-Hochberg (FDR)":
        _, p_corrected, _, _ = smt.multipletests(raw_p_values, alpha=alpha, method='fdr_bh')
    else:
        p_corrected = raw_p_values

    # 3. Build Final DataFrame
    for res, p_adj in zip(temp_results, p_corrected):
        res["P-Value (Adj)"] = p_adj
        res["Significant"] = p_adj < alpha
        res["Significance Level"] = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
        results.append(res)

    return pd.DataFrame(results)

def compute_significant_pairs(data_list, test_type="Auto-Detect", correction_method="None", alpha=0.05, p_value_func=None):
    """
    Wrapper for backward compatibility. Returns list of (i, j, p_adj).
    """
    # Generate dummy labels
    group_labels = [str(i) for i in range(len(data_list))]
    
    df_results = compute_pairwise_stats(data_list, group_labels, test_type, correction_method, alpha)
    
    if df_results.empty:
        return []
        
    sig_pairs = []
    for _, row in df_results.iterrows():
        if row["Significant"]:
            sig_pairs.append((row["g1_index"], row["g2_index"], row["P-Value (Adj)"]))
            
    # Sort by p-value
    sig_pairs.sort(key=lambda x: x[2])
    return sig_pairs


def create_clustermap_streamlit(df, value_cols, sample_col=None, group_col=None, row_group_map=None, transformation='log_zscore', 
                                cluster_cols=True, cluster_rows=True, 
                                show_row_dendrogram=False, show_col_dendrogram=False,
                                show_row_labels=True, show_col_labels=True, figsize=None, font_scale=1.0, cmap='vlag'):
    """
    Generates a clustermap figure for Streamlit using Plotly.
    Supports wide format:
    - value_cols: List of numeric columns (Features)
    - sample_col: Column identifying samples (Columns of heatmap)
    - group_col: Column for grouping/coloring samples
    - row_group_map: Dict or Series mapping feature names to groups (for row coloring)
    - figsize: Tuple (width, height) or None for auto-size (interpreted as relative scale for Plotly)
    - font_scale: Scaling factor for fonts
    - cmap: Colormap name
    """
    # Prepare data
    if sample_col:
        # Handle multiple columns for sample identifier
        if isinstance(sample_col, list):
            if not sample_col: # Empty list
                df_indexed = df.copy()
                df_indexed.index = df_indexed.index.astype(str)
            else:
                # Check for duplicates in the combination of columns
                if df[sample_col].duplicated().any():
                    return None, f"Sample columns '{sample_col}' contain duplicates. Clustermap requires unique samples."
                
                # Create a combined index for display
                df_indexed = df.copy()
                df_indexed.index = df[sample_col].astype(str).agg('_'.join, axis=1)
        else:
            # Single column
            if df[sample_col].duplicated().any():
                 return None, f"Sample column '{sample_col}' contains duplicates. Clustermap requires unique samples."
            df_indexed = df.set_index(sample_col)
    else:
        df_indexed = df.copy()
        df_indexed.index = df_indexed.index.astype(str)

    # Select values
    try:
        # Ensure value_cols is a list
        if isinstance(value_cols, str):
            value_cols = [value_cols]
        
        df_values = df_indexed[value_cols]
        
        # Drop rows with NaNs in value columns to avoid issues
        df_values = df_values.dropna()
        
        if df_values.empty:
            # Diagnostic for incompatible columns
            msg = "No data available after dropping NaNs."
            
            # 1. Check for individual empty columns
            col_counts = {col: df_indexed[col].count() for col in value_cols}
            empty_cols = [col for col, count in col_counts.items() if count == 0]
            
            if empty_cols:
                msg += f"\n\nThe following columns are entirely empty/NaN: {', '.join(empty_cols)}"
            else:
                # 2. Check for incompatible subsets (Minimal Conflict Search)
                incompatible_tuples = []
                import itertools
                
                # We search for the smallest subsets of columns that have no intersection.
                # Start with k=2 (pairs), then k=3, etc.
                # Once we find conflicts at size k, we report them and stop (minimal diagnostics).
                
                # Pre-compute masks for performance
                masks = {col: df_indexed[col].notna().to_numpy() for col in value_cols}
                
                found_conflict = False
                # Limit search depth to avoid performance hit on large lists
                max_k = min(len(value_cols), 5) 
                
                for k in range(2, max_k + 1):
                    # For large N, skipping check if combinations are too huge?
                    # 20C5 is ~15k, which is fast in numpy. 30C5 is ~142k.
                    # Just hard cap operations if needed? 
                    # For now, let's trust max_k=5 is safe enough for typical usage.
                    
                    for combo in itertools.combinations(value_cols, k):
                        # Calculate intersection
                        # Start with first
                        common = masks[combo[0]]
                        short_circuit = False
                        
                        # Iteratively intersect
                        for c_next in combo[1:]:
                            common = common & masks[c_next]
                            if not common.any():
                                short_circuit = True
                                break
                        
                        if short_circuit or not common.any():
                            incompatible_tuples.append(combo)
                            found_conflict = True
                    
                    if found_conflict:
                        break # Found minimal conflicts at this level, stop searching deeper
                
                if incompatible_tuples:
                    msg += "\n\nThe following minimal subsets of variables have no overlapping data (incompatible):\n"
                    for tup in incompatible_tuples[:10]: # Limit output
                         msg += f"- {', '.join(tup)}\n"
                    if len(incompatible_tuples) > 10:
                        msg += f"... and {len(incompatible_tuples)-10} others."
                elif len(value_cols) > 1:
                     msg += f"\n\nVariable combination is incompatible (common intersection is empty), but no simple subset (size <= {max_k}) was found to be exclusively empty."
            
            # 3. Add counts summary
            msg += "\n\nNon-null value counts per column:"
            # Sort by count ascending to highlight problematic ones
            sorted_counts = sorted(col_counts.items(), key=lambda item: item[1])
            for col, count in sorted_counts[:10]: # Show top 10 worst
                msg += f"\n- {col}: {count}"
            if len(sorted_counts) > 10:
                msg += f"\n... (and {len(sorted_counts)-10} more)"

            return None, msg
            
    except KeyError as e:
        return None, f"Missing columns: {e}"

    # Transpose: We want Features (value_cols) as Rows, Samples as Columns
    df_to_plot_base = df_values.T
    
    # Transformation
    if 'log' in transformation:
        min_val = df_to_plot_base[df_to_plot_base > 0].min().min()
        if pd.isna(min_val): min_val = 1e-9
        df_transformed = np.log10(df_to_plot_base + min_val)
    else:
        df_transformed = df_to_plot_base.copy()

    if 'zscore' in transformation:
        # Apply zscore per row (axis=1) -> Standardize each Feature across Samples
        # Use result_type='expand' to ensure we get a DataFrame, not a Series of arrays
        df_to_plot = df_transformed.apply(zscore, axis=1, result_type='expand')
        df_to_plot.columns = df_transformed.columns # Restore Sample IDs as columns
        
        # Check for NaNs (which happen if standard deviation is 0, i.e., constant value)
        if df_to_plot.isna().any().any():
            # Identify constant features
            constant_features = df_to_plot.index[df_to_plot.isna().any(axis=1)].tolist()
            # Drop them
            df_to_plot = df_to_plot.dropna()
            
            # If everything is gone, return specific error
            if df_to_plot.empty:
                return None, f"All selected features are constant (zero variance) across the selected samples and cannot be standardized (Z-score).\n\nConstant features: {', '.join(constant_features)}"
        
        center = 0
        cbar_label = 'Z-score'
    else:
        df_to_plot = df_transformed.dropna()
        center = np.median(df_to_plot.values) if df_to_plot.size > 0 else 0
        cbar_label = 'Value'

    if df_to_plot.empty:
        return None, "No data available after transformation."

    # Initialize clustering variables
    row_dendro_traces = []
    col_dendro_traces = []
    row_dendro_range = None
    col_dendro_range = None
    row_order = df_to_plot.index.tolist()
    col_order = df_to_plot.columns.tolist()
    row_dendro_y_vals = None
    col_dendro_x_vals = None

    # 1. Row Clustering (Features)
    if cluster_rows and df_to_plot.shape[0] > 1:
        try:
            # orientation='left' -> leaves on left (facing heatmap)
            dendro_side = ff.create_dendrogram(df_to_plot, orientation='left', labels=df_to_plot.index)
            row_order = dendro_side['layout']['yaxis']['ticktext']
            row_dendro_y_vals = dendro_side['layout']['yaxis']['tickvals']
            
            # Reorder df
            df_to_plot = df_to_plot.loc[row_order]
            
            if show_row_dendrogram:
                row_dendro_traces = dendro_side['data']
                # Calculate max range for tight layout
                max_d = 0
                for trace in row_dendro_traces:
                    if 'x' in trace and len(trace['x']) > 0:
                        max_d = max(max_d, max(trace['x']))
                row_dendro_range = [0, max_d]
        except Exception as e:
            st.warning(f"Row clustering failed: {e}")

    # 2. Column Clustering (Samples)
    if cluster_cols and df_to_plot.shape[1] > 1:
        try:
            # orientation='bottom' -> leaves on bottom (facing heatmap)
            dendro_top = ff.create_dendrogram(df_to_plot.T, orientation='bottom', labels=df_to_plot.columns)
            col_order = dendro_top['layout']['xaxis']['ticktext']
            col_dendro_x_vals = dendro_top['layout']['xaxis']['tickvals']
            
            # Reorder df
            df_to_plot = df_to_plot[col_order]
            
            if show_col_dendrogram:
                col_dendro_traces = dendro_top['data']
                # Calculate max range for tight layout
                max_d = 0
                for trace in col_dendro_traces:
                    if 'y' in trace and len(trace['y']) > 0:
                        max_d = max(max_d, max(trace['y']))
                col_dendro_range = [0, max_d]
        except Exception as e:
            st.warning(f"Column clustering failed: {e}")

    # Grouping (if not clustered, sort by group)
    groups = None
    if group_col and group_col in df.columns:
        # Ensure df_to_plot is a DataFrame
        if isinstance(df_to_plot, pd.Series):
            df_to_plot = df_to_plot.to_frame()

        groups = df_indexed.loc[df_to_plot.columns, group_col]
        if not cluster_cols:
            groups = groups.sort_values()
            df_to_plot = df_to_plot[groups.index]
            # Re-fetch groups to match sorted order
            groups = df_indexed.loc[df_to_plot.columns, group_col]
        else:
            # If clustered, just align groups to the new order
            groups = groups.loc[df_to_plot.columns]

    # Row Grouping
    row_groups = None
    if row_group_map:
        # row_group_map should be a dict or Series: FeatureName -> Group
        row_groups = pd.Series(row_group_map)
        # Align with current row order
        row_groups = row_groups.reindex(df_to_plot.index)
        
        # If not clustered, we might want to sort by row group?
        # But usually clustering takes precedence.
        # If not clustered, let's sort by row group if provided
        if not cluster_rows:
            row_groups = row_groups.sort_values()
            df_to_plot = df_to_plot.loc[row_groups.index]
            row_groups = row_groups.loc[df_to_plot.index]
            row_order = df_to_plot.index.tolist()
            y_vals = row_order # Update y_vals if not clustered

    # Map cmap to Plotly
    cmap_map = {
        'vlag': 'RdBu', 'coolwarm': 'RdBu', 'RdBu_r': 'RdBu_r',
        'viridis': 'Viridis', 'plasma': 'Plasma', 'inferno': 'Inferno', 
        'magma': 'Magma', 'cividis': 'Cividis'
    }
    plotly_cmap = cmap_map.get(cmap, 'RdBu')
    if cmap in ['vlag', 'coolwarm']: 
        plotly_cmap = 'RdBu_r'

    # Create Figure
    try:
        fig = go.Figure()
        
        # Determine Layout Domains
        # X-Axis: Heatmap [0, hm_x_end] | Row Color Bar [hm_x_end, hm_x_end + 0.05] | Row Dendrogram [start, 1.0]
        
        # Calculate horizontal space allocation
        row_group_width = 0.05 if row_groups is not None else 0.0
        row_dendro_width = 0.15 if show_row_dendrogram else 0.0
        
        hm_x_end = 1.0 - row_group_width - row_dendro_width
        hm_x_domain = [0, hm_x_end]
        
        # Y-Axis: Heatmap [0, hm_y_end] | Group Bar [hm_y_end, hm_y_end + 0.05] | Col Dendrogram [start, 1.0]
        # Calculate vertical space allocation
        group_height = 0.05 if groups is not None else 0.0
        dendro_height = 0.15 if show_col_dendrogram else 0.0
        
        hm_y_end = 1.0 - group_height - dendro_height
        hm_y_domain = [0, hm_y_end]
        
        # Add Main Heatmap
        zmin = -3 if 'zscore' in transformation else None
        zmax = 3 if 'zscore' in transformation else None
        
        # Determine Heatmap Axes Values
        x_vals = col_dendro_x_vals if show_col_dendrogram else col_order
        y_vals = row_dendro_y_vals if show_row_dendrogram else row_order
        
        # Calculate explicit ranges for Heatmap to ensure it touches dendrograms
        hm_x_range = None
        hm_y_range = None
        
        if show_col_dendrogram and col_dendro_x_vals is not None and len(col_dendro_x_vals) > 0:
            vals = sorted(col_dendro_x_vals)
            step = (vals[1] - vals[0]) if len(vals) > 1 else 10.0
            hm_x_range = [min(vals) - step/2, max(vals) + step/2]
            
        if show_row_dendrogram and row_dendro_y_vals is not None and len(row_dendro_y_vals) > 0:
            vals = sorted(row_dendro_y_vals)
            step = (vals[1] - vals[0]) if len(vals) > 1 else 10.0
            hm_y_range = [min(vals) - step/2, max(vals) + step/2]

        fig.add_trace(go.Heatmap(
            z=df_to_plot.values,
            x=x_vals,
            y=y_vals,
            colorscale=plotly_cmap,
            zmid=center if 'zscore' not in transformation else 0,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title=cbar_label, len=0.4, y=0.4),
            xaxis='x',
            yaxis='y'
        ))
        
        # Add Group Color Bar
        if groups is not None:
            unique_groups = groups.unique()
            unique_groups = [g for g in unique_groups if pd.notna(g)]
            colors = pc.qualitative.Plotly * 10
            group_color_map = {g: colors[i % len(colors)] for i, g in enumerate(unique_groups)}
            group_to_int = {g: i for i, g in enumerate(unique_groups)}
            n_groups = len(unique_groups)
            
            cscale = []
            if n_groups > 0:
                for i, g in enumerate(unique_groups):
                    color = group_color_map[g]
                    norm_start = i / n_groups
                    norm_end = (i + 1) / n_groups
                    cscale.append([norm_start, color])
                    cscale.append([norm_end, color])
            else:
                cscale = [[0, 'grey'], [1, 'grey']]

            fig.add_trace(go.Heatmap(
                z=[[group_to_int.get(g, -1) for g in groups]], 
                x=x_vals,
                y=[0], # Dummy y
                colorscale=cscale,
                zmin=0,
                zmax=n_groups,
                showscale=False,
                hoverinfo='text',
                text=[[str(g) for g in groups]],
                xaxis='x', # Share x with main heatmap
                yaxis='y3' # Separate y-axis
            ))
            
            # Add Dummy Traces for Legend
            for g_name in unique_groups:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=10, color=group_color_map[g_name]),
                    name=str(g_name),
                    showlegend=True,
                    legendgroup="groups"
                ))

        # Add Row Group Color Bar (Rows)
        if row_groups is not None:
            unique_row_groups = row_groups.unique()
            unique_row_groups = [g for g in unique_row_groups if pd.notna(g)]
            colors = pc.qualitative.Set2 * 10 # Use different palette
            row_group_color_map = {g: colors[i % len(colors)] for i, g in enumerate(unique_row_groups)}
            row_group_to_int = {g: i for i, g in enumerate(unique_row_groups)}
            n_row_groups = len(unique_row_groups)
            
            cscale_row = []
            if n_row_groups > 0:
                for i, g in enumerate(unique_row_groups):
                    color = row_group_color_map[g]
                    norm_start = i / n_row_groups
                    norm_end = (i + 1) / n_row_groups
                    cscale_row.append([norm_start, color])
                    cscale_row.append([norm_end, color])
            else:
                cscale_row = [[0, 'grey'], [1, 'grey']]

            # Transpose for vertical bar
            z_vals = [[row_group_to_int.get(g, -1)] for g in row_groups]
            
            fig.add_trace(go.Heatmap(
                z=z_vals,
                x=[0], # Dummy x
                y=y_vals,
                colorscale=cscale_row,
                zmin=0,
                zmax=n_row_groups,
                showscale=False,
                hoverinfo='text',
                text=[[str(g)] for g in row_groups],
                xaxis='x3', # Separate x-axis
                yaxis='y'   # Share y with main heatmap
            ))
            
            # Add Dummy Traces for Legend (Row Groups)
            for g_name in unique_row_groups:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=10, symbol='square', color=row_group_color_map[g_name]),
                    name=str(g_name),
                    showlegend=True,
                    legendgroup="row_groups"
                ))

        # Add Row Dendrogram Traces
        for trace in row_dendro_traces:
            trace.xaxis = 'x2'
            trace.yaxis = 'y' # Share y-axis with heatmap
            trace.showlegend = False
            fig.add_trace(trace)

        # Add Column Dendrogram Traces
        for trace in col_dendro_traces:
            trace.xaxis = 'x' # Share x-axis with heatmap
            trace.yaxis = 'y2'
            trace.showlegend = False
            fig.add_trace(trace)

        # Layout Configuration
        width_px = figsize[0] * 50 if figsize else 800
        height_px = figsize[1] * 50 if figsize else 800
        
        # Axis Configurations
        xaxis_config = dict(
            domain=hm_x_domain, 
            tickmode='array', 
            tickvals=x_vals, 
            ticktext=col_order, 
            tickangle=45,
            title='',
            automargin=True,
            showticklabels=show_col_labels,
            showgrid=False,
            zeroline=False
        )
        if hm_x_range: xaxis_config['range'] = hm_x_range
        
        yaxis_config = dict(
            domain=hm_y_domain, 
            tickmode='array', 
            tickvals=y_vals, 
            ticktext=row_order, 
            side='left',
            title='',
            automargin=True,
            showticklabels=show_row_labels,
            showgrid=False,
            zeroline=False
        )
        if hm_y_range: yaxis_config['range'] = hm_y_range

        layout_args = dict(
            width=width_px,
            height=height_px,
            font=dict(size=12 * font_scale),
            xaxis=xaxis_config,
            yaxis=yaxis_config,
            
            # Row Dendrogram Axis (Right)
            xaxis2=dict(
                domain=[1.0 - row_dendro_width, 1.0], 
                range=row_dendro_range,
                showgrid=False, 
                showline=False, 
                showticklabels=False,
                zeroline=False,
                title=''
            ),
            
            # Row Group Color Bar Axis (Between Heatmap and Row Dendrogram)
            xaxis3=dict(
                domain=[hm_x_end, hm_x_end + row_group_width],
                showgrid=False,
                showline=False,
                showticklabels=False,
                zeroline=False,
                title=''
            ),
            
            # Column Dendrogram Axis (Top)
            yaxis2=dict(
                domain=[1.0 - dendro_height, 1.0], 
                range=col_dendro_range,
                showgrid=False, 
                showline=False, 
                showticklabels=False,
                zeroline=False,
                title=''
            ),
            
            # Group Color Bar Axis (Between Heatmap and Col Dendrogram)
            yaxis3=dict(
                domain=[hm_y_end, hm_y_end + group_height],
                showgrid=False,
                showline=False,
                showticklabels=False,
                title=''
            ),
            
            legend=dict(title=dict(text=group_col)) if group_col else None
        )
        
        fig.update_layout(**layout_args)
        
        return fig, None
    except Exception as e:
        return None, str(e)

def add_significance_annotations(fig, data_list, significant_pairs, row=None, col=None):
    """
    Adds significance brackets and stars to an existing figure.
    """
    non_empty_data = [d for d in data_list if len(d) > 0]
    if not non_empty_data:
        return 
    
    max_y = max([max(d) for d in non_empty_data])
    min_y = min([min(d) for d in non_empty_data])
    y_range = max_y - min_y
    if y_range == 0:
        y_range = abs(max_y) * 0.1 if max_y != 0 else 1.0

    # Calculate y-positions for brackets to avoid overlap
    step = y_range * 0.07
    base_y = max_y + y_range * 0.05

    for idx, (i, j, p) in enumerate(significant_pairs):
        y_bracket = base_y + step * idx
        y_star = y_bracket + y_range * 0.01  # Star above bracket

        # Draw bracket
        fig.add_shape(
            type="line",
            x0=i, x1=j,
            y0=y_bracket, y1=y_bracket,
            line=dict(color="black", width=1),
            layer="above",
            row=row, col=col
        )
        fig.add_shape(
            type="line",
            x0=i, x1=i,
            y0=y_bracket - y_range * 0.01, y1=y_bracket,
            line=dict(color="black", width=1),
            layer="above",
            row=row, col=col
        )
        fig.add_shape(
            type="line",
            x0=j, x1=j,
            y0=y_bracket - y_range * 0.01, y1=y_bracket,
            line=dict(color="black", width=1),
            layer="above",
            row=row, col=col
        )

        # Add star above bracket
        if p < 0.001:
            sig_text = f"*** (p={p:.3e})"
        elif p < 0.01:
            sig_text = f"** (p={p:.3f})"
        elif p < 0.05:
            sig_text = f"* (p={p:.3f})"
        else:
            sig_text = "ns"

        fig.add_annotation(
            x=(i + j) / 2,
            y=y_star,
            text=sig_text,
            showarrow=False,
            font=dict(size=12, color="black"),
            yanchor="bottom",
            xanchor="center",
            row=row, col=col
        )

def create_boxplot_with_significance_streamlit(df, group_col, value_col, significant_pairs=None, title="Box Plot with Significance", y_label="Value", palette=None, width=800, height=500, font_family="Arial", font_size=12, template="plotly_white"):
    """
    Create box plot with significance stars for Streamlit.
    Uses df, group_col, value_col instead of lists.
    Significant pairs are pre-computed externally.
    """
    # Get unique groups
    groups = df[group_col].dropna().unique()
    if len(groups) < 2:
        return None  # Not enough groups

    # Create data_list and group_labels
    data_list = [df[df[group_col] == g][value_col].dropna().values for g in groups]
    group_labels = [str(g) for g in groups]

    # Create figure
    fig = go.Figure()

    for i, data in enumerate(data_list):
        color = palette[i] if palette and i < len(palette) else None
        fig.add_trace(go.Box(
            y=data,
            name=group_labels[i],
            marker_color=color,
            boxmean=True,
            line=dict(width=1.5),
            boxpoints='outliers',
            jitter=0,          # Centered outliers
            pointpos=0,        # Centered outliers
            showlegend=False
        ))

    # Calculate ranges for annotations (significance or sample size)
    non_empty_data = [d for d in data_list if len(d) > 0]
    if not non_empty_data:
        return fig 
    
    max_y = max([max(d) for d in non_empty_data])
    min_y = min([min(d) for d in non_empty_data])
    y_range = max_y - min_y
    if y_range == 0:
        y_range = abs(max_y) * 0.1 if max_y != 0 else 1.0

    # Add significance annotations
    add_significance_annotations(fig, data_list, significant_pairs)

    # Add sample sizes
    for i, data in enumerate(data_list):
        fig.add_annotation(
            x=i,
            y=min_y - y_range * 0.05,
            text=f"n={len(data)}",
            showarrow=False,
            font=dict(size=10, color="gray"),
            yanchor="top",
            xanchor="center",
            bgcolor="rgba(255,255,255,0.7)"
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=font_size + 2, family=font_family), x=0.5),
        yaxis_title=y_label,
        xaxis=dict(tickmode='array', tickvals=list(range(len(groups))), ticktext=group_labels, tickfont=dict(size=font_size)),
        showlegend=False,
        template=template,
        width=width,
        height=height,
        margin=dict(l=60, r=60, t=80, b=80),
        font=dict(family=font_family, size=font_size)
    )

    return fig


def adjust_color_to_rgba(color, transparency=0.2):
    """
    Converts a color (hex or rgb format) to an RGBA string with the specified transparency.
    """
    if isinstance(color, str):
        if color.startswith('#'):  # Hex format
            rgba = to_rgba(color, transparency)
        elif color.startswith('rgb'):  # RGB format (e.g., "rgb(228,26,28)")
            rgb_values = list(map(int, re.findall(r'\d+', color)))
            rgba = (*[v / 255 for v in rgb_values], transparency)
        else:
            rgba = to_rgba(color, transparency)  # Named color
    else:
        rgba = to_rgba(color, transparency)

    return f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})"

def get_figure_column_names(fig, df):
    """
    Extracts the column names or data-related attributes from a Plotly figure
    and checks if they match any columns in the provided DataFrame (df).
    """
    columns = set()
    df_columns = set(df.columns)

    for trace in fig.data:
        for attr in ['x', 'y', 'z', 'text', 'labels', 'values']:
            if attr in trace:
                data = trace[attr]
                if isinstance(data, (list, np.ndarray, pd.Series)):
                    if isinstance(data, pd.Series):
                        if data.name in df_columns:
                            columns.add(data.name)
                    elif isinstance(data, (list, np.ndarray)):
                        for col in df_columns:
                            if np.array_equal(data, df[col].values):
                                columns.add(col)
                                break

        if 'name' in trace and trace['name'] in df_columns:
            columns.add(trace['name'])

    return columns

def add_custom_hovertemplate(fig, df):
    """
    Adds a custom hovertemplate to an existing Plotly figure, showing only selected columns on hover.
    """
    selected_cols = st.multiselect(
        "Select columns to display on hover",
        options=sorted(df.columns)
    )

    if len(selected_cols) >= 1:
        for trace in fig.data:
            hovertemplate = trace.get('hovertemplate', "") or ""
            for col in selected_cols: 
                hovertemplate += f"<br>{col}: %{{customdata[{selected_cols.index(col)}]}}"
        
        customdata = df[selected_cols].map(lambda x: 'Not Available' if pd.isna(x) else x)

        fig.update_traces(
            customdata=customdata.values,
            hovertemplate=hovertemplate
        )
    return fig

def convert_to_datetime_or_timedelta(df, col):
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors='coerce')
    elif pd.api.types.is_timedelta64_dtype(df[col]):
        df[col] = pd.to_timedelta(df[col], errors='coerce')
    return df

def resample_data(df, x_col, y_col, color_col, time_granularity, agg):
    if color_col != "None":
        df_grouped = df.set_index(x_col).groupby(color_col)
    else:
        df_grouped = df.set_index(x_col)

    agg_mapping = {
        "None": None,
        "Mean": "mean",
        "Median": "median",
        "Sum": "sum",
        "Count": "count"
    }

    if agg_mapping[agg] is None:
        agg_result = df
    else:
        if pd.api.types.is_timedelta64_dtype(df[x_col]):
            if time_granularity == "Hour":
                freq = 'H'
            elif time_granularity == "Day":
                freq = 'D'
            elif time_granularity == "Month":
                freq = '30D'
            elif time_granularity == "Year":
                freq = '365D'
        else:
            if time_granularity == "Hour":
                freq = 'h'
            elif time_granularity == "Day":
                freq = 'D'
            elif time_granularity == "Month":
                freq = 'ME'
            elif time_granularity == "Year":
                freq = 'YE'

        def custom_agg(series):
            if agg == "Count":
                return series.count()
            elif len(series) < 2:
                return series.iloc[0] if not series.empty else np.nan
            else:
                return series.agg(agg_mapping[agg])

        if color_col != "None":
            agg_result = df_grouped.resample(freq).agg({y_col: custom_agg}).reset_index()
        else:
            agg_result = df_grouped.resample(freq).agg({y_col: custom_agg}).reset_index()

    return agg_result

def create_qq_plot(data, title="Q-Q Plot"):
    """
    Generates a Q-Q plot using Plotly to check for normality.
    """
    # Standardize data
    z_scores = stats.zscore(data)
    # Get theoretical quantiles
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Data'))
    
    # Add regression line (Normal theoretical line)
    fig.add_trace(go.Scatter(x=osm, y=slope*osm + intercept, mode='lines', name='Normal Fit', line=dict(color='red')))
    
    fig.update_layout(
        title=title, 
        xaxis_title="Theoretical Quantiles", 
        yaxis_title="Sample Quantiles",
        showlegend=True
    )
    return fig

def create_bland_altman_plot(df, method1, method2, title="Bland-Altman Plot"):
    """
    Generates a Bland-Altman plot.
    """
    data1 = df[method1]
    data2 = df[method2]
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.mean(diff)
    sd = np.std(diff, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean, y=diff, mode='markers', name='Differences'))
    
    # Mean Difference Line
    fig.add_shape(type="line",
        x0=min(mean), y0=md, x1=max(mean), y1=md,
        line=dict(color="red", width=2, dash="dash"),
        name="Mean Diff"
    )
    fig.add_trace(go.Scatter(x=[min(mean), max(mean)], y=[md, md], mode='lines', name='Mean Diff', line=dict(color='red', dash='dash'), showlegend=False))

    # Limits of Agreement
    upper_loa = md + 1.96 * sd
    lower_loa = md - 1.96 * sd
    
    fig.add_shape(type="line",
        x0=min(mean), y0=upper_loa, x1=max(mean), y1=upper_loa,
        line=dict(color="green", width=2, dash="dot"),
        name="+1.96 SD"
    )
    fig.add_trace(go.Scatter(x=[min(mean), max(mean)], y=[upper_loa, upper_loa], mode='lines', name='+1.96 SD', line=dict(color='green', dash='dot'), showlegend=False))

    fig.add_shape(type="line",
        x0=min(mean), y0=lower_loa, x1=max(mean), y1=lower_loa,
        line=dict(color="green", width=2, dash="dot"),
        name="-1.96 SD"
    )
    fig.add_trace(go.Scatter(x=[min(mean), max(mean)], y=[lower_loa, lower_loa], mode='lines', name='-1.96 SD', line=dict(color='green', dash='dot'), showlegend=False))

    fig.update_layout(
        title=title,
        xaxis_title=f"Mean of {method1} and {method2}",
        yaxis_title=f"Difference ({method1} - {method2})",
        template="plotly_white"
    )
    return fig

def create_roc_curve(df, true_col, score_col, pos_label=None):
    """
    Generates a ROC Curve.
    """
    from sklearn.metrics import auc, roc_curve
    
    y_true = df[true_col]
    y_score = df[score_col]
    
    # Handle NaN
    mask = y_true.notna() & y_score.notna()
    y_true = y_true[mask]
    y_score = y_score[mask]

    if pos_label is None:
        # Infer positive label (assuming 1 or True or the second unique value sorted)
        uniques = sorted(y_true.unique())
        if len(uniques) != 2:
            return None, "Target variable must be binary."
        pos_label = uniques[1]

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:.2f})', line=dict(color='darkorange', width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance', line=dict(color='navy', width=2, dash='dash')))
    
    fig.update_layout(
        title=f"Receiver Operating Characteristic (ROC) - Pos Label: {pos_label}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        xaxis=dict(range=[0.0, 1.0]),
        yaxis=dict(range=[0.0, 1.05])
    )
    return fig, None

def create_correlation_matrix_streamlit(
    data,
    cols,
    method="pearson",
    cmap='Viridis',
    triangle="lower",  # 'lower', 'upper', or 'full'
    cluster_rows=False,
    cluster_cols=False,
    show_row_dendrogram=False,
    show_col_dendrogram=False,
    ):
    """
    Generate an interactive correlation matrix using Plotly, with optional triangular masking and clustering.
    """
    # Filter the data
    data_filtered = data[cols]

    # Calculate the correlation matrix
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        corr_matrix = data_filtered.corr(method=method)

    # Initialize variables for clustering
    row_order = corr_matrix.index.tolist()
    col_order = corr_matrix.columns.tolist()
    row_dendro_traces = []
    col_dendro_traces = []
    row_dendro_y_vals = None
    col_dendro_x_vals = None
    row_dendro_range = None
    col_dendro_range = None
    
    cm_filled = corr_matrix.fillna(0)

    # 1. Row Clustering
    if cluster_rows and len(row_order) > 1:
        try:
            # orientation='left' -> leaves on left (facing heatmap)
            dendro_side = ff.create_dendrogram(cm_filled, orientation='left', labels=cm_filled.index)
            row_order = dendro_side['layout']['yaxis']['ticktext']
            row_dendro_y_vals = dendro_side['layout']['yaxis']['tickvals']
            
            if show_row_dendrogram:
                row_dendro_traces = dendro_side['data']
                # Calculate max range for tight layout
                max_d = 0
                for trace in row_dendro_traces:
                    if 'x' in trace and len(trace['x']) > 0:
                        max_d = max(max_d, max(trace['x']))
                row_dendro_range = [0, max_d]
        except Exception as e:
            print(f"Row clustering failed: {e}")

    # 2. Column Clustering
    if cluster_cols and len(col_order) > 1:
        try:
            # orientation='bottom' -> leaves on bottom (facing heatmap)
            dendro_top = ff.create_dendrogram(cm_filled.T, orientation='bottom', labels=cm_filled.columns)
            col_order = dendro_top['layout']['xaxis']['ticktext']
            col_dendro_x_vals = dendro_top['layout']['xaxis']['tickvals']
            
            if show_col_dendrogram:
                col_dendro_traces = dendro_top['data']
                # Calculate max range for tight layout
                max_d = 0
                for trace in col_dendro_traces:
                    if 'y' in trace and len(trace['y']) > 0:
                        max_d = max(max_d, max(trace['y']))
                col_dendro_range = [0, max_d]
        except Exception as e:
            print(f"Column clustering failed: {e}")

    # Reorder Heatmap Data
    X_ordered = corr_matrix.loc[row_order, col_order]

    # Handle symmetric matrix masking if NOT clustering
    if (not cluster_rows and not cluster_cols):
        mask = np.zeros_like(X_ordered, dtype=bool)
        if triangle == "lower":
            mask[np.triu_indices_from(mask)] = True
        elif triangle == "upper":
            mask[np.tril_indices_from(mask)] = True
        X_ordered = X_ordered.mask(mask)

    # Dynamically adjust figure size
    fig_width = max(6, 0.35 * len(col_order))
    fig_height = max(5, 0.35 * len(row_order))

    # Create Figure
    fig = go.Figure()

    # Determine Heatmap Axes Values
    x_vals = col_dendro_x_vals if show_col_dendrogram else col_order
    y_vals = row_dendro_y_vals if show_row_dendrogram else row_order
    
    # Add Heatmap
    heatmap = go.Heatmap(
        x = x_vals,
        y = y_vals,
        z = X_ordered.values,
        colorscale = cmap,
        colorbar=dict(len=0.4, thickness=10, x=1.05, y=0.4)
    )
    fig.add_trace(heatmap)

    # Add Row Dendrogram Traces
    for trace in row_dendro_traces:
        trace.xaxis = 'x2'
        trace.yaxis = 'y' # Share y-axis with heatmap
        trace.showlegend = False
        fig.add_trace(trace)

    # Add Column Dendrogram Traces
    for trace in col_dendro_traces:
        trace.xaxis = 'x' # Share x-axis with heatmap
        trace.yaxis = 'y2'
        trace.showlegend = False
        fig.add_trace(trace)

    # Define Layout Domains
    hm_x_domain = [0, 1]
    hm_y_domain = [0, 1]
    
    if show_row_dendrogram:
        hm_x_domain = [0, 0.85]
    
    if show_col_dendrogram:
        hm_y_domain = [0, 0.85]

    # Calculate explicit ranges for Heatmap to ensure it touches dendrograms
    hm_x_range = None
    hm_y_range = None
    
    if show_col_dendrogram and col_dendro_x_vals is not None and len(col_dendro_x_vals) > 0:
        vals = sorted(col_dendro_x_vals)
        step = (vals[1] - vals[0]) if len(vals) > 1 else 10.0
        hm_x_range = [min(vals) - step/2, max(vals) + step/2]
        
    if show_row_dendrogram and row_dendro_y_vals is not None and len(row_dendro_y_vals) > 0:
        vals = sorted(row_dendro_y_vals)
        step = (vals[1] - vals[0]) if len(vals) > 1 else 10.0
        hm_y_range = [min(vals) - step/2, max(vals) + step/2]

    # Layout Configuration
    layout_args = dict(
        width=fig_width * 100,
        height=fig_height * 100,
        xaxis=dict(
            domain=hm_x_domain,
            tickmode='array',
            tickvals=x_vals,
            ticktext=col_order,
            tickangle=45,
            automargin=True,
            title='',
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            domain=hm_y_domain,
            tickmode='array',
            tickvals=y_vals,
            ticktext=row_order,
            automargin=True,
            title='',
            showgrid=False,
            zeroline=False
        ),
        # Row Dendrogram Axis (Right)
        xaxis2=dict(
            domain=[0.85, 1.0],
            range=row_dendro_range,
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            title=''
        ),
        # Column Dendrogram Axis (Top)
        yaxis2=dict(
            domain=[0.85, 1.0],
            range=col_dendro_range,
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            title=''
        )
    )

    if hm_x_range: layout_args['xaxis']['range'] = hm_x_range
    if hm_y_range: layout_args['yaxis']['range'] = hm_y_range

    fig.update_layout(**layout_args)

    return fig



def plot_nullity_matrix(df, row_id_col="Index", cluster_cols=True, cluster_rows=False, show_dendrograms=False, show_row_labels=True, show_col_labels=True, figsize=None):
    """
    Generates a Nullity Matrix (Clustermap style).
    - Adapts figure size and font size to data density for readability.
    - Zero gaps between components.
    - Missing Count bar in %.
    """
    # 1. Create Boolean Matrix (1=Missing, 0=Present)
    nullity_df = df.isnull().astype(int)
    
    # 2. Handle Row Labels
    if row_id_col != "Index" and row_id_col in df.columns:
        display_labels = df[row_id_col].astype(str).reset_index(drop=True)
    else:
        display_labels = df.index.astype(str).to_series().reset_index(drop=True)
    
    # 3. Reset Index for Internal Processing
    nullity_df = nullity_df.reset_index(drop=True)
    
    row_order = nullity_df.index.tolist()
    col_order = nullity_df.columns.tolist()
    
    row_dendro_traces = []
    col_dendro_traces = []
    col_dendro_x_vals = None
    y_sequential = list(range(len(nullity_df))) 
    
    # --- CLUSTERING ---
    # 1. Column Clustering (Variables)
    if cluster_cols and len(col_order) > 1:
        try:
            dendro_top = ff.create_dendrogram(nullity_df.T, orientation='bottom', labels=col_order)
            col_order = dendro_top['layout']['xaxis']['ticktext']
            col_dendro_x_vals = dendro_top['layout']['xaxis']['tickvals']
            if show_dendrograms:
                col_dendro_traces = dendro_top['data']
        except Exception as e:
            st.warning(f"Column clustering failed: {e}")

    # 2. Row Clustering (Samples)
    if cluster_rows and len(row_order) > 1:
        try:
            dendro_side = ff.create_dendrogram(nullity_df, orientation='left', labels=list(map(str, row_order)))
            reordered_indices_str = dendro_side['layout']['yaxis']['ticktext']
            row_order = [int(i) for i in reordered_indices_str]
            y_sequential = dendro_side['layout']['yaxis']['tickvals']
            if show_dendrograms:
                row_dendro_traces = dendro_side['data']
        except Exception as e:
            st.warning(f"Row clustering failed: {e}")

    # Reorder DataFrame
    nullity_df = nullity_df.iloc[row_order]
    nullity_df = nullity_df[col_order]
    final_row_labels = display_labels.iloc[row_order]
    
    # --- LAYOUT DOMAIN CALCULATION (Right-to-Left, Zero Gaps) ---
    
    # Settings
    w_row_dendro = 0.11 if (show_dendrograms and cluster_rows) else 0.0
    w_missing = 0.025
    w_gap = 0.0
    
    h_col_dendro = 0.12 if (show_dendrograms and cluster_cols) else 0.0
    
    # 1. Horizontal Domains (X-Axis)
    current_x = 1.0
    
    if w_row_dendro > 0:
        dendro_domain_x = [current_x - w_row_dendro, current_x]
        current_x -= w_row_dendro
    else:
        dendro_domain_x = [1.0, 1.0] 
    
    missing_col_domain_x = [current_x - w_missing, current_x]
    current_x -= w_missing
    
    heatmap_end_x = max(0, current_x)
    heatmap_domain_x = [0, heatmap_end_x]
    
    # 2. Vertical Domains (Y-Axis)
    current_y = 1.0
    
    if h_col_dendro > 0:
        col_dendro_domain_y = [current_y - h_col_dendro, current_y]
        current_y -= h_col_dendro
    else:
        col_dendro_domain_y = [1.0, 1.0]
        
    heatmap_domain_y = [0, max(0, current_y)]

    # --- SIZING & FONTS LOGIC ---
    n_rows = len(row_order)
    n_cols = len(col_order)

    # 1. Calculate Pixels
    if figsize is None:
        # Dynamic Auto-Size
        # Base overhead (margins, legends)
        overhead_h = 200
        overhead_w = 300
        
        # Pixels per cell
        px_per_row_ideal = 15  # Minimum height for readable row label
        px_per_col_ideal = 25  # Minimum width for readable col label
        
        ideal_height = (n_rows * px_per_row_ideal) + overhead_h
        ideal_width = (n_cols * px_per_col_ideal) + overhead_w
        
        # Adjust for Dendrograms (they need space too)
        if show_dendrograms:
            ideal_height += 150
            ideal_width += 150
            
        height_px = max(600, ideal_height)
        width_px = max(900, ideal_width)
    else:
        # User defined fixed size
        width_px = figsize[0] * 50
        height_px = figsize[1] * 50

    # 2. Calculate Adaptive Font Sizes
    # How many pixels are actually allocated to the heatmap area?
    hm_pixel_height = height_px * (heatmap_domain_y[1] - heatmap_domain_y[0])
    hm_pixel_width = width_px * (heatmap_domain_x[1] - heatmap_domain_x[0])
    
    # Pixels available per row/col
    px_per_row_actual = hm_pixel_height / max(1, n_rows)
    px_per_col_actual = hm_pixel_width / max(1, n_cols)
    
    # Clamp font sizes (Min 6px, Max 12px)
    # We subtract a small buffer (e.g. 2px) to prevent touching
    font_size_row = max(6, min(12, int(px_per_row_actual - 2)))
    font_size_col = max(6, min(12, int(px_per_col_actual)))

    # --- FIGURE CONSTRUCTION ---
    fig_matrix = go.Figure()
    
    # Ranges
    hm_x_range = None
    if cluster_cols and col_dendro_x_vals is not None:
        vals = sorted(col_dendro_x_vals)
        step = (vals[1] - vals[0]) if len(vals) > 1 else 10.0
        hm_x_range = [min(vals) - step/2, max(vals) + step/2]

    hm_y_range = None
    if cluster_rows and len(y_sequential) > 0:
        vals = sorted(y_sequential)
        step = (vals[1] - vals[0]) if len(vals) > 1 else 1.0
        hm_y_range = [min(vals) - step/2, max(vals) + step/2]
    
    row_dendro_range = None
    if show_dendrograms and cluster_rows and len(row_dendro_traces) > 0:
        max_d = max([max(t['x']) for t in row_dendro_traces if 'x' in t and len(t['x']) > 0], default=0)
        row_dendro_range = [0, max_d]
    
    col_dendro_range = None
    if show_dendrograms and cluster_cols and len(col_dendro_traces) > 0:
        max_d = max([max(t['y']) for t in col_dendro_traces if 'y' in t and len(t['y']) > 0], default=0)
        col_dendro_range = [0, max_d]

    # 1. Main Heatmap
    x_vals = col_dendro_x_vals if (cluster_cols and col_dendro_x_vals is not None) else col_order
    
    fig_matrix.add_trace(go.Heatmap(
        z=nullity_df.values,
        x=x_vals,
        y=y_sequential, 
        colorscale=[[0, '#eeeeee'], [1, '#444444']], 
        showscale=False,
        xaxis='x',
        yaxis='y',
        hovertemplate='Variable: %{x}<br>Row: %{text}<br>Missing: %{z}<extra></extra>',
        text=[[str(l) for _ in col_order] for l in final_row_labels]
    ))
    
    # 2. Missingness Percentage Column
    total_cols = nullity_df.shape[1]
    row_missing_pct = (nullity_df.sum(axis=1) / total_cols * 100).values.reshape(-1, 1)
    
    fig_matrix.add_trace(go.Heatmap(
        z=row_missing_pct,
        x=[0],
        y=y_sequential,
        colorscale='RdYlGn_r', 
        showscale=True,
        colorbar=dict(
            title=dict(text="Missing %", side="top"),
            thickness=10, 
            len=0.4,
            yanchor="bottom", y=0.0,
            xanchor="left", x=1.005
        ),
        xaxis='x4',
        yaxis='y', 
        hovertemplate='Row: %{text}<br>Missing: %{z:.1f}%<extra></extra>',
        text=[[str(l)] for l in final_row_labels]
    ))
    
    # 3. Dendrograms
    if show_dendrograms and cluster_cols:
        for trace in col_dendro_traces:
            trace['xaxis'] = 'x'
            trace['yaxis'] = 'y2'
            trace['showlegend'] = False
            trace['hoverinfo'] = 'none'
            fig_matrix.add_trace(trace)
        
    if show_dendrograms and cluster_rows:
        for trace in row_dendro_traces:
            trace['xaxis'] = 'x3'
            trace['yaxis'] = 'y'
            trace['showlegend'] = False
            trace['hoverinfo'] = 'none'
            fig_matrix.add_trace(trace)
    
    # --- LAYOUT CONFIG ---
    layout_args = dict(
        width=width_px,
        height=height_px,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=20, b=80), 
        
        # Main X (Bottom)
        xaxis=dict(
            domain=heatmap_domain_x,
            title="",
            side='bottom',
            tickangle=45,
            tickmode='array' if (cluster_cols and col_dendro_x_vals is not None) else 'auto',
            tickvals=col_dendro_x_vals if (cluster_cols and col_dendro_x_vals is not None) else None,
            ticktext=col_order if (cluster_cols and col_dendro_x_vals is not None) else None,
            showticklabels=show_col_labels,
            tickfont=dict(size=font_size_col), # DYNAMIC FONT
            showgrid=False, zeroline=False,
            range=hm_x_range,
            automargin=True
        ),
        
        # Main Y (Left)
        yaxis=dict(
            domain=heatmap_domain_y,
            title=row_id_col,
            side='left',
            tickmode='array',
            tickvals=y_sequential,
            ticktext=final_row_labels,
            showticklabels=show_row_labels,
            tickfont=dict(size=font_size_row), # DYNAMIC FONT
            showgrid=False, zeroline=False,
            range=hm_y_range,
        ),
        
        # Top Dendrogram
        yaxis2=dict(
            domain=col_dendro_domain_y,
            showgrid=False, showticklabels=False, zeroline=False,
            range=col_dendro_range,
        ),
        
        # Right Dendrogram
        xaxis3=dict(
            domain=dendro_domain_x,
            showgrid=False, showticklabels=False, zeroline=False,
            range=row_dendro_range,
        ),
        
        # Missing Bar (Middle)
        xaxis4=dict(
            domain=missing_col_domain_x,
            showgrid=False, showticklabels=False, zeroline=False, title="",
            range=[-0.5, 0.5], 
            side='top'
        )
    )
    
    fig_matrix.update_layout(**layout_args)
    return fig_matrix
