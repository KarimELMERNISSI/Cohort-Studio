import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from openpyxl.styles import PatternFill, Font
from matplotlib.colors import to_hex
import plotly.express as px
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import networkx as nx
from yfiles_graphs_for_streamlit import Node, Edge, EdgeStyle, DashStyle, NodeStyle, NodeShape

####################################### CORRELATION MATRIX ###########################################

def plotly_corr_mat(
    data,
    targets,
    predictors,
    method="pearson",
    selected_color=px.colors.sequential.Viridis,
    corr_method_name=None,
    triangle="lower",  # 'lower', 'upper', or 'full'
    cluster_rows=False,
    cluster_cols=False,
    show_row_dendrogram=False,
    show_col_dendrogram=False,
    correlation_matrix=None
    ):
    """
    Generate an interactive correlation matrix using Plotly, with optional triangular masking.

    Parameters:
    - data (pd.DataFrame): The input data containing the variables.
    - targets (list): List of target variable names.
    - predictors (list): List of predictor variable names.
    - method (str): Correlation method ('pearson', 'spearman', 'kendall').
    - selected_color (str): Colormap for the heatmap.
    - corr_method_name (str or None): Display name for the correlation method (optional).
    - triangle (str): 'lower', 'upper', or 'full' to control the displayed triangle.
    - cluster_rows (bool): Whether to apply hierarchical clustering to reorder rows.
    - cluster_cols (bool): Whether to apply hierarchical clustering to reorder columns.
    - show_row_dendrogram (bool): Whether to show the row dendrogram tree.
    - show_col_dendrogram (bool): Whether to show the column dendrogram tree.
    - correlation_matrix (pd.DataFrame): Pre-calculated correlation matrix (optional).

    Returns:
    - plotly.graph_objs._figure.Figure: Plotly figure object.
    """
    # Filter the data to include only the target and predictor columns
    # Ensure unique columns to avoid duplicates in correlation matrix calculation
    unique_cols = list(set(targets + predictors))
    
    if correlation_matrix is not None:
        # Use provided matrix
        corr_matrix = correlation_matrix.loc[predictors, targets]
    else:
        # Calculate the correlation matrix
        data_filtered = data[unique_cols]
        corr_matrix = data_filtered.corr(method=method).loc[predictors, targets]

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
                # Calculate max range for tight layout (height is on x-axis for orientation='left')
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
            # Transpose for column clustering
            dendro_top = ff.create_dendrogram(cm_filled.T, orientation='bottom', labels=cm_filled.columns)
            col_order = dendro_top['layout']['xaxis']['ticktext']
            col_dendro_x_vals = dendro_top['layout']['xaxis']['tickvals']
            
            if show_col_dendrogram:
                col_dendro_traces = dendro_top['data']
                # Calculate max range for tight layout (height is on y-axis for orientation='bottom')
                max_d = 0
                for trace in col_dendro_traces:
                    if 'y' in trace and len(trace['y']) > 0:
                        max_d = max(max_d, max(trace['y']))
                col_dendro_range = [0, max_d]
        except Exception as e:
            print(f"Column clustering failed: {e}")

    # Reorder Heatmap Data
    X_ordered = corr_matrix.loc[row_order, col_order]

    # Handle symmetric matrix masking if NOT clustering (or if clustering preserves symmetry which is not guaranteed unless we force it)
    # If clustering is enabled, masking usually doesn't make sense unless the order is identical.
    # If cluster_rows and cluster_cols are both True and symmetric, we might want to mask.
    # But for now, let's disable masking if any clustering is active to avoid confusion, or check if orders match.
    
    if (not cluster_rows and not cluster_cols) and set(targets) == set(predictors):
        mask = np.zeros_like(X_ordered, dtype=bool)
        if triangle == "lower":
            mask[np.triu_indices_from(mask)] = True
        elif triangle == "upper":
            mask[np.tril_indices_from(mask)] = True
        X_ordered = X_ordered.mask(mask)

    # Generate correlation method title if not provided
    if corr_method_name is None:
        corr_method_name = method.capitalize()

    # Dynamically adjust figure size based on number of predictors and targets
    fig_width = max(6, 0.35 * len(col_order))
    fig_height = max(5, 0.35 * len(row_order))

    # Create Figure
    fig = go.Figure()

    # Determine Heatmap Axes Values
    # If showing dendrogram, use the numerical values from dendrogram layout.
    # If not, use the categorical labels.
    
    x_vals = col_dendro_x_vals if show_col_dendrogram else col_order
    y_vals = row_dendro_y_vals if show_row_dendrogram else row_order
    
    # Add Heatmap
    heatmap = go.Heatmap(
        x = x_vals,
        y = y_vals,
        z = X_ordered.values,
        colorscale = selected_color,
        colorbar=dict(len=0.4, thickness=10, x=1.05, y=0.4)
    )
    fig.add_trace(heatmap)

    # Add Row Dendrogram Traces
    for trace in row_dendro_traces:
        trace.xaxis = 'x2'
        trace.yaxis = 'y' # Share y-axis with heatmap
        fig.add_trace(trace)

    # Add Column Dendrogram Traces
    for trace in col_dendro_traces:
        trace.xaxis = 'x' # Share x-axis with heatmap
        trace.yaxis = 'y2'
        fig.add_trace(trace)

    # Define Layout Domains
    # Default (No dendrograms)
    hm_x_domain = [0, 1]
    hm_y_domain = [0, 1]
    
    # Use tighter domains to reduce whitespace
    # Heatmap takes 85%, Dendrogram takes 15% (touching)
    if show_row_dendrogram:
        hm_x_domain = [0, 0.85]
    
    if show_col_dendrogram:
        hm_y_domain = [0, 0.85]

    # Calculate explicit ranges for Heatmap to ensure it touches dendrograms
    hm_x_range = None
    hm_y_range = None
    
    if show_col_dendrogram and col_dendro_x_vals is not None and len(col_dendro_x_vals) > 0:
        # Dendrogram ticks are usually centers. We need edges.
        # Assuming uniform spacing
        vals = sorted(col_dendro_x_vals)
        if len(vals) > 1:
            step = vals[1] - vals[0]
        else:
            step = 10.0 # Default
        hm_x_range = [min(vals) - step/2, max(vals) + step/2]
        
    if show_row_dendrogram and row_dendro_y_vals is not None and len(row_dendro_y_vals) > 0:
        vals = sorted(row_dendro_y_vals)
        if len(vals) > 1:
            step = vals[1] - vals[0]
        else:
            step = 10.0
        hm_y_range = [min(vals) - step/2, max(vals) + step/2]

    # Prepare Axis Configs
    xaxis_config = dict(
        domain=hm_x_domain, 
        tickmode='array', 
        tickvals=x_vals, 
        ticktext=col_order, 
        tickangle=45,
        title='',
        automargin=True
    )
    if hm_x_range:
        xaxis_config['range'] = hm_x_range

    yaxis_config = dict(
        domain=hm_y_domain, 
        tickmode='array', 
        tickvals=y_vals, 
        ticktext=row_order, 
        side='left',
        title='',
        automargin=True
    )
    if hm_y_range:
        yaxis_config['range'] = hm_y_range

    # Update Layout
    fig.update_layout(
        # Heatmap Axes
        xaxis=xaxis_config,
        yaxis=yaxis_config,
        
        # Row Dendrogram Axis (Right)
        xaxis2=dict(
            domain=[0.85, 1], 
            range=row_dendro_range,
            showgrid=False, 
            showline=False, 
            showticklabels=False,
            title=''
        ),
        
        # Column Dendrogram Axis (Top)
        yaxis2=dict(
            domain=[0.85, 1], 
            range=col_dendro_range,
            showgrid=False, 
            showline=False, 
            showticklabels=False,
            title=''
        ),
        
        width=fig_width * 100 + (150 if show_row_dendrogram else 0),
        height=fig_height * 100 + (150 if show_col_dendrogram else 0),
        title=f"Correlation Matrix ({corr_method_name})",
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=50),
        autosize=False
    )

    return fig


def plotly_corr_network(
    data,
    targets,
    predictors,
    method="pearson",
    threshold_category="Weak", # Minimum strength to display
    corr_method_name=None,
    node_color="skyblue",
):
    """
    Generate an interactive correlation network graph using Plotly and NetworkX.
    Nodes represent variables, and edges represent correlations.
    Edge styles/attributes depend on correlation strength.

    Parameters:
    - data (pd.DataFrame): Input data.
    - targets (list): Target variables.
    - predictors (list): Predictor variables.
    - method (str): Correlation method.
    - threshold_category (str): Minimum strength category to display edge ('Negligible', 'Weak', 'Moderate', 'Strong', 'Very Strong').
    - corr_method_name (str): Display name for method.
    - node_color (str): Color of nodes.

    Returns:
    - plotly.graph_objects.Figure: The network graph.
    """
    # 1. Calculate Correlation Matrix
    unique_cols = list(set(targets + predictors))
    # Filter numeric columns only to avoid errors
    data_filtered = data[unique_cols].select_dtypes(include=[np.number])
    
    if data_filtered.empty:
        return go.Figure()

    corr_matrix = data_filtered.corr(method=method)

    # 2. Define Thresholds (Same as Excel export)
    strength_data = {
        'pearson': [0.00, 0.10, 0.40, 0.70, 0.90],
        'spearman': [0.00, 0.10, 0.38, 0.68, 0.89],
        'kendall': [0.00, 0.06, 0.26, 0.49, 0.71]
    }
    thresholds = strength_data.get(method, strength_data['pearson'])
    strength_labels = ['Negligible', 'Weak', 'Moderate', 'Strong', 'Very Strong']
    strength_map = dict(zip(strength_labels, thresholds)) # e.g., 'Weak': 0.10

    min_strength_val = strength_map.get(threshold_category, 0.10)

    # 3. Create NetworkX Graph
    G = nx.Graph()
    
    # Add nodes
    for col in corr_matrix.columns:
        G.add_node(col)

    # Add edges
    # Iterate over the upper triangle to avoid duplicates and self-loops
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1 = cols[i]
            col2 = cols[j]
            corr_val = corr_matrix.loc[col1, col2]
            abs_corr = abs(corr_val)

            if abs_corr >= min_strength_val:
                # Determine category
                cat = 'Negligible'
                for idx, val in enumerate(thresholds):
                    if abs_corr >= val:
                        cat = strength_labels[idx]
                
                # Add edge with attributes
                G.add_edge(col1, col2, weight=abs_corr, correlation=corr_val, category=cat)

    # 4. Generate Layout
    # pos = nx.spring_layout(G, seed=42) # Seed for reproducible layout
    # Use circular layout if small number of nodes, else spring
    if len(G.nodes) < 10:
         pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # 5. Create Plotly Traces
    
    # Edge Traces
    # We create separate traces for different categories to style them differently (legend support)
    edge_traces = []
    
    # Define styles for categories
    # Width and Color
    style_map = {
        'Negligible': {'width': 1, 'color': '#d3d3d3', 'dash': 'dot'}, # Light Gray
        'Weak': {'width': 2, 'color': '#a9a9a9', 'dash': 'dash'},     # Dark Gray
        'Moderate': {'width': 3, 'color': '#add8e6', 'dash': 'solid'}, # Light Blue
        'Strong': {'width': 4, 'color': '#0000ff', 'dash': 'solid'},  # Blue
        'Very Strong': {'width': 5, 'color': '#00008b', 'dash': 'solid'} # Dark Blue
    }

    # Group edges by category
    edges_by_cat = {cat: [] for cat in strength_labels}
    for u, v, d in G.edges(data=True):
        edges_by_cat[d['category']].append((u, v, d))

    for cat in strength_labels:
        edges = edges_by_cat[cat]
        if not edges:
            continue
        
        style = style_map[cat]
        edge_x = []
        edge_y = []
        hover_texts = []
        
        for u, v, d in edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            # Hover text needs to be per point, but scatter lines share hover. 
            # Often simpler to put hover on a middle node or just disable edge hover and rely on visual.
            # But we can try to add a middle point for hover if needed. For now, simple lines.
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=style['width'], color=style['color'], dash=style['dash']),
            hoverinfo='none',
            mode='lines',
            name=cat # Add to legend
        )
        edge_traces.append(edge_trace)

    # Node Trace
    node_x = []
    node_y = []
    node_text = []
    node_adjacencies = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_adjacencies.append(len(G.adj[node])) # Degree

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=node_adjacencies,
            size=20,
            colorbar=dict(
                thickness=15,
                title=dict(
                    text='Node Connections',
                    side='right'
                ),
                xanchor='left',
            ),
            line_width=2
        )
    )

    # 6. Create Figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    if corr_method_name is None:
        corr_method_name = method.capitalize()

    fig.update_layout(
        title=f"Correlation Network ({corr_method_name}) - Threshold: {threshold_category}",
        title_font_size=16,
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(title="Correlation Strength")
    )

    return fig


def get_yfiles_network_data(data, targets, predictors, method="pearson", threshold_category="Weak", correlation_matrix=None, focus_node=None):
    """
    Generate nodes and edges for yFiles correlation network.

    Parameters:
    - data (pd.DataFrame): Input data.
    - targets (list): Target variables.
    - predictors (list): Predictor variables.
    - method (str): Correlation method.
    - threshold_category (str): Minimum strength category.
    - correlation_matrix (pd.DataFrame): Pre-calculated correlation matrix (optional).
    - focus_node (str): Variable to focus on (optional). Filters graph to connected component.

    Returns:
    - nodes (list): List of yFiles Node objects.
    - edges (list): List of yFiles Edge objects.
    """
    if correlation_matrix is not None:
        corr_matrix = correlation_matrix
    else:
        unique_cols = list(set(targets + predictors))
        data_filtered = data[unique_cols].select_dtypes(include=[np.number])
        
        if data_filtered.empty:
            return [], []

        corr_matrix = data_filtered.corr(method=method)

    # Thresholds
    strength_data = {
        'pearson': [0.00, 0.10, 0.40, 0.70, 0.90],
        'spearman': [0.00, 0.10, 0.38, 0.68, 0.89],
        'kendall': [0.00, 0.06, 0.26, 0.49, 0.71]
    }
    thresholds = strength_data.get(method, strength_data['pearson'])
    strength_labels = ['Negligible', 'Weak', 'Moderate', 'Strong', 'Very Strong']
    strength_map = dict(zip(strength_labels, thresholds))
    min_strength_val = strength_map.get(threshold_category, 0.10)

    nodes = []
    edges = []
    node_heat = {col: 0.0 for col in corr_matrix.columns}

    # Create Edges first to calculate heat based on visible connections
    # Store adjacency for BFS if needed
    adj = {col: [] for col in corr_matrix.columns}
    
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1 = cols[i]
            col2 = cols[j]
            corr_val = corr_matrix.loc[col1, col2]
            abs_corr = abs(corr_val)

            if abs_corr >= min_strength_val:
                # Accumulate heat (weighted degree)
                node_heat[col1] += abs_corr
                node_heat[col2] += abs_corr
                
                # Build Adjacency
                adj[col1].append(col2)
                adj[col2].append(col1)

                # Determine category
                cat = 'Negligible'
                for idx, val in enumerate(thresholds):
                    if abs_corr >= val:
                        cat = strength_labels[idx]
                
                # Assign properties based on category
                # Base styles (dash/thickness)
                base_styles = {
                    'Negligible': {'dash': 'dotted', 'thickness': 1.0},
                    'Weak': {'dash': 'dashed', 'thickness': 1.5},
                    'Moderate': {'dash': 'solid', 'thickness': 2.5},
                    'Strong': {'dash': 'solid', 'thickness': 3.5},
                    'Very Strong': {'dash': 'solid', 'thickness': 5.0}
                }
                
                # Color palettes
                if corr_val >= 0:
                    # Positive: Blue variations
                    colors = {
                        'Negligible': '#d3d3d3',
                        'Weak': '#add8e6', # LightBlue
                        'Moderate': '#4169e1', # RoyalBlue
                        'Strong': '#0000ff', # Blue
                        'Very Strong': '#00008b' # DarkBlue
                    }
                else:
                    # Negative: Red variations
                    colors = {
                        'Negligible': '#d3d3d3',
                        'Weak': '#f08080', # LightCoral
                        'Moderate': '#cd5c5c', # IndianRed
                        'Strong': '#ff0000', # Red
                        'Very Strong': '#8b0000' # DarkRed
                    }
                
                style = base_styles.get(cat, base_styles['Weak'])
                color = colors.get(cat, '#a9a9a9')
                
                props = {**style, 'color': color}
                
                edges.append(Edge(
                    start=col1,
                    end=col2,
                    properties={
                        "label": f"{corr_val:.2f}",
                        "color": props['color'],
                        "dash": props['dash'],
                        "thickness": props['thickness'],
                        "directed": False
                    }
                ))

    # Determine visible nodes based on focus_node
    if focus_node and focus_node in corr_matrix.columns:
        # Perform BFS to find connected component
        visited = set()
        queue = [focus_node]
        visited.add(focus_node)
        
        while queue:
            curr = queue.pop(0)
            for neighbor in adj.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        visible_nodes = visited
    else:
        visible_nodes = set(corr_matrix.columns)

    # Filter Edges
    final_edges = [e for e in edges if e.start in visible_nodes and e.end in visible_nodes]
    
    # Recalculate heat based on visible edges? 
    # Decision: Keep heat based on visible edges in the *filtered* view to be consistent?
    # Actually, let's recalculate heat for accurate "local" visualization
    node_heat_filtered = {col: 0.0 for col in visible_nodes}
    for e in final_edges:
        # Extract weight from label (a bit hacky but we have it) to avoid re-lookup
        # Or re-lookup
        val = float(e.properties['label'])
        val = abs(val)
        node_heat_filtered[e.start] += val
        node_heat_filtered[e.end] += val
        
    # Normalize heat
    max_heat = max(node_heat_filtered.values()) if node_heat_filtered and max(node_heat_filtered.values()) > 0 else 1.0

    # Create Nodes with heat property
    for col in visible_nodes:
        heat_val = node_heat_filtered.get(col, 0.0) / max_heat
        # Highlight focus node?
        color = "orange" if col == focus_node else "skyblue"
        
        nodes.append(Node(
            id=col,
            properties={
                "label": col,
                "color": color,
                "shape": "ellipse",
                "heat": heat_val
            }
        ))
    
    return nodes, final_edges


def add_correlation_strength_table(writer, method, sheet_name='Correlation Strength Info'):
    """
    Add the correlation strength information table to a new sheet in the Excel file,
    with the row corresponding to the selected method bolded.

    Parameters:
    writer (pd.ExcelWriter): The Pandas ExcelWriter object.
    method (str): The correlation method ('pearson', 'spearman', 'kendall').
    sheet_name (str): The name of the sheet to add the information table.

    Returns:
    None
    """
    # Define the correlation strength thresholds for each method
    strength_data = {
        'Strength (absolute value)': ['Negligible', 'Weak', 'Moderate', 'Strong', 'Very Strong'],
        'pearson': [0.00, 0.10, 0.40, 0.70, 0.90],
        'spearman': [0.00, 0.10, 0.38, 0.68, 0.89],
        'kendall': [0.00, 0.06, 0.26, 0.49, 0.71]
    }

    # Create a DataFrame to represent the full table for all methods
    strength_df = pd.DataFrame({
        'Strength (absolute value)': strength_data['Strength (absolute value)'],
        'Pearson': strength_data['pearson'],
        'Spearman': strength_data['spearman'],
        'Kendall': strength_data['kendall']
    })

    # Write the strength table to a new sheet
    strength_df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Get access to the Excel writer's workbook and the new sheet
    workbook = writer.book
    worksheet = workbook[sheet_name]

    # Define a bold font
    bold_font = Font(bold=True)

    # Determine the column for the selected method to be bolded
    if method == 'pearson':
        column_to_bold = 'B'  # Pearson is in column B
    elif method == 'spearman':
        column_to_bold = 'C'  # Spearman is in column C
    elif method == 'kendall':
        column_to_bold = 'D'  # Kendall is in column D

    # Bold the relevant row in the column corresponding to the method
    for row in range(2, len(strength_df) + 2):  # +2 to account for header row
        worksheet[f'{column_to_bold}{row}'].font = bold_font

    # Auto-adjust column widths
    for col in worksheet.columns:
        max_length = 0
        column = col[0].column_letter  # Get the column name
        for cell in col:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))  # Cast to string for length calculation
            except:
                pass
        adjusted_width = (max_length + 2) * 1.2  # Add extra space and adjust
        worksheet.column_dimensions[column].width = adjusted_width

    print(f"Correlation strength information with bolded {method} column added to {sheet_name} sheet.")


def add_correlation_chart_openpyxl(writer, correlations, method, sheet_name='Correlation Summary', threshold='Strong', top_n=None):
    """
    Add an Excel sheet showing correlations sorted from greatest to lowest,
    with correlation strength labels based on the selected method (Pearson, Spearman, Kendall).
    Filters correlations based on the strength threshold (default: 'strong' or more) and limits the display to the top N correlations.

    Parameters:
    writer (pd.ExcelWriter): The Pandas ExcelWriter object.
    correlations (pd.DataFrame): The correlation matrix.
    method (str): The correlation method ('pearson', 'spearman', 'kendall').
    sheet_name (str): The name of the sheet where the chart will be added.
    threshold (str): Minimum strength category to include in the chart ('negligible', 'weak', 'moderate', 'strong', 'very strong').
    top_n (int): Number of top correlations to display (default: 20).

    Returns:
    None
    """
    # Flatten the correlation matrix and extract the pairs with correlation values
    corr_flat = correlations.stack().reset_index()
    corr_flat.columns = ['Predictor', 'Target', 'Correlation']

    # Remove self-correlations (where Predictor == Target)
    corr_flat = corr_flat[corr_flat['Predictor'] != corr_flat['Target']]

    # Sort by absolute correlation value in descending order
    corr_flat['AbsCorrelation'] = corr_flat['Correlation'].abs()
    corr_flat_sorted = corr_flat.sort_values(by='AbsCorrelation', ascending=False)

    # Define strength categories based on the selected method
    strength_data = {
        'pearson': [0.00, 0.10, 0.40, 0.70, 0.90],
        'spearman': [0.00, 0.10, 0.38, 0.68, 0.89],
        'kendall': [0.00, 0.06, 0.26, 0.49, 0.71]
    }

    # Map the selected method to the appropriate thresholds
    selected_thresholds = strength_data[method]

    # Categorize correlation strengths based on absolute values
    def categorize_strength(value):
        if value >= selected_thresholds[4]:
            return 'Very Strong'
        elif value >= selected_thresholds[3]:
            return 'Strong'
        elif value >= selected_thresholds[2]:
            return 'Moderate'
        elif value >= selected_thresholds[1]:
            return 'Weak'
        else:
            return 'Negligible'

    corr_flat_sorted['Strength'] = corr_flat_sorted['AbsCorrelation'].apply(categorize_strength)

    # Define the order of strength categories for filtering
    strength_order = ['negligible', 'weak', 'moderate', 'strong', 'very strong']

    # Filter based on the threshold
    filter_index = strength_order.index(threshold.lower())
    filtered_strengths = strength_order[filter_index:]
    
    # Filter the data to include only the desired strength categories
    if corr_flat_sorted.empty:
        corr_flat_filtered = pd.DataFrame(columns=corr_flat_sorted.columns)
    else:
        # Ensure Strength column is treated as string
        corr_flat_sorted['Strength'] = corr_flat_sorted['Strength'].astype(str)
        corr_flat_filtered = corr_flat_sorted[corr_flat_sorted['Strength'].str.lower().isin(filtered_strengths)]

    # Limit to top N correlations
    if top_n:
        corr_flat_filtered = corr_flat_filtered.head(top_n)

    # Write the filtered correlation pairs and their strength categories to a new sheet
    corr_flat_filtered.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Filtered correlations (threshold: {threshold}, top {top_n}) added to {sheet_name} sheet.")


def generate_palette(style: str = "blue_white_red") -> list:
    """
    Generate a diverging color palette for correlation matrix visualization.
    
    Parameters:
    style (str): Color style to use:
        - "blue_white_red": Clear blue (negative) → white (zero) → red (positive)
        - "coolwarm": Matplotlib's coolwarm colormap (color-blind friendly)
        - "RdBu_r": Red-Blue reversed (classic scientific)
        - "legacy": Original seaborn palette
    
    Returns:
    list of str: List of 256 color hex codes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    if style == "blue_white_red":
        # High-contrast blue-white-red gradient
        # Deep blue → Light blue → White → Light red → Deep red
        colors = [
            "#053061",  # Deep blue (-1.0)
            "#2166ac",  # Blue (-0.75)
            "#4393c3",  # Light blue (-0.5)
            "#92c5de",  # Pale blue (-0.25)
            "#f7f7f7",  # White (0.0)
            "#f4a582",  # Pale red (+0.25)
            "#d6604d",  # Light red (+0.5)
            "#b2182b",  # Red (+0.75)
            "#67001f",  # Deep red (+1.0)
        ]
        cmap = LinearSegmentedColormap.from_list("blue_white_red", colors, N=256)
    elif style == "coolwarm":
        cmap = plt.get_cmap("coolwarm")
    elif style == "RdBu_r":
        cmap = plt.get_cmap("RdBu_r")
    else:  # legacy
        cmap = sns.diverging_palette(240, 10, s=70, l=85, as_cmap=True)
    
    return [to_hex(cmap(i / 255)) for i in range(256)]


def get_color_from_value(value: float, min_val: float, max_val: float, palette: list) -> str:
    """
    Get the fill color based on the correlation value using an existing color palette.
    Uses symmetric scaling centered at zero for correlation matrices.

    Parameters:
    value (float): The correlation value to determine the color.
    min_val (float): Minimum value in the correlation matrix for scaling.
    max_val (float): Maximum value in the correlation matrix for scaling.
    palette (list of str): List of color hex codes.

    Returns:
    str: The color code in hexadecimal format.
    """
    # Handle NaN values by returning neutral color (white)
    if np.isnan(value):
        return palette[len(palette) // 2]  # Return middle color (white/neutral)
    
    # Use symmetric scaling centered at 0 for correlations
    # This ensures 0 is always white/neutral regardless of data range
    abs_max = max(abs(min_val), abs(max_val), 1.0)  # At least 1.0 for standard correlations
    
    # Normalize value to [-1, 1] range, then to [0, 1] for palette indexing
    # value -1 → 0, value 0 → 0.5, value +1 → 1
    normalized_value = (value + abs_max) / (2 * abs_max)
    normalized_value = max(0, min(1, normalized_value))  # Clamp to [0, 1]

    # Find the closest color in the palette
    index = int(normalized_value * (len(palette) - 1))
    return palette[index]



def _write_corr_to_sheet(
    writer: pd.ExcelWriter,
    correlations: pd.DataFrame,
    counts: pd.DataFrame,
    sheet_name: str,
    method: str
) -> None:
    """
    Helper function to write the correlation matrix to a sheet and apply formatting.

    Parameters:
    writer (pd.ExcelWriter): The Pandas ExcelWriter object.
    correlations (pd.DataFrame): The correlation matrix.
    counts (pd.DataFrame): The matrix containing counts of rows used for each correlation.
    sheet_name (str): The name of the sheet in the Excel file.
    method (str): Correlation method ('pearson', 'spearman', etc.).

    Returns:
    None
    """
    # Write the correlation matrix (numeric values) to the Excel sheet
    correlations.to_excel(writer, sheet_name=sheet_name)

    # Access the workbook and sheet
    workbook = writer.book
    worksheet = workbook[sheet_name]

    # Find min and max values for normalization
    min_val = -1
    max_val = 1

    # Generate the color palette
    palette = generate_palette()

    # Apply formatting
    num_rows, num_cols = correlations.shape

    for row in range(2, num_rows + 2):
        for col in range(2, num_cols + 2):
            cell = worksheet.cell(row=row, column=col)
            
            predictor = correlations.index[row - 2]
            target = correlations.columns[col - 2]
            count = counts.loc[predictor, target]
            correlation_value = correlations.loc[predictor, target]
            
            # Format the cell with correlation value and count
            if pd.notna(correlation_value):  # If correlation value exists
                cell.value = f"{correlation_value:.2f} ({count})"
            else:  # If correlation value is NaN, only display the count
                cell.value = f"- ({count})"
            
            # Extract numeric value for color scaling, handle NaN
            try:
                numeric_value = float(correlation_value)
            except (ValueError, TypeError):
                numeric_value = 0  # Default if correlation_value is NaN
            
            # Apply color based on normalized value
            color_hex = get_color_from_value(numeric_value, min_val, max_val, palette)
            cell.fill = PatternFill(start_color=color_hex[1:],  # Strip the '#' character
                                     end_color=color_hex[1:],    # Strip the '#' character
                                     fill_type='solid')

            # Apply number formatting to ensure correlation values are displayed as x.xx
            cell.number_format = '0.00'

    # Auto-adjust column widths
    for col in worksheet.columns:
        max_length = 0
        column = col[0].column_letter  # Get the column name
        for cell in col:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))  # Cast to string for length calculation
            except:
                pass
        adjusted_width = (max_length + 2) * 1.2  # Add extra space and adjust
        worksheet.column_dimensions[column].width = adjusted_width

    print(f'Correlation matrix with custom formatting saved to {sheet_name}')


def custom_corr_mat_to_excel(data, targets, predictors, group_column=None, file_name='correlation_matrix', method='pearson', threshold='Strong'):
    """
    Calculate the correlation matrix for the given targets and predictors, and write it to an Excel file
    with conditional formatting based on float values for the correlation matrix.
    The formatted values with counts (x.xx (count)) are updated after applying the color formatting.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the target and predictor columns.
    targets (list): List of target column names.
    predictors (list): List of predictor column names.
    group_column (str): Optional. Column name to group data by. A sheet will be created for each group.
    file_name (str): Name of the Excel file to write to (default is 'correlation_matrix.xlsx').
    method (str): Correlation method ('pearson', 'spearman', etc.).

    Returns:
    None
    """
    file_name = f"{file_name}.xlsx"
    
    # Create a Pandas Excel writer object
    with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
        
        # If no grouping column is provided, calculate the global correlation matrix
        if group_column is None:
            correlations, counts = _compute_corr_and_counts(data, targets, predictors, method)
            _write_corr_to_sheet(writer, correlations, counts, sheet_name=f"Global({method})", method=method)
        
        else:
            # 1. Global correlation matrix (all data)
            correlations, counts = _compute_corr_and_counts(data, targets, predictors, method)
            _write_corr_to_sheet(writer, correlations, counts, sheet_name=f"Global({method})", method=method)
            
            # 2. Grouped correlation matrices
            for group_value, group_data in data.groupby(group_column, observed=False):  # Suppress observed warning
                # Filter the data for the current group
                group_correlations, group_counts = _compute_corr_and_counts(group_data, targets, predictors, method)
                
                # Create a new sheet for the current group, truncate if name exceeds 31 characters
                sheet_name = f"{group_column}_{group_value}({method})"
                if len(sheet_name) > 31:
                    sheet_name = sheet_name[:31]
                _write_corr_to_sheet(writer, group_correlations, group_counts, sheet_name=sheet_name, method=method)

        # Add a chart for sorted correlation scores
        add_correlation_chart_openpyxl(writer, correlations, method=method, threshold=threshold)

        # Add the correlation strength information
        add_correlation_strength_table(writer, method=method)
    
    print(f'Correlation matrix saved to {file_name}')


def _compute_corr_and_counts(data, targets, predictors, method):
    """
    Compute the correlation matrix and count of valid (non-NaN) observations for each pair of variables.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the target and predictor columns.
    targets (list): List of target column names.
    predictors (list): List of predictor column names.
    method (str): Correlation method ('pearson', 'spearman', etc.).

    Returns:
    pd.DataFrame: Correlation matrix (with NaN for missing correlations).
    pd.DataFrame: Matrix of counts (number of valid rows used for each correlation).
    """
    # Initialize an empty correlation matrix
    correlations = pd.DataFrame(index=predictors, columns=targets)
    
    # Create a count matrix for non-NaN values
    counts = pd.DataFrame(index=predictors, columns=targets)
    
    for predictor in predictors:
        for target in targets:
            # Count valid rows (non-NaN) for this pair
            valid_rows = data[[predictor, target]].dropna().shape[0]
            counts.loc[predictor, target] = valid_rows
            
            # Only compute correlation if there are more than 1 valid rows
            if valid_rows > 1:
                correlations.loc[predictor, target] = data[[predictor, target]].corr(method=method).iloc[0, 1]
                # 'auto' to be decided here later (Kendall's Tau should be preferred over Spearman's correlation when there is very little data and many rank ties, if 'auto' mode' to select most suitable method based on data)
            else:
                # Set correlation as NaN but keep the count
                correlations.loc[predictor, target] = None  # Or use float('nan')
    
    return correlations, counts
