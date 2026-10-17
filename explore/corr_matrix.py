import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import to_hex
from openpyxl.styles import Font, PatternFill

from explore.correlation_strength import (
    STRENGTH_ORDER,
    STRENGTH_THRESHOLDS,
    CorrelationMethod,
    categorize_strength,
)

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
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
        zmid = 0,  # diverging scale centred on zero correlation (audit S13)
        zmin = -1,
        zmax = 1,
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



def get_yfiles_network_data(data, targets, predictors, method="pearson", threshold_category="Weak", correlation_matrix=None, focus_node=None):
    """Deprecated: use :func:`graph.builders.correlation.build_correlation_graph`.

    Kept for backward compatibility; returns plain-dict nodes/edges in the historical shape
    (``{"id", "properties"}`` / ``{"start", "end", "id", "properties"}``).
    """
    warnings.warn(
        "get_yfiles_network_data is deprecated; use graph.builders.correlation.build_correlation_graph",
        DeprecationWarning,
        stacklevel=2,
    )
    from graph.builders.correlation import build_correlation_graph, to_legacy_dicts

    if correlation_matrix is None:
        unique_cols = list(dict.fromkeys(list(targets) + list(predictors)))
        data_filtered = data[unique_cols].select_dtypes(include=[np.number]) if unique_cols else pd.DataFrame()
        if data_filtered.empty:
            return [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            correlation_matrix = data_filtered.corr(method=method)

    spec = build_correlation_graph(correlation_matrix, method=method, min_strength=threshold_category, focus=focus_node)
    return to_legacy_dicts(spec)


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
    # Thresholds come from the shared table (explore.correlation_strength)
    strength_df = pd.DataFrame({
        'Strength (absolute value)': [c.value for c in STRENGTH_ORDER],
        'Pearson': list(STRENGTH_THRESHOLDS[CorrelationMethod.PEARSON]),
        'Spearman': list(STRENGTH_THRESHOLDS[CorrelationMethod.SPEARMAN]),
        'Kendall': list(STRENGTH_THRESHOLDS[CorrelationMethod.KENDALL]),
    })

    # Write the strength table to a new sheet
    strength_df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Get access to the Excel writer's workbook and the new sheet
    workbook = writer.book
    worksheet = workbook[sheet_name]

    # Define a bold font
    bold_font = Font(bold=True)

    # Determine the column for the selected method to be bolded
    method_lower = str(method).lower()
    if method_lower == 'pearson':
        column_to_bold = 'B'  # Pearson is in column B
    elif method_lower == 'spearman':
        column_to_bold = 'C'  # Spearman is in column C
    elif method_lower == 'kendall':
        column_to_bold = 'D'  # Kendall is in column D
    else:
        column_to_bold = 'B'

    # Bold the relevant row in the column corresponding to the method
    for row in range(2, len(strength_df) + 2):  # +2 to account for header row
        worksheet[f'{column_to_bold}{row}'].font = bold_font

    # Auto-adjust column widths
    for col in worksheet.columns:
        max_length = 0
        column = col[0].column_letter  # Get the column name
        for cell in col:
            try:
                max_length = max(max_length, len(str(cell.value)))  # Cast to string for length calculation
            except Exception:
                pass  # a value that cannot be printed only leaves the column width as it is
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
    empty_summary = pd.DataFrame(columns=['Predictor', 'Target', 'Correlation', 'AbsCorrelation', 'Strength'])

    if correlations is None or correlations.empty:
        empty_summary.to_excel(writer, sheet_name=sheet_name, index=False)
        return

    # Flatten the correlation matrix and extract the pairs with correlation values
    corr_flat = correlations.stack().reset_index()
    if corr_flat.empty:
        empty_summary.to_excel(writer, sheet_name=sheet_name, index=False)
        return

    corr_flat.columns = ['Predictor', 'Target', 'Correlation']

    # Remove self-correlations (where Predictor == Target)
    corr_flat = corr_flat[corr_flat['Predictor'] != corr_flat['Target']]

    # Coerce numeric correlations and drop missing/non-numeric pairs
    corr_flat['Correlation'] = pd.to_numeric(corr_flat['Correlation'], errors='coerce')
    corr_flat = corr_flat.dropna(subset=['Correlation'])

    if corr_flat.empty:
        empty_summary.to_excel(writer, sheet_name=sheet_name, index=False)
        return

    # Sort by absolute correlation value in descending order
    corr_flat['AbsCorrelation'] = corr_flat['Correlation'].abs()
    corr_flat_sorted = corr_flat.sort_values(by='AbsCorrelation', ascending=False)

    corr_flat_sorted['Strength'] = corr_flat_sorted['AbsCorrelation'].apply(
        lambda value: categorize_strength(float(value), method).value
    )

    # Define the order of strength categories for filtering
    strength_order = ['negligible', 'weak', 'moderate', 'strong', 'very strong']

    # Filter based on the threshold
    threshold_str = str(threshold).lower()
    if threshold_str in strength_order:
        filter_index = strength_order.index(threshold_str)
        filtered_strengths = strength_order[filter_index:]
    else:
        filtered_strengths = strength_order

    # Filter the data to include only the desired strength categories
    corr_flat_sorted['Strength'] = corr_flat_sorted['Strength'].astype(str)
    corr_flat_filtered = corr_flat_sorted[corr_flat_sorted['Strength'].str.lower().isin(filtered_strengths)]

    # Limit to top N correlations
    if top_n:
        corr_flat_filtered = corr_flat_filtered.head(top_n)

    # Write the filtered correlation pairs and their strength categories to a new sheet
    corr_flat_filtered.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Filtered correlations (threshold: {threshold}, top {top_n}) added to {sheet_name} sheet.")


from functools import cache


@cache
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
                max_length = max(max_length, len(str(cell.value)))  # Cast to string for length calculation
            except Exception:
                pass  # a value that cannot be printed only leaves the column width as it is
        adjusted_width = (max_length + 2) * 1.2  # Add extra space and adjust
        worksheet.column_dimensions[column].width = adjusted_width

    print(f'Correlation matrix with custom formatting saved to {sheet_name}')


def custom_corr_mat_to_excel(
    data,
    targets,
    predictors,
    group_column=None,
    file_name='correlation_matrix',
    method='pearson',
    threshold='Strong'
):
    """
    Calculate the correlation matrix for the given targets and predictors, and write it to an Excel file
    with conditional formatting based on float values for the correlation matrix.
    The formatted values with counts (x.xx (count)) are updated after applying the color formatting.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the target and predictor columns.
    targets (list): List of target column names.
    predictors (list): List of predictor column names.
    group_column (str): Optional. Column name to group data by. A sheet will be created for each group.
    file_name (str or BytesIO): Name of the Excel file or BytesIO buffer to write to.
    method (str): Correlation method ('pearson', 'spearman', etc.).
    threshold (str): Correlation strength threshold for the summary chart.

    Returns:
    None
    """
    if isinstance(file_name, str):
        target_file = file_name if file_name.endswith('.xlsx') else f"{file_name}.xlsx"
    else:
        target_file = file_name

    method_clean = str(method).lower()

    # Guard against empty targets, predictors, or data to guarantee a valid workbook with at least one sheet
    if not targets or not predictors or data is None or getattr(data, "empty", False):
        with pd.ExcelWriter(target_file, engine='openpyxl') as writer:
            pd.DataFrame({"Status": ["No data or variables selected for correlation analysis."]}).to_excel(
                writer, sheet_name="Correlation Summary", index=False
            )
        return

    # Create a Pandas Excel writer object
    with pd.ExcelWriter(target_file, engine='openpyxl') as writer:
        
        # If no grouping column is provided, calculate the global correlation matrix
        if group_column is None:
            correlations, counts = _compute_corr_and_counts(data, targets, predictors, method=method_clean)
            _write_corr_to_sheet(writer, correlations, counts, sheet_name=f"Global({method_clean})", method=method_clean)
        
        else:
            # 1. Global correlation matrix (all data)
            correlations, counts = _compute_corr_and_counts(data, targets, predictors, method=method_clean)
            _write_corr_to_sheet(writer, correlations, counts, sheet_name=f"Global({method_clean})", method=method_clean)
            
            # 2. Grouped correlation matrices
            for group_value, group_data in data.groupby(group_column, observed=False):
                group_correlations, group_counts = _compute_corr_and_counts(group_data, targets, predictors, method=method_clean)
                raw_sheet_name = f"{group_column}_{group_value}({method_clean})"
                clean_sheet_name = re.sub(r'[\/\\?*:[\]]', '_', str(raw_sheet_name))[:31].strip()
                if not clean_sheet_name:
                    clean_sheet_name = "Group"
                _write_corr_to_sheet(writer, group_correlations, group_counts, sheet_name=clean_sheet_name, method=method_clean)

        # Add a chart for sorted correlation scores
        add_correlation_chart_openpyxl(writer, correlations, method=method_clean, threshold=threshold)

        # Add the correlation strength information
        add_correlation_strength_table(writer, method=method_clean)
    
    print(f'Correlation matrix saved to {target_file}')


MIN_PERIODS = 10  # pairwise correlations need at least this many complete pairs (audit S13)


def _compute_corr_and_counts(data, targets, predictors, method, min_periods=MIN_PERIODS):
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
    correlations = pd.DataFrame(index=predictors, columns=targets, dtype=object)
    
    # Create a count matrix for non-NaN values
    counts = pd.DataFrame(index=predictors, columns=targets, dtype=object)
    
    for predictor in predictors:
        for target in targets:
            # Handle diagonal (self-correlation) explicitly to prevent duplicate column indexing
            if predictor == target:
                if predictor in data.columns:
                    series = data[predictor].dropna()
                    valid_rows = int(len(series))
                    counts.loc[predictor, target] = valid_rows
                    if valid_rows > 1 and series.nunique() > 1:
                        correlations.loc[predictor, target] = 1.0
                    else:
                        correlations.loc[predictor, target] = None
                else:
                    counts.loc[predictor, target] = 0
                    correlations.loc[predictor, target] = None
                continue

            # Handle missing columns gracefully
            if predictor not in data.columns or target not in data.columns:
                counts.loc[predictor, target] = 0
                correlations.loc[predictor, target] = None
                continue

            pair_df = data[[predictor, target]].dropna()
            valid_rows = len(pair_df)
            counts.loc[predictor, target] = valid_rows
            
            # Enough complete pairs (threshold capped at the dataset size so tiny cohorts still work) and non-zero variance
            if (
                valid_rows >= max(2, min(min_periods, len(data)))
and pair_df[predictor].nunique() > 1 
                and pair_df[target].nunique() > 1
            ):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=Warning)
                    try:
                        val = pair_df.corr(method=method).iloc[0, 1]
                        correlations.loc[predictor, target] = float(val) if pd.notna(val) else None
                    except Exception:
                        correlations.loc[predictor, target] = None
            else:
                correlations.loc[predictor, target] = None
    
    return correlations, counts
