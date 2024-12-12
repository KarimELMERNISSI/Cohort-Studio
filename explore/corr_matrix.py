import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from openpyxl.styles import PatternFill, Font
from matplotlib.colors import to_hex
import plotly.express as px

####################################### CORRELATION MATRIX ###########################################

def plotly_corr_mat(
    data,
    targets,
    predictors,
    method="pearson",
    selected_color=px.colors.sequential.Viridis,
    corr_method_name=None,
    triangle="lower",  # 'lower', 'upper', or 'full'
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

    Returns:
    - plotly.graph_objs._figure.Figure: Plotly figure object.
    """
    # Filter the data to include only the target and predictor columns
    data_filtered = data[targets + predictors]

    # Calculate the correlation matrix
    corr_matrix = data_filtered.corr(method=method).loc[predictors, targets]

    # Handle symmetric matrix for identical targets and predictors
    if set(targets) == set(predictors):  # Check if target and predictor sets are identical
        mask = np.zeros_like(corr_matrix, dtype=bool)
        if triangle == "lower":
            mask[np.triu_indices_from(mask)] = True
        elif triangle == "upper":
            mask[np.tril_indices_from(mask)] = True
        corr_matrix = corr_matrix.mask(mask)

    # Generate correlation method title if not provided
    if corr_method_name is None:
        corr_method_name = method.capitalize()

    # Dynamically adjust figure size based on number of predictors and targets
    fig_width = max(10, 1.5 * len(targets))  # Ensure a minimum width
    fig_height = max(8, 1 * len(predictors))  # Ensure a minimum height

    # Create the interactive heatmap
    fig = px.imshow(
        corr_matrix,
        color_continuous_scale=selected_color,
        labels=dict(color="Correlation Coefficient"),
        title=f"Correlation Matrix ({corr_method_name})",
    )

    # Adjust the layout for text size and figure dimensions
    fig.update_layout(
        title_x=0.5,
        title_font=dict(size=20),
        width=fig_width * 100,  # Scale width for more space
        height=fig_height * 100,  # Scale height for more space
        font=dict(size=12),  # Smaller font size for better readability
        xaxis=dict(tickangle=45, tickmode='array', tickvals=list(range(len(corr_matrix.columns)))),
        yaxis=dict(tickmode='array', tickvals=list(range(len(corr_matrix.index)))),
        margin=dict(l=50, r=50, t=50, b=50),  # Adjust margins, reduced top margin to remove extra space
        autosize=True
    )

    # Adjust the axis labels' font size
    fig.update_xaxes(title_font=dict(size=14), tickfont=dict(size=10))
    fig.update_yaxes(title_font=dict(size=14), tickfont=dict(size=10))

    # Adjust the color bar to be on the right, properly sized and positioned
    fig.update_layout(
        coloraxis_colorbar=dict(
            len=0.8,  # Length of the color bar, scales to 80% of the figure height
            thickness=20,  # Width of the color bar
            ticks="outside",  # Position ticks outside of the color bar
            ticklen=5,  # Length of the ticks
            tickfont=dict(size=10),  # Font size for the color bar ticks
            x=1.05,  # Position color bar on the right side
            xanchor="left"  # Anchor the color bar to the left (relative to x=1.05)
        )
    )

    # Return the figure
    return fig


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
    corr_flat_filtered = corr_flat_sorted[corr_flat_sorted['Strength'].str.lower().isin(filtered_strengths)]

    # Limit to top N correlations
    if top_n:
        corr_flat_filtered = corr_flat_filtered.head(top_n)

    # Write the filtered correlation pairs and their strength categories to a new sheet
    corr_flat_filtered.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Filtered correlations (threshold: {threshold}, top {top_n}) added to {sheet_name} sheet.")


def generate_palette():
    """
    Generate a diverging palette similar to sns.diverging_palette(240, 10, s=70, l=85, as_cmap=True).
    
    Returns:
    list of str: List of color hex codes.
    """
    palette = sns.diverging_palette(240, 10, s=70, l=85, as_cmap=True)
    return [to_hex(palette(i)) for i in range(256)]


def get_color_from_value(value: float, min_val: float, max_val: float, palette: list) -> str:
    """
    Get the fill color based on the correlation value using an existing color palette.

    Parameters:
    value (float): The correlation value to determine the color.
    min_val (float): Minimum value in the correlation matrix for scaling.
    max_val (float): Maximum value in the correlation matrix for scaling.
    palette (list of str): List of color hex codes.

    Returns:
    str: The color code in hexadecimal format.
    """
    # Handle NaN values by defaulting to 0
    if np.isnan(value):
        value = 0
    
    # Normalize value between 0 and 1
    normalized_value = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5

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
