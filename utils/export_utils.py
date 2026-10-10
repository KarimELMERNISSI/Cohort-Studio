"""
Export Utilities.

Functions for exporting DataFrames to various formats (Excel, etc.)
"""
from io import BytesIO

import pandas as pd


def to_excel(df):
    """
    Convert a DataFrame to Excel bytes for download.
    
    Args:
        df: pandas DataFrame to export
        
    Returns:
        bytes: Excel file as bytes
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=True, sheet_name='Sheet1')
    return output.getvalue()


def to_excel_sheets(dataframes_dict: dict[str, pd.DataFrame]) -> bytes:
    """
    Converts multiple DataFrames into an Excel file with separate sheets.
    
    Args:
        dataframes_dict: Dictionary mapping sheet names to pandas DataFrames.
        
    Returns:
        bytes: Multi-sheet Excel workbook as bytes.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in dataframes_dict.items():
            safe_name = str(sheet_name)[:27]
            df.to_excel(writer, sheet_name=safe_name, index=True)
    return output.getvalue()


def to_csv_bytes(df: pd.DataFrame, index: bool = False) -> bytes:
    """
    Converts a pandas DataFrame to UTF-8 encoded CSV bytes.
    
    Args:
        df: The pandas DataFrame to convert.
        index: Whether to include the DataFrame index column.
        
    Returns:
        bytes: CSV content encoded as UTF-8 bytes.
    """
    return df.to_csv(index=index).encode("utf-8")


def fig_to_html_str(fig: Any, include_plotlyjs: str = "cdn") -> str:
    """
    Exports a Plotly figure object into a standalone interactive HTML string.
    
    Args:
        fig: Plotly Graph Object figure.
        include_plotlyjs: Strategy for including Plotly JavaScript library ('cdn' or True).
        
    Returns:
        str: Standalone HTML markup containing the interactive figure.
    """
    try:
        return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=True)
    except Exception as e:
        return f"<!-- Failed to export figure to HTML: {e!s} -->"


def fig_to_json_str(fig: Any) -> str:
    """
    Exports a Plotly figure object to a JSON specification string.
    
    Args:
        fig: Plotly Graph Object figure.
        
    Returns:
        str: Serialized JSON representation of figure traces and layout.
    """
    try:
        return fig.to_json()
    except Exception as e:
        return "{}"


def get_publication_plot_config(
    filename: str = "clinical_figure",
    format_type: str = "svg",
    width: int = 1200,
    height: int = 800,
    scale: int = 3,
) -> dict[str, Any]:
    """
    Generates a Plotly modebar configuration dict for publication-quality exports.
    
    Configures the in-browser camera icon to export vector SVG or 300+ DPI PNG
    directly without requiring server-side rendering dependencies.
    
    Args:
        filename: Base filename for downloaded graphic.
        format_type: Image format ('svg', 'png', 'jpeg', 'webp').
        width: Image width in pixels.
        height: Image height in pixels.
        scale: Resolution scaling multiplier (e.g. 3 for publication print quality).
        
    Returns:
        dict: Configuration dictionary for plotly_chart(fig, config=config).
    """
    return {
        "displaylogo": False,
        "modeBarButtonsToAdd": ["hoverclosest", "hovercompare"],
        "toImageButtonOptions": {
            "format": format_type,
            "filename": filename,
            "height": height,
            "width": width,
            "scale": scale,
        },
    }
