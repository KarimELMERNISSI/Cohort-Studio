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


def to_excel_sheets(dataframes_dict):
    """
    Converts multiple DataFrames into an Excel file with separate sheets.
    
    Args:
        dataframes_dict: dict mapping sheet names to DataFrames
        
    Returns:
        bytes: Excel file as bytes
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in dataframes_dict.items():
            if type(sheet_name) != 'str':
                sheet_name = str(sheet_name)
            df.to_excel(writer, sheet_name=sheet_name[:27], index=True)
    return output.getvalue()
