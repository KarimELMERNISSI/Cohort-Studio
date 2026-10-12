"""Legacy loaders kept for the pages (the shell lives in ui/, new imports go through ingest/)."""
import functools
import hashlib
import io
import os
from typing import Any

import pandas as pd
import streamlit as st

import manage.file_handling as mf

# @dataclass
# class Page:
#     title: str
#     function: Callable
#     icon: str = ""

def detect_encoding(file_content: bytes, num_lines: int = 100) -> str:
    """
    Detects the encoding of file bytes using charset-normalizer.

    Parameters:
    - file_content (bytes): The file content in bytes.
    - num_lines (int): The number of lines to read from the file for encoding detection. Default is 100.

    Returns:
    - str: The detected encoding of the file.

    Note:
    The function attempts to detect the encoding by analyzing a portion of the file's content. 
    It reads the specified number of lines and uses the charset-normalizer library to determine the encoding.
    """
    from charset_normalizer import from_bytes

    # Attempt to decode as UTF-8 first
    try:
        file_content.decode('utf-8')
        print("UTF-8 encoding detected with high priority")
        return 'utf-8'
    except UnicodeDecodeError:
        pass  # Continue with charset-normalizer detection if UTF-8 decoding fails

    # Split bytes into lines and analyze the first 'num_lines' lines
    lines = file_content.split(b'\n')
    # Reconstruct the chunk from lines to pass to charset-normalizer
    chunk = b'\n'.join(lines[:num_lines])
    
    results = from_bytes(chunk)
    best_match = results.best()
    
    if best_match:
        # If we have a match, we can check if it's better than scanning the full content
        # But usually scanning the chunk is enough if the chunk is representative.
        # Let's trust the chunk first, but if confidence is low, scan full.
        
        # However, to keep it simple and robust:
        if best_match.coherence < 0.5: # Arbitrary threshold, but reasonable
             results_full = from_bytes(file_content)
             best_match_full = results_full.best()
             if best_match_full and best_match_full.coherence > best_match.coherence:
                 print(f"Encoding detection result (full scan): {best_match_full.encoding}")
                 return best_match_full.encoding

        print(f"Encoding detection result: {best_match.encoding}")
        return best_match.encoding
    
    # Fallback to full scan if no match on chunk
    results_full = from_bytes(file_content)
    best_match_full = results_full.best()
    
    if best_match_full:
        print(f"Encoding detection result (full scan fallback): {best_match_full.encoding}")
        return best_match_full.encoding

    return 'utf-8'  # Fallback to utf-8 if no encoding detected


def detect_separator(text: str) -> str:
    """
    Detect the most likely separator in a CSV text.
    
    Parameters:
    - text (str): Sample of the CSV content
    
    Returns:
    - str: Detected separator
    """
    # Common separators to check
    possible_separators = [',', ';', '\t', '|']
    
    # Count occurrences of each separator in the first few lines
    lines = text.split('\n')[:10]  # Check first 10 lines
    separator_counts = {sep: 0 for sep in possible_separators}
    
    for line in lines:
        if line.strip():  # Skip empty lines
            for sep in possible_separators:
                count = line.count(sep)
                # Only count if more than one occurrence (likely a real separator)
                if count > 0:
                    separator_counts[sep] += count

    # Find the separator with the most consistent count across lines
    if separator_counts:
        most_common = max(separator_counts.items(), key=lambda x: x[1])
        if most_common[1] > 0:
            return most_common[0]
    
    return ','  # Default to comma if no clear separator found


def load_csv_with_separator(file_bytes: bytes, encoding: str, num_lines: int = 30) -> pd.DataFrame | None:
    """
    Load a CSV from bytes with automatic delimiter detection.
    
    Parameters:
    - file_bytes (bytes): The file content in bytes
    - encoding (str): File encoding
    - num_lines (int): Number of lines to analyze for delimiter detection
    
    Returns:
    - Optional[pd.DataFrame]: Loaded DataFrame or None if error
    """
    try:
        # Decode the bytes to string
        text = file_bytes.decode(encoding)
        
        # Detect separator
        separator = detect_separator(text)
        st.info(f"Detected separator: '{separator}'")

        # Create DataFrame from bytes
        df = pd.read_csv(
            io.BytesIO(file_bytes),
            sep=separator,
            encoding=encoding,
            on_bad_lines='warn'  # More permissive parsing
        )
        return df

    except Exception as e:
        st.error(f"Error loading CSV file: {e!s}")
        return None


def load_dataframe(uploaded_file) -> pd.DataFrame | None:
    """
    Load DataFrame from uploaded Streamlit file.
    
    Parameters:
    - uploaded_file: Streamlit UploadedFile object
    
    Returns:
    - Optional[pd.DataFrame]: Loaded DataFrame or None if error
    """
    try:
        if uploaded_file is None:
            return None

        if isinstance(uploaded_file, str):
            current_path = os.getcwd()
            print("Current Path:", current_path)
            return mf.load_dataframe(uploaded_file)
            
        # Get file extension
        file_extension = uploaded_file.name.lower().split('.')[-1]
        
        # Read file bytes
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer for subsequent reads
        
        if file_extension == 'csv':
            # Detect encoding
            encoding = detect_encoding(file_bytes)
            st.info(f"Detected encoding: {encoding}")
            
            # Load CSV with automatic separator detection
            df = load_csv_with_separator(file_bytes, encoding)
            if df is not None:
                st.success("CSV file loaded successfully!")
                return df
            
        elif file_extension in ['xls', 'xlsx']:
            try:
                uploaded_file.seek(0)
                xls = pd.ExcelFile(uploaded_file, engine='openpyxl')
                sheet_names = xls.sheet_names
                if len(sheet_names) == 1:
                    # Single sheet: load directly (unchanged behavior)
                    df = pd.read_excel(xls, sheet_name=sheet_names[0])
                    st.success("Excel file loaded successfully!")
                    return df
                else:
                    # Multiple sheets detected: return sentinel so caller shows UI
                    return {"__multi_sheet__": True, "sheet_names": sheet_names}
            except Exception as e:
                st.error(f"Error reading Excel file: {e!s}")
                return None
        elif file_extension == 'parquet':
            try:
                df = pd.read_parquet(io.BytesIO(file_bytes))
                st.success("Parquet file loaded successfully!")
                return df
            except Exception as e:
                st.error(f"Error reading Parquet file: {e!s}")
                return None
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None

    except Exception as e:
        st.error(f"Error loading file: {e!s}")
        return None


def _load_excel_with_selection(file_bytes: bytes, sheet_names: list, selected_sheets: list, 
                               merge_key: str = None) -> pd.DataFrame | None:
    """
    Load one or more sheets from an Excel file.
    
    Parameters:
    - file_bytes: Raw bytes of the Excel file
    - sheet_names: All sheet names in the file
    - selected_sheets: List of sheet names the user selected
    - merge_key: Column name to merge on (None for single-sheet mode)
    
    Returns:
    - DataFrame or None
    """
    try:
        xls = pd.ExcelFile(io.BytesIO(file_bytes), engine='openpyxl')
        
        if len(selected_sheets) == 1:
            df = pd.read_excel(xls, sheet_name=selected_sheets[0])
            st.success(f"Sheet '{selected_sheets[0]}' loaded successfully!")
            return df
        
        # Load and merge multiple sheets
        dataframes = []
        for sheet in selected_sheets:
            sheet_df = pd.read_excel(xls, sheet_name=sheet)
            if merge_key and merge_key not in sheet_df.columns:
                st.warning(f"Merge key '{merge_key}' not found in sheet '{sheet}'. Skipping.")
                continue
            dataframes.append(sheet_df)
        
        if not dataframes:
            st.error("No valid sheets to load.")
            return None
        
        if merge_key:
            merged_df = functools.reduce(
                lambda left, right: pd.merge(left, right, on=merge_key, how='outer'),
                dataframes
            )
            st.success(f"Merged {len(dataframes)} sheets on '{merge_key}' → "
                       f"{merged_df.shape[0]} rows × {merged_df.shape[1]} columns")
            return merged_df
        else:
            # Concatenate if no merge key (stack vertically)
            concat_df = pd.concat(dataframes, ignore_index=True)
            st.success(f"Concatenated {len(dataframes)} sheets → "
                       f"{concat_df.shape[0]} rows × {concat_df.shape[1]} columns")
            return concat_df
    except Exception as e:
        st.error(f"Error loading Excel sheets: {e!s}")
        return None


def create_empty_config() -> dict[str, Any]:
    """Create an empty configuration structure."""
    return {
        "mask_families": {},
        "transformations": [],
        "computed_columns": {},
        "thresholds": {
            "ZSCORE_THRESHOLD": 3,
            "IQR_TOLERANCE": 1.5,
            "CATEGORICAL_OUTLIER_THRESHOLD": 1.0
        },
        "statistical_tests": {},
        "comparisons": {},
        "output_format": {
            "CSV": False,
            "XLSX": True,
            "PDF": False,
            "PNG": False
        },
        "folder_names": {
            "DATA_FOLDER": "./data/",
            "ENRICHMENT_FOLDER": "./data/enrichment/",
            "ENRICHED_FOLDER": "./data/enriched/",
            "DESCRIPTIVE_FOLDER": "./data/descriptive/",
            "HYPOTHESIS_TESTING_FOLDER": "./data/descriptive/hypothesis_testing/",
            "OUTLIERS_FOLDER": "./data/outliers/",
            "COMPARISONS_FOLDER": "./data/comparisons/"
        },
        "data_enrichments": {},
        "unicode_latex_mapping": {}
    }

def get_file_hash(uploaded_file) -> str:
    """Generate a hash of the file content using hashlib."""
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()
