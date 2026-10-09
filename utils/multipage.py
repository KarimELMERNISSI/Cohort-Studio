# multipage.py
import base64
import functools
import hashlib
import io
import os
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from PIL import Image

import manage.file_handling as mf
from app_pages.rag_sidebar import render_rag_sidebar
from manage.transformation_manager import TransformationManager

# @dataclass
# class Page:
#     title: str
#     function: Callable
#     icon: str = "📄"

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
    hash_md5 = hashlib.md5()
    for chunk in uploaded_file.getvalue():
        hash_md5.update(bytes([chunk]))
    return hash_md5.hexdigest()


def _finalize_upload(df, uploaded_file, is_new_file=True):
    """Finalize the upload: save source, init transformation trace, store data."""
    st.session_state["data"] = df
    st.session_state["last_uploaded_file_hash"] = get_file_hash(uploaded_file)

    # Initialize Transformation Manager if needed
    if 'transformation_manager' not in st.session_state:
        st.session_state.transformation_manager = TransformationManager()

    # Initialize Session Trace
    st.session_state.transformation_manager.initialize_session(
        uploaded_file.name, username=st.session_state.get('username')
    )

    # Save source file to artifacts for reproducibility
    session_id = st.session_state.transformation_manager.session_id
    artifact_dir = os.path.join("data", "traces", "artifacts", session_id)
    os.makedirs(artifact_dir, exist_ok=True)

    file_ext = os.path.splitext(uploaded_file.name)[1]
    source_path = os.path.join(artifact_dir, f"source{file_ext}")

    uploaded_file.seek(0)
    with open(source_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Update manager with the stored path
    st.session_state.transformation_manager.source_dataset = source_path
    st.session_state.transformation_manager.save_trace()

    label = "initial" if is_new_file else "new version"
    st.session_state.transformation_manager.add_step(
        "initial_load",
        {"filename": uploaded_file.name, "source_path": source_path},
        f"Uploaded {label} dataset: {uploaded_file.name}"
    )
    st.success("Data uploaded successfully!")


def _render_sheet_selection_ui(sheet_names: list, file_bytes: bytes, uploaded_file):
    """Render the multi-sheet selection UI and return the loaded DataFrame or None."""
    st.markdown(f"📑 **{len(sheet_names)} sheets detected:** {', '.join(sheet_names)}")

    load_mode = st.radio(
        "How would you like to load this file?",
        ["Load a single sheet", "Merge multiple sheets"],
        key="excel_load_mode",
        horizontal=True
    )

    if load_mode == "Load a single sheet":
        selected_sheet = st.selectbox(
            "Select sheet to load",
            sheet_names,
            key="excel_single_sheet_select"
        )
        if st.button("📥 Load Sheet", key="btn_load_single_sheet"):
            df = _load_excel_with_selection(file_bytes, sheet_names, [selected_sheet])
            if df is not None:
                _finalize_upload(df, uploaded_file)
                # Clear pending state
                st.session_state.pop('excel_sheets_pending', None)
                st.session_state.pop('excel_sheet_names', None)
                st.session_state.pop('excel_file_bytes', None)
                st.rerun()
    else:
        selected_sheets = st.multiselect(
            "Select sheets to merge",
            sheet_names,
            default=sheet_names,
            key="excel_multi_sheet_select"
        )

        if len(selected_sheets) >= 2:
            # Read columns from first selected sheet to suggest merge key
            try:
                preview_xls = pd.ExcelFile(io.BytesIO(file_bytes), engine='openpyxl')
                # Find common columns across all selected sheets
                common_cols = None
                for s in selected_sheets:
                    cols = set(pd.read_excel(preview_xls, sheet_name=s, nrows=0).columns)
                    common_cols = cols if common_cols is None else common_cols & cols
                common_cols = sorted(common_cols) if common_cols else []
            except Exception:
                common_cols = []

            if common_cols:
                merge_key = st.selectbox(
                    "Select merge key (row ID column)",
                    common_cols,
                    key="excel_merge_key"
                )
            else:
                st.warning("No common columns found across selected sheets. Sheets will be concatenated vertically.")
                merge_key = None

            if st.button("📥 Merge & Load Sheets", key="btn_load_merge_sheets"):
                df = _load_excel_with_selection(file_bytes, sheet_names, selected_sheets, merge_key)
                if df is not None:
                    _finalize_upload(df, uploaded_file)
                    # Clear pending state
                    st.session_state.pop('excel_sheets_pending', None)
                    st.session_state.pop('excel_sheet_names', None)
                    st.session_state.pop('excel_file_bytes', None)
                    st.rerun()
        elif len(selected_sheets) == 1:
            st.info("Select at least 2 sheets to merge, or switch to single-sheet mode.")
        else:
            st.info("Please select at least one sheet.")


def handle_data_upload():
    """Handle data upload and perform file comparison."""
    uploaded_file = st.session_state.get('uploaded_file', None)

    with st.sidebar.expander("Data Upload"):
        # Display file uploader in the sidebar
        uploaded_file = st.file_uploader("Upload File (XLSX, CSV)", type=['xlsx', 'csv', 'parquet'], key="file_uploader")

        # Show multi-sheet selection UI if pending
        if st.session_state.get('excel_sheets_pending', False):
            _render_sheet_selection_ui(
                st.session_state['excel_sheet_names'],
                st.session_state['excel_file_bytes'],
                st.session_state['uploaded_file']
            )
            return  # Don't proceed with normal flow while sheet selection is pending

        # If a new file is uploaded, check if the file has changed
        if uploaded_file is not None:
            st.session_state['uploaded_file'] = uploaded_file  # Store file in session state

            # Initialize Transformation Manager if needed
            if 'transformation_manager' not in st.session_state:
                st.session_state.transformation_manager = TransformationManager()

            # Determine if this is a new file or same file
            is_new_file = "last_uploaded_file_hash" not in st.session_state
            file_changed = False
            if not is_new_file:
                current_file_hash = get_file_hash(uploaded_file)
                last_file_hash = st.session_state["last_uploaded_file_hash"]
                file_changed = current_file_hash != last_file_hash
                if not file_changed:
                    st.info("The file has not changed. No need to reload data.")
                    return

            # Attempt to load the file
            result = load_dataframe(uploaded_file)

            # Check if multi-sheet sentinel was returned
            if isinstance(result, dict) and result.get("__multi_sheet__"):
                # Store state for sheet selection UI
                uploaded_file.seek(0)
                st.session_state['excel_sheets_pending'] = True
                st.session_state['excel_sheet_names'] = result['sheet_names']
                st.session_state['excel_file_bytes'] = uploaded_file.read()
                uploaded_file.seek(0)
                st.rerun()
            elif result is not None:
                # Normal single-sheet or CSV/Parquet load
                _finalize_upload(result, uploaded_file, is_new_file=is_new_file)
                if not is_new_file:
                    st.session_state["working_df"] = result


class Page:
    def __init__(self, title: str, function: Callable, icon: str | Path = "📄"):
        self.title = title  # Internal page title
        self.function = function
        self.icon = self._process_icon(icon)

    def _process_icon(self, icon: str | bytes | Image.Image) -> str | Image.Image:
        """
        Process the icon, supporting emojis, file paths, URLs, and image bytes
        
        Args:
            icon: Icon source (emoji, file path, URL, or image bytes)
        
        Returns:
            Processed icon (emoji or PIL Image)
        """
        # If it's an emoji, return as-is
        if isinstance(icon, str) and len(icon) <= 4:
            return icon
        
        # If it's already a PIL Image, return it
        if isinstance(icon, Image.Image):
            return icon
        
        try:
            # Try to handle bytes
            if isinstance(icon, bytes):
                return Image.open(BytesIO(icon))
            
            # Try to handle string (file path or URL)
            elif isinstance(icon, str):
                # Check if it's a URL
                if icon.startswith(('http://', 'https://')):
                    response = requests.get(icon)
                    return Image.open(BytesIO(response.content))
                
                # Try as a file path
                return Image.open(icon)
            
            # Fallback
            return "📊"
        
        except Exception as e:
            print(f"Error processing icon: {e}")
            return "📄"
        


class MultiPageApp:
    def __init__(self, default_page: str = None):
        self.pages: dict[str, Page] = {}
        self.default_page = default_page
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'config' not in st.session_state:
            st.session_state.config = create_empty_config()


    def add_page(self, title: str, function: Callable, icon: str | Path = "📄") -> None:
        """
        Add a new page to the app with a custom display label and icon.
        
        Args:
            title (str): The title of the page
            function (Callable): The function to render the page
            icon (Union[str, Path], optional): An emoji or path to an image file. Defaults to "📄".
        """
        self.pages[title] = Page(title=title, function=function, icon=icon)


    def _render_sidebar_icon(self, icon: str | Image.Image, page_name: str, is_widget: bool = False) -> str:
        """
        Render icons for the sidebar using Markdown or plain text for widgets.
        
        Args:
            icon: Icon to render (emoji or PIL Image).
            page_name: Name of the page.
            is_widget: If True, return plain text for use in widgets like `checkbox`.
        
        Returns:
            A string suitable for use in Markdown or plain text, depending on `is_widget`.
        """
        if not icon:
            return page_name

        if isinstance(icon, Image.Image):
            buffered = BytesIO()
            icon.thumbnail((60, 60))
            icon.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            if is_widget:
                return page_name
            else:
                return f"""
                <div style="display: flex; align-items: center;">
                    <img src="data:image/png;base64,{img_str}" style="margin-right:10px;" width="20"/>
                    <span>{page_name}</span>
                </div>
                """

        if isinstance(icon, str) and icon.strip():
            return f"{icon} {page_name}"

        return page_name


    def _render_icon(self, icon: str | Image.Image, width: int = 30) -> str:
        """
        Render icons for the page title (HTML-supported).
        
        Args:
            icon: Icon to render
            width: Width of the icon in pixels
        
        Returns:
            Rendered HTML for display, or empty string if no icon.
        """
        if not icon:
            return ""

        if isinstance(icon, Image.Image):
            buffered = BytesIO()
            icon.thumbnail((width, width))
            icon.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f'<img src="data:image/png;base64,{img_str}" width="{width}" height="{width}" style="vertical-align:middle; margin-right:10px;">'

        if isinstance(icon, str) and icon.strip():
            return f"{icon} "

        return ""


    def _handle_data_upload(self) -> None:
        """Handle data upload in sidebar."""
        try:
            handle_data_upload()
        except Exception as e:
            st.error(f"Error uploading file: {e!s}")


    def run(self) -> None:
        """Run the multi-page app."""
        st.sidebar.title("Navigation")

        # Add data upload section in sidebar
        self._handle_data_upload()

        # Render RAG Sidebar (Global)
        render_rag_sidebar()

        # Custom navigation labels in the sidebar
        page_list = list(self.pages.keys())
        default_index = 0
        if self.default_page and self.default_page in page_list:
            default_index = page_list.index(self.default_page)
        
        selected_page = st.sidebar.selectbox(
            "Go to",
            page_list,
            index=default_index,
            format_func=lambda x: self._render_sidebar_icon(self.pages[x].icon, x, is_widget=True)
        )

        # Show current dataset info if data is available
        if st.session_state.data is not None:
            with st.sidebar.expander("Current Dataset Info"):
                st.write(f"Rows: {len(st.session_state.data)}")
                st.write(f"Columns: {len(st.session_state.data.columns)}")

        # Render page title
        icon_html = self._render_icon(self.pages[selected_page].icon, width=40)
        if icon_html:
            st.markdown(f"<h1 style='display: flex; align-items: center;'>{icon_html} {selected_page}</h1>", unsafe_allow_html=True)
        else:
            st.title(selected_page)
        

        # Execute the function for the selected page
        self.pages[selected_page].function()

        # Add Logo at the bottom of sidebar
        st.sidebar.markdown("---")
        st.sidebar.image("assets/karim-app-logo.png", width="stretch")


    # def run(self) -> None:
    #     """Run the multi-page app."""
    #     st.sidebar.title("Navigation")

    #     # Add data upload section in sidebar
    #     self._handle_data_upload()

    #     # Only custom navigation labels in the sidebar
    #     selected_page = st.sidebar.selectbox(
    #         "Go to",
    #         list(self.pages.keys()),
    #         format_func=lambda x: f"{self.pages[x].icon} {x}"
    #     )
        
    #     # Show current dataset info if data is available
    #     if st.session_state.data is not None:
    #         with st.sidebar.expander("Current Dataset Info"):
    #             st.write(f"Rows: {len(st.session_state.data)}")
    #             st.write(f"Columns: {len(st.session_state.data.columns)}")

    #     # Execute the function for the selected page
    #     st.title(f"{self.pages[selected_page].icon} {selected_page}")
    #     self.pages[selected_page].function()
