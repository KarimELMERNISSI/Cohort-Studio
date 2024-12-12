# multipage.py
import streamlit as st
from typing import Dict, Any, Callable, Optional, Union
from dataclasses import dataclass
import pandas as pd
import io
import chardet
import hashlib
import manage.file_handling as mf
import os
from pathlib import Path
import requests
from io import BytesIO
from PIL import Image
import base64

# @dataclass
# class Page:
#     title: str
#     function: Callable
#     icon: str = "📄"

def detect_encoding(file_content: bytes, num_lines: int = 100) -> str:
    """
    Detects the encoding of file bytes using chardet.

    Parameters:
    - file_content (bytes): The file content in bytes.
    - num_lines (int): The number of lines to read from the file for encoding detection. Default is 100.

    Returns:
    - str: The detected encoding of the file.

    Note:
    The function attempts to detect the encoding by analyzing a portion of the file's content. 
    It reads the specified number of lines and uses the chardet library to determine the encoding.
    """
    # Attempt to decode as UTF-8 first
    try:
        file_content.decode('utf-8')
        print("UTF-8 encoding detected with high priority")
        return 'utf-8'
    except UnicodeDecodeError:
        pass  # Continue with chardet detection if UTF-8 decoding fails

    detector = chardet.UniversalDetector(should_rename_legacy=True)
    detector.reset()

    # Split bytes into lines and analyze the first 'num_lines' lines
    lines = file_content.split(b'\n')
    for line in lines[:num_lines]:
        detector.feed(line)
        if detector.done:
            break
    detector.close()
    result = detector.result

    # If no encoding is found, fall back to chardet.detect on the entire content
    if result['encoding'] is None:
        result = chardet.detect(file_content)
    else:
        # Compare confidence with a full content scan and return the better match
        result_full = chardet.detect(file_content)
        if result_full['confidence'] > result['confidence']:
            print(f"Encoding detection result: {result_full}")
            return result_full['encoding']

    print(f"Encoding detection result: {result}")
    return result['encoding'] or 'utf-8'  # Fallback to utf-8 if no encoding detected


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


def load_csv_with_separator(file_bytes: bytes, encoding: str, num_lines: int = 30) -> Optional[pd.DataFrame]:
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
        st.error(f"Error loading CSV file: {str(e)}")
        return None


def load_dataframe(uploaded_file) -> Optional[pd.DataFrame]:
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
                df = pd.read_excel(uploaded_file, engine='openpyxl')
                st.success("Excel file loaded successfully!")
                return df
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")
                return None
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def create_empty_config() -> Dict[str, Any]:
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


def handle_data_upload():
    """Handle data upload and perform file comparison."""
    uploaded_file = st.session_state.get('uploaded_file', None)

    with st.sidebar.expander("Data Upload"):
        # Display file uploader in the sidebar
        uploaded_file = st.file_uploader("Upload File (XLSX, CSV)", type=['xlsx', 'csv'], key="file_uploader")

        # If a new file is uploaded, check if the file has changed
        if uploaded_file is not None:
            st.session_state['uploaded_file'] = uploaded_file  # Store file in session state

            # First-time upload or file change detection
            if "last_uploaded_file_hash" not in st.session_state:
                # First file upload
                st.session_state["data"] = load_dataframe(uploaded_file)
                st.session_state["last_uploaded_file_hash"] = get_file_hash(uploaded_file)
                st.success("Data uploaded successfully!")
            else:
                # Compare hashes of current and previous files
                current_file_hash = get_file_hash(uploaded_file)
                last_file_hash = st.session_state["last_uploaded_file_hash"]

                if current_file_hash != last_file_hash:
                    # File changed, reload data
                    st.session_state["data"] = load_dataframe(uploaded_file)
                    st.session_state["working_df"] = load_dataframe(uploaded_file)
                    st.session_state["last_uploaded_file_hash"] = current_file_hash
                    st.success("Data uploaded successfully!")
                else:
                    st.info("The file has not changed. No need to reload data.")


class Page:
    def __init__(self, title: str, function: Callable, icon: Union[str, Path] = "📄"):
        self.title = title  # Internal page title
        self.function = function
        self.icon = self._process_icon(icon)

    def _process_icon(self, icon: Union[str, bytes, Image.Image]) -> Union[str, Image.Image]:
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
    def __init__(self):
        self.pages: Dict[str, Page] = {}
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'config' not in st.session_state:
            st.session_state.config = create_empty_config()


    def add_page(self, title: str, function: Callable, icon: Union[str, Path] = "📄") -> None:
        """
        Add a new page to the app with a custom display label and icon.
        
        Args:
            title (str): The title of the page
            function (Callable): The function to render the page
            icon (Union[str, Path], optional): An emoji or path to an image file. Defaults to "📄".
        """
        self.pages[title] = Page(title=title, function=function, icon=icon)


    def _render_sidebar_icon(self, icon: Union[str, Image.Image], page_name: str, is_widget: bool = False) -> str:
        """
        Render icons for the sidebar using Markdown or plain text for widgets.
        
        Args:
            icon: Icon to render (emoji or PIL Image).
            page_name: Name of the page.
            is_widget: If True, return plain text for use in widgets like `checkbox`.
        
        Returns:
            A string suitable for use in Markdown or plain text, depending on `is_widget`.
        """
        # Case 1: Emoji
        if isinstance(icon, str) and len(icon) <= 4:  # Assume emoji
            return f"{icon} {page_name}"

        # Case 2: Image
        if isinstance(icon, Image.Image):
            # Convert image to Base64 string
            buffered = BytesIO()
            icon.thumbnail((60, 60))  # Resize to a smaller size for sidebar
            icon.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            if is_widget:
                # For widgets, return plain text without HTML
                return f"📊 {page_name}"
            else:
                # Return an HTML string for rendering in Markdown
                return f"""
                <div style="display: flex; align-items: center;">
                    <img src="data:image/png;base64,{img_str}" style="margin-right:10px;" width="20"/>
                    <span>{page_name}</span>
                </div>
                """

        # Case 3: Default fallback
        return f"📄 {page_name}"  # Fallback to a default emoji



    def _render_icon(self, icon: Union[str, Image.Image], width: int = 30) -> str:
        """
        Render icons for the page title (HTML-supported).
        
        Args:
            icon: Icon to render (emoji or PIL Image)
            width: Width of the icon in pixels
        
        Returns:
            Rendered HTML for display
        """
        if isinstance(icon, str):
            return icon  # Emojis work directly
        
        if isinstance(icon, Image.Image):
            # Convert the image to a base64-encoded string for rendering in HTML
            buffered = BytesIO()
            icon.thumbnail((width, width))  # Resize while maintaining aspect ratio
            icon.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return f'<img src="data:image/png;base64,{img_str}" width="{width}" height="{width}" style="vertical-align:middle; margin-right:10px;">'
        
        return "📄"


    def _handle_data_upload(self) -> None:
        """Handle data upload in sidebar."""
        with st.sidebar.expander("Data Upload"):
            try:
                handle_data_upload()
            except Exception as e:
                st.error(f"Error uploading file: {str(e)}")


    def run(self) -> None:
        """Run the multi-page app."""
        st.sidebar.title("Navigation")

        # Add data upload section in sidebar
        self._handle_data_upload()

        # Custom navigation labels in the sidebar
        selected_page = st.sidebar.selectbox(
            "Go to",
            list(self.pages.keys()),
            format_func=lambda x: self._render_sidebar_icon(self.pages[x].icon, x, is_widget=True)
        )

        # Show current dataset info if data is available
        if st.session_state.data is not None:
            with st.sidebar.expander("Current Dataset Info"):
                st.write(f"Rows: {len(st.session_state.data)}")
                st.write(f"Columns: {len(st.session_state.data.columns)}")

        # Render page title with icon
        icon_html = self._render_icon(self.pages[selected_page].icon, width=40)
        st.markdown(f"<h1 style='display: flex; align-items: center;'>{icon_html} {selected_page}</h1>", unsafe_allow_html=True)
        

        # Execute the function for the selected page
        self.pages[selected_page].function()


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
