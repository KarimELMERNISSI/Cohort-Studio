import os
import streamlit as st


def normalize_path(path_str: str) -> str:
    """
    Normalizes a file path by replacing backslashes with forward slashes.
    Crucial for handling Windows paths in a Linux (Docker) environment, 
    ensuring os.path.basename returns the correct filename.
    """
    if not path_str:
        return path_str
    return path_str.replace('\\', '/')

def resolve_path(original_path: str, search_dirs: list = None) -> str | None:
    """
    Attempts to resolve a file path.
    1. Checks if original_path exists validly.
    2. If not, extracts the filename and searches in `search_dirs`.
    
    Args:
        original_path (str): The absolute or relative path from the trace.
        search_dirs (list, optional): List of directories to search in. Defaults to ['data'].

    Returns:
        str | None: The valid path if found, or None if not found.
    """
    if not original_path:
        return None

    # 1. Check if the path works as-is (absolute or valid relative)
    if os.path.exists(original_path):
        return original_path

    # 2. Extract filename and search in common directories
    filename = os.path.basename(original_path)
    
    if search_dirs is None:
        search_dirs = ["data", "data/uploads", "data/traces", "."]

    for directory in search_dirs:
        candidate_path = os.path.join(directory, filename)
        if os.path.exists(candidate_path):
            st.toast(f"Build-in path resolution found file: {candidate_path} (original: {original_path})", icon="🧠")
            return candidate_path
            
    return None
