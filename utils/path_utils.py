import logging
import os

logger = logging.getLogger(__name__)



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

    from utils.session_scope import get_current_scope

    scope = get_current_scope()
    # 1. The recorded path itself, but only inside the user's workspace when a user is signed in
    #    (trace files are user-supplied JSON and must not point anywhere on the server; audit E4).
    if scope is not None:
        try:
            candidate = scope.resolve_within(original_path)
            if candidate.exists():
                return str(candidate)
        except ValueError:
            pass
    elif os.path.exists(original_path):
        return original_path

    # 2. Extract filename and search in common directories
    filename = os.path.basename(original_path)
    
    if search_dirs is None:
        if scope:
            search_dirs = [str(p) for p in scope.search_dirs()]
        else:
            from utils.data_paths import get_data_root, get_traces_dir, get_uploads_dir

            search_dirs = [get_data_root(), get_uploads_dir(), get_traces_dir()]

    for directory in search_dirs:
        candidate_path = os.path.join(directory, filename)
        if os.path.exists(candidate_path):
            logger.info("Path resolution found %s for %s", candidate_path, original_path)
            return candidate_path
            
    return None
