import logging
import os

from utils.logging_config import single_line

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
            pass  # outside the workspace: only the file name is searched for below
    elif os.path.exists(original_path):
        return original_path
    elif not os.path.isabs(original_path):  # recorded relative to the data folder (AppState.recorded_path)
        from core.workspace import resolve_within
        from utils.data_paths import get_data_root

        try:
            candidate = resolve_within(get_data_root(), original_path)
        except ValueError:
            candidate = None
        if candidate is not None and candidate.exists():
            return str(candidate)

    # 2. Extract filename and search in common directories (a path recorded on Windows keeps its
    #    backslashes, which are not separators on Linux, e.g. in the Docker image)
    filename = os.path.basename(normalize_path(original_path))
    
    if search_dirs is None:
        if scope:
            search_dirs = [str(p) for p in scope.search_dirs()]
        else:
            from utils.data_paths import get_data_root, get_traces_dir, get_uploads_dir

            search_dirs = [get_data_root(), get_uploads_dir(), get_traces_dir()]

    if filename in ("", ".", ".."):
        return None
    from core.workspace import resolve_within

    for directory in search_dirs:
        try:  # the recorded name, looked up in each search directory, never outside it
            candidate_path = str(resolve_within(directory, filename))
        except ValueError:
            continue
        if os.path.isfile(candidate_path):
            logger.info("Path resolution found %s for %s", single_line(candidate_path), single_line(original_path))
            return candidate_path

    return None
