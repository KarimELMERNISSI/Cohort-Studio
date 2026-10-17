import difflib
import json
import os

import pandas as pd
import streamlit as st

from core.lineage import verify_trace
from core.workspace import safe_filename
from ingest.loaders import UPLOAD_TYPES
from manage.reproduction_manager import run_trace

# Import necessary modules for reproduction logic
from manage.trace_documenter import TraceDocumenter
from ui.progress import current_trace_session, mark_step
from ui.state import get_state
from utils.data_paths import get_traces_dir, get_uploads_dir
from utils.path_utils import normalize_path, resolve_path
from utils.session_scope import get_current_scope


def _uploads_dir() -> str:
    scope = get_current_scope()
    return str(scope.uploads_dir()) if scope else get_uploads_dir()


def _traces_dir() -> str:
    scope = get_current_scope()
    return str(scope.traces_dir()) if scope else get_traces_dir()


def _artifacts_dir() -> str:
    scope = get_current_scope()
    path = str(scope.artifacts_dir()) if scope else os.path.join(get_traces_dir(), "artifacts")
    os.makedirs(path, exist_ok=True)
    return path


def _resolve_user_path(text: str | None) -> str | None:
    """A manually entered path must exist inside the user's workspace (audit E4)."""
    if not text:
        return None
    fixed = text.replace("\\", "/")
    scope = get_current_scope()
    if scope is None:  # pages are behind sign-in: without a user there is no workspace to read from
        st.error("Sign in to reference a file of your workspace by its path.")
        return None
    bases = [scope.root, *scope.search_dirs()]
    for base in bases:
        try:
            candidate = scope.resolve_within(fixed if os.path.isabs(fixed) else os.path.join(str(base), fixed))
        except ValueError:
            continue
        if candidate.exists():
            return str(candidate)
    st.error("Only files inside your workspace can be referenced. Upload the file instead.")
    return None



_INTEGRITY_LABELS = {
    "verified": "integrity verified",
    "unverifiable": "integrity not verifiable",
    "tampered": "MODIFIED after recording",
    "discontinuous": "gap in the data chain",
}


def render_trace_integrity(trace: dict) -> None:
    """The hash chain of the trace (schema 2): verified, not verifiable (older traces) or broken."""
    check = verify_trace(trace)
    if check.status == "verified":
        st.success(
            f"Integrity verified: {check.n_steps} chained steps, head hash `{(check.head_hash or '')[:16]}`. "
            "The trace is tamper-evident; it is not an electronically signed record.",
            icon=":material/verified:",
        )
    elif check.status == "unverifiable":
        st.info(check.message, icon=":material/info:")
    else:
        st.error(f"Integrity check failed: {check.message}", icon=":material/gpp_bad:")


def app():
    
    # Option to upload trace directly
    uploaded_trace = st.file_uploader("Upload Trace File (.json)", type=["json"])
    
    trace_data = None
    selected_trace_file = None
    trace_path = None

    if uploaded_trace:
        try:
            trace_data = json.load(uploaded_trace)
            st.success(f"Result loaded from **{uploaded_trace.name}**")
            # If uploaded, use the uploaded file's directory as a pseudo trace path if needed, or None
            trace_path = os.path.join(_uploads_dir(), safe_filename(uploaded_trace.name))
        except Exception as e:
            st.error(f"Error parsing JSON: {e}")
            trace_path = None
    else:
        trace_dir = _traces_dir()
        if not os.path.exists(trace_dir):
            st.info("No analysis traces found (and no file uploaded).")
            return

        # List available traces
        traces = [f for f in os.listdir(trace_dir) if f.endswith(".json")]
        
        # Traces live in the user's workspace; the legacy name-prefix filter only applies without a scope
        username = st.session_state.get('username')
        if get_current_scope() is None and username and username != 'admin':
            traces = [f for f in traces if f.startswith(f"{username}_")]
        
        if not traces:
            st.info("No analysis traces found in your workspace.")
            return

        # Sort traces by modification time (newest first)
        traces.sort(key=lambda x: os.path.getmtime(os.path.join(trace_dir, x)), reverse=True)

        trace_options = {}
        for t in traces:
            try:
                t_path = os.path.join(trace_dir, t)
                with open(t_path, 'r', encoding='utf-8') as f:
                    t_data = json.load(f)
                    step_count = len(t_data.get('steps', []))
                    suffix = "step" if step_count == 1 else "steps"
                    integrity = _INTEGRITY_LABELS.get(verify_trace(t_data).status, "")
                    label = f"{t} [{step_count} {suffix}, {integrity}]"
                    trace_options[label] = t
            except Exception:
                # If a trace is corrupt or unreadable, just show the filename
                trace_options[t] = t

        selected_option = st.selectbox("Select Analysis Session", list(trace_options.keys()))
        selected_trace_file = trace_options[selected_option] if selected_option else None
        
        if selected_trace_file:
            trace_path = os.path.join(trace_dir, selected_trace_file)
            with open(trace_path, 'r', encoding='utf-8') as f:
                trace_data = json.load(f)
                


    if trace_data is not None and not (isinstance(trace_data, dict) and isinstance(trace_data.get("steps"), list)):
        st.error("This file is not a CohortStudio trace: it has no list of steps.")
        return
    if trace_data:
        # Checked on the trace as recorded, before file paths are resolved for this machine below
        render_trace_integrity(trace_data)

    # --- GLOBAL PERSISTENCE LOGIC (Fixes Loop for both Drag & Drop and Select) ---
    # Initialize persistent storage for resolved paths
    if 'resolved_paths' not in st.session_state:
        st.session_state['resolved_paths'] = {}

    # Apply persisted resolutions
    if trace_data:
        # Update source if previously resolved
        if trace_data.get('source_dataset') in st.session_state['resolved_paths']:
            trace_data['source_dataset'] = st.session_state['resolved_paths'][trace_data['source_dataset']]
        
        # Update steps if previously resolved
        for step in trace_data.get('steps', []):
            if step['function'] == 'enrichment':
                    original = step['params'].get('enrichment_file_path')
                    if original in st.session_state['resolved_paths']:
                        step['params']['enrichment_file_path'] = st.session_state['resolved_paths'][original]

    if trace_data:
            
        # --- PRE-FLIGHT CHECK ---
        # Use a dict to ensure unique paths and prevent duplicate widget keys
        missing_files_map = {}
        
        # Check source
        source_dataset = trace_data.get('source_dataset')
        if source_dataset and not resolve_path(source_dataset):
             missing_files_map[source_dataset] = {'type': 'source', 'path': source_dataset, 'description': 'Source Dataset'}

        # Check enrichment steps
        steps = trace_data.get('steps', [])
        for i, step in enumerate(steps):
             if step['function'] == 'enrichment':
                 enrich_path = step['params'].get('enrichment_file_path')
                 if enrich_path and not resolve_path(enrich_path):
                      if enrich_path not in missing_files_map:
                          missing_files_map[enrich_path] = {'type': 'enrichment', 'path': enrich_path, 'description': f"Enrichment File (Step {i+1})"}
                      else:
                          # Start appending usage info if reused
                          if "Steps" not in missing_files_map[enrich_path]['description']:
                               missing_files_map[enrich_path]['description'] += f", {i+1}"
        
        missing_files = list(missing_files_map.values())
        
        if missing_files:
            with st.expander("Missing Resources", expanded=True):
                st.warning("The following files referenced in the trace could not be found.")

                # --- NEW: Artifact Folder Selection & Fixes ---
                
                # Suffix suggestion & Auto-detection by Trace Name
                suggested_folder = ""
                potential_paths = []
                
                # 1. Try to find a folder matching the trace name (e.g. session_date.json -> folder session_date)
                # 1. Try to find a folder matching the trace name (e.g. session_date.json -> folder session_date)
                trace_basename = None
                if selected_trace_file:
                    trace_basename = os.path.splitext(selected_trace_file)[0]
                elif uploaded_trace:
                    trace_basename = os.path.splitext(uploaded_trace.name)[0]

                if trace_basename:
                    # Common locations for such a folder
                    possible_matches = []
                    # If we have a real trace path (not just a temp upload one), check near it
                    if trace_path and "uploads" not in trace_path:
                         possible_matches.append(os.path.join(os.path.dirname(trace_path), "artifacts", trace_basename))
                         possible_matches.append(os.path.join(os.path.dirname(trace_path), trace_basename))
                    
                    # Always check standard data locations
                    possible_matches.append(os.path.join(_artifacts_dir(), trace_basename))
                    for pm in possible_matches:
                         if os.path.exists(pm) and os.path.isdir(pm):
                             potential_paths.append(pm)

                # 2. Add generic 'artifacts' folders
                if trace_path: 
                    potential_paths.append(os.path.join(os.path.dirname(trace_path), "artifacts")) # data/traces/artifacts
                potential_paths.append(_artifacts_dir())

                for p in potential_paths:
                    if os.path.exists(p) and os.path.isdir(p):
                        suggested_folder = p
                        break
                
                help_text="Enter path to folder containing missing files."
                if suggested_folder:
                     help_text += f" Found likely candidate: `{suggested_folder}`"

                artifact_folder_input = st.text_input("Select Artifact Root Folder (Optional batch resolution)",
                                              value=suggested_folder if suggested_folder else "",
                                              help=help_text)
                artifact_folder_input = _resolve_user_path(artifact_folder_input) or ""

                # Auto-fix Windows paths (replace \ with /)
                if artifact_folder_input:
                     artifact_folder = artifact_folder_input.replace("\\", "/")
                     if artifact_folder != artifact_folder_input:
                          st.caption(f"Auto-corrected path to: `{artifact_folder}`")
                else:
                     artifact_folder = artifact_folder_input

                # Check for remaining Windows path issues (e.g. C:)
                if artifact_folder and (":/" in artifact_folder) and os.name == 'posix':
                    st.warning(
                        "You entered a Windows-style absolute path (e.g., `C:/...`). "
                        "Since the app is running in Docker (Linux), it cannot access your host's `C:` drive directly.\n\n"
                        "**Solution:**\n"
                        "1. Ensure your artifacts are inside the project's `data/` folder (mounted to `/app/data`).\n"
                        "2. Enter the path relative to the container, e.g., `/app/data/traces/artifacts`."
                    )
                
                if artifact_folder and os.path.exists(artifact_folder):
                    # Handle if user pointed to a file instead of a folder
                    if os.path.isfile(artifact_folder):
                         st.info(f"You selected a file (`{os.path.basename(artifact_folder)}`). Using its parent directory as the artifact folder.")
                         artifact_folder = os.path.dirname(artifact_folder)

                    st.success(f"Scanning folder: `{artifact_folder}`")
                    
                    folder_files = []
                    file_map = {} # Map lower_case -> real_filename
                    
                    # Show files in this folder as a control
                    try:
                        folder_files = os.listdir(artifact_folder)
                        # Build case-insensitive map
                        for f in folder_files:
                            file_map[f.lower()] = f
                            
                        with st.expander(f"View files in `{os.path.basename(artifact_folder)}` ({len(folder_files)} files)", expanded=False):
                            st.write(folder_files)
                    except Exception as e:
                        st.error(f"Could not list files: {e}")

                    resolved_count = 0
                    files_to_remove = []

                    for path_key, item in missing_files_map.items():
                         filename = os.path.basename(item['path'])
                         
                         candidate = None
                         # 1. Exact Match
                         if os.path.exists(os.path.join(artifact_folder, filename)):
                             candidate = os.path.join(artifact_folder, filename)
                         # 2. Case-Insensitive Match
                         elif filename.lower() in file_map:
                             real_name = file_map[filename.lower()]
                             candidate = os.path.join(artifact_folder, real_name)

                         if candidate and os.path.exists(candidate):
                             # Update trace data in memory
                             trace_data_updated = False
                             if item['type'] == 'source':
                                 trace_data['source_dataset'] = candidate
                                 trace_data_updated = True
                             elif item['type'] == 'enrichment':
                                 # Need to find which steps used this path and update them
                                 for step in trace_data.get('steps', []):
                                     if step['function'] == 'enrichment' and step['params'].get('enrichment_file_path') == item['path']:
                                         step['params']['enrichment_file_path'] = candidate
                                         trace_data_updated = True
                             
                             if trace_data_updated:
                                 # Persist this resolution in session state to prevent loops on rerun
                                 st.session_state['resolved_paths'][item['path']] = candidate
                                 
                                 st.toast(f"Resolved: {filename} -> {os.path.basename(candidate)}", icon="")
                                 files_to_remove.append(path_key)
                                 resolved_count += 1
                    
                    # Remove resolved files from missing map
                    for k in files_to_remove:
                        del missing_files_map[k]
                    
                    if resolved_count > 0:
                        st.info(f"Batch resolved {resolved_count} files!")
                        if not missing_files_map:
                             st.balloons()
                             st.rerun() # All done!

                # --- End Artifact Folder Selection ---

                # Re-evaluate missing files list after potential batch resolution
                missing_files = list(missing_files_map.values())

                if missing_files:
                    st.markdown("---")
                    st.caption("Please upload missing files, select from artifacts, or provide their location manually:")
                    
                    for item in missing_files:
                        cols = st.columns([2, 2, 2, 2]) # Added column for Dropdown
                        with cols[0]:
                            st.markdown(f"**{item['description']}**")
                            st.caption(f"`{os.path.basename(normalize_path(item['path']))}`")
                        
                        with cols[1]:
                            # File Uploader
                            uploaded_missing = st.file_uploader("Upload", type=UPLOAD_TYPES, key=f"pre_upload_{item['path']}")
                            if uploaded_missing:
                                save_path = os.path.join(_uploads_dir(), safe_filename(uploaded_missing.name))
                                with open(save_path, "wb") as f:
                                    f.write(uploaded_missing.getbuffer())
                                
                                # Persist resolution
                                st.session_state['resolved_paths'][item['path']] = save_path
                                st.toast(f"Uploaded & Resolved: {uploaded_missing.name}", icon="")
                                st.rerun()

                        with cols[2]:
                            # Dropdown (Select from Artifacts)
                            # Dropdown (Select from Artifacts)
                            if 'folder_files' in locals() and folder_files:
                                folder_files.sort() # Ensure consistent order
                                
                                # Fuzzy match for pre-selection
                                missing_basename = os.path.basename(normalize_path(item['path']))
                                match_index = 0
                                widget_key = f"pre_select_{item['path']}"
                                
                                # Try to find exact or close match
                                matches = difflib.get_close_matches(missing_basename, folder_files, n=1, cutoff=0.6)
                                if matches:
                                    best_match = matches[0]
                                    
                                    # FORCE FIX: Explicitly update session state if currently empty
                                    # This ensures the widget picks up the value immediately
                                    if widget_key not in st.session_state or st.session_state[widget_key] == "":
                                         st.session_state[widget_key] = best_match

                                    try:
                                        # Index passed to selectbox is 0-based index of OPTIONS. 
                                        # Options are [""] + folder_files.
                                        match_index = folder_files.index(best_match) + 1
                                    except ValueError:
                                        match_index = 0
                                
                                selected_artifact = st.selectbox(
                                    "Select from Artifacts", 
                                    [""] + folder_files, 
                                    index=match_index,
                                    key=widget_key
                                )
                                
                                if selected_artifact:
                                    candidate = os.path.join(artifact_folder, selected_artifact)
                                    if os.path.exists(candidate):
                                        st.session_state['resolved_paths'][item['path']] = candidate
                                        st.toast(f"Selected: {selected_artifact}", icon="")
                                        st.rerun()
                            else:
                                st.empty()

                        with cols[3]:
                            # Manual Input
                            manual_path = st.text_input("Or enter path", key=f"pre_manual_{item['path']}")
                            if manual_path:
                                # Auto-fix Windows paths
                                manual_path_fixed = manual_path.replace("\\", "/")
                                
                                # Verify existence inside the workspace
                                valid_path = _resolve_user_path(manual_path_fixed)
                                
                                if valid_path:
                                     # Persist resolution
                                     st.session_state['resolved_paths'][item['path']] = valid_path
                                     st.toast(f"Path Verified: {os.path.basename(valid_path)}", icon="")
                                     st.rerun() 

                    st.info("Uploaded files are saved to your workspace uploads folder, which is checked during reproduction.")
                    st.divider()

        # --- END PRE-FLIGHT CHECK ---
            
        source_dataset = trace_data.get('source_dataset')
        session_id = trace_data.get('session_id', 'Unknown Session')
        st.markdown(f"**Session:** `{session_id}`")
        st.markdown(f"**Source Dataset:** `{source_dataset}`")
        
        # Download Report Button
        try:
            documenter = TraceDocumenter(trace_data)
            report_buffer = documenter.generate_report()
            st.download_button(
                label="Download Transformation Report (.docx)",
                data=report_buffer,
                file_name=f"trace_report_{session_id}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                # the report of the session's own trace documents the current data (workflow progress)
                on_click=mark_step if session_id == current_trace_session() else "ignore",
                args=("report", "Word report exported") if session_id == current_trace_session() else None,
            )
        except Exception as e:
            st.warning(f"Report generation unavailable: {e}")
        
        steps = trace_data.get('steps', [])
        
        if not steps:
            st.warning("This trace has no steps recorded.")
            return

        # Rerun Mode Selection
        rerun_mode = st.radio(
            "Replay Mode",
            ["Fast (Use Snapshots)", "Full Replay (From Source)"],
            help="Fast: Uses saved intermediate files if available. Full Replay: Re-executes all steps from the original source dataset."
        )

        st.subheader("Analysis Process Path")
        
        for i, step in enumerate(steps):
            # Create a visual step container
            with st.container():
                col1, col2, col3 = st.columns([0.5, 4, 1.5])
                
                # Step Number
                with col1:
                    st.markdown(f"<h3 style='text-align: center; color: #4CAF50;'>{i+1}</h3>", unsafe_allow_html=True)
                
                # Step Details
                with col2:
                    st.markdown(f"**{step.get('description', 'Operation')}**")
                    st.caption(f"Function: `{step['function']}` | Time: {step['timestamp']}")
                    
                    # Show available data links (Artifacts)
                    artifacts = []
                    if step.get('output_dataset_path') and os.path.exists(step.get('output_dataset_path')):
                        artifacts.append(f"Snapshot: `{os.path.basename(step.get('output_dataset_path'))}`")
                    
                    if step.get('stats_cat_path') and os.path.exists(step.get('stats_cat_path')):
                        artifacts.append("Categorical Stats")
                    
                    if step.get('stats_num_path') and os.path.exists(step.get('stats_num_path')):
                        artifacts.append("Numerical Stats")
                        
                    if step.get('corr_matrix_path') and os.path.exists(step.get('corr_matrix_path')):
                        artifacts.append("Correlation Matrix")
                    
                    if artifacts:
                        st.markdown(" | ".join(artifacts))

                # Action Button
                with col3:
                    if st.button(f"Reproduce to Step {i+1}", key=f"jump_{i}", width='stretch'):
                        jump_to_step(trace_data, i, rerun_mode)
                
                st.markdown("---")

def jump_to_step(trace_data, target_step_index, rerun_mode):
    """
    Reproduces the analysis up to the target step index.
    """
    steps = trace_data.get('steps', [])
    target_step = steps[target_step_index]
    session_id = trace_data.get('session_id', 'unknown_session')
    
    # 1. Fast Mode: Try to load snapshot
    snapshot_path = target_step.get('output_dataset_path')
    resolved_snapshot = resolve_path(snapshot_path) if snapshot_path else None
    
    if rerun_mode == "Fast (Use Snapshots)" and resolved_snapshot:
        try:
            with st.spinner(f"Loading snapshot for Step {target_step_index+1}..."):
                # Determine file type
                path = resolve_path(target_step.get('output_dataset_path'))
                if not path:
                    st.warning(f"Snapshot not found: {target_step.get('output_dataset_path')}")
                    raise FileNotFoundError("Snapshot not found")

                if path.endswith('.parquet'):
                    df = pd.read_parquet(path)
                elif path.endswith('.csv'):
                    df = pd.read_csv(path)
                else:
                    st.error("Unknown file format for snapshot.")
                    return
                
                # The snapshot becomes the session dataset, linked to this trace (AppState)
                get_state().install_replay(df, trace_data, upto=target_step_index, mode="snapshot")
                st.success(f"Successfully loaded state at Step {target_step_index+1} (Fast Mode)")
                return
        except Exception as e:
            st.warning(f"Failed to load snapshot: {e}. Falling back to replay.")
            # Fallback to replay
    
    # 2. Full replay through the single engine (manage.reproduction_manager, audit E10)
    overrides = dict(st.session_state.get('resolved_paths', {}))
    source_dataset = trace_data.get('source_dataset')
    # a trace is user-supplied JSON: its paths are looked up in the workspace only (resolve_path), never as typed
    if (not source_dataset or resolve_path(str(source_dataset)) is None) and steps:
        first_step = steps[0]
        candidate_path = (first_step.get('params') or {}).get('source_path') if first_step.get('function') == 'initial_load' else None
        if candidate_path and resolve_path(candidate_path):
            source_dataset = candidate_path
    resolved_source = resolve_path(source_dataset) if source_dataset else None
    if source_dataset and not resolved_source:
        st.warning(f"Could not find source dataset at: `{source_dataset}`")
        st.info("The file path differs on this machine. Locate the file inside your workspace or upload it.")
        col_manual, col_upload = st.columns(2)
        with col_manual:
            manual_source = st.text_input("Enter path to source dataset:", value=os.path.basename(str(source_dataset)),
                                          key=f"manual_source_path_{session_id}", help="A path inside your workspace.")
        with col_upload:
            uploaded_source = st.file_uploader("Or upload source file:", type=UPLOAD_TYPES,
                                               key=f"upload_source_{session_id}", help="Stored in your workspace uploads folder.")
        if uploaded_source:
            save_path = os.path.join(_uploads_dir(), safe_filename(uploaded_source.name))
            with open(save_path, "wb") as f:
                f.write(uploaded_source.getbuffer())
            resolved_source = save_path
        elif manual_source:
            resolved_source = _resolve_user_path(manual_source)
        if not resolved_source:
            return
        st.session_state['resolved_paths'][str(source_dataset)] = resolved_source
        overrides[str(source_dataset)] = resolved_source

    progress_bar = st.progress(0, text="Starting reproduction...")

    def _on_progress(pct, msg):
        progress_bar.progress(min(max(int(pct), 0), 100) / 100, text=msg)

    with st.spinner(f"Replaying analysis up to Step {target_step_index + 1}..."):
        result = run_trace(trace_data, upto=target_step_index, path_overrides=overrides, progress_callback=_on_progress)
    progress_bar.empty()

    if result.missing_file is not None:
        step_index, missing_path = result.missing_file
        st.warning(f"Step {step_index + 1} needs a file that was not found: `{missing_path}`")
        key_suffix = f"{step_index}_{session_id}"
        col_m, col_u = st.columns(2)
        with col_m:
            manual = st.text_input(f"Enter path for the file of step {step_index + 1}:", value=os.path.basename(missing_path),
                                   key=f"manual_enrich_path_{key_suffix}", help="A path inside your workspace.")
        with col_u:
            uploaded = st.file_uploader(f"Or upload it (step {step_index + 1}):", type=UPLOAD_TYPES,
                                        key=f"upload_enrich_{key_suffix}", help="Stored in your workspace uploads folder.")
        resolved = None
        if uploaded:
            resolved = os.path.join(_uploads_dir(), safe_filename(uploaded.name))
            with open(resolved, "wb") as f:
                f.write(uploaded.getbuffer())
        elif manual:
            resolved = _resolve_user_path(manual)
        if resolved:
            st.session_state['resolved_paths'][missing_path] = resolved
            st.toast("File located; click the step again to replay.")
        return
    if result.error:
        st.error(result.error)
        return
    for note in result.warnings:
        st.warning(note)
    if result.reproducible and result.verified_steps:
        st.success(
            f"Reproduced exactly: the replayed table matches the {result.verified_steps} recorded fingerprints.",
            icon=":material/verified:",
        )
    elif result.not_replayable:
        st.info(f"{len(result.not_replayable)} step(s) could not be recomputed; see the warnings above.")
    get_state().install_replay(result.df, trace_data, upto=target_step_index, mode="full replay",
                               single_imputed=result.single_imputed, exact_imputed=result.exact_imputed)
    st.success(f"Successfully reproduced state at Step {target_step_index + 1} (full replay, {result.steps_applied} step(s))")
