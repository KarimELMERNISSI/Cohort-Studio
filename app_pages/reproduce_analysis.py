import streamlit as st
import pandas as pd
import os
import json
from manage.db_manager import DBManager
from manage.trace_documenter import TraceDocumenter
# Import necessary modules for reproduction logic
import enrich.external_data as eed
from app_pages.transformation_logic import apply_variable_transformation
from app_pages.data_enrichment import apply_imputer, evaluate_formula_safely
from utils.multipage import load_dataframe
from utils.multipage import load_dataframe
from utils.path_utils import resolve_path, normalize_path
import difflib

def app():
    
    # Option to upload trace directly
    uploaded_trace = st.file_uploader("📂 Upload Trace File (.json)", type=["json"])
    
    trace_data = None
    selected_trace_file = None
    trace_path = None

    if uploaded_trace:
        try:
            trace_data = json.load(uploaded_trace)
            st.success(f"Result loaded from **{uploaded_trace.name}**")
            # If uploaded, use the uploaded file's directory as a pseudo trace path if needed, or None
            trace_path = os.path.join("data", "uploads", uploaded_trace.name) 
        except Exception as e:
            st.error(f"Error parsing JSON: {e}")
            trace_path = None
    else:
        trace_dir = "data/traces"
        if not os.path.exists(trace_dir):
            st.info("No analysis traces found (and no file uploaded).")
            return

        # List available traces
        traces = [f for f in os.listdir(trace_dir) if f.endswith(".json")]
        
        if not traces:
            st.info("No analysis traces found in data/traces/")
            return

        # Sort traces by modification time (newest first)
        traces.sort(key=lambda x: os.path.getmtime(os.path.join(trace_dir, x)), reverse=True)

        trace_options = {}
        for t in traces:
            try:
                t_path = os.path.join(trace_dir, t)
                with open(t_path, 'r') as f:
                    t_data = json.load(f)
                    step_count = len(t_data.get('steps', []))
                    suffix = "step" if step_count == 1 else "steps"
                    label = f"{t} [{step_count} {suffix}]"
                    trace_options[label] = t
            except Exception:
                # If a trace is corrupt or unreadable, just show the filename
                trace_options[t] = t

        selected_option = st.selectbox("Select Analysis Session", list(trace_options.keys()))
        selected_trace_file = trace_options[selected_option] if selected_option else None
        
        if selected_trace_file:
            trace_path = os.path.join(trace_dir, selected_trace_file)
            with open(trace_path, 'r') as f:
                trace_data = json.load(f)
                


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
            with st.expander("⚠️ Missing Resources", expanded=True):
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
                    possible_matches.extend([
                        os.path.join("data", "traces", "artifacts", trace_basename),
                        os.path.join("data", trace_basename)
                    ])
                    for pm in possible_matches:
                         if os.path.exists(pm) and os.path.isdir(pm):
                             potential_paths.append(pm)

                # 2. Add generic 'artifacts' folders
                if trace_path: 
                    potential_paths.append(os.path.join(os.path.dirname(trace_path), "artifacts")) # data/traces/artifacts
                potential_paths.append(os.path.join("data", "artifacts"))

                for p in potential_paths:
                    if os.path.exists(p) and os.path.isdir(p):
                        suggested_folder = p
                        break
                
                help_text="Enter path to folder containing missing files."
                if suggested_folder:
                     help_text += f" Found likely candidate: `{suggested_folder}`"

                artifact_folder_input = st.text_input("📂 Select Artifact Root Folder (Optional batch resolution)", 
                                              value=suggested_folder if suggested_folder else "",
                                              help=help_text)
                
                # Auto-fix Windows paths (replace \ with /)
                if artifact_folder_input:
                     artifact_folder = artifact_folder_input.replace("\\", "/")
                     if artifact_folder != artifact_folder_input:
                          st.caption(f"ℹ️ Auto-corrected path to: `{artifact_folder}`")
                else:
                     artifact_folder = artifact_folder_input

                # Check for remaining Windows path issues (e.g. C:)
                if artifact_folder and (":/" in artifact_folder) and os.name == 'posix':
                    st.warning(
                        "⚠️ You entered a Windows-style absolute path (e.g., `C:/...`). "
                        "Since the app is running in Docker (Linux), it cannot access your host's `C:` drive directly.\n\n"
                        "**Solution:**\n"
                        "1. Ensure your artifacts are inside the project's `data/` folder (mounted to `/app/data`).\n"
                        "2. Enter the path relative to the container, e.g., `/app/data/traces/artifacts`."
                    )
                
                if artifact_folder and os.path.exists(artifact_folder):
                    # Handle if user pointed to a file instead of a folder
                    if os.path.isfile(artifact_folder):
                         st.info(f"ℹ️ You selected a file (`{os.path.basename(artifact_folder)}`). Using its parent directory as the artifact folder.")
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
                            
                        with st.expander(f"📄 View files in `{os.path.basename(artifact_folder)}` ({len(folder_files)} files)", expanded=False):
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
                                 
                                 st.toast(f"Resolved: {filename} -> {os.path.basename(candidate)}", icon="✅")
                                 files_to_remove.append(path_key)
                                 resolved_count += 1
                    
                    # Remove resolved files from missing map
                    for k in files_to_remove:
                        del missing_files_map[k]
                    
                    if resolved_count > 0:
                        st.info(f"✅ Batch resolved {resolved_count} files!")
                        if not missing_files_map:
                             st.balloons()
                             st.rerun() # All done!

                # --- End Artifact Folder Selection ---

                # Re-evaluate missing files list after potential batch resolution
                missing_files = list(missing_files_map.values())

                if missing_files:
                    st.markdown("---")
                    st.caption("Please upload missing files, select from artifacts, or provide their location manually:")
                    
                    os.makedirs("data/uploads", exist_ok=True)
                    
                    for item in missing_files:
                        cols = st.columns([2, 2, 2, 2]) # Added column for Dropdown
                        with cols[0]:
                            st.markdown(f"**{item['description']}**")
                            st.caption(f"`{os.path.basename(normalize_path(item['path']))}`")
                        
                        with cols[1]:
                            # File Uploader
                            uploaded_missing = st.file_uploader(f"Upload", type=['csv', 'xlsx', 'parquet'], key=f"pre_upload_{item['path']}")
                            if uploaded_missing:
                                save_path = os.path.join("data/uploads", uploaded_missing.name)
                                with open(save_path, "wb") as f:
                                    f.write(uploaded_missing.getbuffer())
                                
                                # Persist resolution
                                st.session_state['resolved_paths'][item['path']] = save_path
                                st.toast(f"Uploaded & Resolved: {uploaded_missing.name}", icon="✅")
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
                                        st.toast(f"Selected: {selected_artifact}", icon="✅")
                                        st.rerun()
                            else:
                                st.empty()

                        with cols[3]:
                            # Manual Input
                            manual_path = st.text_input(f"Or enter path", key=f"pre_manual_{item['path']}")
                            if manual_path:
                                # Auto-fix Windows paths
                                manual_path_fixed = manual_path.replace("\\", "/")
                                
                                # Verify existence
                                valid_path = None
                                if os.path.exists(manual_path_fixed):
                                    valid_path = manual_path_fixed
                                elif os.path.exists(os.path.join("data", manual_path_fixed)):
                                    valid_path = os.path.join("data", manual_path_fixed)
                                
                                if valid_path:
                                     # Persist resolution
                                     st.session_state['resolved_paths'][item['path']] = valid_path
                                     st.toast(f"Path Verified: {os.path.basename(valid_path)}", icon="✅")
                                     st.rerun() 

                    st.info("Uploaded files are automatically saved to `data/uploads/`, which is checked during reproduction.")
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
                label="📄 Download Transformation Report (.docx)",
                data=report_buffer,
                file_name=f"trace_report_{session_id}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
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
                        artifacts.append(f"✅ Snapshot: `{os.path.basename(step.get('output_dataset_path'))}`")
                    
                    if step.get('stats_cat_path') and os.path.exists(step.get('stats_cat_path')):
                        artifacts.append("📊 Categorical Stats")
                    
                    if step.get('stats_num_path') and os.path.exists(step.get('stats_num_path')):
                        artifacts.append("📈 Numerical Stats")
                        
                    if step.get('corr_matrix_path') and os.path.exists(step.get('corr_matrix_path')):
                        artifacts.append("📉 Correlation Matrix")
                    
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
                
                st.session_state['working_df'] = df
                st.session_state['data'] = df # Update main data reference too
                st.success(f"Successfully loaded state at Step {target_step_index+1} (Fast Mode)")
                return
        except Exception as e:
            st.warning(f"Failed to load snapshot: {e}. Falling back to replay.")
            # Fallback to replay
    
    # 2. Full Replay (or Fallback)
    with st.spinner(f"Replaying analysis up to Step {target_step_index+1}..."):
        source_dataset = trace_data.get('source_dataset')
        
        # Fallback: Check first step for source_path (for newer traces)
        if (not source_dataset or not os.path.exists(source_dataset)) and len(steps) > 0:
            first_step = steps[0]
            if first_step['function'] == 'initial_load' and 'source_path' in first_step['params']:
                candidate_path = first_step['params']['source_path']
                if os.path.exists(candidate_path):
                    source_dataset = candidate_path

        # Handle initial load
        if source_dataset:
            resolved_source = resolve_path(source_dataset)
            if not resolved_source:
                st.warning(f"Could not find source dataset at: `{source_dataset}`")
                st.info("It seems the file path is different (possibly due to Docker/OS differences). Please locate the file manually.")
                
                # Check if we have a manual override in session state from a previous run or input
                manual_path_key = f"manual_source_path_{session_id}"
                
                col_manual, col_upload = st.columns(2)
                with col_manual:
                    manual_source = st.text_input("Enter path to source dataset:", value=os.path.basename(source_dataset), key=manual_path_key)
                
                with col_upload:
                    uploaded_source = st.file_uploader("Or upload source file:", type=['csv', 'xlsx', 'parquet'], key=f"upload_source_{session_id}")
                
                if uploaded_source:
                    # Save uploaded file
                    os.makedirs("data/uploads", exist_ok=True)
                    save_path = os.path.join("data/uploads", uploaded_source.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_source.getbuffer())
                    resolved_source = save_path
                    st.success(f"Using uploaded file: {save_path}")
                elif manual_source and os.path.exists(manual_source):
                     resolved_source = manual_source
                elif manual_source and os.path.exists(os.path.join("data", manual_source)):
                     resolved_source = os.path.join("data", manual_source)
            
            if resolved_source and os.path.exists(resolved_source):
                source_dataset = resolved_source # Update to valid path
                # Determine file type
                if source_dataset.endswith('.csv'):
                    df = pd.read_csv(source_dataset)
                elif source_dataset.endswith('.xlsx'):
                    df = pd.read_excel(source_dataset)
                elif source_dataset.endswith('.parquet'):
                    df = pd.read_parquet(source_dataset)
                else:
                    st.error(f"Unknown file type for source dataset: {source_dataset}")
                    return
            else:
                 if resolved_source:
                     st.error(f"Still cannot find file at: {resolved_source}")
                 return

        else:
            # Try DB Manager
            db_manager = DBManager()
            df, msg = db_manager.load_dataset(source_dataset)
        
        if df is None:
            st.error(f"Could not load source dataset '{source_dataset}'")
            return
            
        current_df = df.copy()
        
        # Initialize Progress Bar
        progress_bar = st.progress(0, text="Starting reproduction...")
        
        # Replay loop
        for i in range(target_step_index + 1):
            step = steps[i]
            func_name = step['function']
            description = step.get('description', func_name)
            
            # Update Progress
            progress_bar.progress((i + 1) / (target_step_index + 1), text=f"Step {i+1}: {description}")
            
            params = step['params']
            
            if func_name == "initial_load":
                continue # Already loaded

            try:
                if func_name == "enrichment":
                    # Logic to replay enrichment
                    enrichment_file_path = params.get("enrichment_file_path")
                    identifier = params.get("identifier")
                    strategy = params.get("strategy", "left")
                    conflict_resolution = params.get("conflict_resolution", "add")
                    
                    if enrichment_file_path:
                        resolved_enrich_path = resolve_path(enrichment_file_path)
                        
                        if not resolved_enrich_path:
                             st.warning(f"Enrichment file not found: `{enrichment_file_path}`")
                             manual_enrich_key = f"manual_enrich_path_{i}_{session_id}"
                             
                             col_enrich_manual, col_enrich_upload = st.columns(2)
                             with col_enrich_manual:
                                 manual_enrich = st.text_input(f"Enter path for enrichment file (Step {i+1}):", value=os.path.basename(enrichment_file_path), key=manual_enrich_key)
                             
                             with col_enrich_upload:
                                 uploaded_enrich = st.file_uploader(f"Or upload enrichment file (Step {i+1}):", type=['csv', 'xlsx'], key=f"upload_enrich_{i}_{session_id}")
                             
                             if uploaded_enrich:
                                 os.makedirs("data/uploads", exist_ok=True)
                                 save_path_enrich = os.path.join("data/uploads", uploaded_enrich.name)
                                 with open(save_path_enrich, "wb") as f:
                                     f.write(uploaded_enrich.getbuffer())
                                 resolved_enrich_path = save_path_enrich
                                 st.success(f"Using uploaded enrichment file: {save_path_enrich}")
                             elif manual_enrich and os.path.exists(manual_enrich):
                                 resolved_enrich_path = manual_enrich
                             elif manual_enrich and os.path.exists(os.path.join("data", manual_enrich)):
                                 resolved_enrich_path = os.path.join("data", manual_enrich)

                        if resolved_enrich_path and os.path.exists(resolved_enrich_path):
                            enrichment_df = load_dataframe(resolved_enrich_path)
                        if enrichment_df is not None:
                            # Ensure identifier is a list/set as expected by add_data
                            # In data_enrichment.py it handles str vs list, but let's be safe
                            # eed.add_data expects left_id_names and right_id_names
                            
                            # Calculate additional columns (all columns in enrichment_df except identifier)
                            # Note: In data_enrichment.py, it calculates this. We should do the same.
                            identifier_set = {identifier} if isinstance(identifier, str) else set(identifier)
                            additional_columns = list(set(enrichment_df.columns) - identifier_set)

                            current_df = eed.add_data(
                                df=current_df, 
                                additional_df=enrichment_df, 
                                left_id_names=identifier, 
                                right_id_names=identifier, 
                                additional_cols=additional_columns,
                                strategy=strategy, 
                                conflict_resolution=conflict_resolution
                            )
                        else:
                            st.error(f"Failed to load enrichment file: {enrichment_file_path}")
                            return
                    else:
                        st.warning(f"Enrichment file not found: {enrichment_file_path}")

                elif func_name == "variable_computation":
                     # Replay variable computation
                     formula = params.get("formula")
                     variable_name = params.get("variable_name")
                     
                     success, msg, current_df = evaluate_formula_safely(formula, variable_name, current_df)
                     if not success:
                         st.error(f"Failed to replay variable computation: {msg}")
                         return

                elif func_name == "imputation":
                    # Replay imputation
                    numerical_imputation_method = params.get("numerical_imputation_method", "Mean")
                    categorical_imputation_method = params.get("categorical_imputation_method", "Most_Frequent")
                    cat_encoder = params.get("cat_encoder", False)
                    num_scaler = params.get("num_scaler", False)
                    remainder_columns = params.get("remainder_columns", "auto")
                    remainder_strategy = params.get("remainder_strategy", "passthrough")
                    remainder_threshold = params.get("remainder_threshold", 0.3)
                    
                    current_df, _ = apply_imputer(
                        data=current_df,
                        numerical_imputation_method=numerical_imputation_method,
                        categorical_imputation_method=categorical_imputation_method,
                        cat_encoder=cat_encoder,
                        num_scaler=num_scaler,
                        remainder_columns=remainder_columns,
                        remainder_strategy=remainder_strategy,
                        remainder_threshold=remainder_threshold,
                        debug=False
                    )

                elif func_name == "variable_transformation":
                    # Replay variable transformation
                    result_df = apply_variable_transformation(current_df, params)
                    if result_df is not None:
                        current_df = pd.concat([current_df, result_df], axis=1)
                    else:
                        st.warning(f"Variable transformation returned no data at step {i+1}")

                elif func_name == "data_type_conversion":
                    # Replay data type conversion
                    columns = params.get("columns", [])
                    new_type = params.get("new_type")
                    
                    for col in columns:
                        if new_type == "datetime":
                            current_df[col] = pd.to_datetime(current_df[col])
                        else:
                            current_df[col] = current_df[col].astype(new_type)

                elif func_name == "missing_value_handling":
                    # Replay missing value handling
                    strategy = params.get("strategy")
                    columns = params.get("columns", [])
                    fill_value = params.get("fill_value")

                    if strategy == "Drop rows":
                        current_df.dropna(subset=columns, inplace=True)
                    elif strategy == "Fill with mean":
                        for col in columns:
                            current_df[col].fillna(current_df[col].mean(), inplace=True)
                    elif strategy == "Fill with median":
                        for col in columns:
                            current_df[col].fillna(current_df[col].median(), inplace=True)
                    elif strategy == "Fill with mode":
                        for col in columns:
                            # Safely handle mode if empty
                            mode_val = current_df[col].mode()
                            if not mode_val.empty:
                                current_df[col].fillna(mode_val[0], inplace=True)
                    elif strategy == "Fill with value":
                         for col in columns:
                            current_df[col].fillna(fill_value, inplace=True)

                # ... Add other replay handlers as needed ...
                    
            except Exception as e:
                st.error(f"Error replaying step {i+1} ({func_name}): {e}")
                return

        st.session_state['working_df'] = current_df
        st.session_state['data'] = current_df
        st.session_state['enriched_df'] = current_df
        progress_bar.empty() # clear progress bar
        st.success(f"Successfully reproduced state at Step {target_step_index+1} (Full Replay)")
