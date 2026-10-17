import json
import os
import re

import pandas as pd
import streamlit as st

from core.workspace import UserScope


def get_user_taxonomy_dir(username):
    """Taxonomy directory of the user's workspace (no shared fallback; audit E4)."""
    if not username:
        return None
    return str(UserScope.for_user(username).taxonomy_dir())

def get_taxonomy_versions(username=None):
    """Returns list of (version_int, filepath) sorted descending."""
    target_dir = get_user_taxonomy_dir(username)
    if not target_dir or not os.path.exists(target_dir):
        return []
    
    # Scan for directories starting with 'v'
    versions = []
    for item in os.listdir(target_dir):
        full_path = os.path.join(target_dir, item)
        if os.path.isdir(full_path):
            # Check for 'vX' pattern
            match = re.search(r"^v(\d+)$", item)
            if match:
                v_num = int(match.group(1))
                # Confirm taxonomy file exists inside
                tax_path = os.path.join(full_path, "taxonomy_metadata.json")
                if os.path.exists(tax_path):
                    versions.append((v_num, tax_path))
            
    return sorted(versions, key=lambda x: x[0], reverse=True)
# Ensure clean imports consistent with your environment
from graph.builders.taxonomy import (
    CATEGORY_LABELS,
    RELATION_LABELS,
    build_taxonomy_graph,
    make_formula_id,
    parse_formula_id,
)
from graph.model import LayoutHint, RenderOptions
from graph.ui import current_theme, render_graph, render_graph_exports, renderer_selector
from ui.components import no_dataset

# Lazy-compatible imports for clustering (module itself is light)
from utils.clustering_utils import (
    compute_cluster_profiles,
    fit_dbscan,
    fit_gaussian_mixture,
    fit_kmeans,
    prepare_data_for_clustering,
    run_famd,
    run_pca,
    run_tsne,
    run_umap,
)
from utils.optional import available_methods

# ==========================================
# ==========================================
# 1. GRAPH HELPER FUNCTIONS
# ==========================================

def repair_taxonomy_links(taxonomy, formulas_registry):
    """
    Ensures taxonomy variables have correct related_formula_ids based on registry.
    Run this on load to fix desynchronized files.
    """
    if not taxonomy or not formulas_registry:
        return taxonomy
        
    # Create reverse map of formula usage
    var_to_formulas = {} # var_id -> set(f_id)
    
    for f_id, f_data in formulas_registry.items():
        # Inputs
        for inp in f_data.get('input_variables', []):
            if inp not in var_to_formulas: var_to_formulas[inp] = set()
            var_to_formulas[inp].add(f_id)
            
        # Output
        out = f_data.get('output_variable')
        if out:
             if out not in var_to_formulas: var_to_formulas[out] = set()
             var_to_formulas[out].add(f_id)
             
    # Update Taxonomy
    for var_id, f_ids in var_to_formulas.items():
        if var_id in taxonomy:
            # Get existing
            current = set(taxonomy[var_id].get('related_formula_ids', []))
            # Merge
            new_set = current.union(f_ids)
            if len(new_set) > len(current):
                taxonomy[var_id]['related_formula_ids'] = list(new_set)
                
    return taxonomy

# Graph construction and styling live in graph.builders.taxonomy and graph.palette.

# ==========================================
# 2. MAIN APP LOGIC
# ==========================================

def load_and_repair_taxonomy(target_path, rag_manager=None):
    """
    Robust loading function:
    1. Loads Taxonomy & Formulas
    2. Repairs Links immediately
    3. Updates Session State
    4. Syncs with RAG Manager (if active)
    """
    try:
        # 1. Load Taxonomy
        with open(target_path, encoding="utf-8") as f:
            loaded_taxonomy = json.load(f)

        # 2. Load Formulas
        formulas_path = os.path.join(os.path.dirname(target_path), "formulas_metadata.json")
        loaded_formulas = {}
        if os.path.exists(formulas_path):
            with open(formulas_path, encoding="utf-8") as f:
                loaded_formulas = json.load(f)

        # 3. Repair Links
        # This modifies loaded_taxonomy in place
        loaded_taxonomy = repair_taxonomy_links(loaded_taxonomy, loaded_formulas)

        # 4. Update Session State
        st.session_state.offline_taxonomy = loaded_taxonomy
        st.session_state.offline_formulas = loaded_formulas
        st.session_state["loaded_path"] = target_path

        # 5. Update RAG Manager
        if rag_manager:
            rag_manager.variable_taxonomy = loaded_taxonomy
            rag_manager.formulas_registry = loaded_formulas
            
        return True, len(loaded_formulas)
    except Exception as e:
        return False, str(e)

def app():
    """
    Main application function of the Variable map page (Knowledge > Variable map).

    Handles:
    1. Loading and saving taxonomy versions.
    2. Visualizing variables and formulas as a graph.
    3. Generating new taxonomies using RAG.
    4. Enriching existing taxonomies with external knowledge.
    """

    # --- 1. Session State & Setup ---
    if 'offline_taxonomy' not in st.session_state:
        st.session_state.offline_taxonomy = None
    if 'offline_formulas' not in st.session_state:
        st.session_state.offline_formulas = {}

    rag_manager = st.session_state.get('rag_manager')
    
    # HOTFIX: Check for stale instance (missing new method)
    if rag_manager and not hasattr(rag_manager, 'enrich_variable_taxonomy'):
        st.warning("Detecting updated code... Re-initializing AI system.")
        st.session_state.pop('rag_manager', None)
        rag_manager = None
        st.rerun()

    rag_active = rag_manager is not None and rag_manager.initialized

    # --- 2. Sidebar Controls ---
    st.sidebar.header("Data Management")
    
    # Mode Indicator
    if rag_active:
        st.sidebar.success("AI System Active")
    else:
        st.sidebar.warning("Offline Mode")

    # Persistence (Save/Load)
    col_p1, col_p2 = st.sidebar.columns(2)
    
    
    # Load Logic
    username = st.session_state.get('username')
    if not username:
        st.sidebar.error("No user detected. Please login.")
    avail_versions = get_taxonomy_versions(username)
    
    if avail_versions:
        # Dropdown for version selection
        v_options = [f"v{v[0]}" for v in avail_versions]
        v_paths = {f"v{v[0]}": v[1] for v in avail_versions}
        
        selected_v_label = col_p1.selectbox("Select Version", v_options, index=0, label_visibility="collapsed")
        target_path = v_paths[selected_v_label]
        
        if col_p1.button("Load"):
            success, info = load_and_repair_taxonomy(target_path, rag_manager if rag_active else None)
            
            if success:
                st.toast(f"Loaded {info} formulas and synced!", icon="")
                # Trigger a single rerun to refresh UI with new state
                st.rerun()
            else:
                 st.sidebar.error(f"Load failed: {info}")
    else:
        col_p1.warning("No saved versions.")

    # --- 3. Resolve Taxonomy Source ---
    taxonomy = None
    if rag_active and rag_manager.variable_taxonomy:
        taxonomy = rag_manager.variable_taxonomy
    elif st.session_state.offline_taxonomy:
        taxonomy = st.session_state.offline_taxonomy
        
    formulas_registry = st.session_state.get("offline_formulas", {})
    
    # Sync Manager if needed
    if rag_active:
        mgr_registry = getattr(rag_manager, 'formulas_registry', {})
        if not mgr_registry and formulas_registry:
            rag_manager.formulas_registry = formulas_registry
        elif mgr_registry and not formulas_registry:
            # Rare case: Manager has info, session state doesn't? Sync back?
            # Usually session state is master. Let's assume session state wins, or we merge?
            # For safety, let's trust session state, but if session state is empty, take manager.
            formulas_registry = mgr_registry
            st.session_state.offline_formulas = mgr_registry

    # Save Button (Active only if taxonomy exists)
    if taxonomy:
        # Determine Next Version
        current_versions = get_taxonomy_versions(username)
        next_v = 1
        if current_versions:
            next_v = current_versions[0][0] + 1
        
        # Save Controls
        if col_p2.button(f"Save New (v{next_v})"):
            try:
                # User Isolation: Save to user folder
                user_dir = get_user_taxonomy_dir(username)
                if not user_dir:
                    raise ValueError("Sign in to save a taxonomy version.")
                new_dir = os.path.join(user_dir, f"v{next_v}")
                os.makedirs(new_dir, exist_ok=True)
                
                new_path = os.path.join(new_dir, "taxonomy_metadata.json")
                with open(new_path, "w") as f:
                    json.dump(taxonomy, f, indent=2)
                
                # Save Formulas
                new_formulas_path = os.path.join(new_dir, "formulas_metadata.json")
                with open(new_formulas_path, "w") as f:
                    json.dump(formulas_registry, f, indent=2)
                
                st.session_state["loaded_path"] = new_path # Update context
                st.toast(f"Saved Version v{next_v}!", icon="")
                st.rerun() 
            except Exception as e:
                st.sidebar.error(f"Save failed: {e}")

        # Overwrite Option
        loaded_path = st.session_state.get("loaded_path")
        if loaded_path and os.path.exists(loaded_path):
            current_v_name = os.path.basename(loaded_path)
            # Use a simpler label or icon to save space
            if col_p2.button("Overwrite"):
                try:
                    with open(loaded_path, "w") as f:
                        json.dump(taxonomy, f, indent=2)
                    
                    # Overwrite Formulas
                    # Overwrite Formulas
                    loaded_formulas_path = os.path.join(os.path.dirname(loaded_path), "formulas_metadata.json")
                    with open(loaded_formulas_path, "w") as f:
                         json.dump(formulas_registry, f, indent=2)
                    st.toast(f"Overwritten {current_v_name}!", icon="")
                except Exception as e:
                    st.sidebar.error(f"Overwrite failed: {e}")
    else:
        col_p2.button("Save", disabled=True)

    st.sidebar.divider()
    
    # --- 4. Persistent Graph Actions ---
    with st.sidebar.expander("Graph Actions", expanded=True):
        
        # A. Enrichment (Only if taxonomy exists)
        if taxonomy:
            st.markdown("**Taxonomy Enrichment**")
            st.caption("Expand graph with external knowledge.")
            enrich_dist = st.slider("Neighborhood Distance", 1, 3, 1, help="Depth of external variables to find.")
            
            if st.button("Enrich Taxonomy"):
                if not rag_active:
                     st.error("RAG System Required")
                else:
                    # Get columns info context
                    columns_info = []
                    if "working_df" in st.session_state:
                         df = st.session_state["working_df"]
                         for col in df.columns:
                            info = {"name": col}
                            # Basic Stats
                            if pd.api.types.is_numeric_dtype(df[col]):
                                col_data = df[col].dropna()
                                if not col_data.empty:
                                    info["stats"] = {
                                        "min": float(col_data.min()),
                                        "max": float(col_data.max()),
                                        "mean": float(col_data.mean())
                                    }
                            else:
                                info["top_values"] = {str(k): v for k, v in df[col].value_counts().head(5).to_dict().items()}
                            columns_info.append(info)
                    
                    # Progress Bar
                    prog_bar = st.sidebar.progress(0, text="Enriching taxonomy...")
                    def update_prog(p, t):
                        prog_bar.progress(p, text=t)
                    
                    # Call Manager
                    candidates, formulas, error = rag_manager.enrich_variable_taxonomy(
                        taxonomy, 
                        columns_info, 
                        distance=enrich_dist, 
                        progress_callback=update_prog
                    )
                    
                    prog_bar.empty()
                    
                    if error:
                        st.sidebar.error(f"Enrichment failed: {error}")
                    else:
                        if candidates or formulas:
                            st.session_state["enrichment_candidates"] = candidates
                            st.session_state["enrichment_formulas"] = formulas
                            st.sidebar.success(f"Found {len(candidates)} vars & {len(formulas)} formulas! Review in Overview tab.")
                            st.toast(f"Found {len(candidates)} new items!", icon="")
                        else:
                            st.sidebar.info("No new candidates found.")
        if st.button("New Taxonomy", help="Generate fresh taxonomy from dataset"):
            if not rag_active:
                 st.sidebar.error("RAG Required")
            elif "working_df" not in st.session_state:
                 st.sidebar.error("Load dataset first")
            else:
                 # Progress Bar Setup
                 prog_bar = st.progress(0, text="Initializing generation...")
                 def update_progress(p, text):
                     prog_bar.progress(p, text=text)

                 with st.spinner("Generating..."):
                    df = st.session_state["working_df"]
                    columns_info = [{"name": c} for c in df.columns]
                    current_renames = st.session_state.get("rename_dict", {})
                    
                    new_taxonomy, error = rag_manager.generate_variable_taxonomy(
                        columns_info, 
                        existing_mapping=current_renames,
                        deep_analysis=True,
                        progress_callback=update_progress
                    )
                    
                    prog_bar.empty()
                    
                    if error:
                        st.sidebar.error(f"Error: {error}")
                    else:
                        st.sidebar.success("Generated!")
                        st.session_state.rag_manager.variable_taxonomy = new_taxonomy
                        st.session_state.offline_taxonomy = new_taxonomy
                        # ALSO UPDATE FORMULAS
                        st.session_state.rag_manager.formulas_registry = st.session_state.rag_manager.formulas_registry 
                        st.session_state.offline_formulas = st.session_state.rag_manager.formulas_registry
                        st.session_state["loaded_path"] = None 
                        st.rerun()

    # --- 4. Main Content Logic ---
    
    # Case A: No Taxonomy (Empty State)
    if not taxonomy:
        st.info("No taxonomy loaded: load a saved version or generate a new one.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Load Existing")
            if os.path.exists("taxonomy_metadata.json"):
                 st.info("Legacy `taxonomy_metadata.json` found in root. Consider moving it to data/taxonomy.")
            
            # User Isolation: Get from session
            username = st.session_state.get('username')
            local_versions = get_taxonomy_versions(username)
            if local_versions:
                latest_v, latest_path = local_versions[0]
                st.success(f"Found {len(local_versions)} versions. Latest: v{latest_v}")
                
                v_opts = [f"v{v[0]}" for v in local_versions]
                sel_v = st.selectbox("Choose Version", v_opts, index=0)
                
                # Find path for selected version
                path_to_load = next(v[1] for v in local_versions if f"v{v[0]}" == sel_v)

                if st.button(f"Load {sel_v}", type="primary", width='stretch'):
                    # the same loader as the sidebar: taxonomy, formulas, link repair and AI-engine sync in one
                    # click (this button used to read the taxonomy only, so the formulas needed a second load)
                    success, info = load_and_repair_taxonomy(path_to_load, rag_manager if rag_active else None)
                    if success:
                        st.toast(f"Loaded {sel_v} with {info} formulas.")
                        st.rerun()
                    else:
                        st.error(f"Load failed: {info}")
            else:
                st.warning("No saved taxonomy version in your workspace yet.")

        with col2:
            st.subheader("Generate New")
            if not rag_active:
                st.warning("RAG System is not initialized.")
                st.markdown("You need to initialize the AI system to generate a new taxonomy from your data.")
            else:
                if "working_df" not in st.session_state:
                    no_dataset()
                else:
                    st.markdown("Analyze your dataset to generate a medical knowledge graph.")
                    if st.button("Generate Taxonomy", type="primary", width='stretch'):
                        # ... Generation Logic ...
                        progress_bar = st.progress(0, text="Starting taxonomy generation...")
                        def update_progress(percent, text):
                            progress_bar.progress(percent, text=text)
                        
                        df = st.session_state["working_df"]
                        columns_info = []
                        for col in df.columns:
                            info = {"name": col}
                            # Simple stats gathering
                            if pd.api.types.is_numeric_dtype(df[col]):
                                col_data = df[col].dropna()
                                if not col_data.empty:
                                    info["stats"] = {
                                        "min": float(col_data.min()),
                                        "max": float(col_data.max()),
                                        "mean": float(col_data.mean())
                                    }
                            else:
                                info["top_values"] = {str(k): v for k, v in df[col].value_counts().head(5).to_dict().items()}
                            columns_info.append(info)
                        
                        current_renames = st.session_state.get("rename_dict", {})
                        
                        new_taxonomy, error = rag_manager.generate_variable_taxonomy(
                            columns_info, 
                            existing_mapping=current_renames,
                            deep_analysis=True,
                            progress_callback=update_progress
                        )
                        progress_bar.empty()
                        
                        if error:
                            st.error(f"Error: {error}")
                        else:
                            st.success("Taxonomy generated successfully!")
                            st.session_state.rag_manager.variable_taxonomy = new_taxonomy
                            st.rerun()
        return

    # Case B: Taxonomy Exists (Render Dashboard)
    
    # --- Sidebar Filters & Search ---
    st.sidebar.header("Explore Data")
    search_query = st.sidebar.text_input("Search Variables", placeholder="Name...").lower()
    
    # Formula Search (Added here for consistency)
    f_query = st.sidebar.text_input("Search Formulas", placeholder="BMI, BSA...").lower()
    
    deep_search = st.sidebar.checkbox("Search in details (Description, Formulas)", value=False, help="Enable to search within variable descriptions and related formulas.")
    
    # Category Filter
    # SAFEGUARD: Filter out non-dict items to prevent 'list has no attribute get' error
    all_categories = sorted(list(set(v.get('category', 'Uncategorized') for v in taxonomy.values() if isinstance(v, dict))))
    selected_category = st.sidebar.selectbox("Filter by Category", ["All"] + all_categories)
    
    # Filter Logic
    filtered_vars = {}
    for var, data in taxonomy.items():
        if not isinstance(data, dict): continue
        # Category Match
        cat_match = (selected_category == "All" or data.get('category') == selected_category)
        
        # Search Match
        if deep_search:
             # Broad search
             search_text = f"{var} {data.get('standard_name', '')} {data.get('description', '')} {data.get('category', '')} {data.get('related_formulas', [])!s} {data.get('clinical_usage', '')}".lower()
        else:
             # Narrow search (Names only)
             search_text = f"{var} {data.get('standard_name', '')}".lower()
             
        text_match = search_query in search_text
        
        if cat_match and text_match:
            filtered_vars[var] = data
            
    st.sidebar.markdown(f"**Found {len(filtered_vars)} variables**")
    
    # Export Button
    st.sidebar.download_button(
        label="Export JSON",
        data=json.dumps(taxonomy, indent=2),
        file_name="variable_taxonomy.json",
        mime="application/json"
    )

    # --- Main Content Tabs ---
    tab_graph, tab_overview, tab_details, tab_formulas, tab_clustering, tab_refine = st.tabs(["Knowledge Graph", "Overview", "Variable Details", "Formulas Details", "Population Clustering", "Refinement"])
    
    # --- TAB 1: OVERVIEW ---
    with tab_overview:
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Variables", len(taxonomy))
        col2.metric("Standardized", len([v for v in taxonomy.values() if v.get('standard_name')]))
        col3.metric("With Formulas", len([v for v in taxonomy.values() if v.get('related_formula_ids') or v.get('related_formulas')]))
        
        st.divider()
        
        table_data = []
        for var, data in filtered_vars.items():
            # Robust Formula Count
            f_count = len(data.get('related_formula_ids', []))
            if f_count == 0:
                f_count = len(data.get('related_formulas', []))

            table_data.append({
                "Original Name": var,
                "Type": data.get('node_type', 'Variable').replace("-", " ").title(),
                "Standard Name": data.get('standard_name', 'N/A'),
                "Category": data.get('category', 'N/A'),
                "Formulas": f_count,
                "Context": "Documented" if data.get('clinical_usage') else "None"
            })
        
        st.dataframe(
            pd.DataFrame(table_data), 
            width='stretch', 
            column_config={
                "Original Name": st.column_config.TextColumn("Original Name", help="Name in the dataset"),
                "Type": st.column_config.TextColumn("Type", width="small"),
                "Standard Name": st.column_config.TextColumn("Standard Name", width="medium"),
                "Category": st.column_config.TextColumn("Category", width="small"),
                "Formulas": st.column_config.ProgressColumn("Formulas", min_value=0, max_value=5, format="%d"),
                "Context": st.column_config.TextColumn("Context", width="small"),
            },
            hide_index=True
        )

        # --- Enrichment Proposals Review ---
        candidates = st.session_state.get('enrichment_candidates', {})
        formulas = st.session_state.get('enrichment_formulas', {})

        if candidates or formulas:
            st.divider()
            st.markdown("### Enrichment Proposals")
            st.info("The AI has identified potential additions. Review and merge.")
            
            selected_vars = []
            
            if candidates:
                # Prepare DataFrame for Editor
                cand_list = []
                for k, v in candidates.items():
                    cand_list.append({
                        "Keep": True,
                        "Variable": k,
                        "Type": v.get('node_type', 'Input-External'),
                        "Description": v.get('description', ''),
                        "Formulas": str(v.get('relationships', [])),
                        "Category": v.get('category', 'Suggested')
                    })
                
                cand_df = pd.DataFrame(cand_list)
                
                edited_df = st.data_editor(
                    cand_df,
                    column_config={
                        "Keep": st.column_config.CheckboxColumn("Keep?", help="Select to add to graph"),
                        "Variable": st.column_config.TextColumn("Variable Name", disabled=True),
                    },
                    width='stretch',
                    hide_index=True
                )
                selected_vars = edited_df[edited_df["Keep"] == True]["Variable"].tolist()
            else:
                st.info("No new variables to add (all found concepts already exist or were unified).")

            if formulas:
                st.success(f"{len(formulas)} New Formulas identified connecting these variables.")
                with st.expander("View Formulas"):
                    st.json(formulas)
            
            c_merge, c_discard = st.columns([1, 4])
            
            if c_merge.button("Merge Selected", type="primary"):
                # Filter Selected (already done above if candidates exist)
                
                if not selected_vars:
                    st.warning("No variables selected.")
                else:
                    count = 0
                    # Use the currently active taxonomy as base
                    current_tax = taxonomy if taxonomy else {}
                    # Copy to avoid direct mutation issues before assignment
                    current_tax = current_tax.copy()
                    
                    for var in selected_vars:
                        if var in candidates:
                            current_tax[var] = candidates[var]
                            count += 1
                    
                    # Update State (Taxonomy)
                    st.session_state.offline_taxonomy = current_tax
                    if rag_manager:
                        rag_manager.variable_taxonomy = current_tax
                    
                    # Merge Formulas (Blind merge for now)
                    new_formulas = st.session_state.get("enrichment_formulas", {})
                    if new_formulas:
                         current_formulas = st.session_state.get("offline_formulas", {})
                         current_formulas.update(new_formulas)
                         st.session_state.offline_formulas = current_formulas
                         
                         # Ensure Outputs of newly added formulas are also merged into Taxonomy
                         # (Even if user didn't explicitly check them)
                         for f in new_formulas.values():
                             out_var = f.get('output_variable')
                             if out_var and out_var not in current_tax:
                                 if out_var in candidates:
                                      current_tax[out_var] = candidates[out_var]
                                      count += 1 # Count implicitly merged vars
                        
                    # State Update with implicit adds
                    st.session_state.offline_taxonomy = current_tax
                    if rag_manager:
                        rag_manager.variable_taxonomy = current_tax

                    # Clear Candidates
                    st.session_state.enrichment_candidates = {}
                    st.session_state["enrichment_formulas"] = {}
                    
                    st.toast(f"Merged {count} vars & {len(new_formulas)} formulas!", icon="")
                    st.rerun()
            
            if c_discard.button("Discard All"):
                st.session_state.enrichment_candidates = {}
                st.rerun()

    # --- TAB 2: VARIABLE DETAILS ---
    with tab_details:
        if not filtered_vars:
            st.warning("No variables found.")
        else:
            col_sel, col_content = st.columns([1, 3])
            with col_sel:
                selected_var_name = st.radio("Select Variable", sorted(filtered_vars.keys()), label_visibility="collapsed")
            
            with col_content:
                if selected_var_name:
                    var_data = filtered_vars[selected_var_name]
                    st.subheader(f"{var_data.get('standard_name', selected_var_name)}")
                    st.caption(f"Original Name: `{selected_var_name}`")
                    st.markdown(f"**Category:** `{var_data.get('category', 'Uncategorized')}`")
                    st.divider()
                    st.markdown("### Definition")
                    st.info(var_data.get('description', 'No description available.'))
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("### Related Formulas")
                        formula_ids = var_data.get('related_formula_ids', [])
                        legacy_formulas = var_data.get('related_formulas', [])
                        
                        if formula_ids and formulas_registry:
                            for fid in formula_ids:
                                f_data = formulas_registry.get(fid, {})
                                fname = f_data.get('name', fid)
                                fexpr = f_data.get('expression', '')
                                st.markdown(f"- **{fname}**")
                                if fexpr: st.caption(f"  `{fexpr}`")
                        elif legacy_formulas:
                            for f in legacy_formulas:
                                st.markdown(f"- {f}")
                        else:
                            st.caption("No known formulas.")
                    with c2:
                        st.markdown("### Clinical Usage")
                        usage = var_data.get('clinical_usage', "")
                        if usage:
                            st.markdown(usage)
                        else:
                            st.caption("No clinical context.")

    # --- TAB 3: KNOWLEDGE GRAPH ---
    with tab_graph:
        st.subheader("Interactive Clinical Knowledge Graph")

        graph_source = filtered_vars
        if len(graph_source) > 150:
            st.warning(f"Rendering {len(graph_source)} variables. The graph might be slow.")

        # Renderer-agnostic graph; the full taxonomy restores variables hidden by the filters
        spec = build_taxonomy_graph(
            graph_source,
            formulas_registry,
            full_taxonomy=taxonomy,
            formula_query=f_query,
            theme=current_theme(),
        )
        view = spec
        if spec.nodes:
            stats = spec.stats()
            c_f1, c_f2, c_f3 = st.columns(3)
            node_options = sorted(stats.nodes_by_category)
            sel_node_cats = c_f1.multiselect(
                "Node types", node_options, default=node_options,
                format_func=lambda c: CATEGORY_LABELS.get(c, c), key="kg_node_types",
            )
            edge_options = sorted(stats.edges_by_category)
            sel_edge_cats = c_f2.multiselect(
                "Edge types", edge_options, default=edge_options,
                format_func=lambda c: RELATION_LABELS.get(c, c), key="kg_edge_types",
            )
            renderer_selector(container=c_f3)
            view = spec.filter(node_categories=sel_node_cats, edge_categories=sel_edge_cats)

        if not view.nodes:
            st.info("No nodes to display based on current filters.")
        else:
            view_stats = view.stats()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Nodes", view_stats.node_count)
            m2.metric("Total Edges", view_stats.edge_count)
            m3.metric("Formulas", view_stats.nodes_by_category.get("formula", 0))
            m4.metric("Categories", view_stats.nodes_by_category.get("category", 0))

            with st.expander("Detailed Metrics", expanded=False):
                d1, d2 = st.columns(2)
                with d1:
                    st.write("**Nodes by Type**")
                    st.dataframe(
                        pd.DataFrame([{"Type": CATEGORY_LABELS.get(k, k), "Count": v}
                                      for k, v in view_stats.nodes_by_category.items()]),
                        width="stretch", hide_index=True,
                    )
                with d2:
                    st.write("**Edges by Type**")
                    st.dataframe(
                        pd.DataFrame([{"Type": RELATION_LABELS.get(k, k), "Count": v}
                                      for k, v in view_stats.edges_by_category.items()]),
                        width="stretch", hide_index=True,
                    )

            selection = render_graph(
                view,
                RenderOptions(key="main_kg_view", height=750, selectable=True, layout=LayoutHint.HIERARCHICAL),
            )
            render_graph_exports(view, RenderOptions(key="main_kg_view"), file_stem="knowledge_graph")

            # 6. Interaction panel, driven by the renderer's selection
            if selection.node_ids:
                st.divider()
                st.markdown(f"### Selection Details ({len(selection.node_ids)} items)")

                for sel_id in selection.node_ids:
                    node = view.node(sel_id)
                    if node is None:
                        continue
                    with st.expander(f"Item: {node.label}", expanded=True):
                        if sel_id in taxonomy:
                            data = taxonomy[sel_id]
                            c1, c2 = st.columns(2)
                            with c1:
                                st.write(f"**Standard Name:** {data.get('standard_name')}")
                                st.write(f"**Category:** {data.get('category')}")
                                st.info(data.get('description') or "No description.")
                            with c2:
                                st.write("**Clinical Usage:**")
                                st.caption(data.get('clinical_usage', 'N/A'))
                                if data.get('related_formulas'):
                                    st.write("**Formulas:**")
                                    for f in data['related_formulas']:
                                        st.code(f, language="text")
                        elif node.category == "category":
                            st.write("**Type:** Category node")
                            st.info(f"Groups all variables belonging to '{node.label}' category.")
                        elif node.category == "formula":
                            st.markdown("#### Formula Details")
                            registry = formulas_registry or {}
                            f_key = node.data.get("formula_key") or parse_formula_id(sel_id) or sel_id
                            f_data = (
                                registry.get(f_key)
                                or registry.get(make_formula_id(f_key))
                                or registry.get(parse_formula_id(f_key) or "")
                            )
                            if f_data:
                                st.markdown("#### Formula Content")
                                if f_data.get('markdown_formula'):
                                    st.latex(f_data['markdown_formula'])
                                elif f_data.get('expression'):
                                    st.code(f_data['expression'], language="text")
                                elif f_data.get('formula'):
                                    st.code(f_data['formula'], language="python")
                                else:
                                    st.caption("No mathematical expression available.")

                                st.success(f"**Description:** {f_data.get('description', 'No description.')}")

                                c_in, c_out = st.columns(2)
                                with c_in:
                                    st.markdown("**Inputs**")
                                    for inv in f_data.get('input_variables', []):
                                        st.markdown(f"- `{inv}`")
                                with c_out:
                                    st.markdown("**Derived Variable**")
                                    st.markdown(f"`{f_data.get('output_variable', 'Unknown')}`")
                            else:
                                st.info(f"Logic: {node.tooltip or 'No details'}")
                        else:
                            st.write(f"**Type:** {CATEGORY_LABELS.get(node.category, node.category)}")
                            st.write(f"**Name:** {node.label}")
                            if node.tooltip:
                                st.info(node.tooltip)

    # --- TAB 3: FORMULA DETAILS (NEW) ---
    with tab_formulas:
        st.subheader("Formula Taxonomy")
        
        # Filter Formulas (Query comes from Sidebar)
        # f_query is defined in sidebar section above
        
        # Filter logic
        filtered_forms = []
        for fid, fdata in formulas_registry.items():
            fname = fdata.get('name', '').lower()
            fdesc = fdata.get('description', '').lower()
            if f_query in fname or f_query in fdesc or f_query in fid.lower():
                filtered_forms.append((fid, fdata))
        
        f_col1, f_col2 = st.columns([1, 2])
        
        with f_col1:
            if not filtered_forms:
                st.warning("No formulas found.")
                if not formulas_registry:
                    st.info("Hint: Try reloading the Taxonomy (vX) from the sidebar to fetch definitions.")
                sel_fdata = None
            else:
                st.markdown(f"**Found {len(filtered_forms)} formulas**")
                # Create readable labels
                f_labels = [f"{f[1].get('name', 'Unknown')} ({f[0]})" for f in filtered_forms]
                
                sel_f_idx = st.radio("Select Formula", range(len(filtered_forms)), format_func=lambda x: f_labels[x], label_visibility="collapsed")
                sel_fid, sel_fdata = filtered_forms[sel_f_idx]
        
        with f_col2:
            if sel_fdata:
                st.markdown(f"### {sel_fdata.get('name', 'Unnamed Formula')}")
                st.caption(f"ID: `{sel_fid}`")
                st.divider()
                
                # Math Display
                st.markdown("#### Formula Content")
                if sel_fdata.get('markdown_formula'):
                    st.latex(sel_fdata['markdown_formula'])
                elif sel_fdata.get('expression'):
                    st.code(sel_fdata['expression'], language="text")
                elif sel_fdata.get('formula'):
                    st.code(sel_fdata['formula'], language="python")
                else:
                    st.caption("No mathematical expression available.")
                
                st.success(f"**Description:** {sel_fdata.get('description', 'No description.')}")
                
                c_in, c_out = st.columns(2)
                with c_in:
                    st.markdown("**Inputs**")
                    for inv in sel_fdata.get('input_variables', []):
                        st.markdown(f"- `{inv}`")
                with c_out:
                    st.markdown("**Derived Variable**")
                    st.markdown(f"`{sel_fdata.get('output_variable', 'Unknown')}`")
                
                if sel_fdata.get('references'):
                    st.markdown("#### References")
                    for ref in sel_fdata['references']:
                        st.markdown(f"- {ref}")
            else:
                st.info("Select a formula from the left panel to view its details.")
    
    # --- TAB 5: POPULATION CLUSTERING ---
    with tab_clustering:
        st.subheader("Population Clustering & Dimensionality Reduction")
        
        # Educational Introduction
        with st.expander("What is Clustering Analysis?", expanded=False):
            st.markdown("""
            ### Understanding Clustering in Epidemiology
            
            **Clustering** identifies natural subgroups (phenotypes) in your population based on 
            similarities across multiple variables.
            
            ---
            
            #### Clustering Algorithms
            
            | Method | How it Works | Best For | Output |
            |:-------|:-------------|:---------|:-------|
            | **K-Means** | Minimizes distance to K centroids | Well-separated, spherical clusters | Hard assignment (1 cluster per sample) |
            | **DBSCAN** | Groups points in dense regions | Arbitrary shapes, **outlier detection** | Hard + Noise label (-1) |
            | **Gaussian Mixture** | Fits K Gaussian distributions | Overlapping clusters, probabilistic | **Soft assignment** (probability per cluster) |
            
            ##### Parameter Guide
            
            | Algorithm | Parameter | Meaning | Typical Values |
            |:----------|:----------|:--------|:---------------|
            | **K-Means** | K | Number of clusters to find | 2-10 (use Elbow/Silhouette) |
            | **DBSCAN** | eps | Max distance for neighbors | 0.3-1.0 (depends on data scale) |
            | **DBSCAN** | min_samples | Min points to form cluster | 5-10 |
            | **Gaussian Mixture** | n_components | Number of Gaussians = clusters | 2-10 (use BIC/AIC) |
            
            > **Note:** In GMM, "components" = clusters. Each Gaussian distribution represents one subpopulation.
            > GMM gives **soft assignments**: each sample has a probability of belonging to each cluster.
            
            ---
            
            #### Dimensionality Reduction Methods
            
            | Method | What it Preserves | Speed | When to Use |
            |:-------|:------------------|:------|:------------|
            | **PCA** | Global structure & variance | Very fast | First exploration, interpretable axes |
            | **t-SNE** | Local neighborhoods | Slow | Visualizing clusters (< 3000 samples) |
            | **UMAP** | Local + some global | Fast | Large datasets, better than t-SNE |
            
            ##### Key Differences
            
            - **PCA**: Linear projection. Axes have meaning (loadings). Distances are meaningful.
            - **t-SNE**: Non-linear. Great for cluster visualization. **Distances BETWEEN clusters are meaningless!**
            - **UMAP**: Non-linear like t-SNE but preserves more global structure. Generally preferred.
            
            ---
            
            #### Important Caveats
            
            1. **Standardization**: Variables are auto-standardized (mean=0, SD=1) before analysis
            2. **Missing values**: Rows with any missing value are dropped
            3. **Interpretation**: Clusters are exploratory, validate with clinical knowledge
            4. **Overfitting**: More clusters ≠ better. Aim for 3-5 interpretable groups
            """)
        
        # Check if data is loaded
        if "working_df" not in st.session_state:
            st.warning("Please load a dataset from the Main View page first.")
        else:
            df = st.session_state["working_df"]
            
            # Identify column types
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.error("Need at least 2 numeric variables for clustering.")
            else:
                # Configuration Panel
                st.markdown("### Configuration")
                
                with st.expander("Analysis Configuration", expanded=True):
                    col_cfg1, col_cfg2 = st.columns(2)
                    
                    with col_cfg1:
                        # Combine lists for selection
                        all_candidates = numeric_cols + categorical_cols
                        selected_vars = st.multiselect(
                            "Select Analysis Variables",
                            options=all_candidates,
                            default=numeric_cols[:min(3, len(numeric_cols))],
                            help="Choose numeric and categorical variables to include."
                        )
                        st.caption(f"Selected: {len(selected_vars)} variables")
                        
                        dim_method = st.selectbox(
                            "Dimensionality Reduction",
                            available_methods(["PCA", "FAMD", "t-SNE", "UMAP"])[0],
                            help="Method to project data to 2D for visualization."
                        )
                        
                    with col_cfg2:
                        cluster_method = st.selectbox(
                            "Clustering Algorithm",
                            ["K-Means", "DBSCAN", "Gaussian Mixture"],
                            help="Algorithm to identify subgroups."
                        )
                        
                        # Method-specific parameters
                        if cluster_method == "K-Means":
                            n_clusters = st.slider("Number of Clusters (K)", 2, 10, 3)
                        elif cluster_method == "DBSCAN":
                            eps = st.slider("Epsilon (neighborhood size)", 0.1, 2.0, 0.5, 0.1)
                            min_samples = st.slider("Min Samples", 2, 20, 5)
                        else:  # GMM
                            n_clusters = st.slider("Number of Components", 2, 10, 3)
                        
                        color_by = st.selectbox(
                            "Color by (optional)",
                            ["Cluster"] + categorical_cols[:10],
                            help="Color points by cluster or existing group variable."
                        )
                
                if len(selected_vars) < 2:
                    st.warning("Please select at least 2 variables (can be mixed numeric/categorical).")
                else:
                    # Run Analysis Button
                    if st.button("Run Clustering Analysis", type="primary", width='content'):
                        with st.spinner("Running analysis..."):
                            try:
                                # Prepare dataify variable types in selection
                                sel_numeric = [v for v in selected_vars if v in numeric_cols]
                                sel_categorical = [v for v in selected_vars if v in categorical_cols]
                                
                                # Prepare data
                                prep_data, valid_idx = prepare_data_for_clustering(
                                    df, 
                                    numeric_cols=sel_numeric, 
                                    categorical_cols=sel_categorical,
                                    handle_missing="drop"
                                )
                                
                                if len(prep_data) < 10:
                                    st.error("Not enough valid samples after removing missing values.")
                                else:
                                    # Run dimensionality reduction
                                    if dim_method == "PCA":
                                        embeddings, dim_info = run_pca(prep_data, n_components=2)
                                    elif dim_method == "t-SNE":
                                        embeddings, dim_info = run_tsne(prep_data, n_components=2, max_samples=3000)
                                    elif dim_method == "UMAP":
                                        try:
                                            embeddings, dim_info = run_umap(prep_data, n_components=2, max_samples=5000)
                                        except ImportError:
                                            st.error("UMAP not installed. Run: `pip install umap-learn`")
                                            embeddings, dim_info = run_pca(prep_data, n_components=2)
                                            dim_method = "PCA (fallback)"
                                    elif dim_method == "FAMD":
                                        try:
                                            # Use the raw selected data (imputed) for FAMD if available, or just prep_data
                                            # prep_data is already numeric+imputed if standardized?
                                            # wait, prepare_data_for_clustering standardizes numeric columns.
                                            # FAMD usually needs mix. 
                                            # For now, let's pass prep_data which might be mixed if we updated prepare_data...
                                            # Checking prepare_data_for_clustering -> it does standardization on numeric.
                                            # It keeps categorical columns if passed.
                                            # So we can pass prep_data directly.
                                            embeddings, dim_info = run_famd(prep_data, n_components=2)
                                        except ImportError:
                                            st.error("Prince not installed. Run: `pip install prince`")
                                            embeddings, dim_info = run_pca(prep_data, n_components=2)
                                            dim_method = "PCA (fallback)"
                                        except Exception as e:
                                            st.error(f"FAMD Error: {e}")
                                            embeddings, dim_info = run_pca(prep_data, n_components=2)
                                            dim_method = "PCA (fallback)"
                                    
                                    # Run clustering
                                    if cluster_method == "K-Means":
                                        labels, clust_info = fit_kmeans(embeddings, n_clusters)
                                    elif cluster_method == "DBSCAN":
                                        labels, clust_info = fit_dbscan(embeddings, eps, min_samples)
                                    else:  # GMM
                                        labels, clust_info = fit_gaussian_mixture(embeddings, n_clusters)
                                    
                                    # Store results in session state
                                    st.session_state['clustering_results'] = {
                                        'embeddings': embeddings,
                                        'labels': labels,
                                        'valid_idx': valid_idx,
                                        'dim_info': dim_info,
                                        'clust_info': clust_info,
                                        'dim_method': dim_method,
                                        'cluster_method': cluster_method,
                                        'selected_vars': selected_vars
                                    }
                                    st.success(f"Analysis complete! Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters.")
                            
                            except Exception as e:
                                st.error(f"Error: {e!s}")
                    
                    # Display results if available
                    if 'clustering_results' in st.session_state:
                        results = st.session_state['clustering_results']
                        embeddings = results['embeddings']
                        labels = results['labels']
                        valid_idx = results['valid_idx']
                        
                        st.markdown("---")
                        st.markdown("### Results")
                        
                        # Visualization tabs
                        viz_tabs = st.tabs(["Scatter Plot", "Cluster Profiles", "Optimal K"])
                        
                        with viz_tabs[0]:
                            import plotly.express as px
                            
                            # Prepare plot data
                            plot_df = pd.DataFrame({
                                'Dim 1': embeddings[:, 0],
                                'Dim 2': embeddings[:, 1],
                                'Cluster': [f"Cluster {l}" if l >= 0 else "Noise" for l in labels]
                            })
                            
                            # Add color_by variable if not cluster
                            if color_by != "Cluster" and color_by in df.columns:
                                plot_df['Color'] = df.loc[valid_idx, color_by].values
                                color_col = 'Color'
                            else:
                                color_col = 'Cluster'
                            
                            
                            # Warning for dropped columns
                            used_features = results['dim_info'].get('feature_names', [])
                            if used_features:
                                dropped = list(set(selected_vars) - set(used_features))
                                if dropped:
                                    st.info(f"Note: Columns excluded from projection: {', '.join(dropped)}")

                            # Axis labels
                            dim_info = results['dim_info']
                            axis_labels = {"x": "Dim 1", "y": "Dim 2"}
                            
                            if results['dim_method'] == 'PCA':
                                ratio = dim_info.get('explained_variance_ratio', [0, 0])
                                if len(ratio) < 2: ratio = ratio + [0]*(2-len(ratio))
                                axis_labels = {"x": f"PC1 ({ratio[0]:.1%})", "y": f"PC2 ({ratio[1]:.1%})"}
                            
                            elif results['dim_method'] == 'FAMD':
                                inertia = dim_info.get('explained_inertia', [0, 0])
                                if len(inertia) < 2: inertia = inertia + [0]*(2-len(inertia))
                                axis_labels = {"x": f"Dim 1 ({inertia[0]:.1f}%)", "y": f"Dim 2 ({inertia[1]:.1f}%)"}

                            fig = px.scatter(
                                plot_df,
                                x='Dim 1',
                                y='Dim 2',
                                color=color_col,
                                title=f"{results['dim_method']} Projection with {results['cluster_method']} Clustering",
                                template="plotly_white",
                                labels={'Dim 1': axis_labels['x'], 'Dim 2': axis_labels['y']},
                                hover_data={'Dim 1': False, 'Dim 2': False, 'Cluster': True}
                            )
                            fig.update_layout(height=500)
                            fig.update_traces(marker=dict(size=8, opacity=0.7))
                            st.plotly_chart(fig, width="stretch")
                            
                            # Interpretation
                            with st.expander("How to Interpret"):
                                st.markdown("""
                                **Reading the Plot:**
                                - Each point = one sample
                                - Nearby points = similar profiles
                                - Colors = cluster assignments or group
                                - Well-separated clusters = distinct subpopulations
                                
                                **Caveats:**
                                - 2D projection may distort distances
                                - t-SNE/UMAP: distances between clusters are NOT meaningful
                                - Always validate clusters with clinical knowledge
                                """)
                        
                            with viz_tabs[1]:
                                st.markdown("##### Cluster Profiles")
                                st.caption("Mean values (numeric) and Mode (categorical) for each cluster.")
                                
                                # Access selections from session state or logic above (need to be robust)
                                # We didn't store sel_numeric/sel_categorical in session state, 
                                # but we can re-derive them from results['selected_vars']
                                all_selected = results['selected_vars']
                                current_numeric = [c for c in all_selected if c in numeric_cols]
                                current_categorical = [c for c in all_selected if c in categorical_cols]
                                
                                profiles = compute_cluster_profiles(
                                    df.loc[valid_idx], labels, 
                                    numeric_cols=current_numeric,
                                    categorical_cols=current_categorical
                                )
                                st.dataframe(profiles, width="stretch")
                            
                                # Download
                                st.download_button(
                                    "Download Profiles",
                                    data=profiles.to_csv(),
                                    file_name="cluster_profiles.csv",
                                    mime="text/csv"
                                )
                                
                                with st.expander("How to Interpret"):
                                    st.markdown("""
                                    **Reading Cluster Profiles:**
                                    - Each row = one cluster
                                    - Values = mean of each variable in that cluster
                                    - Compare across clusters to characterize phenotypes
                                    
                                    **Example Interpretation:**
                                    - "Cluster 0 has high BMI, high glucose → metabolic phenotype"
                                    - "Cluster 2 has low all biomarkers → healthy controls"
                                    """)
                        
                        with viz_tabs[2]:
                            import plotly.graph_objects as go

                            from utils.clustering_utils import optimal_k_analysis
                            
                            # Only for K-Means/GMM
                            if results['cluster_method'] != "DBSCAN":
                                with st.spinner("Computing optimal K..."):
                                    opt_analysis = optimal_k_analysis(embeddings, k_range=range(2, 11))
                                
                                fig = go.Figure()
                                
                                # Elbow plot
                                fig.add_trace(go.Scatter(
                                    x=opt_analysis['k_values'],
                                    y=opt_analysis['inertias'],
                                    mode='lines+markers',
                                    name='Inertia (Elbow)',
                                    yaxis='y1'
                                ))
                                
                                # Silhouette
                                fig.add_trace(go.Scatter(
                                    x=opt_analysis['k_values'],
                                    y=opt_analysis['silhouette_scores'],
                                    mode='lines+markers',
                                    name='Silhouette Score',
                                    yaxis='y2',
                                    line=dict(color='green')
                                ))
                                
                                fig.update_layout(
                                    title="Optimal Number of Clusters",
                                    xaxis_title="K (Number of Clusters)",
                                    yaxis=dict(title="Inertia", side="left"),
                                    yaxis2=dict(title="Silhouette", side="right", overlaying="y"),
                                    template="plotly_white"
                                )
                                st.plotly_chart(fig, width='content')
                                
                                st.info(f"**Recommended K (by Silhouette):** {opt_analysis['optimal_k_silhouette']}")
                                
                                with st.expander("How to Choose K"):
                                    st.markdown("""
                                    **Elbow Method:** Look for "bend" in the curve
                                    - Steep drop → adding clusters helps
                                    - Flat → diminishing returns
                                    
                                    **Silhouette Score:** Higher is better (max = 1)
                                    - > 0.5 = good clustering
                                    - 0.25-0.5 = acceptable
                                    - < 0.25 = poor separation
                                    
                                    **Practical Advice:**
                                    - Consider clinical interpretability
                                    - More clusters ≠ better
                                    - 3-5 clusters often sufficient
                                    """)
                            else:
                                st.info("DBSCAN automatically determines the number of clusters based on density.")
                                st.markdown(f"""
                                **DBSCAN Results:**
                                - Clusters found: **{results['clust_info']['n_clusters']}**
                                - Noise points: **{results['clust_info']['n_noise']}** ({results['clust_info']['noise_ratio']:.1%})
                                
                                Adjust `eps` and `min_samples` to tune clustering.
                                """)
            
    # --- TAB 6: REFINEMENT ---
    with tab_refine:
        st.subheader("Taxonomy Refinement")
        
        if not rag_active:
            st.warning("AI System Required")
            st.info("Refining the taxonomy requires the RAG system to be initialized.")
        else:
            # --- Refine Existing Variable ---
            with st.expander("Refine Existing Variable (AI)", expanded=False):
                st.caption("Select variables to refine, provide natural language feedback, and let the AI update the taxonomy.")
                
                vars_to_refine = st.multiselect(
                    "Select variables:",
                    options=sorted(filtered_vars.keys()),
                    format_func=lambda x: f"{x} ({filtered_vars[x].get('standard_name', 'Unknown')})"
                )

                if vars_to_refine:
                    feedback_dict = {}
                    st.write("### Provide Feedback")
                    for var in vars_to_refine:
                        current_desc = filtered_vars[var].get('description', 'No description')
                        st.markdown(f"**{var}**: `{current_desc}`")
                        feedback = st.text_area(f"Feedback for {var}:", key=f"feed_{var}", placeholder="e.g., 'This definition is incorrect...'")
                        if feedback:
                            feedback_dict[var] = feedback
                        st.divider()
                    
                    if st.button("Refine Selected Variables", type="primary"):
                        if not feedback_dict:
                            st.warning("Please provide feedback for at least one variable.")
                        else:
                            with st.spinner("Refining taxonomy..."):
                                # Get columns info context
                                columns_info = []
                                if "working_df" in st.session_state:
                                    df = st.session_state["working_df"]
                                    for col in df.columns:
                                        info = {"name": col}
                                        if pd.api.types.is_numeric_dtype(df[col]):
                                            info["stats"] = {"min": float(df[col].min()), "max": float(df[col].max()), "mean": float(df[col].mean())}
                                        columns_info.append(info)
                                
                                updated_taxonomy, error = rag_manager.refine_variable_taxonomy(
                                    taxonomy, 
                                    feedback_dict, 
                                    columns_info
                                )
                                
                                if error:
                                    st.error(f"Refinement failed: {error}")
                                else:
                                    st.success("Taxonomy updated successfully!")
                                    st.session_state.rag_manager.variable_taxonomy = updated_taxonomy
                                    st.rerun()

            # --- NEW: Add Missing Concept Node ---
            st.divider()
            with st.expander("Add Missing Concept Node (AI)", expanded=False):
                st.caption("Use AI to find and add specific derived concepts (e.g. medical scores, ratios) to the taxonomy.")
                
                
                # --- NEW LAYOUT: Align Search/Suggestions and Clues/Category ---
                c_top_1, c_top_2 = st.columns([3, 1])
                with c_top_1:
                    refine_search_hint = st.text_input("Concept Name / Medical Score:", placeholder="e.g. HOMA-IR, MAP, BMI", key="refine_search_hint")
                with c_top_2:
                    refine_num_sugg = st.number_input("Suggestions", min_value=1, max_value=5, value=3, key="refine_num_sugg")
                
                # Get existing categories safely
                ex_cats = sorted(list(set(v.get('category', 'Uncategorized') for v in taxonomy.values() if isinstance(v, dict)))) if taxonomy else []

                c_mid_1, c_mid_2 = st.columns([3, 1])
                with c_mid_1:
                     # Involved Variables Clue
                    refine_clues = st.multiselect(
                        "Involved Variables (Clue):",
                        options=sorted(taxonomy.keys()) if taxonomy else [],
                        format_func=lambda x: f"{x} ({taxonomy[x].get('standard_name', '')})" if taxonomy and x in taxonomy else x,
                        help="Limit the search to specific variables to guide the AI."
                    )
                with c_mid_2:
                    # Target Category Selection
                    refine_target_category = st.selectbox("Target Category (Optional):", ["Auto (AI)"] + ex_cats, key="refine_target_category")

                if st.button("Generate Concept Suggestions", type="primary"):
                    if not refine_search_hint:
                        st.warning("Please enter a concept name.")
                    else:
                        # Create a container for progress details
                        status_container = st.status("AI Agent Working...", expanded=True)
                        p_bar = status_container.progress(0, text="Initializing...")
                        
                        def update_progress(p, msg):
                            # Ensure p is 0-100 (int) or 0.0-1.0 (float) -> Progress expects 0.0-1.0 or 0-100
                            # Our backend sends 0-100 integers usually
                            p_bar.progress(p, text=msg)
                            # Update status label too for history
                            status_container.write(f"checking: {msg}")

                        # Get columns context
                        # Default: All columns in dataframe or taxonomy
                        cols_context = list(taxonomy.keys()) if taxonomy else []
                        if "working_df" in st.session_state:
                            cols_context = list(st.session_state["working_df"].columns)
                        
                        # Apply Clue Filter (Restriction)
                        if refine_clues:
                            cols_context = refine_clues
                        
                        # Store for later use in "Add to Graph"
                        st.session_state['refine_active_clues'] = refine_clues if refine_clues else []
                        st.session_state['refine_target_cat'] = refine_target_category

                        # Re-use the enrichment logic
                        sugg_text, err = rag_manager.suggest_computed_variables_with_validation(
                            cols_context,
                            search_hint=refine_search_hint,
                            num_suggestions=refine_num_sugg,
                            suggestion_mode="Go To Target",
                            use_taxonomy=True,
                            existing_categories=ex_cats,  # Pass existing categories
                            progress_callback=update_progress
                        )
                        
                        status_container.update(label="Generation Complete!", state="complete", expanded=False)
                            
                        if err:
                            st.error(err)
                        else:
                            st.session_state['refine_suggestions_text'] = sugg_text

                # Display Results
                if 'refine_suggestions_text' in st.session_state:
                    s_text = st.session_state['refine_suggestions_text']
                    
                    # Parse JSON safely
                    # Parse JSON safely using robust utility
                    from utils.llm_utils import parse_json_safe
                    s_data = parse_json_safe(s_text, default={"suggestions": []})
                        
                    # Normalize to list
                    if isinstance(s_data, dict):
                        s_list = s_data.get('suggestions', [])
                    else:
                        s_list = s_data if isinstance(s_data, list) else []
                        
                    if not s_list:
                        st.info("No suggestions returned.")
                    else:
                        st.markdown("### Suggested Concepts")
                        
                        for idx, item in enumerate(s_list):
                            with st.container(border=True):
                                c1, c2 = st.columns([3, 1])
                                with c1:
                                    st.markdown(f"**{item.get('name', 'Unknown')}**")
                                    st.caption(item.get('description', ''))
                                    # Formula
                                    f_code = item.get('formula', 'N/A')
                                    st.code(f_code, language='python')
                                    # Inputs
                                    ins = item.get('input_variables', [])
                                    # Display raw inputs from AI + Clues if any
                                    active_clues = st.session_state.get('refine_active_clues', [])
                                    # Merge for display
                                    display_inputs = list(set(ins + active_clues))
                                    
                                    if display_inputs:
                                        st.write(f"**Inputs:** {', '.join(display_inputs)}")

                                    # Category
                                    target_cat_pref = st.session_state.get('refine_target_cat', 'Auto (AI)')
                                    if target_cat_pref != 'Auto (AI)':
                                        disp_cat = target_cat_pref
                                        cat_source = "(User)"
                                    else:
                                        disp_cat = item.get('suggestion_category', 'Generated')
                                        cat_source = "(AI)"
                                    
                                    st.write(f"**Category:** {disp_cat} {cat_source}")
                                
                                with c2:
                                    if st.button("Add to Graph", key=f"add_refine_{idx}"):
                                        # Logic to Add Node and Formula
                                        
                                        # 1. Define IDs
                                        new_var_id = item.get('name')
                                        # Create a simpler ID for formula to avoid collisions? 
                                        # Use hash or timestamp, or just FORM_VARNAME
                                        new_form_id = make_formula_id(new_var_id)
                                        
                                        # MERGE CLUES AS INPUTS
                                        final_inputs = list(set(item.get('input_variables', []) + st.session_state.get('refine_active_clues', [])))
                                        
                                        # Determine Category
                                        target_cat_pref = st.session_state.get('refine_target_cat', 'Auto (AI)')
                                        if target_cat_pref != 'Auto (AI)':
                                            final_category = target_cat_pref
                                        else:
                                            final_category = item.get('suggestion_category', 'Generated')

                                        # 2. Update Taxonomy (Variable Node)
                                        # We tag it as 'External-Derived' as requested
                                        new_var_node = {
                                            "node_type": "External-Derived",
                                            "category": final_category,

                                            "description": item.get('description', ''),
                                            "standard_name": item.get('title', new_var_id),
                                            "related_formula_ids": [new_form_id],
                                            "source_type": "AI-Refinement",
                                            "clinical_usage": item.get('clinical_usage', '')
                                        }
                                        
                                        # 3. Update Formulas Registry (Formula Node)
                                        # Use title for the name if available, otherwise cleaner ID
                                        formula_display_name = item.get('title', new_var_id)
                                        
                                        new_form_node = {
                                            "node_type": "Formula",
                                            "id": new_form_id, # explicit ID
                                            "name": formula_display_name,
                                            "description": f"Formula for {new_var_id}",
                                            "output_variable": new_var_id,
                                            "input_variables": final_inputs,
                                            "expression": f_code,
                                            "formula": f_code, # duplicate for safety
                                            "markdown_formula": item.get('markdown_formula', '')
                                        }
                                        
                                        # 4. Commit to Session State
                                        if st.session_state.offline_taxonomy is None:
                                            st.session_state.offline_taxonomy = {}
                                        st.session_state.offline_taxonomy[new_var_id] = new_var_node
                                        st.session_state.offline_formulas[new_form_id] = new_form_node
                                        
                                        # 5. Link Inputs (Update their related_formula_ids)
                                        # This ensures that looking at an Input variable shows it contributes to this Formula
                                        for inp in final_inputs:

                                            if inp in st.session_state.offline_taxonomy:
                                                # Get node
                                                inp_node = st.session_state.offline_taxonomy[inp]
                                                # Init list if missing
                                                if 'related_formula_ids' not in inp_node:
                                                    inp_node['related_formula_ids'] = []
                                                # Add if not present
                                                if new_form_id not in inp_node['related_formula_ids']:
                                                    inp_node['related_formula_ids'].append(new_form_id)
                                        
                                        if rag_manager:
                                            rag_manager.variable_taxonomy = st.session_state.offline_taxonomy
                                        
                                        st.toast(f"Added {new_var_id} to taxonomy!", icon="")

                                        # Clear suggestions to reset state?
                                        del st.session_state['refine_suggestions_text']
                                        st.rerun()


            


if __name__ == "__main__":
    app()
