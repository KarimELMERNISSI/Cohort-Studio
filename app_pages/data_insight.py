import streamlit as st
import pandas as pd
import json
import os
import re
import networkx as nx
import glob
import time

TAXONOMY_DIR = os.path.join("data", "taxonomy")

def get_taxonomy_versions():
    """Returns list of (version_int, filepath) sorted descending."""
    if not os.path.exists(TAXONOMY_DIR):
        return []
    
    # Scan for directories starting with 'v'
    versions = []
    for item in os.listdir(TAXONOMY_DIR):
        full_path = os.path.join(TAXONOMY_DIR, item)
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
from yfiles_graphs_for_streamlit import StreamlitGraphWidget, Node, Edge, EdgeStyle, DashStyle, Layout, LabelStyle, NodeStyle, NodeShape

# ==========================================
# 1. GRAPH HELPER FUNCTIONS
# ==========================================

def get_graph_data(taxonomy_data, formulas_registry=None, full_taxonomy_ref=None):
    nodes = []
    edges = []
    existing_node_ids = set()
    
    TYPE_CONFIG = {
        "Input":    {"role": "input"},
        "Derived":  {"role": "derived"},
        "Outcome":  {"role": "outcome"},
        "Formula":  {"role": "formula"},
        "Category": {"role": "category"},
        "Input-External":   {"role": "input-external"},
        "Derived-External": {"role": "derived-external"},
        # Explicit Internal Types
        "Input-Internal":   {"role": "input"},
        "Derived-Internal": {"role": "derived"}
    }

    def add_node_safe(n_id, label, n_type, description=""):
        # Ensure IDs are robust strings
        safe_id = str(n_id)
        if safe_id not in existing_node_ids:
            role = TYPE_CONFIG.get(n_type, {}).get("role", "input")
            
            nodes.append(Node(
                id=safe_id,
                properties={
                    "label": str(label),
                    "type": n_type,
                    "role": role,
                    "tooltip": description
                }
            ))
            existing_node_ids.add(safe_id)

    # --- Main Parsing Loop ---
    # 1. Variables & Categories
    for var_id, attributes in taxonomy_data.items():
        var_id_str = str(var_id)
        
        # Variable Node
        node_type = attributes.get("node_type", "Input")
        # Remap Input -> Input-Internal
        if node_type == "Input":
             node_type = "Input-Internal"
             
        label = attributes.get("standard_name", var_id_str)
        desc = attributes.get("description", "")
        
        add_node_safe(var_id_str, label, node_type, desc)

        # Category Node & Edge
        category = attributes.get("category", "Uncategorized")
        cat_id = f"CAT_{category}"
        add_node_safe(cat_id, category, "Category", "Data Category")
        
        # Edge: Category -> Variable (Parent -> Child for layout)
        edges.append(Edge(
            start=cat_id, 
            end=var_id_str, 
            properties={
                "label": "", 
                "style": "dashed", 
                "color": "#BDBDBD", 
                "directed": False 
            }
        ))

    # 2. Formula Logic (Structured vs Legacy)
    if formulas_registry:
        # --- Pre-compute Lookups for Robust Matching ---
        std_to_orig = {}
        desc_to_orig = {} 
        
        for k, v in taxonomy_data.items():
            # Standard Name Lookup
            s_name = str(v.get('standard_name', '')).lower()
            if s_name: std_to_orig[s_name] = k
            
            # Description Lookup (First 50 chars as a heuristic key? or just contain?)
            # Let's use exact description match for now to be safe, or significant words?
            # User said "description for context". 
            # Let's try matching if the input var name is IN the description? No, risky.
            # Let's just stick to Standard Name and maybe "Original Name" if stored.
            pass

        # --- NEW STRUCTURED PATH ---
        for f_id, f_data in formulas_registry.items():
            f_name = f_data.get('name', f_id)
            f_desc = f_data.get('description', '')
            formula_node_id = f"FORM_{f_id}" # Namespace it
            
            # Helper to resolve ID
            def resolve_id(raw_name):
                r = str(raw_name).strip()
                r_lower = r.lower()
                
                # 1. Check if ID exists directly
                if r in existing_node_ids: return r
                if r in taxonomy_data: return r
                if full_taxonomy_ref and r in full_taxonomy_ref: return r
                
                # 2. Check Standard Name (Exact & Case-insensitive)
                if r_lower in std_to_orig: return std_to_orig[r_lower]
                
                # Check Standard Name in FULL taxonomy if available
                if full_taxonomy_ref:
                     for k, v in full_taxonomy_ref.items():
                         if str(v.get('standard_name', '')).lower() == r_lower:
                             return k

                # 3. Check if 'r' is a substring of any Standard Name (Fuzzy)
                # (Only if r is long enough to be significant)
                if len(r) > 4:
                    for s_name, t_id in std_to_orig.items():
                        if r_lower in s_name or s_name in r_lower:
                            return t_id
                            
                return r # Fallback to raw (External)

            # --- FILTER CHECK: Is this formula relevant? ---
            # It is relevant if:
            # A) Its Output Variable is in the CURRENT FILTERED SET (taxonomy_data)
            # B) Any of its Input Variables is in the CURRENT FILTERED SET
            
            is_relevant = False
            resolved_inputs = [resolve_id(i) for i in f_data.get('input_variables', [])]
            resolved_output = resolve_id(f_data.get('output_variable')) if f_data.get('output_variable') else None
            
            # Check Output
            if resolved_output and resolved_output in taxonomy_data:
                is_relevant = True
            
            # Check Inputs
            if not is_relevant:
                for inp_id in resolved_inputs:
                    if inp_id in taxonomy_data:
                        is_relevant = True
                        break
            
            if not is_relevant:
                continue

            add_node_safe(formula_node_id, f_name, "Formula", f_desc)

            # Inputs -> Formula
            for inp_var in f_data.get('input_variables', []):
                inp_id = resolve_id(inp_var)
                
                # If resolved to existing, it uses that node's type/role. 
                # If new, we treat as External.
                if inp_id not in existing_node_ids:
                    # Check if it is a HIDDEN internal node (in full taxonomy but not current view)
                    is_hidden_internal = full_taxonomy_ref and inp_id in full_taxonomy_ref
                    
                    if is_hidden_internal:
                        # Add it as a proper Input Node (Internal)
                        # Fetch details from full taxonomy
                        hidden_data = full_taxonomy_ref[inp_id]
                        
                        # Use Internal Type explicitly if it was 'Input'
                        n_type = hidden_data.get('node_type', 'Input')
                        if n_type == "Input":
                            n_type = "Input-Internal"
                            
                        add_node_safe(inp_id, hidden_data.get('standard_name', inp_id), n_type, hidden_data.get('description', ''))
                        
                        # --- FIX: Also restore Category Link for this hidden node ---
                        category = hidden_data.get("category", "Uncategorized")
                        cat_id = f"CAT_{category}"
                        add_node_safe(cat_id, category, "Category", "Data Category")
                        
                        edges.append(Edge(
                            start=cat_id, 
                            end=inp_id, 
                            properties={
                                "label": "", 
                                "style": "dashed", 
                                "color": "#BDBDBD", 
                                "directed": False 
                            }
                        ))
                    else:
                        add_node_safe(inp_id, inp_id, "Input-External")
                
                edges.append(Edge(
                    start=inp_id, 
                    end=formula_node_id, 
                    properties={
                        "label": "input", 
                        "style": "solid", 
                        "color": "#4285F4", 
                        "directed": True
                    }
                ))
            
                # Formula -> Output
            out_var = f_data.get('output_variable')
            if out_var:
                out_id = resolve_id(out_var)
                
                if out_id not in existing_node_ids:
                    # Check if it is a HIDDEN internal node
                    is_hidden_internal = full_taxonomy_ref and out_id in full_taxonomy_ref
                    
                    if is_hidden_internal:
                        # Restore Hidden Node
                        hidden_data = full_taxonomy_ref[out_id]
                        # Note: We enforce 'Derived-Internal' type/role because it is an output here, 
                        # even if taxonomy says 'Input' (though ideally taxonomy matches)
                        add_node_safe(out_id, hidden_data.get('standard_name', out_id), "Derived-Internal", hidden_data.get('description', ''))
                        
                        # Restore Category
                        category = hidden_data.get("category", "Uncategorized")
                        cat_id = f"CAT_{category}"
                        add_node_safe(cat_id, category, "Category", "Data Category")
                        
                        edges.append(Edge(
                            start=cat_id, 
                            end=out_id, 
                            properties={
                                "label": "", 
                                "style": "dashed", 
                                "color": "#BDBDBD", 
                                "directed": False 
                            }
                        ))
                    else:
                        add_node_safe(out_id, out_id, "Derived-External") 
                
                # Update role/type if it was existing (or just added) to reflect it's being calculated here
                for n in nodes:
                    if n.id == out_id:
                        current_type = n.properties.get('type', '')
                        if "External" in current_type:
                             n.properties['type'] = 'Derived-External'
                             n.properties['role'] = 'derived-external'
                        else:
                             n.properties['type'] = 'Derived-Internal'
                             n.properties['role'] = 'derived-internal'

                edges.append(Edge(
                    start=formula_node_id, 
                    end=out_id, 
                    properties={
                        "label": "yields", 
                        "style": "solid", 
                        "color": "#EA4335", 
                        "directed": True
                    }
                ))
                
    else:
        # --- LEGACY REGEX PATH ---
        for var_id, attributes in taxonomy_data.items():
            var_id_str = str(var_id)
            raw_formulas = attributes.get("related_formulas", [])
            if isinstance(raw_formulas, str): raw_formulas = [raw_formulas]
            
            for text in raw_formulas:
                match_input = re.search(r"Parameter in:?\s*(.*?)(?:\.|:|\(|$)", text, re.IGNORECASE)
                match_output = re.search(r"Result of:?\s*(.*?)(?:\.|:|\(|$)", text, re.IGNORECASE)
    
                if match_input:
                    form_name = match_input.group(1).strip()
                    form_id = f"FORM_{hash(form_name)}"
                    add_node_safe(form_id, form_name, "Formula", text)
                    
                    edges.append(Edge(
                        start=var_id_str, 
                        end=form_id, 
                        properties={
                            "label": "input", 
                            "style": "solid", 
                            "color": "#4285F4", 
                            "directed": True
                        }
                    ))
                
                elif match_output:
                    form_name = match_output.group(1).strip()
                    form_id = f"FORM_{hash(form_name)}"
                    add_node_safe(form_id, form_name, "Formula", text)
                    
                    if var_id_str in existing_node_ids:
                         for node in nodes:
                             if node.id == var_id_str:
                                 node.properties['role'] = 'derived'
                                 node.properties['type'] = 'Derived'
                                 break
    
                    edges.append(Edge(
                        start=form_id, 
                        end=var_id_str, 
                        properties={
                            "label": "yields", 
                            "style": "solid", 
                            "color": "#EA4335", 
                            "directed": True
                        }
                    ))
    
    # 4. Relationships (Common)
    for var_id, attributes in taxonomy_data.items():
        var_id_str = str(var_id)
        for target in attributes.get("relationships", []):
            if target in taxonomy_data:
                edges.append(Edge(
                    start=var_id_str, 
                    end=str(target), 
                    properties={
                        "label": "relates to", 
                        "style": "dotted", 
                        "color": "#9E9E9E", 
                        "directed": False
                    }
                ))

    return nodes, edges

# --- Visual Mappers ---

def get_node_style(node):
    role = node['properties'].get('role', 'input')
    styles = {
        "category": {"color": "#E0E0E0", "shape": NodeShape.ROUND_RECTANGLE},
        "input":    {"color": "#4285F4", "shape": NodeShape.ELLIPSE},
        "derived":  {"color": "#FBBC05", "shape": NodeShape.ELLIPSE},
        "formula":  {"color": "#EA4335", "shape": NodeShape.HEXAGON},
        "input-external":   {"color": "#AECBFA", "shape": NodeShape.ELLIPSE}, # Lighter Blue
        "derived-external": {"color": "#FEEFC3", "shape": NodeShape.ELLIPSE}  # Lighter Yellow
    }
    config = styles.get(role, styles["input"])
    # Reverting scale change to fix crash
    return NodeStyle(color=config["color"], shape=config["shape"])

def get_edge_style(edge):
    props = edge['properties']
    style_str = props.get('style', 'solid')
    
    if style_str == 'dashed': d_style = DashStyle.DASH
    elif style_str == 'dotted': d_style = DashStyle.DOT
    else: d_style = DashStyle.SOLID
    
    return EdgeStyle(
        color=props.get('color', '#000000'),
        dash_style=d_style,
        directed=props.get('directed', True),
        thickness=2.0
    )

def get_node_label_style(node):
    return LabelStyle(
        color="#000000", 
        text_position="center", 
        # Removing wrapping and background to ensure text is always rendered
    )

def get_edge_label_style(edge):
    return LabelStyle(
        background_color="#FFFFFFCC", 
        color="#333333", 
        text_position="center",
        wrapping="word"
    )

# ==========================================
# 2. MAIN APP LOGIC
# ==========================================

def app():
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
    st.sidebar.header("📂 Data Management")
    
    # Mode Indicator
    if rag_active:
        st.sidebar.success("🟢 AI System Active")
    else:
        st.sidebar.warning("🔴 Offline Mode")

    # Persistence (Save/Load)
    col_p1, col_p2 = st.sidebar.columns(2)
    
    # Load Logic
    # Load Logic
    avail_versions = get_taxonomy_versions()
    
    if avail_versions:
        # Dropdown for version selection
        v_options = [f"v{v[0]}" for v in avail_versions]
        v_paths = {f"v{v[0]}": v[1] for v in avail_versions}
        
        selected_v_label = col_p1.selectbox("Select Version", v_options, index=0, label_visibility="collapsed")
        target_path = v_paths[selected_v_label]
        
        if col_p1.button("📂 Load"):
            try:
                with open(target_path, "r") as f:
                    loaded_taxonomy = json.load(f)
                st.session_state.offline_taxonomy = loaded_taxonomy
                st.session_state["loaded_path"] = target_path # Track for overwrite
                
                if rag_active:
                    rag_manager.variable_taxonomy = loaded_taxonomy
                
                # Load Formulas if present
                # Path is .../vX/taxonomy_metadata.json -> .../vX/formulas_metadata.json
                formulas_path = os.path.join(os.path.dirname(target_path), "formulas_metadata.json")

                if os.path.exists(formulas_path):
                    with open(formulas_path, "r") as f:
                        loaded_formulas = json.load(f)
                    st.session_state.offline_formulas = loaded_formulas
                    st.toast(f"Loaded {len(loaded_formulas)} formulas from disk.", icon="🧮")
                    if rag_active:
                        rag_manager.formulas_registry = loaded_formulas
                else:
                    st.toast(f"No formula file found at {formulas_path}", icon="⚠️")
                    st.session_state.offline_formulas = {}
                    if rag_active:
                        rag_manager.formulas_registry = {}
                
                st.toast(f"Loaded {selected_v_label} successfully!", icon="✅")
                # Force rerun to update UI
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Load failed: {e}")
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
        current_versions = get_taxonomy_versions()
        next_v = 1
        if current_versions:
            next_v = current_versions[0][0] + 1
        
        # Save Controls
        if col_p2.button(f"💾 Save New (v{next_v})"):
            try:
                new_dir = os.path.join(TAXONOMY_DIR, f"v{next_v}")
                os.makedirs(new_dir, exist_ok=True)
                
                new_path = os.path.join(new_dir, "taxonomy_metadata.json")
                with open(new_path, "w") as f:
                    json.dump(taxonomy, f, indent=2)
                
                # Save Formulas
                new_formulas_path = os.path.join(new_dir, "formulas_metadata.json")
                with open(new_formulas_path, "w") as f:
                    json.dump(formulas_registry, f, indent=2)
                
                st.session_state["loaded_path"] = new_path # Update context
                st.toast(f"Saved Version v{next_v}!", icon="💾")
                st.rerun() 
            except Exception as e:
                st.sidebar.error(f"Save failed: {e}")

        # Overwrite Option
        loaded_path = st.session_state.get("loaded_path")
        if loaded_path and os.path.exists(loaded_path):
            current_v_name = os.path.basename(loaded_path)
            # Use a simpler label or icon to save space
            if col_p2.button(f"⚠️ Overwrite"):
                try:
                    with open(loaded_path, "w") as f:
                        json.dump(taxonomy, f, indent=2)
                    
                    # Overwrite Formulas
                    # Overwrite Formulas
                    loaded_formulas_path = os.path.join(os.path.dirname(loaded_path), "formulas_metadata.json")
                    with open(loaded_formulas_path, "w") as f:
                         json.dump(formulas_registry, f, indent=2)
                    st.toast(f"Overwritten {current_v_name}!", icon="💾")
                except Exception as e:
                    st.sidebar.error(f"Overwrite failed: {e}")
    else:
        col_p2.button("💾 Save", disabled=True)

    st.sidebar.divider()
    
    # --- 4. Persistent Graph Actions ---
    with st.sidebar.expander("⚡ Graph Actions", expanded=True):
        
        # A. Enrichment (Only if taxonomy exists)
        if taxonomy:
            st.markdown("**Taxonomy Enrichment**")
            st.caption("Expand graph with external knowledge.")
            enrich_dist = st.slider("Neighborhood Distance", 1, 3, 1, help="Depth of external variables to find.")
            
            if st.button("✨ Enrich Taxonomy"):
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
                            st.toast(f"Found {len(candidates)} new items!", icon="✨")
                        else:
                            st.sidebar.info("No new candidates found.")
        if st.button("🚀 New Taxonomy", help="Generate fresh taxonomy from dataset"):
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
        st.info("👋 Welcome to Data Interpretation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📂 Load Existing")
            if os.path.exists("taxonomy_metadata.json"):
                 st.info("Legacy `taxonomy_metadata.json` found in root. Consider moving it to data/taxonomy.")
            
            local_versions = get_taxonomy_versions()
            if local_versions:
                latest_v, latest_path = local_versions[0]
                st.success(f"Found {len(local_versions)} versions. Latest: v{latest_v}")
                
                v_opts = [f"v{v[0]}" for v in local_versions]
                sel_v = st.selectbox("Choose Version", v_opts, index=0)
                
                # Find path for selected version
                path_to_load = next(v[1] for v in local_versions if f"v{v[0]}" == sel_v)

                if st.button(f"Load {sel_v}", type="primary", use_container_width=True):
                    with open(path_to_load, "r") as f:
                        st.session_state.offline_taxonomy = json.load(f)
                    st.rerun()
            else:
                st.warning("No saved taxonomy files found in `data/taxonomy/`.")

        with col2:
            st.subheader("✨ Generate New")
            if not rag_active:
                st.warning("⚠️ RAG System is not initialized.")
                st.markdown("You need to initialize the AI system to generate a new taxonomy from your data.")
            else:
                if "working_df" not in st.session_state:
                    st.error("No dataset loaded. Please go to Home page.")
                else:
                    st.markdown("Analyze your dataset to generate a medical knowledge graph.")
                    if st.button("🚀 Generate Taxonomy", type="primary", width='stretch'):
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
    st.sidebar.header("🔍 Explore Data")
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
             search_text = f"{var} {data.get('standard_name', '')} {data.get('description', '')} {data.get('category', '')} {str(data.get('related_formulas', []))} {data.get('clinical_usage', '')}".lower()
        else:
             # Narrow search (Names only)
             search_text = f"{var} {data.get('standard_name', '')}".lower()
             
        text_match = search_query in search_text
        
        if cat_match and text_match:
            filtered_vars[var] = data
            
    st.sidebar.markdown(f"**Found {len(filtered_vars)} variables**")
    
    # Export Button
    st.sidebar.download_button(
        label="📥 Export JSON",
        data=json.dumps(taxonomy, indent=2),
        file_name="variable_taxonomy.json",
        mime="application/json"
    )

    # --- Main Content Tabs ---
    tab_overview, tab_details, tab_formulas, tab_graph, tab_refine = st.tabs(["📋 Overview", "🔍 Variable Details", "🧮 Formulas Details", "🕸️ Knowledge Graph", "🛠️ Refinement"])
    
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
                "Context": "✅" if data.get('clinical_usage') else "❌"
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
            st.markdown("### ✨ Enrichment Proposals")
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
                st.success(f"➕ {len(formulas)} New Formulas identified connecting these variables.")
                with st.expander("View Formulas"):
                    st.json(formulas)
            
            c_merge, c_discard = st.columns([1, 4])
            
            if c_merge.button("✅ Merge Selected", type="primary"):
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
                    
                    st.toast(f"Merged {count} vars & {len(new_formulas)} formulas!", icon="🎉")
                    time.sleep(1)
                    st.rerun()
            
            if c_discard.button("❌ Discard All"):
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
                    st.subheader(f"📌 {var_data.get('standard_name', selected_var_name)}")
                    st.caption(f"Original Name: `{selected_var_name}`")
                    st.markdown(f"**Category:** `{var_data.get('category', 'Uncategorized')}`")
                    st.divider()
                    st.markdown("### 📝 Definition")
                    st.info(var_data.get('description', 'No description available.'))
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("### 🧮 Related Formulas")
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
                        st.markdown("### 🏥 Clinical Usage")
                        usage = var_data.get('clinical_usage', "")
                        if usage:
                            st.markdown(usage)
                        else:
                            st.caption("No clinical context.")

    # --- TAB 3: KNOWLEDGE GRAPH ---
    with tab_graph:
        st.subheader("Interactive Clinical Knowledge Graph")
        
        col_legend = st.columns([1])[0]
        
        with col_legend:
            with st.expander("Legend", expanded=False):
                st.markdown("""
                **Nodes:**  
                🔵 **Input Var** 🟡 **Derived Var** 🛑 **Formula** ⬜ **Category**  
                <span style='color:#AECBFA'><b>●</b></span> **Ext. Input** <span style='color:#FEEFC3'><b>●</b></span> **Ext. Derived**

                **Edges:**  
                <span style='color:#4285F4'><b>──▶</b></span> **Input to Formula** (Blue)  
                <span style='color:#EA4335'><b>──▶</b></span> **Yields Variable** (Red)  
                <span style='color:#BDBDBD'><b>- - -</b></span> **Category Member** (Dashed)  
                <span style='color:#9E9E9E'><b>• • •</b></span> **Related To** (Dotted)
                """, unsafe_allow_html=True)
                


        # 2. Build Graph Elements (Using filtered data)
        # Use filtered_vars directly, consistent with sidebar
        graph_source = filtered_vars

        if len(graph_source) > 150:
            st.warning(f"⚠️ Rendering {len(graph_source)} variables. The graph might be slow.")

        # Pass FULL taxonomy as reference to allow finding hidden nodes
        nodes, edges = get_graph_data(graph_source, formulas_registry, full_taxonomy_ref=taxonomy)
        
        # --- FILTERS ---
        if nodes:
            c_f1, c_f2 = st.columns(2)
            
            # Helper for robust access
            def get_props(item):
                return getattr(item, 'properties', item.get('properties', {}) if isinstance(item, dict) else {})

            # 1. Node Types
            all_node_types = sorted(list(set(get_props(n).get('type', 'Unknown') for n in nodes)))
            sel_node_types = c_f1.multiselect("Node Types", all_node_types, default=all_node_types)
            
            # 2. Edge Types
            # Map edge labels to readable types
            def get_edge_type(e):
                p = get_props(e)
                l = p.get('label', '')
                if not l and p.get('style') == 'dashed': return 'Category Link'
                if not l and p.get('style') == 'dotted': return 'Related'
                return l if l else 'Unknown'

            all_edge_types = sorted(list(set(get_edge_type(e) for e in edges)))
            # Default: specific common types
            sel_edge_types = c_f2.multiselect("Edge Types", all_edge_types, default=all_edge_types)
            
            # Apply Filters
            filtered_nodes = [n for n in nodes if get_props(n).get('type') in sel_node_types]
            filtered_node_ids = set(n.id for n in filtered_nodes)
            
            # DEBUG
            # num_formulas_raw = len([n for n in nodes if get_props(n).get('type') == 'Formula'])
            # if num_formulas_raw > 0: st.sidebar.info(f"Raw Formulas found: {num_formulas_raw}")
            
            # Filter Edges: Both source and target must exist, AND edge type must be selected
            filtered_edges = [
                e for e in edges 
                if e.start in filtered_node_ids 
                and e.end in filtered_node_ids
                and get_edge_type(e) in sel_edge_types
            ]
            
            # Override for display
            nodes = filtered_nodes
            edges = filtered_edges

        # --- Metrics Display ---
        if nodes:
            # Count Nodes by Type
            node_counts = {}
            for n in nodes:
                ntype = get_props(n).get('type', 'Unknown')
                node_counts[ntype] = node_counts.get(ntype, 0) + 1
            
            edge_count = len(edges)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Nodes", len(nodes))
            m2.metric("Total Edges", edge_count)
            m3.metric("Formulas", node_counts.get('Formula', 0))
            m4.metric("Categories", node_counts.get('Category', 0))
            
            with st.expander("Detailed Metrics", expanded=False):
                d1, d2 = st.columns(2)
                with d1:
                    st.write("**Nodes by Type**")
                    st.dataframe(pd.DataFrame(list(node_counts.items()), columns=["Type", "Count"]), hide_index=True)
                with d2:
                    st.write("**Edge Summary**")
                    st.info(f"Visualizing interactions between {len(nodes)} entities.")
        
        if not nodes:
            st.info("No nodes to display based on current filters.")
        else:
            # 3. Initialize Widget
            widget = StreamlitGraphWidget(nodes=nodes, edges=edges)
            
            # 4. Apply Mappings
            widget.node_label_mapping = lambda n: n['properties']['label']
            widget.node_styles_mapping = get_node_style
            widget.node_label_style_mapping = get_node_label_style
            widget.node_tooltip_mapping = lambda n: n['properties']['tooltip']
            
            widget.edge_label_mapping = lambda e: e['properties']['label']
            widget.edge_styles_mapping = get_edge_style
            widget.edge_label_styles_mapping = get_edge_label_style

            # 5. Render
            widget.height = 750 
            
            # Rendering and getting selection
            selection_data = widget.show(key="main_kg_view", graph_layout=Layout.HIERARCHIC)
            
            # 6. Interaction Panel
            if selection_data:
                selected_ids = []
                
                # Recursive flatten function to handle lists, tuples, nested structures
                def flatten_and_collect(data):
                    if isinstance(data, (list, tuple)):
                        for item in data:
                            flatten_and_collect(item)
                    elif data is not None:
                        # Append as string, trim whitespace
                        s = str(data).strip()
                        if s: selected_ids.append(s)

                flatten_and_collect(selection_data)
                
                # Deduplicate
                selected_ids = list(set(selected_ids))

                if selected_ids:
                    st.divider()
                    st.markdown(f"### 🔎 Selection Details ({len(selected_ids)} items)")
                    
                    for sel_id in selected_ids:
                        with st.expander(f"Item: {sel_id}", expanded=True):
                            # Case A: It's a Variable in our Taxonomy
                            if sel_id in taxonomy:
                                data = taxonomy[sel_id]
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.write(f"**Standard Name:** {data.get('standard_name')}")
                                    st.write(f"**Category:** {data.get('category')}")
                                    st.info(data.get('description'))
                                with c2:
                                    st.write("**Clinical Usage:**")
                                    st.caption(data.get('clinical_usage', 'N/A'))
                                    if data.get('related_formulas'):
                                        st.write("**Formulas:**")
                                        for f in data['related_formulas']:
                                            st.code(f, language="text")
                            
                            # Case B: It's a Generated Node (Category or Formula)
                            else:
                                # Search in graph nodes
                                node_obj = next((n for n in nodes if n.id == sel_id), None)
                                if node_obj:
                                    role = node_obj.properties.get('role', '').title()
                                    label = node_obj.properties.get('label', '')
                                    st.write(f"**Type:** {role} Node")
                                    st.write(f"**Name:** {label}")
                                    
                                    if role == "Category":
                                        st.info(f"Groups all variables belonging to '{label}' category.")
                                    elif role == "Formula":
                                        st.markdown("#### 🧮 Formula Details")
                                        
                                        # Try to resolve valid Formula ID
                                        f_id = sel_id.replace("FORM_", "")
                                        f_data = formulas_registry.get(f_id)
                                        
                                        if f_data:
                                            # Rich Display from Registry
                                            c_fd1, c_fd2 = st.columns(2)
                                            # Math Display
                                            st.markdown("#### 📐 Formula Content")
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
                                                st.markdown("**📥 Inputs**")
                                                for inv in f_data.get('input_variables', []):
                                                    st.markdown(f"- `{inv}`")
                                            with c_out:
                                                st.markdown("**📤 Derived Variable**")
                                                st.markdown(f"`{f_data.get('output_variable', 'Unknown')}`")
                                        else:
                                            # Fallback to Node Props
                                            st.info(f"Logic: {node_obj.properties.get('tooltip', 'No details')}")
                                else:
                                    st.caption(f"ID not found in graph registry: {sel_id}")

    # --- TAB 3: FORMULA DETAILS (NEW) ---
    with tab_formulas:
        st.subheader("🧮 Formula Taxonomy")
        
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
                    st.info("💡 Hint: Try reloading the Taxonomy (vX) from the sidebar to fetch definitions.")
                sel_fdata = None
            else:
                st.markdown(f"**Found {len(filtered_forms)} formulas**")
                # Create readable labels
                f_opts = [f[0] for f in filtered_forms]
                f_labels = [f"{f[1].get('name', 'Unknown')} ({f[0]})" for f in filtered_forms]
                
                sel_f_idx = st.radio("Select Formula", range(len(filtered_forms)), format_func=lambda x: f_labels[x], label_visibility="collapsed")
                sel_fid, sel_fdata = filtered_forms[sel_f_idx]
        
        with f_col2:
            if sel_fdata:
                st.markdown(f"### {sel_fdata.get('name', 'Unnamed Formula')}")
                st.caption(f"ID: `{sel_fid}`")
                st.divider()
                
                # Math Display
                st.markdown("#### 📐 Formula Content")
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
                    st.markdown("**📥 Inputs**")
                    for inv in sel_fdata.get('input_variables', []):
                        st.markdown(f"- `{inv}`")
                with c_out:
                    st.markdown("**📤 Derived Variable**")
                    st.markdown(f"`{sel_fdata.get('output_variable', 'Unknown')}`")
                
                if sel_fdata.get('references'):
                    st.markdown("#### 📚 References")
                    for ref in sel_fdata['references']:
                        st.markdown(f"- {ref}")
            else:
                st.info("Select a formula from the left panel to view its details.")
            
    # --- TAB 4: REFINEMENT ---
    with tab_refine:
        st.subheader("🛠️ Taxonomy Refinement")
        
        if not rag_active:
            st.warning("⚠️ AI System Required")
            st.info("Refining the taxonomy requires the RAG system to be initialized.")
        else:
            st.info("Select variables to refine and provide feedback for the AI.")
            
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

if __name__ == "__main__":
    app()