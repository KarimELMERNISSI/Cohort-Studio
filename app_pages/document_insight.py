import json
import os

import streamlit as st
from yfiles_graphs_for_streamlit import (
    DashStyle,
    Edge,
    EdgeStyle,
    LabelStyle,
    Layout,
    Node,
    NodeShape,
    NodeStyle,
    StreamlitGraphWidget,
)

# ==========================================
# 1. GRAPH HELPER FUNCTIONS (Adapted for Docs)
# ==========================================

def filter_graph_by_docs(graph_json, visible_docs):
    """
    Returns a subset of the graph JSON containing only nodes/edges/formulas linked to visible_docs.
    """
    if not graph_json or not visible_docs:
        return {"nodes": [], "edges": [], "formulas": []}
        
    visible_set = set(visible_docs)
    
    # 1. Filter Nodes
    kept_nodes = []
    valid_node_ids = set()
    
    for n in graph_json.get("nodes", []):
        # Check citations
        citations = n.get("citations", [])
        
        # Fallback for legacy/single-doc nodes without citations list
        # (Though extraction now ensures citations, existing session state might be old)
        if not citations and n.get("source_text"):
             # Assume it belongs if we can't track it? Or strict?
             # If strictly generated from merge, it has citations.
             pass
             
        relevant_cits = [c for c in citations if c.get("doc") in visible_set]
        
        if relevant_cits:
            # Create a shallow copy to avoid mutating session state
            n_copy = n.copy()
            n_copy["citations"] = relevant_cits
            kept_nodes.append(n_copy)
            valid_node_ids.add(n.get("id"))
            
    # 2. Filter Edges
    kept_edges = []
    for e in graph_json.get("edges", []):
        if e.get("source") in valid_node_ids and e.get("target") in valid_node_ids:
            kept_edges.append(e)
            
    # 3. Filter Formulas
    kept_formulas = [f for f in graph_json.get("formulas", []) if f.get("doc") in visible_set]
    
    # 4. Filter Summaries
    kept_summaries = [s for s in graph_json.get("summaries", []) if s.get("doc") in visible_set]
    
    return {
        "nodes": kept_nodes,
        "edges": kept_edges,
        "formulas": kept_formulas,
        "summaries": kept_summaries
    }

def get_document_graph_data(graph_json, focus_node_id=None):
    nodes = []
    edges = []
    existing_node_ids = set()

    # Pre-filter check
    valid_neighbors = set()
    if focus_node_id:
        valid_neighbors.add(focus_node_id)
        for e in graph_json.get("edges", []):
            s = str(e.get("source"))
            t = str(e.get("target"))
            if s == focus_node_id: valid_neighbors.add(t)
            if t == focus_node_id: valid_neighbors.add(s)

    # Dynamic Type Coloring
    TYPE_COLORS = {
        "Concept": "#4285F4",    # Blue
        "Metric": "#34A853",     # Green
        "Finding": "#EA4335",    # Red
        "Disease": "#FBBC05",    # Yellow
        "Treatment": "#AA46BB",  # Purple
        "Method": "#00ACC1",     # Cyan
        "Other": "#9E9E9E"       # Grey
    }

    def add_node_safe(n_id, n_type, desc, source_text, page_ref):
        safe_id = str(n_id)
        if safe_id not in existing_node_ids:
            # Determine color based on type (fuzzy match)
            color = TYPE_COLORS["Other"]
            for key, c in TYPE_COLORS.items():
                if key.lower() in n_type.lower():
                    color = c
                    break
            
            nodes.append(Node(
                id=safe_id,
                properties={
                    "label": safe_id,
                    "type": n_type,
                    "color": color,
                    "description": desc,
                    "source_text": source_text,
                    "page_reference": page_ref
                }
            ))
            existing_node_ids.add(safe_id)

    # 1. Parse Nodes
    for n in graph_json.get("nodes", []):
        nid = n.get("id")
        
        # Filter: Skip if focusing and not in neighbors
        if focus_node_id and nid not in valid_neighbors:
             continue
             
        add_node_safe(
            nid, 
            n.get("type", "Concept"), 
            n.get("description", ""), 
            n.get("source_text", ""),
            n.get("page_reference", "")
        )

    # 2. Parse Edges
    for e in graph_json.get("edges", []):
        src = str(e.get("source"))
        tgt = str(e.get("target"))
        
        # Filter edge if focusing
        if focus_node_id:
            if src not in valid_neighbors or tgt not in valid_neighbors:
                continue
        
        # Ensure endpoints exist
        # Pass empty strings for inferred nodes
        if src not in existing_node_ids:
            add_node_safe(src, "Inferred", "Inferred from relationship", "", "")
        if tgt not in existing_node_ids:
            add_node_safe(tgt, "Inferred", "Inferred from relationship", "", "")

        edges.append(Edge(
            start=src,
            end=tgt,
            properties={
                "label": e.get("relation", "relates to"),
                "description": e.get("description", ""),
                "color": "#BDBDBD"
            }
        ))

    return nodes, edges

def get_node_style(node):
    color = node['properties'].get('color', '#9E9E9E')
    return NodeStyle(color=color, shape=NodeShape.ELLIPSE)

def get_edge_style(edge):
    return EdgeStyle(
        color=edge['properties'].get('color', '#BDBDBD'),
        dash_style=DashStyle.SOLID,
        directed=True,
        thickness=2.0
    )

def get_node_label_style(node):
    return LabelStyle(color="#000000", text_position="center")

def get_edge_label_style(edge):
    return LabelStyle(background_color="#FFFFFFCC", color="#333333", text_position="center", wrapping="word")


# ==========================================
# 2. PERSISTENCE HELPERS
# ==========================================
import datetime

GRAPH_STORAGE_DIR = os.path.join("data", "knowledge_graphs")

def get_saved_graphs(username=None):
    """Returns list of saved graph filenames (without extension). Filters by username."""
    if not os.path.exists(GRAPH_STORAGE_DIR):
        return []
    
    files = [f.replace(".json", "") for f in os.listdir(GRAPH_STORAGE_DIR) if f.endswith(".json")]
    
    # User Isolation
    if not username:
        return []
        
    if username != 'admin':
        files = [f for f in files if f.startswith(f"{username}_")]
        
    return sorted(files)

def save_graph(name, graph_json, source_docs, username=None):
    """Saves graph + metadata to disk."""
    if not os.path.exists(GRAPH_STORAGE_DIR):
        os.makedirs(GRAPH_STORAGE_DIR)
        
    safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '_', '-')]).strip()
    if not safe_name: return False, "Invalid name"
    
    # User Isolation: Prefix
    if username and username != 'admin' and not safe_name.startswith(f"{username}_"):
        safe_name = f"{username}_{safe_name}"

    filepath = os.path.join(GRAPH_STORAGE_DIR, f"{safe_name}.json")
    
    payload = {
        "meta": {
            "name": safe_name,
            "date": str(datetime.datetime.now()),
            "source_docs": source_docs
        },
        "graph": graph_json
    }
    
    try:
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)
        return True, None
    except Exception as e:
        return False, str(e)

def load_graph(name):
    """Loads graph payload."""
    filepath = os.path.join(GRAPH_STORAGE_DIR, f"{name}.json")
    if not os.path.exists(filepath):
        return None, "File not found"
        
    try:
        with open(filepath, "r") as f:
            payload = json.load(f)
        return payload, None
    except Exception as e:
        return None, str(e)


# ==========================================
# 3. MAIN APP
# ==========================================

def app():
    # st.title("Document Knowledge Graph") # Removed per user preference previously
    
    rag_manager = st.session_state.get('rag_manager')
    # Use persistence even if RAG not active? Preferably yes, but RAG required for new generation.
    
    # Sidebar: Document Selection
    st.sidebar.header("Documents Selection")

    # --- PERSISTENCE: LOAD ---
    username = st.session_state.get('username')
    saved_graphs = get_saved_graphs(username)
    if saved_graphs:
        with st.sidebar.expander("Load Saved Analysis", expanded=False):
            selected_load = st.selectbox("Select Analysis", [""] + saved_graphs, index=0)
            if selected_load and st.button("Load Graph"):
                payload, err = load_graph(selected_load)
                if err:
                    st.sidebar.error(f"Load failed: {err}")
                else:
                    st.session_state.doc_graph_json = payload.get("graph")
                    st.session_state.doc_graph_source = payload.get("meta", {}).get("source_docs", [])
                    st.success(f"Loaded '{selected_load}'!")
                    st.rerun()
    
    st.sidebar.divider()

    docs = []
    if rag_manager and rag_manager.initialized:
        docs = rag_manager.get_available_documents()
    
    if not docs and (not rag_manager or not rag_manager.initialized):
        st.sidebar.warning("AI System not initialized. You can only load saved graphs.")
    
    selected_docs = st.sidebar.multiselect("Choose Document(s)", docs, default=st.session_state.get("doc_graph_source", []))
    
    # State Management for Graph
    if "doc_graph_json" not in st.session_state:
        st.session_state.doc_graph_json = None
    if "doc_graph_source" not in st.session_state:
        st.session_state.doc_graph_source = []
        
    # Generate Button
    if rag_manager and rag_manager.initialized:
        if st.sidebar.button("Generate Graph", type="primary", disabled=len(selected_docs)==0):
            progress_bar = st.progress(0, text="Starting extraction...")
            def update_progress(p, t):
                progress_bar.progress(p, text=t)
                
            json_res, error = rag_manager.extract_merged_graph_from_docs(selected_docs, progress_callback=update_progress)
            
            progress_bar.empty()
            
            if error:
                st.error(f"Extraction failed: {error}")
            else:
                st.session_state.doc_graph_json = json_res
                st.session_state.doc_graph_source = selected_docs
                st.success(f"Graph merged from {len(selected_docs)} documents!")
                st.rerun()

    # Reset if document changed and graph exists for old one?
    if st.session_state.doc_graph_json and set(st.session_state.doc_graph_source) != set(selected_docs):
        st.warning(f"Displaying graph for **{len(st.session_state.doc_graph_source)} docs**. Click 'Generate Graph' to update.")
        
    # --- PERSISTENCE: SAVE ---
    if st.session_state.doc_graph_json:
        with st.sidebar.expander("Save Analysis", expanded=True):
            save_name = st.text_input("Analysis Name", placeholder="e.g. Heart Failure Study")
            if st.button("Save Graph"):
                if not save_name:
                    st.sidebar.error("Please enter a name.")
                else:
                    success, msg = save_graph(save_name, st.session_state.doc_graph_json, st.session_state.doc_graph_source, username=st.session_state.get('username'))
                    if success:
                        st.sidebar.success("Saved!")
                        # st.rerun() # Optional, to update load list
                    else:
                        st.sidebar.error(f"Save failed: {msg}")

    # Main Graph Display
    if st.session_state.doc_graph_json:
        
        # --- NEW: Filter by Document ---
        st.divider()
        all_graph_docs = sorted(list(set(st.session_state.doc_graph_source))) # or iterate citations to be safe
        visible_docs = st.sidebar.multiselect("Filter Visible Documents", all_graph_docs, default=all_graph_docs)
        
        # Apply Filtering
        filtered_graph = filter_graph_by_docs(st.session_state.doc_graph_json, visible_docs)
        
        # Create Tabs
        tab_graph, tab_formulas, tab_coverage, tab_summary, tab_chat = st.tabs(["Knowledge Graph", "Explicit Formulas", "Dataset Coverage", "Document Summary", "Chat with Document"])
        
        with tab_graph:
            col_graph, col_details = st.columns([2, 1])
            
            with col_graph:
                # Get Selection for filtering
                selected_entity = st.session_state.get("doc_graph_selected_node", None)
                focus_mode = st.toggle("Focus on Selected Entity", value=False)
                
                target_node = selected_entity if (focus_mode and selected_entity) else None
                
                # USE FILTERED GRAPH
                nodes, edges = get_document_graph_data(filtered_graph, focus_node_id=target_node)
                
                if not nodes:
                    st.warning("No nodes found for the selected documents.")
                
                w = StreamlitGraphWidget(nodes=nodes, edges=edges)

                # Apply Mappings (Properties, not Init Args)
                w.node_label_mapping = lambda n: n['properties']['label']
                w.node_styles_mapping = get_node_style
                w.node_label_style_mapping = get_node_label_style
                w.node_tooltip_mapping = lambda n: n['properties']['description']
                
                w.edge_label_mapping = lambda e: e['properties']['label']
                w.edge_styles_mapping = get_edge_style
                w.edge_label_styles_mapping = get_edge_label_style

                w.height = 800
                
                w.show(key="doc_graph_widget", graph_layout=Layout.HIERARCHIC)
                
            with col_details:
                st.subheader("Node Details")
                st.info("Select an entity to verify its source(s) in the documents.")
                
                with st.expander("Explore Entities", expanded=True):
                    all_nodes = filtered_graph.get("nodes", [])
                    node_names = sorted([n["id"] for n in all_nodes])
                    
                    # Sync with session state
                    selected_node_name = st.selectbox("Select Entity", node_names, key="doc_graph_selected_node")
                    
                    if selected_node_name:
                        node_data = next((n for n in all_nodes if n["id"] == selected_node_name), None)
                        if node_data:
                            st.markdown(f"**Type:** {node_data.get('type')}")
                            st.markdown(f"**Description:** {node_data.get('description')}")
                            
                            st.divider()
                            st.caption("Documents & Citations:")
                            
                            citations = node_data.get('citations', [])
                            # Backward compatibility if single doc extraction exists in session
                            if not citations and node_data.get('source_text'):
                                citations = [{
                                    "doc": "Single Document", 
                                    "page": node_data.get('page_reference', "?"),
                                    "text": node_data.get('source_text')
                                }]
                                
                            for idx, cit in enumerate(citations):
                                p_val = str(cit.get('page', '?')).lower().replace('page ', '').replace('page', '').strip()
                                st.markdown(f"**{idx+1}. {cit.get('doc')}** (Page {p_val})")
                                st.markdown(f"> *{cit.get('text')}*")
                                st.caption("---")

        with tab_formulas:
            formulas = filtered_graph.get("formulas", [])
            if not formulas:
                st.info("No explicit mathematical formulas found in the visible documents.")
            else:
                st.subheader(f"Found {len(formulas)} Formulas")
                for f in formulas:
                    p_val = str(f.get('page', '?')).lower().replace('page ', '').replace('page', '').strip()
                    with st.expander(f"{f.get('name', 'Formula')} ({f.get('doc')}, Page {p_val})"):
                        st.code(f.get('expression', 'N/A'))
                        st.write(f.get('description', ''))
 
        with tab_chat:
            st.subheader("Chat with Visible Documents (v1.1)")
            
            if st.button("Clear Chat History"):
                st.session_state.doc_chat_history = []
                st.rerun()
            
            # Chat History
            if "doc_chat_history" not in st.session_state:
                st.session_state.doc_chat_history = []
                
            # Display Chat
            for msg in st.session_state.doc_chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            
            # Input
            if prompt := st.chat_input("Ask about these documents..."):
                # Add user message
                st.session_state.doc_chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)
                    
                # Get response
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing documents..."):
                        # Use visible docs for context
                        response = rag_manager.chat_with_specific_doc(visible_docs, prompt)
                        st.write(response)
                        st.session_state.doc_chat_history.append({"role": "assistant", "content": response})

        with tab_coverage:
            st.subheader("Dataset Variable Coverage Analysis")
            st.info("Check if variables in your loaded dataset are defined or discussed in these documents.")
            
            if "working_df" not in st.session_state:
                st.warning("No dataset loaded. Please go to 'Data Preparation' to load a CSV first.")
            else:
                df_cols = list(st.session_state["working_df"].columns)
                
                if st.button("Check Coverage"):
                    with st.spinner("Matching dataset variables to knowledge graph nodes..."):
                        # Use filtered graph
                        coverage_results = rag_manager.match_columns_to_graph(df_cols, filtered_graph)
                        st.session_state.doc_coverage_results = coverage_results
                
                if "doc_coverage_results" in st.session_state:
                    res = st.session_state.doc_coverage_results
                    
                    # Compute Stats
                    covered = [k for k,v in res.items() if v["match_found"]]
                    st.metric("Coverage Score", value=f"{len(covered)} / {len(df_cols)} Variables", delta=f"{len(covered)/len(df_cols):.1%} of dataset")
                    
                    # Display Results
                    for col in res:
                        match_info = res[col]
                        found = match_info["match_found"]
                        
                        icon = "Covered" if found else "Missing"
                        # Use container/expander for clean look
                        with st.expander(f"{icon} {col}", expanded=found):
                            if found:
                                node = match_info["node"]
                                st.markdown(f"**Matched Concept:** `{node.get('id')}` (Confidence: {match_info.get('confidence',0):.2f})")
                                st.markdown(f"**Description:** {node.get('description')}")
                                
                                # Show all citations for this node from visible docs
                                cit_list = node.get("citations", [])
                                if cit_list:
                                    st.caption("Sources:")
                                    for cit in cit_list:
                                        st.markdown(f"- **{cit.get('doc')}** (p.{cit.get('page')}): {cit.get('text')}")
                            else:
                                st.caption("No direct mention found.")

        with tab_summary:
            st.subheader("Document Summaries")
            summaries = filtered_graph.get("summaries", [])
            
            if not summaries:
                st.info("No summaries available. Try regenerating the graph.")
            else:
                for s in summaries:
                    with st.container(border=True):
                        st.markdown(f"### {s.get('doc', 'Unknown Document')}")
                        st.caption(f"**Title**: {s.get('title', 'N/A')}")
                        st.caption(f"**Type**: {s.get('doc_type', 'N/A')}")
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("#### Objective")
                            st.write(s.get('objective', ''))
                            
                            st.markdown("#### Methods")
                            st.write(s.get('methods', ''))
                        
                        with c2:
                            st.markdown("#### Key Findings")
                            st.write(s.get('key_findings', ''))
                            
                            st.markdown("#### Clinical Significance")
                            st.write(s.get('significance', ''))
                            
                        if s.get("top_concepts"):
                            st.markdown("**Top Concepts:**")
                            # Simple chips
                            st.markdown(" ".join([f"`{c}`" for c in s.get("top_concepts", [])]))
                        
    else:
        st.info("Select a document and click 'Generate Graph' to start.")
