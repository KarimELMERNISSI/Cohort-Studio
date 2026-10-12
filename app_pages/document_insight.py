import json

import streamlit as st
from graph.builders.document import build_document_graph, filter_by_documents
from graph.model import LayoutHint, RenderOptions
from graph.ui import current_theme, render_graph, render_graph_exports, renderer_selector

# ==========================================
# 1. GRAPH HELPER FUNCTIONS (Adapted for Docs)
# ==========================================

# Filtering, merging and graph construction live in graph.builders.document.


# ==========================================
# 2. PERSISTENCE HELPERS
# ==========================================
import datetime

from core.workspace import UserScope, resolve_within, safe_filename
from ui.components import no_dataset


def _graph_dir(username):
    """Knowledge-graph directory of the user's workspace (audit E4)."""
    if not username:
        return None
    return UserScope.for_user(username).knowledge_graphs_dir()


def get_saved_graphs(username=None):
    """Returns the saved graph names of the user's workspace."""
    directory = _graph_dir(username)
    if directory is None or not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.json"))


def save_graph(name, graph_json, source_docs, username=None):
    """Saves graph + metadata into the user's workspace."""
    directory = _graph_dir(username)
    if directory is None:
        return False, "Sign in to save graphs"
    safe_name = safe_filename(name, default="")
    if not safe_name:
        return False, "Invalid name"
    filepath = directory / f"{safe_name}.json"

    payload = {
        "meta": {
            "name": safe_name,
            "date": str(datetime.datetime.now()),
            "source_docs": source_docs
        },
        "graph": graph_json
    }

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return True, None
    except Exception as e:
        return False, str(e)


def load_graph(name, username=None):
    """Loads a graph payload from the user's workspace."""
    directory = _graph_dir(username)
    if directory is None:
        return None, "Sign in to load graphs"
    try:
        filepath = resolve_within(directory, f"{safe_filename(name)}.json")
    except ValueError as e:
        return None, str(e)
    if not filepath.exists():
        return None, "File not found"

    try:
        with open(filepath, "r", encoding="utf-8") as f:
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
                payload, err = load_graph(selected_load, username=username)
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
    
    raw_default = st.session_state.get("doc_graph_source", [])
    valid_defaults = [d for d in raw_default if d in docs] if isinstance(raw_default, list) else []
    selected_docs = st.sidebar.multiselect("Choose Document(s)", docs, default=valid_defaults)
    
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
        filtered_graph = filter_by_documents(st.session_state.doc_graph_json, visible_docs).to_json_dict()
        
        # Create Tabs
        tab_graph, tab_formulas, tab_coverage, tab_summary, tab_chat = st.tabs(["Knowledge Graph", "Explicit Formulas", "Dataset Coverage", "Document Summary", "Chat with Document"])
        
        with tab_graph:
            col_graph, col_details = st.columns([2, 1])
            
            with col_graph:
                selected_entity = st.session_state.get("doc_graph_selected_node", None)
                c_focus, c_engine = st.columns(2)
                focus_mode = c_focus.toggle("Focus on Selected Entity", value=False, key="doc_graph_focus_mode")
                renderer_selector(container=c_engine)
                target_node = selected_entity if (focus_mode and selected_entity) else None

                spec = build_document_graph(filtered_graph, focus=target_node, theme=current_theme())
                if not spec.nodes:
                    st.warning("No nodes found for the selected documents.")
                else:
                    selection = render_graph(
                        spec,
                        RenderOptions(key="doc_graph_widget", height=800, selectable=True, layout=LayoutHint.HIERARCHICAL),
                    )
                    # A node selected in the graph becomes the entity shown in the details panel
                    if selection.node_ids:
                        clicked = selection.node_ids[0]
                        if clicked in {n.get("id") for n in filtered_graph.get("nodes", [])}:
                            st.session_state["doc_graph_selected_node"] = clicked
                    render_graph_exports(spec, RenderOptions(key="doc_graph_widget"), file_stem="document_graph")
                
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
                no_dataset()
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
