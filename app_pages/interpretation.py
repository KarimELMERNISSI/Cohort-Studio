import streamlit as st
import pandas as pd
import json
# from yfiles_graphs_for_streamlit import GraphWidget # Not used directly due to issues

def app():
    
    if 'rag_manager' not in st.session_state or not st.session_state.rag_manager.initialized:
        st.warning("RAG System is not initialized. Please configure it in the sidebar.")
        return

    rag_manager = st.session_state.rag_manager
    taxonomy = rag_manager.variable_taxonomy
    
    if not taxonomy:
        st.info("No variable taxonomy available. You can generate it here.")
        
        if "working_df" not in st.session_state:
            st.warning("No data loaded. Please go to the Home page to load your dataset.")
            st.page_link("app_pages/home.py", label="Go to Home Page", icon="🏠")
            return

        if st.button("Generate Taxonomy", type="primary"):
            progress_bar = st.progress(0, text="Starting taxonomy generation...")
            def update_progress(percent, text):
                progress_bar.progress(percent, text=text)
            
            # Generate columns_info from working_df
            df = st.session_state["working_df"]
            columns_info = []
            for col in df.columns:
                info = {"name": col}
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
            
            # Get current manual renames to guide the taxonomy
            current_renames = st.session_state.get("rename_dict", {})
            
            taxonomy, error = rag_manager.generate_variable_taxonomy(
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
                st.rerun()
        return

    # --- Sidebar Filters & Search ---
    st.sidebar.header("🔍 Explore Data")
    
    # Search
    search_query = st.sidebar.text_input("Search Variables", placeholder="Name, description, formula...").lower()
    
    # Category Filter
    categories = set()
    for var_data in taxonomy.values():
        cat = var_data.get('category', 'Uncategorized')
        categories.add(cat)
    
    selected_category = st.sidebar.selectbox("Filter by Category", ["All"] + sorted(list(categories)))
    
    # Filter Logic
    filtered_vars = {}
    for var, data in taxonomy.items():
        # Category Match
        cat_match = (selected_category == "All" or data.get('category') == selected_category)
        
        # Search Match
        search_text = f"{var} {data.get('standard_name', '')} {data.get('description', '')} {str(data.get('related_formulas', []))}".lower()
        text_match = search_query in search_text
        
        if cat_match and text_match:
            filtered_vars[var] = data
            
    st.sidebar.markdown(f"**Found {len(filtered_vars)} variables**")
    
    # Export Button
    st.sidebar.divider()
    st.sidebar.download_button(
        label="📥 Export Taxonomy (JSON)",
        data=json.dumps(taxonomy, indent=2),
        file_name="variable_taxonomy.json",
        mime="application/json"
    )

    if not filtered_vars:
        st.warning("No variables match your search filters.")
        return

    # --- Main Content Tabs ---
    tab_overview, tab_details, tab_graph, tab_refine = st.tabs(["📋 Overview", "🔍 Variable Details", "🕸️ Knowledge Graph", "🛠️ Refinement"])
    
    # --- TAB 1: OVERVIEW ---
    with tab_overview:
        st.subheader("Dataset Overview")
        
        # Stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Variables", len(taxonomy))
        col2.metric("Standardized", len([v for v in taxonomy.values() if v.get('standard_name')]))
        col3.metric("With Formulas", len([v for v in taxonomy.values() if v.get('related_formulas')]))
        
        st.divider()
        
        # Table
        table_data = []
        for var, data in filtered_vars.items():
            table_data.append({
                "Original Name": var,
                "Standard Name": data.get('standard_name', 'N/A'),
                "Category": data.get('category', 'N/A'),
                "Formulas": len(data.get('related_formulas', [])),
                "Context": "✅" if data.get('clinical_usage') else "❌"
            })
        
        st.dataframe(
            pd.DataFrame(table_data), 
            width='stretch',
            column_config={
                "Original Name": st.column_config.TextColumn("Original Name", help="Name in the dataset"),
                "Standard Name": st.column_config.TextColumn("Standard Concept", help="Medical standard name"),
            }
        )

    # --- TAB 2: VARIABLE DETAILS ---
    with tab_details:
        col_sel, col_content = st.columns([1, 3])
        
        with col_sel:
            selected_var_name = st.radio("Select Variable", sorted(filtered_vars.keys()), label_visibility="collapsed")
            
        with col_content:
            if selected_var_name:
                var_data = filtered_vars[selected_var_name]
                
                # Header
                st.subheader(f"📌 {var_data.get('standard_name', selected_var_name)}")
                st.caption(f"Original Name: `{selected_var_name}`")
                
                # Tags
                st.markdown(f"**Category:** `{var_data.get('category', 'Uncategorized')}`")
                
                st.divider()
                
                # Description
                st.markdown("### 📝 Definition")
                st.info(var_data.get('description', 'No description available.'))
                
                # Formulas & Context
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### 🧮 Related Formulas")
                    formulas = var_data.get('related_formulas', [])
                    if formulas:
                        for f in formulas:
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
        st.subheader("Variable Relationship Network")
        st.caption("Visualizing connections between variables through medical formulas.")
        
        try:
            nodes = []
            edges = []
            
            # Track created categories to avoid duplicates
            created_categories = set()
            
            for var, data in filtered_vars.items():
                # 1. Variable Node
                cat = data.get('category', 'Uncategorized')
                node_type = data.get('node_type', 'Input')
                
                # Style logic based on Category AND Node Type
                var_shape = 'rectangle'
                var_color = '#ADD8E6' # lightblue (default)
                
                # Base color by category
                if 'Vital' in cat: var_color = '#F08080' # lightcoral
                elif 'Lab' in cat: var_color = '#90EE90' # lightgreen
                elif 'Demographics' in cat: var_color = '#FFB6C1' # lightpink
                
                # Shape by Node Type (overrides category shape preference if needed, or combines)
                if node_type == 'Outcome':
                    var_shape = 'octagon'
                    var_color = '#FFD700' # Gold for outcomes
                elif node_type == 'Derived':
                    var_shape = 'diamond'
                # else Input -> rectangle/round-rectangle
                
                # Build rich description for tooltip/details
                desc_html = f"""
                <b>{data.get('standard_name', var)}</b><br>
                <i>{cat}</i> | <i>{node_type}</i><br><br>
                {data.get('description', '')}<br><br>
                <b>Clinical Usage:</b> {data.get('clinical_usage', 'N/A')}
                """
                
                nodes.append({
                    'id': var,
                    'label': f"{data.get('standard_name', var)}\n({var})",
                    'properties': {
                        'color': var_color,
                        'shape': var_shape,
                        'description': data.get('description', ''),
                        'detail': desc_html, 
                        'tooltip': desc_html
                    }
                })
                
                # 2. Category Node & Edge (Hierarchical)
                if cat not in created_categories:
                    nodes.append({
                        'id': f"cat_{cat}",
                        'label': cat,
                        'properties': {
                            'color': '#E0E0E0', # lightgray
                            'shape': 'hexagon', 
                            'description': f"Category: {cat}",
                            'detail': f"<b>Category: {cat}</b><br>Grouping for related variables."
                        }
                    })
                    created_categories.add(cat)
                
                # Edge: Category -> Variable
                edges.append({
                    'id': f"e_cat_{cat}_{var}",
                    'start': f"cat_{cat}",
                    'end': var,
                    'label': '',
                    'properties': {'color': '#CCCCCC', 'style': 'dashed'} 
                })

                # 3. Formula Nodes & Edges
                formulas = data.get('related_formulas', [])
                if isinstance(formulas, str):
                    formulas = [formulas] 
                    
                for i, formula in enumerate(formulas):
                    formula_id = f"formula_{hash(formula)}"
                    
                    if not any(n['id'] == formula_id for n in nodes):
                        f_label = formula[:30] + "..." if len(formula) > 30 else formula
                        nodes.append({
                            'id': formula_id,
                            'label': f_label,
                            'properties': {
                                'color': '#FFFFE0', # lightyellow
                                'shape': 'ellipse',
                                'description': formula,
                                'detail': f"<b>Formula:</b><br>{formula}"
                            }
                        })
                    
                    # Edge: Variable -> Formula
                    edges.append({
                        'id': f"e_{var}_{formula_id}",
                        'start': var,
                        'end': formula_id,
                        'label': '',
                        'properties': {'color': '#A0A0A0'}
                    })
                    
                # 4. Direct Relationships (New)
                relationships = data.get('relationships', [])
                for rel_var in relationships:
                    if rel_var in filtered_vars: # Only link if target exists in filter
                        edges.append({
                            'id': f"e_rel_{var}_{rel_var}",
                            'start': var,
                            'end': rel_var,
                            'label': 'related',
                            'properties': {'color': '#FFA500'} # Orange for direct relations
                        })
            
            # Inject CSS to:
            # 1. Make the iframe responsive (80vh)
            # 2. Hide yFiles logo and about button (attempting common classes)
            st.markdown(
                """
                <style>
                iframe[title="yfiles_graphs_for_streamlit.Streamlit_Graph_Widget"] {
                    height: 80vh !important;
                }
                /* Attempt to hide yFiles branding inside the iframe (Note: this might not work if cross-origin, 
                   but usually streamlit components are same-origin or allow styling) 
                   Actually, we can't style inside the iframe from here easily. 
                   But we can try to hide the footer if it's outside. 
                   If it's inside, we might be out of luck without JS injection.
                */
                </style>
                """,
                unsafe_allow_html=True
            )
            
            try:
                from yfiles_graphs_for_streamlit import _component_func
                
                # Layout selector
                layout_option = st.radio("Graph Layout", ["hierarchic", "organic", "radial"], horizontal=True, index=0)
                
                # Render graph and capture selection
                selected_node_id = _component_func(
                    nodes=nodes,
                    edges=edges,
                    directed=True, 
                    graph_layout=layout_option, 
                    key="knowledge_graph"
                )
                
                # --- Selection Details Panel ---
                if selected_node_id:
                    # Find the node data
                    selected_data = next((n for n in nodes if n['id'] == selected_node_id), None)
                    
                    if selected_data:
                        with st.container():
                            st.info(f"Selected: {selected_data.get('label', 'Unknown')}")
                            
                            # Parse the HTML detail we created or use raw properties
                            # We stored rich info in 'detail' property, but let's reconstruct it nicely here
                            # since we have the ID, we can look up the original variable data if it's a variable
                            
                            var_id = selected_node_id
                            if var_id in filtered_vars:
                                data = filtered_vars[var_id]
                                c1, c2 = st.columns([2, 1])
                                with c1:
                                    st.markdown(f"### {data.get('standard_name', var_id)}")
                                    st.markdown(f"**Category:** {data.get('category', 'N/A')}")
                                    st.markdown(f"**Description:** {data.get('description', 'N/A')}")
                                with c2:
                                    st.markdown("**Clinical Usage:**")
                                    st.caption(data.get('clinical_usage', 'N/A'))
                                    
                                    formulas = data.get('related_formulas', [])
                                    if formulas:
                                        st.markdown("**Formulas:**")
                                        for f in formulas:
                                            st.code(f)
                            elif var_id.startswith("cat_"):
                                st.markdown(f"### Category: {var_id.replace('cat_', '')}")
                                st.write("Grouping node for variables.")
                            elif var_id.startswith("formula_"):
                                st.markdown("### Formula")
                                st.code(selected_data.get('properties', {}).get('description', ''))
                    
            except ImportError:
                st.error("Could not import _component_func from yfiles_graphs_for_streamlit")
            except Exception as e:
                st.error(f"Error rendering graph: {e}")
                st.write(type(e))
        
        except Exception as e:
            st.error(f"Error in Knowledge Graph tab: {e}")

    # --- TAB 4: REFINEMENT ---
    with tab_refine:
        st.subheader("🛠️ Taxonomy Refinement")
        st.info("Select variables that need correction and provide feedback. The AI will re-process them using your input.")
        
        # Selection
        vars_to_refine = st.multiselect(
            "Select variables to refine:",
            options=sorted(filtered_vars.keys()),
            format_func=lambda x: f"{x} ({filtered_vars[x].get('standard_name', 'Unknown')})"
        )
        
        if vars_to_refine:
            feedback_dict = {}
            st.write("### Provide Feedback")
            for var in vars_to_refine:
                current_desc = filtered_vars[var].get('description', 'No description')
                st.markdown(f"**{var}**: `{current_desc}`")
                feedback = st.text_area(f"Feedback for {var}:", key=f"feed_{var}", placeholder="e.g., 'This is actually a date', 'Unit is wrong, should be mg/dL'")
                if feedback:
                    feedback_dict[var] = feedback
                st.divider()
            
            if st.button("Refine Selected Variables", type="primary"):
                if not feedback_dict:
                    st.warning("Please provide feedback for at least one variable.")
                else:
                    with st.spinner("Refining taxonomy based on your feedback..."):
                        # Get columns info from home page component if possible, or pass empty list
                        columns_info = []
                        if "working_df" in st.session_state:
                            df = st.session_state["working_df"]
                            # Reconstruct minimal info
                            for col in df.columns:
                                info = {"name": col}
                                if pd.api.types.is_numeric_dtype(df[col]):
                                    info["stats"] = {
                                        "min": float(df[col].min()),
                                        "max": float(df[col].max()),
                                        "mean": float(df[col].mean())
                                    }
                                else:
                                    info["top_values"] = {str(k): v for k, v in df[col].value_counts().head(5).to_dict().items()}
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
