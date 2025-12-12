import streamlit as st
import os
from manage.rag_manager import RAGManager # test

def render_rag_sidebar():
    """Renders the RAG & AI Settings sidebar component."""
    
    # Initialize RAG Manager if not present
    if 'rag_manager' not in st.session_state:
        st.session_state.rag_manager = RAGManager()
    else:
        # Hot Reload Fix: Check if class definition changed (Zombie Object)
        if st.session_state.rag_manager.__class__ is not RAGManager:
            # Re-initialize but keep API key
            old_key = getattr(st.session_state.rag_manager, 'api_key', None)
            st.session_state.rag_manager = RAGManager(api_key=old_key)
            st.toast("Updated AI System (Code Reload)", icon="🔄")

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    with st.sidebar.expander("🧠 RAG & AI Settings", expanded=False):
        st.caption("Configure AI Assistant")
        
        # API Key Handling
        env_api_key = os.getenv("GOOGLE_API_KEY", "")
        api_key = st.text_input("Google API Key", value=env_api_key, type="password", help="Required for AI features", key="rag_api_key_input")
        
        if api_key:
            st.session_state.rag_manager.api_key = api_key
            os.environ["GOOGLE_API_KEY"] = api_key
            
            # --- NEW SDK LOGIC START ---
            from google import genai
            try:
                # 1. Init Client
                client = genai.Client(api_key=api_key)
                
                # 2. Update Manager's client
                st.session_state.rag_manager.client = client
                
                # 3. List & Filter Models
                all_models = list(client.models.list())
                text_models = []
                
                # Filter for text generation capability (Mock/Heuristic based on name)
                # The 'google.genai' SDK Model object structure differs from 'google.generativeai'
                # fallback to name-based filtering to be safe.
                for m in all_models:
                    name_lower = m.name.lower()
                    # Exclude embeddings, imagen, veo, audio-only
                    if "embedding" in name_lower or "imagen" in name_lower or "veo" in name_lower:
                        continue
                    if "gemini" in name_lower or "gemma" in name_lower:
                        text_models.append(m)

                # 4. Group by Family
                families = {"Gemini": [], "Gemma": [], "Other": []}
                for m in text_models:
                    name_lower = m.name.lower()
                    if "gemini" in name_lower:
                        families["Gemini"].append(m)
                    elif "gemma" in name_lower:
                        families["Gemma"].append(m)
                    else:
                        families["Other"].append(m)
                
                # Family Selector
                available_families = [f for f, models in families.items() if models]
                selected_family = st.selectbox("Model Family", available_families, index=0, key="rag_model_family")
                
                # 5. Smart Version Filtering for Selected Family
                candidate_models = families[selected_family]
                model_options = []
                
                # Heuristic: Hide older versions if newer exist
                # e.g. hide '001' if '002' exists
                model_names = [m.name for m in candidate_models]
                for m in candidate_models:
                    name = m.name
                    # Check if essentially deprecated
                    is_old = False
                    if name.endswith("-001"):
                        # Look for -002 or no-suffix version
                        base = name[:-4] # remove -001
                        if f"{base}-002" in model_names or base in model_names:
                           is_old = True
                    
                    if not is_old:
                        model_options.append(name)

                # Sort: Latest/Highest versions first
                # Descending sort usually puts 'latest' or higher numbers first
                model_options.sort(reverse=True)

                if model_options:
                    selected_model = st.selectbox("Select AI Model", model_options, index=0, key="rag_model_select")
                else:
                    selected_model = "gemini-1.5-flash"
                    st.warning("No models found for this family. Using default.")
                    
            except Exception as e:
                st.error(f"Error configuring Google GenAI: {e}")
                selected_model = "gemini-1.5-flash"
        else:
            selected_model = "models/gemini-1.5-flash"

        # File Uploader for Documents
        uploaded_files = st.file_uploader("Upload Medical Literature (PDF)", type=["pdf"], accept_multiple_files=True, key="rag_file_uploader")
        if uploaded_files:
            docs_dir = "DOCUMENTS"
            if not os.path.exists(docs_dir):
                os.makedirs(docs_dir)
            
            new_files_count = 0
            for uploaded_file in uploaded_files:
                file_path = os.path.join(docs_dir, uploaded_file.name)
                # Only write if file doesn't exist or overwrite is intended (here we overwrite)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                new_files_count += 1
            
            if new_files_count > 0:
                st.success(f"Uploaded {new_files_count} documents. Please click 'Initialize/Update' below to process them.")

        # File Selection for Embedding
        selected_files = None
        docs_dir = "DOCUMENTS"
        if os.path.exists(docs_dir):
            all_files = [f for f in os.listdir(docs_dir) if f.lower().endswith('.pdf')]
            if all_files:
                # Default to all files if not explicitly changed
                selected_files = all_files
                
                with st.expander(f"📂 Select Documents in your library ({len(all_files)} available)", expanded=False):
                    selected_files = st.multiselect(
                        "Select Documents to Embed",
                        options=all_files,
                        default=all_files,
                        help="Select specific files to include in the RAG knowledge base.",
                        label_visibility="collapsed"
                    )
            else:
                st.info("No PDF documents found in DOCUMENTS folder.")

        # Adherence Slider
        # Use session state to persist value
        if 'rag_adherence_score' not in st.session_state:
            st.session_state.rag_adherence_score = 0.5

        adherence_score = st.slider(
            "Document Adherence vs Model Knowledge",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.rag_adherence_score,
            step=0.1,
            help="0.0: Rely mostly on Model Knowledge (Creative). 1.0: Rely strictly on Documents (Fact-based).",
            key="rag_adherence_slider"
        )
        
        # Check if slider changed and update immediately if initialized
        if adherence_score != st.session_state.rag_adherence_score:
            st.session_state.rag_adherence_score = adherence_score
            if st.session_state.rag_manager.initialized:
                success, msg = st.session_state.rag_manager.update_adherence_score(adherence_score)
                if success:
                    st.toast(msg, icon="✅")
                else:
                    st.error(msg)

        # Temperature Slider
        if 'rag_temperature' not in st.session_state:
            st.session_state.rag_temperature = 0.3

        temperature = st.slider(
            "Model Creativity (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.rag_temperature,
            step=0.1,
            help="0.0: Deterministic/Focused. 1.0: Creative/Random.",
            key="rag_temp_slider"
        )
        if temperature != st.session_state.rag_temperature:
             st.session_state.rag_temperature = temperature

        # Option to use existing database
        use_existing_db = st.checkbox("Use Existing Database (Skip Re-indexing)", value=True, help="If checked, loads the existing vector database instead of rebuilding it. Useful to avoid file lock errors and save time.")

        if st.button("Initialize/Update RAG System", key="rag_init_button"):
            with st.spinner("Initializing RAG System..."):
                # Progress Bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(percent, message):
                    progress_bar.progress(percent)
                    status_text.text(message)

                # Get dataset columns if available
                dataset_columns = None
                if 'data' in st.session_state and st.session_state.data is not None:
                    dataset_columns = st.session_state.data.columns.tolist()
                elif 'working_df' in st.session_state and st.session_state.working_df is not None:
                    dataset_columns = st.session_state.working_df.columns.tolist()

                success, msg = st.session_state.rag_manager.initialize_system(
                    model_name=selected_model,
                    adherence_score=adherence_score,
                    temperature=temperature,
                    dataset_columns=dataset_columns,
                    use_existing_db=use_existing_db,
                    progress_callback=update_progress,
                    selected_files=selected_files
                )
                
                # Clear progress on completion
                progress_bar.empty()
                status_text.empty()
                
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
        
        if st.session_state.rag_manager.initialized:
            st.success("✅ RAG System Active")
            
            # Display Global Context if available
            if hasattr(st.session_state.rag_manager, 'global_context'):
                gc = st.session_state.rag_manager.global_context
                if isinstance(gc, dict):
                    st.markdown("---")
                    st.markdown("### 🌍 Global Context")
                    st.markdown(f"**Domain:** {gc.get('domain', 'N/A')}")
                    st.caption(gc.get('context_description', ''))
                    
                    with st.expander("🎭 Expert Roles", expanded=False):
                        roles = gc.get('roles', {})
                        st.markdown(f"**Strict:** {roles.get('strict', 'N/A')}")
                        st.markdown(f"**Balanced:** {roles.get('balanced', 'N/A')}")
                        st.markdown(f"**Creative:** {roles.get('creative', 'N/A')}")
                else:
                     st.info(f"Context: {gc}")
        else:
            st.warning("⚠️ RAG System Inactive")
