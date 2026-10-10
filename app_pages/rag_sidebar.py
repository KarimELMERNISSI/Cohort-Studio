"""
RAG and AI Configuration Sidebar Component.

Provides a unified, provider-agnostic interface for configuring:
- AI Providers: Google Gemini, Ollama (Local Open Source), OpenAI, Mistral AI
- Model selection and dynamic model discovery
- Local and cloud embedding models
- Document retrieval parameters and clinical adherence calibration
"""

import os

import streamlit as st
from dotenv import load_dotenv

from manage.rag import ConnectorFactory, ProviderConfig, ProviderType

load_dotenv()


def render_rag_sidebar():
    """Renders the professional RAG & AI Configuration sidebar."""
    from manage.rag_manager import RAGManager

    # Initialize RAG Manager in session state if not already present
    if "rag_manager" not in st.session_state:
        st.session_state.rag_manager = RAGManager()
    else:
        # Hot-reload safety check if class definition was updated in memory
        if st.session_state.rag_manager.__class__ is not RAGManager:
            old_key = getattr(st.session_state.rag_manager, "api_key", None)
            st.session_state.rag_manager = RAGManager(api_key=old_key)

    with st.sidebar.expander("RAG & AI Configuration", expanded=False):
        st.caption("Model Providers, Embeddings & Literature Retrieval")

        # Provider Selector
        provider_display_names = {
            "gemini": "Google Gemini (Cloud)",
            "ollama": "Ollama (Local / Open-Source)",
            "openai": "OpenAI (Cloud)",
            "mistral": "Mistral AI (European Sovereign)",
        }
        
        provider_options = list(provider_display_names.keys())
        default_provider_idx = 0
        
        selected_provider = st.selectbox(
            "AI Provider",
            options=provider_options,
            format_func=lambda x: provider_display_names.get(x, x),
            index=default_provider_idx,
            key="rag_provider_select",
            help="Choose between sovereign local open-source models (Ollama) or managed cloud models."
        )

        api_key = None
        base_url = None
        selected_model = ""
        selected_embedding_model = ""

        # --- 1. OLLAMA (LOCAL OPEN SOURCE) ---
        if selected_provider == "ollama":
            st.caption("100% Offline, HIPAA/GDPR private inference. Zero telemetry.")
            base_url = st.text_input(
                "Ollama Host Endpoint",
                value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                key="rag_ollama_base_url",
                help="URL of the local or network Ollama service."
            )

            # Discover local Ollama models dynamically
            ollama_cfg = ProviderConfig(provider=ProviderType.OLLAMA, base_url=base_url)
            ollama_llm = ConnectorFactory.get_llm_connector(ollama_cfg)
            discovered_models = ollama_llm.get_available_models()

            ollama_model_options = discovered_models if discovered_models else [
                "llama3.2:latest",
                "llama3.1:8b",
                "mistral:latest",
                "deepseek-r1:8b",
                "qwen2.5:7b"
            ]

            selected_model = st.selectbox(
                "Chat Model",
                options=ollama_model_options,
                index=0,
                key="rag_ollama_model_select",
                help="Local model for question answering and taxonomy generation."
            )

            ollama_embed_options = [
                "nomic-embed-text:latest",
                "bge-m3:latest",
                "all-minilm:latest",
                "mxbai-embed-large:latest",
            ]
            selected_embedding_model = st.selectbox(
                "Local Embedding Model",
                options=ollama_embed_options,
                index=0,
                key="rag_ollama_embed_select",
                help="Local vector embedding model (must be pulled in Ollama: 'ollama pull <model>')."
            )

        # --- 2. GOOGLE GEMINI ---
        elif selected_provider == "gemini":
            env_api_key = os.getenv("GOOGLE_API_KEY", "")
            api_key = st.text_input(
                "Google GenAI API Key",
                value=env_api_key,
                type="password",
                help="Required for Gemini inference.",
                key="rag_gemini_api_key"
            )
            if api_key:
                st.session_state.rag_manager.api_key = api_key
                os.environ["GOOGLE_API_KEY"] = api_key

            gemini_model_options = [
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemini-1.5-pro",
                "gemini-flash-latest",
            ]
            selected_model = st.selectbox(
                "Gemini Model",
                options=gemini_model_options,
                index=0,
                key="rag_gemini_model_select"
            )

            gemini_embed_options = [
                "models/text-embedding-004",
                "models/gemini-embedding-001",
            ]
            selected_embedding_model = st.selectbox(
                "Embedding Model",
                options=gemini_embed_options,
                index=0,
                key="rag_gemini_embed_select"
            )

        # --- 3. OPENAI ---
        elif selected_provider == "openai":
            env_api_key = os.getenv("OPENAI_API_KEY", "")
            api_key = st.text_input(
                "OpenAI API Key",
                value=env_api_key,
                type="password",
                help="Required for OpenAI inference.",
                key="rag_openai_api_key"
            )
            if api_key:
                st.session_state.rag_manager.api_key = api_key
                os.environ["OPENAI_API_KEY"] = api_key

            openai_model_options = [
                "gpt-4o-mini",
                "gpt-4o",
                "o3-mini",
            ]
            selected_model = st.selectbox(
                "OpenAI Model",
                options=openai_model_options,
                index=0,
                key="rag_openai_model_select"
            )

            openai_embed_options = [
                "text-embedding-3-small",
                "text-embedding-3-large",
            ]
            selected_embedding_model = st.selectbox(
                "Embedding Model",
                options=openai_embed_options,
                index=0,
                key="rag_openai_embed_select"
            )

        # --- 4. MISTRAL AI ---
        elif selected_provider == "mistral":
            env_api_key = os.getenv("MISTRAL_API_KEY", "")
            api_key = st.text_input(
                "Mistral API Key",
                value=env_api_key,
                type="password",
                help="Required for Mistral AI inference.",
                key="rag_mistral_api_key"
            )
            if api_key:
                st.session_state.rag_manager.api_key = api_key
                os.environ["MISTRAL_API_KEY"] = api_key

            mistral_model_options = [
                "mistral-small-latest",
                "mistral-large-latest",
                "codestral-latest",
            ]
            selected_model = st.selectbox(
                "Mistral Model",
                options=mistral_model_options,
                index=0,
                key="rag_mistral_model_select"
            )

            mistral_embed_options = ["mistral-embed"]
            selected_embedding_model = st.selectbox(
                "Embedding Model",
                options=mistral_embed_options,
                index=0,
                key="rag_mistral_embed_select"
            )

        st.markdown("---")
        st.markdown("##### Document Library")

        # Document Upload
        uploaded_files = st.file_uploader(
            "Upload Literature (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            key="rag_file_uploader",
            help="PDF files will be stored in DOCUMENTS and ingested into the knowledge base."
        )
        if uploaded_files:
            docs_dir = "DOCUMENTS"
            os.makedirs(docs_dir, exist_ok=True)
            new_files_count = 0
            for uploaded_file in uploaded_files:
                file_path = os.path.join(docs_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                new_files_count += 1
            if new_files_count > 0:
                st.success(f"Stored {new_files_count} document(s). Click Initialize below to index.")

        # Document Selection
        selected_files = None
        docs_dir = "DOCUMENTS"
        if os.path.exists(docs_dir):
            all_files = [f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")]
            if all_files:
                with st.expander(f"Select Documents ({len(all_files)} available)", expanded=False):
                    selected_files = st.multiselect(
                        "Active Documents for Ingestion",
                        options=all_files,
                        default=all_files,
                        label_visibility="collapsed",
                        key="rag_selected_files_multiselect"
                    )
            else:
                st.info("No PDF files currently stored in DOCUMENTS folder.")

        st.markdown("---")
        st.markdown("##### Retrieval Parameters")

        # Adherence Slider
        if "rag_adherence_score" not in st.session_state:
            st.session_state.rag_adherence_score = 0.5

        adherence_score = st.slider(
            "Adherence: Medical Knowledge vs Documents",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.rag_adherence_score,
            step=0.1,
            help="0.0 = General Medical Knowledge. 1.0 = Strict Document Factuality.",
            key="rag_adherence_slider"
        )
        if adherence_score != st.session_state.rag_adherence_score:
            st.session_state.rag_adherence_score = adherence_score
            if st.session_state.rag_manager.initialized:
                success, msg = st.session_state.rag_manager.update_adherence_score(adherence_score)
                if success:
                    st.toast(msg)
                else:
                    st.error(msg)

        # Temperature Slider
        if "rag_temperature" not in st.session_state:
            st.session_state.rag_temperature = 0.3

        temperature = st.slider(
            "Creativity (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.rag_temperature,
            step=0.1,
            help="0.0 = Deterministic. 1.0 = Highly Creative.",
            key="rag_temp_slider"
        )
        if temperature != st.session_state.rag_temperature:
            st.session_state.rag_temperature = temperature

        use_existing_db = st.checkbox(
            "Use Existing Database (Fast Load)",
            value=True,
            help="Reuse already computed vector embeddings to skip re-indexing."
        )

        if st.button("Initialize / Reindex RAG System", key="rag_init_button"):
            with st.spinner("Initializing AI Engine..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(percent: int, message: str):
                    progress_bar.progress(percent)
                    status_text.text(message)

                dataset_columns = None
                if "data" in st.session_state and st.session_state.data is not None:
                    dataset_columns = st.session_state.data.columns.tolist()
                elif "working_df" in st.session_state and st.session_state.working_df is not None:
                    dataset_columns = st.session_state.working_df.columns.tolist()

                # Set API key on manager if provided
                if api_key:
                    st.session_state.rag_manager.api_key = api_key

                success, msg = st.session_state.rag_manager.initialize_system(
                    provider=selected_provider,
                    model_name=selected_model,
                    embedding_model=selected_embedding_model,
                    base_url=base_url,
                    adherence_score=adherence_score,
                    temperature=temperature,
                    dataset_columns=dataset_columns,
                    use_existing_db=use_existing_db,
                    progress_callback=update_progress,
                    selected_files=selected_files,
                )

                progress_bar.empty()
                status_text.empty()

                if success:
                    st.success(msg)
                else:
                    st.error(msg)

        # Status and Context Summary
        if st.session_state.rag_manager.initialized:
            st.success("RAG System Active")
            if hasattr(st.session_state.rag_manager, "global_context"):
                gc = st.session_state.rag_manager.global_context
                if isinstance(gc, dict):
                    st.markdown("---")
                    st.markdown("##### Inferred Clinical Context")
                    st.write(f"**Domain:** {gc.get('domain', 'General Medical')}")
                    if gc.get("context_description"):
                        st.caption(gc.get("context_description"))
        else:
            st.info("RAG System Inactive (Initialize above)")
