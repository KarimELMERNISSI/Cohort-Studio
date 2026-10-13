"""
AI engine configuration (Settings page).

Provider-agnostic controls for the AI providers (local Ollama by default; Google Gemini, OpenAI and
Mistral as cloud providers that require an explicit data-sharing consent), model and embedding
selection, the document library, retrieval parameters and engine initialisation. Laid out as cards:
provider and models, data sharing and document library on the left; retrieval and the engine on the
right. Widget keys are stable because other pages read them from the session.
"""

import os

import streamlit as st
from dotenv import load_dotenv

from core.workspace import safe_filename
from manage.rag import ConnectorFactory, ProviderConfig, ProviderType
from utils.session_scope import get_current_scope

load_dotenv()

CLOUD_PROVIDERS = {"gemini", "openai", "mistral"}
PROVIDER_NAMES = {
    "ollama": "Ollama (local, open-source)",
    "gemini": "Google Gemini (cloud)",
    "openai": "OpenAI (cloud)",
    "mistral": "Mistral AI (cloud, EU)",
}
OLLAMA_CHAT_FALLBACK = ["llama3.2:latest", "llama3.1:8b", "mistral:latest", "deepseek-r1:8b", "qwen2.5:7b"]
OLLAMA_EMBEDDINGS = ["nomic-embed-text:latest", "bge-m3:latest", "all-minilm:latest", "mxbai-embed-large:latest"]
CLOUD_CATALOG = {
    "gemini": {
        "env": "GOOGLE_API_KEY",
        "label": "Google GenAI API key",
        "chat": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.8-flash", "gemini-3.5-flash", "gemini-flash-latest",
                 "gemini-pro-latest", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        "embed": ["models/gemini-embedding-2", "models/text-embedding-005", "models/text-embedding-004", "models/gemini-embedding-001"],
    },
    "openai": {
        "env": "OPENAI_API_KEY",
        "label": "OpenAI API key",
        "chat": ["gpt-4o-mini", "gpt-4o", "o3-mini"],
        "embed": ["text-embedding-3-small", "text-embedding-3-large"],
    },
    "mistral": {
        "env": "MISTRAL_API_KEY",
        "label": "Mistral API key",
        "chat": ["mistral-small-latest", "mistral-large-latest", "codestral-latest"],
        "embed": ["mistral-embed"],
    },
}


def _documents_dir() -> str:
    """The signed-in user's document library (audit E4); the legacy shared folder outside a session."""
    scope = get_current_scope()
    return str(scope.documents_dir()) if scope else "DOCUMENTS"


@st.cache_data(ttl=60, show_spinner=False)
def _discover_ollama_models(base_url: str) -> list[str]:
    """Ollama model list, probed at most once a minute per endpoint (audit E11)."""
    try:
        cfg = ProviderConfig(provider=ProviderType.OLLAMA, base_url=base_url)
        return list(ConnectorFactory.get_llm_connector(cfg).get_available_models() or [])
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def _discover_cloud_models(_manager, provider: str, api_key: str) -> list[str]:
    """Cloud model list, fetched at most once an hour per key (audit E11)."""
    try:
        return list(_manager.get_available_models(provider) or [])
    except Exception:
        return []


def _api_key_input(env_var: str, label: str, key: str, help_text: str) -> str | None:
    """Session-scoped API key entry.

    The server-side key is never shown or pre-filled and nothing is written to ``os.environ``
    (audit E6). Leaving the field empty falls back to the server-configured key, if any.
    """
    typed = st.text_input(label, value="", type="password", help=help_text, key=key, icon=":material/key:")
    if os.getenv(env_var):
        st.caption("A server-side key is configured; leave this empty to use it.")
    st.session_state.rag_manager.api_key = typed or None
    return typed or None


def _ensure_manager() -> None:
    from manage.rag_manager import RAGManager

    if "rag_manager" not in st.session_state:
        st.session_state.rag_manager = RAGManager()
    elif st.session_state.rag_manager.__class__ is not RAGManager:  # hot-reload safety
        old_key = getattr(st.session_state.rag_manager, "api_key", None)
        st.session_state.rag_manager = RAGManager(api_key=old_key)


# ----------------------------------------------------------------------------- cards

def _provider_card() -> dict:
    """Provider, connection and model choice; returns the engine configuration."""
    cfg: dict = {"api_key": None, "base_url": None}
    with st.container(border=True):
        st.subheader("Provider and models", icon=":material/smart_toy:", anchor=False)
        provider = st.selectbox(
            "AI provider",
            options=list(PROVIDER_NAMES),
            format_func=lambda p: PROVIDER_NAMES.get(p, p),
            index=0,  # local-first default: nothing leaves the server unless a cloud provider is chosen
            key="rag_provider_select",
            help="Ollama runs on this server or your network; the cloud providers need an API key and your consent.",
        )
        cfg["provider"] = provider
        if provider == "ollama":
            st.badge("Local: no data leaves this server", icon=":material/lock:", color="green")
            cfg["base_url"] = st.text_input(
                "Ollama endpoint",
                value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                key="rag_ollama_base_url",
                help="URL of the local or network Ollama service.",
                icon=":material/dns:",
            )
            chat_options = _discover_ollama_models(cfg["base_url"]) or OLLAMA_CHAT_FALLBACK
            embed_options = OLLAMA_EMBEDDINGS
            chat_key, embed_key = "rag_ollama_model_select", "rag_ollama_embed_select"
            chat_help = "Local model for question answering and taxonomy generation."
            embed_help = "Local embedding model (pull it first: 'ollama pull <model>')."
        else:
            catalog = CLOUD_CATALOG[provider]
            st.badge("Cloud: requires consent", icon=":material/cloud:", color="orange")
            cfg["api_key"] = _api_key_input(
                catalog["env"], catalog["label"], f"rag_{provider}_api_key", f"Required for {PROVIDER_NAMES[provider]} inference."
            )
            chat_options = list(catalog["chat"])
            if provider == "gemini" and cfg["api_key"] and hasattr(st.session_state.rag_manager, "get_available_models"):
                for model in _discover_cloud_models(st.session_state.rag_manager, "gemini", cfg["api_key"]):
                    if model not in chat_options:
                        chat_options.append(model)
            embed_options = catalog["embed"]
            chat_key, embed_key = f"rag_{provider}_model_select", f"rag_{provider}_embed_select"
            chat_help = "Model used for question answering and taxonomy generation."
            embed_help = "Embedding model used to index your documents."
        col_chat, col_embed = st.columns(2)
        cfg["model"] = col_chat.selectbox("Chat model", chat_options, index=0, key=chat_key, help=chat_help)
        cfg["embedding"] = col_embed.selectbox("Embedding model", embed_options, index=0, key=embed_key, help=embed_help)
    return cfg


def _consent_card(provider: str) -> bool:
    """Explicit, session-scoped confirmation before any data can be sent to a cloud provider (audit E6)."""
    with st.container(border=True):
        st.subheader("Data sharing", icon=":material/shield_person:", anchor=False)
        if provider not in CLOUD_PROVIDERS:
            st.session_state.rag_manager.cloud_consent = False
            st.caption("The local provider keeps every prompt, document and dataset excerpt on this server.")
            return True
        st.warning(
            "Column names and types, up to 3 sample rows, the most frequent values of each column, your anomaly "
            "criteria, the text of indexed PDFs and your chat messages. Do not enable this for identifiable patient "
            "data unless your data-protection approval covers it.",
            title=f"What leaves this server with {PROVIDER_NAMES[provider]}",
            icon=":material/cloud_upload:",
        )
        consent = st.checkbox(
            "I confirm this data may be sent to the selected cloud provider",
            key="rag_cloud_consent",
            help="Required before any data is sent to a cloud AI provider. Applies to this session only.",
        )
        st.session_state.rag_manager.cloud_consent = bool(consent)
        return bool(consent)


def _library_card() -> list[str] | None:
    """Upload PDFs into the user's document library and choose the ones to index."""
    with st.container(border=True):
        st.subheader("Document library", icon=":material/library_books:", anchor=False)
        uploaded_files = st.file_uploader(
            "Add literature (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            key="rag_file_uploader",
            help="PDF files are stored in your workspace and indexed into your knowledge base.",
        )
        docs_dir = _documents_dir()
        if uploaded_files:
            os.makedirs(docs_dir, exist_ok=True)
            for uploaded_file in uploaded_files:
                with open(os.path.join(docs_dir, safe_filename(uploaded_file.name)), "wb") as handle:
                    handle.write(uploaded_file.getbuffer())
            st.success(f"Stored {len(uploaded_files)} document(s). Initialise the engine to index them.", icon=":material/check:")
        all_files = sorted(f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")) if os.path.exists(docs_dir) else []
        if not all_files:
            st.caption("No PDF in your library yet. Indexed papers power the Documents page and the AI suggestions.")
            return None
        return st.multiselect(
            f"Documents to index ({len(all_files)} in your library)",
            options=all_files,
            default=all_files,
            key="rag_selected_files_multiselect",
            help="Only the selected documents are embedded when the engine is initialised.",
        )


def _retrieval_card() -> tuple[float, float]:
    with st.container(border=True):
        st.subheader("Retrieval", icon=":material/tune:", anchor=False)
        st.session_state.setdefault("rag_adherence_score", 0.5)
        st.session_state.setdefault("rag_temperature", 0.3)
        adherence = st.slider(
            "Adherence to your documents",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.rag_adherence_score,
            step=0.1,
            key="rag_adherence_slider",
            help="0 = general medical knowledge, 1 = answers strictly grounded in your documents.",
        )
        if adherence != st.session_state.rag_adherence_score:
            st.session_state.rag_adherence_score = adherence
            if st.session_state.rag_manager.initialized:
                ok, msg = st.session_state.rag_manager.update_adherence_score(adherence)
                (st.toast if ok else st.error)(msg)
        temperature = st.slider(
            "Creativity (temperature)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.rag_temperature,
            step=0.1,
            key="rag_temp_slider",
            help="0 = deterministic answers, 1 = more varied wording.",
        )
        st.session_state.rag_temperature = temperature
    return adherence, temperature


def _engine_card(cfg: dict, consent: bool, adherence: float, temperature: float, selected_files: list[str] | None) -> None:
    manager = st.session_state.rag_manager
    with st.container(border=True):
        st.subheader("AI engine", icon=":material/bolt:", anchor=False)
        if manager.initialized:
            st.badge("Active", icon=":material/check_circle:", color="green")
        else:
            st.badge("Not initialised", icon=":material/radio_button_unchecked:", color="gray")
        use_existing_db = st.toggle(
            "Reuse the existing index",
            value=True,
            key="rag_use_existing_db",
            help="Skips re-embedding documents that are already indexed (faster).",
        )
        if not consent:
            st.caption("Confirm the data-sharing notice to enable initialisation.")
        if st.button(
            "Initialise / reindex",
            key="rag_init_button",
            type="primary",
            width="stretch",
            icon=":material/rocket_launch:",
            disabled=not consent,
        ):
            with st.status("Initialising the AI engine", expanded=True) as status:
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(percent: int, message: str) -> None:
                    progress_bar.progress(percent)
                    status_text.caption(message)

                dataset = st.session_state.get("data")
                if dataset is None:
                    dataset = st.session_state.get("working_df")
                if cfg.get("api_key"):
                    manager.api_key = cfg["api_key"]
                success, msg = manager.initialize_system(
                    provider=cfg["provider"],
                    model_name=cfg["model"],
                    embedding_model=cfg["embedding"],
                    base_url=cfg.get("base_url"),
                    adherence_score=adherence,
                    temperature=temperature,
                    dataset_columns=dataset.columns.tolist() if dataset is not None else None,
                    use_existing_db=use_existing_db,
                    progress_callback=update_progress,
                    selected_files=selected_files,
                )
                status.update(label=msg if success else "Initialisation failed", state="complete" if success else "error")
                if not success:
                    st.error(msg)
        context = getattr(manager, "global_context", None) if manager.initialized else None
        if isinstance(context, dict):
            st.caption("Inferred clinical context")
            st.markdown(f"**{context.get('domain', 'General medical')}**")
            if context.get("context_description"):
                st.caption(context["context_description"])


def render_ai_settings() -> None:
    """AI provider, keys, consent, document library, retrieval parameters and engine (Settings page)."""
    _ensure_manager()
    left, right = st.columns([3, 2], gap="large")
    with left:
        cfg = _provider_card()
        consent = _consent_card(cfg["provider"])
        selected_files = _library_card()
    with right:
        adherence, temperature = _retrieval_card()
        _engine_card(cfg, consent, adherence, temperature, selected_files)


__all__ = ["CLOUD_PROVIDERS", "render_ai_settings"]
