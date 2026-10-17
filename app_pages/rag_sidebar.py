"""
AI engine configuration (Settings page).

Provider-agnostic controls for the AI providers (local Ollama by default; Google Gemini, OpenAI, Mistral
and Anthropic Claude as cloud providers that require an explicit data-sharing consent; Claude has no
embedding model, so another provider embeds the documents), model and embedding
selection, the document library, retrieval parameters and engine initialisation. Laid out as cards:
provider and models, data sharing and document library on the left; retrieval, the vector database
(LanceDB or Qdrant, see ``manage.vectorstores``) and the engine on the right. Widget keys are stable
because other pages read them from the session.
"""

import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from core.workspace import remove_pdf, safe_filename
from manage.rag import ConnectorFactory, ProviderConfig, ProviderType
from manage.vectorstores import (
    ENGINE_SUMMARIES,
    ENGINE_TITLES,
    IndexManifest,
    MachineProfile,
    VectorEngine,
    VectorStoreSettings,
    estimate_chunks,
    probe_machine,
    qdrant_embedded_warnings,
    recommend,
    resolve_engine,
)
from manage.vectorstores.qdrant_index import QdrantIndex, user_collection
from ui.components import confirm
from utils.data_paths import get_vectors_dir
from utils.session_scope import get_current_scope

load_dotenv()

CLOUD_PROVIDERS = {"gemini", "openai", "mistral", "anthropic"}
PROVIDER_NAMES = {
    "ollama": "Ollama (local, open-source)",
    "gemini": "Google Gemini (cloud)",
    "openai": "OpenAI (cloud)",
    "mistral": "Mistral AI (cloud, EU)",
    "anthropic": "Anthropic Claude (cloud)",
}
# Claude answers but has no embedding model: these providers can embed the documents for it.
EMBEDDING_PROVIDERS = ("ollama", "gemini", "openai", "mistral")
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
    "anthropic": {
        "env": "ANTHROPIC_API_KEY",
        "label": "Anthropic API key",
        "chat": ["claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"],
        "embed": [],  # no embedding API: see EMBEDDING_PROVIDERS
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
            if provider in ("gemini", "anthropic") and cfg["api_key"] and hasattr(
                st.session_state.rag_manager, "get_available_models"
            ):
                for model in _discover_cloud_models(st.session_state.rag_manager, provider, cfg["api_key"]):
                    if model not in chat_options:
                        chat_options.append(model)
            embed_options = catalog["embed"]
            chat_key, embed_key = f"rag_{provider}_model_select", f"rag_{provider}_embed_select"
            chat_help = "Model used for question answering and taxonomy generation."
            embed_help = "Embedding model used to index your documents."
            if provider == "anthropic":
                chat_help += " The most capable model is listed first; the app's temperature setting does not apply to Claude."
                embed_options, embed_help = _claude_embedding_block(cfg)
                embed_key = f"rag_anthropic_{cfg['embedding_provider']}_embed_select"
        col_chat, col_embed = st.columns(2)
        cfg["model"] = col_chat.selectbox("Chat model", chat_options, index=0, key=chat_key, help=chat_help)
        cfg["embedding"] = col_embed.selectbox("Embedding model", embed_options, index=0, key=embed_key, help=embed_help)
    return cfg


def _claude_embedding_block(cfg: dict) -> tuple[list[str], str]:
    """Claude has no embedding model: choose the provider that embeds the documents (local Ollama by default)."""
    st.caption(
        "Claude answers; another provider embeds your documents for search. With local Ollama the whole library "
        "is indexed on this server, and only the passages retrieved for a question are sent to Claude."
    )
    embed_provider = st.selectbox(
        "Embedding provider",
        EMBEDDING_PROVIDERS,
        format_func=lambda p: PROVIDER_NAMES.get(p, p),
        index=0,
        key="rag_claude_embed_provider",
        help="Who turns your documents into vectors for search. Ollama keeps the documents on this server; a cloud "
        "provider needs its own API key and receives the text of the indexed PDFs.",
    )
    cfg["embedding_provider"] = embed_provider
    if embed_provider == "ollama":
        cfg["embedding_base_url"] = st.text_input(
            "Ollama endpoint (embeddings)",
            value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            key="rag_claude_ollama_base_url",
            help="URL of the local or network Ollama service that embeds the documents.",
            icon=":material/dns:",
        )
        return OLLAMA_EMBEDDINGS, "Local embedding model (pull it first: 'ollama pull <model>')."
    catalog = CLOUD_CATALOG[embed_provider]
    typed = st.text_input(
        f"{catalog['label']} (embeddings)",
        value="",
        type="password",
        key=f"rag_claude_{embed_provider}_embed_key",
        icon=":material/key:",
        help=f"Key for {PROVIDER_NAMES[embed_provider]} embeddings. Never stored; empty uses the server key if configured.",
    )
    if os.getenv(catalog["env"]):
        st.caption("A server-side key is configured for the embeddings; leave this empty to use it.")
    cfg["embedding_api_key"] = typed or None
    return list(catalog["embed"]), f"{PROVIDER_NAMES[embed_provider]} embedding model used to index your documents."


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
    """Upload PDFs into the user's document library, choose the ones to index and remove the others (the
    index is rebuilt from the remaining documents at the next initialisation: ``manage.vectorstores.index_matches``)."""
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
        key = "rag_selected_files_multiselect"
        selected = st.session_state.setdefault(key, all_files)
        if any(f not in all_files for f in selected):
            st.session_state[key] = [f for f in selected if f in all_files]  # a removed document leaves the choice
        chosen = st.multiselect(
            f"Documents to index ({len(all_files)} in your library)",
            options=all_files,
            key=key,
            help="Only the selected documents are embedded when the engine is initialised.",
        )
        with st.expander("Remove a document"):
            for name in all_files:
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.markdown(f":material/picture_as_pdf: {name}", width="stretch")
                    if st.button(
                        "Remove",
                        key=f"rag_remove_document_{name}",
                        icon=":material/delete:",
                        help="Deletes this PDF from your library.",
                    ):
                        confirm(
                            f"Remove {name}?",
                            "Deletes the PDF from your library. The AI engine keeps its current index until it is "
                            "initialised again; the index is then rebuilt from the remaining documents.",
                            lambda name=name: remove_pdf(docs_dir, name),
                            confirm_label="Remove",
                        )
        return chosen


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


@st.cache_data(ttl=300, show_spinner=False)
def _machine(storage: str) -> MachineProfile:
    """CPU, memory and free disk, probed at most every five minutes."""
    return probe_machine(Path(storage))


def _vectors_dir() -> Path:
    scope = get_current_scope()
    return scope.vectors_dir() if scope else Path(get_vectors_dir())


def _vector_card(selected_files: list[str] | None) -> dict:
    """Choice of the vector database for this machine and library; returns the engine settings."""
    settings = VectorStoreSettings.from_env()
    base_dir = _vectors_dir()
    docs_dir = Path(_documents_dir())
    sizes = [(docs_dir / f).stat().st_size for f in (selected_files or []) if (docs_dir / f).exists()]
    built = [m for m in (IndexManifest.load(base_dir / e.value) for e in VectorEngine) if m is not None]
    chunks = max([m.chunks for m in built], default=0) or estimate_chunks(sizes)
    dim = built[0].dim if built else 768
    machine = _machine(str(base_dir))
    with st.container(border=True):
        st.subheader("Vector database", icon=":material/database:", anchor=False)
        st.session_state.setdefault("rag_qdrant_url", settings.qdrant_url or "")
        server_url = (st.session_state.get("rag_qdrant_url") or "").strip()
        advice = recommend(machine, chunks=chunks, dim=dim, qdrant_server=bool(server_url))
        default = resolve_engine(None, settings, advice)
        choice = st.segmented_control(
            "Engine",
            options=[e.value for e in VectorEngine],
            format_func=lambda v: ENGINE_TITLES[VectorEngine(v)],
            default=default.value,
            key="rag_vector_engine",
            help="Where the chunks of your documents and their vectors are stored and searched.",
        )
        engine = VectorEngine(choice or default.value)
        st.caption(ENGINE_SUMMARIES[engine])
        if engine is advice.engine:
            st.badge("Recommended for this machine", icon=":material/recommend:", color="green")
        else:
            st.badge(f"Recommended here: {ENGINE_TITLES[advice.engine]}", icon=":material/info:", color="gray")
        with st.expander("Why, and this machine"):
            for reason in advice.reasons:
                st.markdown(f"- {reason}")
            memory = f"{machine.memory_gb:.0f} GB memory" if machine.memory_gb else "memory unknown"
            if machine.available_gb:
                memory += f" ({machine.available_gb:.1f} GB free)"
            disk = f"{machine.disk_free_gb:,.0f} GB free disk" if machine.disk_free_gb is not None else "free disk unknown"
            source = "current index" if built else "rough estimate from the PDF sizes"
            st.caption(f"{machine.cpu_count} CPU cores · {memory} · {disk} · library: ~{chunks:,} chunks ({source}).")

        qdrant_url = qdrant_key = None
        if engine is VectorEngine.QDRANT:
            server_url = st.text_input(
                "Qdrant server URL",
                key="rag_qdrant_url",
                placeholder="http://localhost:6333 (empty: embedded in your workspace)",
                help="Leave empty for embedded Qdrant in your workspace. A server (Docker, self-hosted or Qdrant Cloud) "
                "is shared by every session, with one collection per user. Default: CMP_QDRANT_URL.",
                icon=":material/dns:",
            ).strip()
            qdrant_url = server_url or None
            if qdrant_url:
                qdrant_key = st.text_input(
                    "Qdrant API key", value="", type="password", key="rag_qdrant_api_key", icon=":material/key:",
                    help="Only for secured servers. Never stored; empty uses CMP_QDRANT_API_KEY when configured.",
                ) or settings.qdrant_api_key
                if st.button("Test the connection", key="rag_qdrant_ping", icon=":material/network_check:"):
                    scope = get_current_scope()
                    probe = QdrantIndex(
                        base_dir / "qdrant", url=qdrant_url, api_key=qdrant_key,
                        collection=user_collection(scope.username if scope else None),
                    )
                    ok, message = probe.ping()
                    (st.success if ok else st.error)(message)
            else:
                for note in qdrant_embedded_warnings(machine, chunks=chunks, dim=dim):
                    st.caption(f":orange[:material/warning:] {note}")
        for warning in advice.warnings:
            st.warning(warning, icon=":material/storage:")

        manifest = IndexManifest.load(base_dir / engine.value)
        if manifest is not None:
            st.caption(
                f":material/inventory_2: Index: {manifest.chunks:,} chunks from {len(manifest.documents)} document(s), "
                f"embedded with {manifest.embedding} ({manifest.dim} dimensions), built {manifest.built_at[:16].replace('T', ' ')}."
            )
        else:
            st.caption(":material/inventory_2: No index for this engine yet: initialise the engine to build it.")

        scope = get_current_scope()
        legacy = scope.legacy_chroma_dir() if scope else None
        if legacy is not None and legacy.is_dir() and not any(legacy.iterdir()):
            legacy.rmdir()  # an empty folder left by earlier versions: nothing to keep
        if legacy is not None and legacy.exists():
            st.info(
                "An index from ChromaDB (earlier versions) is still in your workspace. It is no longer used: "
                "initialise the engine to rebuild the index with the selected database. Your PDFs are kept.",
                icon=":material/history:",
            )
            if st.button("Remove the old ChromaDB index", key="rag_remove_chroma", icon=":material/delete:"):
                confirm(
                    "Remove the old ChromaDB index?",
                    "Deletes the ChromaDB folder of your workspace. Your PDFs and the new index are not affected.",
                    lambda: shutil.rmtree(legacy, ignore_errors=True),
                    confirm_label="Remove",
                )
    return {"vector_engine": engine.value, "qdrant_url": qdrant_url, "qdrant_api_key": qdrant_key}


def _engine_card(
    cfg: dict,
    consent: bool,
    adherence: float,
    temperature: float,
    selected_files: list[str] | None,
    vector: dict,
) -> None:
    manager = st.session_state.rag_manager
    with st.container(border=True):
        st.subheader("AI engine", icon=":material/bolt:", anchor=False)
        if manager.initialized:
            engine_name = ENGINE_TITLES.get(VectorEngine(manager.vector_engine)) if getattr(manager, "vector_engine", None) else None
            st.badge(f"Active · {engine_name} index" if engine_name else "Active", icon=":material/check_circle:", color="green")
        else:
            st.badge("Not initialised", icon=":material/radio_button_unchecked:", color="gray")
        use_existing_db = st.toggle(
            "Reuse the existing index",
            value=True,
            key="rag_use_existing_db",
            help="Uses the index already built with the selected vector database and embedding model (faster), when "
            "it holds the selected documents: after a document is added or removed, the index is rebuilt. Turn it "
            "off after changing the embedding model.",
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
                    embedding_provider=cfg.get("embedding_provider"),
                    embedding_api_key=cfg.get("embedding_api_key"),
                    embedding_base_url=cfg.get("embedding_base_url"),
                    vector_engine=vector.get("vector_engine"),
                    qdrant_url=vector.get("qdrant_url"),
                    qdrant_api_key=vector.get("qdrant_api_key"),
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
        vector = _vector_card(selected_files)
        _engine_card(cfg, consent, adherence, temperature, selected_files, vector)


__all__ = ["CLOUD_PROVIDERS", "render_ai_settings"]
