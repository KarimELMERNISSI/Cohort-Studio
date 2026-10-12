"""Settings page: AI provider and consent, graph engine, import defaults, workspace maintenance."""

import streamlit as st

from app_pages.rag_sidebar import render_ai_settings
from ui.components import confirm, page_header, section
from ui.shell import clear_workspace_statistics
from ui.state import get_state

state = get_state()
page_header("Settings", "AI provider and data-sharing consent, graph engine, import defaults and workspace maintenance.")

tab_ai, tab_graph, tab_import, tab_workspace = st.tabs(["AI provider", "Graph engine", "Import defaults", "Workspace"])

with tab_ai:
    st.caption("The AI engine is optional. Local Ollama is the default: nothing leaves the server. "
               "Cloud providers require an explicit confirmation of what would be sent.")
    render_ai_settings()

with tab_graph:
    from graph.ui import renderer_selector

    section("Graph rendering engine")
    renderer_selector()
    st.caption("Plotly + NetworkX renders offline. The optional yFiles engine loads its runtime from yworks.com and reports "
               "anonymised graph-structure analytics; see the About page.")

with tab_import:
    section("Defaults applied when a file is imported")
    st.checkbox("Treat dates as day-first (31/12/2024) by default", key="import_default_dayfirst",
                help="European date order. You can still change it per file in the import flow.")
    st.caption("Encoding, delimiter and decimal separator are always detected per file and shown for review.")

with tab_workspace:
    section("Your workspace")
    try:
        st.code(str(state.db_manager.dataset_dir), language=None)
    except Exception:
        pass
    st.caption("Datasets, statistics caches, traces, uploads, taxonomies, knowledge graphs, documents and the vector store "
               "are private to your account.")

    def _clear() -> None:
        n = clear_workspace_statistics(state)
        st.session_state["_settings_notice"] = f"Removed {n} cached statistics file(s)."

    if st.button("Clear my cached statistics", help="Forces a full recomputation of descriptive statistics for saved versions."):
        confirm("Clear cached statistics?", "Cached descriptive statistics of your saved versions will be recomputed on the next visit. "
                "Datasets and traces are not affected.", _clear, confirm_label="Clear")
    if notice := st.session_state.pop("_settings_notice", None):
        st.success(notice)
