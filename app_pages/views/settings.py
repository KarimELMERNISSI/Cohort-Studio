"""Settings page: AI engine and consent, graph engine, import defaults, workspace maintenance."""

import streamlit as st

from ui.components import confirm, page_header
from ui.preferences import save_from
from ui.shell import clear_workspace_statistics
from ui.state import get_state
from utils.optional import extra_installed, missing_message

state = get_state()
page_header("Settings", "AI engine and data-sharing consent, graph engine, import defaults and your workspace.")

tab_ai, tab_graph, tab_import, tab_workspace = st.tabs(
    [":material/smart_toy: AI engine", ":material/hub: Graph engine", ":material/upload_file: Import defaults",
     ":material/folder_shared: Workspace"]
)

with tab_ai:
    if extra_installed("ai"):
        from app_pages.rag_sidebar import render_ai_settings

        render_ai_settings()
    else:  # the AI connectors and vector databases are an optional install (follow-up programme M4)
        st.info(missing_message("The AI engine (variable map, document library, AI suggestions)", "ai"),
                icon=":material/extension:")

with tab_graph, st.container(border=True, width=760):
    from ui.graph_view import renderer_selector

    st.subheader("Graph rendering engine", icon=":material/hub:", anchor=False)
    # kept for the next session ("automatic", the server's default, keeps no preference)
    renderer_selector(explain=True)
    st.caption("Plotly + NetworkX renders offline. The optional yFiles engine loads its runtime from yworks.com and reports "
               "anonymised graph-structure analytics; see the About page.")

with tab_import, st.container(border=True, width=760):
    st.subheader("Defaults applied when a file is imported", icon=":material/upload_file:", anchor=False)
    # kept for the session: Import, the page that reads it, does not show this toggle
    st.toggle("Treat dates as day-first (31/12/2024) by default", key="import_default_dayfirst",
              help="European date order. You can still change it per file in the import flow.", persist_state="session",
              on_change=save_from, args=("import_default_dayfirst", "import_dayfirst"))  # kept for the next session
    st.caption("Encoding, delimiter and decimal separator are always detected per file and shown for review.")

with tab_workspace, st.container(border=True, width=760):
    st.subheader("Your workspace", icon=":material/folder_shared:", anchor=False)
    try:
        st.code(str(state.db_manager.dataset_dir), language=None)
    except Exception:
        pass  # the folder path is informative only: the page works without it
    st.caption("Datasets, statistics caches, traces, uploads, taxonomies, knowledge graphs, documents and the vector store "
               "are private to your account.")

    def _clear() -> None:
        n = clear_workspace_statistics(state)
        st.session_state["_settings_notice"] = f"Removed {n} cached statistics file(s)."

    if st.button("Clear my cached statistics", icon=":material/delete_sweep:",
                 help="Forces a full recomputation of descriptive statistics for saved versions."):
        confirm("Clear cached statistics?", "Cached descriptive statistics of your saved versions will be recomputed on the next visit. "
                "Datasets and traces are not affected.", _clear, confirm_label="Clear")
    if notice := st.session_state.pop("_settings_notice", None):
        st.success(notice, icon=":material/check:")
