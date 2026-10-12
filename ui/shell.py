"""The application shell: sidebar with the dataset card, AI status, handbook link and user menu."""

from __future__ import annotations

import streamlit as st

from ui.auth import LOGO, logout
from ui.components import confirm, page_link
from ui.state import AppState

ORIGIN_LABELS = {"upload": "imported file", "saved": "saved version", "sample": "sample cohort", "path": "server file"}


@st.dialog("Change dataset")
def _change_dialog(state: AppState) -> None:
    versions = state.saved_versions()
    if versions:
        chosen = st.selectbox("Saved versions in your workspace", versions, help="Newest first.")
        if st.button("Load this version", type="primary", width="stretch"):
            ok, msg = state.load_saved_version(chosen)
            if ok:
                st.rerun()
            st.error(msg)
    else:
        st.caption("No saved versions yet.")
    st.divider()
    page_link("import", "Import a new file", icon=":material/upload_file:")
    if st.button("Load the sample cohort", width="stretch"):
        state.load_sample_cohort()
        st.rerun()


@st.dialog("Save a version")
def _save_dialog(state: AppState) -> None:
    ref = state.dataset
    name = st.text_input("Dataset name", value=ref.name if ref else "cohort",
                         help="Versions are numbered automatically (name_v1, name_v2, ...).")
    st.caption("The current working dataset (with renames, enrichments and filters applied) is saved as Parquet in your workspace.")
    if st.button("Save", type="primary", width="stretch"):
        ok, msg, saved = state.save_version(name)
        if ok:
            st.success(f"Saved as {saved}")
            st.rerun()
        st.error(msg)


def _dataset_card(state: AppState) -> None:
    with st.container(border=True):
        ref = state.dataset
        df = state.df
        if ref is None or df is None:
            st.markdown("**No dataset loaded**")
            st.caption("Import a file or reopen a saved version.")
            page_link("import", "Import", icon=":material/upload_file:", width="stretch")
            if st.button("Load the sample cohort", key="shell_load_sample", width="stretch"):
                state.load_sample_cohort()
                st.rerun()
            return
        title_col, badge_col = st.columns([3, 1], vertical_alignment="center")
        title_col.markdown(f"**{ref.name}**")
        if ref.version:
            badge_col.badge(f"v{ref.version}", color="blue")
        st.caption(f"{len(df):,} rows x {df.shape[1]} columns - {ORIGIN_LABELS.get(ref.origin, ref.origin)}")
        col_change, col_save = st.columns(2)
        if col_change.button("Change", key="shell_change_dataset", width="stretch"):
            _change_dialog(state)
        if col_save.button("Save version", key="shell_save_version", width="stretch"):
            _save_dialog(state)


def _ai_status(state: AppState) -> None:
    status = state.ai_status()
    provider = status["provider"] or "ollama"
    if status["initialized"]:
        local = provider == "ollama"
        st.badge(f"AI: {provider} ({'local' if local else 'cloud'})", icon=":material/smart_toy:", color="green" if local else "orange")
    else:
        st.badge("AI engine: off", icon=":material/smart_toy:", color="grey")
    page_link("settings", "Settings", icon=":material/settings:")


def _user_menu(state: AppState) -> None:
    st.caption(f"Signed in as **{state.username}** ({state.role or 'researcher'})")
    if st.button("Log out", key="shell_logout", width="stretch"):
        logout()


def render_sidebar(state: AppState) -> None:
    with st.sidebar:
        try:
            st.image(LOGO, width=150)
        except Exception:
            pass
        _dataset_card(state)
        _ai_status(state)
        page_link("handbook", "Handbook", icon=":material/menu_book:")
        st.divider()
        _user_menu(state)


def clear_workspace_statistics(state: AppState) -> int:
    """Delete this user's cached statistics files; returns how many were removed."""
    import glob
    import os

    removed = 0
    for path in glob.glob(os.path.join(state.db_manager.stats_dir, "stats_*.parquet")):
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass
    return removed


__all__ = ["clear_workspace_statistics", "confirm", "render_sidebar"]
