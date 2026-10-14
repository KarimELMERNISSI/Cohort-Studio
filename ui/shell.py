"""The application shell: brand logo, and a sidebar with the dataset card, AI status and user menu."""

from __future__ import annotations

import streamlit as st

from manage.iam.authz import is_admin_role
from ui.auth import logout
from ui.components import confirm, page_link
from ui.components.population import render_population_selector
from ui.state import AppState

ORIGIN_LABELS = {
    "upload": ("imported file", ":material/upload_file:"),
    "saved": ("saved version", ":material/inventory_2:"),
    "sample": ("sample cohort", ":material/science:"),
    "path": ("server file", ":material/folder_open:"),
    "replay": ("replayed trace", ":material/history:"),
}


@st.dialog("Change dataset", icon=":material/swap_horiz:")
def _change_dialog(state: AppState) -> None:
    versions = state.saved_versions()
    if versions:
        chosen = st.selectbox("Saved versions in your workspace", versions, help="Newest first.")
        if st.button(
            "Load this version", type="primary", width="stretch", icon=":material/download:"
        ):
            ok, msg = state.load_saved_version(chosen)
            if ok:
                st.rerun()
            st.error(msg)
    else:
        st.caption("No saved versions yet.")
    st.space("small")
    with st.container(horizontal=True):
        page_link("import", "Import a new file", icon=":material/upload_file:")
        if st.button("Load the sample cohort", icon=":material/science:"):
            state.load_sample_cohort()
            st.rerun()


@st.dialog("Save a version", icon=":material/save:")
def _save_dialog(state: AppState) -> None:
    ref = state.dataset
    name = st.text_input(
        "Dataset name",
        value=ref.name if ref else "cohort",
        help="Versions are numbered automatically (name_v1, name_v2, ...).",
    )
    st.caption(
        "The current working dataset (with renames, enrichments and filters applied) is saved as Parquet in your workspace."
    )
    if st.button("Save", type="primary", width="stretch", icon=":material/save:"):
        ok, msg, saved = state.save_version(name)
        if ok:
            st.success(f"Saved as {saved}")
            st.rerun()
        st.error(msg)


def _dataset_card(state: AppState) -> None:
    with st.container(border=True, key="cmp-dataset-card"):
        st.caption("Current dataset")
        ref = state.dataset
        df = state.df
        if ref is None or df is None:
            st.markdown("**:material/dataset: No dataset loaded**")
            page_link("import", "Import a file", icon=":material/upload_file:", width="stretch")
            if st.button(
                "Load the sample cohort",
                key="shell_load_sample",
                width="stretch",
                icon=":material/science:",
            ):
                state.load_sample_cohort()
                st.rerun()
            return
        st.markdown(f"**:material/dataset: {ref.name}**")
        origin_label, origin_icon = ORIGIN_LABELS.get(
            ref.origin, (ref.origin, ":material/description:")
        )
        with st.container(horizontal=True, gap="small"):
            if ref.version:
                st.badge(f"v{ref.version}", color="primary")
            st.badge(origin_label, icon=origin_icon, color="gray")
        st.caption(f"{len(df):,} rows · {df.shape[1]} columns")
        draft = state.staging_status()
        if draft.n_steps:
            st.badge(
                f"Enrich draft: {draft.n_steps} change{'s' if draft.n_steps != 1 else ''}"
                + ("" if draft.can_apply else " (outdated)"),
                icon=":material/edit_note:",
                color="blue" if draft.can_apply else "orange",
            )
        render_population_selector(state)
        with st.container(horizontal=True, gap="small"):
            if st.button(
                "Change", key="shell_change_dataset", width="stretch", icon=":material/swap_horiz:"
            ):
                _change_dialog(state)
            if st.button("Save", key="shell_save_version", width="stretch", icon=":material/save:"):
                _save_dialog(state)


def _ai_status(state: AppState) -> None:
    status = state.ai_status()
    provider = status["provider"] or "ollama"
    if status["initialized"]:
        local = provider == "ollama"
        st.badge(
            f"AI engine: {provider} ({'local' if local else 'cloud'})",
            icon=":material/smart_toy:",
            color="green" if local else "orange",
        )
    else:
        st.badge("AI engine off", icon=":material/smart_toy:", color="gray")


def _user_menu(state: AppState) -> None:
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown(f":material/account_circle: **{state.username}**")
        admin = is_admin_role(state.role)
        st.badge(
            "admin" if admin else (state.role or "researcher"), color="violet" if admin else "gray"
        )
    if st.button("Log out", key="shell_logout", type="tertiary", icon=":material/logout:"):
        logout()


def render_sidebar(state: AppState) -> None:
    with st.sidebar:
        _dataset_card(state)
        _ai_status(state)
        page_link("settings", "Settings", icon=":material/settings:")
        page_link("handbook", "Handbook", icon=":material/menu_book:")
        st.space("medium")
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
