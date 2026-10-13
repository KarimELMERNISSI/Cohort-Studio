"""Start page: welcome banner, dataset figures, the three-step workflow, saved versions, sample cohort, AI status."""

import streamlit as st

from ui.components import page_header, page_link, status_chip, switch_to
from ui.components.workflow_map import workflow_map
from ui.state import SAMPLE_DICTIONARY_PATH, get_state
from ui.workflow import BRANCHES
from utils.session_scope import get_current_scope

state = get_state()
username = st.session_state.get("username") or "researcher"
page_header(
    f"Welcome, {username}",
    "Load a cohort, check its quality, then explore and analyse it. Everything you do is traced and reproducible.",
    chapter="00_Researcher_Handbook",
    eyebrow="Start",
    icon=":material/home:",
)

# ---------------------------------------------------------------------------- dataset at a glance
df = state.df
ref = state.dataset
versions = state.saved_versions()
with st.container(horizontal=True, gap="medium"):
    if df is not None and ref is not None:
        missing = float(df.isna().to_numpy().mean() * 100) if df.size else 0.0
        st.metric("Current dataset", ref.label, border=True, icon=":material/dataset:")
        st.metric("Rows", f"{len(df):,}", border=True, icon=":material/table_rows:")
        st.metric("Columns", df.shape[1], border=True, icon=":material/view_column:")
        st.metric("Missing cells", f"{missing:.1f}%", border=True, icon=":material/grid_off:",
                  help="Share of empty cells across the whole table.")
    else:
        st.metric("Current dataset", "None loaded", border=True, icon=":material/dataset:")
    st.metric("Saved versions", len(versions), border=True, icon=":material/inventory_2:")

# ---------------------------------------------------------------------------- workflow
st.subheader("Your workflow", icon=":material/route:", anchor=False)
st.caption("Three core steps take you from a raw file to results. Optional paths branch off whenever they help: "
           "enrich the cohort, map your variables to your field, or learn from your documents.")
has_data = state.has_data
ai_on = state.ai_status()["initialized"]
with st.container(border=True):
    target = workflow_map(has_data=has_data, ai_on=ai_on)
if target:
    switch_to(target)
steps = (
    (1, ":material/upload_file:", "Load a dataset", "Import a CSV, Excel or Parquet file, reopen a saved version, or try the sample cohort.",
     "ok" if has_data else "info", "done" if has_data else "start here", (("import", "Import a file"),)),
    (2, ":material/fact_check:", "Validate the data", "Quality score, missingness pattern, clinical anomalies, inclusion criteria and outliers.",
     "info" if has_data else "neutral", "ready" if has_data else "waiting", (("validate", "Open Validate"),)),
    (3, ":material/science:", "Explore and analyse", "Descriptive statistics, correlations, hypothesis tests, epidemiological measures and survival.",
     "info" if has_data else "neutral", "ready" if has_data else "waiting", (("explore", "Statistics"), ("analyze", "Hypothesis tests"))),
)
for column, (number, icon, title, text, status, status_label, links) in zip(st.columns(3, gap="medium"), steps, strict=True):
    with column, st.container(border=True, height="stretch"):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge(f"Step {number}", color="blue")
            status_chip(status_label, status)
        st.markdown(f"#### {icon} {title}")
        st.caption(text)
        with st.container(horizontal=True):
            for key, label in links:
                page_link(key, label)

# ---------------------------------------------------------------------------- optional paths
st.subheader("Optional paths", icon=":material/alt_route:", anchor=False)
scope = get_current_scope()
docs_dir = scope.documents_dir() if scope else None
n_docs = len(list(docs_dir.glob("*.pdf"))) if docs_dir is not None and docs_dir.exists() else 0
for column, branch in zip(st.columns(len(BRANCHES), gap="medium"), BRANCHES, strict=True):
    with column, st.container(border=True, height="stretch"):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge("Optional", color="gray")
            if branch.needs_ai:
                status_chip("AI engine on" if ai_on else "needs the AI engine", "ok" if ai_on else "caution")
            elif branch.key == "enrich":
                status_chip("ready" if has_data else "load data first", "info" if has_data else "neutral")
        st.markdown(f"#### {branch.icon} {branch.title}")
        st.caption(f"*{branch.when[:1].upper()}{branch.when[1:]}.* {branch.text}")
        if branch.key == "documents":
            st.caption(f":material/picture_as_pdf: {n_docs} PDF(s) in your library")
        with st.container(horizontal=True):
            for key, label in branch.links:
                page_link(key, label)

# ---------------------------------------------------------------------------- versions, sample, AI
st.space("small")
left, right = st.columns([3, 2], gap="medium")

with left, st.container(border=True):
    st.subheader("Saved versions", icon=":material/inventory_2:", anchor=False)
    if versions:
        for saved in versions[:8]:
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(f":material/description: `{saved}`", width="stretch")
                if st.button("Load", key=f"start_load_{saved}", icon=":material/download:"):
                    ok, msg = state.load_saved_version(saved)
                    if ok:
                        st.rerun()
                    st.error(msg)
        if len(versions) > 8:
            st.caption(f"{len(versions) - 8} more in the sidebar's Change dialog.")
    else:
        st.caption("No saved versions in your workspace yet. Versions are created from the sidebar (Save) or when importing.")

with right:
    with st.container(border=True):
        st.subheader("Sample cohort", icon=":material/science:", anchor=False)
        st.caption("500 synthetic patients with a treatment effect, a prognostic score, time-to-event data, "
                   "documented missingness and one data-entry anomaly.")
        with st.container(horizontal=True, vertical_alignment="center"):
            if st.button("Load the sample cohort", key="start_sample", type="primary", icon=":material/play_arrow:"):
                state.load_sample_cohort()
                st.rerun()
            with st.popover("Data dictionary", icon=":material/menu_book:"):
                try:
                    st.markdown(SAMPLE_DICTIONARY_PATH.read_text(encoding="utf-8"))
                except OSError:
                    st.caption("Dictionary not found (run dev_scripts/make_sample_cohort.py).")

    with st.container(border=True):
        st.subheader("AI engine", icon=":material/smart_toy:", anchor=False)
        ai = state.ai_status()
        if ai["initialized"]:
            st.badge(f"Active: {ai['provider']}", icon=":material/check_circle:", color="green")
        else:
            st.badge("Off", icon=":material/radio_button_unchecked:", color="gray")
            st.caption("Optional. Local Ollama by default, so no data leaves the server.")
        page_link("settings", "Configure in Settings", icon=":material/settings:")
