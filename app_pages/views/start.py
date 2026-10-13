"""Start page: welcome banner, dataset figures, the three-step workflow, saved versions, sample cohort, AI status."""

import streamlit as st

from core.progress import DONE, STALE
from ui.components import page_header, page_link, status_chip, switch_to
from ui.components.workflow_map import workflow_map
from ui.progress import current_progress
from ui.state import SAMPLE_DICTIONARY_PATH, get_state
from ui.workflow import BRANCHES

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
progress = current_progress()  # ticks, "run again" badges and the suggested next step
with st.container(border=True):
    target = workflow_map(progress=progress, ai_on=ai_on)
if target:
    switch_to(target)


def progress_chip(step: str) -> None:
    """The step's status for the current data, as on the map."""
    state_ = progress.state(step)
    if state_ == DONE:
        status_chip("done", "ok")
    elif state_ == STALE:
        status_chip("run again", "caution")
    elif step == progress.next_step:
        status_chip("suggested next" if has_data else "start here", "info")
    else:
        status_chip("to do" if has_data else "waiting", "neutral")


steps = (
    (1, "load", ":material/upload_file:", "Load a dataset", "Import a CSV, Excel or Parquet file, reopen a saved version, or try the sample cohort.",
     (("import", "Import a file"),)),
    (2, "validate", ":material/fact_check:", "Validate the data", "Quality score, missingness pattern, clinical anomalies, inclusion criteria and outliers.",
     (("validate", "Open Validate"),)),
    (3, "analyse", ":material/science:", "Explore and analyse", "Descriptive statistics, correlations, hypothesis tests, epidemiological measures and survival.",
     (("explore", "Statistics"), ("analyze", "Hypothesis tests"))),
)
for column, (number, step, icon, title, text, links) in zip(st.columns(3, gap="medium"), steps, strict=True):
    with column, st.container(border=True, height="stretch"):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge(f"Step {number}", color="blue")
            progress_chip(step)
        st.markdown(f"#### {icon} {title}")
        st.caption(text)
        st.caption(f":material/history: {progress.text(step)}")
        with st.container(horizontal=True):
            for key, label in links:
                page_link(key, label)

# ---------------------------------------------------------------------------- optional paths
st.subheader("Optional paths", icon=":material/alt_route:", anchor=False)
BRANCH_STEP = {"enrich": "enrich", "variable_map": "varmap", "documents": "docs", "reproduce": "report"}
for column, branch in zip(st.columns(len(BRANCHES), gap="medium"), BRANCHES, strict=True):
    step = BRANCH_STEP[branch.key]
    with column, st.container(border=True, height="stretch"):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge("Optional", color="gray")
            if branch.needs_ai:
                status_chip("AI engine on" if ai_on else "needs the AI engine", "ok" if ai_on else "caution")
            if progress.state(step) in (DONE, STALE) or (has_data and step == progress.next_step):
                progress_chip(step)
            elif not branch.needs_ai:
                status_chip("ready" if has_data else "load data first", "info" if has_data else "neutral")
        st.markdown(f"#### {branch.icon} {branch.title}")
        st.caption(f"*{branch.when[:1].upper()}{branch.when[1:]}.* {branch.text}")
        if progress.text(step):
            st.caption(f":material/history: {progress.text(step)}")
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
