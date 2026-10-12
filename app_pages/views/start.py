"""Start page: the three-step workflow, saved versions, the sample cohort and the AI status."""

import streamlit as st

from ui.components import page_header, page_link, section, status_chip
from ui.state import SAMPLE_DICTIONARY_PATH, get_state

state = get_state()
page_header("Start", "Load a cohort, check its quality, then explore and analyse it.", chapter="00_Researcher_Handbook")

left, right = st.columns([3, 2], gap="large")

with left:
    section("Your workflow")
    has_data = state.has_data
    steps = (
        ("1. Load a dataset", "Import a CSV, Excel or Parquet file, or reopen a saved version.",
         "ok" if has_data else "neutral", (("import", "Import a file"),)),
        ("2. Validate the data", "Quality score, missingness pattern, anomalies, inclusion criteria and outliers.",
         "info" if has_data else "neutral", (("validate", "Open Validate"),)),
        ("3. Explore and analyse", "Descriptive statistics, correlations, hypothesis tests, epidemiological measures and power.",
         "info" if has_data else "neutral", (("explore", "Statistics & correlations"), ("analyze", "Hypothesis tests"))),
    )
    for title, text, status, links in steps:
        with st.container(border=True):
            head, chip = st.columns([5, 1], vertical_alignment="center")
            head.markdown(f"**{title}**")
            with chip:
                status_chip({"ok": "done", "info": "ready", "neutral": "waiting"}[status], status)
            st.caption(text)
            cols = st.columns(len(links))
            for col, (key, label) in zip(cols, links, strict=True):
                with col:
                    page_link(key, label)
    if state.dataset is not None and state.df is not None:
        ref = state.dataset
        st.info(f"Current dataset: **{ref.label}** - {len(state.df):,} rows x {state.df.shape[1]} columns.")

with right:
    section("Saved versions")
    versions = state.saved_versions()
    if versions:
        for saved in versions[:8]:
            c1, c2 = st.columns([3, 1], vertical_alignment="center")
            c1.markdown(f"`{saved}`")
            if c2.button("Load", key=f"start_load_{saved}", width="stretch"):
                ok, msg = state.load_saved_version(saved)
                if ok:
                    st.rerun()
                st.error(msg)
        if len(versions) > 8:
            st.caption(f"{len(versions) - 8} more in the sidebar's Change dialog.")
    else:
        st.caption("No saved versions in your workspace yet. Versions are created from the sidebar (Save version) or when importing.")

    section("Sample cohort")
    st.caption("500 synthetic patients with a treatment effect, a prognostic score, time-to-event data, documented missingness and one data-entry anomaly.")
    if st.button("Load the sample cohort", key="start_sample", type="primary"):
        state.load_sample_cohort()
        st.rerun()
    with st.expander("Data dictionary"):
        try:
            st.markdown(SAMPLE_DICTIONARY_PATH.read_text(encoding="utf-8"))
        except OSError:
            st.caption("Dictionary not found (run dev_scripts/make_sample_cohort.py).")

    section("AI engine")
    ai = state.ai_status()
    if ai["initialized"]:
        st.caption(f"Active provider: {ai['provider']}.")
    else:
        st.caption("Optional. Off until you choose a provider in Settings (local Ollama by default; no data leaves the server).")
    page_link("settings", "Settings")
