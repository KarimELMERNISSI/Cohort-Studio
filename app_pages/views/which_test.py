"""Which test? Four questions -> the recommended analysis, why, and where to run it."""

import streamlit as st

from ui.components import glossary_popover, page_header, page_link, section, status_chip
from ui.guidance import DESIGNS, DISTRIBUTIONS, GROUPS, OUTCOMES, recommend

page_header("Which test?", "Answer four questions to get the recommended analysis, the effect size to report and where to run it.",
            chapter="05_Epidemiology")

left, right = st.columns([2, 3], gap="large")
with left:
    section("Your question")
    outcome = st.selectbox("What kind of outcome are you comparing?", OUTCOMES, key="chooser_outcome",
                           help="Continuous: a measurement (e.g. blood pressure). Binary: yes/no. Ordinal: ordered levels (e.g. stage). "
                                "Time to event: follow-up with censoring.")
    groups = st.selectbox("How many groups?", GROUPS, key="chooser_groups",
                          help="Groups are the levels of the factor you compare (e.g. treatment arms).")
    design = st.selectbox("How were the groups formed?", DESIGNS, key="chooser_design",
                          help="Paired: the same subjects measured twice (before/after, two methods on the same sample).")
    distribution = st.selectbox("What does the outcome look like?", DISTRIBUTIONS, key="chooser_distribution",
                                help="Skewed outcomes, outliers or very small groups favour rank-based tests. "
                                     "Welch procedures already tolerate unequal variances.")
    glossary_popover()

with right:
    section("Recommendation")
    rec = recommend(outcome, groups, design, distribution)
    with st.container(border=True):
        head, chip = st.columns([4, 1], vertical_alignment="center")
        head.markdown(f"**{rec.test}**")
        with chip:
            status_chip("available" if rec.available else "planned", "ok" if rec.available else "caution")
        st.markdown(f"**Why:** {rec.why}")
        st.markdown(f"**Report:** {rec.effect_size}")
        st.markdown(f"**Where:** {rec.where}")
        for caveat in rec.caveats:
            st.caption(caveat)
        if rec.alternatives:
            st.caption("Alternatives: " + "; ".join(rec.alternatives))
        if rec.available:
            page_link(rec.page, "Open the analysis page", icon=":material/science:")
    st.caption("Assumption checks are shown as diagnostics next to every result; they never replace the choice above.")
