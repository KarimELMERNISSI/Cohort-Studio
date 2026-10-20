"""Survival & regression page: Kaplan-Meier, log-rank, Cox, logistic and count regression."""

import streamlit as st

from app_pages.analysis_survival import run_survival
from ui.components import no_dataset, page_header
from ui.components.mi_mode import mi_banner
from ui.components.population import population_banner
from ui.state import get_state

page_header(
    "Survival & regression",
    "Time-to-event curves and hazard ratios, adjusted odds ratios and incidence-rate ratios.",
    chapter="05_Epidemiology",
)
population_banner()
df = st.session_state.get("data")
if df is None:
    no_dataset()
else:
    state = get_state()
    run_survival(df, state.analyzer(), mi=mi_banner(state))
