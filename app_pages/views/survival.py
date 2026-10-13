"""Survival & regression page: Kaplan-Meier, log-rank, Cox, logistic and count regression."""

import streamlit as st

from app_pages.analysis_survival import run_survival
from ui.components import no_dataset, page_header
from utils.data_analyzer import DataAnalyzer

page_header(
    "Survival & regression",
    "Time-to-event curves and hazard ratios, adjusted odds ratios and incidence-rate ratios.",
    chapter="05_Epidemiology",
)
df = st.session_state.get("data")
if df is None:
    no_dataset()
else:
    run_survival(df, DataAnalyzer(df))
