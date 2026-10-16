"""Variable map page: taxonomy of variables, formulas, clustering and the knowledge graph."""

import streamlit as st

from app_pages.data_insight import app
from ui.components import page_header
from utils.optional import extra_installed, missing_message

page_header("Variable map", "Taxonomy of your variables, derived formulas, clustering and the knowledge graph.",
            chapter="06_Data_Insight")
if not extra_installed("ai"):
    st.info(missing_message("Building the taxonomy of the variables with the AI engine", "ai"),
            icon=":material/extension:")
app()
