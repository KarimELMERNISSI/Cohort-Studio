"""Documents page: literature knowledge graph and document chat."""

import streamlit as st

from app_pages.document_insight import app
from ui.components import page_header
from utils.optional import extra_installed, missing_message

page_header("Documents", "Knowledge graph extracted from your document library, and document chat.",
            chapter="07_Document_Insight")
if not extra_installed("ai"):
    st.info(missing_message("Indexing documents, chatting with them and extracting their knowledge graph", "ai"),
            icon=":material/extension:")
app()
