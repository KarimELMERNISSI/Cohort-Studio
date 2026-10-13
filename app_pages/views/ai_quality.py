"""AI quality page (administrators): RAG evaluation and embedding health."""

from app_pages.rag_monitoring import render_rag_monitoring
from ui.components import page_header

page_header("AI quality", "Quality of the AI answers over time: faithfulness, relevance, retrieval and embedding health.")
render_rag_monitoring()
