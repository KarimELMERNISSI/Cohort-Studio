"""Documents page: literature knowledge graph and document chat."""

from app_pages.document_insight import app
from ui.components import page_header

page_header("Documents", "Knowledge graph extracted from your document library, and document chat.",
            chapter="07_Document_Insight")
app()
