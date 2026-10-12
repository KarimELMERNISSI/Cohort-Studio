"""Explore page: descriptive statistics, column renaming, correlations, performance benchmark."""

from app_pages.home import app
from ui.components import page_header

page_header("Statistics & correlations", "Descriptive statistics by variable type, column renaming and the correlation network.",
            chapter="01_Home")
app()
