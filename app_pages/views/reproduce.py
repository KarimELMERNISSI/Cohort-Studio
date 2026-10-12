"""Reproduce page: traces, artifacts, replay and reports."""

from app_pages.reproduce_analysis import app
from ui.components import page_header

page_header("Versions & traces", "Replay a recorded analysis, resolve its source files and generate the report.",
            chapter="08_Reproduction")
app()
