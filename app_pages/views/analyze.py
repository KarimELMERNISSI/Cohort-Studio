"""Analyze page: hypothesis tests, epidemiological measures, ANCOVA, power, z-scores."""

from app_pages.epidemiology import app
from ui.components import page_header

page_header("Hypothesis tests & epidemiology", "Group comparisons, 2x2 measures of association, ANCOVA, power and reference z-scores.",
            chapter="05_Epidemiology")
app()
