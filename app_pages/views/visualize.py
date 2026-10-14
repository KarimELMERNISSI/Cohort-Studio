"""Visualize page: distributions, group comparisons, time series, ROC, Bland-Altman."""

from app_pages.visualization import app
from ui.components import page_header
from ui.components.population import population_banner

page_header("Visualize", "Distributions, group comparisons, time series and diagnostic plots.", chapter="04_Visualization")
population_banner(purpose="plots")
app()
