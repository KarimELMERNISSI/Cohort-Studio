"""Validate page: data quality dashboard, anomalies, inclusion criteria, outliers, version comparison."""

from app_pages.data_monitoring import app
from ui.components import page_header

page_header("Validate", "Quality score, missingness, clinical anomalies, inclusion criteria, outliers and version comparison.",
            chapter="02_Data_Validation")
app()
