"""Validate page: data quality dashboard, implausible values, inclusion criteria, outliers, version comparison."""

from app_pages.data_monitoring import app
from ui.components import page_header

page_header("Validate", "Quality score, missingness, implausible values, inclusion criteria, outliers and version comparison.",
            chapter="02_Data_Validation")
app()
