"""Enrich page: external data merge, computed variables, imputation, transformations."""

from app_pages.data_enrichment import app
from ui.components import page_header

page_header("Enrich", "Merge external data, derive variables, impute missing values and transform columns.",
            chapter="03_Data_Enrichment")
app()
