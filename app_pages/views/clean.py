"""Clean page: how values are written, fixed (a variable's values recoded into a new column; text read as dates)."""

from app_pages.data_cleaning import app
from ui.components import page_header

page_header(
    "Clean",
    "Fix how values are written. Rows are never removed here; each fix is one traced step.",
    chapter="02_Data_Validation",
)
app()
