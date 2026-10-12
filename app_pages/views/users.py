"""Users page (administrators)."""

from app_pages.users_management import app
from ui.components import page_header

page_header("Users", "Activate accounts, reset passwords and remove users.")
app()
