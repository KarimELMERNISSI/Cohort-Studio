"""Import page: the three-step import flow."""

from ui.components import page_header
from ui.import_flow import render_import_flow
from ui.state import get_state

page_header("Import", "Choose a file, check how it is read, then name and load it.", chapter="01_Home")
render_import_flow(get_state())
