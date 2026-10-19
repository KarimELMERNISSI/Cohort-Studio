"""CohortStudio - entry point: configuration, authentication, navigation and shell."""

from pathlib import Path

from dotenv import load_dotenv

# .env first, before any module reads the configuration (identity provider, data folder, logging, AI keys); a
# variable already set in the environment wins, which is how Docker passes them
load_dotenv(Path(__file__).with_name(".env"), override=False)

import streamlit as st

from manage.iam import IAMConfigurationError, get_iam_manager
from manage.iam.authz import is_admin_role
from ui import nav, shell
from ui.auth import LOGO, require_login
from ui.state import get_state
from ui.theme import APP_NAME, apply_theme, show_logo
from utils.logging_config import configure_logging


@st.cache_resource(show_spinner=False)
def _bootstrap_admin_once() -> str:
    """Create the initial administrator from CMP_ADMIN_* settings, once per server process."""
    _, message = get_iam_manager().ensure_bootstrap_admin()
    return message


def main() -> None:
    configure_logging()
    st.set_page_config(page_title=APP_NAME, page_icon=LOGO, layout="wide")
    apply_theme()
    show_logo()
    try:
        _bootstrap_admin_once()
    except IAMConfigurationError as exc:  # never another identity provider in place of the one configured
        st.error(f"CohortStudio cannot start: {exc}", icon=":material/gpp_bad:")
        st.stop()

    if not require_login():
        st.stop()

    state = get_state()
    state.begin_run()  # the dataset views of this run (AppState is the only writer of the data)
    is_admin = is_admin_role(st.session_state.get("user_role"))  # privileges come from the stored role
    page = st.navigation(nav.build_navigation(is_admin), position=nav.NAV_POSITION)
    shell.render_sidebar(state)
    page.run()
    state.end_run()  # strict mode (tests): the page must not have modified the views in place


if __name__ == "__main__":
    main()
