"""CohortStudio - entry point: configuration, authentication, navigation and shell."""

from pathlib import Path

from utils.settings import SettingsError, load_settings

# .env first, then every setting validated, before any module reads the configuration (identity provider, data
# folder, logging, AI keys); a variable already set in the environment wins, which is how Docker passes them
try:
    settings = load_settings(Path(__file__).with_name(".env"))
    from manage.connectors import check_connector_settings  # reads no setting when imported

    check_connector_settings(settings)  # each CMP_* connector setting names a connector of this version
except SettingsError as exc:  # stopped before a module reads the wrong value; the message never shows a value
    import sys

    import streamlit as st

    # .env is read once per server process: a correction applies after a restart
    message = f"CohortStudio cannot start: {exc} Correct the environment or .env, then restart the application."
    print(message, file=sys.stderr)  # the server log (docker compose logs), where the page is not seen
    st.error(message, icon=":material/gpp_bad:")
    st.stop()

import streamlit as st

from manage.ai_payload import disable_remote_tracing
from manage.iam import IAMConfigurationError, get_iam_manager
from manage.iam.authz import is_admin_role
from ui import nav, shell
from ui.auth import LOGO, require_login
from ui.preferences import seed_session
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
    # no prompt goes to a tracing service (LangSmith), whatever .env says: it would go round the data-sharing
    # confirmation of Settings
    disable_remote_tracing()
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
    seed_session()  # once per user: the choices of Settings kept from the last session

    state = get_state()
    state.begin_run()  # the dataset views of this run (AppState is the only writer of the data)
    is_admin = is_admin_role(st.session_state.get("user_role"))  # privileges come from the stored role
    page = st.navigation(nav.build_navigation(is_admin), position=nav.NAV_POSITION)
    shell.render_sidebar(state)
    page.run()
    state.end_run()  # strict mode (tests): the page must not have modified the views in place


if __name__ == "__main__":
    main()
