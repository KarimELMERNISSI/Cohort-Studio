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
    print(message, file=sys.stderr)  # noqa: T201 (the server log, docker compose logs, where the page is not seen)
    st.error(message, icon=":material/gpp_bad:")
    st.stop()

import streamlit as st

from core.expressions import WorkerUnavailable
from manage.ai_payload import disable_remote_tracing
from manage.iam import IAMConfigurationError
from manage.iam.authz import is_admin_role
from manage.workers.pool import BUSY
from ui import nav, shell
from ui.auth import LOGO, require_login
from ui.preferences import seed_session
from ui.services import stop_engine_off_list
from ui.state import get_state
from ui.theme import APP_NAME, apply_theme, show_logo
from ui.waiting import computing
from utils.logging_config import configure_logging


@st.cache_resource(show_spinner=False)
def _bootstrap_admin_once() -> str:
    """Create the initial administrator from CMP_ADMIN_* settings, once per server process."""
    from ui.services import shared

    _, message = shared().iam.ensure_bootstrap_admin()
    return message


@st.cache_resource(show_spinner=False)
def _expression_workers_once() -> bool:
    """The workers that compute every formula and criterion under a time limit and a memory limit, installed once
    per server process (manage.workers); unless the development switch keeps them in this process, which the
    log says."""
    from manage.workers import install_for

    return install_for(settings)


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
    _expression_workers_once()

    if not require_login():
        st.stop()
    seed_session()  # once per user: the choices of Settings kept from the last session
    # an AI engine on an Ollama address an administrator has removed stops before the page can use it
    stop_engine_off_list()

    state = get_state()
    is_admin = is_admin_role(st.session_state.get("user_role"))  # privileges come from the stored role
    page = st.navigation(nav.build_navigation(is_admin), position=nav.NAV_POSITION)
    # the views, the sidebar and the page run where a formula or criterion that waits on a worker says why the page
    # waits, and a click elsewhere or Stop ends it; no worker to compute it: the page says so in the waiting line's
    # place, and the next run asks again
    line = None
    try:
        with computing() as line:
            state.begin_run()  # the dataset views of this run (AppState is the only writer of the data)
            shell.render_sidebar(state)
            page.run()
    except WorkerUnavailable as exc:
        with line.container() if line is not None else st.container():
            st.error(str(exc), icon=":material/hourglass_empty:" if str(exc) == BUSY else ":material/error:")
            st.button("Try again", key="worker_try_again", icon=":material/refresh:")
    state.end_run()  # strict mode (tests): the page must not have modified the views in place


if __name__ == "__main__":
    main()
