"""Suite-wide settings."""

import os
import sys

import dotenv
import pytest

# The tests never read the developer's .env (API keys, identity provider, data folder): main.py loads it at
# start-up, and the AppTests run main.py. The loading itself is tested in a subprocess (tests/test_startup_config.py).
dotenv.load_dotenv = lambda *args, **kwargs: False

# Strict session state: a page that modifies the dataset views in place, or assigns the dataset keys
# directly, raises instead of being silently corrected (ui/state.py, follow-up programme M1).
os.environ.setdefault("CMP_STRICT_STATE", "1")

# A bootstrap administrator password, so that the app's start-up never issues a first-run setup token: whichever
# AppTest started the app first would otherwise see the first-run page instead of the login form. The first-run
# setup has its own tests (tests/test_first_admin.py, tests/ui_layer/test_first_admin_page.py).
os.environ.setdefault("CMP_ADMIN_PASSWORD", "test-bootstrap-password-1")


@pytest.fixture(autouse=True)
def _fresh_graph_manager():
    """Each test starts without the process-wide graph engine manager: one configured by an earlier test (an engine,
    offline mode) would decide for the next."""
    yield
    manager = sys.modules.get("graph.manager")
    if manager is not None:
        manager._GLOBAL_MANAGER = None  # type: ignore[attr-defined]
