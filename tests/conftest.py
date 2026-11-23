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

# The AppTests run main.py, which would start the worker processes for formulas and criteria: the tests compute in
# this process, through the round trip below (the processes have their own tests: tests/test_expression_workers.py)
os.environ.setdefault("CMP_FORMULA_IN_PROCESS", "1")

# Every expression the tests compute goes through the worker's wire in memory (manage.workers.wire.round_trip): the
# job and its answer pickled and read back as a worker's are, the answer by the restricted reader. What a worker
# computes is so shown to be what this process computes, at the speed of the suite. A test of this process's own
# computation installs nothing (core.expressions.runner.running(None)).
from core.expressions.runner import install as _install_runner
from manage.workers.wire import round_trip as _round_trip

_install_runner(_round_trip)

# Every log record without an exception's text, as in the application (main.py and every command of manage_users.py
# install it through configure_logging): installed here once, so that what a test reads in the log does not depend
# on whether an AppTest ran main.py before it. A test of a call site's own words asks for ``plain_log_records``.
from utils.logging_config import records_without_values as _records_without_values

_records_without_values()


@pytest.fixture
def plain_log_records():
    """Log records as they were made before the cleaning (``records_without_values``), an exception's text whole:
    for a test that a logging call, or a handler's filter, keeps a value out of the log by itself. Not for a test
    that runs main.py in an AppTest or calls ``configure_logging``: either installs the cleaning again."""
    import logging

    installed = logging.getLogRecordFactory()
    logging.setLogRecordFactory(getattr(installed, "wraps", None) or logging.LogRecord)
    yield
    logging.setLogRecordFactory(installed)


@pytest.fixture(autouse=True)
def _ollama_server_runs_it_here(monkeypatch: pytest.MonkeyPatch):
    """No test reaches an Ollama server: asked where it runs a model (an engine's start, a Send of Data > Clean), the
    server answers that it runs it itself. A test of a silent or a remote server sets its own answer."""
    import manage.rag.providers.ollama_server as server

    monkeypatch.setattr(server, "runs_at", lambda base_url, model: None)


@pytest.fixture(autouse=True)
def _fresh_graph_manager():
    """Each test starts without the process-wide graph engine manager: one configured by an earlier test (an engine,
    offline mode) would decide for the next."""
    yield
    manager = sys.modules.get("graph.manager")
    if manager is not None:
        manager._GLOBAL_MANAGER = None  # type: ignore[attr-defined]
