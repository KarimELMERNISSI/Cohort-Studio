"""API keys never reach the log: a provider's error can quote the key it refused. Nor does an exception's text as
Python writes it: an error raised while computing on the data can quote a cell, a file's error its path."""

import errno
import io
import logging
import os
import threading
import tomllib
from pathlib import Path

import pytest

from utils.logging_config import (
    NOT_WRITTEN,
    REDACTED,
    RedactSecrets,
    configure_logging,
    exception_kind,
    exception_said,
    records_without_values,
    redact_secrets,
    traceback_without_values,
)

# the shapes of a Google and an OpenAI key, built here so that no secret scanner sees one
GOOGLE = "AIza" + "Sy0abcDEF_ghi-JKL" * 2 + "m"
OPENAI = "sk-proj-" + "a1B2c3" * 5
# a cell of the data, and the account a path names
MARK = "zq-marker-7731"
ACCOUNT = "zq-account-55"


def test_api_keys_are_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    suspended = "403 PERMISSION_DENIED. Consumer 'api_key:{}' has been suspended."
    assert redact_secrets(suspended.format(GOOGLE)) == suspended.format(REDACTED)
    assert OPENAI not in redact_secrets(f"Incorrect API key provided: {OPENAI}")
    value = "f" * 16
    for field in ("x-api-key: {}", "Authorization: Bearer {}", "api_key={}"):
        assert redact_secrets(field.format(value)) == field.format(REDACTED)

    # no recognisable shape: redacted because it is the configured key
    mistral = "m" * 31 + "7"
    monkeypatch.setenv("MISTRAL_API_KEY", mistral)
    refused = "401 Unauthorized for key {}"
    assert redact_secrets(refused.format(mistral)) == refused.format(REDACTED)


def test_ordinary_messages_and_the_setup_token_are_kept() -> None:
    setup = "Create the first administrator with this one-time setup token: {} (or set CMP_ADMIN_PASSWORD)."
    for message in (
        "Embedding attempt with models/gemini-embedding-2 failed: 429 RESOURCE_EXHAUSTED.",
        setup.format("Zx9_" * 8),
        "Loading the existing lancedb index...",
    ):
        assert redact_secrets(message) == message


def test_the_log_shows_neither_the_key_nor_its_traceback(plain_log_records: None) -> None:
    """The handler's filter alone, on records as Python makes them."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RedactSecrets())
    logger = logging.getLogger("tests.redaction")
    logger.addHandler(handler)
    logger.propagate = False
    refusal = f"Consumer 'api_key:{GOOGLE}' has been suspended."
    try:
        logger.warning("Embedding attempt failed: %s", refusal)
        try:
            raise RuntimeError(refusal)
        except RuntimeError:
            logger.exception("All Gemini embedding attempts failed")
    finally:
        logger.removeHandler(handler)
    written = stream.getvalue()
    assert GOOGLE not in written
    assert written.count(REDACTED) == 2 and "RuntimeError" in written


def test_configure_logging_filters_every_root_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    root = logging.getLogger()
    handler = logging.StreamHandler(io.StringIO())
    monkeypatch.setattr("utils.logging_config._configured", threading.Event())
    monkeypatch.setattr(root, "handlers", [handler])
    monkeypatch.setattr(root, "level", root.level)
    configure_logging()
    configure_logging()
    assert sum(isinstance(f, RedactSecrets) for f in handler.filters) == 1


# --------------------------------------------------------------------------- an exception's text


def _path() -> str:
    return "C:\\data\\users\\" + ACCOUNT + "\\datasets\\labs_v1.parquet"


def _fail() -> None:
    """A conversion that quotes a cell, then a lookup raised from it (its cause), then an error raised while that
    one is handled (its context), with a note and a path."""
    try:
        try:
            float(MARK)
        except ValueError as first:
            raise KeyError(MARK) from first
    except KeyError:
        error = RuntimeError(f"the row of '{MARK}' was not written to {_path()}")
        error.add_note(f'seen beside "{MARK}"')
        raise error


def test_an_exceptions_kind_is_its_class_and_for_a_system_error_the_systems_own_words() -> None:
    try:
        float(MARK)
    except ValueError as exc:
        assert exception_kind(exc) == "ValueError"
    # the words are looked up from the number, never read from the exception: its own can name the file
    full = OSError(errno.ENOSPC, f"no room for '{MARK}' in {_path()}", _path())
    assert exception_kind(full) == f"OSError: {os.strerror(errno.ENOSPC)}"
    assert exception_kind(OSError(errno.EACCES, "refused", _path())).startswith("PermissionError: ")
    for exc in (
        full,
        OSError(f"cannot read {_path()}"),
        OSError(10**12, MARK),
        OSError(True, MARK),
    ):
        assert MARK not in exception_kind(exc) and ACCOUNT not in exception_kind(exc)
    assert exception_kind(OSError(f"cannot read {_path()}")) == "OSError"


def test_a_librarys_error_number_is_not_read_as_the_systems() -> None:
    """A library's subclass of OSError holds its own number: an SSL error's 1 is not "Operation not permitted", and
    would send an operator after a permission. Nor are words written for a number the system does not know."""
    import socket
    import ssl

    assert exception_kind(
        ssl.SSLError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    ) == ("SSLError")
    assert exception_kind(ssl.SSLCertVerificationError(1, "certificate verify failed")) == (
        "SSLCertVerificationError"
    )
    assert exception_kind(socket.gaierror(11001, "getaddrinfo failed")) == "gaierror"
    # one of Python's own errors with a number the system has no words for (a Windows socket error's)
    unknown = next(
        n for n in range(10061, 10200) if os.strerror(n).lower().startswith("unknown error")
    )
    refused = OSError(unknown, "refused")  # on Windows, 10061 makes a ConnectionRefusedError
    assert exception_kind(refused) == type(refused).__name__


def test_an_exception_is_said_by_its_kind_and_its_text_without_values() -> None:
    try:
        float(MARK)
    except ValueError as exc:
        assert exception_said(exc) == 'ValueError: could not convert string to float: "…"'
    assert exception_said(OSError(2, "No such file or directory", _path())) == (
        'FileNotFoundError: [Errno 2] No such file or directory: "…"'
    )
    # a path no quote surrounds: its file's name
    assert (
        exception_said(RuntimeError(f"cannot open {_path()}"))
        == "RuntimeError: cannot open labs_v1.parquet"
    )
    # a folder whose name holds spaces: nothing of the path but the file's name
    spaced = "C:\\Users\\Jean Dupont\\data\\users\\" + ACCOUNT + "\\v1.parquet"
    assert exception_said(RuntimeError(f"not found: {spaced} (twice)")) == (
        "RuntimeError: not found: v1.parquet (twice)"
    )
    # punctuation ends the path, and the words after it are kept
    assert exception_said(RuntimeError("see /srv/my data/users/a b/labs.csv, and/or km/h")) == (
        "RuntimeError: see labs.csv, and/or km/h"
    )
    # words followed by another separator read as a folder's name: more is removed, never less
    assert exception_said(RuntimeError("open /srv/users/a b/labs.csv for reading/writing")) == (
        "RuntimeError: open writing"
    )
    assert exception_said(ZeroDivisionError()) == "ZeroDivisionError"
    assert exception_said(TimeoutError("line one\nline two")) == "TimeoutError: line one\\nline two"

    class Unwritable(Exception):
        def __str__(self) -> str:
            raise RuntimeError(MARK)

    assert exception_said(Unwritable()) == "Unwritable"


def test_a_traceback_keeps_its_frames_and_names_each_exception_by_its_kind() -> None:
    with pytest.raises(RuntimeError) as raised:
        _fail()
    written = traceback_without_values(raised.value)
    assert MARK not in written and ACCOUNT not in written and "labs_v1" not in written
    # the three exceptions of the chain, each with its frames, in Python's order and with its words between them
    assert written.count("Traceback (most recent call last):") == 3
    lines = written.splitlines()
    kinds = [lines.index(kind) for kind in ("ValueError", "KeyError", "RuntimeError")]
    assert kinds == sorted(kinds) and lines[-1] == "RuntimeError"  # neither a text nor the note
    assert "direct cause of the following exception" in written
    assert "During handling of the above exception" in written
    assert "test_logging_config.py" in written and "in _fail" in written


def _raised(make):
    try:
        make()
    except Exception as exc:
        return exc
    raise AssertionError("nothing was raised")


def test_a_traceback_holds_no_value_that_no_quote_surrounds() -> None:
    """Texts that :func:`exception_said` would keep: a numeric identifier, a date as it was typed, a year."""
    import datetime

    import pandas as pd
    from dateutil import parser

    frame = pd.DataFrame({"x": [1.0]}, index=[5])
    # each value is put together here, so that no line of code a traceback shows holds it
    identifier, typed, year = int("9900" + "17"), MARK + " 1961", int("120" + "21")
    for exc, value in (
        (_raised(lambda: frame.loc[identifier]), str(identifier)),
        (_raised(lambda: parser.parse(typed)), MARK),
        (_raised(lambda: datetime.date(year, 3, 4)), str(year)),
    ):
        assert value in str(exc)  # Python's own text holds it
        written = traceback_without_values(exc)
        assert value not in written and written.splitlines()[-1] == type(exc).__name__


def test_a_groups_members_are_named_by_their_kind() -> None:
    group = ExceptionGroup(
        f"two failures for '{MARK}'",
        [ValueError(f"bad value '{MARK}'"), OSError(13, "Permission denied", _path())],
    )
    written = traceback_without_values(group)
    assert MARK not in written and ACCOUNT not in written
    assert written.splitlines() == [
        "ExceptionGroup",
        "  Exception 1 of 2 of the group:",
        "  ValueError",
        "  Exception 2 of 2 of the group:",
        f"  PermissionError: {os.strerror(13)}",
    ]


def test_an_exception_that_is_its_own_context_ends() -> None:
    first, second = ValueError("first"), KeyError("second")
    first.__context__, second.__context__ = second, first
    assert traceback_without_values(first).splitlines().count("ValueError") == 1


@pytest.fixture
def own_handler():
    """A logger with its own handler, which does not propagate: as Streamlit's loggers are, past the root handlers'
    filters."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("tests.own_handler")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    yield logger, stream
    logger.removeHandler(handler)


def test_no_record_holds_an_exceptions_text_whichever_logger_writes_it(own_handler) -> None:
    logger, stream = own_handler
    try:
        float(MARK)
    except ValueError as exc:
        logger.error("Uncaught app execution", exc_info=exc)
        logger.warning("could not read (%s)", exc)
        logger.warning("could not read (%(why)s)", {"why": exc})
        logger.warning(exc)
        logger.exception("as logger.exception writes it")
    written = stream.getvalue()
    assert MARK not in written and "could not convert" not in written
    lines = written.splitlines()
    assert (
        lines.count("ValueError") == 3
    )  # the two tracebacks' last lines, and the exception given as the message
    assert lines.count("could not read (ValueError)") == 2
    assert written.count("Traceback (most recent call last):") == 2
    assert "Uncaught app execution" in written and "test_logging_config.py" in written


def test_the_exception_leaves_the_record(own_handler) -> None:
    """A handler that formats the exception itself (a JSON formatter, a log service's) would write its text whole:
    the record keeps the traceback as text, and no exception."""
    logger, _ = own_handler
    seen: list[logging.LogRecord] = []

    class Keeps(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record)

    keeps = Keeps()
    logger.addHandler(keeps)
    try:
        try:
            raise RuntimeError(f"401 Unauthorized for key {OPENAI} and '{MARK}'")
        except RuntimeError as exc:
            logger.exception("The provider refused: %s", exc)
    finally:
        logger.removeHandler(keeps)
    (record,) = seen
    # the message as any handler writes it (another handler's filter may have formatted it already)
    assert record.exc_info is None and record.getMessage() == "The provider refused: RuntimeError"
    assert record.exc_text.startswith("Traceback (most recent call last):")
    assert record.exc_text.endswith("\nRuntimeError")
    assert OPENAI not in record.exc_text and MARK not in record.exc_text


def test_records_are_made_without_values_once(own_handler, monkeypatch: pytest.MonkeyPatch) -> None:
    installed = logging.getLogRecordFactory()
    records_without_values()
    assert (
        logging.getLogRecordFactory() is installed
    )  # the suite installs it (conftest), as the application does

    # a record that cannot be cleaned is not written with its values, and the code that logs goes on
    def refuses(record: logging.LogRecord) -> None:
        raise RuntimeError("no")

    monkeypatch.setattr("utils.logging_config._without_exception_text", refuses)
    logger, stream = own_handler
    try:
        float(MARK)
    except ValueError as exc:
        logger.warning("could not read (%s)", exc, exc_info=exc)
    assert stream.getvalue().strip() == NOT_WRITTEN


def test_a_pages_uncaught_error_reaches_streamlits_log_without_its_values() -> None:
    """Streamlit logs a page's uncaught error with its traceback on its own logger, which does not propagate."""
    from streamlit.testing.v1 import AppTest

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    streamlit_log = logging.getLogger("streamlit.error_util")
    streamlit_log.addHandler(handler)
    # the cell is put together in the page, so that no line of its code holds it
    page = f"parts = {MARK.split('-')!r}\ncell = '-'.join(parts)\nfloat(cell)\n"
    try:
        at = AppTest.from_string(page).run()
    finally:
        streamlit_log.removeHandler(handler)
    assert len(at.exception) == 1 and MARK in at.exception[0].value  # the page shows it to its user
    written = stream.getvalue()
    assert "Uncaught app execution" in written and "Traceback (most recent call last):" in written
    assert written.splitlines()[-1] == "ValueError" and "float(cell)" in written
    assert MARK not in written and "could not convert" not in written


def test_streamlit_never_prints_a_pages_error_through_rich() -> None:
    """With ``logger.enableRich`` on, Streamlit's default whenever the package rich is installed, a page's uncaught
    error is printed to the console by rich, its text whole, and never reaches the log: the committed configuration
    turns it off, and the server reads it so."""
    from streamlit import config

    root = Path(__file__).resolve().parents[1]
    committed = tomllib.loads((root / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert committed["logger"]["enableRich"] is False
    if Path.cwd() == root:  # the configuration Streamlit read for this run is the committed one
        assert config.get_option("logger.enableRich") is False
