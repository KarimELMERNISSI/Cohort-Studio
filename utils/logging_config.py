"""Central logging configuration. Call ``configure_logging()`` once from the application entry point.

Every record is written without API keys (:func:`redact_secrets`): a provider's error can quote the key it refused,
as Google does for a suspended key ("Consumer 'api_key:...' has been suspended").

No record holds an exception's text as Python writes it: an error raised while computing on the data can quote a
cell ("could not convert string to float: 'Jean Dupont'"), and a file's error its path, which names the account.
- A logging call names a caught exception by its kind (``type(exc).__name__`` or :func:`exception_kind`) where the
  failure comes from the data, or whenever it is unsure, and through :func:`exception_said` for a service's error;
  never by the exception itself, its ``str``, or a message built from it beforehand. A quality gate reads the
  logging calls for it (``tests/test_quality_gates.py``).
- Under those calls, :func:`records_without_values` writes every exception that reaches a record as itself by its
  kind alone, whichever logger makes the record. Each entry point installs it through :func:`configure_logging`:
  ``main.py``, every command of ``manage_users.py``, and the test suite.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import traceback
from collections.abc import Mapping
from typing import Any

from utils.settings import get_settings, secret_values

_configured = threading.Event()  # set once the root logger is configured
_NOISY_LIBRARIES = ("httpx", "httpcore", "urllib3", "lancedb", "qdrant_client", "portalocker", "watchdog", "PIL", "matplotlib", "numba")

REDACTED = "[redacted]"
_KEYS = (
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),  # Google
    re.compile(r"\bsk-[0-9A-Za-z_\-]{20,}"),  # OpenAI, Anthropic
)
_KEY_FIELD = re.compile(
    r"(?i)\b(api[_-]?key|x-api-key|authorization)(['\"]?\s*[:=]\s*['\"]?(?:bearer\s+)?)([^\s'\",;&}]{8,})"
)


_URL_CREDENTIALS = re.compile(r"(?<=://)[^/\s@]+@")  # http://user:password@host


def redact_secrets(text: str) -> str:
    """The text without API keys: the values of the configured secrets (variables named ``*API_KEY``,
    ``*_SECRET`` or ``*_PASSWORD``, as set and without their surrounding spaces), Google, OpenAI and Anthropic keys,
    the credentials of an address (``http://user:password@host``), and the value of an ``api_key: ...``,
    ``x-api-key: ...`` or ``Authorization: ...`` field."""
    for value in secret_values():  # read as they are, never validated: a wrong setting cannot stop the redaction
        if value in text:
            text = text.replace(value, REDACTED)
    for pattern in _KEYS:
        text = pattern.sub(REDACTED, text)
    text = _URL_CREDENTIALS.sub(f"{REDACTED}@", text)
    return _KEY_FIELD.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)


def single_line(value: object) -> str:
    """
    ``value`` on one line, for a log entry: a typed name, path or formula cannot forge another entry. Each
    character that is not printable is written as its escape (``\\n``, ``\\x1b``, ``\\u2028``): a line break
    of another kind (U+2028, U+0085) breaks a line in many log viewers, and a terminal's escape rewrites the
    screen of whoever reads the log.
    """
    text = str(value)
    if not text.isprintable():
        text = "".join(
            one if one.isprintable() else one.encode("unicode_escape").decode("ascii") for one in text
        )
    # no line break is left (each was written as its escape): the replace changes nothing, and says so to the code
    # scan, which cannot tell a printable text from one that holds a line break
    return text.replace("\n", "\\n")


AUDIT_LOGGER = "cohortstudio.audit"


def audit_logger() -> logging.Logger:
    """
    The logger of the record an organisation answers from: who confirmed that study data may go to an AI
    recipient, who withdrew it, the confirmations a sign-out ended, and the engines started under one. At INFO
    whatever ``CMP_LOG_LEVEL`` says: that setting quiets the rest of the log, and must not silence this record.
    The record is the server's log, kept as long as the log is (``.env.example`` says how).
    """
    found = logging.getLogger(AUDIT_LOGGER)
    found.setLevel(logging.INFO)
    return found


# a quoted string (not an apostrophe inside a word, as in "the file's"), and a path of two parts or more (not a URL's).
# A folder's name can hold spaces (C:\Users\Jean Dupont\...): a part with spaces counts while a separator follows it,
# so the last part, the file's name, ends at the first space
_QUOTED_TEXT = re.compile(r"(?<![A-Za-z0-9])'[^'\n]*'(?![A-Za-z0-9])|\"[^\"\n]*\"")
_PATH_PART = r"[^\\/\s'\"():;,]+"
_PATH_TEXT = re.compile(
    rf"(?<![\w:/.])(?:[A-Za-z]:)?(?:[\\/]{_PATH_PART}(?: {_PATH_PART})*(?=[\\/]))+[\\/]{_PATH_PART}"
)


def without_values(text: object) -> str:
    """
    An error's text for a record that travels (a replay's error, the log): every quoted string, which may be a cell of
    the data ("could not convert string to float: 'Jean Dupont'"), becomes "…", and a path its file name (an absolute
    path names the account); on one line.
    """
    said = _QUOTED_TEXT.sub('"…"', str(text))
    said = _PATH_TEXT.sub(lambda m: m.group(0).replace("\\", "/").rsplit("/", 1)[-1], said)
    return single_line(said)


def exception_kind(exc: BaseException) -> str:
    """
    An exception for the log by its kind alone: its class's name, and for one of Python's own system errors that
    holds an error number the system's words for that number ("OSError: No space left on device"), looked up here
    and never read from the exception, whose own words can name the file. Not for a library's subclass, whose
    number is not the system's (an ``ssl.SSLError`` holds 1, which would read "Operation not permitted"; a
    ``socket.gaierror`` its own codes), nor when the system does not know the number. What the cleaning of records
    writes for every exception it meets (:func:`records_without_values`), and what a call site writes for a failure
    on the data, or whenever it is unsure whose error it is (``type(exc).__name__`` says the same without the
    system's words).
    """
    kind = type(exc).__name__
    if not (isinstance(exc, OSError) and type(exc).__module__ == "builtins"):
        return kind
    number = exc.errno
    if not isinstance(number, int) or isinstance(number, bool):
        return kind
    try:
        words = os.strerror(number)
    except (ValueError, OverflowError):  # a number the system does not know
        return kind
    return kind if not words or words.lower().startswith("unknown error") else f"{kind}: {words}"


def exception_said(exc: BaseException) -> str:
    """
    An exception for the log, where a call site chooses it for a service's error (an identity or AI provider's
    reply, a store's refusal): its kind, then its text without quoted strings and paths (:func:`without_values`).
    Where the failure comes from a computation on the data, log the kind alone (:func:`exception_kind`): a text can
    hold a value that no quote surrounds ("KeyError: 990017", "year must be in 1..9999, not 12021").
    """
    kind = type(exc).__name__
    try:
        text = without_values(exc)
    except Exception:  # an exception whose own text cannot be written
        return kind
    return f"{kind}: {text}" if text else kind


_CAUSE = "\nThe above exception was the direct cause of the following exception:\n\n"
_CONTEXT = "\nDuring handling of the above exception, another exception occurred:\n\n"


def _traceback_lines(exc: BaseException, lines: list[str], seen: set[int]) -> None:
    seen.add(id(exc))
    cause, context = exc.__cause__, exc.__context__
    if cause is not None and id(cause) not in seen:
        _traceback_lines(cause, lines, seen)
        lines.append(_CAUSE)
    elif context is not None and not exc.__suppress_context__ and id(context) not in seen:
        _traceback_lines(context, lines, seen)
        lines.append(_CONTEXT)
    if exc.__traceback__ is not None:
        lines.append("Traceback (most recent call last):\n")
        lines.extend(traceback.format_tb(exc.__traceback__))
    lines.append(exception_kind(exc) + "\n")
    if isinstance(exc, BaseExceptionGroup):
        for position, member in enumerate(exc.exceptions, 1):
            if id(member) in seen:
                continue
            lines.append(f"  Exception {position} of {len(exc.exceptions)} of the group:\n")
            inner: list[str] = []
            _traceback_lines(member, inner, seen)
            lines.extend("  " + line for line in "".join(inner).splitlines(keepends=True))


def traceback_without_values(exc: BaseException) -> str:
    """
    The traceback of ``exc`` for the log: the frames of every exception of its chain as Python writes them (the
    files, the lines and the code's own text), and each exception by its kind alone, the members of a group
    included. Neither its text nor its notes: Python's own last line holds the text whole, and the error is most
    often one on the data, whose text can hold a value no quote surrounds.
    """
    lines: list[str] = []
    _traceback_lines(exc, lines, set())
    return "".join(lines).rstrip("\n")


NOT_WRITTEN = "A log record could not be written without the values it may hold; it is not shown"


def _kind(value: Any) -> Any:
    return exception_kind(value) if isinstance(value, BaseException) else value


def _without_exception_text(record: logging.LogRecord) -> None:
    record.msg = _kind(record.msg)
    args: Any = record.args
    if isinstance(args, tuple):
        if any(isinstance(arg, BaseException) for arg in args):
            record.args = tuple(_kind(arg) for arg in args)
    elif isinstance(args, Mapping) and any(isinstance(arg, BaseException) for arg in args.values()):
        record.args = {key: _kind(arg) for key, arg in args.items()}
    info = record.exc_info
    if info and info[1] is not None:
        if not record.exc_text:
            record.exc_text = redact_secrets(traceback_without_values(info[1]))
        # the exception leaves the record: a handler that formats it itself (a JSON formatter, a log service's
        # handler) would write its text whole; the standard formatter writes exc_text
        record.exc_info = None


def records_without_values() -> None:
    """
    Every log record of this process, whichever logger writes it, without the text of an exception: an exception
    given as the message or as an argument is written by its kind alone, and so is each exception of a traceback
    (:func:`traceback_without_values`), which then leaves the record. This is the net for what the application did
    not word itself, above all a page's uncaught error, most often one on the data. Done when the record is made,
    not on a handler: Streamlit's loggers have their own handlers and do not propagate, so the traceback of a page's
    uncaught error ("Uncaught app execution") never passes the root handlers' filters; a library's logger is covered
    the same way. A text already formatted into the message (an f-string, ``str(exc)``) cannot be undone here: the
    application's own calls never do it (see the module's docstring), a library's may. Installed once per process
    (:func:`configure_logging`); a second call changes nothing.
    """
    made = logging.getLogRecordFactory()
    if getattr(made, "wraps", None) is not None:
        return

    def record_without_values(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = made(*args, **kwargs)
        try:
            _without_exception_text(record)
        except Exception:  # never a record with its values, and never an error raised into the code that logs
            record.msg, record.args, record.exc_info, record.exc_text = NOT_WRITTEN, None, None, None
        return record

    record_without_values.wraps = made  # type: ignore[attr-defined]  # the factory this one cleans the records of
    logging.setLogRecordFactory(record_without_values)


class RedactSecrets(logging.Filter):
    """Rewrites each record without API keys, its traceback included."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg, record.args = redact_secrets(record.getMessage()), None
        except Exception:  # a malformed call is reported by the handler as usual
            return True
        if record.exc_info and not record.exc_text:
            record.exc_text = redact_secrets(logging.Formatter().formatException(record.exc_info))
        return True


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger from ``CMP_LOG_LEVEL`` (default INFO), and every record of the process without an
    exception's values (:func:`records_without_values`). Safe to call repeatedly."""
    records_without_values()
    if _configured.is_set():
        return
    level_name = (level or get_settings().logging.level).upper()
    root = logging.getLogger()
    root.setLevel(getattr(logging, level_name, logging.INFO))
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
    for handler in root.handlers:
        if not any(isinstance(f, RedactSecrets) for f in handler.filters):
            handler.addFilter(RedactSecrets())
    for name in _NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)
    _configured.set()
