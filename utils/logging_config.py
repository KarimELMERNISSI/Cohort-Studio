"""Central logging configuration. Call ``configure_logging()`` once from the application entry point.

Every record is written without API keys (:func:`redact_secrets`): a provider's error can quote the key it refused,
as Google does for a suspended key ("Consumer 'api_key:...' has been suspended").
"""

from __future__ import annotations

import logging
import re

from utils.settings import get_settings, secret_values

_CONFIGURED = False
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
    """``value`` on one line, for a log entry: a typed name, path or formula cannot forge another entry."""
    return str(value).replace("\r", "\\r").replace("\n", "\\n")


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
    """Configure the root logger from ``CMP_LOG_LEVEL`` (default INFO). Safe to call repeatedly."""
    global _CONFIGURED
    if _CONFIGURED:
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
    _CONFIGURED = True
