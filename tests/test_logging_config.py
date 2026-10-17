"""API keys never reach the log: a provider's error can quote the key it refused."""

import io
import logging

import pytest

from utils.logging_config import REDACTED, RedactSecrets, configure_logging, redact_secrets

# the shapes of a Google and an OpenAI key, built here so that no secret scanner sees one
GOOGLE = "AIza" + "Sy0abcDEF_ghi-JKL" * 2 + "m"
OPENAI = "sk-proj-" + "a1B2c3" * 5


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


def test_the_log_shows_neither_the_key_nor_its_traceback() -> None:
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
    monkeypatch.setattr("utils.logging_config._CONFIGURED", False)
    monkeypatch.setattr(root, "handlers", [handler])
    monkeypatch.setattr(root, "level", root.level)
    configure_logging()
    configure_logging()
    assert sum(isinstance(f, RedactSecrets) for f in handler.filters) == 1
