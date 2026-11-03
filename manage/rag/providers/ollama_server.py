"""
What an Ollama server says of its models, asked without the chat connector: which of them it runs elsewhere. Ollama
0.12 and later run a cloud model on ollama.com through the local server, and so any copy of one (``ollama cp``) or
model built on one (``FROM``), whatever its name; the server's ``/api/show`` and ``/api/tags`` answer ``remote_host``
for them. A name's tag (":cloud", "-cloud") says so only for the models pulled under it (manage.ai_payload.cloud_model).
An older Ollama, or a server that cannot be reached, answers nothing: no model is then known to run elsewhere by this
route.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_HOST = "http://localhost:11434"
TIMEOUT = (
    3.0  # seconds: the question is asked before an engine starts and when Settings lists the models
)


def _ask(base_url: str | None, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    try:
        import httpx  # the "ai" extra
    except ImportError:
        return None
    url = (base_url or DEFAULT_HOST).rstrip("/") + path
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.request(method, url, json=body)
        return response.json() if response.status_code == 200 else None
    except Exception as exc:  # unreachable, refused, not JSON: nothing is known
        logger.debug("The Ollama server did not say where its models run (%s)", type(exc).__name__)
        return None


def _remote(entry: Any) -> str | None:
    if not isinstance(entry, dict):
        return None
    host = entry.get("remote_host") or ("https://ollama.com" if entry.get("remote_model") else None)
    return str(host) if host else None


class NoAnswer(Exception):
    """The Ollama server did not say where it runs a model: unreachable, an error, or no answer in time."""


def remote_host(base_url: str | None, model: str | None) -> str | None:
    """The address that runs ``model`` for this Ollama server when the server does not run it itself (a cloud model,
    or a copy of one), as ``/api/show`` answers; None when the server runs it, or cannot say (for what Settings
    shows: a decision on what is sent reads :func:`runs_at`)."""
    if not model:
        return None
    return _remote(_ask(base_url, "POST", "/api/show", {"model": model}))


def runs_at(base_url: str | None, model: str | None) -> str | None:
    """Where the server runs ``model``: the address that runs it when the server does not run it itself, None when it
    does (an Ollama older than the cloud models answers without saying: it runs every model itself). Raises
    :class:`NoAnswer` when the server does not answer in time, so that "it runs here" is never read in a silence: a
    copy of a cloud model would then be sent every value, as to a model on this machine."""
    if not model:
        return None
    answer = _ask(base_url, "POST", "/api/show", {"model": model})
    if answer is None:
        raise NoAnswer(model)
    return _remote(answer)


def remote_models(base_url: str | None) -> dict[str, str]:
    """The models the server lists that it runs elsewhere, by name, with the address that runs each (``/api/tags``)."""
    found = _ask(base_url, "GET", "/api/tags")
    models = found.get("models", []) if isinstance(found, dict) else []
    return {
        str(entry["name"]): host
        for entry in models
        if isinstance(entry, dict) and entry.get("name") and (host := _remote(entry))
    }


__all__ = ["NoAnswer", "remote_host", "remote_models", "runs_at"]
