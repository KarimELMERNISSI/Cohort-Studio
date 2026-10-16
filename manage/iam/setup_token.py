"""
First-run administrator (follow-up programme M4).

A fresh installation (a new Docker container) has no administrator. When no bootstrap password is configured
(``CMP_ADMIN_PASSWORD`` / ``CMP_ADMIN_PASSWORD_FILE``), the server issues a one-time setup token and writes it to
its log; the login page then offers to create the first administrator with it. Whoever can read the server log
operates the server, which is the trust this relies on (as Jupyter's token does).

Only a SHA-256 hash of the token is stored, in the data root. The token is valid until it is used or until an
administrator exists; a restart without an administrator issues a new one.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import secrets
from datetime import UTC, datetime
from pathlib import Path

TOKEN_FILE = "setup_token.json"


def _path(directory: str | os.PathLike[str]) -> Path:
    return Path(directory) / TOKEN_FILE


def _digest(token: str) -> str:
    return hashlib.sha256(token.strip().encode("utf-8")).hexdigest()


def issue(directory: str | os.PathLike[str]) -> str:
    """A new token (the previous one, if any, stops working); only its hash is written."""
    path = _path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(24)
    record = {"sha256": _digest(token), "issued_at": datetime.now(UTC).isoformat(timespec="seconds")}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record), encoding="utf-8")
    os.replace(tmp, path)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)
    return token


def pending(directory: str | os.PathLike[str]) -> bool:
    return _path(directory).is_file()


def check(directory: str | os.PathLike[str], token: str) -> bool:
    """True when ``token`` is the current setup token (constant-time comparison)."""
    try:
        record = json.loads(_path(directory).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(token.strip()) and hmac.compare_digest(str(record.get("sha256", "")), _digest(token))


def clear(directory: str | os.PathLike[str]) -> None:
    with contextlib.suppress(FileNotFoundError):
        _path(directory).unlink()


__all__ = ["TOKEN_FILE", "check", "clear", "issue", "pending"]
