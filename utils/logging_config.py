"""Central logging configuration. Call ``configure_logging()`` once from the application entry point."""

from __future__ import annotations

import logging
import os

_CONFIGURED = False
_NOISY_LIBRARIES = ("httpx", "httpcore", "urllib3", "chromadb", "watchdog", "PIL", "matplotlib", "numba")


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger from ``CMP_LOG_LEVEL`` (default INFO). Safe to call repeatedly."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = (level or os.getenv("CMP_LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    root.setLevel(getattr(logging, level_name, logging.INFO))
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
    for name in _NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)
    _CONFIGURED = True
