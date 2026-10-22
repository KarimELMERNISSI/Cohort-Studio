"""
The identity providers, a connector family on :mod:`core.registry`: :data:`IAM_PROVIDERS`. The deployment's setting,
``CMP_IAM_PROVIDER``, chooses one (no automatic choice, no user's choice; any other value stops start-up):

- local: accounts in a DuckDB file on this server, passwords hashed with bcrypt (the default);
- better_auth: a Better Auth service, through its admin API;
- disabled: no sign-in.

The names are stored in deployments' settings: they never change, and the aliases are the spellings that were already
accepted. The package is light: the providers are loaded when first used (PEP 562).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from core.registry import ConnectorRegistry

if TYPE_CHECKING:
    from ..base import BaseIAMProvider
    from .better_auth import BetterAuthIAMProvider
    from .disabled import DisabledIAMProvider
    from .local_duckdb import LocalDuckDBIAMProvider

IAM_PROVIDERS: ConnectorRegistry[BaseIAMProvider] = ConnectorRegistry(
    "identity provider", setting="CMP_IAM_PROVIDER", auto=None, setting_only=True
)
IAM_PROVIDERS.register("local", "manage.iam.providers.local_duckdb:LocalDuckDBIAMProvider")
IAM_PROVIDERS.register(
    "better_auth",
    "manage.iam.providers.better_auth:BetterAuthIAMProvider",
    aliases=("betterauth", "better-auth", "remote"),
)
IAM_PROVIDERS.register(
    "disabled",
    "manage.iam.providers.disabled:DisabledIAMProvider",
    aliases=("none", "off", "bypass"),
)

# where each provider is defined, loaded on first use (written out, so the layer test sees the edges)
_LAZY = {
    "BetterAuthIAMProvider": "manage.iam.providers.better_auth",
    "DisabledIAMProvider": "manage.iam.providers.disabled",
    "LocalDuckDBIAMProvider": "manage.iam.providers.local_duckdb",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # the next access does not come here
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = [
    "IAM_PROVIDERS",
    "BetterAuthIAMProvider",
    "DisabledIAMProvider",
    "LocalDuckDBIAMProvider",
]
