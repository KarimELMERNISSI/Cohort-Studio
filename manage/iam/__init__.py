"""
Identity and Access Management (IAM) Subsystem.

Provides modular, framework-agnostic user authentication and authorization
supporting Local DuckDB, Better Auth (RESTful remote), and Disabled modes: the identity providers of
:data:`manage.iam.providers.IAM_PROVIDERS`, loaded when first used.
"""

from typing import TYPE_CHECKING, Any

from .base import BaseIAMProvider
from .manager import IAMConfigurationError, IAMManager, get_iam_manager
from .schemas import (
    AuthResult,
    IAMConfig,
    IAMProviderType,
    UserCredentials,
    UserProfile,
    UserRole,
)

if TYPE_CHECKING:
    from .providers.better_auth import BetterAuthIAMProvider
    from .providers.disabled import DisabledIAMProvider
    from .providers.local_duckdb import LocalDuckDBIAMProvider

# the providers' classes, loaded on first use (manage.iam.providers)
_PROVIDERS = ("BetterAuthIAMProvider", "DisabledIAMProvider", "LocalDuckDBIAMProvider")


def __getattr__(name: str) -> Any:
    if name not in _PROVIDERS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import providers

    value = getattr(providers, name)
    globals()[name] = value  # the next access does not come here
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_PROVIDERS})


__all__ = [
    "BaseIAMProvider",
    "LocalDuckDBIAMProvider",
    "BetterAuthIAMProvider",
    "DisabledIAMProvider",
    "IAMConfigurationError",
    "IAMManager",
    "get_iam_manager",
    "IAMConfig",
    "IAMProviderType",
    "UserRole",
    "UserProfile",
    "UserCredentials",
    "AuthResult",
]
