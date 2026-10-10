"""
Identity and Access Management (IAM) Subsystem.

Provides modular, framework-agnostic user authentication and authorization
supporting Local DuckDB, Better Auth (RESTful remote), and Disabled modes.
"""

from .base import BaseIAMProvider
from .manager import IAMManager, get_iam_manager
from .providers.better_auth import BetterAuthIAMProvider
from .providers.disabled import DisabledIAMProvider
from .providers.local_duckdb import LocalDuckDBIAMProvider
from .schemas import (
    AuthResult,
    IAMConfig,
    IAMProviderType,
    UserCredentials,
    UserProfile,
    UserRole,
)

__all__ = [
    "BaseIAMProvider",
    "LocalDuckDBIAMProvider",
    "BetterAuthIAMProvider",
    "DisabledIAMProvider",
    "IAMManager",
    "get_iam_manager",
    "IAMConfig",
    "IAMProviderType",
    "UserRole",
    "UserProfile",
    "UserCredentials",
    "AuthResult",
]
