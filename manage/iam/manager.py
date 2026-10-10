"""
IAM Manager and Factory.

Provides a unified, framework-agnostic facade for identity and access management.
Dynamically resolves and instantiates the active IAM provider (Local DuckDB,
Better Auth, or Disabled) based on environment configuration or runtime parameters.
"""

import logging
import os
from typing import Any

from .base import BaseIAMProvider
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

logger = logging.getLogger(__name__)


class IAMManager:
    """
    Central facade managing IAM operations and provider lifecycle.
    """

    def __init__(self, config: IAMConfig | None = None) -> None:
        """
        Initialize IAM Manager with configuration or environment variables.
        """
        self._config = config or self._resolve_config_from_env()
        self._provider: BaseIAMProvider = self._build_provider(self._config)

    @classmethod
    def _resolve_config_from_env(cls) -> IAMConfig:
        """Read IAM configuration from process environment variables."""
        raw_type = os.getenv("CMP_IAM_PROVIDER", "local").lower().strip()
        provider_type = IAMProviderType.LOCAL
        if raw_type in ("better_auth", "betterauth", "remote"):
            provider_type = IAMProviderType.BETTER_AUTH
        elif raw_type in ("disabled", "none", "off", "bypass"):
            provider_type = IAMProviderType.DISABLED

        better_auth_url = os.getenv("CMP_BETTER_AUTH_URL", "http://localhost:3000/api/auth")
        better_auth_api_key = os.getenv("CMP_BETTER_AUTH_API_KEY")
        cache_ttl = int(os.getenv("CMP_IAM_CACHE_TTL", "300"))

        return IAMConfig(
            provider_type=provider_type,
            better_auth_url=better_auth_url,
            better_auth_api_key=better_auth_api_key,
            session_cache_ttl_seconds=cache_ttl,
        )

    @classmethod
    def _build_provider(cls, config: IAMConfig) -> BaseIAMProvider:
        """Construct the designated provider instance."""
        p_type = config.provider_type
        logger.info(f"Instantiating IAM provider: {p_type.value}")

        if p_type == IAMProviderType.BETTER_AUTH:
            return BetterAuthIAMProvider(
                base_url=config.better_auth_url,
                api_key=config.better_auth_api_key,
                cache_ttl_seconds=config.session_cache_ttl_seconds,
            )
        elif p_type == IAMProviderType.DISABLED:
            return DisabledIAMProvider()
        else:
            return LocalDuckDBIAMProvider(db_path=config.duckdb_path)

    @property
    def provider(self) -> BaseIAMProvider:
        """Get the active IAM provider instance."""
        return self._provider

    @property
    def is_enabled(self) -> bool:
        """Whether authentication gating is active."""
        return self._provider.is_enabled

    @property
    def provider_name(self) -> str:
        """Name of the active provider."""
        return self._provider.provider_name

    def set_provider(self, provider: BaseIAMProvider) -> None:
        """
        Manually inject a custom or mock IAM provider.
        """
        self._provider = provider
        logger.info(f"Swapped active IAM provider to {provider.provider_name}")

    def reset_provider(self) -> None:
        """Reset provider to default environment configuration."""
        self._config = self._resolve_config_from_env()
        self._provider = self._build_provider(self._config)

    # --- Unified IAM Operations ---

    def authenticate(self, credentials: UserCredentials) -> AuthResult:
        """Authenticate user credentials."""
        return self._provider.authenticate(credentials)

    def verify_user(self, username: str, password: str) -> tuple[bool, str]:
        """
        Convenience verification method returning (success, message).
        Maintains complete drop-in compatibility with legacy DBManager calls.
        """
        res = self._provider.authenticate(UserCredentials(username=username, password=password))
        return res.success, res.message

    def create_user(
        self,
        username: str,
        password: str,
        role: UserRole = UserRole.RESEARCHER,
        auto_activate: bool = False,
    ) -> tuple[bool, str]:
        """Provision a new user account."""
        return self._provider.create_user(
            username=username,
            password=password,
            role=role,
            auto_activate=auto_activate,
        )

    def verify_session(self, token: str) -> AuthResult:
        """Verify an existing session token."""
        return self._provider.verify_session(token)

    def get_user(self, username: str) -> UserProfile | None:
        """Fetch user profile."""
        return self._provider.get_user(username)

    def list_users(self) -> list[UserProfile]:
        """List registered user profiles."""
        return self._provider.list_users()

    def get_all_users(self) -> list[str]:
        """Return list of all registered usernames."""
        return [u.username for u in self._provider.list_users()]

    def get_all_users_with_status(self) -> list[dict[str, Any]]:
        """
        Return structured dictionary list of all users with status.
        Maintains complete drop-in compatibility with legacy DBManager calls.
        """
        users = self._provider.list_users()
        result: list[dict[str, Any]] = []
        for u in users:
            result.append({
                "username": u.username,
                "is_active": u.is_active,
                "created_at": u.created_at,
                "status": "Active" if u.is_active else "Pending Approval",
                "role": u.role,
            })
        return result

    def activate_user(self, username: str) -> tuple[bool, str]:
        """Activate user account."""
        return self._provider.activate_user(username)

    def deactivate_user(self, username: str) -> tuple[bool, str]:
        """Deactivate user account."""
        return self._provider.deactivate_user(username)

    def delete_user(self, username: str) -> tuple[bool, str]:
        """Delete user account."""
        return self._provider.delete_user(username)

    def update_user_password(self, username: str, new_password: str) -> tuple[bool, str]:
        """Update user credentials."""
        return self._provider.update_password(username, new_password)


# Global singleton instance and accessor
_GLOBAL_IAM_MANAGER: IAMManager | None = None


def get_iam_manager(config: IAMConfig | None = None) -> IAMManager:
    """
    Retrieve or initialize the global IAMManager instance.
    """
    global _GLOBAL_IAM_MANAGER
    if _GLOBAL_IAM_MANAGER is None or config is not None:
        _GLOBAL_IAM_MANAGER = IAMManager(config=config)
    return _GLOBAL_IAM_MANAGER
