"""
IAM Manager and Factory.

Provides a unified, framework-agnostic facade for identity and access management.
Dynamically resolves and instantiates the active IAM provider (Local DuckDB,
Better Auth, or Disabled) based on environment configuration or runtime parameters.
"""

import logging
import os
from typing import Any

from . import setup_token
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

    @staticmethod
    def _setup_dir() -> str:
        from utils.data_paths import get_data_root

        return get_data_root()

    def _has_active_admin(self) -> bool:
        return any(str(u.role) == UserRole.ADMIN.value and u.is_active for u in self._provider.list_users())

    def setup_pending(self) -> bool:
        """True when the first administrator must be created with the setup token of the server log."""
        if not self._provider.is_enabled or not setup_token.pending(self._setup_dir()):
            return False
        try:
            return not self._has_active_admin()
        except Exception:
            return False

    def issue_setup_token(self) -> str | None:
        """A new first-run setup token (the previous one stops working); None when no setup is needed."""
        if not self._provider.is_enabled or self._has_active_admin():
            return None
        return setup_token.issue(self._setup_dir())

    def create_first_admin(self, token: str, username: str, password: str) -> tuple[bool, str]:
        """Create the first administrator with the one-time setup token (follow-up programme M4)."""
        if not self.setup_pending():
            return False, "No first-run setup is pending: an administrator exists, or no setup token was issued."
        if not setup_token.check(self._setup_dir(), token):
            return False, "The setup token is not valid. Copy it from the server log (the most recent one)."
        ok, msg = self._provider.create_user(username, password, role=UserRole.ADMIN, auto_activate=True)
        if ok:
            setup_token.clear(self._setup_dir())
            logger.info("First administrator '%s' created with the setup token.", username)
        return ok, msg

    def ensure_bootstrap_admin(self) -> tuple[bool, str]:
        """
        Create the initial administrator when no active administrator exists.

        Credentials come from CMP_ADMIN_USERNAME (default 'admin') and CMP_ADMIN_PASSWORD or
        CMP_ADMIN_PASSWORD_FILE. Without them, a one-time setup token is written to the server log and the
        login page creates the first administrator with it (manage/iam/setup_token.py). Nothing happens when
        the IAM provider is disabled or when an active administrator already exists.
        """
        if not self._provider.is_enabled:
            return False, "IAM disabled; no bootstrap needed"
        try:
            users = self._provider.list_users()
        except Exception as exc:  # provider unreachable: never block application start-up
            logger.warning("Admin bootstrap skipped, could not list users: %s", exc)
            return False, f"Could not list users: {exc}"
        if any(str(u.role) == UserRole.ADMIN.value and u.is_active for u in users):
            setup_token.clear(self._setup_dir())  # a token left from before the administrator existed
            return False, "An active administrator already exists"

        username = os.getenv("CMP_ADMIN_USERNAME", "admin").strip()
        password = os.getenv("CMP_ADMIN_PASSWORD")
        password_file = os.getenv("CMP_ADMIN_PASSWORD_FILE")
        if not password and password_file and os.path.isfile(password_file):
            with open(password_file, encoding="utf-8") as handle:
                password = handle.read().strip()
        if not password:
            if setup_token.pending(self._setup_dir()):  # keep the token already in the log (server start-up)
                logger.warning(
                    "No administrator account exists. A first-run setup token is pending: it was written to the "
                    "server log when it was issued. `python manage_users.py setup-token` issues a new one."
                )
                return False, "Setup token pending"
            token = setup_token.issue(self._setup_dir())
            logger.warning(
                "No administrator account exists. Create the first administrator on the login page with this "
                "one-time setup token: %s (or set CMP_ADMIN_PASSWORD, or run "
                "`python manage_users.py create-admin <username>`).",
                token,
            )
            return False, "Setup token issued"
        if self._provider.get_user(username) is not None:
            logger.warning(
                "Bootstrap account '%s' exists but is not an active administrator; "
                "use `python manage_users.py` to activate it or create another administrator.",
                username,
            )
            return False, f"User '{username}' exists but is not an active administrator"
        ok, msg = self._provider.create_user(username, password, role=UserRole.ADMIN, auto_activate=True)
        if ok:
            logger.info("Bootstrap administrator '%s' created from environment configuration.", username)
        return ok, msg


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
