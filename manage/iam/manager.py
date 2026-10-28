"""
IAM Manager and Factory.

Provides a unified, framework-agnostic facade for identity and access management.
Dynamically resolves and instantiates the active IAM provider (Local DuckDB,
Better Auth, or Disabled) based on environment configuration or runtime parameters.
"""

import logging
import os
from typing import Any

from core.workspace import workspace_exists
from utils.logging_config import single_line
from utils.settings import Settings, get_settings

from . import setup_token
from .base import BaseIAMProvider
from .providers import IAM_PROVIDERS
from .schemas import (
    AuthResult,
    IAMConfig,
    IAMProviderType,
    UserCredentials,
    UserProfile,
    UserRole,
)

logger = logging.getLogger(__name__)


class IAMConfigurationError(ValueError):
    """The identity provider is misconfigured: the app must not start with another one in its place."""


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
    def _resolve_config_from_env(cls, settings: Settings | None = None) -> IAMConfig:
        """
        The IAM configuration of the settings (utils.settings; the current ones by default). An unknown
        CMP_IAM_PROVIDER raises IAMConfigurationError: a misspelt provider used to give local accounts without a word,
        so a server meant to sign users in elsewhere ran with its own accounts instead.
        """
        iam = (settings or get_settings()).iam
        raw_type = (iam.provider or "").strip() or "local"
        name = IAM_PROVIDERS.canonical(raw_type)  # the registered names and their aliases, in any case
        if name is None:  # the variable and the valid names, never the value (a pasted key?)
            raise IAMConfigurationError(f"{IAM_PROVIDERS.setting_problem(raw_type)}.")
        provider_type = IAMProviderType(name)

        key = iam.better_auth_api_key
        return IAMConfig(
            provider_type=provider_type,
            better_auth_url=iam.better_auth_url,
            better_auth_api_key=key.get_secret_value() if key is not None else None,
            session_cache_ttl_seconds=iam.cache_ttl,
        )

    @classmethod
    def _build_provider(cls, config: IAMConfig) -> BaseIAMProvider:
        """Construct the designated provider instance."""
        p_type = config.provider_type
        logger.info(f"Instantiating IAM provider: {p_type.value}")
        if p_type.value not in IAM_PROVIDERS.names():
            # CUSTOM gave local accounts in its place: a custom provider is installed with IAMManager.set_provider
            raise IAMConfigurationError(
                f"the identity provider {p_type.value!r} cannot be built from the configuration; install a custom "
                "provider with IAMManager.set_provider."
            )
        return IAM_PROVIDERS.get(p_type.value).from_config(config)

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

    @property
    def provider_type(self) -> IAMProviderType:
        """The configured provider (CMP_IAM_PROVIDER, or the configuration given)."""
        return self._config.provider_type

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
        *,
        reuse_folder: bool = False,
    ) -> tuple[bool, str]:
        """
        Provision a new user account. A name whose workspace folder exists is refused, with the message of a name taken
        (core.workspace.workspace_exists): the folder may hold another account's files, such as the folder an earlier
        version gave an account that has not signed in since. The server's operator may reuse it (``reuse_folder``,
        manage_users.py create-admin).
        """
        if not reuse_folder and workspace_exists(username):
            return False, "Username already exists"
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
            logger.info("First administrator '%s' created with the setup token.", single_line(username))
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

        iam = get_settings().iam
        username = iam.admin_username
        password = iam.admin_password.get_secret_value() if iam.admin_password is not None else None
        password_file = iam.admin_password_file
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
