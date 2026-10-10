"""
Disabled IAM Provider (Bypass / No-Auth Mode).

Allows seamless operation in single-researcher offline workstations or automated
pipelines where user authentication is explicitly disabled. Automatically provides
a pre-authorized administrator session without gating the application.
"""

from ..base import BaseIAMProvider
from ..schemas import AuthResult, UserCredentials, UserProfile, UserRole


class DisabledIAMProvider(BaseIAMProvider):
    """
    Bypass provider that allows unrestricted access as a local researcher.
    """

    def __init__(self, default_username: str = "local_researcher") -> None:
        self.default_username = default_username
        self._default_user = UserProfile(
            username=default_username,
            role=UserRole.ADMIN,
            is_active=True,
            created_at=None,
        )

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return "disabled"

    @property
    def is_enabled(self) -> bool:
        """Authentication is disabled."""
        return False

    def authenticate(self, credentials: UserCredentials) -> AuthResult:
        """Always return active authorized session."""
        username = credentials.username or self.default_username
        return AuthResult(
            success=True,
            user=UserProfile(username=username, role=UserRole.ADMIN, is_active=True),
            token="disabled_auth_token",
            message="IAM is disabled. Granted local researcher access.",
        )

    def create_user(
        self,
        username: str,
        password: str,
        role: UserRole = UserRole.RESEARCHER,
        auto_activate: bool = False,
    ) -> tuple[bool, str]:
        """No-op user provisioning."""
        return True, f"IAM disabled: Simulated user creation for '{username}'"

    def verify_session(self, token: str) -> AuthResult:
        """Always validate token."""
        return AuthResult(
            success=True,
            user=self._default_user,
            token="disabled_auth_token",
            message="Session active (IAM disabled)",
        )

    def get_user(self, username: str) -> UserProfile | None:
        """Return default user profile."""
        return UserProfile(username=username, role=UserRole.ADMIN, is_active=True)

    def list_users(self) -> list[UserProfile]:
        """Return default user."""
        return [self._default_user]

    def activate_user(self, username: str) -> tuple[bool, str]:
        return True, f"IAM disabled: User '{username}' is already active"

    def deactivate_user(self, username: str) -> tuple[bool, str]:
        return False, "Cannot deactivate users in disabled IAM mode"

    def delete_user(self, username: str) -> tuple[bool, str]:
        return False, "Cannot delete users in disabled IAM mode"

    def update_password(self, username: str, new_password: str) -> tuple[bool, str]:
        return True, f"IAM disabled: Simulated password update for '{username}'"
