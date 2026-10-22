"""
Base Abstract Interface for Identity and Access Management (IAM) Providers.

Defines the contract that all authentication and user management backends
must implement, enabling framework-agnostic switching between local, remote,
or disabled authentication.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

from core.registry import Availability

from .schemas import AuthResult, IAMConfig, UserCredentials, UserProfile, UserRole


class BaseIAMProvider(ABC):
    """
    Abstract Base Class for IAM Providers.

    Subclasses implement credential verification, user provisioning, role
    management, and session token resolution against specific backends.
    """

    # its name in CMP_IAM_PROVIDER (manage.iam.providers.IAM_PROVIDERS), stored in deployments: never renamed
    name: ClassVar[str] = ""
    display_name: ClassVar[str] = ""  # in sentences: the Users page

    @classmethod
    def from_config(cls, config: IAMConfig) -> "BaseIAMProvider":
        """The provider the configuration describes (IAMManager builds the configured one so). A provider not
        registered in IAM_PROVIDERS is installed with IAMManager.set_provider instead."""
        raise NotImplementedError(f"{cls.__name__} is installed with IAMManager.set_provider")

    @classmethod
    def availability(cls) -> Availability:
        """Whether it can run here: its libraries are the application's own (a Better Auth service that does not
        answer is said at sign-in)."""
        return Availability(True)

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique name identifier of the IAM provider."""

    @property
    def is_enabled(self) -> bool:
        """
        Indicates whether authentication is actively enforced.
        Default is True; disabled/bypass providers override to False.
        """
        return True

    @abstractmethod
    def authenticate(self, credentials: UserCredentials) -> AuthResult:
        """
        Authenticate user credentials and evaluate account activation status.

        Args:
            credentials: UserCredentials containing username and password.

        Returns:
            AuthResult with success status, user profile, and informative message.
        """

    @abstractmethod
    def create_user(
        self,
        username: str,
        password: str,
        role: UserRole = UserRole.RESEARCHER,
        auto_activate: bool = False,
    ) -> tuple[bool, str]:
        """
        Provision a new user account.

        Args:
            username: Unique alphanumeric username.
            password: Raw password string to be securely hashed.
            role: Assigned clinical user role.
            auto_activate: Whether account is active immediately without admin review.

        Returns:
            Tuple of (success_flag, status_message).
        """

    @abstractmethod
    def verify_session(self, token: str) -> AuthResult:
        """
        Validate an active bearer token or session identifier.

        Args:
            token: Session token or bearer string.

        Returns:
            AuthResult with user details if token is valid.
        """

    @abstractmethod
    def get_user(self, username: str) -> UserProfile | None:
        """
        Retrieve structured user profile by username.

        Args:
            username: Target user account name.

        Returns:
            UserProfile if found, else None.
        """

    @abstractmethod
    def list_users(self) -> list[UserProfile]:
        """
        List all registered platform users with activation and role status.

        Returns:
            List of UserProfile instances.
        """

    @abstractmethod
    def activate_user(self, username: str) -> tuple[bool, str]:
        """
        Activate a pending user account (administrator authorization).

        Args:
            username: Target user account name.

        Returns:
            Tuple of (success_flag, status_message).
        """

    @abstractmethod
    def deactivate_user(self, username: str) -> tuple[bool, str]:
        """
        Suspend an active user account.

        Args:
            username: Target user account name.

        Returns:
            Tuple of (success_flag, status_message).
        """

    @abstractmethod
    def delete_user(self, username: str) -> tuple[bool, str]:
        """
        Permanently remove a user account from the directory.

        Args:
            username: Target user account name.

        Returns:
            Tuple of (success_flag, status_message).
        """

    @abstractmethod
    def update_password(self, username: str, new_password: str) -> tuple[bool, str]:
        """
        Update credentials for an existing account.

        Args:
            username: Target user account name.
            new_password: New raw password string.

        Returns:
            Tuple of (success_flag, status_message).
        """
