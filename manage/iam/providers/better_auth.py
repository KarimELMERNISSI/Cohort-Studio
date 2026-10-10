"""
Better Auth Remote IAM Provider.

Enables seamless integration with modern authentication frameworks such as
Better Auth (and compatible RESTful identity microservices). Features connection
pooling, short-lived in-memory TTL session caching to eliminate UI latency, and
resilient error handling.
"""

import logging
import time
from typing import Any

import requests

from ..base import BaseIAMProvider
from ..schemas import AuthResult, UserCredentials, UserProfile, UserRole

logger = logging.getLogger(__name__)


class BetterAuthIAMProvider(BaseIAMProvider):
    """
    Remote IAM Provider integrating with Better Auth or compatible REST IAM services.
    
    Includes an in-memory TTL cache to ensure that Streamlit rerun cycles do not
    degrade performance with repeated synchronous network requests.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        cache_ttl_seconds: int = 300,
        request_timeout: float = 3.0,
    ) -> None:
        """
        Initialize the Better Auth provider.

        Args:
            base_url: Base endpoint URL of the Better Auth service (e.g., "http://localhost:3000/api/auth").
            api_key: Optional Bearer or service API key for administrative queries.
            cache_ttl_seconds: Duration to cache verified sessions in memory (default 300s).
            request_timeout: HTTP request timeout in seconds (default 3.0s).
        """
        self.base_url = (base_url or "http://localhost:3000/api/auth").rstrip("/")
        self.api_key = api_key
        self.cache_ttl_seconds = cache_ttl_seconds
        self.request_timeout = request_timeout

        # Persistent session for HTTP keep-alive connection reuse
        self._session = requests.Session()
        if self.api_key:
            self._session.headers.update({"Authorization": f"Bearer {self.api_key}"})

        # In-memory session cache: token_or_key -> (expiry_timestamp, AuthResult)
        self._session_cache: dict[str, tuple[float, AuthResult]] = {}

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return "better_auth"

    def _get_cached_auth(self, cache_key: str) -> AuthResult | None:
        """Check in-memory TTL cache to avoid redundant network round-trips."""
        if cache_key in self._session_cache:
            expiry, result = self._session_cache[cache_key]
            if time.time() < expiry:
                return result
            # Cache entry expired
            del self._session_cache[cache_key]
        return None

    def _set_cached_auth(self, cache_key: str, result: AuthResult) -> None:
        """Store verified authentication result in TTL cache."""
        if result.success:
            self._session_cache[cache_key] = (time.time() + self.cache_ttl_seconds, result)

    def health_check(self) -> tuple[bool, str]:
        """Verify connectivity to the Better Auth service."""
        try:
            resp = self._session.get(f"{self.base_url}/health", timeout=self.request_timeout)
            if resp.status_code == 200:
                return True, "Better Auth service is healthy and reachable"
            return False, f"Better Auth returned status code {resp.status_code}"
        except Exception as e:
            return False, f"Better Auth connection failed: {e}"

    def authenticate(self, credentials: UserCredentials) -> AuthResult:
        """
        Authenticate credentials via Better Auth sign-in endpoint.
        """
        cache_key = f"auth:{credentials.username}:{hash(credentials.password)}"
        cached = self._get_cached_auth(cache_key)
        if cached:
            return cached

        try:
            payload = {
                "username": credentials.username,
                "email": credentials.username if "@" in credentials.username else f"{credentials.username}@local.cohort",
                "password": credentials.password,
            }
            # Standard Better Auth sign-in endpoint
            resp = self._session.post(
                f"{self.base_url}/sign-in/username",
                json=payload,
                timeout=self.request_timeout,
            )

            # Fallback to email endpoint if username endpoint returns 404
            if resp.status_code == 404:
                resp = self._session.post(
                    f"{self.base_url}/sign-in/email",
                    json=payload,
                    timeout=self.request_timeout,
                )

            if resp.status_code == 200:
                data = resp.json()
                user_data = data.get("user", {})
                token = data.get("token") or data.get("session", {}).get("token")

                role_str = str(user_data.get("role", "researcher")).lower()
                role_enum = UserRole.ADMIN if role_str == "admin" else UserRole.RESEARCHER
                is_active = bool(user_data.get("isActive", True))

                if not is_active:
                    return AuthResult(
                        success=False,
                        message="Account pending approval by administrator",
                    )

                profile = UserProfile(
                    username=user_data.get("username", credentials.username),
                    email=user_data.get("email"),
                    role=role_enum,
                    is_active=is_active,
                    created_at=user_data.get("createdAt"),
                    metadata=user_data,
                )
                auth_result = AuthResult(
                    success=True,
                    user=profile,
                    token=token or f"better_auth_{credentials.username}",
                    message="Login successful via Better Auth",
                )
                self._set_cached_auth(cache_key, auth_result)
                return auth_result

            elif resp.status_code in (401, 403):
                return AuthResult(success=False, message="Invalid credentials or account inactive")
            else:
                return AuthResult(
                    success=False,
                    message=f"Better Auth server responded with status {resp.status_code}",
                )

        except requests.RequestException as e:
            logger.warning(f"Better Auth network error: {e}")
            return AuthResult(
                success=False,
                message=f"Better Auth service unreachable: {e}",
            )
        except Exception as e:
            logger.error(f"Better Auth unexpected error: {e}")
            return AuthResult(success=False, message=f"Authentication error: {e}")

    def create_user(
        self,
        username: str,
        password: str,
        role: UserRole = UserRole.RESEARCHER,
        auto_activate: bool = False,
    ) -> tuple[bool, str]:
        """
        Provision a new user account through Better Auth sign-up endpoint.
        """
        try:
            payload = {
                "username": username,
                "email": username if "@" in username else f"{username}@local.cohort",
                "password": password,
                "role": role.value,
                "isActive": auto_activate or (username == "admin"),
            }
            resp = self._session.post(
                f"{self.base_url}/sign-up/email",
                json=payload,
                timeout=self.request_timeout,
            )
            if resp.status_code in (200, 201):
                if auto_activate or username == "admin":
                    return True, "User created and activated successfully via Better Auth"
                return True, "Account created! Awaiting administrator approval."
            elif resp.status_code == 409 or "exists" in resp.text.lower():
                return False, "Username or email already exists in Better Auth"
            return False, f"Failed to register user: HTTP {resp.status_code} - {resp.text}"
        except Exception as e:
            return False, f"Better Auth connection error during sign-up: {e}"

    def verify_session(self, token: str) -> AuthResult:
        """
        Verify existing session token using Better Auth get-session endpoint.
        """
        cache_key = f"session:{token}"
        cached = self._get_cached_auth(cache_key)
        if cached:
            return cached

        try:
            headers = {"Authorization": f"Bearer {token}"}
            resp = self._session.get(
                f"{self.base_url}/get-session",
                headers=headers,
                timeout=self.request_timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                user_data = data.get("user", {})
                role_str = str(user_data.get("role", "researcher")).lower()
                role_enum = UserRole.ADMIN if role_str == "admin" else UserRole.RESEARCHER

                profile = UserProfile(
                    username=user_data.get("username", user_data.get("email", "unknown")),
                    email=user_data.get("email"),
                    role=role_enum,
                    is_active=bool(user_data.get("isActive", True)),
                    created_at=user_data.get("createdAt"),
                )
                auth_result = AuthResult(
                    success=True,
                    user=profile,
                    token=token,
                    message="Session valid",
                )
                self._set_cached_auth(cache_key, auth_result)
                return auth_result
            return AuthResult(success=False, message="Session expired or invalid")
        except Exception as e:
            return AuthResult(success=False, message=f"Session verification error: {e}")

    def get_user(self, username: str) -> UserProfile | None:
        """Fetch user profile from Better Auth admin endpoint."""
        try:
            resp = self._session.get(
                f"{self.base_url}/admin/users/{username}",
                timeout=self.request_timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                role_str = str(data.get("role", "researcher")).lower()
                return UserProfile(
                    username=data.get("username", username),
                    email=data.get("email"),
                    role=UserRole.ADMIN if role_str == "admin" else UserRole.RESEARCHER,
                    is_active=bool(data.get("isActive", True)),
                    created_at=data.get("createdAt"),
                )
            return None
        except Exception as e:
            logger.error(f"Error fetching user from Better Auth: {e}")
            return None

    def list_users(self) -> list[UserProfile]:
        """Fetch user list from Better Auth admin endpoint."""
        try:
            resp = self._session.get(
                f"{self.base_url}/admin/users",
                timeout=self.request_timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                users_raw = data.get("users", data) if isinstance(data, dict) else data
                profiles: list[UserProfile] = []
                for u in users_raw:
                    role_str = str(u.get("role", "researcher")).lower()
                    profiles.append(
                        UserProfile(
                            username=u.get("username", u.get("email", "unknown")),
                            email=u.get("email"),
                            role=UserRole.ADMIN if role_str == "admin" else UserRole.RESEARCHER,
                            is_active=bool(u.get("isActive", True)),
                            created_at=u.get("createdAt"),
                        )
                    )
                return profiles
            return []
        except Exception as e:
            logger.error(f"Error listing users from Better Auth: {e}")
            return []

    def activate_user(self, username: str) -> tuple[bool, str]:
        """Activate user via Better Auth admin API."""
        try:
            resp = self._session.post(
                f"{self.base_url}/admin/users/{username}/activate",
                json={"isActive": True},
                timeout=self.request_timeout,
            )
            if resp.status_code in (200, 204):
                return True, f"User '{username}' activated successfully in Better Auth"
            return False, f"Better Auth activation error: HTTP {resp.status_code}"
        except Exception as e:
            return False, f"Failed to activate user in Better Auth: {e}"

    def deactivate_user(self, username: str) -> tuple[bool, str]:
        """Deactivate user via Better Auth admin API."""
        if username == "admin":
            return False, "Cannot deactivate administrator account"
        try:
            resp = self._session.post(
                f"{self.base_url}/admin/users/{username}/deactivate",
                json={"isActive": False},
                timeout=self.request_timeout,
            )
            if resp.status_code in (200, 204):
                return True, f"User '{username}' deactivated in Better Auth"
            return False, f"Better Auth deactivation error: HTTP {resp.status_code}"
        except Exception as e:
            return False, f"Failed to deactivate user in Better Auth: {e}"

    def delete_user(self, username: str) -> tuple[bool, str]:
        """Delete user account in Better Auth directory."""
        if username == "admin":
            return False, "Cannot delete administrator account"
        try:
            resp = self._session.delete(
                f"{self.base_url}/admin/users/{username}",
                timeout=self.request_timeout,
            )
            if resp.status_code in (200, 204):
                return True, f"User '{username}' deleted from Better Auth"
            return False, f"Better Auth deletion error: HTTP {resp.status_code}"
        except Exception as e:
            return False, f"Failed to delete user in Better Auth: {e}"

    def update_password(self, username: str, new_password: str) -> tuple[bool, str]:
        """Reset user password via Better Auth admin API."""
        try:
            resp = self._session.post(
                f"{self.base_url}/admin/users/{username}/set-password",
                json={"newPassword": new_password},
                timeout=self.request_timeout,
            )
            if resp.status_code in (200, 204):
                return True, f"Password updated for user '{username}' in Better Auth"
            return False, f"Better Auth password update error: HTTP {resp.status_code}"
        except Exception as e:
            return False, f"Failed to update password in Better Auth: {e}"
