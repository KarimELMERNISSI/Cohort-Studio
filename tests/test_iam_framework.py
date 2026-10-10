"""
Comprehensive Unit Tests for the Modular IAM Subsystem.

Tests LocalDuckDB, Better Auth (remote with in-memory TTL caching),
Disabled provider, and the IAMManager facade.
"""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from manage.iam import (
    AuthResult,
    BaseIAMProvider,
    BetterAuthIAMProvider,
    DisabledIAMProvider,
    IAMConfig,
    IAMManager,
    IAMProviderType,
    LocalDuckDBIAMProvider,
    UserCredentials,
    UserProfile,
    UserRole,
    get_iam_manager,
)


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    """Provide a path to a temporary isolated DuckDB file."""
    return str(tmp_path / "test_users.duckdb")


@pytest.fixture
def local_provider(temp_db_path: str) -> LocalDuckDBIAMProvider:
    """Instantiate an isolated LocalDuckDBIAMProvider."""
    return LocalDuckDBIAMProvider(db_path=temp_db_path)


class TestLocalDuckDBIAMProvider:
    """Test suite for LocalDuckDBIAMProvider."""

    def test_admin_auto_activation(self, local_provider: LocalDuckDBIAMProvider) -> None:
        """Admin user is created and automatically activated."""
        ok, msg = local_provider.create_user("admin", "AdminPassword123!")
        assert ok is True

        res = local_provider.authenticate(UserCredentials(username="admin", password="AdminPassword123!"))
        assert res.success is True
        assert res.user is not None
        assert res.user.role == UserRole.ADMIN
        assert res.user.is_active is True

    def test_standard_user_activation_flow(self, local_provider: LocalDuckDBIAMProvider) -> None:
        """Standard users require administrator activation before login."""
        ok, _ = local_provider.create_user("researcher_dr_lee", "Password456!")
        assert ok is True

        # Attempt login before approval
        res_before = local_provider.authenticate(
            UserCredentials(username="researcher_dr_lee", password="Password456!")
        )
        assert res_before.success is False
        assert "pending approval" in res_before.message.lower()

        # Activate user
        act_ok, _ = local_provider.activate_user("researcher_dr_lee")
        assert act_ok is True

        # Attempt login after approval
        res_after = local_provider.authenticate(
            UserCredentials(username="researcher_dr_lee", password="Password456!")
        )
        assert res_after.success is True
        assert res_after.user is not None
        assert res_after.user.is_active is True

    def test_duplicate_username_rejection(self, local_provider: LocalDuckDBIAMProvider) -> None:
        """Creating an existing username must fail."""
        local_provider.create_user("duplicate_user", "Pass1")
        ok, msg = local_provider.create_user("duplicate_user", "Pass2")
        assert ok is False
        assert "already exists" in msg.lower()

    def test_admin_lockout_protection(self, local_provider: LocalDuckDBIAMProvider) -> None:
        """Admins cannot be deactivated or deleted."""
        local_provider.create_user("admin", "secret")
        deact_ok, deact_msg = local_provider.deactivate_user("admin")
        assert deact_ok is False
        assert "cannot deactivate admin" in deact_msg.lower()

        del_ok, del_msg = local_provider.delete_user("admin")
        assert del_ok is False
        assert "cannot delete admin" in del_msg.lower()

    def test_update_password_and_token_verification(self, local_provider: LocalDuckDBIAMProvider) -> None:
        """Password updates take immediate effect and tokens can be verified."""
        local_provider.create_user("user_tom", "OldPass1", auto_activate=True)

        upd_ok, _ = local_provider.update_password("user_tom", "NewPass2")
        assert upd_ok is True

        # Old password fails
        assert local_provider.authenticate(UserCredentials(username="user_tom", password="OldPass1")).success is False

        # New password succeeds
        res = local_provider.authenticate(UserCredentials(username="user_tom", password="NewPass2"))
        assert res.success is True
        assert res.token is not None

        # Session token verification
        sess_res = local_provider.verify_session(res.token)
        assert sess_res.success is True
        assert sess_res.user is not None
        assert sess_res.user.username == "user_tom"


class TestDisabledIAMProvider:
    """Test suite for DisabledIAMProvider (bypass mode)."""

    def test_bypass_behavior(self) -> None:
        """Disabled provider indicates is_enabled=False and auto-authenticates."""
        provider = DisabledIAMProvider()
        assert provider.is_enabled is False
        assert provider.provider_name == "disabled"

        res = provider.authenticate(UserCredentials(username="", password=""))
        assert res.success is True
        assert res.user is not None
        assert res.user.role == UserRole.ADMIN

        users = provider.list_users()
        assert len(users) == 1
        assert users[0].is_active is True


class TestBetterAuthIAMProvider:
    """Test suite for BetterAuthIAMProvider with mock HTTP and TTL caching."""

    @patch("requests.Session.post")
    def test_authenticate_success_and_ttl_cache_hit(self, mock_post: MagicMock) -> None:
        """
        Verifies that Better Auth authenticates via HTTP, and a subsequent call
        within TTL returns from in-memory cache without hitting the network.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "token": "bearer_jwt_better_auth_token_123",
            "user": {
                "username": "dr_watson",
                "email": "watson@hospital.org",
                "role": "researcher",
                "isActive": True,
                "createdAt": "2026-10-10T12:00:00Z",
            },
        }
        mock_post.return_value = mock_response

        provider = BetterAuthIAMProvider(base_url="http://mock-auth:3000/api/auth", cache_ttl_seconds=60)

        # First call: hits HTTP endpoint
        creds = UserCredentials(username="dr_watson", password="SecurePassword!")
        res1 = provider.authenticate(creds)

        assert res1.success is True
        assert res1.user is not None
        assert res1.user.username == "dr_watson"
        assert res1.token == "bearer_jwt_better_auth_token_123"
        assert mock_post.call_count == 1

        # Second call: must hit in-memory TTL cache (no additional HTTP call!)
        res2 = provider.authenticate(creds)
        assert res2.success is True
        assert res2.user is not None
        assert res2.user.username == "dr_watson"
        # call_count remains 1 because network request was skipped!
        assert mock_post.call_count == 1

    @patch("requests.Session.post")
    def test_authenticate_network_failure_resilience(self, mock_post: MagicMock) -> None:
        """Better Auth provider fails gracefully returning error message if network drops."""
        mock_post.side_effect = requests.RequestException("Connection refused")

        provider = BetterAuthIAMProvider(base_url="http://offline-auth:3000/api/auth")
        res = provider.authenticate(UserCredentials(username="dr_watson", password="pass"))

        assert res.success is False
        assert "unreachable" in res.message.lower()

    @patch("requests.Session.post")
    def test_create_user_endpoint(self, mock_post: MagicMock) -> None:
        """Tests user creation via Better Auth sign-up endpoint."""
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_post.return_value = mock_resp

        provider = BetterAuthIAMProvider(base_url="http://mock-auth:3000/api/auth")
        ok, msg = provider.create_user("new_user", "password123")
        assert ok is True
        assert "created" in msg.lower()


class TestIAMManager:
    """Test suite for IAMManager facade and environment configuration."""

    def test_manager_local_default(self, temp_db_path: str) -> None:
        """Default configuration instantiates LocalDuckDBIAMProvider."""
        config = IAMConfig(provider_type=IAMProviderType.LOCAL, duckdb_path=temp_db_path)
        manager = IAMManager(config=config)

        assert manager.is_enabled is True
        assert manager.provider_name == "local_duckdb"

        ok, _ = manager.create_user("admin", "AdminSecret!", auto_activate=True)
        assert ok is True

        valid, _ = manager.verify_user("admin", "AdminSecret!")
        assert valid is True

        all_users = manager.get_all_users()
        assert "admin" in all_users

    def test_manager_disabled_mode(self) -> None:
        """Disabled provider mode operates seamlessly."""
        config = IAMConfig(provider_type=IAMProviderType.DISABLED)
        manager = IAMManager(config=config)

        assert manager.is_enabled is False
        assert manager.provider_name == "disabled"

        valid, _ = manager.verify_user("anyone", "anypass")
        assert valid is True

    def test_manager_environment_switching(self, monkeypatch: pytest.MonkeyPatch, temp_db_path: str) -> None:
        """Environment variable CMP_IAM_PROVIDER alters instantiated provider."""
        monkeypatch.setenv("CMP_IAM_PROVIDER", "disabled")
        manager_disabled = IAMManager()
        assert manager_disabled.is_enabled is False

        monkeypatch.setenv("CMP_IAM_PROVIDER", "better_auth")
        monkeypatch.setenv("CMP_BETTER_AUTH_URL", "http://test-server:3000/api/auth")
        manager_better = IAMManager()
        assert manager_better.provider_name == "better_auth"
