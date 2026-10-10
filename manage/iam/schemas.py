"""
Identity and Access Management (IAM) Schemas and Data Models.

Defines strict Pydantic models for user profiles, credentials, authentication
results, and IAM provider configurations.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class IAMProviderType(str, Enum):
    """Supported IAM provider backends."""
    LOCAL = "local"
    BETTER_AUTH = "better_auth"
    DISABLED = "disabled"
    CUSTOM = "custom"


class UserRole(str, Enum):
    """Clinical user authorization roles."""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    AUDITOR = "auditor"
    GUEST = "guest"


class UserProfile(BaseModel):
    """
    Structured representation of a registered platform user.
    """
    username: str
    email: str | None = None
    role: UserRole = UserRole.RESEARCHER
    is_active: bool = False
    created_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": True}


class UserCredentials(BaseModel):
    """User authentication credentials for sign-in."""
    username: str
    password: str


class AuthResult(BaseModel):
    """Result of an authentication or verification attempt."""
    success: bool
    user: UserProfile | None = None
    token: str | None = None
    message: str = ""


class IAMConfig(BaseModel):
    """
    Configuration specification for the IAM subsystem.
    """
    provider_type: IAMProviderType = IAMProviderType.LOCAL
    duckdb_path: str | None = None
    better_auth_url: str | None = None
    better_auth_api_key: str | None = None
    session_cache_ttl_seconds: int = 300
    auto_activate_admin: bool = True
