"""
Authorization and credential-policy helpers shared by the UI and the CLI.

Privileges are derived from the role stored by the IAM provider (``UserProfile.role``), never from
the username string (audit finding E5).
"""

from __future__ import annotations

import re

from utils.settings import get_settings

from .schemas import UserRole

# the CMP_MIN_PASSWORD_LENGTH setting, read once; tests patch this attribute
MIN_PASSWORD_LENGTH = get_settings().iam.min_password_length
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,31}$")


def is_admin_role(role: str | UserRole | None) -> bool:
    """True when the role value denotes an administrator."""
    if role is None:
        return False
    value = role.value if isinstance(role, UserRole) else str(role)
    return value.lower() == UserRole.ADMIN.value


def validate_username(username: str | None) -> str | None:
    """Return an error message when the username is unacceptable, else None."""
    if not username or not USERNAME_PATTERN.match(username.strip()):
        return "Username must be 3-32 characters: letters, digits, '.', '_' or '-' (no spaces)."
    return None


def validate_password(password: str | None) -> str | None:
    """Return an error message when the password is unacceptable, else None."""
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    return None


def validate_new_credentials(username: str | None, password: str | None, confirm: str | None) -> str | None:
    """Validate a sign-up form; returns the first problem found or None."""
    problem = validate_username(username) or validate_password(password)
    if problem:
        return problem
    if password != confirm:
        return "Passwords do not match."
    return None
