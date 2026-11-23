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
# what the hash of a local account takes (bcrypt): a longer password was refused by the hash, in a programmer's words
MAX_PASSWORD_BYTES = 72
CANNOT_BE_STORED = "Password holds a character that cannot be stored."
# a new name starts and ends with a letter or a digit: it is its own workspace folder (core.workspace.safe_dirname,
# which gives any other name a code of its own)
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,30}[A-Za-z0-9]$")


def is_admin_role(role: str | UserRole | None) -> bool:
    """True when the role value denotes an administrator."""
    if role is None:
        return False
    value = role.value if isinstance(role, UserRole) else str(role)
    return value.lower() == UserRole.ADMIN.value


def validate_username(username: str | None) -> str | None:
    """Return an error message when the username is unacceptable, else None."""
    if not username or not USERNAME_PATTERN.match(username.strip()):
        return (
            "Username must be 3-32 characters: letters, digits, '.', '_' or '-' (no spaces), beginning and ending "
            "with a letter or a digit."
        )
    return None


def validate_password(password: str | None) -> str | None:
    """
    Return an error message when the password is unacceptable, else None: shorter than the minimum (counted in
    characters), made of spaces only, or longer than the hash takes. The one rule of every way a password is set:
    ``IAMManager`` applies it too, so that no caller can forget it. The words never quote the password.
    """
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    if not password.strip():
        return "Password cannot be made of spaces only."
    try:
        size = len(password.encode("utf-8"))
    except UnicodeEncodeError:  # bytes that were no text where it was read (an environment variable): by no word of it
        return CANNOT_BE_STORED
    if chr(0) in password:  # a file in another encoding read as text (UTF-16 without its mark): nobody types it
        return CANNOT_BE_STORED
    if size > MAX_PASSWORD_BYTES:
        return (
            f"Password is too long: at most {MAX_PASSWORD_BYTES} bytes (a letter with an accent counts for two or "
            "more)."
        )
    return None


def validate_new_credentials(username: str | None, password: str | None, confirm: str | None) -> str | None:
    """Validate a sign-up form; returns the first problem found or None."""
    problem = validate_username(username) or validate_password(password)
    if problem:
        return problem
    if password != confirm:
        return "Passwords do not match."
    return None
