"""Role checks and credential policy (audit E5)."""

from manage.iam.authz import (
    CANNOT_BE_STORED,
    MAX_PASSWORD_BYTES,
    MIN_PASSWORD_LENGTH,
    is_admin_role,
    validate_new_credentials,
    validate_password,
    validate_username,
)
from manage.iam.schemas import UserRole


def test_is_admin_role_accepts_enum_and_strings() -> None:
    assert is_admin_role(UserRole.ADMIN)
    assert is_admin_role("admin") and is_admin_role("ADMIN")
    assert not is_admin_role("researcher") and not is_admin_role(None) and not is_admin_role("")


def test_username_policy() -> None:
    assert validate_username("dr.lee_2") is None
    assert validate_username("ab") is not None  # too short
    assert validate_username("../etc") is not None  # path characters
    assert validate_username("name with space") is not None
    assert validate_username(None) is not None


def test_password_policy() -> None:
    assert MIN_PASSWORD_LENGTH >= 12
    assert validate_password("x" * MIN_PASSWORD_LENGTH) is None
    assert validate_password("x" * (MIN_PASSWORD_LENGTH - 1)) is not None
    assert validate_password("") is not None


def test_a_password_is_counted_in_characters_and_fits_the_hash() -> None:
    """The minimum counts characters, not bytes; a password of spaces is none; the hash of a local account takes 72
    bytes, and a longer password was refused by the hash itself, in a programmer's words."""
    accent = chr(0xE9)  # two bytes in UTF-8
    assert validate_password(accent * MIN_PASSWORD_LENGTH) is None  # 12 characters, 24 bytes
    assert validate_password(" " * MIN_PASSWORD_LENGTH) == "Password cannot be made of spaces only."
    assert validate_password("  a phrase with spaces  ") is None  # spaces are part of a password
    assert validate_password("x" * MAX_PASSWORD_BYTES) is None
    too_long = (
        "Password is too long: at most 72 bytes (a letter with an accent counts for two or more)."
    )
    assert validate_password("x" * (MAX_PASSWORD_BYTES + 1)) == too_long
    assert validate_password(accent * 36) is None and validate_password(accent * 37) == too_long
    # what no form lets anyone type: a NUL (a file in another encoding read as text), bytes that were no text
    assert validate_password("twelve-chars" + chr(0)) == CANNOT_BE_STORED
    assert validate_password("twelve-chars" + chr(0xDCE4)) == CANNOT_BE_STORED


def test_new_credentials_order_of_checks() -> None:
    good = "correct-horse-battery-staple"
    assert validate_new_credentials("dr_lee", good, good) is None
    assert "Username" in validate_new_credentials("!bad", good, good)
    assert "at least" in validate_new_credentials("dr_lee", "short", "short")
    assert "match" in validate_new_credentials("dr_lee", good, good + "x")


def test_a_new_name_is_its_own_workspace_folder() -> None:
    """A final '.', '_' or '-' is dropped from folder names: "drlee." would share drlee's workspace."""
    for name in ("drlee.", "drlee_", "drlee-", ".drlee"):
        assert validate_username(name) is not None, name
    for name in ("drlee", "dr.lee", "dr_lee-2"):
        assert validate_username(name) is None, name
