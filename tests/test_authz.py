"""Role checks and credential policy (audit E5)."""

from manage.iam.authz import (
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


def test_new_credentials_order_of_checks() -> None:
    good = "correct-horse-battery-staple"
    assert validate_new_credentials("dr_lee", good, good) is None
    assert "Username" in validate_new_credentials("!bad", good, good)
    assert "at least" in validate_new_credentials("dr_lee", "short", "short")
    assert "match" in validate_new_credentials("dr_lee", good, good + "x")
