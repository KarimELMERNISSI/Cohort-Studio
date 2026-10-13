"""
Administrative command-line tool for user accounts (replaces reset_password.py).

Examples:
    python manage_users.py create-admin ops_admin
    python manage_users.py activate dr_lee
    python manage_users.py deactivate dr_lee
    python manage_users.py reset-password dr_lee
    python manage_users.py list

Passwords are read from the CMP_ADMIN_PASSWORD environment variable when set, otherwise prompted
interactively. They are never accepted on the command line, so they do not end up in shell history.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys

from manage.iam import UserRole, get_iam_manager
from manage.iam.authz import validate_password, validate_username


def _fail(message: str) -> None:
    sys.exit(f"Error: {message}")


def _read_password(prompt: str) -> str:
    env_password = os.getenv("CMP_ADMIN_PASSWORD")
    if env_password:
        return env_password
    first = getpass.getpass(prompt)
    second = getpass.getpass("Confirm password: ")
    if first != second:
        _fail("passwords do not match.")
    return first


def cmd_create_admin(args: argparse.Namespace) -> None:
    problem = validate_username(args.username)
    if problem:
        _fail(problem)
    password = _read_password("Administrator password: ")
    problem = validate_password(password)
    if problem:
        _fail(problem)
    ok, msg = get_iam_manager().create_user(args.username, password, role=UserRole.ADMIN, auto_activate=True)
    print(msg if ok else f"Error: {msg}")
    sys.exit(0 if ok else 1)


def cmd_activate(args: argparse.Namespace) -> None:
    ok, msg = get_iam_manager().activate_user(args.username)
    print(msg if ok else f"Error: {msg}")
    sys.exit(0 if ok else 1)


def cmd_deactivate(args: argparse.Namespace) -> None:
    ok, msg = get_iam_manager().deactivate_user(args.username)
    print(msg if ok else f"Error: {msg}")
    sys.exit(0 if ok else 1)


def cmd_reset_password(args: argparse.Namespace) -> None:
    iam = get_iam_manager()
    if iam.get_user(args.username) is None:
        _fail(f"user '{args.username}' not found.")
    password = _read_password("New password: ")
    problem = validate_password(password)
    if problem:
        _fail(problem)
    ok, msg = iam.update_user_password(args.username, password)
    print(msg if ok else f"Error: {msg}")
    sys.exit(0 if ok else 1)


def cmd_list(_: argparse.Namespace) -> None:
    users = get_iam_manager().list_users()
    if not users:
        print("No users registered.")
        return
    print(f"{'username':<24} {'role':<12} {'active':<8} created_at")
    for user in users:
        print(f"{user.username:<24} {str(user.role):<12} {str(user.is_active):<8} {user.created_at or ''}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage CohortStudio user accounts.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create-admin", help="create an ACTIVE administrator account")
    p.add_argument("username")
    p.set_defaults(func=cmd_create_admin)

    p = sub.add_parser("activate", help="activate a pending account")
    p.add_argument("username")
    p.set_defaults(func=cmd_activate)

    p = sub.add_parser("deactivate", help="deactivate an account")
    p.add_argument("username")
    p.set_defaults(func=cmd_deactivate)

    p = sub.add_parser("reset-password", help="set a new password for an account")
    p.add_argument("username")
    p.set_defaults(func=cmd_reset_password)

    p = sub.add_parser("list", help="list accounts")
    p.set_defaults(func=cmd_list)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
