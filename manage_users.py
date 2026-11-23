"""
Administrative command-line tool for user accounts (replaces reset_password.py).

Examples:
    python manage_users.py create-admin ops_admin
    python manage_users.py activate dr_lee
    python manage_users.py deactivate dr_lee
    python manage_users.py reset-password dr_lee
    python manage_users.py list
    python manage_users.py setup-token     # a new one-time token to create the first administrator
    python manage_users.py bootstrap       # at start-up: administrator from CMP_ADMIN_*, or a setup token

Passwords are typed at a prompt, twice, and never shown. They are never accepted on the command line, so they do
not end up in shell history. One exception: create-admin for the account named by CMP_ADMIN_USERNAME takes
CMP_ADMIN_PASSWORD when it is set (an installation scripted without a terminal), as the start-up does.
"""

from __future__ import annotations

import argparse
import getpass
import sys
import warnings
from pathlib import Path

from utils.settings import SettingsError, get_settings, load_settings

# .env first, then every setting validated, as main.py does: the identity provider and the CMP_ADMIN_* settings
try:
    load_settings(Path(__file__).with_name(".env"))
except SettingsError as exc:
    sys.exit(f"Error: {exc}")

from core.workspace import workspace_exists
from manage.iam import IAMConfigurationError, UserRole, get_iam_manager
from manage.iam.authz import validate_password, validate_username


def _fail(message: str) -> None:
    sys.exit(f"Error: {message}")


NO_TERMINAL = (
    "the password is typed on a terminal, and there is none here: run the command in a terminal (in a container: "
    "docker exec -it <container> python manage_users.py ...)."
)


def _at_a_terminal() -> bool:
    """Whether somebody can type a password here: on Windows the null device says it is a terminal too, and the
    prompt then waits for keys nobody types (manage.replay.commandline)."""
    from manage.replay.commandline import at_a_terminal

    return at_a_terminal(sys.stdin)


def _read_password(prompt: str, *, from_setting: bool = False) -> str:
    """
    A password typed twice at a prompt, unseen. ``from_setting``: CMP_ADMIN_PASSWORD in its place when it is set,
    for the one account it is the password of (create-admin of CMP_ADMIN_USERNAME). Read for every command, it gave
    the account of a reset-password the administrator's password, without a question.
    """
    env_password = get_settings().iam.admin_password
    if from_setting and env_password is not None:
        return env_password.get_secret_value()
    if not _at_a_terminal():
        _fail(NO_TERMINAL)
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)  # a prompt that would show what is typed
        try:
            first = getpass.getpass(prompt)
            second = getpass.getpass("Confirm password: ")
        except (getpass.GetPassWarning, EOFError):
            _fail(NO_TERMINAL)
    if first != second:
        _fail("passwords do not match.")
    return first


def cmd_create_admin(args: argparse.Namespace) -> None:
    problem = validate_username(args.username)
    if problem:
        _fail(problem)
    password = _read_password(
        "Administrator password: ", from_setting=args.username == get_settings().iam.admin_username
    )
    problem = validate_password(password)
    if problem:
        _fail(problem)
    # a folder that exists may hold another account's files (the earlier folder of an account that has not signed in
    # since): reused only when the operator says so, after a users database was replaced and its folders kept
    if not args.reuse_folder and workspace_exists(args.username):
        _fail(
            f"the workspace folder of {args.username!r} exists and may hold another account's files. If it is this "
            "account's own (a users database replaced, its folders kept), run again with --reuse-folder."
        )
    ok, msg = get_iam_manager().create_user(
        args.username, password, role=UserRole.ADMIN, auto_activate=True, reuse_folder=args.reuse_folder
    )
    if not ok:
        _fail(msg)
    print(f"Administrator '{args.username}' created.")  # the command's own words: the accounts' are for a refusal
    sys.exit(0)


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
    if not ok:
        _fail(msg)
    print(f"Password updated for user '{args.username}'.")
    sys.exit(0)


def cmd_list(_: argparse.Namespace) -> None:
    users = get_iam_manager().list_users()
    if not users:
        print("No users registered.")
        return
    print(f"{'username':<24} {'role':<12} {'active':<8} created_at")
    for user in users:
        print(f"{user.username:<24} {str(user.role):<12} {str(user.is_active):<8} {user.created_at or ''}")


def cmd_setup_token(_: argparse.Namespace) -> None:
    token = get_iam_manager().issue_setup_token()
    if token is None:
        _fail("no first-run setup is needed: an active administrator exists, or accounts are disabled.")
    print(f"One-time setup token: {token}")
    print("Open CohortStudio and create the first administrator with it; any previous token no longer works.")


def cmd_bootstrap(_: argparse.Namespace) -> None:
    """Run before the server (Docker entry point): the administrator from CMP_ADMIN_*, or a setup token in the log
    when no password is set. A password that is set and cannot be used ends the command with the reason; the
    server then started says the same on its page."""
    _, message = get_iam_manager().ensure_bootstrap_admin()
    print(f"Administrator bootstrap: {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage CohortStudio user accounts.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create-admin", help="create an ACTIVE administrator account")
    p.add_argument("username")
    p.add_argument(
        "--reuse-folder",
        action="store_true",
        help="give the account the workspace folder of its name although it exists (it must be this account's)",
    )
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

    p = sub.add_parser("setup-token", help="issue a new one-time token to create the first administrator")
    p.set_defaults(func=cmd_setup_token)

    p = sub.add_parser("bootstrap", help="create the administrator from CMP_ADMIN_* or issue a setup token")
    p.set_defaults(func=cmd_bootstrap)
    return parser


def main(argv: list[str] | None = None) -> None:
    from utils.logging_config import configure_logging

    args = build_parser().parse_args(argv)
    # for every command: the log's handler and level, and no exception's text in a record (a warning of the
    # identity provider, or a traceback, would otherwise reach the terminal as Python writes it)
    configure_logging()
    try:
        args.func(args)
    # an unknown CMP_IAM_PROVIDER, or a first administrator's password that is set and cannot be used: the words
    # name the settings, never a value
    except IAMConfigurationError as exc:
        _fail(str(exc))


if __name__ == "__main__":
    main()
