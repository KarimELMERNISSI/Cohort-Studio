"""
Command-line utility to reset or create user accounts in DuckDB.
Usage:
    python reset_password.py [username] [password]
If arguments are omitted, prompts interactively.
"""

import sys
import getpass
from manage.db_manager import DBManager


def reset_or_create_user(username: str, password: str) -> None:
    """Reset password for an existing user or create a new activated user.

    Args:
        username: The username to create or reset.
        password: The new password to set.
    """
    if not username or not password:
        print("Error: Username and password cannot be empty.")
        sys.exit(1)

    if len(password) < 4:
        print("Error: Password must be at least 4 characters long.")
        sys.exit(1)

    db = DBManager()
    users = db.get_all_users()

    if username in users:
        success, msg = db.update_user_password(username, password)
        if success:
            db.activate_user(username)
            print(f"Success: Password for '{username}' updated and account activated.")
        else:
            print(f"Error: {msg}")
            sys.exit(1)
    else:
        success, msg = db.create_user(username, password, auto_activate=True)
        if success:
            db.activate_user(username)
            print(f"Success: User '{username}' created and activated.")
        else:
            print(f"Error: {msg}")
            sys.exit(1)


def main() -> None:
    if len(sys.argv) >= 3:
        username = sys.argv[1].strip()
        password = sys.argv[2].strip()
    elif len(sys.argv) == 2:
        username = sys.argv[1].strip()
        password = getpass.getpass(prompt="Enter new password: ").strip()
    else:
        username = input("Enter username (e.g., admin): ").strip()
        password = getpass.getpass(prompt="Enter new password: ").strip()

    reset_or_create_user(username, password)


if __name__ == "__main__":
    main()
