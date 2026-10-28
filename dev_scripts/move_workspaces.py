"""
Move the workspace folders earlier versions gave accounts to their folders now (core.workspace.safe_dirname).

Earlier versions replaced each run of characters other than ASCII letters, digits, "_", "." and "-" in a username
by "_", so an account such as "j.smith@chl.lu" now has another folder ("_j.smith_at_chl.lu"). A local account's
folder is moved at its first sign-in; with another identity provider, this script moves them after review.

A folder is moved only when it is clearly one account's (core.workspace.planned_move): the identity provider must list
every account, under the name each signs in with. Check the count it prints: a provider may list accounts in pages.
Without --apply, the script prints the plan and moves nothing.

Usage:
    python dev_scripts/move_workspaces.py            # the plan
    python dev_scripts/move_workspaces.py --apply    # moves what the plan marks "move"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.workspace import earlier_workspaces, move_workspace, planned_move


def possible_owners(names: list[str]) -> list[str]:
    """The listed names, and the part before "@" of each listed e-mail: Better Auth lists an account that signs in
    with a username under an e-mail when it has no username field (``name@local.cohort``)."""
    return list(dict.fromkeys([*names, *(name.split("@", 1)[0] for name in names if "@" in name)]))


def plan(names: list[str]) -> list[tuple[str, str, str]]:
    """(account, its earlier folders, "move" or why the folder stays) for each account with an earlier folder."""
    owners = possible_owners(names)
    rows = []
    for name in names:
        found = earlier_workspaces(name)
        if not found:
            continue
        earlier, reason = planned_move(name, owners)
        rows.append(
            (
                name,
                ", ".join(folder.name for folder in found),
                "move" if earlier is not None else reason,
            )
        )
    return rows


def apply(names: list[str]) -> list[str]:
    """Move each folder the plan marks "move", planned again at the moment of its move."""
    owners = possible_owners(names)
    done = []
    for name in names:
        earlier, _ = planned_move(name, owners)
        if earlier is not None:
            done.append(move_workspace(name, earlier))
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--apply", action="store_true", help="move the folders the plan marks 'move'"
    )
    args = parser.parse_args(argv)

    from utils.settings import SettingsError, load_settings

    try:  # .env first, as main.py and manage_users.py do: the identity provider
        load_settings(Path(__file__).resolve().parents[1] / ".env")
    except SettingsError as exc:
        print(f"Error: {exc}")
        return 2
    from manage.iam import get_iam_manager

    names = [user.username for user in get_iam_manager().list_users()]
    print(f"{len(names)} accounts listed: check that the identity provider listed every account.")
    rows = plan(names)
    if not rows:
        print("No earlier workspace folder to move.")
        return 0
    for account, folders, verdict in rows:
        print(f"{account}: {folders} -> {verdict}")
    if args.apply:
        for line in apply(names):
            print(line)
    else:
        print("Nothing moved: run again with --apply to move the folders marked 'move'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
