"""
One-shot migration of the legacy flat data layout to per-user workspaces.

Before (audit E4): data/datasets/<user>_<name>_vN.parquet, data/traces/<user>_session_*.json,
data/taxonomy/<user>/vN/, data/knowledge_graphs/<user>_<name>.json and the shared DOCUMENTS/.
After: <DATA_ROOT>/users/<user>/{datasets,stats,traces,taxonomy,knowledge_graphs,documents,vectors}/.

Files whose owner cannot be inferred from a username prefix are assigned to the administrator account
(``--admin``, default: the first active administrator in the user directory, else ``admin``).

Usage:
    python dev_scripts/migrate_user_layout.py --dry-run
    python dev_scripts/migrate_user_layout.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.workspace import UserScope
from utils.data_paths import get_data_root


def known_users() -> tuple[list[str], str | None]:
    try:
        from manage.iam import get_iam_manager

        users = get_iam_manager().list_users()
        names = [u.username for u in users]
        admin = next((u.username for u in users if str(u.role) == "admin" and u.is_active), None)
        return names, admin
    except Exception as exc:  # pragma: no cover - environment specific
        print(f"warning: could not read the user directory ({exc}); only prefix inference is used")
        return [], None


def owner_of(name: str, users: list[str], default: str) -> tuple[str, str]:
    """(owner, remaining name) by longest matching '<user>_' prefix."""
    for user in sorted(users, key=len, reverse=True):
        if name.startswith(f"{user}_"):
            return user, name[len(user) + 1:]
    return default, name


def move(src: Path, dst: Path, dry_run: bool) -> None:
    print(f"{src}  ->  {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print("  skipped: destination exists")
        return
    shutil.move(str(src), str(dst))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the plan without moving anything")
    parser.add_argument("--admin", help="account that receives files without a user prefix")
    args = parser.parse_args()

    root = Path(get_data_root()).resolve()
    users, detected_admin = known_users()
    admin = args.admin or detected_admin or os.getenv("CMP_ADMIN_USERNAME", "admin")
    print(f"data root: {root}\nusers: {users or '(none found)'}\nfallback owner: {admin}\n")

    def scope(user: str) -> UserScope:
        return UserScope.for_user(user)

    # datasets and statistics
    datasets = root / "datasets"
    if datasets.exists():
        for f in sorted(datasets.glob("*.parquet")):
            owner, rest = owner_of(f.name, users, admin)
            move(f, scope(owner).datasets_dir() / rest, args.dry_run)
    stats = root / "stats"
    if stats.exists():
        for f in sorted(stats.glob("stats_*.parquet")):
            # stats_<type>_<dataset>.parquet: owner comes from the dataset name
            parts = f.name.split("_", 2)
            dataset = parts[2] if len(parts) == 3 else f.name
            owner, rest = owner_of(dataset, users, admin)
            move(f, scope(owner).stats_dir() / f"{parts[0]}_{parts[1]}_{rest}" if len(parts) == 3 else scope(owner).stats_dir() / f.name, args.dry_run)

    # traces, artifacts, snapshots
    traces = root / "traces"
    if traces.exists():
        for f in sorted(traces.glob("*.json")):
            owner, rest = owner_of(f.name, users, admin)
            move(f, scope(owner).traces_dir() / rest, args.dry_run)
        artifacts = traces / "artifacts"
        if artifacts.exists():
            for d in sorted(p for p in artifacts.iterdir() if p.is_dir()):
                owner, rest = owner_of(d.name, users, admin)
                move(d, scope(owner).artifacts_dir() / rest, args.dry_run)
        snapshots = traces / "snapshots"
        if snapshots.exists():
            for f in sorted(p for p in snapshots.iterdir() if p.is_file()):
                move(f, scope(admin).snapshots_dir() / f.name, args.dry_run)

    # taxonomy versions
    taxonomy = root / "taxonomy"
    if taxonomy.exists():
        for d in sorted(p for p in taxonomy.iterdir() if p.is_dir() and p.name != "_no_user_"):
            for version in sorted(p for p in d.iterdir() if p.is_dir()):
                move(version, scope(d.name).taxonomy_dir() / version.name, args.dry_run)

    # knowledge graphs
    graphs = root / "knowledge_graphs"
    if graphs.exists():
        for f in sorted(graphs.glob("*.json")):
            owner, rest = owner_of(f.name, users, admin)
            move(f, scope(owner).knowledge_graphs_dir() / rest, args.dry_run)

    # the shared document library goes to the administrator
    documents = Path("DOCUMENTS")
    if documents.exists():
        for f in sorted(documents.glob("*.pdf")):
            move(f, scope(admin).documents_dir() / f.name, args.dry_run)

    print("\ndone" + (" (dry run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
