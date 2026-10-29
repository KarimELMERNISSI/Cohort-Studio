"""
Per-user storage scope (audit finding E4).

Every user gets a directory tree under ``<DATA_ROOT>/users/<username>/`` and all persistence goes
through a :class:`UserScope`, instead of filename prefixes checked inconsistently across pages.
Paths supplied by users (uploads, manual file paths) are resolved with :meth:`UserScope.resolve_within`
so they can never leave the user's workspace.

This module has no Streamlit dependency; ``ui.session_scope`` derives the current scope from the
session state.
"""

from __future__ import annotations

import contextlib
import hashlib
import ntpath
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

# a new name starts and ends with a letter or a digit: it is its own workspace folder (safe_dirname)
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,30}[A-Za-z0-9]$")
# what earlier versions did to a name to make its folder (earlier_folders)
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
# a name that is its own folder: ASCII letters and digits, with "_", "." and "-" inside, at most 64 characters
_FOLDER_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,62}[A-Za-z0-9])?")
_DEVICE_NAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
)  # no folder of Windows can have these names, with or without an extension
_KEPT_IN_CODE = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.-")
KINDS: tuple[str, ...] = (
    "datasets",
    "stats",
    "traces",
    "uploads",
    "taxonomy",
    "knowledge_graphs",
    "documents",
    "vectors",
    "rag_eval_logs",
    "protocols",
)


def is_device_name(name: str) -> bool:
    """A name Windows keeps for a device (CON, NUL, COM1...), with or without an extension: no folder can have it."""
    return name.split(".")[0].lower() in _DEVICE_NAMES


def data_root() -> Path:
    """The application data root (``DATA_ROOT``), resolved through ``utils.data_paths`` so tests can patch it."""
    from utils.data_paths import get_data_root

    return Path(get_data_root()).resolve()


def validate_username(username: str | None) -> str:
    """Strict validation for new accounts; raises ``ValueError`` otherwise."""
    value = (username or "").strip()
    if not USERNAME_PATTERN.match(value):
        raise ValueError(
            "Username must be 3-32 characters: letters, digits, '.', '_' or '-' (no spaces)."
        )
    return value


def safe_dirname(username: str | None) -> str:
    """
    The workspace folder of a username, never a path escape; two names never share one.

    - A name that is a folder name already is its own folder: ASCII letters and digits, with "_", "." or "-" inside,
      at most 64 characters, and not a device name of Windows. Every account created in the application has one.
    - Any other name, such as an identity provider's e-mail, gets "_" and a readable code of it: "j.smith@chl.lu" gets
      "_j.smith_at_chl.lu". No folder name starts with "_", and two names never get the same code: "_" is doubled,
      "@" is written "_at_", and any other character (and a final ".") is written "_x", its code point in
      hexadecimal, and "_" ("o'neil@x.org" gets "_o_x27_neil_at_x.org").
    - A code longer than 64 characters: "_" and 32 hexadecimal characters of the name's SHA-256.

    Earlier versions replaced each run of such characters by "_": "j.smith@chl.lu" and "j.smith_chl.lu" had one folder.
    Names are never normalised or case-folded, so two spellings an identity provider may tell apart keep two folders
    (on Windows, names differing only by case are one folder: local sign-up refuses such a name).
    """
    name = (username or "").strip()
    if not name:
        raise ValueError("A workspace directory needs a username.")
    if _FOLDER_NAME.fullmatch(name) and not is_device_name(name):
        return name
    code = _code(name)
    if len(code) <= 64:
        return code
    return "_" + hashlib.sha256(name.encode("utf-8", "surrogatepass")).hexdigest()[:32]


def _code(name: str) -> str:
    """The readable code of a name that is no folder name (safe_dirname); it starts with "_"."""
    last = len(name) - 1
    out = ["_"]
    for position, character in enumerate(name):
        if character == "_":
            out.append("__")
        elif character == "@":
            out.append("_at_")
        elif character in _KEPT_IN_CODE and not (character == "." and position == last):
            out.append(character)
        else:  # Windows drops a final "." from a folder name
            out.append(f"_x{ord(character):x}_")
    return "".join(out)


def earlier_folders(username: str | None) -> list[str]:
    """
    The folders earlier versions gave ``username`` that are not its folder now: the name with each run of characters
    other than ASCII letters, digits, "_", "." and "-" replaced by "_" and those three trimmed from its ends, and "_"
    with 32 hexadecimal characters of its SHA-256.
    """
    name = (username or "").strip()
    if not name:
        return []
    found = []
    trimmed = _UNSAFE_CHARS.sub("_", name).strip("._-")
    if trimmed and len(trimmed) <= 64:
        found.append(trimmed)
    found.append("_" + hashlib.sha256(name.encode("utf-8", "surrogatepass")).hexdigest()[:32])
    current = safe_dirname(name)
    return [folder for folder in dict.fromkeys(found) if folder != current]


def workspace_exists(username: str | None) -> bool:
    """Whether the folder ``username`` would have exists already. A new account never takes it: it may hold another
    account's files, such as the folder an earlier version gave an account that has not signed in since."""
    name = (username or "").strip()
    if not name:
        return False
    try:  # safe_dirname keeps it inside users/; resolve_within says so where code scanning reads it
        return resolve_within(data_root() / "users", safe_dirname(name)).exists()
    except ValueError:  # a link leading outside the data folder: never given to a new account
        return True


def earlier_workspaces(username: str | None) -> list[Path]:
    """The folders an earlier version gave ``username`` that exist and are not its folder now."""
    users = data_root() / "users"
    return [users / folder for folder in earlier_folders(username) if (users / folder).exists()]


def planned_move(username: str | None, accounts: Iterable[str]) -> tuple[Path | None, str]:
    """
    The earlier folder of ``username`` to move to its folder now, and why not when it cannot be: (folder, "") when
    exactly one exists, it is a folder and no link or junction, the folder now holds no file, ``accounts`` (every
    account) holds ``username``, and no other account in it could own the earlier folder, as its folder now or an
    earlier one; (None, reason) when it cannot be moved; (None, "") when there is none.
    """
    name = (username or "").strip()
    found = earlier_workspaces(name)
    if not found:
        return None, ""
    if len(found) > 1:
        return (
            None,
            f"several earlier folders ({', '.join(f.name for f in found)}) are left for an administrator",
        )
    earlier = found[0]
    if earlier.is_symlink() or earlier.is_junction() or not earlier.is_dir():
        return (
            None,
            f"the earlier folder {earlier.name} is a link, never moved: it is left for an administrator",
        )
    target = earlier.parent / safe_dirname(name)
    if target.is_symlink() or target.is_junction():
        return (
            None,
            f"the folder {target.name} is a link, never replaced: {earlier.name} is left for an administrator",
        )
    if target.exists() and any(path.is_file() for path in target.rglob("*")):
        return (
            None,
            f"the earlier folder {earlier.name} is left for an administrator: {target.name} holds files",
        )
    names = [str(other).strip() for other in accounts]
    if name not in names:
        return (
            None,
            f"the list of accounts does not hold this one: the folder {earlier.name} is left",
        )
    owned = os.path.normcase(earlier.name)
    for other in names:
        if not other or other == name:
            continue
        if owned in {
            os.path.normcase(folder) for folder in (safe_dirname(other), *earlier_folders(other))
        }:
            return (
                None,
                f"another account could own the folder {earlier.name}: it is left for an administrator",
            )
    return earlier, ""


def move_workspace(username: str | None, earlier: Path) -> str:
    """Move ``earlier`` (from :func:`planned_move`) to the folder of ``username`` now. An empty folder there, created on
    this visit, is replaced; never one that holds a file."""
    target = earlier.parent / safe_dirname((username or "").strip())
    if target.is_symlink() or target.is_junction():  # planned_move refuses it: never followed
        raise ValueError(f"the folder {target.name} is a link: nothing is moved")
    if target.exists():
        for folder, _, _ in sorted(
            os.walk(target), key=lambda walked: len(walked[0]), reverse=True
        ):
            os.rmdir(folder)  # fails on a file: nothing is ever deleted
    earlier.rename(target)
    return f"the folder {earlier.name} was moved to {target.name}"


def move_earlier_workspace(
    username: str | None, accounts: Callable[[], Iterable[str]]
) -> tuple[bool, str]:
    """
    At the sign-in of a local account: move the folder an earlier version gave it to its folder now, once, when it is
    clearly its own (:func:`planned_move`; ``accounts`` is asked only when there is a folder to move). Returns whether
    it moved and what happened, for the server log ("" when there was nothing to move).
    """
    if not earlier_workspaces(username):
        return False, ""

    def listed() -> Iterator[str]:  # the accounts, asked only after the checks on the folders
        yield from accounts()

    earlier, reason = planned_move(username, listed())
    if earlier is None:
        return False, reason
    return True, move_workspace(username, earlier)


def safe_filename(name: str | None, default: str = "file") -> str:
    """File-system safe base name (no separators, no leading dots); falls back to ``default``."""
    raw = Path(str(name or "")).name  # drop any directory part
    value = re.sub(r"[^A-Za-z0-9 _.()-]+", "_", raw).strip().strip(".")
    return value or default


def file_name_of(source: object) -> str:
    """
    The file's name in a path written on any system (both separators are read: an index can move between systems),
    "" for none or a folder. What leaves the server names a file by it, never by the folders it is in, which name the
    user's account and the server's layout.
    """
    return ntpath.basename(str(source or ""))


def network_share(path: str | Path) -> str:
    """
    The network share or device a Windows path names (``\\\\server\\share``, ``//server/share``, ``\\\\?\\UNC\\...``,
    ``\\\\.\\pipe\\...``), normalised; "" for a local path, and always on Linux, where such a path is local. Opening it
    connects to that server, which then receives the account's credentials.
    """
    drive = os.path.splitdrive(str(path))[0]
    return os.path.normcase(drive) if drive[:2] in ("\\\\", "//") else ""


def resolve_within(base: Path, candidate: str | Path) -> Path:
    """Resolve ``candidate`` (absolute or relative to ``base``) and require it to stay inside ``base``.

    Written as normalise-then-check-the-prefix (``os.path.realpath``, then ``startswith``), the form that code
    scanning (CodeQL) recognises as confining a path chosen by the user. A path on another network share is refused
    before it is opened (:func:`network_share`): ``realpath`` would connect to it.
    """
    root = os.path.realpath(base)
    joined = os.path.join(root, candidate)  # an absolute candidate replaces root
    share = network_share(joined)
    if share and share != network_share(root):
        raise ValueError(f"'{candidate}' is outside the allowed directory {root}.")
    path = os.path.realpath(joined)
    if path == root:
        return Path(root)
    inside = root if root.endswith(os.sep) else root + os.sep  # a drive root already ends with one
    if not path.startswith(inside):
        raise ValueError(f"'{candidate}' is outside the allowed directory {root}.")
    return Path(path)


_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def path_lock(path: Path) -> threading.Lock:
    """The lock of one file, shared by the sessions of this process: hold it to read a file and write it again."""
    with _locks_guard:
        return _locks.setdefault(str(path), threading.Lock())


def write_atomic(path: Path, text: str, *, prefix: str = ".tmp.") -> None:
    """
    Replace ``path`` with ``text`` in one step: written beside it, then moved over it, so that a crash never leaves it
    half written. The file holds exactly ``text`` (UTF-8, its line ends as given, on every system). On Windows the
    move is tried again for a moment while another program holds the file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(
                stream.fileno()
            )  # on the disk before the move: a power cut never leaves the file empty
        for attempt in range(5):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:  # Windows: the file open elsewhere for a moment
                if attempt == 4:
                    raise
                time.sleep(0.05)
    finally:
        with contextlib.suppress(OSError):
            os.remove(tmp)


def remove_pdf(folder: str | Path, name: str) -> bool:
    """Delete the PDF ``name`` of ``folder`` (a document of the library), never a file outside it; False when
    there is no such PDF."""
    base = Path(folder).resolve()
    path = resolve_within(base, name)
    if path.parent != base or path.suffix.lower() != ".pdf" or not path.is_file():
        return False
    path.unlink()
    return True


@dataclass(frozen=True)
class UserScope:
    """All storage locations of one user."""

    username: str
    role: str = "researcher"

    @classmethod
    def for_user(cls, username: str | None, role: str | None = None) -> UserScope:
        return cls(username=safe_dirname(username), role=str(role or "researcher"))

    @property
    def is_admin(self) -> bool:
        return self.role.lower() == "admin"

    @property
    def root(self) -> Path:
        path = data_root() / "users" / self.username
        path.mkdir(parents=True, exist_ok=True)
        return path

    def dir(self, kind: str) -> Path:
        if kind not in KINDS:
            raise ValueError(f"Unknown storage kind {kind!r}")
        path = self.root / kind
        path.mkdir(parents=True, exist_ok=True)
        return path

    def datasets_dir(self) -> Path:
        return self.dir("datasets")

    def stats_dir(self) -> Path:
        return self.dir("stats")

    def traces_dir(self) -> Path:
        return self.dir("traces")

    def artifacts_dir(self, session_id: str | None = None) -> Path:
        path = self.traces_dir() / "artifacts"
        if session_id:
            path = path / safe_filename(session_id, "session")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def snapshots_dir(self) -> Path:
        path = self.traces_dir() / "snapshots"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def uploads_dir(self) -> Path:
        return self.dir("uploads")

    def taxonomy_dir(self) -> Path:
        return self.dir("taxonomy")

    def knowledge_graphs_dir(self) -> Path:
        return self.dir("knowledge_graphs")

    def documents_dir(self) -> Path:
        return self.dir("documents")

    def vectors_dir(self) -> Path:
        """Vector indexes of the document library, one folder per engine (``vectors/lancedb``, ``vectors/qdrant``)."""
        return self.dir("vectors")

    def rag_eval_logs_dir(self) -> Path:
        return self.dir("rag_eval_logs")

    def protocols_dir(self) -> Path:
        """The study definitions (manage.protocol_store), one folder per study."""
        return self.dir("protocols")

    def resolve_within(self, candidate: str | Path, kind: str | None = None) -> Path:
        """A path the user may read: inside the whole workspace, or inside one storage kind."""
        base = self.dir(kind) if kind else self.root
        return resolve_within(base, candidate)

    def search_dirs(self) -> list[Path]:
        """Directories searched when a trace references a file by name."""
        return [
            self.uploads_dir(),
            self.artifacts_dir(),
            self.snapshots_dir(),
            self.datasets_dir(),
            self.root,
        ]


def all_user_scopes() -> list[UserScope]:
    """Scopes for every user directory present on disk (administration and migration)."""
    users_dir = data_root() / "users"
    if not users_dir.exists():
        return []
    return [UserScope(username=p.name) for p in sorted(users_dir.iterdir()) if p.is_dir()]


__all__ = [
    "KINDS",
    "USERNAME_PATTERN",
    "UserScope",
    "all_user_scopes",
    "data_root",
    "earlier_folders",
    "earlier_workspaces",
    "is_device_name",
    "move_earlier_workspace",
    "move_workspace",
    "planned_move",
    "network_share",
    "path_lock",
    "resolve_within",
    "safe_dirname",
    "safe_filename",
    "validate_username",
    "workspace_exists",
    "write_atomic",
]
