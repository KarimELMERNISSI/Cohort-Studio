"""
The study definitions of a user's workspace (M6): a working copy saved as the user works, and numbered revisions that
are never rewritten.

    <workspace>/protocols/<study_id>/
        working.json                                replaced in one step at each save
        revisions/r0001.number                      the number taken, before its revision is written; never deleted
        revisions/r0001-<checksum12>.json           written once; read back with its checksum checked
        history/working-<YYYYmmddTHHMMSS>-NNN.json  copies another session had saved: the last 10, none over 30 days

There is one store, with no setting and no selector: a Git or a database store would be a second class of
:class:`ProtocolStore`. A store needs the user's workspace, or a folder given explicitly (the tests, a command line):
there is no shared folder of study definitions. A file is read up to 5 MB, and none larger is written; one that cannot
be read gives none, logged with the study and the reason, never with its content. A file that cannot be written or
deleted raises ProtocolFileError, whose message names no path. Files keep the order of the families and of their rules,
which the fingerprint covers. A revision's number is taken by creating its ``.number`` file, which only one save can
do, in any process, and which stays when the revision is deleted: a number is never given twice. The working copy's tag
and its history are kept in one process (the application's).

A copy in the history holds what the working copy held when another session replaced it: a text corrected since (a
name typed into a rule) stays there until :meth:`ProtocolStore.discard_history` deletes the copies, or for 30 days.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from core.protocol import StudyProtocol, is_study_id, next_revision
from core.workspace import UserScope, path_lock, resolve_within, write_atomic
from utils.environment import app_version

logger = logging.getLogger(__name__)

MAX_BYTES = 5 * 1024 * 1024
HISTORY = 10  # copies kept when another session's working copy is replaced
HISTORY_DAYS = 30  # and none kept longer
_STALE_TMP = 3600  # seconds: a temporary file older than this is left by a write that never ended
WORKING = "working.json"
_REVISION = re.compile(r"r(\d{4,})-([0-9a-f]{12})\.json")
_NUMBER = re.compile(r"r(\d{4,})\.number")


class ProtocolUnchanged(Exception):
    """A revision would repeat the latest one."""


class ProtocolFileError(Exception):
    """A file of a study definition could not be written or deleted. The message names no path."""


@dataclass(frozen=True)
class Stored:
    """A working copy and the tag of its file, given back on the next save."""

    protocol: StudyProtocol
    etag: str


@dataclass(frozen=True)
class Saved:
    """A saved working copy: its new tag, and the history file of the copy it replaced when another session had saved
    one since the tag given (None otherwise)."""

    etag: str
    kept: str | None = None


@dataclass(frozen=True)
class StudyInfo:
    study_id: str
    title: str
    revision: int | None  # the latest readable revision
    updated_at: str


@dataclass(frozen=True)
class RevisionInfo:
    number: int
    checksum: str  # of its body as written (core.protocol.body_checksum)
    parent: str | None  # the checksum of the revision it follows
    saved_at: str


def _etag(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M")


def _text(protocol: StudyProtocol) -> str:
    # never sort the keys: the order of the families and of their rules is part of the study
    return json.dumps(protocol.to_file(), indent=2, ensure_ascii=False) + "\n"


def _file_error(study_id: str, name: str, exc: OSError, done: str) -> ProtocolFileError:
    """The error to raise for a file that could not be written or deleted: logged and told without its path."""
    logger.warning("Study %s: %s not %s (%s)", study_id, name, done, type(exc).__name__)
    return ProtocolFileError(
        f"the study definition's {name} could not be {done} ({type(exc).__name__})"
    )


class ProtocolStore(ABC):
    """Where the study definitions are kept."""

    @abstractmethod
    def studies(self) -> list[StudyInfo]:
        """The studies, by id."""

    @abstractmethod
    def working(self, study_id: str) -> Stored | None:
        """The study's working copy, none when there is none or it cannot be read."""

    @abstractmethod
    def save_working(self, protocol: StudyProtocol, *, expected: str | None) -> Saved:
        """
        Save the working copy. ``expected`` is the tag of the copy this session read or saved last (None for none):
        a copy another session saved since is kept in the history, and named in the result.
        """

    @abstractmethod
    def history(self, study_id: str) -> list[str]:
        """The copies kept from other sessions, oldest first."""

    @abstractmethod
    def discard_history(self, study_id: str) -> int:
        """Delete the kept copies, so that a text corrected in the working copy is gone from them too; how many."""

    @abstractmethod
    def revisions(self, study_id: str) -> list[RevisionInfo]:
        """The study's readable revisions, by number."""

    @abstractmethod
    def revision(self, study_id: str, number: int) -> StudyProtocol | None:
        """One revision, none when it does not exist or cannot be read."""

    @abstractmethod
    def save_revision(self, protocol: StudyProtocol) -> StudyProtocol:
        """Save the body as the next revision and return it; ProtocolUnchanged when it repeats the latest."""


class WorkspaceProtocolStore(ProtocolStore):
    """The study definitions as JSON files in the user's workspace (``protocols``)."""

    def __init__(self, scope: UserScope | None, *, root: Path | None = None) -> None:
        if root is None:
            if scope is None:
                raise ValueError(
                    "study definitions are kept in a user's workspace, and no user is given"
                )
            root = scope.protocols_dir()
        self.scope = scope
        self.root = Path(root)

    def _folder(self, study_id: str) -> Path:
        if not is_study_id(study_id):
            raise ValueError(
                f"{study_id!r} is not a study id (lower-case letters, digits, '_' and '-')"
            )
        return resolve_within(self.root, study_id)

    def _read(self, path: Path, study_id: str) -> bytes | None:
        try:
            with open(path, "rb") as stream:
                data = stream.read(MAX_BYTES + 1)
        except FileNotFoundError:
            return None
        except OSError as exc:
            logger.warning(
                "Study %s: %s cannot be read (%s)", study_id, path.name, type(exc).__name__
            )
            return None
        if len(data) > MAX_BYTES:
            logger.warning(
                "Study %s: %s is larger than 5 MB, so it is not read", study_id, path.name
            )
            return None
        return data

    def _parse(self, data: bytes, name: str, study_id: str) -> StudyProtocol | None:
        try:
            parsed = json.loads(data.decode("utf-8"))
        except UnicodeDecodeError, ValueError, RecursionError:
            logger.warning("Study %s: %s is not a JSON file", study_id, name)
            return None
        try:
            protocol, problems = StudyProtocol.from_file(parsed)
        except (
            Exception
        ) as exc:  # a file crafted to break the reading: none, and the type alone is logged
            logger.warning("Study %s: %s is not read (%s)", study_id, name, type(exc).__name__)
            return None
        if protocol is None:
            logger.warning("Study %s: %s is not read: %s", study_id, name, "; ".join(problems))
            return None
        if protocol.meta.study_id != study_id:
            logger.warning("Study %s: %s names another study", study_id, name)
            return None
        return protocol

    def _load(self, path: Path, study_id: str) -> tuple[StudyProtocol, bytes] | None:
        data = self._read(path, study_id)
        protocol = self._parse(data, path.name, study_id) if data is not None else None
        return (protocol, data) if protocol is not None and data is not None else None

    def _sized(self, text: str, study_id: str, name: str) -> str:
        """``text``, unless it is larger than a file the store reads back."""
        if len(text.encode("utf-8")) > MAX_BYTES:
            logger.warning("Study %s: %s not saved (larger than 5 MB)", study_id, name)
            raise ProtocolFileError(f"the study definition's {name} would be larger than 5 MB")
        return text

    def studies(self) -> list[StudyInfo]:
        if not self.root.is_dir():
            return []
        out = []
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir() or not is_study_id(folder.name):
                continue
            try:
                stored = self.working(folder.name)
                revisions = self.revisions(folder.name)
                latest = self.revision(folder.name, revisions[-1].number) if revisions else None
            except Exception as exc:  # one study that cannot be listed leaves the others listed
                logger.warning("Study %s is not listed (%s)", folder.name, type(exc).__name__)
                continue
            shown = stored.protocol if stored is not None else latest
            out.append(
                StudyInfo(
                    folder.name,
                    shown.meta.title if shown is not None else "",
                    revisions[-1].number if revisions else None,
                    shown.meta.saved_at if shown is not None else "",
                )
            )
        return out

    def working(self, study_id: str) -> Stored | None:
        loaded = self._load(self._folder(study_id) / WORKING, study_id)
        if loaded is None:
            return None
        protocol, data = loaded
        if protocol.meta.revision is not None:
            logger.warning("Study %s: %s holds a revision, not a working copy", study_id, WORKING)
            return None
        return Stored(protocol, _etag(data))

    def save_working(self, protocol: StudyProtocol, *, expected: str | None) -> Saved:
        study_id = protocol.meta.study_id
        folder = self._folder(study_id)
        meta = replace(protocol.meta, revision=None, saved_at=_now(), app_version=app_version())
        text = self._sized(_text(StudyProtocol(protocol.body, meta)), study_id, WORKING)
        path = folder / WORKING
        try:
            with path_lock(path):
                current = self._read(path, study_id)
                kept = None
                # another session's copy, kept when it is one of this study's (a file it cannot read is replaced)
                if current is not None and _etag(current) != expected:
                    if self._parse(current, WORKING, study_id) is not None:
                        kept = self._keep(folder, current)
                write_atomic(path, text, prefix=".working.")
        except OSError as exc:
            raise _file_error(study_id, WORKING, exc, "saved") from None
        self._prune(folder)
        return Saved(_etag(text.encode("utf-8")), kept)

    def _keep(self, folder: Path, data: bytes) -> str:
        """Keep a working copy another session saved in the history, and name its file."""
        history = folder / "history"
        history.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        # after this second's copies: a name the pruning freed would sort first, and go at once
        same = re.compile(rf"working-{stamp}-(\d+)\.json")
        taken = [int(m[1]) for p in history.iterdir() if (m := same.fullmatch(p.name))]
        for n in itertools.count(max(taken, default=0) + 1):
            name = f"working-{stamp}-{n:03d}.json"
            try:
                with open(history / name, "xb") as stream:
                    stream.write(data)
                break
            except FileExistsError:
                continue
        return name

    def _prune(self, folder: Path) -> None:
        """The history's copies beyond the last HISTORY or older than HISTORY_DAYS, and the temporary files of writes
        that never ended, are deleted at each save."""
        now = time.time()
        copies = sorted((folder / "history").glob("working-*.json"))
        for i, path in enumerate(copies):
            with contextlib.suppress(OSError):
                if i < len(copies) - HISTORY or now - path.stat().st_mtime > HISTORY_DAYS * 86400:
                    path.unlink()
        for tmp in folder.glob(".working.*.tmp"):
            with contextlib.suppress(OSError):
                if now - tmp.stat().st_mtime > _STALE_TMP:
                    tmp.unlink()

    def history(self, study_id: str) -> list[str]:
        folder = self._folder(study_id) / "history"
        return sorted(p.name for p in folder.glob("working-*.json")) if folder.is_dir() else []

    def discard_history(self, study_id: str) -> int:
        folder = self._folder(study_id) / "history"
        names = self.history(study_id)
        for name in names:
            try:
                (folder / name).unlink(missing_ok=True)
            except OSError as exc:
                raise _file_error(study_id, "history", exc, "deleted") from None
        return len(names)

    def _numbers(self, folder: Path) -> list[int]:
        """The numbers taken: by a revision's file, readable or not, or by its ``.number`` file."""
        if not folder.is_dir():
            return []
        return sorted(
            {
                int(m[1])
                for p in folder.iterdir()
                if (m := _REVISION.fullmatch(p.name) or _NUMBER.fullmatch(p.name))
            }
        )

    def _take_number(self, folder: Path, first: int) -> int:
        """The first number from ``first`` that no save has taken, taken by creating its ``.number`` file: only one
        save can, in any process."""
        for number in itertools.count(first):
            try:
                (folder / f"r{number:04d}.number").open("x").close()
                return number
            except FileExistsError:
                continue
        raise AssertionError("unreachable")

    def _revision_file(self, folder: Path, number: int) -> Path | None:
        for path in folder.glob(f"r{number:04d}-*.json") if folder.is_dir() else ():
            match = _REVISION.fullmatch(path.name)
            if match and int(match[1]) == number:
                return path
        return None

    def revision(self, study_id: str, number: int) -> StudyProtocol | None:
        path = self._revision_file(self._folder(study_id) / "revisions", number)
        loaded = self._load(path, study_id) if path is not None else None
        if path is None or loaded is None:
            return None
        protocol = loaded[0]
        if protocol.meta.revision != number or not path.name.endswith(
            f"-{protocol.checksum[:12]}.json"
        ):
            logger.warning(
                "Study %s: %s does not hold the revision its name says", study_id, path.name
            )
            return None
        return protocol

    def revisions(self, study_id: str) -> list[RevisionInfo]:
        out = []
        for number in self._numbers(self._folder(study_id) / "revisions"):
            protocol = self.revision(study_id, number)
            if protocol is not None:
                out.append(
                    RevisionInfo(
                        number, protocol.checksum, protocol.meta.parent, protocol.meta.saved_at
                    )
                )
        return out

    def save_revision(self, protocol: StudyProtocol) -> StudyProtocol:
        study_id = protocol.meta.study_id
        folder = self._folder(study_id) / "revisions"
        with path_lock(folder):
            numbers = self._numbers(folder)
            latest = next(
                (
                    found
                    for n in reversed(numbers)
                    if (found := self.revision(study_id, n)) is not None
                ),
                None,
            )
            if latest is not None and latest.hash == protocol.hash:  # as this version reads both
                raise ProtocolUnchanged(
                    f"the study definition is unchanged since revision {latest.meta.revision}"
                )
            meta = replace(protocol.meta, saved_at=_now(), app_version=app_version())
            first = max(numbers, default=0) + 1
            # refused before a number is taken
            self._sized(
                _text(next_revision(latest, protocol.body, meta, number=first)),
                study_id,
                "revision",
            )
            try:
                folder.mkdir(parents=True, exist_ok=True)
                number = self._take_number(folder, first)
                new = next_revision(latest, protocol.body, meta, number=number)
                # exclusive creation: a revision file is never rewritten
                name = f"r{number:04d}-{new.checksum[:12]}.json"
                with open(folder / name, "x", encoding="utf-8", newline="") as stream:
                    stream.write(_text(new))
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError as exc:
                raise _file_error(study_id, "revision", exc, "saved") from None
        return new


__all__ = [
    "HISTORY",
    "HISTORY_DAYS",
    "MAX_BYTES",
    "ProtocolFileError",
    "ProtocolStore",
    "ProtocolUnchanged",
    "RevisionInfo",
    "Saved",
    "Stored",
    "StudyInfo",
    "WorkspaceProtocolStore",
]
