"""
The files a replay may read (M5b step 5): where the paths a trace records are looked for.

A trace is user-supplied JSON, so in a session a path it records is never read as written. :class:`PathResolver`
finds each file in the places the session may read:

- the files the user chose for recorded paths on the Reproduce page (``named``), first, in the same places;
- the user's workspace (``scope``): the recorded path when it lies inside it, else the file's name in its folders;
- for a command line (M8), only the files named on it, found by their names (:meth:`PathResolver.for_files`);
- for tests and scripts (:meth:`PathResolver.unconfined`): the recorded path when it exists, else the file's name in
  the shared folders of the data root.

A resolver given none of these reads no file, as a replay given no resolver: a session without a signed-in user
reads nothing (every signed-in user has a workspace, :func:`core.workspace.safe_dirname`). A saved version a trace
names is looked up in the datasets folder only (:meth:`PathResolver.dataset`): a name such as
``../../bob/datasets/x`` finds nothing. A path on a network share is never opened
(:func:`core.workspace.network_share`).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from core.workspace import UserScope, network_share, resolve_within
from utils.logging_config import single_line

logger = logging.getLogger(__name__)


def normalize_path(path_str: str) -> str:
    """A path with forward slashes: one recorded on Windows keeps its backslashes, which are no separators on Linux."""
    return path_str.replace("\\", "/") if path_str else path_str


def read_version(path: str) -> pd.DataFrame | None:
    """A saved version's table: read by pandas, or by DuckDB for a file pandas cannot read. None when it is missing."""
    try:
        return pd.read_parquet(path)  # the row ids come back as the index
    except FileNotFoundError:
        return None
    except Exception:
        import duckdb

        try:
            with duckdb.connect() as con:
                return con.execute("SELECT * FROM read_parquet(?)", [path]).df()
        except Exception:
            return None


@dataclass(frozen=True)
class PathResolver:
    """Where the files a trace names are looked for (see the module docstring). ``PathResolver()`` reads none."""

    scope: UserScope | None = None
    # recorded path -> the file chosen for it
    named: Mapping[str, str] = field(default_factory=dict)
    files: tuple[str, ...] | None = None  # only these files, found by their names; () reads none
    shared: bool = False  # without a user: the shared folders of the data root (unconfined())

    @classmethod
    def for_files(cls, files: Iterable[str | Path]) -> PathResolver:
        """Only these files, found by their names (a command line's): nothing else is read."""
        return cls(files=tuple(str(Path(f).resolve()) for f in files))

    @classmethod
    def unconfined(cls) -> PathResolver:
        """Tests and scripts, without a user: the recorded path when it exists, else its name in the shared folders of
        the data root. Never for a session: a trace could name any file of the server."""
        return cls(shared=True)

    def with_named(self, named: Mapping[str, str]) -> PathResolver:
        """The same places, and the files the user chose for recorded paths (Reproduce)."""
        return replace(self, named=dict(named))

    def resolve(self, recorded: object) -> str | None:
        """The file to read for a path a trace records, or None when the session may read none."""
        if not isinstance(recorded, str) or not recorded:
            return None
        chosen = self._chosen(recorded)
        if chosen is not None:
            return chosen
        name = os.path.basename(normalize_path(recorded))
        if name in ("", ".", ".."):
            return None
        if self.files is not None:
            return self._named_file(name)
        if self.scope is not None:
            # the recorded path itself, only inside the user's workspace (audit E4)
            try:
                candidate = self.scope.resolve_within(recorded)
                if candidate.is_file():
                    return str(candidate)
            except ValueError:
                pass  # outside the workspace: only the file's name is looked for below
        elif not self.shared:
            return None
        elif not network_share(recorded) and os.path.isfile(recorded):
            return recorded
        elif not os.path.isabs(recorded):
            # recorded relative to the data folder (AppState.recorded_path)
            from utils.data_paths import get_data_root

            try:
                candidate = resolve_within(Path(get_data_root()), recorded)
            except ValueError:
                candidate = None
            if candidate is not None and candidate.is_file():
                return str(candidate)
        for directory in self._search_dirs():
            try:  # the recorded name, in each folder, never outside it
                found = str(resolve_within(Path(directory), name))
            except ValueError:
                continue
            if os.path.isfile(found):
                logger.info(
                    "Path resolution found %s for %s", single_line(found), single_line(recorded)
                )
                return found
        return None

    def dataset(self, name: object) -> str | None:
        """The saved version ``name`` (``<datasets>/<name>.parquet``), never a file outside that folder: the name comes
        from a trace."""
        if not isinstance(name, str) or not name or name in (".", ".."):
            return None
        if os.path.basename(normalize_path(name)) != name:
            return None
        if self.files is not None:
            return self._named_file(f"{name}.parquet")
        if self.scope is not None:
            folder = self.scope.datasets_dir()
        elif self.shared:
            from utils.data_paths import get_datasets_dir

            folder = Path(get_datasets_dir())
        else:
            return None
        try:
            path = resolve_within(folder, f"{name}.parquet")
        except ValueError:
            return None
        return str(path) if path.is_file() else None

    def _chosen(self, recorded: str) -> str | None:
        """The file the user chose for ``recorded``, in the places this resolver reads."""
        chosen = self.named.get(recorded)
        if not chosen:
            return None
        if self.files is not None:
            if network_share(chosen):
                return None
            chosen = str(Path(chosen).resolve())
            if chosen not in self.files:
                return None
        elif self.scope is not None:
            try:
                chosen = str(self.scope.resolve_within(chosen))
            except ValueError:
                return None
        elif not self.shared or network_share(chosen):
            return None
        return chosen if os.path.isfile(chosen) else None

    def _named_file(self, name: str) -> str | None:
        return next(
            (f for f in self.files or () if os.path.basename(f) == name and os.path.isfile(f)), None
        )

    def _search_dirs(self) -> list[str]:
        if self.scope is not None:
            return [str(p) for p in self.scope.search_dirs()]
        from utils.data_paths import get_data_root, get_traces_dir, get_uploads_dir

        return [get_data_root(), get_uploads_dir(), get_traces_dir()]


__all__ = ["PathResolver", "normalize_path", "read_version"]
