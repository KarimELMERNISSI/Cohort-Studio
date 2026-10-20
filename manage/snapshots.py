"""
The snapshots of the Enrich draft (M5b).

Each change prepared on the Enrich page writes a copy of the table to the snapshots folder of the workspace
(``traces/snapshots``) before it joins the draft. Applying the draft writes the snapshot's path into the trace: the
replay uses it for a step it cannot recompute, and Reproduce loads it in its fast mode. A change that never reaches
the trace left its copy behind: a draft discarded, redone on newer data or dropped when another dataset is loaded,
and a change that changed nothing. :func:`delete_unreferenced` deletes those copies.

A file is deleted only when it is a ``.parquet`` file directly in the snapshots folder and no trace of the workspace
names it. The replay also finds a file by its name alone (``utils.path_utils.resolve_path``), so a trace that names a
file of the same name in any folder keeps it. When a trace cannot be read, nothing is deleted.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from core.workspace import UserScope, resolve_within
from utils.data_paths import get_data_root, get_traces_dir
from utils.logging_config import single_line

logger = logging.getLogger(__name__)

SUFFIX = ".parquet"


def _reason(exc: BaseException) -> str:
    """Why an operation failed, without the full path (it names the server's account)."""
    return (exc.strerror if isinstance(exc, OSError) else None) or type(exc).__name__


def folders(scope: UserScope | None) -> tuple[Path, Path, Path]:
    """
    The folder that recorded paths are relative to, the traces folder and the snapshots folder of ``scope``; outside
    a session, those of the data folder (as ``AppState.recorded_path`` and the Enrich page use them).
    """
    if scope is not None:
        return scope.root, scope.traces_dir(), scope.snapshots_dir()
    traces = Path(get_traces_dir())
    return Path(get_data_root()), traces, traces / "snapshots"


def _name(value: str) -> str:
    """The file name of a recorded path, whatever its separators (a trace recorded on Windows keeps backslashes)."""
    return value.replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _parquet_names(value: Any) -> Iterator[str]:
    """The names of the .parquet files mentioned anywhere in a trace."""
    if isinstance(value, str):
        if value.casefold().endswith(SUFFIX):
            yield _name(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _parquet_names(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _parquet_names(item)


def referenced(traces: Path, steps: Iterable[Mapping[str, Any]] = ()) -> set[str] | None:
    """
    The names (case-folded) of the .parquet files named by the traces saved in ``traces`` and by ``steps`` (the
    session's own trace); ``None`` when a trace cannot be read.
    """
    names = set(_parquet_names(list(steps)))
    for path in sorted(traces.glob("*.json")):
        try:
            names.update(_parquet_names(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError, RecursionError) as exc:  # RecursionError: nested too deeply
            name = single_line(path.name)
            logger.warning("Snapshots kept: the trace %s cannot be read (%s)", name, _reason(exc))
            return None
    return names


def snapshot_file(value: str | None, root: Path, snapshots: Path) -> Path | None:
    """The file a recorded path names when it is a .parquet file directly in ``snapshots``, otherwise ``None``."""
    if not value:
        return None
    try:  # relative to the workspace (AppState.recorded_path), or absolute
        path = resolve_within(snapshots, os.path.join(os.path.realpath(root), str(value)))
    except ValueError:
        return None  # outside the snapshots folder: never deleted
    if path.parent != Path(os.path.realpath(snapshots)) or path.suffix.casefold() != SUFFIX:
        return None
    return path


def delete_unreferenced(
    paths: Iterable[str | None],
    scope: UserScope | None,
    steps: Iterable[Mapping[str, Any]] = (),
) -> list[str]:
    """
    Delete the snapshots among ``paths`` (recorded paths of changes that will never reach the trace) that no trace
    names, ``steps`` included; returns the names deleted. Best effort: a file that cannot be deleted is logged and
    kept.
    """
    try:
        root, traces, snapshots = folders(scope)
    except OSError as exc:
        logger.warning("Snapshots kept: the workspace cannot be read (%s)", _reason(exc))
        return []
    files = [f for f in (snapshot_file(p, root, snapshots) for p in paths) if f is not None]
    files = [f for f in dict.fromkeys(files) if f.is_file()]
    if not files:
        return []
    names = referenced(traces, steps)
    if names is None:
        return []
    deleted = []
    for path in files:
        if path.name.casefold() in names:
            continue
        try:
            path.unlink()
        except OSError as exc:
            logger.warning("Snapshot %s not deleted (%s)", single_line(path.name), _reason(exc))
            continue
        deleted.append(path.name)
    if deleted:
        logger.info("Deleted %d snapshot(s) of changes that never reached the trace", len(deleted))
    return deleted


__all__ = ["delete_unreferenced", "folders", "referenced", "snapshot_file"]
