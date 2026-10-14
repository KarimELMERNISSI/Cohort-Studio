"""
Content fingerprints and row identity of data frames (audit E8).

A fingerprint identifies the *content* a result was computed on, so cached results can follow the data
(imputation, clipping, renaming, filtering) instead of a dataset name or a row count. Row identity
(``__row_id``, kept in the index) lets traces, populations and imputations name the same rows across
filters, merges, saves and reloads.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

ROW_ID = "__row_id"


def frame_fingerprint(df: pd.DataFrame, *, index: bool = True, length: int = 16) -> str:
    """Hex digest of a frame's shape, column names, dtypes and values (and row labels when ``index``).

    Equal frames give equal fingerprints. Changing a value, a column name, the column order, a dtype or,
    with ``index``, a row label gives a different one. Cells pandas cannot hash (lists, dicts) are hashed
    through their string form.
    """
    digest = hashlib.sha256()
    digest.update(
        repr((df.shape, [str(c) for c in df.columns], [str(t) for t in df.dtypes])).encode()
    )
    try:
        values = pd.util.hash_pandas_object(df, index=index)
    except TypeError:
        values = pd.util.hash_pandas_object(df.astype(str), index=index)
    digest.update(values.to_numpy().tobytes())
    return digest.hexdigest()[:length]


def frame_sentinel(df: pd.DataFrame, sample: int = 256) -> str:
    """Cheap fingerprint: shape, names, dtypes and up to ``sample`` evenly spaced rows (for per-run checks)."""
    if len(df) > sample:
        part = df.iloc[np.linspace(0, len(df) - 1, sample).astype(int)]
    else:
        part = df
    return f"{df.shape}:{frame_fingerprint(part)}"


def ensure_row_ids(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` with a unique integer row identity in an index named ``__row_id``.

    A persisted ``__row_id`` column becomes the index; an existing unique integer index keeps its labels
    (rows filtered out of a table keep the ids of the original rows); anything else is numbered 0..n-1.
    """
    if ROW_ID in df.columns:
        out = df.set_index(ROW_ID)
    elif df.index.name == ROW_ID:
        out = df
    elif pd.api.types.is_integer_dtype(df.index.dtype) and df.index.is_unique:
        out = df.rename_axis(index=ROW_ID)
    else:
        out = df.set_axis(pd.RangeIndex(len(df), name=ROW_ID), axis=0)
    if not out.index.is_unique:
        raise ValueError(f"row identifiers ({ROW_ID}) must be unique")
    return out


def extend_row_ids(df: pd.DataFrame, used: Iterable[int] = ()) -> pd.DataFrame:
    """Row ids after a join: rows keep the id they carry (``__row_id`` column or index); rows without one
    (added by an outer or right join) or repeating an earlier row's id (one-to-many join) get fresh ids
    above every id in ``used`` and in ``df``, so an id is never given to two different rows."""
    raw = df[ROW_ID] if ROW_ID in df.columns else pd.Series(df.index, index=df.index)
    ids = pd.to_numeric(raw, errors="coerce").astype("float64").reset_index(drop=True)
    fresh = (ids.isna() | ids.duplicated(keep="first")).to_numpy()
    taken = [int(i) for i in used] + [int(i) for i in ids[~fresh]]
    start = max(taken) + 1 if taken else 0
    values = ids.to_numpy(copy=True)
    values[fresh] = np.arange(start, start + int(fresh.sum()))
    body = df.drop(columns=[ROW_ID]) if ROW_ID in df.columns else df
    return body.set_axis(pd.Index(values.astype("int64"), name=ROW_ID), axis=0)


def ids_to_ranges(ids: Iterable[int]) -> list[list[int]]:
    """Sorted unique integer ids as inclusive ``[first, last]`` ranges (compact in traces)."""
    values = np.unique(np.asarray(list(ids), dtype="int64"))
    if values.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(values) != 1)
    starts = np.concatenate(([values[0]], values[breaks + 1]))
    ends = np.concatenate((values[breaks], [values[-1]]))
    return [[int(a), int(b)] for a, b in zip(starts, ends, strict=True)]


def ranges_to_ids(ranges: Sequence[Sequence[int]]) -> np.ndarray:
    """Inverse of :func:`ids_to_ranges`."""
    if not ranges:
        return np.array([], dtype="int64")
    return np.concatenate([np.arange(int(a), int(b) + 1, dtype="int64") for a, b in ranges])


def ids_fingerprint(ids: Iterable[int], length: int = 16) -> str:
    """Fingerprint of a set of row ids (order does not matter)."""
    values = np.unique(np.asarray(list(ids), dtype="int64"))
    return hashlib.sha256(values.tobytes()).hexdigest()[:length]


__all__ = [
    "ROW_ID",
    "ensure_row_ids",
    "extend_row_ids",
    "frame_fingerprint",
    "frame_sentinel",
    "ids_fingerprint",
    "ids_to_ranges",
    "ranges_to_ids",
]
