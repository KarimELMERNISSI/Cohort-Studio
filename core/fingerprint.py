"""
Content fingerprints of data frames (audit E8).

A fingerprint identifies the *content* a result was computed on, so cached results can follow the data
(imputation, clipping, renaming, filtering) instead of a dataset name or a row count.
"""

from __future__ import annotations

import hashlib

import pandas as pd


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


__all__ = ["frame_fingerprint"]
