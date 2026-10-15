"""
Files of a multiple imputation saved with a dataset version (in the ``mi/`` folder of the datasets).

``<version>.cells.parquet``  the imputed cells: row id, column, imputation number, value (number or text)
``<version>.rows.parquet``   row id, hash of the model columns at fitting time, stratum (binding)
``<version>.json``           the binding (spec, methods, population), the engine information, diagnostics

The observed data are the saved version itself; together they rebuild the ``MIData`` and its status
against the current analysis population. Writes are atomic (temporary file, then replace).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biostats.mi.binding import MIBinding, StoredMI
from biostats.mi.data import MIData

FOLDER = "mi"


def mi_dir(dataset_dir: str | Path) -> Path:
    return Path(dataset_dir) / FOLDER


def _paths(dataset_dir: str | Path, name: str) -> dict[str, Path]:
    folder = mi_dir(dataset_dir)
    return {
        "cells": folder / f"{name}.cells.parquet",
        "rows": folder / f"{name}.rows.parquet",
        "meta": folder / f"{name}.json",
    }


def _atomic(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_mi(stored: StoredMI, dataset_dir: str | Path, name: str) -> list[Path]:
    """Write the three files of ``stored`` for the saved version ``name``."""
    paths = _paths(dataset_dir, name)
    paths["cells"].parent.mkdir(parents=True, exist_ok=True)
    mi = stored.mi
    parts = []
    for col, frame in mi.imputations.items():
        long = frame.stack(future_stack=True).rename("value").reset_index()
        long.columns = ["row_id", "imp", "value"]
        numeric = pd.api.types.is_numeric_dtype(mi.base[col]) and not pd.api.types.is_bool_dtype(
            mi.base[col]
        )
        parts.append(
            pd.DataFrame(
                {
                    "row_id": long["row_id"].to_numpy(),
                    "column": col,
                    "imp": long["imp"].astype("int16").to_numpy(),
                    "num": pd.to_numeric(long["value"], errors="coerce").astype(float)
                    if numeric
                    else np.nan,
                    "text": None if numeric else long["value"].astype(str),
                }
            )
        )
    cells = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["row_id", "column", "imp", "num", "text"])
    )
    cells["text"] = cells["text"].astype("string")
    _atomic(paths["cells"], lambda p: cells.to_parquet(p, index=False))
    rows = pd.DataFrame(
        {"row_id": stored.hashes.index.to_numpy(), "hash": stored.hashes.to_numpy(dtype="uint64")}
    )
    if stored.strata is not None:
        rows["stratum"] = stored.strata.loc[stored.hashes.index].astype(str).to_numpy()
    _atomic(paths["rows"], lambda p: rows.to_parquet(p, index=False))
    meta = {
        "binding": stored.binding.model_dump(),
        "m": mi.m,
        "methods": dict(mi.methods),
        "info": _jsonable(dict(mi.info)),
        "diagnostics": _jsonable(stored.diagnostics),
    }
    _atomic(
        paths["meta"],
        lambda p: p.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8"),
    )
    return list(paths.values())


def has_mi(dataset_dir: str | Path, name: str) -> bool:
    return all(p.exists() for p in _paths(dataset_dir, name).values())


def load_mi(dataset_dir: str | Path, name: str, base: pd.DataFrame) -> StoredMI | None:
    """Rebuild the imputation saved with version ``name`` on its observed data ``base`` (None if absent)."""
    if not has_mi(dataset_dir, name):
        return None
    paths = _paths(dataset_dir, name)
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    cells = pd.read_parquet(paths["cells"])
    m = int(meta["m"])
    imputations: dict[str, pd.DataFrame] = {}
    for col, part in cells.groupby("column", sort=False):
        numeric = pd.api.types.is_numeric_dtype(base[col]) and not pd.api.types.is_bool_dtype(
            base[col]
        )
        values = part["num"] if numeric else part["text"].astype(object)
        wide = pd.DataFrame(
            {"row_id": part["row_id"], "imp": part["imp"].astype(int), "value": values}
        )
        frame = wide.pivot(index="row_id", columns="imp", values="value")
        frame = frame.reindex(columns=range(1, m + 1))
        frame.index.name = base.index.name
        imputations[str(col)] = frame
    rows = pd.read_parquet(paths["rows"])
    hashes = pd.Series(
        rows["hash"].to_numpy(dtype="uint64"), index=pd.Index(rows["row_id"], name=base.index.name)
    )
    strata = (
        pd.Series(rows["stratum"].to_numpy(), index=hashes.index)
        if "stratum" in rows.columns
        else None
    )
    absent = hashes.index.difference(base.index)
    if len(absent):
        raise ValueError(
            f"{len(absent)} imputed row(s) are not in the saved data: the imputation cannot be restored"
        )
    mi = MIData(
        base=base.loc[hashes.index],  # the rows the imputation was fitted on (the population of the time)
        imputations=imputations,
        m=m,
        methods=meta.get("methods", {}),
        info=meta.get("info", {}),
    )
    return StoredMI(
        mi=mi,
        binding=MIBinding(**meta["binding"]),
        hashes=hashes,
        strata=strata,
        diagnostics=meta.get("diagnostics", {}),
    )


def delete_mi(dataset_dir: str | Path, name: str) -> None:
    for path in _paths(dataset_dir, name).values():
        path.unlink(missing_ok=True)


__all__ = ["FOLDER", "delete_mi", "has_mi", "load_mi", "mi_dir", "save_mi"]
