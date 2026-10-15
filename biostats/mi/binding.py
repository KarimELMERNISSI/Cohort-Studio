"""
What a multiple imputation was fitted on, and whether it still fits the data being analysed.

An imputation is fitted on the rows of the analysis population at that time. The binding keeps a hash of
the model columns for every one of those rows: membership and observed values are compared row by row,
so a change elsewhere in the table (a new derived variable) leaves the imputation valid. Statuses:

=========== ==========================================================================================
fresh       the same rows with the same observed values
by_subset   a union of whole strata of the ``by`` variable: exact, each stratum was imputed on its own
subset      a strict part of the rows: usable with a caveat (the imputation model saw other rows), or
            run again on the current rows
stale       rows that were not imputed, or observed values that changed: run the imputation again
=========== ==========================================================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from biostats.mi.data import MIData

Status = Literal["fresh", "by_subset", "subset", "stale"]


def row_hashes(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """One 64-bit hash per row of the model columns (missing cells included), indexed by row id."""
    return pd.util.hash_pandas_object(frame[list(columns)], index=True).astype("uint64")


class MIBinding(BaseModel):
    model_config = ConfigDict(frozen=True)

    columns: list[str]  # model columns whose observed values the imputation used
    by: str | None = None
    data_fp: str = ""  # fingerprint of the data at fitting time (provenance)
    population_fp: str = ""
    population: dict[str, Any] | None = None  # population snapshot (criteria, counts)
    n_rows: int
    created_at: str
    spec: dict[str, Any]  # the ImputationSpec
    methods: dict[str, str]


@dataclass
class StoredMI:
    mi: MIData
    binding: MIBinding
    hashes: pd.Series  # row id -> hash of the model columns at fitting time
    strata: pd.Series | None = None  # row id -> stratum label (``by``)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MIStatus:
    kind: Status
    reason: str
    n_rows: int  # rows of the current data the imputation covers

    @property
    def usable(self) -> bool:
        return self.kind in ("fresh", "by_subset")


def bind(
    run: Any,
    *,
    data_fp: str = "",
    population_fp: str = "",
    population: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> StoredMI:
    """Attach to an imputation run (biostats.mi.engine.ImputationRun) what it was fitted on."""
    mi: MIData = run.mi
    columns = [c for c in run.spec.columns if c in mi.base.columns]
    by = run.spec.by
    binding = MIBinding(
        columns=columns,
        by=by,
        data_fp=data_fp,
        population_fp=population_fp,
        population=population,
        n_rows=len(mi.base),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        spec=run.spec.model_dump(),
        methods=dict(run.methods),
    )
    diag = {
        "rhat": dict(run.rhat),
        "dropped": dict(run.dropped),
        "warnings": list(run.warnings),
        "elapsed": run.elapsed,
        # chain means and SDs per iteration (lists: they are saved with the version for the trace plots)
        "trace": {
            name: {k: v.tolist() for k, v in values.items()} for name, values in run.trace.items()
        },
    }
    return StoredMI(
        mi=mi,
        binding=binding,
        hashes=row_hashes(mi.base, columns),
        strata=mi.base[by].astype(str) if by else None,
        diagnostics={**diag, **(diagnostics or {})},
    )


def status(stored: StoredMI, frame: pd.DataFrame) -> MIStatus:
    """Whether ``stored`` fits ``frame`` (the current analysis rows)."""
    columns = stored.binding.columns
    gone = [c for c in columns if c not in frame.columns]
    if gone:
        return MIStatus("stale", f"column(s) {', '.join(gone)} no longer exist", 0)
    rows = frame.index
    fitted = stored.hashes.index
    new = rows.difference(fitted)
    if len(new):
        return MIStatus("stale", f"{len(new)} row(s) of the current population were not imputed", 0)
    current = row_hashes(frame, columns)
    changed = int((current != stored.hashes.loc[rows]).sum())
    if changed:
        return MIStatus(
            "stale", f"observed values changed in {changed} row(s) since the imputation", 0
        )
    if len(rows) == len(fitted):
        return MIStatus("fresh", "same rows and values as when it was imputed", len(rows))
    if stored.strata is not None:
        kept = stored.strata.loc[rows]
        whole = all(
            stored.strata[stored.strata == level].index.isin(rows).all() for level in kept.unique()
        )
        if whole:
            return MIStatus(
                "by_subset",
                f"whole {stored.binding.by} strata of the imputed rows; each stratum was imputed on its own",
                len(rows),
            )
    return MIStatus(
        "subset",
        f"{len(rows)} of the {len(fitted)} imputed rows: the imputation model also used rows outside the "
        "current population",
        len(rows),
    )


def restrict(stored: StoredMI, rows: pd.Index) -> MIData:
    """The imputed data restricted to ``rows`` (all imputed before)."""
    return stored.mi.subset(rows)


def hashes_equal(a: pd.Series, b: pd.Series) -> bool:
    return bool(
        len(a) == len(b) and np.array_equal(a.sort_index().to_numpy(), b.sort_index().to_numpy())
    )


__all__ = [
    "MIBinding",
    "MIStatus",
    "Status",
    "StoredMI",
    "bind",
    "hashes_equal",
    "restrict",
    "row_hashes",
    "status",
]
