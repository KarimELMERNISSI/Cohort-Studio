"""
A multiply imputed dataset, stored compactly (like R's ``mids``).

The observed table is kept once, with its missing cells; for each imputed column a small frame holds the
m imputed values of its missing cells (rows indexed by the row id, columns 1..m). A completed dataset is
the observed table with imputation i written into its missing cells. Memory grows with the number of
missing cells, not with m copies of the table.

Formats: ``long()`` returns the mice "long" layout (``.imp`` 0..m, ``.id`` the row id) that R
(``mice::as.mids``) and Stata (``mi import flong``) read; ``from_long`` reads it back, and
``from_completed`` builds the object from m completed tables.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MIData:
    base: pd.DataFrame  # the observed data, NaN in the missing cells; index = row ids
    imputations: Mapping[str, pd.DataFrame]  # column -> (row id x imputation 1..m) imputed values
    m: int
    methods: Mapping[str, str] = field(default_factory=dict)  # column -> imputation method
    info: Mapping[str, Any] = field(default_factory=dict)  # seed, engine, population, model ...

    def __post_init__(self) -> None:
        if self.m < 1:
            raise ValueError("m must be at least 1")
        if not self.base.index.is_unique:
            raise ValueError("the row ids (index of the observed data) must be unique")
        for col, frame in self.imputations.items():
            if col not in self.base.columns:
                raise ValueError(f"imputed column {col!r} is not in the data")
            if list(frame.columns) != list(range(1, self.m + 1)):
                raise ValueError(f"the imputations of {col!r} must be columns 1..{self.m}")
            missing = self.base.index[self.base[col].isna()]
            if not frame.index.isin(missing).all():
                raise ValueError(f"imputed values of {col!r} target cells that are observed")

    # --- description --------------------------------------------------------------------------------

    @property
    def imputed_columns(self) -> list[str]:
        return [c for c in self.base.columns if c in self.imputations]

    def n_imputed(self) -> dict[str, int]:
        """Number of imputed cells per column."""
        return {c: len(self.imputations[c]) for c in self.imputed_columns}

    def levels(self, columns: Sequence[str]) -> dict[str, list[str]]:
        """Factor levels of the observed data (what every completed dataset must share)."""
        from biostats.design import factor_levels

        return factor_levels(self.base, columns)

    # --- completed datasets -------------------------------------------------------------------------

    def complete(self, i: int, columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Completed dataset i (1..m); 0 returns the observed data with its missing cells."""
        if not 0 <= i <= self.m:
            raise ValueError(f"imputation {i} does not exist (0..{self.m})")
        cols = list(columns) if columns is not None else list(self.base.columns)
        out = self.base[cols].copy()
        if i == 0:
            return out
        for col in cols:
            frame = self.imputations.get(col)
            if frame is None or frame.empty:
                continue
            values = frame[i]
            if isinstance(out[col].dtype, pd.CategoricalDtype):
                out[col] = out[col].astype(object)
                out.loc[frame.index, col] = values.to_numpy()
                out[col] = pd.Categorical(out[col], categories=self.base[col].cat.categories)
            else:
                out.loc[frame.index, col] = values.to_numpy()
        return out

    def completed(self, columns: Sequence[str] | None = None) -> Iterator[tuple[int, pd.DataFrame]]:
        for i in range(1, self.m + 1):
            yield i, self.complete(i, columns)

    def subset(self, rows: pd.Index | Sequence[Any]) -> MIData:
        """The same imputations restricted to ``rows`` (row ids of the observed data)."""
        index = pd.Index(rows)
        unknown = index.difference(self.base.index)
        if len(unknown):
            raise ValueError(f"{len(unknown)} row id(s) are not in the imputed data")
        imputations = {
            col: frame.loc[frame.index.intersection(index)]
            for col, frame in self.imputations.items()
        }
        return MIData(
            base=self.base.loc[index],
            imputations=imputations,
            m=self.m,
            methods=self.methods,
            info=self.info,
        )

    # --- formats ------------------------------------------------------------------------------------

    def long(self, *, include_original: bool = True) -> pd.DataFrame:
        """The mice long layout: ``.imp`` (0 = observed data) and ``.id`` (row id) followed by the columns."""
        parts = []
        for i in range(0 if include_original else 1, self.m + 1):
            frame = self.complete(i).rename_axis(".id").reset_index()
            frame.insert(0, ".imp", i)
            parts.append(frame)
        return pd.concat(parts, ignore_index=True)

    def flong(self) -> pd.DataFrame:
        """
        The Stata "flong" layout: ``_mi_m`` (0 = observed data) and ``_mi_id`` (row id) followed by the columns;
        in Stata, ``mi import flong, m(_mi_m) id(_mi_id) imputed(<the imputed variables>)``.
        """
        return self.long().rename(columns={".imp": "_mi_m", ".id": "_mi_id"})

    @classmethod
    def from_completed(
        cls,
        base: pd.DataFrame,
        completed: Sequence[pd.DataFrame],
        *,
        methods: Mapping[str, str] | None = None,
        info: Mapping[str, Any] | None = None,
    ) -> MIData:
        """Build from the observed data and m completed tables (same index and columns as ``base``)."""
        if not completed:
            raise ValueError("no completed dataset")
        imputations: dict[str, pd.DataFrame] = {}
        for col in base.columns:
            missing = base.index[base[col].isna()]
            if len(missing) == 0:
                continue
            imputations[col] = pd.DataFrame(
                {i + 1: frame.loc[missing, col].to_numpy() for i, frame in enumerate(completed)},
                index=missing,
            )
        return cls(
            base=base,
            imputations=imputations,
            m=len(completed),
            methods=dict(methods or {}),
            info=dict(info or {}),
        )

    @classmethod
    def from_long(cls, long: pd.DataFrame, *, imp_col: str = ".imp", id_col: str = ".id") -> MIData:
        """
        Read the mice long layout. With the ``.imp == 0`` rows, those give the observed data; without them a
        cell is taken as imputed when its value differs between the imputations.
        """
        data_cols = [c for c in long.columns if c not in (imp_col, id_col)]
        groups = {
            int(k): g.set_index(id_col)[data_cols].sort_index()
            for k, g in long.groupby(imp_col, sort=True)
        }
        completed = [groups[k] for k in sorted(groups) if k > 0]
        if not completed:
            raise ValueError("the long data hold no imputed dataset (.imp > 0)")
        if 0 in groups:
            base = groups[0]
        else:
            base = completed[0].copy()
            for col in data_cols:
                stacked = pd.concat([c[col] for c in completed], axis=1)
                varies = stacked.astype(object).nunique(axis=1, dropna=False) > 1
                if varies.any():
                    base[col] = (
                        base[col].astype(object)
                        if not pd.api.types.is_numeric_dtype(base[col])
                        else base[col].astype(float)
                    )
                    base.loc[varies, col] = np.nan
        return cls.from_completed(base, completed)


__all__ = ["MIData"]
