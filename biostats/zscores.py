"""
Reference-population Z-scores.

Audit finding S5: the page tested ``if ref_group_col and ref_group_val`` so a control group coded
``0`` or ``False`` silently fell back to mean 0 / SD 1 (the "Z-scores" were the raw values), and the
manual-reference mode had no inputs. This module does the computation explicitly and raises when a
reference cannot be established.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

Method = Literal["cohort", "control", "manual"]


@dataclass(frozen=True)
class ReferenceStats:
    mean: float
    std: float
    n: int
    source: str


def _control_mask(df: pd.DataFrame, ref_col: str, ref_value: Any) -> pd.Series:
    mask = df[ref_col] == ref_value
    if not mask.any():
        mask = df[ref_col].astype(str).str.strip() == str(ref_value).strip()
    return mask


def reference_zscores(
    df: pd.DataFrame,
    variables: list[str],
    method: Method,
    *,
    ref_col: str | None = None,
    ref_value: Any | None = None,
    manual: dict[str, tuple[float, float]] | None = None,
    ddof: int = 1,
    suffix: str = "_Z",
) -> tuple[pd.DataFrame, dict[str, ReferenceStats]]:
    """Return a copy of ``df`` with ``<var>_Z`` columns and the reference statistics used.

    Methods: ``cohort`` (whole-cohort mean/SD), ``control`` (mean/SD of the rows where
    ``ref_col == ref_value``; ``ref_value`` may be 0/False), ``manual`` (``{var: (mean, sd)}``).
    """
    if not variables:
        raise ValueError("Select at least one variable.")
    out = df.copy()
    stats: dict[str, ReferenceStats] = {}

    # the reference rows: all, unless a control group is chosen
    mask = pd.Series(True, index=df.index)
    if method == "control":
        if ref_col is None or ref_value is None:
            raise ValueError(
                "A grouping variable and a reference level are required for the control-group method."
            )
        mask = _control_mask(df, ref_col, ref_value)
        if not mask.any():
            raise ValueError(f"No rows found where {ref_col!r} equals {ref_value!r}.")
    elif method == "manual":
        manual = manual or {}
        missing = [v for v in variables if v not in manual]
        if missing:
            raise ValueError(f"Manual reference values are missing for: {', '.join(missing)}")
    elif method != "cohort":
        raise ValueError(f"Unknown method {method!r}")

    for var in variables:
        values = pd.to_numeric(df[var], errors="coerce")
        if method == "cohort":
            ref = values
            source = "whole cohort"
        elif method == "control":
            ref = values[mask]
            source = f"{ref_col} = {ref_value}"
        else:
            mu, sigma = manual[var]  # type: ignore[index]
            if sigma is None or float(sigma) <= 0:
                raise ValueError(f"The reference SD for {var!r} must be > 0.")
            stats[var] = ReferenceStats(
                mean=float(mu), std=float(sigma), n=0, source="manual reference values"
            )
            out[f"{var}{suffix}"] = (values - float(mu)) / float(sigma)
            continue
        ref = ref.dropna()
        if len(ref) < 2:
            raise ValueError(f"The reference population for {var!r} has fewer than 2 observations.")
        mu = float(ref.mean())
        sigma = float(ref.std(ddof=ddof))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"The reference SD for {var!r} is zero; Z-scores are undefined.")
        stats[var] = ReferenceStats(mean=mu, std=sigma, n=int(len(ref)), source=source)
        out[f"{var}{suffix}"] = (values - mu) / sigma
    return out, stats


__all__ = ["Method", "ReferenceStats", "reference_zscores"]
