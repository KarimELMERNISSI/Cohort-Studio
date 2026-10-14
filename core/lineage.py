"""
Lineage of the session data: what each change did, and a tamper-evident chain of trace steps.

Every trace step written since schema 2 carries ``prev_hash`` and ``hash``: the SHA-256 of the previous
step's hash and the step's canonical JSON. Editing, removing or reordering a step breaks the chain, which
:func:`verify_trace` reports. This makes the trace tamper-*evident*; it is not an electronic signature.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import PurePath
from typing import Any

import numpy as np
import pandas as pd

from core.fingerprint import frame_fingerprint

TRACE_SCHEMA = 2


def normalise(value: Any) -> Any:
    """JSON-able copy of ``value``: numpy scalars to Python, NaN/inf to None, tuples/sets to lists."""
    if isinstance(value, Mapping):
        return {str(k): normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalise(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((normalise(v) for v in value), key=str)
    if isinstance(value, np.ndarray):
        return [normalise(v) for v in value.tolist()]
    if isinstance(value, (pd.Index, pd.Series)):
        return [normalise(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, PurePath):
        return str(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (pd.DataFrame,)):
        return "DataFrame (not serialisable)"
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(
        normalise(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def genesis_hash(session_id: str | None) -> str:
    """Start of the chain of a trace session."""
    return hashlib.sha256(
        canonical_json({"schema_version": TRACE_SCHEMA, "session_id": session_id}).encode("utf-8")
    ).hexdigest()


def step_hash(step: Mapping[str, Any], prev_hash: str) -> str:
    body = {k: v for k, v in step.items() if k != "hash"}
    return hashlib.sha256((prev_hash + canonical_json(body)).encode("utf-8")).hexdigest()


def seal_step(step: Mapping[str, Any], prev_hash: str) -> dict[str, Any]:
    """The step, normalised, with ``prev_hash`` and ``hash`` set."""
    sealed = normalise(dict(step))
    sealed.pop("hash", None)
    sealed["prev_hash"] = prev_hash
    sealed["hash"] = step_hash(sealed, prev_hash)
    return sealed


VERIFIED, UNVERIFIABLE, TAMPERED, DISCONTINUOUS = (
    "verified",
    "unverifiable",
    "tampered",
    "discontinuous",
)


@dataclass(frozen=True)
class TraceVerification:
    status: str  # VERIFIED | UNVERIFIABLE | TAMPERED | DISCONTINUOUS
    n_steps: int
    head_hash: str | None = None
    first_bad_step: int | None = None  # 1-based
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == VERIFIED


def verify_trace(trace: Mapping[str, Any]) -> TraceVerification:
    """Check the hash chain of a schema-2 trace and the continuity of its data fingerprints."""
    steps = list(trace.get("steps") or [])
    if int(trace.get("schema_version") or 1) < TRACE_SCHEMA:
        return TraceVerification(
            UNVERIFIABLE,
            len(steps),
            message="Recorded before traces were chained (schema 1): integrity cannot be verified.",
        )
    prev = genesis_hash(trace.get("session_id"))
    last_fp: str | None = None
    for i, step in enumerate(steps, start=1):
        if step.get("prev_hash") != prev or step.get("hash") != step_hash(step, prev):
            return TraceVerification(
                TAMPERED,
                len(steps),
                first_bad_step=i,
                message=f"Step {i} was modified, removed or reordered after it was recorded.",
            )
        prev = str(step["hash"])
        data = step.get("data") or {}
        if data.get("fp_in") and last_fp and data["fp_in"] != last_fp:
            return TraceVerification(
                DISCONTINUOUS,
                len(steps),
                head_hash=prev,
                first_bad_step=i,
                message=f"Step {i} starts from data that the previous step did not produce.",
            )
        if data.get("fp_out"):
            last_fp = str(data["fp_out"])
    head = trace.get("head_hash")
    if head and head != prev:
        return TraceVerification(
            TAMPERED,
            len(steps),
            first_bad_step=len(steps),
            message="The last steps of the trace were removed.",
        )
    return TraceVerification(VERIFIED, len(steps), head_hash=prev, message="Hash chain intact.")


@dataclass(frozen=True)
class Change:
    """What one step did to the table (for the trace and the participant flow)."""

    rows_in: int
    rows_out: int
    cols_in: int
    cols_out: int
    fp_in: str
    fp_out: str
    columns_added: tuple[str, ...] = ()
    columns_removed: tuple[str, ...] = ()
    columns_retyped: tuple[str, ...] = ()
    rows_added: int = 0
    rows_removed: int = 0

    @property
    def changed(self) -> bool:
        return self.fp_in != self.fp_out

    def to_dict(self) -> dict[str, Any]:
        return normalise(asdict(self))


def describe_change(
    before: pd.DataFrame | None,
    after: pd.DataFrame,
    *,
    fp_in: str | None = None,
    fp_out: str | None = None,
) -> Change:
    if before is None:
        before = pd.DataFrame()
    cols_before = [str(c) for c in before.columns]
    cols_after = [str(c) for c in after.columns]
    common = [c for c in after.columns if c in before.columns]
    retyped = tuple(str(c) for c in common if str(before[c].dtype) != str(after[c].dtype))
    ids_before = set(before.index.tolist()) if len(before) else set()
    ids_after = set(after.index.tolist())
    return Change(
        rows_in=len(before),
        rows_out=len(after),
        cols_in=before.shape[1],
        cols_out=after.shape[1],
        fp_in=fp_in or (frame_fingerprint(before) if len(before.columns) else ""),
        fp_out=fp_out or frame_fingerprint(after),
        columns_added=tuple(c for c in cols_after if c not in cols_before),
        columns_removed=tuple(c for c in cols_before if c not in cols_after),
        columns_retyped=retyped,
        rows_added=len(ids_after - ids_before),
        rows_removed=len(ids_before - ids_after),
    )


__all__ = [
    "DISCONTINUOUS",
    "TAMPERED",
    "TRACE_SCHEMA",
    "UNVERIFIABLE",
    "VERIFIED",
    "Change",
    "TraceVerification",
    "canonical_json",
    "describe_change",
    "genesis_hash",
    "normalise",
    "seal_step",
    "step_hash",
    "verify_trace",
]
