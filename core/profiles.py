"""
Column profiles (M8, the study package): what each column holds after a step, written so that two environments can
compare it.

A frame's fingerprint (``core.fingerprint.frame_fingerprint``) rests on pandas' own hashing and on its names of the
types, which a pandas version may change: it is compared only between identical environments. A profile is built from
the values themselves: the column's kind, its rows and missing values, a digest of its values in the order of the row
ids, and for a numeric column the sums that say how far two columns are apart when their digests differ (a variable
computed with another BLAS can differ in its last bit).

A profile reads the values, not how pandas keeps them: whole numbers held as integers or as floats, a column of
True and False held as booleans or as objects, dates held in any unit or as Python dates, text held as objects, as
strings or as categories give the same profile.

The sums of a numeric column are what a comparison may allow for (``core.verification.compare_profiles``): the sum,
the sum of the absolute values, the sum of the squares, the two ends, and two sums weighted by the place of each row
(``mix`` and ``mix2``, with unrelated weights: they move when two rows exchange their values, which no other sum
sees). Each is allowed the rounding error of a sum, a few units in the last place of the sum of the absolute values,
never a share of the column's spread: one value changed by 0.2 among a million is a failure.

A profile holds a column's smallest and largest values, which are one participant's: profiles belong to the
reproduction tier of a package (encrypted, beside the data), never to a shareable one. ``FORMAT`` marks one wherever
it is nested, so that a writer can refuse it.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd

from core.fingerprint import ids_fingerprint

FORMAT = "cohortstudio.column-profile"
PROFILE_VERSION = (
    1  # of the profile's content and of its digests: a change makes older profiles not comparable
)

# the kinds of column (stored ids)
NUMBER, TEXT, DATE, DURATION, BOOLEAN = "number", "text", "date", "duration", "boolean"
KINDS = (NUMBER, TEXT, DATE, DURATION, BOOLEAN)

_EXACT_FLOAT = 2.0**53  # whole numbers up to here are the same as floats and as integers
_GOLDEN, _ROOT_TWO = 0.6180339887498949, 1.4142135623730951  # weights of `mix` and `mix2`


def _digest(*chunks: bytes) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _number(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def _weights(size: int, step: float) -> np.ndarray:
    """A weight in [0, 1) for each place: the fractional parts of the multiples of an irrational number never
    repeat, and two rows whose weights are close for one such number are far apart for another."""
    return np.modf(np.arange(1, size + 1, dtype="float64") * step)[0]


def _numeric_profile(series: pd.Series) -> dict[str, Any]:
    """
    ``whole`` says every value is a whole number (two such columns are equal or they are not: no rounding error).
    Whole numbers within 2**53 are digested as floats, so that integers and floats holding the same values agree;
    larger integers as 64-bit integers, which a float would round.
    """
    missing = series.isna().to_numpy()
    integers = pd.api.types.is_integer_dtype(series.dtype)
    real = pd.to_numeric(series, errors="coerce").to_numpy(dtype="<f8", na_value=np.nan)
    present = ~np.isnan(real)
    values = real[present]
    if integers and len(values) and float(np.abs(values).max()) > _EXACT_FLOAT:
        whole = series.fillna(0).to_numpy(dtype="<i8", na_value=0)
        digest = _digest(b"i8", whole.tobytes(), missing.astype("u1").tobytes())
    else:
        real = np.where(
            present, real + 0.0, np.nan
        )  # one NaN, and -0.0 as 0.0: equal values, equal bytes
        digest = _digest(b"f8", real.tobytes())
    # the sums are taken over the finite values, and the infinite ones counted: one infinite value (a ratio with a
    # zero denominator) would otherwise leave every sum without a number, and the column without a comparison
    is_finite = np.isfinite(real)
    finite = real[is_finite]
    found: dict[str, Any] = {
        "whole": bool(len(values) == len(finite) and (finite == np.floor(finite)).all()),
        "digest": digest,
        "plus_inf": int(np.isposinf(values).sum()),
        "minus_inf": int(np.isneginf(values).sum()),
    }
    if len(finite):
        places = np.flatnonzero(is_finite)
        with np.errstate(over="ignore", invalid="ignore"):
            found |= {
                "sum": _number(float(finite.sum())),
                "sum_abs": _number(float(np.abs(finite).sum())),
                "sum_sq": _number(float(np.square(finite).sum())),
                "min": _number(float(finite.min())),
                "max": _number(float(finite.max())),
                "mix": _number(float(np.sum(finite * _weights(len(real), _GOLDEN)[places]))),
                "mix2": _number(float(np.sum(finite * _weights(len(real), _ROOT_TWO)[places]))),
            }
    return found


def _days_and_rest(series: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dates as whole days since 1970 and the nanoseconds within the day, in 64-bit integers, and where they are
    missing: the same numbers whatever unit pandas keeps the column in, and no overflow for a date far from now
    (9999-12-31 does not fit in nanoseconds since 1970). A time zone is read as UTC.
    """
    missing = series.isna().to_numpy()
    if getattr(series.dt, "tz", None) is not None:
        series = series.dt.tz_convert("UTC").dt.tz_localize(None)
    values = series.to_numpy()
    whole_days = values.astype("datetime64[D]")
    rest = (values - whole_days).astype("timedelta64[ns]").astype("<i8")
    days = whole_days.astype("<i8")
    days[missing] = 0
    rest[missing] = 0
    return days, rest, missing


def _iso(day: int, rest: int) -> str:
    """A date in ISO form, with its time when it has one (to the nanosecond, trailing zeros dropped)."""
    text = str(np.datetime64(int(day), "D"))
    if not rest:
        return text
    seconds, fraction = divmod(int(rest), 1_000_000_000)
    clock = f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
    return f"{text}T{clock}" + (f".{fraction:09d}".rstrip("0") if fraction else "")


def _as_held(series: pd.Series) -> pd.Series:
    """
    A column of objects as the values it holds: booleans, whole or real numbers, dates. Text, and a column mixing
    kinds, stays as it is (read as text). A typed column is returned as it is.
    """
    if series.dtype != object:
        return series
    held = pd.api.types.infer_dtype(series, skipna=True)
    try:
        if held == "boolean":
            return series.astype("boolean")
        if held == "integer":
            return series.astype("Int64")
        if held in ("floating", "mixed-integer-float", "decimal"):
            return pd.to_numeric(series, errors="raise").astype("float64")
        if held in ("date", "datetime", "datetime64"):
            return pd.to_datetime(series, errors="raise")
        if held in ("timedelta", "timedelta64"):
            return pd.to_timedelta(series, errors="raise")
    except TypeError, ValueError, OverflowError:
        pass  # values that do not convert after all (a number too large, a date out of range): read as text
    return series


def _text_profile(series: pd.Series) -> dict[str, Any]:
    """Text, categories and mixed values through their text: the lengths of every value in bytes (-1 for a missing
    one), then the bytes of them all. The lengths come first, so two columns cannot share their bytes by chance."""
    values = series.astype(object).tolist()
    texts = [
        None
        if v is None or v is pd.NA or v is pd.NaT or (isinstance(v, float) and math.isnan(v))
        else str(v)
        for v in values
    ]
    encoded = [None if t is None else t.encode("utf-8", "surrogatepass") for t in texts]
    lengths = np.fromiter(
        (-1 if e is None else len(e) for e in encoded), dtype="<i8", count=len(encoded)
    )
    digest = _digest(b"text", lengths.tobytes(), b"".join(e for e in encoded if e is not None))
    return {"digest": digest, "distinct": len({t for t in texts if t is not None})}


def column_profile(series: pd.Series) -> dict[str, Any]:
    """
    The profile of a column, its rows taken in the order given (:func:`frame_profile` puts them in the order of the
    row ids): ``kind``, ``rows``, ``missing``, ``digest`` (SHA-256 of its values), and by kind: ``whole``, ``sum``,
    ``sum_abs``, ``sum_sq``, ``min``, ``max``, ``mix``, ``mix2`` over its finite values and ``plus_inf``,
    ``minus_inf`` (number); ``min``, ``max`` in ISO form (date) or in
    nanoseconds (duration); ``true`` and ``false`` (boolean); ``distinct`` (text).
    """
    series = _as_held(series)
    profile: dict[str, Any] = {"rows": len(series), "missing": int(series.isna().sum())}
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        missing = series.isna().to_numpy()
        truth = series.fillna(False).to_numpy(dtype="u1", na_value=0)
        return {
            "kind": BOOLEAN,
            **profile,
            "digest": _digest(b"b1", truth.tobytes(), missing.astype("u1").tobytes()),
            "true": int(truth[~missing].sum()),
            "false": int((~missing).sum() - truth[~missing].sum()),
        }
    if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_complex_dtype(dtype):
        return {"kind": NUMBER, **profile, **_numeric_profile(series)}
    if pd.api.types.is_datetime64_any_dtype(dtype):
        days, rest, missing = _days_and_rest(series)
        ends = {}
        if (~missing).any():
            order = np.lexsort((rest[~missing], days[~missing]))
            first, last = order[0], order[-1]
            ends = {
                "min": _iso(days[~missing][first], rest[~missing][first]),
                "max": _iso(days[~missing][last], rest[~missing][last]),
            }
        return {
            "kind": DATE,
            **profile,
            "digest": _digest(
                b"date", days.tobytes(), rest.tobytes(), missing.astype("u1").tobytes()
            ),
            **ends,
        }
    if pd.api.types.is_timedelta64_dtype(dtype):
        missing = series.isna().to_numpy()
        whole = series.to_numpy().astype("timedelta64[ns]").astype("<i8")
        whole[missing] = 0
        present = whole[~missing]
        ends = {"min": int(present.min()), "max": int(present.max())} if len(present) else {}
        return {
            "kind": DURATION,
            **profile,
            "digest": _digest(b"td", whole.tobytes(), missing.astype("u1").tobytes()),
            **ends,
        }
    return {"kind": TEXT, **profile, **_text_profile(series)}


def frame_profile(df: pd.DataFrame) -> dict[str, Any]:
    """
    The profile of a table: its rows, the fingerprint of its row ids (``core.fingerprint.ids_fingerprint``; None
    when they are not whole numbers, and the rows are then taken in the order given), its columns' names in order,
    and each column's profile in the same order (by place: two columns can bear one name).
    """
    rows_fp: str | None = None
    if pd.api.types.is_integer_dtype(df.index.dtype):
        try:
            rows_fp = ids_fingerprint(df.index)
            if not df.index.is_monotonic_increasing:
                df = df.sort_index(kind="stable")  # once, for every column
        except TypeError, ValueError, OverflowError:  # ids beyond 64 bits
            rows_fp = None
    return {
        "format": FORMAT,
        "version": PROFILE_VERSION,
        "rows": len(df),
        "rows_fp": rows_fp,
        "names": [str(c) for c in df.columns],
        "columns": [column_profile(df.iloc[:, i]) for i in range(df.shape[1])],
    }


__all__ = [
    "BOOLEAN",
    "DATE",
    "DURATION",
    "FORMAT",
    "KINDS",
    "NUMBER",
    "PROFILE_VERSION",
    "TEXT",
    "column_profile",
    "frame_profile",
]
