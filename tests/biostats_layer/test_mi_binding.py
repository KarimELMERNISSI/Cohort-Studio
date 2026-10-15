"""Binding an imputation to the rows it was fitted on, its statuses and its files (M2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biostats.mi import ImputationSpec, impute
from biostats.mi.binding import bind, restrict, status
from manage.mi_store import delete_mi, has_mi, load_mi, mi_dir, save_mi


def _data(n: int = 240, seed: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "arm": rng.choice(["a", "b"], n),
            "age": rng.normal(60, 8, n),
            "smoker": rng.choice(["no", "yes"], n).astype(object),
        }
    )
    df["sbp"] = 100 + 0.5 * df["age"] + rng.normal(0, 8, n)
    df.loc[rng.random(n) < 0.15, "age"] = np.nan
    df.loc[rng.random(n) < 0.10, "smoker"] = np.nan
    df.index = pd.RangeIndex(1000, 1000 + n, name="__row_id")
    return df


def _stored(df: pd.DataFrame, **spec):
    run = impute(df, ImputationSpec(columns=["age", "smoker", "sbp"], m=3, maxit=3, **spec))
    return bind(run, data_fp="fp-data", population_fp="fp-pop", population={"label": "all rows"})


def test_statuses_follow_rows_and_observed_values() -> None:
    df = _data()
    stored = _stored(df)
    assert (
        stored.binding.columns == ["age", "smoker", "sbp"]
        and stored.binding.methods["age"] == "pmm"
    )
    fresh = status(stored, df)
    assert fresh.kind == "fresh" and fresh.usable and fresh.n_rows == len(df)
    unrelated = df.assign(bmi=27.0)  # a new derived variable does not touch the model columns
    assert status(stored, unrelated).kind == "fresh"
    part = status(stored, df.iloc[:100])
    assert part.kind == "subset" and not part.usable and "100 of the 240" in part.reason
    changed = df.copy()
    changed.loc[changed.index[3], "sbp"] += 1
    assert (
        status(stored, changed).kind == "stale"
        and "changed in 1 row" in status(stored, changed).reason
    )
    grown = pd.concat([df, df.iloc[:2].set_axis([5000, 5001]).rename_axis("__row_id")])
    assert status(stored, grown).kind == "stale" and "2 row(s)" in status(stored, grown).reason
    assert status(stored, df.drop(columns="smoker")).kind == "stale"


def test_whole_strata_are_an_exact_subset() -> None:
    df = _data()
    stored = _stored(df, by="arm")
    only_a = df[df["arm"] == "a"]
    by_subset = status(stored, only_a)
    assert by_subset.kind == "by_subset" and by_subset.usable
    assert status(stored, only_a.iloc[:-1]).kind == "subset"
    narrowed = restrict(stored, only_a.index)
    assert narrowed.base.index.equals(only_a.index)
    assert narrowed.imputations["age"].index.isin(only_a.index).all()
    with pytest.raises(ValueError, match="not in the imputed data"):
        stored.mi.subset([1, 2])


def test_flong_layout_for_stata() -> None:
    stored = _stored(_data())
    flong = stored.mi.flong()
    assert list(flong.columns[:2]) == ["_mi_m", "_mi_id"] and set(flong["_mi_m"]) == {0, 1, 2, 3}


def test_saved_imputation_reloads_identically_and_stays_fresh(tmp_path: Path) -> None:
    df = _data()
    stored = _stored(df, by="arm")
    written = save_mi(stored, tmp_path, "cohort_v2")
    assert all(p.exists() for p in written) and has_mi(tmp_path, "cohort_v2")
    assert all(p.parent == mi_dir(tmp_path) for p in written)  # kept out of the dataset list
    df.to_parquet(tmp_path / "cohort_v2.parquet")
    base = pd.read_parquet(tmp_path / "cohort_v2.parquet")
    loaded = load_mi(tmp_path, "cohort_v2", base)
    assert loaded is not None
    for col in ["age", "smoker"]:
        expected = stored.mi.imputations[col]
        got = loaded.mi.imputations[col]
        assert got.index.equals(expected.index)
        assert (got.astype(str).to_numpy() == expected.astype(str).to_numpy()).all()
    assert loaded.binding == stored.binding and loaded.mi.m == 3
    assert status(loaded, base).kind == "fresh"
    assert status(loaded, base[base["arm"] == "b"]).kind == "by_subset"
    assert loaded.diagnostics["rhat"].keys() == stored.diagnostics["rhat"].keys()
    delete_mi(tmp_path, "cohort_v2")
    assert not has_mi(tmp_path, "cohort_v2") and load_mi(tmp_path, "cohort_v2", base) is None
