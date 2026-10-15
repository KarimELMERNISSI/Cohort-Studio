"""Replay of a multiple imputation (mi_impute steps): re-computed and checked against the recorded cells."""

from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn

from biostats.mi import ImputationSpec, impute
from manage.reproduction_manager import apply_steps


def _data(n: int = 200, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.normal(size=n), "g": rng.choice(["a", "b"], n).astype(object)})
    df["y"] = df["x"] + rng.normal(size=n)
    df.loc[rng.random(n) < 0.2, "x"] = np.nan
    df.loc[rng.random(n) < 0.1, "g"] = np.nan
    return df


def _step(df: pd.DataFrame, spec: ImputationSpec, cells_fp: str | None = None) -> dict:
    run = impute(df, spec)
    return {
        "function": "mi_impute",
        "params": {
            "spec": spec.model_dump(),
            "cells_fp": cells_fp or run.mi.fingerprint(),
            "versions": {"numpy": np.__version__, "scikit-learn": sklearn.__version__},
        },
        "description": "Multiple imputation",
    }


def test_fingerprint_follows_the_imputed_cells() -> None:
    df = _data()
    spec = ImputationSpec(columns=["x", "g", "y"], m=3, maxit=2, seed=5)
    first, again = impute(df, spec).mi, impute(df, spec).mi
    assert first.fingerprint() == again.fingerprint()
    other = impute(df, spec.model_copy(update={"seed": 6})).mi
    assert other.fingerprint() != first.fingerprint()


def test_replay_recomputes_the_imputation_and_checks_its_cells() -> None:
    df = _data()
    spec = ImputationSpec(columns=["x", "g", "y"], m=3, maxit=2, seed=5)
    result = apply_steps(df, [_step(df, spec)])
    assert result.error is None and result.reproducible and not result.mi_divergence
    assert result.imputations[0].m == 3 and result.df is not None and result.df.equals(df)
    tampered = apply_steps(df, [_step(df, spec, cells_fp="0" * 16)])
    assert tampered.mi_divergence == [0] and not tampered.reproducible
    assert any("re-computed imputation differs" in w for w in tampered.warnings)
