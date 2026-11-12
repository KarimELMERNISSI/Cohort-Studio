"""
A pooled result names its imputation by what was drawn, and the imputation's step records what the draws depend on
(M8, the study package): the name is the fingerprint of the imputed cells that the trace's ``mi_impute`` step
records, the step holds the libraries, the machine and every BLAS library with its kernel, and a replay whose
imputation differs says which of them changed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from biostats.mi import ImputationSpec, impute
from biostats.mi.binding import bind
from biostats.mi.engine import drawn_with, drawn_with_changes
from manage.replay.engine import apply_steps


def _data(n: int = 120, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.normal(0, 1, n), "g": rng.choice(["a", "b"], n)})
    df["y"] = 2 * df["x"] + (df["g"] == "b") + rng.normal(0, 1, n)
    df.loc[rng.random(n) < 0.2, "y"] = np.nan
    df.loc[rng.random(n) < 0.1, "x"] = np.nan
    return df


SPEC = ImputationSpec(columns=["x", "g", "y"], m=3, maxit=2, seed=5)


def test_the_step_records_what_the_draws_depend_on_and_nothing_of_the_machine_s_own() -> None:
    found = drawn_with()
    assert set(found) == {"versions", "platform"}
    assert set(found["versions"]) == {
        "numpy",
        "scikit-learn",
        "scipy",
    }  # polyreg's solver is SciPy's
    assert {"system", "machine"} <= set(found["platform"]) <= {"system", "machine", "blas"}
    for pool in found["platform"].get("blas", []):
        assert set(pool) == {"library", "version", "kernel", "threading", "threads"}
    # never a path or a host name
    import socket

    text = repr(found)
    assert "\\" not in text and "/" not in text and ".dll" not in text and ".so" not in text
    assert socket.gethostname().lower() not in text.lower()
    assert drawn_with() == found  # the same process says the same


def test_what_changed_since_a_step_was_drawn_is_named() -> None:
    now = drawn_with()
    assert drawn_with_changes(now, now) == []
    older = {
        "versions": {**now["versions"], "scipy": "1.15.0"},
        # a system and a machine that no test runs on: an unchanged value is, rightly, not listed (recorded as
        # "Linux, aarch64", this test failed on a Linux runner, and would have on an ARM one)
        "platform": {
            "system": "SunOS",
            "machine": "sparc64",
            "blas": [
                {
                    "library": "openblas",
                    "version": "0.3.27",
                    "kernel": "neoversen1",
                    "threading": "pthreads",
                    "threads": 8,
                }
            ],
        },
    }
    changes = drawn_with_changes(older, now)
    assert changes[0] == f"scipy 1.15.0 -> {now['versions']['scipy']}"
    assert f"system SunOS -> {now['platform']['system']}" in changes
    assert f"machine sparc64 -> {now['platform']['machine']}" in changes
    assert any(
        c.startswith("BLAS openblas 0.3.27 neoversen1 (pthreads, 8 threads) -> ") for c in changes
    )
    # a change of threading layer alone is seen and printed; a process without threadpoolctl says so
    one = {"library": "openblas", "version": "0.3.27", "kernel": "haswell", "threads": 8}
    layers = [
        {**now, "platform": {**now["platform"], "blas": [{**one, "threading": layer}]}}
        for layer in ("pthreads", "openmp")
    ]
    assert drawn_with_changes(*layers) == [
        "BLAS openblas 0.3.27 haswell (pthreads, 8 threads) -> openblas 0.3.27 haswell (openmp, 8 threads)"
    ]
    bare = {**now, "platform": {k: v for k, v in now["platform"].items() if k != "blas"}}
    assert drawn_with_changes(layers[0], bare) == [
        "BLAS openblas 0.3.27 haswell (pthreads, 8 threads) -> not available"
    ]
    # a step written before these were recorded: said, since any of them may have changed
    before = {
        "versions": {
            "numpy": now["versions"]["numpy"],
            "scikit-learn": now["versions"]["scikit-learn"],
        }
    }
    assert drawn_with_changes(before, now) == [
        "scipy not recorded",
        "system not recorded",
        "machine not recorded",
        "BLAS not recorded",
    ]


def test_a_replay_whose_imputation_differs_says_what_changed() -> None:
    df = _data()
    drawn = drawn_with()

    def step(**params) -> dict:
        return {
            "function": "mi_impute",
            "params": {"spec": SPEC.model_dump(), "cells_fp": "0" * 16, **params},
            "description": "Multiple imputation",
        }

    same = apply_steps(df, [step(**drawn)])
    assert same.mi_divergence == [0]
    assert same.warnings == [
        "step 1 (mi_impute): the re-computed imputation differs from the recorded one"
    ]
    elsewhere = apply_steps(
        df,
        [
            step(
                versions={**drawn["versions"], "scipy": "1.15.0"},
                platform={**drawn["platform"], "machine": "sparc64"},
            )
        ],
    )
    (warning,) = elsewhere.warnings
    assert warning.endswith(
        f"(changed since: scipy 1.15.0 -> {drawn['versions']['scipy']}, "
        f"machine sparc64 -> {drawn['platform']['machine']})"
    )
    earlier = apply_steps(
        df, [step(versions={"numpy": "1.26.0", "scikit-learn": drawn["versions"]["scikit-learn"]})]
    )
    assert "numpy 1.26.0 -> " in earlier.warnings[0] and "scipy not recorded" in earlier.warnings[0]


def test_the_name_is_the_fingerprint_of_the_imputed_cells_kept_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from biostats.mi import MIData
    from ui.components.mi_mode import mi_token
    from ui.results import describe_settings

    df = _data()
    stored = bind(impute(df, SPEC))
    expected = stored.mi.fingerprint()
    assert stored.cells_fp == expected and len(expected) == 16
    calls: list[int] = []
    real = MIData.fingerprint
    monkeypatch.setattr(
        MIData, "fingerprint", lambda self, *a, **k: calls.append(1) or real(self, *a, **k)
    )
    assert stored.cells_fp == expected and calls == []  # not computed again

    def state(kind: str):
        return SimpleNamespace(mi=stored, mi_status=lambda: SimpleNamespace(kind=kind))

    name = mi_token(state("fresh"))
    assert name == f"mi:{stored.cells_fp}"
    assert mi_token(state("subset")) == f"mi:{stored.cells_fp}:subset"
    assert mi_token(SimpleNamespace(mi=None)) == ""
    # the same specification and seed drawn again: the same cells, the same name
    again = bind(impute(df, SPEC))
    assert again.binding.run_id != stored.binding.run_id and again.cells_fp == stored.cells_fp
    other = bind(impute(df, SPEC.model_copy(update={"seed": 6})))
    assert other.cells_fp != stored.cells_fp
    # the dossier's words of the settings never show the name, in this form or in the earlier one
    assert (
        describe_settings(("age", "sex", name, name + ":subset", f"{'d' * 32}:2026-11-10"))
        == "age, sex"
    )
