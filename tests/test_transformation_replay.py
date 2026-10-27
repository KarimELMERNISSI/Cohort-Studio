"""
Transformations: the Enrich page and the replay share one engine, so a recorded transformation replays to the same
table (M5). Before, the page had its own copy: a transformation applied twice got a renamed column on the page but
a duplicate column on replay, and PCA was not seeded, so its replay differed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from enrich.transformation_logic import apply_variable_transformation
from utils.optional import method_available

STATS = "{initial_variable}_{method_applied}"
CONFIGS = {
    "mean": {"transformation_type": "Statistics", "transformation": "Mean", "columns": ["age"]},
    "z_score": {
        "transformation_type": "Scaling",
        "transformation": "Z-Score Standardization",
        "columns": ["age", "sbp"],
    },
    "min_max": {"transformation_type": "Scaling", "transformation": "Min-Max Normalization", "columns": ["sbp"]},
    "log": {
        "transformation_type": "Scaling",
        "transformation": "Natural Logarithm Transformation",
        "columns": ["weight"],
    },
    "pca": {
        "transformation_type": "Dimensionality Reduction",
        "transformation": "PCA",
        "columns": ["age", "weight", "sbp"],
        "naming_pattern": "Name_{method_applied}_Component",
        "n_components": 2,
    },
    "famd": {
        "transformation_type": "Dimensionality Reduction",
        "transformation": "FAMD",
        "columns": ["age_years", "weight", "arm", "nyha"],
        "naming_pattern": "Name_{method_applied}_Component",
        "n_components": 2,
    },
    "k_means": {
        "transformation_type": "Clustering",
        "transformation": "K-Means",
        "columns": ["age", "weight"],
        "naming_pattern": "Cluster_{method_applied}",
        "n_clusters": 3,
    },
    "one_hot": {
        "transformation_type": "Encoding",
        "transformation": "One-Hot Encoding",
        "columns": ["arm"],
        "naming_pattern": "{initial_variable}_{category}",
    },
    "ordinal": {
        "transformation_type": "Encoding",
        "transformation": "Ordinal Encoding",
        "columns": ["nyha"],
        "naming_pattern": "{initial_variable}_{category}",
        "category_orders": {"nyha": ["I", "II", "III"]},
    },
}


def _params(name: str) -> dict:
    return {"naming_pattern": STATS, **CONFIGS[name]}


def _cohort(n: int = 40, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.normal(60, 10, n).round(1),
            "age_years": rng.integers(40, 80, n),  # an integer column: FAMD takes it as a number
            "weight": rng.normal(75, 12, n).round(1),
            "sbp": rng.normal(130, 15, n).round(0),
            "arm": rng.choice(["control", "drug"], n),
            "nyha": rng.choice(["I", "II", "III"], n),
        }
    )


def _record(tmp_path: Path, steps: list[dict]):
    """A session that applies ``steps`` as the Enrich page does: engine, draft, apply."""
    from core.session import DataSession
    from core.state import ParseOptions
    from ingest.loaders import read_path
    from manage.transformation_manager import TransformationManager

    src = tmp_path / "cohort.csv"
    _cohort().to_csv(src, index=False)
    tm = TransformationManager(trace_dir=str(tmp_path / "traces"))
    tm.initialize_session("cohort.csv")
    tm.source_dataset = str(src)
    session = DataSession(trace=tm)
    options = ParseOptions()
    session.install(
        read_path(str(src), options).df,
        params={"source_path": str(src), "parse_options": options.to_dict()},
    )
    for params in steps:
        working = session.working_frame()
        new_columns = apply_variable_transformation(working, params)
        assert new_columns is not None
        session.stage(
            pd.concat([working, new_columns], axis=1),
            "variable_transformation",
            params,
            f"Applied {params['transformation']}",
        )
        session.commit_staging()
    return tm, session


def _replay(tm):
    from manage.paths import PathResolver
    from manage.reproduction_manager import run_trace

    # as written to and read from the trace file
    return run_trace(json.loads(json.dumps(tm.get_trace())), paths=PathResolver.unconfined())


@pytest.mark.parametrize("name", list(CONFIGS))
def test_a_recorded_transformation_replays_to_the_same_table(tmp_path: Path, name: str) -> None:
    from core.fingerprint import frame_fingerprint

    if not method_available(CONFIGS[name]["transformation"]):  # PCA runs with prince (the 'ml' extra)
        pytest.skip(f"{CONFIGS[name]['transformation']} needs an optional library that is not installed")
    tm, session = _record(tmp_path, [_params(name)])
    result = _replay(tm)
    assert result.error is None and result.reproducible, result.warnings
    assert frame_fingerprint(result.df) == session.data_fp


def test_a_transformation_applied_twice_replays_with_the_same_names(tmp_path: Path) -> None:
    from core.fingerprint import frame_fingerprint

    z_score = _params("z_score")
    tm, session = _record(tmp_path, [z_score, z_score])
    created = [c for c in session.base.columns if "Z-Score" in c]
    assert created == [
        "age_Z-Score_Standardization",
        "sbp_Z-Score_Standardization",
        "age_Z-Score_Standardization_1",
        "sbp_Z-Score_Standardization_1",
    ]
    result = _replay(tm)
    assert result.error is None and result.reproducible, result.warnings
    assert frame_fingerprint(result.df) == session.data_fp


def test_reduction_and_clustering_need_complete_columns() -> None:
    df = _cohort()
    df.loc[3, "age"] = np.nan
    for name in ("pca", "k_means"):
        with pytest.raises(ValueError, match="missing values"):
            apply_variable_transformation(df, _params(name))
    with pytest.raises(ValueError, match="at least two columns"):
        apply_variable_transformation(_cohort(), {**_params("pca"), "columns": ["age"]})


def test_famd_takes_an_integer_column_as_a_number_and_refuses_a_date() -> None:
    """prince alone analysed the age in years as one category per value (M5)."""
    pytest.importorskip("prince", reason="FAMD runs with prince (the 'ml' extra)")
    from biostats.reduction import fit_famd

    df = _cohort()
    columns = CONFIGS["famd"]["columns"]
    new = apply_variable_transformation(df, _params("famd"))
    expected = fit_famd(df[columns], 2).transform(df[columns])
    np.testing.assert_allclose(pd.DataFrame(new).to_numpy(), expected.to_numpy())
    dated = df.assign(visit=pd.date_range("2024-01-01", periods=len(df), freq="D"))
    with pytest.raises(ValueError, match="a date"):
        apply_variable_transformation(dated, {**_params("famd"), "columns": ["age", "arm", "visit"]})
