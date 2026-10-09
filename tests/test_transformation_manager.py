"""
Unit tests for TransformationManager (Session Tracing, Data Lineage, and Serialization).
"""
import pytest
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Generator
from manage.transformation_manager import TransformationManager


@pytest.fixture
def isolated_transformation_manager(tmp_path: Path) -> TransformationManager:
    """Provides a TransformationManager instance writing to an isolated directory."""
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    return TransformationManager(trace_dir=str(trace_dir))


class TestTransformationManager:
    """Test suite for TransformationManager session lifecycle, serialization, and lineage."""

    def test_initialize_session(self, isolated_transformation_manager: TransformationManager) -> None:
        """Session initialization formats session ID and creates initial trace file."""
        isolated_transformation_manager.initialize_session("patient_cohort_2026.csv")
        
        assert isolated_transformation_manager.session_id is not None
        assert "patient_cohort_2026" in isolated_transformation_manager.session_id
        assert isolated_transformation_manager.source_dataset == "patient_cohort_2026.csv"
        assert len(isolated_transformation_manager.history) == 0

        # Trace file should have been written to disk
        trace_path = isolated_transformation_manager.get_trace_path()
        assert trace_path is not None
        assert Path(trace_path).exists()

    def test_initialize_session_with_user_isolation(self, isolated_transformation_manager: TransformationManager) -> None:
        """Non-admin user sessions are prefixed with their username."""
        isolated_transformation_manager.initialize_session("trial_alpha.csv", username="dr_smith")
        assert isolated_transformation_manager.session_id.startswith("dr_smith_session_")

        # Admin user session does not get prefixed
        isolated_transformation_manager.initialize_session("trial_alpha.csv", username="admin")
        assert not isolated_transformation_manager.session_id.startswith("admin_")
        assert isolated_transformation_manager.session_id.startswith("session_")

    def test_add_step_and_serialization(self, isolated_transformation_manager: TransformationManager) -> None:
        """Adding a transformation step converts complex parameters (numpy, pandas) safely."""
        isolated_transformation_manager.initialize_session("baseline.csv")

        # Complex params containing numpy scalars, dicts, lists, and a dataframe
        complex_params = {
            "threshold": np.float64(0.05),
            "iterations": np.int32(100),
            "features": ["age", "bmi"],
            "metadata": {"scale": 1.5},
            "dataframe_sample": pd.DataFrame({"a": [1, 2]}),
        }

        isolated_transformation_manager.add_step(
            function_name="filter_outliers",
            params=complex_params,
            description="Filtered BMI outliers beyond 3 SD",
            output_dataset_path="/data/cleaned_step1.parquet"
        )

        assert len(isolated_transformation_manager.history) == 1
        step = isolated_transformation_manager.history[0]
        assert step["function"] == "filter_outliers"
        assert step["description"] == "Filtered BMI outliers beyond 3 SD"
        assert step["output_dataset_path"] == "/data/cleaned_step1.parquet"

        # Verify serialization
        assert isinstance(step["params"]["threshold"], float)
        assert isinstance(step["params"]["iterations"], int)
        assert step["params"]["dataframe_sample"] == "DataFrame/Series (Not Serializable)"

    def test_save_and_load_trace(self, isolated_transformation_manager: TransformationManager) -> None:
        """Trace files can be saved and restored with full fidelity."""
        isolated_transformation_manager.initialize_session("study_x.csv")
        isolated_transformation_manager.add_step(
            function_name="impute_missing",
            params={"strategy": "median"},
            description="Imputed missing values",
            output_dataset_path="/data/imputed.parquet"
        )

        trace_path = isolated_transformation_manager.get_trace_path()
        session_id = isolated_transformation_manager.session_id

        # Create a new manager instance and load the trace
        new_manager = TransformationManager(trace_dir=isolated_transformation_manager.trace_dir)
        ok, msg = new_manager.load_trace(trace_path)

        assert ok is True
        assert new_manager.session_id == session_id
        assert new_manager.source_dataset == "study_x.csv"
        assert len(new_manager.history) == 1
        assert new_manager.history[0]["function"] == "impute_missing"

    def test_load_nonexistent_trace_fails_gracefully(self, isolated_transformation_manager: TransformationManager) -> None:
        """Loading a missing trace file returns (False, error) without raising an uncaught exception."""
        ok, msg = isolated_transformation_manager.load_trace("/nonexistent/path/trace.json")
        assert ok is False
        assert "not found" in msg.lower()

    def test_get_available_datasets_from_traces(self, isolated_transformation_manager: TransformationManager) -> None:
        """Scans traces directory and lists datasets with user filtering."""
        # Session 1: dr_who
        mgr1 = TransformationManager(trace_dir=isolated_transformation_manager.trace_dir)
        mgr1.initialize_session("cardio.csv", username="dr_who")
        mgr1.add_step(
            function_name="normalize",
            params={"method": "zscore"},
            description="Normalized features",
            output_dataset_path="/data/cardio_norm.parquet"
        )

        # Session 2: dr_watson
        mgr2 = TransformationManager(trace_dir=isolated_transformation_manager.trace_dir)
        mgr2.initialize_session("neuro.csv", username="dr_watson")
        mgr2.add_step(
            function_name="binning",
            params={"bins": 5},
            description="Binned age categories",
            output_dataset_path="/data/neuro_binned.parquet"
        )

        # dr_who sees only dr_who's datasets
        who_datasets = isolated_transformation_manager.get_available_datasets_from_traces(username="dr_who")
        assert len(who_datasets) == 1
        assert who_datasets[0]["file_path"] == "/data/cardio_norm.parquet"

        # dr_watson sees only dr_watson's datasets
        watson_datasets = isolated_transformation_manager.get_available_datasets_from_traces(username="dr_watson")
        assert len(watson_datasets) == 1
        assert watson_datasets[0]["file_path"] == "/data/neuro_binned.parquet"

        # admin sees both datasets
        admin_datasets = isolated_transformation_manager.get_available_datasets_from_traces(username="admin")
        assert len(admin_datasets) == 2
