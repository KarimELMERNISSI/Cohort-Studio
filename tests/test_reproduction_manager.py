"""
Unit Tests for Reproduction Manager (manage/reproduction_manager.py).

Tests deterministic pipeline execution, progress callbacks, multi-step replay,
and error handling.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import app_pages.data_enrichment
from manage.reproduction_manager import execute_trace_pipeline


@pytest.fixture
def sample_cohort_csv(tmp_path: Path) -> Path:
    """Create a temporary source dataset CSV for pipeline execution."""
    df = pd.DataFrame({
        "patient_id": [101, 102, 103, 104, 105],
        "age": [45, 52, 61, 39, 70],
        "systolic_bp": [120, 140, 135, 110, 160],
        "cholesterol": [180.0, 220.0, np.nan, 195.0, 240.0],
    })
    csv_path = tmp_path / "baseline_cohort.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_enrichment_csv(tmp_path: Path) -> Path:
    """Create a secondary enrichment file for merging."""
    df = pd.DataFrame({
        "patient_id": [101, 102, 103, 104, 105],
        "smoking_status": ["Never", "Former", "Current", "Never", "Former"],
        "diabetes": [0, 1, 1, 0, 1],
    })
    csv_path = tmp_path / "lifestyle_enrichment.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


class TestReproductionPipelineExecution:
    """Test suite for execute_trace_pipeline."""

    def test_missing_source_dataset_field(self) -> None:
        """Trace missing 'source_dataset' returns error."""
        trace: dict[str, Any] = {"steps": []}
        df, err = execute_trace_pipeline(trace)
        assert df is None
        assert "missing 'source_dataset'" in err.lower()

    def test_nonexistent_source_file(self) -> None:
        """Trace referencing a nonexistent file returns error."""
        trace = {"source_dataset": "nonexistent_file_xyz_123.csv", "steps": []}
        df, err = execute_trace_pipeline(trace)
        assert df is None
        assert "not found" in err.lower()

    def test_execute_empty_steps_pipeline(self, sample_cohort_csv: Path) -> None:
        """Trace with no steps successfully returns copy of source dataset."""
        trace = {
            "source_dataset": str(sample_cohort_csv),
            "steps": [],
        }
        progress_calls = []

        def on_progress(pct: int, msg: str) -> None:
            progress_calls.append((pct, msg))

        df, err = execute_trace_pipeline(trace, progress_callback=on_progress)
        assert err is None
        assert df is not None
        assert df.shape == (5, 4)
        assert len(progress_calls) >= 1
        assert progress_calls[-1][0] == 100

    def test_execute_enrichment_step(
        self, sample_cohort_csv: Path, sample_enrichment_csv: Path
    ) -> None:
        """Enrichment step successfully merges additional columns into current dataframe."""
        trace = {
            "source_dataset": str(sample_cohort_csv),
            "steps": [
                {
                    "function": "enrichment",
                    "description": "Merge lifestyle data",
                    "params": {
                        "enrichment_file_path": str(sample_enrichment_csv),
                        "identifier": "patient_id",
                        "strategy": "inner",
                        "conflict_resolution": "keep_left",
                    },
                }
            ],
        }

        df, err = execute_trace_pipeline(trace)
        assert err is None
        assert df is not None
        assert "smoking_status" in df.columns
        assert "diabetes" in df.columns
        assert df.shape[0] == 5

    def test_execute_enrichment_missing_file_error(self, sample_cohort_csv: Path) -> None:
        """Enrichment step with invalid path fails gracefully."""
        trace = {
            "source_dataset": str(sample_cohort_csv),
            "steps": [
                {
                    "function": "enrichment",
                    "params": {
                        "enrichment_file_path": "nonexistent_enrichment.csv",
                        "identifier": "patient_id",
                    },
                }
            ],
        }

        df, err = execute_trace_pipeline(trace)
        assert df is None
        assert "could not find enrichment file" in err.lower()

    def test_unknown_step_skipped_safely(self, sample_cohort_csv: Path) -> None:
        """Unknown function in trace is skipped with warning, pipeline succeeds."""
        trace = {
            "source_dataset": str(sample_cohort_csv),
            "steps": [
                {
                    "function": "unsupported_custom_operator",
                    "description": "Experimental filter",
                    "params": {},
                }
            ],
        }

        df, err = execute_trace_pipeline(trace)
        assert err is None
        assert df is not None
        assert df.shape == (5, 4)

    def test_execute_variable_transformation_step(self, sample_cohort_csv: Path) -> None:
        """Variable transformation step applies calculation and appends result."""
        trace = {
            "source_dataset": str(sample_cohort_csv),
            "steps": [
                {
                    "function": "variable_transformation",
                    "description": "Log transform cholesterol",
                    "params": {
                        "column": "systolic_bp",
                        "operation": "log",
                        "new_column": "log_systolic",
                    },
                }
            ],
        }

        with patch("manage.reproduction_manager.apply_variable_transformation") as mock_transform:
            mock_transform.return_value = pd.DataFrame({
                "log_systolic": [4.78, 4.94, 4.90, 4.70, 5.07]
            })

            df, err = execute_trace_pipeline(trace)
            assert err is None
            assert df is not None
            assert "log_systolic" in df.columns
            assert mock_transform.called

    def test_step_exception_handling(self, sample_cohort_csv: Path) -> None:
        """Exceptions raised inside step execution are caught and returned as clean error."""
        trace = {
            "source_dataset": str(sample_cohort_csv),
            "steps": [
                {
                    "function": "imputation",
                    "params": {},
                }
            ],
        }

        with patch("app_pages.data_enrichment.apply_imputer", side_effect=ValueError("Corrupt imputation matrix")):
            df, err = execute_trace_pipeline(trace)
            assert df is None
            assert "error during step 1" in err.lower()
            assert "corrupt imputation matrix" in err.lower()
