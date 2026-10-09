"""
Unit tests for DBManager (User Authentication and DuckDB Dataset Persistence).
"""
from collections.abc import Generator
from pathlib import Path

import pandas as pd
import pytest

import utils.data_paths as dp
from manage.db_manager import DBManager


@pytest.fixture
def isolated_db_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[DBManager]:
    """Provides a DBManager instance configured within an isolated temporary directory."""
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    manager = DBManager()
    yield manager


class TestDBManagerUserAuth:
    """Test suite for user credential management and activation workflow."""

    def test_create_and_verify_admin_user(self, isolated_db_manager: DBManager) -> None:
        """Admin user is automatically activated upon creation."""
        ok, msg = isolated_db_manager.create_user("admin", "admin_secret123")
        assert ok is True
        assert "activated" in msg.lower()

        # Verify correct credentials
        valid, auth_msg = isolated_db_manager.verify_user("admin", "admin_secret123")
        assert valid is True
        assert "successful" in auth_msg.lower()

        # Verify invalid credentials
        invalid, err_msg = isolated_db_manager.verify_user("admin", "wrong_password")
        assert invalid is False
        assert "invalid" in err_msg.lower()

    def test_create_standard_user_pending_activation(self, isolated_db_manager: DBManager) -> None:
        """Standard user is created inactive by default and requires activation."""
        ok, msg = isolated_db_manager.create_user("analyst_jane", "JanePass2026!")
        assert ok is True
        assert "approval" in msg.lower()

        # Login fails before activation
        valid_before, msg_before = isolated_db_manager.verify_user("analyst_jane", "JanePass2026!")
        assert valid_before is False
        assert "pending" in msg_before.lower()

        # Activate user
        act_ok, act_msg = isolated_db_manager.activate_user("analyst_jane")
        assert act_ok is True

        # Login succeeds after activation
        valid_after, msg_after = isolated_db_manager.verify_user("analyst_jane", "JanePass2026!")
        assert valid_after is True
        assert "successful" in msg_after.lower()

    def test_prevent_duplicate_username(self, isolated_db_manager: DBManager) -> None:
        """Registering the same username twice must fail."""
        ok1, _ = isolated_db_manager.create_user("researcher_bob", "pass1")
        assert ok1 is True

        ok2, err = isolated_db_manager.create_user("researcher_bob", "pass2")
        assert ok2 is False
        assert "already exists" in err.lower()

    def test_user_activation_deactivation_cycle(self, isolated_db_manager: DBManager) -> None:
        """Admins can activate and deactivate user accounts."""
        isolated_db_manager.create_user("analyst_sam", "PassWord999!", auto_activate=True)
        
        # Deactivate
        deact_ok, _ = isolated_db_manager.deactivate_user("analyst_sam")
        assert deact_ok is True

        valid, _ = isolated_db_manager.verify_user("analyst_sam", "PassWord999!")
        assert valid is False

        # Reactivate
        react_ok, _ = isolated_db_manager.activate_user("analyst_sam")
        assert react_ok is True

        valid_again, _ = isolated_db_manager.verify_user("analyst_sam", "PassWord999!")
        assert valid_again is True

    def test_admin_cannot_be_deactivated_or_deleted(self, isolated_db_manager: DBManager) -> None:
        """Safety checks preventing accidental admin lockout."""
        isolated_db_manager.create_user("admin", "pass")
        
        deact_ok, deact_msg = isolated_db_manager.deactivate_user("admin")
        assert deact_ok is False
        assert "cannot deactivate admin" in deact_msg.lower()

        del_ok, del_msg = isolated_db_manager.delete_user("admin")
        assert del_ok is False
        assert "cannot delete admin" in del_msg.lower()

    def test_update_password(self, isolated_db_manager: DBManager) -> None:
        """Updating user password securely hashes and applies the new secret."""
        isolated_db_manager.create_user("clara", "old_pass", auto_activate=True)
        
        upd_ok, _ = isolated_db_manager.update_user_password("clara", "new_secret_pass")
        assert upd_ok is True

        # Old password rejected
        old_valid, _ = isolated_db_manager.verify_user("clara", "old_pass")
        assert old_valid is False

        # New password accepted
        new_valid, _ = isolated_db_manager.verify_user("clara", "new_secret_pass")
        assert new_valid is True

    def test_get_all_users_and_status(self, isolated_db_manager: DBManager) -> None:
        """Status overview lists users with their activation state."""
        isolated_db_manager.create_user("admin", "p1")
        isolated_db_manager.create_user("user_a", "p2", auto_activate=False)
        isolated_db_manager.create_user("user_b", "p3", auto_activate=True)

        users = isolated_db_manager.get_all_users()
        assert "admin" in users
        assert "user_a" in users
        assert "user_b" in users

        statuses = isolated_db_manager.get_all_users_with_status()
        status_map = {u["username"]: u["status"] for u in statuses}
        assert status_map["admin"] == "Active"
        assert status_map["user_a"] == "Pending Approval"
        assert status_map["user_b"] == "Active"


class TestDBManagerDatasetPersistence:
    """Test suite for Parquet dataset versioning and DuckDB persistence."""

    def test_save_and_load_dataframe(self, isolated_db_manager: DBManager) -> None:
        """Saves and reloads a pandas DataFrame via DuckDB parquet copy."""
        test_df = pd.DataFrame({
            "patient_id": [101, 102, 103],
            "measurement": [12.5, 14.8, 11.2],
        })

        ok, msg = isolated_db_manager.save_dataframe(test_df, "test_cohort")
        assert ok is True
        assert "test_cohort.parquet" in msg

        loaded_df, load_msg = isolated_db_manager.load_dataframe("test_cohort")
        assert loaded_df is not None
        assert len(loaded_df) == 3
        assert list(loaded_df["patient_id"]) == [101, 102, 103]

    def test_save_dataset_versioning(self, isolated_db_manager: DBManager) -> None:
        """Sequential saves increment the version number automatically."""
        df1 = pd.DataFrame({"col": [1, 2]})
        df2 = pd.DataFrame({"col": [1, 2, 3]})

        ok1, _, v1_name = isolated_db_manager.save_dataset(df1, "clinical_study")
        assert ok1 is True
        assert v1_name == "clinical_study_v1"

        ok2, _, v2_name = isolated_db_manager.save_dataset(df2, "clinical_study")
        assert ok2 is True
        assert v2_name == "clinical_study_v2"

        available = isolated_db_manager.get_available_datasets()
        assert "clinical_study_v1" in available
        assert "clinical_study_v2" in available

    def test_user_isolated_datasets(self, isolated_db_manager: DBManager) -> None:
        """Datasets saved with a username are prefixed and filtered appropriately."""
        df = pd.DataFrame({"val": [42]})
        ok, _, fname = isolated_db_manager.save_dataset(df, "my_private_data", username="doctor_who")
        assert ok is True
        assert fname == "doctor_who_my_private_data_v1"

        # doctor_who sees their dataset
        doc_datasets = isolated_db_manager.get_available_datasets(username="doctor_who")
        assert "doctor_who_my_private_data_v1" in doc_datasets

        # another user does not see doctor_who's dataset
        other_datasets = isolated_db_manager.get_available_datasets(username="doctor_strange")
        assert "doctor_who_my_private_data_v1" not in other_datasets

        # admin sees all datasets
        admin_datasets = isolated_db_manager.get_available_datasets(username="admin")
        assert "doctor_who_my_private_data_v1" in admin_datasets

    def test_stats_caching_and_clearing(self, isolated_db_manager: DBManager) -> None:
        """Statistical summaries can be saved, loaded, and cleared from cache."""
        stats_df = pd.DataFrame({"mean": [50.0], "std": [10.0]})
        save_ok, _ = isolated_db_manager.save_stats(stats_df, "test_set", "numerical")
        assert save_ok is True

        loaded_stats, _ = isolated_db_manager.load_stats("test_set", "numerical")
        assert loaded_stats is not None
        assert loaded_stats["mean"].iloc[0] == 50.0

        # Clear stats for this dataset
        clear_ok, _ = isolated_db_manager.clear_stats("test_set")
        assert clear_ok is True

        loaded_again, _ = isolated_db_manager.load_stats("test_set", "numerical")
        assert loaded_again is None
