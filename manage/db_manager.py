import glob
import logging
import os
import re
from typing import Any

import duckdb
import pandas as pd

from core.workspace import UserScope
from utils.data_paths import get_datasets_dir, get_stats_dir


def _sql_literal(value: str) -> str:
    """Escape a string for use inside a single-quoted DuckDB literal (COPY ... TO does not bind parameters)."""
    return value.replace("'", "''")


logger = logging.getLogger(__name__)


def write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write a dataset version: pyarrow keeps the dtypes and the row ids (the ``__row_id`` index); DuckDB
    is the fallback for the rare column pyarrow refuses (the ids then travel as a ``__row_id`` column)."""
    frame = df.copy(deep=False)
    frame.columns = [str(c) for c in frame.columns]
    try:
        frame.to_parquet(path, index=frame.index.name is not None)
        return
    except Exception as exc:
        logger.warning("pyarrow could not write %s (%s); writing it with DuckDB", os.path.basename(path), exc)
    flat = frame.reset_index() if frame.index.name else frame
    con = duckdb.connect()
    try:
        con.register("flat_frame", flat)
        con.execute(f"COPY (SELECT * FROM flat_frame) TO '{_sql_literal(path)}' (FORMAT PARQUET)")
    finally:
        con.close()


class DBManager:
    """Helper class to manage DuckDB connections and data persistence using Parquet with versioning"""
    def __init__(self, db_path: str = ":memory:", scope: UserScope | None = None) -> None:
        # Per-user workspace (audit E4): datasets and statistics live under <DATA_ROOT>/users/<username>/. The
        # session's services pass its user's scope (manage.services); without one (tests, scripts, the command line)
        # the shared directories are used.
        self.scope: UserScope | None = scope
        shared_dir = get_datasets_dir()
        if self.scope is not None:
            self.dataset_dir: str = str(self.scope.datasets_dir())
            self.stats_dir: str = str(self.scope.stats_dir())
        else:
            self.dataset_dir = shared_dir
            self.stats_dir = get_stats_dir()
        # The DuckDB catalog is shared by design (the accounts are manage.iam's).
        self.db_path: str = os.path.join(shared_dir, "cohort_data.duckdb")

        # Don't keep a persistent connection - open/close as needed

    def _get_connection(self):
        """Get a new DuckDB connection."""
        try:
            return duckdb.connect(self.db_path)
        except Exception as e:
            logging.error(f"Failed to connect to DuckDB: {e}")
            return None

    def _get_next_version(self, base_name: str) -> int:
        """Determines the next version number for a given base name."""
        files = glob.glob(os.path.join(self.dataset_dir, f"{base_name}_v*.parquet"))
        if not files:
            return 1
        versions = []
        for f in files:
            try:
                # Extract version number assuming format name_vX.parquet
                match = re.search(rf"{base_name}_v(\d+)\.parquet", f)
                if match:
                    versions.append(int(match.group(1)))
            except Exception:
                continue
        return max(versions) + 1 if versions else 1

    def save_dataframe(self, df: pd.DataFrame, name: str = "main_data", folder: str | None = None) -> tuple[bool, str]:
        """Persists a pandas DataFrame to a Parquet file via DuckDB (Low level)."""
        con = self._get_connection()
        if con is None:
            return False, "Database connection not available"
        try:
            target_folder = folder if folder else self.dataset_dir
            file_path = os.path.join(target_folder, f"{name}.parquet")
            con.execute(f"COPY (SELECT * FROM df) TO '{_sql_literal(file_path)}' (FORMAT PARQUET)")
            return True, f"Data saved to '{file_path}'"
        except Exception as e:
            return False, str(e)
        finally:
            con.close()

    def save_dataset(self, df: pd.DataFrame, base_name: str = "dataset") -> tuple[bool, str, str | None]:
        """Persists a pandas DataFrame to a versioned Parquet file (row ids and dtypes are kept), in the workspace."""
        try:
            version = self._get_next_version(base_name)
            file_name = f"{base_name}_v{version}"
            file_path = os.path.join(self.dataset_dir, f"{file_name}.parquet")
            write_parquet(df, file_path)
            return True, f"Data saved as '{file_name}'", file_name
        except Exception as e:
            return False, str(e), None

    def load_dataframe(self, name: str = "main_data", folder: str | None = None) -> tuple[pd.DataFrame | None, str]:
        """Loads a Parquet file into a pandas DataFrame via DuckDB (Low level)."""
        con = self._get_connection()
        if con is None:
            return None, "Database connection not available"
        try:
            target_folder = folder if folder else self.dataset_dir
            file_path = os.path.join(target_folder, f"{name}.parquet")
            df = con.execute("SELECT * FROM read_parquet(?)", [file_path]).df()
            return df, f"Data loaded from '{file_path}'"
        except Exception as e:
            return None, f"Failed to load '{name}': {e!s}"
        finally:
            con.close()

    def load_dataset(self, file_name: str) -> tuple[pd.DataFrame | None, str]:
        """
        Loads a Parquet file into a pandas DataFrame via DuckDB.
        
        Parameters:
        -----------
        file_name : str
            The name of the file to load (without extension)
            
        Returns:
        --------
        tuple
            (DataFrame, message) - The loaded DataFrame and a success/error message
        """
        path = os.path.join(self.dataset_dir, f"{file_name}.parquet")
        try:
            return pd.read_parquet(path), f"Data loaded from '{path}'"  # row ids come back as the index
        except FileNotFoundError:
            return None, f"Failed to load '{file_name}': file not found"
        except Exception:
            return self.load_dataframe(file_name)  # files pandas cannot read (DuckDB fallback)
            
    def lineage_path(self, file_name: str) -> str:
        return os.path.join(self.dataset_dir, f"{file_name}.lineage.json")

    def save_lineage(self, file_name: str, lineage: dict[str, Any]) -> None:
        """The lineage sidecar of a saved version (where it comes from: trace, fingerprint, population)."""
        from core.lineage import canonical_json

        path = self.lineage_path(file_name)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(canonical_json(lineage))
        os.replace(tmp, path)

    def load_lineage(self, file_name: str) -> dict[str, Any] | None:
        import json

        path = self.lineage_path(file_name)
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            logger.warning("Unreadable lineage sidecar: %s", os.path.basename(path))
            return None

    def get_available_datasets(self) -> list[str]:
        """The saved versions of the workspace's datasets, the newest first."""
        files = glob.glob(os.path.join(self.dataset_dir, "*.parquet"))
        # Filter out stats files, temp files, and User DB
        dataset_files = [f for f in files if not os.path.basename(f).startswith("stats_") and not os.path.basename(f).startswith("tmp_") and not os.path.basename(f) == "users.parquet" and not os.path.basename(f) == "users.duckdb"]
        
        # Sort by modification time (newest first)
        dataset_files.sort(key=os.path.getmtime, reverse=True)
        
        return [os.path.basename(f).replace(".parquet", "") for f in dataset_files]

    def save_stats(self, df: pd.DataFrame, dataset_name: str, stats_type: str) -> tuple[bool, str]:
        """Save statistical results to parquet cache."""
        return self.save_dataframe(df, f"stats_{stats_type}_{dataset_name}", folder=self.stats_dir)

    def load_stats(self, dataset_name, stats_type):
        """Load statistical results from parquet cache."""
        return self.load_dataframe(f"stats_{stats_type}_{dataset_name}", folder=self.stats_dir)

    def clear_stats(self, dataset_name):
        """Delete cached statistics for a given dataset to force recalculation."""
        try:
            # Stats are saved as stats_numerical_{dataset_name} and stats_categorical_{dataset_name}
            for stats_type in ["numerical", "categorical"]:
                file_path = os.path.join(self.stats_dir, f"stats_{stats_type}_{dataset_name}.parquet")
                if os.path.exists(file_path):
                    os.remove(file_path)
            return True, "Statistics cache cleared."
        except Exception as e:
            return False, f"Failed to clear stats: {e}"

    def clear_all_stats(self):
        """Delete ALL cached statistics files."""
        try:
            files = glob.glob(os.path.join(self.stats_dir, "*.parquet"))
            count = 0
            for f in files:
                try:
                    os.remove(f)
                    count += 1
                except OSError:
                    pass  # a file in use or already gone is left out of the count
            return True, f"Cleared {count} cached statistics files."
        except Exception as e:
            return False, f"Failed to clean cache: {e}"
