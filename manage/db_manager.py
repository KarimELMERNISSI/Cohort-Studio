import duckdb
import glob
import re
import os
import logging
import pandas as pd

class DBManager:
    """Helper class to manage DuckDB connections and data persistence using Parquet with versioning"""
    def __init__(self, db_path="cohort_data.duckdb"):
        self.db_path = db_path
        # Don't keep a persistent connection - open/close as needed

    def _get_connection(self):
        """Get a new DuckDB connection."""
        try:
            return duckdb.connect(self.db_path)
        except Exception as e:
            logging.error(f"Failed to connect to DuckDB: {e}")
            return None

    def _get_next_version(self, base_name):
        """Determines the next version number for a given base name."""
        files = glob.glob(f"{base_name}_v*.parquet")
        if not files:
            return 1
        versions = []
        for f in files:
            try:
                # Extract version number assuming format name_vX.parquet
                match = re.search(rf"{base_name}_v(\d+)\.parquet", f)
                if match:
                    versions.append(int(match.group(1)))
            except:
                continue
        return max(versions) + 1 if versions else 1

    def save_dataframe(self, df, name="main_data"):
        """Persists a pandas DataFrame to a Parquet file via DuckDB (Low level)."""
        con = self._get_connection()
        if con is None:
            return False, "Database connection not available"
        try:
            file_path = f"{name}.parquet"
            con.execute(f"COPY (SELECT * FROM df) TO '{file_path}' (FORMAT PARQUET)")
            return True, f"Data saved to '{file_path}'"
        except Exception as e:
            return False, str(e)
        finally:
            con.close()

    def save_dataset(self, df, base_name="dataset"):
        """Persists a pandas DataFrame to a Parquet file with automatic versioning."""
        con = self._get_connection()
        if con is None:
            return False, "Database connection not available", None
        try:
            version = self._get_next_version(base_name)
            file_name = f"{base_name}_v{version}"
            file_path = f"{file_name}.parquet"
            
            # Use DuckDB to write parquet efficiently
            con.execute(f"COPY (SELECT * FROM df) TO '{file_path}' (FORMAT PARQUET)")
            return True, f"Data saved as '{file_name}'", file_name
        except Exception as e:
            return False, str(e), None
        finally:
            con.close()

    def load_dataframe(self, name="main_data"):
        """Loads a Parquet file into a pandas DataFrame via DuckDB (Low level)."""
        con = self._get_connection()
        if con is None:
            return None, "Database connection not available"
        try:
            file_path = f"{name}.parquet"
            df = con.execute(f"SELECT * FROM read_parquet('{file_path}')").df()
            return df, f"Data loaded from '{file_path}'"
        except Exception as e:
            return None, f"Failed to load '{name}': {str(e)}"
        finally:
            con.close()

    def load_dataset(self, file_name):
        """Loads a Parquet file into a pandas DataFrame via DuckDB."""
        return self.load_dataframe(file_name)
            
    def get_available_tables(self):
        """List available parquet files in the current directory."""
        files = glob.glob("*.parquet")
        return [f.replace(".parquet", "") for f in files if not f.startswith("stats_")]

    def get_available_datasets(self):
        """List available parquet files in the current directory, sorted by modification time."""
        files = glob.glob("*.parquet")
        # Filter out stats files
        dataset_files = [f for f in files if not f.startswith("stats_")]
        # Sort by modification time (newest first)
        dataset_files.sort(key=os.path.getmtime, reverse=True)
        return [f.replace(".parquet", "") for f in dataset_files]

    def save_stats(self, df, dataset_name, stats_type):
        """Save statistical results to parquet cache."""
        return self.save_dataframe(df, f"stats_{stats_type}_{dataset_name}")

    def load_stats(self, dataset_name, stats_type):
        """Load statistical results from parquet cache."""
        return self.load_dataframe(f"stats_{stats_type}_{dataset_name}")
