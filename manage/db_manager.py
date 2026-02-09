import duckdb
import glob
import re
import os
import logging
import bcrypt

class DBManager:
    """Helper class to manage DuckDB connections and data persistence using Parquet with versioning"""
    def __init__(self, db_path=":memory:"):
        self.dataset_dir = os.path.join("data", "datasets")
        self.stats_dir = os.path.join("data", "stats")
        self.db_path = os.path.join(self.dataset_dir, "cohort_data.duckdb")
        self.user_db_path = os.path.join(self.dataset_dir, "users.duckdb")
        
        # Ensure directories exist
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)
        
        # Initialize User DB
        self.init_user_db()
        
        # Don't keep a persistent connection - open/close as needed

    def _get_user_connection(self):
        """Get a new DuckDB connection for users DB."""
        try:
            return duckdb.connect(self.user_db_path)
        except Exception as e:
            logging.error(f"Failed to connect to User DuckDB: {e}")
            return None

    def init_user_db(self):
        """Initialize the users table."""
        con = self._get_user_connection()
        if con:
            try:
                con.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        username TEXT PRIMARY KEY,
                        password_hash TEXT
                    )
                """)
            except Exception as e:
                logging.error(f"Failed to init user DB: {e}")
            finally:
                con.close()

    def create_user(self, username, password):
        """Create a new user."""
        con = self._get_user_connection()
        if con:
            try:
                # Check if user exists
                res = con.execute("SELECT 1 FROM users WHERE username = ?", [username]).fetchone()
                if res:
                    return False, "Username already exists"
                
                # Hash password
                hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                
                con.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", [username, hashed])
                return True, "User created successfully"
            except Exception as e:
                return False, f"Error creating user: {e}"
            finally:
                con.close()
        return False, "Database connection failed"

    def verify_user(self, username, password):
        """Verify user credentials."""
        con = self._get_user_connection()
        if con:
            try:
                res = con.execute("SELECT password_hash FROM users WHERE username = ?", [username]).fetchone()
                if res:
                    stored_hash = res[0]
                    if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
                        return True, "Login successful"
                return False, "Invalid username or password"
            except Exception as e:
                return False, f"Error verifying user: {e}"
            finally:
                con.close()
        return False, "Database connection failed"

    def get_all_users(self):
        """Get list of all registered usernames."""
        con = self._get_user_connection()
        if con:
            try:
                res = con.execute("SELECT username FROM users ORDER BY username").fetchall()
                return [row[0] for row in res]
            except Exception as e:
                logging.error(f"Error fetching users: {e}")
                return []
            finally:
                con.close()
        return []

    def update_user_password(self, username, new_password):
        """Update a user's password."""
        con = self._get_user_connection()
        if con:
            try:
                # Check if user exists
                res = con.execute("SELECT 1 FROM users WHERE username = ?", [username]).fetchone()
                if not res:
                    return False, "User not found"
                
                # Hash new password
                hashed = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                con.execute("UPDATE users SET password_hash = ? WHERE username = ?", [hashed, username])
                return True, f"Password updated for '{username}'"
            except Exception as e:
                return False, f"Error updating password: {e}"
            finally:
                con.close()
        return False, "Database connection failed"

    def delete_user(self, username):
        """Delete a user account. Protects admin from deletion."""
        if username == 'admin':
            return False, "Cannot delete admin account"
        
        con = self._get_user_connection()
        if con:
            try:
                # Check if user exists
                res = con.execute("SELECT 1 FROM users WHERE username = ?", [username]).fetchone()
                if not res:
                    return False, "User not found"
                
                con.execute("DELETE FROM users WHERE username = ?", [username])
                return True, f"User '{username}' deleted successfully"
            except Exception as e:
                return False, f"Error deleting user: {e}"
            finally:
                con.close()
        return False, "Database connection failed"

    def _get_connection(self):
        """Get a new DuckDB connection."""
        try:
            return duckdb.connect(self.db_path)
        except Exception as e:
            logging.error(f"Failed to connect to DuckDB: {e}")
            return None

    def _get_next_version(self, base_name):
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
            except:
                continue
        return max(versions) + 1 if versions else 1

    def save_dataframe(self, df, name="main_data", folder=None):
        """Persists a pandas DataFrame to a Parquet file via DuckDB (Low level)."""
        con = self._get_connection()
        if con is None:
            return False, "Database connection not available"
        try:
            target_folder = folder if folder else self.dataset_dir
            file_path = os.path.join(target_folder, f"{name}.parquet")
            con.execute(f"COPY (SELECT * FROM df) TO '{file_path}' (FORMAT PARQUET)")
            return True, f"Data saved to '{file_path}'"
        except Exception as e:
            return False, str(e)
        finally:
            con.close()

    def save_dataset(self, df, base_name="dataset", username=None):
        """Persists a pandas DataFrame to a Parquet file with automatic versioning."""
        con = self._get_connection()
        if con is None:
            return False, "Database connection not available", None
        try:
            # Prefix functionality for user isolation
            if username and username != 'admin':
                display_name = base_name
                base_name = f"{username}_{base_name}"
            
            version = self._get_next_version(base_name)
            file_name = f"{base_name}_v{version}"
            file_path = os.path.join(self.dataset_dir, f"{file_name}.parquet")
            
            # Use DuckDB to write parquet efficiently
            con.execute(f"COPY (SELECT * FROM df) TO '{file_path}' (FORMAT PARQUET)")
            return True, f"Data saved as '{file_name}'", file_name
        except Exception as e:
            return False, str(e), None
        finally:
            con.close()

    def load_dataframe(self, name="main_data", folder=None):
        """Loads a Parquet file into a pandas DataFrame via DuckDB (Low level)."""
        con = self._get_connection()
        if con is None:
            return None, "Database connection not available"
        try:
            target_folder = folder if folder else self.dataset_dir
            file_path = os.path.join(target_folder, f"{name}.parquet")
            df = con.execute(f"SELECT * FROM read_parquet('{file_path}')").df()
            return df, f"Data loaded from '{file_path}'"
        except Exception as e:
            return None, f"Failed to load '{name}': {str(e)}"
        finally:
            con.close()

    def load_dataset(self, file_name):
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
        return self.load_dataframe(file_name)
            
    def get_available_tables(self):
        """List available parquet files in the current directory."""
        files = glob.glob("*.parquet")
        return [f.replace(".parquet", "") for f in files if not f.startswith("stats_") and not f.startswith("tmp_")]

    def get_available_datasets(self, username=None):
        """
        List available parquet files in the current directory, sorted by modification time.
        Filters by username if provided (unless admin).
        """
        # files = glob.glob("*.parquet")
        files = glob.glob(os.path.join(self.dataset_dir, "*.parquet"))
        # Filter out stats files, temp files, and User DB
        dataset_files = [f for f in files if not os.path.basename(f).startswith("stats_") and not os.path.basename(f).startswith("tmp_") and not os.path.basename(f) == "users.parquet" and not os.path.basename(f) == "users.duckdb"]
        
        # Sort by modification time (newest first)
        dataset_files.sort(key=os.path.getmtime, reverse=True)
        
        all_files = [os.path.basename(f).replace(".parquet", "") for f in dataset_files]
        
        if username and username != 'admin':
            return [f for f in all_files if f.startswith(f"{username}_")]
        
        return all_files

    def save_stats(self, df, dataset_name, stats_type):
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
                except:
                    pass
            return True, f"Cleared {count} cached statistics files."
        except Exception as e:
            return False, f"Failed to clean cache: {e}"
