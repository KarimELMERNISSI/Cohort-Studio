"""
Local DuckDB IAM Provider.

Default, zero-dependency embedded authentication backend utilizing DuckDB
and bcrypt password hashing. Designed for single-node deployments, air-gapped
clinical environments, and full backward compatibility.
"""

import logging
import os
from typing import Any

import bcrypt
import duckdb

import utils.data_paths as dp
from ..base import BaseIAMProvider
from ..schemas import AuthResult, UserCredentials, UserProfile, UserRole

logger = logging.getLogger(__name__)


class LocalDuckDBIAMProvider(BaseIAMProvider):
    """
    Local embedded IAM backend using DuckDB and bcrypt.
    """

    def __init__(self, db_path: str | None = None) -> None:
        """
        Initialize the local DuckDB IAM provider.

        Args:
            db_path: Optional custom path to users.duckdb file. If None,
                     resolves via standard data paths.
        """
        if db_path:
            self.user_db_path = db_path
        else:
            self.user_db_path = os.path.join(dp.get_datasets_dir(), "users.duckdb")

        self._init_database()

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return "local_duckdb"

    def _get_connection(self) -> duckdb.DuckDBPyConnection | None:
        """
        Acquire an on-demand DuckDB connection.
        Closes explicitly per operation to prevent Windows file lock conflicts.
        """
        try:
            return duckdb.connect(self.user_db_path)
        except Exception as e:
            logger.error(f"Failed to connect to User DuckDB at {self.user_db_path}: {e}")
            return None

    def _init_database(self) -> None:
        """Initialize user directory table and apply schema migrations."""
        con = self._get_connection()
        if not con:
            return
        try:
            con.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT,
                    is_active BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    role TEXT DEFAULT 'researcher'
                )
            """)

            # Safe migrations for pre-existing tables
            try:
                con.execute("ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT FALSE")
            except Exception:
                pass

            try:
                con.execute("ALTER TABLE users ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            except Exception:
                pass

            try:
                con.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'researcher'")
            except Exception:
                pass

            # Administrator account is perpetually active
            con.execute("UPDATE users SET is_active = TRUE, role = 'admin' WHERE username = 'admin'")
        except Exception as e:
            logger.error(f"Error initializing local user database: {e}")
        finally:
            con.close()

    def authenticate(self, credentials: UserCredentials) -> AuthResult:
        """
        Authenticate local credentials against bcrypt password hashes.
        """
        con = self._get_connection()
        if not con:
            return AuthResult(success=False, message="Database connection failed")

        try:
            res = con.execute(
                "SELECT password_hash, is_active, role, created_at FROM users WHERE username = ?",
                [credentials.username],
            ).fetchone()

            if not res:
                return AuthResult(success=False, message="Invalid username or password")

            stored_hash = res[0]
            is_active = bool(res[1]) if len(res) > 1 and res[1] is not None else True
            raw_role = str(res[2]) if len(res) > 2 and res[2] is not None else "researcher"
            created_at = str(res[3]) if len(res) > 3 and res[3] is not None else None

            # Verify bcrypt hash
            if bcrypt.checkpw(credentials.password.encode("utf-8"), stored_hash.encode("utf-8")):
                if not is_active:
                    return AuthResult(
                        success=False,
                        message="Account pending approval. Please contact administrator.",
                    )

                role_enum = UserRole.ADMIN if raw_role == "admin" else UserRole.RESEARCHER
                profile = UserProfile(
                    username=credentials.username,
                    role=role_enum,
                    is_active=True,
                    created_at=created_at,
                )
                return AuthResult(
                    success=True,
                    user=profile,
                    token=f"local_token_{credentials.username}",
                    message="Login successful",
                )

            return AuthResult(success=False, message="Invalid username or password")
        except Exception as e:
            logger.error(f"Exception during user verification: {e}")
            return AuthResult(success=False, message=f"Authentication error: {e}")
        finally:
            con.close()

    def create_user(
        self,
        username: str,
        password: str,
        role: UserRole = UserRole.RESEARCHER,
        auto_activate: bool = False,
    ) -> tuple[bool, str]:
        """
        Provision a new local user with bcrypt password hashing.
        """
        if not username or not password:
            return False, "Username and password cannot be empty"

        con = self._get_connection()
        if not con:
            return False, "Database connection failed"

        try:
            # Check for existing username
            res = con.execute("SELECT 1 FROM users WHERE username = ?", [username]).fetchone()
            if res:
                return False, "Username already exists"

            hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            is_active = True if username == "admin" or auto_activate else False
            role_val = "admin" if username == "admin" else role.value

            con.execute(
                "INSERT INTO users (username, password_hash, is_active, role) VALUES (?, ?, ?, ?)",
                [username, hashed, is_active, role_val],
            )

            if is_active:
                return True, "User created and activated successfully"
            return True, "Account created! Please wait for admin approval to access the app."
        except Exception as e:
            logger.error(f"Error creating user {username}: {e}")
            return False, f"Error creating user: {e}"
        finally:
            con.close()

    def verify_session(self, token: str) -> AuthResult:
        """
        Validate local session token.
        """
        if not token or not token.startswith("local_token_"):
            return AuthResult(success=False, message="Invalid session token")

        username = token.replace("local_token_", "")
        user = self.get_user(username)
        if user and user.is_active:
            return AuthResult(success=True, user=user, token=token, message="Session valid")
        return AuthResult(success=False, message="Session expired or account inactive")

    def get_user(self, username: str) -> UserProfile | None:
        """Retrieve user profile from local database."""
        con = self._get_connection()
        if not con:
            return None
        try:
            res = con.execute(
                "SELECT username, is_active, role, created_at FROM users WHERE username = ?",
                [username],
            ).fetchone()
            if not res:
                return None

            raw_role = str(res[2]) if len(res) > 2 and res[2] is not None else "researcher"
            role_enum = UserRole.ADMIN if raw_role == "admin" else UserRole.RESEARCHER
            return UserProfile(
                username=res[0],
                is_active=bool(res[1]),
                role=role_enum,
                created_at=str(res[3]) if len(res) > 3 and res[3] is not None else None,
            )
        except Exception as e:
            logger.error(f"Error fetching user {username}: {e}")
            return None
        finally:
            con.close()

    def list_users(self) -> list[UserProfile]:
        """List all users in local database."""
        con = self._get_connection()
        if not con:
            return []
        try:
            res = con.execute(
                "SELECT username, is_active, role, created_at FROM users ORDER BY is_active ASC, username ASC"
            ).fetchall()
            users: list[UserProfile] = []
            for row in res:
                raw_role = str(row[2]) if len(row) > 2 and row[2] is not None else "researcher"
                role_enum = UserRole.ADMIN if raw_role == "admin" else UserRole.RESEARCHER
                users.append(
                    UserProfile(
                        username=row[0],
                        is_active=bool(row[1]) if row[1] is not None else True,
                        role=role_enum,
                        created_at=str(row[3]) if len(row) > 3 and row[3] is not None else None,
                    )
                )
            return users
        except Exception as e:
            logger.error(f"Error listing users: {e}")
            return []
        finally:
            con.close()

    def activate_user(self, username: str) -> tuple[bool, str]:
        """Activate a user account."""
        con = self._get_connection()
        if not con:
            return False, "Database connection failed"
        try:
            res = con.execute("SELECT 1 FROM users WHERE username = ?", [username]).fetchone()
            if not res:
                return False, "User not found"
            con.execute("UPDATE users SET is_active = TRUE WHERE username = ?", [username])
            return True, f"User '{username}' has been activated"
        except Exception as e:
            return False, f"Error activating user: {e}"
        finally:
            con.close()

    def deactivate_user(self, username: str) -> tuple[bool, str]:
        """Deactivate a user account (admin protected)."""
        if username == "admin":
            return False, "Cannot deactivate admin account"
        con = self._get_connection()
        if not con:
            return False, "Database connection failed"
        try:
            res = con.execute("SELECT 1 FROM users WHERE username = ?", [username]).fetchone()
            if not res:
                return False, "User not found"
            con.execute("UPDATE users SET is_active = FALSE WHERE username = ?", [username])
            return True, f"User '{username}' has been deactivated"
        except Exception as e:
            return False, f"Error deactivating user: {e}"
        finally:
            con.close()

    def delete_user(self, username: str) -> tuple[bool, str]:
        """Delete a user account (admin protected)."""
        if username == "admin":
            return False, "Cannot delete admin account"
        con = self._get_connection()
        if not con:
            return False, "Database connection failed"
        try:
            res = con.execute("SELECT 1 FROM users WHERE username = ?", [username]).fetchone()
            if not res:
                return False, "User not found"
            con.execute("DELETE FROM users WHERE username = ?", [username])
            return True, f"User '{username}' deleted successfully"
        except Exception as e:
            return False, f"Error deleting user: {e}"
        finally:
            con.close()

    def update_password(self, username: str, new_password: str) -> tuple[bool, str]:
        """Update password for an existing user."""
        con = self._get_connection()
        if not con:
            return False, "Database connection failed"
        try:
            res = con.execute("SELECT 1 FROM users WHERE username = ?", [username]).fetchone()
            if not res:
                return False, "User not found"
            hashed = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            con.execute("UPDATE users SET password_hash = ? WHERE username = ?", [hashed, username])
            return True, f"Password updated for user '{username}'"
        except Exception as e:
            return False, f"Error updating password: {e}"
        finally:
            con.close()
