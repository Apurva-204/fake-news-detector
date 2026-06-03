"""
app/services/auth_service.py
----------------------------
SQLite-based user authentication service.

Responsibilities:
- Initialize the users database and table
- Register new users (with hashed passwords)
- Authenticate users on login
- Fetch user records by ID or username

Password hashing uses werkzeug.security (bcrypt-style pbkdf2).
No plaintext passwords are ever stored.
"""

import sqlite3
import os
from typing import Optional
from werkzeug.security import generate_password_hash, check_password_hash


# -------------------------------------------------------------------
# Database helpers
# -------------------------------------------------------------------

def _get_connection(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection with row_factory for dict-like access."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    return conn


def init_db(db_path: str) -> None:
    """
    Create the users table if it does not already exist.

    Schema:
        id        INTEGER PRIMARY KEY AUTOINCREMENT
        username  TEXT    UNIQUE NOT NULL
        email     TEXT    UNIQUE NOT NULL
        password  TEXT    NOT NULL  (hashed)
        role      TEXT    DEFAULT 'user'   ('user' | 'admin')
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None
    conn = _get_connection(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT    NOT NULL UNIQUE,
                email      TEXT    NOT NULL UNIQUE,
                password   TEXT    NOT NULL,
                role       TEXT    NOT NULL DEFAULT 'user',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    finally:
        conn.close()


# -------------------------------------------------------------------
# User CRUD
# -------------------------------------------------------------------

def register_user(
    db_path: str,
    username: str,
    email: str,
    password: str,
    role: str = "user",
) -> tuple[bool, str]:
    """
    Register a new user.

    Args:
        db_path  : Path to the SQLite database file.
        username : Desired username (must be unique).
        email    : User email address (must be unique).
        password : Plaintext password (will be hashed before storage).
        role     : 'user' or 'admin'.

    Returns:
        (success: bool, message: str)
    """
    # Basic validation
    username = username.strip()
    email    = email.strip().lower()
    password = password.strip()

    if not username or not email or not password:
        return False, "All fields are required."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if "@" not in email or "." not in email:
        return False, "Please enter a valid email address."

    hashed_pw = generate_password_hash(password)

    conn = _get_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)",
            (username, email, hashed_pw, role),
        )
        conn.commit()
        return True, "Account created successfully. Please log in."
    except sqlite3.IntegrityError as e:
        error_str = str(e).lower()
        if "username" in error_str:
            return False, "That username is already taken."
        if "email" in error_str:
            return False, "An account with that email already exists."
        return False, "Registration failed. Please try again."
    finally:
        conn.close()


def authenticate_user(
    db_path: str,
    username: str,
    password: str,
) -> tuple[Optional[dict], str]:
    """
    Authenticate a user by username + password.

    Args:
        db_path  : Path to the SQLite database file.
        username : Username to look up.
        password : Plaintext password to verify.

    Returns:
        (user_dict | None, message: str)
        user_dict contains: id, username, email, role, created_at
    """
    username = username.strip()
    password = password.strip()

    if not username or not password:
        return None, "Please enter your username and password."

    conn = _get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT id, username, email, password, role, created_at FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None, "Invalid username or password."

    if not check_password_hash(row["password"], password):
        return None, "Invalid username or password."

    user = {
        "id":         row["id"],
        "username":   row["username"],
        "email":      row["email"],
        "role":       row["role"],
        "created_at": row["created_at"],
    }
    return user, "Login successful."


def get_user_by_id(db_path: str, user_id: int) -> Optional[dict]:
    """
    Fetch a user record by primary key (used to restore session data).

    Args:
        db_path : Path to the SQLite database file.
        user_id : The user's integer ID.

    Returns:
        dict with user fields, or None if not found.
    """
    conn = _get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT id, username, email, role, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    return {
        "id":         row["id"],
        "username":   row["username"],
        "email":      row["email"],
        "role":       row["role"],
        "created_at": row["created_at"],
    }
