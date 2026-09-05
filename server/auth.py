"""
User accounts, login, and access control for the cisTEM3 API.

Bearer-token auth, not cookies: the API runs with wide-open CORS
(Access-Control-Allow-Origin: *) so cistem3.html can be opened as a
local file or served from any port -- cookie-based sessions need
credentials:'include' plus a non-wildcard origin, which doesn't fit that.
A token in an Authorization header sidesteps it entirely and has no CSRF
surface (CSRF specifically exploits ambient cookie auth).

Tokens are opaque, random, and tracked server-side in SESSIONS (not JWTs)
so they're actually revocable on logout -- this app already has a trivial
place to put that state, so there's no reason to give up revocability for
a stateless-token scheme.

Users and sessions are global (cross-project), unlike everything in db.py,
so they get their own small SQLite file here rather than living in any
one project's database.
"""

import os
import secrets
import sqlite3
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from flask import g, jsonify, request
from werkzeug.security import check_password_hash, generate_password_hash

AUTH_DB_PATH = Path(__file__).parent / "data" / "auth.db"
ADMIN_CREDENTIALS_PATH = Path(__file__).parent / "data" / "admin_credentials.txt"

SESSION_TTL_SECONDS = 30 * 24 * 3600
SESSION_TOUCH_THRESHOLD_SECONDS = 3600

# Fixed dummy hash checked on a "username not found" login attempt so that
# path takes comparable time to a real check_password_hash() call --
# otherwise response latency alone reveals which usernames exist.
_DUMMY_HASH = generate_password_hash("not-a-real-password")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS USERS(
  USER_ID INTEGER PRIMARY KEY,
  USERNAME TEXT NOT NULL UNIQUE,
  PASSWORD_HASH TEXT NOT NULL,
  ROLE TEXT NOT NULL DEFAULT 'user' CHECK(ROLE IN ('admin','user')),
  DISPLAY_NAME TEXT,
  CREATED_AT TEXT NOT NULL,
  CREATED_BY_USER_ID INTEGER
);

CREATE TABLE IF NOT EXISTS SESSIONS(
  TOKEN TEXT PRIMARY KEY,
  USER_ID INTEGER NOT NULL,
  CREATED_AT INTEGER NOT NULL,
  LAST_SEEN_AT INTEGER,
  EXPIRES_AT INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON SESSIONS(EXPIRES_AT);
"""


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def now_epoch():
    return int(time.time())


def get_conn():
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(AUTH_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.executescript(SCHEMA_SQL)
    return conn


def _public_user(row):
    return {
        "id": row["USER_ID"],
        "username": row["USERNAME"],
        "role": row["ROLE"],
        "display_name": row["DISPLAY_NAME"] or row["USERNAME"],
        "created_at": row["CREATED_AT"],
    }


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def create_user(username, password, role, display_name=None, created_by_user_id=None):
    username = (username or "").strip()
    if not username:
        raise ValueError("username is required")
    if not password or len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    if role not in ("admin", "user"):
        raise ValueError("role must be 'admin' or 'user'")

    conn = get_conn()
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO USERS(USERNAME, PASSWORD_HASH, ROLE, DISPLAY_NAME, CREATED_AT, "
                "CREATED_BY_USER_ID) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    username, generate_password_hash(password), role,
                    (display_name or "").strip() or username, now_iso(), created_by_user_id,
                ),
            )
            user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        raise ValueError("username '{}' is already taken".format(username))
    finally:
        conn.close()

    return get_user_by_id(user_id)


def get_user_by_id(user_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM USERS WHERE USER_ID=?", (user_id,)).fetchone()
    conn.close()
    return _public_user(row) if row else None


def list_users():
    conn = get_conn()
    rows = conn.execute(
        "SELECT USER_ID, USERNAME, ROLE, DISPLAY_NAME, CREATED_AT FROM USERS ORDER BY USERNAME"
    ).fetchall()
    conn.close()
    return [_public_user(r) for r in rows]


def user_count():
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) FROM USERS").fetchone()[0]
    conn.close()
    return n


# ---------------------------------------------------------------------------
# Login / sessions
# ---------------------------------------------------------------------------

def authenticate(username, password):
    conn = get_conn()
    row = conn.execute("SELECT * FROM USERS WHERE USERNAME=?", ((username or "").strip(),)).fetchone()
    conn.close()

    if row is None:
        check_password_hash(_DUMMY_HASH, password or "")  # timing-attack guard
        return None
    if not check_password_hash(row["PASSWORD_HASH"], password or ""):
        return None
    return _public_user(row)


def create_session(user_id):
    conn = get_conn()
    with conn:
        conn.execute("DELETE FROM SESSIONS WHERE EXPIRES_AT < ?", (now_epoch(),))
        token = secrets.token_hex(32)
        now = now_epoch()
        conn.execute(
            "INSERT INTO SESSIONS(TOKEN, USER_ID, CREATED_AT, LAST_SEEN_AT, EXPIRES_AT) "
            "VALUES (?, ?, ?, ?, ?)",
            (token, user_id, now, now, now + SESSION_TTL_SECONDS),
        )
    conn.close()
    return token


def revoke_session(token):
    if not token:
        return
    conn = get_conn()
    with conn:
        conn.execute("DELETE FROM SESSIONS WHERE TOKEN=?", (token,))
    conn.close()


def validate_token(token):
    if not token:
        return None
    conn = get_conn()
    row = conn.execute(
        "SELECT s.EXPIRES_AT, s.LAST_SEEN_AT, u.* FROM SESSIONS s "
        "JOIN USERS u ON u.USER_ID = s.USER_ID WHERE s.TOKEN=?",
        (token,),
    ).fetchone()

    if row is None:
        conn.close()
        return None

    now = now_epoch()
    if row["EXPIRES_AT"] < now:
        with conn:
            conn.execute("DELETE FROM SESSIONS WHERE TOKEN=?", (token,))
        conn.close()
        return None

    if row["LAST_SEEN_AT"] is None or now - row["LAST_SEEN_AT"] > SESSION_TOUCH_THRESHOLD_SECONDS:
        with conn:
            conn.execute(
                "UPDATE SESSIONS SET LAST_SEEN_AT=?, EXPIRES_AT=? WHERE TOKEN=?",
                (now, now + SESSION_TTL_SECONDS, token),
            )
    conn.close()
    return _public_user(row)


def get_bearer_token():
    header = request.headers.get("Authorization", "")
    if not header.lower().startswith("bearer "):
        return None
    return header[7:].strip()


def get_current_user():
    token = get_bearer_token()
    return validate_token(token) if token else None


# ---------------------------------------------------------------------------
# Password change / reset
# ---------------------------------------------------------------------------

def verify_password(user_id, password):
    conn = get_conn()
    row = conn.execute("SELECT PASSWORD_HASH FROM USERS WHERE USER_ID=?", (user_id,)).fetchone()
    conn.close()
    if row is None:
        return False
    return check_password_hash(row["PASSWORD_HASH"], password or "")


def set_password(user_id, new_password):
    if not new_password or len(new_password) < 8:
        raise ValueError("password must be at least 8 characters")
    conn = get_conn()
    with conn:
        cur = conn.execute(
            "UPDATE USERS SET PASSWORD_HASH=? WHERE USER_ID=?",
            (generate_password_hash(new_password), user_id),
        )
    found = cur.rowcount > 0
    conn.close()
    if not found:
        raise ValueError("user not found")


def set_role(user_id, role):
    """Promote or demote an account. Takes effect on the target's next
    request, since every request looks its user up afresh -- no session
    needs revoking. The route guards against removing the last admin."""
    if role not in ("admin", "user"):
        raise ValueError("role must be 'admin' or 'user'")
    conn = get_conn()
    with conn:
        cur = conn.execute("UPDATE USERS SET ROLE=? WHERE USER_ID=?", (role, user_id))
    found = cur.rowcount > 0
    conn.close()
    if not found:
        raise ValueError("user not found")
    return get_user_by_id(user_id)


def admin_count():
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) FROM USERS WHERE ROLE='admin'").fetchone()[0]
    conn.close()
    return n


def revoke_user_sessions(user_id, keep_token=None):
    """Invalidate a user's other sessions after their password changes --
    self-service change keeps the session that made the request alive
    (keep_token); an admin-driven reset has no session of the target
    user's to keep, so it clears all of them."""
    conn = get_conn()
    with conn:
        if keep_token:
            conn.execute("DELETE FROM SESSIONS WHERE USER_ID=? AND TOKEN<>?", (user_id, keep_token))
        else:
            conn.execute("DELETE FROM SESSIONS WHERE USER_ID=?", (user_id,))
    conn.close()


# ---------------------------------------------------------------------------
# Route guards
# ---------------------------------------------------------------------------

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        user = get_current_user()
        if user is None:
            return jsonify({"error": "authentication required"}), 401
        g.current_user = user
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        user = get_current_user()
        if user is None:
            return jsonify({"error": "authentication required"}), 401
        if user["role"] != "admin":
            return jsonify({"error": "admin access required"}), 403
        g.current_user = user
        return view(*args, **kwargs)
    return wrapped


def project_access_required(view):
    """401 unauthenticated, 404 unknown project, 403 not owner/not admin."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        import db  # local import: db.py has no dependency on auth.py otherwise

        user = get_current_user()
        if user is None:
            return jsonify({"error": "authentication required"}), 401

        project_id = kwargs.get("project_id")
        if not db.project_exists(project_id):
            return jsonify({"error": "project not found"}), 404

        if user["role"] != "admin":
            owner_id = db.get_project_owner(project_id)
            if owner_id != user["id"]:
                return jsonify({"error": "you do not have access to this project"}), 403

        g.current_user = user
        return view(*args, **kwargs)
    return wrapped


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_admin_if_needed():
    if user_count() > 0:
        return

    password = secrets.token_urlsafe(16)
    conn = get_conn()
    try:
        with conn:
            conn.execute(
                "INSERT INTO USERS(USERNAME, PASSWORD_HASH, ROLE, DISPLAY_NAME, CREATED_AT) "
                "VALUES ('admin', ?, 'admin', 'Administrator', ?)",
                (generate_password_hash(password), now_iso()),
            )
    except sqlite3.IntegrityError:
        # another process won the race to create the first admin -- fine
        conn.close()
        return
    conn.close()

    ADMIN_CREDENTIALS_PATH.write_text(
        "cisTEM3 -- initial admin account\n"
        "username: admin\n"
        "password: {}\n"
        "\n"
        "Log in and create real accounts for your team; this file is only\n"
        "written once, when the very first admin account is created.\n".format(password)
    )
    os.chmod(ADMIN_CREDENTIALS_PATH, 0o600)

    print("=" * 70)
    print("No users found -- created the initial admin account:")
    print("  username: admin")
    print("  password: {}".format(password))
    print("(also written to {})".format(ADMIN_CREDENTIALS_PATH))
    print("=" * 70)
