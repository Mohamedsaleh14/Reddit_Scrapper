# db/reader.py

import sqlite3
from datetime import datetime, UTC
from config.config_loader import get_config
from pathlib import Path

config = get_config()
DB_PATH = config["database"]["path"]

# Initialize shared read-only connection with WAL mode
_conn = None

def _get_connection():
    global _conn
    if _conn is None:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL;")
    return _conn

def is_already_processed(post_id: str) -> bool:
    """Check if a post or comment has already been processed."""
    conn = _get_connection()
    try:
        result = conn.execute("SELECT 1 FROM history WHERE id = ?", (post_id,)).fetchone()
        return result is not None
    except sqlite3.Error as e:
        print(f"SQLite error during is_already_processed: {e}")
        return False

def get_all_posts_by_tag(tag: str) -> list:
    conn = _get_connection()
    rows = conn.execute("""
        SELECT * FROM posts
        WHERE tags LIKE ?
        ORDER BY processed_at DESC
    """, (f"%{tag}%",)).fetchall()
    return [dict(row) for row in rows]

def get_posts_by_ids(post_ids: set, require_unprocessed: bool = False) -> list:
    """Retrieve full post records for a set of IDs, optionally skipping already-insighted ones."""
    if not post_ids:
        return []

    conn = _get_connection()
    placeholders = ",".join("?" for _ in post_ids)
    query = f"SELECT * FROM posts WHERE id IN ({placeholders})"
    if require_unprocessed:
        query += " AND (insight_processed IS NULL OR insight_processed = 0)"

    try:
        rows = conn.execute(query, tuple(post_ids)).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        print(f"[SQLite get_posts_by_ids Error] {e}")
        return []

def get_post_parent_mapping(post_ids: set) -> dict:
    """Return {id: parent_post_id} for a set of IDs. parent_post_id is None for posts."""
    if not post_ids:
        return {}
    conn = _get_connection()
    placeholders = ",".join("?" for _ in post_ids)
    try:
        rows = conn.execute(
            f"SELECT id, parent_post_id FROM posts WHERE id IN ({placeholders})",
            tuple(post_ids)
        ).fetchall()
        return {row["id"]: row["parent_post_id"] for row in rows}
    except sqlite3.Error as e:
        print(f"[SQLite get_post_parent_mapping Error] {e}")
        return {}

def get_top_insights_from_today(limit=10) -> list:
    today = datetime.now(UTC).date().isoformat()
    conn = _get_connection()
    try:
        rows = conn.execute("""
            SELECT * FROM posts
            WHERE date(processed_at) = ? AND insight_processed = 1
            ORDER BY roi_weight DESC, relevance_score DESC
            LIMIT ?
        """, (today, limit)).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        print(f"[SQLite get_top_insights_from_today Error] {e}")
        return []
