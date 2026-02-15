# db/schema.py

import sqlite3
import os
from config.config_loader import get_config
from utils.logger import setup_logger
from utils.helpers import ensure_directory_exists

log = setup_logger()
config = get_config()
DB_PATH = config["database"]["path"]

def create_tables():
    """Create the SQLite tables if they don't exist."""
    ensure_directory_exists(os.path.dirname(DB_PATH))
    log.info(f"Initializing database at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS posts (
        id TEXT PRIMARY KEY,
        url TEXT,
        title TEXT,
        body TEXT,
        subreddit TEXT,
        created_utc REAL,
        last_active REAL,
        processed_at TEXT,
        relevance_score REAL,
        emotion_score REAL,
        pain_score REAL,
        tags TEXT,
        roi_weight INTEGER,
        community_type TEXT,
        type TEXT,  -- 'post' or 'comment'
        post_body TEXT,  -- parent post body for comments
        parent_post_id TEXT,  -- links comments to their parent post (for dedup)
        implementability_score REAL,
        technical_depth_score REAL,
        insight_processed INTEGER DEFAULT 0,
        insight_processed_at TEXT
    );
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id TEXT PRIMARY KEY,
        processed_at TEXT
    );
    """)

    # Migration: add technical_depth_score column if missing (for existing databases)
    try:
        c.execute("ALTER TABLE posts ADD COLUMN technical_depth_score REAL")
        log.info("Added technical_depth_score column to posts table")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Migration: add parent_post_id column if missing (for existing databases)
    try:
        c.execute("ALTER TABLE posts ADD COLUMN parent_post_id TEXT")
        log.info("Added parent_post_id column to posts table")
    except sqlite3.OperationalError:
        pass  # Column already exists

    c.execute("CREATE INDEX IF NOT EXISTS idx_posts_processed_at ON posts(processed_at);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_posts_relevance ON posts(relevance_score);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_posts_roi ON posts(roi_weight);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_posts_subreddit ON posts(subreddit);")

    conn.commit()
    conn.close()
    log.info("Database tables created successfully")

if __name__ == "__main__":
    create_tables()
    print(f"Database initialized at {DB_PATH}")
