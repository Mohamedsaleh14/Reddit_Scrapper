# db/cleaner.py

import sqlite3
from datetime import datetime, timedelta
from config.config_loader import get_config
from utils.logger import setup_logger

log = setup_logger()
config = get_config()
DB_PATH = config["database"]["path"]

def clean_old_entries():
    """Remove posts and history entries older than the configured retention period."""
    retention_days = config["database"]["retention_days"]
    cutoff_date = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()
    log.debug(f"cutoff date is {cutoff_date}")

    log.info(f"Cleaning entries older than {retention_days} days...")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    DELETE FROM posts
    WHERE processed_at < ?
    """, (cutoff_date,))
    posts_deleted = c.rowcount

    c.execute("""
    DELETE FROM history
    WHERE processed_at < ?
    """, (cutoff_date,))
    history_deleted = c.rowcount

    conn.commit()
    conn.close()

    log.info(f"Cleaned {posts_deleted} posts and {history_deleted} history entries.")
    return posts_deleted, history_deleted
