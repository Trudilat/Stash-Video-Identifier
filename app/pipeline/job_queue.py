import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

_DATA_DIR = Path(".data")
_DATA_DIR.mkdir(exist_ok=True)
DB_PATH = _DATA_DIR / "jobs.db"


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                scene_id        TEXT PRIMARY KEY,
                stash_path      TEXT NOT NULL,
                duration        REAL NOT NULL,
                status          TEXT DEFAULT 'pending',
                error           TEXT,
                existing_title  TEXT,
                title           TEXT,
                tag_names       TEXT,
                proc_seconds    REAL,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Migration: add existing_title to databases created before this column existed
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN existing_title TEXT")
        except Exception:
            pass  # column already exists


def reset_in_progress():
    """On startup, move any stuck in_progress jobs back to pending."""
    with _conn() as conn:
        conn.execute(
            "UPDATE jobs SET status='pending' WHERE status='in_progress'"
        )


def add_jobs(scenes: list[dict]) -> int:
    """
    Insert scenes into the queue, skipping ones already present.
    Returns the number of newly added jobs.
    """
    added = 0
    with _conn() as conn:
        for scene in scenes:
            path = scene["files"][0]["path"]
            duration = scene["files"][0]["duration"]
            existing_title = scene.get("title") or ""
            cur = conn.execute(
                "INSERT OR IGNORE INTO jobs (scene_id, stash_path, duration, existing_title) VALUES (?,?,?,?)",
                (scene["id"], path, duration, existing_title),
            )
            if cur.rowcount:
                added += 1
    return added


def get_next_job() -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status='pending' ORDER BY rowid LIMIT 1"
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE jobs SET status='in_progress', updated_at=CURRENT_TIMESTAMP WHERE scene_id=?",
                (row["scene_id"],),
            )
            return dict(row)
    return None


def mark_done(scene_id: str, title: str, tag_names: list[str], proc_seconds: float):
    with _conn() as conn:
        conn.execute(
            """UPDATE jobs
               SET status='done', title=?, tag_names=?, proc_seconds=?, updated_at=CURRENT_TIMESTAMP
               WHERE scene_id=?""",
            (title, json.dumps(tag_names), proc_seconds, scene_id),
        )


def reset_failed() -> int:
    """Reset all failed jobs back to pending so they are retried. Returns count."""
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE jobs SET status='pending', error=NULL, updated_at=CURRENT_TIMESTAMP "
            "WHERE status='failed'"
        )
        return cur.rowcount


def mark_failed(scene_id: str, error: str):
    with _conn() as conn:
        conn.execute(
            "UPDATE jobs SET status='failed', error=?, updated_at=CURRENT_TIMESTAMP WHERE scene_id=?",
            (error, scene_id),
        )


def get_counts() -> dict:
    with _conn() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN status='pending'     THEN 1 ELSE 0 END) AS pending,
                SUM(CASE WHEN status='in_progress' THEN 1 ELSE 0 END) AS in_progress,
                SUM(CASE WHEN status='done'        THEN 1 ELSE 0 END) AS done,
                SUM(CASE WHEN status='failed'      THEN 1 ELSE 0 END) AS failed,
                SUM(CASE WHEN status='done'        THEN proc_seconds ELSE 0 END) AS total_proc_sec,
                SUM(CASE WHEN status='done'        THEN duration     ELSE 0 END) AS total_video_sec,
                SUM(CASE WHEN status='pending'     THEN duration     ELSE 0 END) AS remaining_video_sec
            FROM jobs
        """).fetchone()
        return dict(row)


def clear_queue():
    with _conn() as conn:
        conn.execute("DELETE FROM jobs")


def matches_folder(stash_path: str, folder: str, recursive: bool) -> bool:
    """Check whether a Stash path belongs to the given folder config."""
    if not stash_path.startswith(folder):
        return False
    if recursive:
        return True
    rel = stash_path[len(folder):]
    return "/" not in rel
