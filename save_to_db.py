import sqlite3
import uuid
from typing import Dict, Any, List

DB_PATH = "analytics.db"

def _create_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS videos (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id          TEXT UNIQUE,
        filename          TEXT,
        source            TEXT,
        duration_sec      REAL,
        fps               REAL,
        total_frames      INTEGER,
        uploaded_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS video_summary (
        video_id TEXT PRIMARY KEY,
        total_cars INTEGER,
        distinct_plates INTEGER,
        avg_cars_per_second REAL,
        duration_sec REAL,
        avg_car_visibility_percent REAL,
        average_confidence REAL,
        most_active_start_sec REAL,
        most_active_end_sec REAL,
        avg_cars_per_frame REAL,
        FOREIGN KEY(video_id) REFERENCES videos(video_id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS vehicles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT,
        track_id INTEGER,
        first_seen_sec REAL,
        last_seen_sec REAL,
        dwell_time_seconds REAL,
        type TEXT,
        color TEXT,
        brand TEXT,
        model TEXT,
        license_plate TEXT,
        license_region TEXT,
        is_moving INTEGER,
        image_path TEXT,
        UNIQUE(video_id, track_id),
        FOREIGN KEY(video_id) REFERENCES videos(video_id)
    );""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS color_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT,
        color TEXT,
        count INTEGER,
        FOREIGN KEY(video_id) REFERENCES videos(video_id)
    );""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS brand_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT,
        brand TEXT,
        logo_count INTEGER,
        FOREIGN KEY(video_id) REFERENCES videos(video_id)
    );""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS model_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT,
        make TEXT,
        model TEXT,
        count INTEGER,
        FOREIGN KEY(video_id) REFERENCES videos(video_id)
    );""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS region_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT,
        region TEXT,
        plate_count INTEGER,
        FOREIGN KEY(video_id) REFERENCES videos(video_id)
    );""")

    conn.commit()

def save_analytics_to_db(analytics: Dict[str, Any], db_path: str = DB_PATH) -> Dict[str, int]:
    row_counts = {
        "videos": 0, "video_summary": 0, "vehicles": 0,
        "color_stats": 0, "brand_stats": 0,
        "model_stats": 0, "region_stats": 0
    }

    conn = sqlite3.connect(db_path)
    _create_tables(conn)
    cur = conn.cursor()

    video = analytics["video"]
    video_id = analytics.get("video_id") or uuid.uuid4().hex[:8]

    cur.execute("""
        INSERT OR IGNORE INTO videos
          (video_id, filename, source, duration_sec, fps, total_frames)
        VALUES (?, ?, ?, ?, ?, ?);
    """, (
        video_id,
        video.get("filename"),
        video.get("source"),
        video.get("duration_sec"),
        video.get("fps"),
        video.get("total_frames"),
    ))
    row_counts["videos"] += cur.rowcount

    vs = analytics["video_summary"]
    cur.execute("""
        INSERT OR REPLACE INTO video_summary (
            video_id, total_cars, distinct_plates, avg_cars_per_second,
            duration_sec, avg_car_visibility_percent, average_confidence,
            most_active_start_sec, most_active_end_sec, avg_cars_per_frame
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        video_id,
        vs.get("total_cars"),
        vs.get("distinct_plates"),
        vs.get("avg_cars_per_second"),
        vs.get("duration_sec"),
        vs.get("avg_car_visibility_percent"),
        vs.get("average_confidence"),
        vs.get("most_active_start_sec"),
        vs.get("most_active_end_sec"),
        vs.get("avg_cars_per_frame")
    ))
    row_counts["video_summary"] += cur.rowcount

    vehicles: List[dict] = analytics["vehicles"]
    cur.executemany("""
        INSERT OR REPLACE INTO vehicles
          (video_id, track_id, first_seen_sec, last_seen_sec, dwell_time_seconds,
           type, color, brand, model, license_plate, license_region, is_moving, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, [
        (
            video_id,
            v["track_id"],
            v["first_seen_sec"],
            v["last_seen_sec"],
            v["dwell_time_seconds"],
            v["type"],
            v["color"],
            v["brand"],
            v["model"],
            v["license_plate"],
            v["license_region"],
            int(v["is_moving"]),
            v["image_path"],
        )
        for v in vehicles
    ])
    row_counts["vehicles"] += cur.rowcount

    def _bulk_insert(table: str, cols: List[str], rows: List[tuple]):
        if not rows:
            return
        placeholders = ",".join(["?"] * len(cols))
        collist = ",".join(cols)
        sql = f"INSERT OR REPLACE INTO {table} ({collist}) VALUES ({placeholders});"
        cur.executemany(sql, rows)
        row_counts[table] += cur.rowcount

    _bulk_insert(
        "color_stats",
        ["video_id", "color", "count"],
        [(video_id, c["color"], c["count"]) for c in analytics["color_stats"]]
    )

    _bulk_insert(
        "brand_stats",
        ["video_id", "brand", "logo_count"],
        [(video_id, b["brand"], b["logo_count"]) for b in analytics["brand_stats"]]
    )

    _bulk_insert(
        "model_stats",
        ["video_id", "make", "model", "count"],
        [(video_id, m["make"], m["model"], m["count"]) for m in analytics["model_stats"]]
    )

    _bulk_insert(
        "region_stats",
        ["video_id", "region", "plate_count"],
        [(video_id, r["region"], r["plate_count"]) for r in analytics["region_stats"]]
    )

    conn.commit()
    conn.close()
    return row_counts

