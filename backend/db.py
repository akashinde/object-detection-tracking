import os
import sqlite3
from save_to_db import _create_tables

DB_PATH = 'analytics.db'
UPLOAD_FOLDER = 'videos/uploads'
PROCESSED_FOLDER = 'videos/processed'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_normalized_db_tables():
    conn = sqlite3.connect(DB_PATH)
    _create_tables(conn)
    conn.close()
