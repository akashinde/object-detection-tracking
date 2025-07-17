import sqlite3

DB_PATH = 'detections.db'

def truncate_all_tables(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all user tables (ignore sqlite internal tables)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        cursor.execute(f'DELETE FROM "{table}";')
        # Optionally reset auto-increment keys:
        cursor.execute(f'DELETE FROM sqlite_sequence WHERE name="{table}";')

    conn.commit()
    conn.close()
    print(f"All tables in {db_path} have been emptied.")

if __name__ == "__main__":
    truncate_all_tables(DB_PATH)