import sqlite3
import sys
import logging

DB_PATH = 'detections.db'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def truncate_all_tables(db_path):
    try:
        logging.info(f"Connecting to database: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all user tables (ignore sqlite internal tables)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]
        logging.info(f"Found tables: {tables}")

        for table in tables:
            logging.info(f"Truncating table: {table}")
            cursor.execute(f'DELETE FROM "{table}";')
            # Optionally reset auto-increment keys:
            cursor.execute(f'DELETE FROM sqlite_sequence WHERE name="{table}";')

        conn.commit()
        logging.info(f"All tables in {db_path} have been emptied.")
    except Exception as e:
        logging.error(f"Error while truncating tables: {e}")
    finally:
        conn.close()
        logging.info("Database connection closed.")

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else DB_PATH
    truncate_all_tables(db_path)