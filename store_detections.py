import json
import sqlite3
import os
import re

def create_tables(conn):
    """Create normalized database tables"""
    conn.execute('''
        CREATE TABLE IF NOT EXISTS CAR_TYPES (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS CAR_MODELS (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS CAR_COLORS (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS VIDEOS (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS CARS (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            track_id INTEGER,
            label TEXT,
            type_id INTEGER,
            model_id INTEGER,
            color_id INTEGER,
            license_plate TEXT,
            license_plate_confidence REAL,
            track_frame_counts INTEGER,
            scene_count INTEGER,
            dwell_time_seconds REAL,
            image_path TEXT,
            FOREIGN KEY(video_id) REFERENCES VIDEOS(id),
            FOREIGN KEY(type_id) REFERENCES CAR_TYPES(id),
            FOREIGN KEY(model_id) REFERENCES CAR_MODELS(id),
            FOREIGN KEY(color_id) REFERENCES CAR_COLORS(id)
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS STATES (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            state TEXT,
            car_id INTEGER,
            FOREIGN KEY(car_id) REFERENCES CARS(id)
        )
    ''')
    # --- Add peak_hour_range and peak_hour_count to SUMMARY_STATS ---
    conn.execute('''
        CREATE TABLE IF NOT EXISTS SUMMARY_STATS (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            total_cars INTEGER,
            unique_models INTEGER,
            unique_license_plates INTEGER,
            average_dwell_time REAL,
            color_counts_json TEXT,
            peak_hour_range TEXT,
            peak_hour_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()

def get_or_create_car_type(conn, type_name):
    """Get car type record (static table, no insert)"""
    cur = conn.execute('SELECT id FROM CAR_TYPES WHERE name=?', (type_name,))
    row = cur.fetchone()
    if row:
        return row[0]
    # If not found, return None (or handle as needed)
    return None

def get_or_create_car_model(conn, model_name):
    """Get or create car model record"""
    cur = conn.execute('SELECT id FROM CAR_MODELS WHERE name=?', (model_name,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur = conn.execute('INSERT INTO CAR_MODELS (name) VALUES (?)', (model_name,))
    conn.commit()
    return cur.lastrowid

def get_or_create_car_color(conn, color_name):
    """Get or create car color record"""
    cur = conn.execute('SELECT id FROM CAR_COLORS WHERE name=?', (color_name,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur = conn.execute('INSERT INTO CAR_COLORS (name) VALUES (?)', (color_name,))
    conn.commit()
    return cur.lastrowid

def get_or_create_video(conn, filename):
    cur = conn.execute('SELECT id FROM VIDEOS WHERE filename=?', (filename,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur = conn.execute('INSERT INTO VIDEOS (filename) VALUES (?)', (filename,))
    conn.commit()
    return cur.lastrowid

def get_or_create_car(conn, car_data, video_id):
    """Get or create car record for a specific video"""
    type_id = get_or_create_car_type(conn, car_data['type'])
    if type_id is None:
        type_id = get_or_create_car_type(conn, 'other')
    model_id = get_or_create_car_model(conn, car_data['model'])
    color_name = car_data['color'] if car_data.get('color') else 'unknown'
    color_id = get_or_create_car_color(conn, color_name)
    license_plate_confidence = car_data.get('license_plate_confidence')
    # Fix image_path: always use videos/processed/<video_name>/car_<track_id>.jpg
    image_path = car_data.get('image_path')
    if not image_path:
        # Get video filename from car_data or fallback
        video_filename = car_data.get('video_filename') or car_data.get('label', '').split('_')[0] or 'unknown_video.mp4'
        video_name = os.path.splitext(os.path.basename(video_filename))[0]
        image_path = f"videos/processed/{video_name}/car_{car_data['track_id']}.jpg"
    # Check if car already exists for this video and track_id
    cur = conn.execute('SELECT id FROM CARS WHERE video_id=? AND track_id=?', (video_id, car_data['track_id']))
    row = cur.fetchone()
    if row:
        # Update existing car
        conn.execute('''
            UPDATE CARS SET 
                label=?, type_id=?, model_id=?, color_id=?, license_plate=?, license_plate_confidence=?,
                track_frame_counts=?, scene_count=?, dwell_time_seconds=?, image_path=?
            WHERE id=?
        ''', (
            car_data['label'], type_id, model_id, color_id, car_data['license_plate'], license_plate_confidence,
            car_data['track_frame_counts'], car_data['scene_count'], car_data['dwell_time_seconds'],
            image_path,
            row[0]
        ))
        conn.commit()
        return row[0]
    else:
        # Create new car
        cur = conn.execute('''
            INSERT INTO CARS (video_id, track_id, label, type_id, model_id, color_id, license_plate, license_plate_confidence,
                            track_frame_counts, scene_count, dwell_time_seconds, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            video_id, car_data['track_id'], car_data['label'], type_id, model_id, color_id,
            car_data['license_plate'], license_plate_confidence, car_data['track_frame_counts'], 
            car_data['scene_count'], car_data['dwell_time_seconds'], image_path
        ))
        conn.commit()
        return cur.lastrowid

def get_or_create_state(conn, state, car_id):
    """Get or create state record"""
    cur = conn.execute('SELECT id FROM STATES WHERE state=? AND car_id=?', (state, car_id))
    row = cur.fetchone()
    if row:
        return row[0]
    cur = conn.execute('INSERT INTO STATES (state, car_id) VALUES (?, ?)', (state, car_id))
    conn.commit()
    return cur.lastrowid

def extract_state_from_plate(plate):
    """Extract state code from license plate"""
    if not plate:
        return None
    match = re.match(r'([A-Z]{2})', plate.upper())
    return match.group(1) if match else None

def store_summary_stats(conn, summary_data, demographics=None):
    """Store summary statistics"""
    import json as _json
    # Use color_distribution from demographics if provided, else fallback
    color_counts = None
    if demographics and 'color_distribution' in demographics:
        color_counts = demographics['color_distribution']
    elif 'color_distribution' in summary_data:
        color_counts = summary_data['color_distribution']
    else:
        color_counts = summary_data.get('all_detected_colors', {})
    conn.execute('''
        INSERT INTO SUMMARY_STATS (total_cars, unique_models, unique_license_plates, average_dwell_time, color_counts_json, peak_hour_range, peak_hour_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        summary_data['total_cars'],
        summary_data['unique_models'],
        summary_data['unique_license_plates'],
        summary_data['average_dwell_time'],
        _json.dumps(color_counts),
        summary_data.get('peak_hour_range'),
        summary_data.get('peak_hour_count')
    ))
    conn.commit()

def main(video_filename=None):
    json_path = 'frontend_data.json'  # Use transformed data
    db_path = 'detections.db'
    if not os.path.exists(json_path):
        print(f"File {json_path} not found. Please run transform.py first to generate transformed data.")
        return
    with open(json_path, 'r') as f:
        transformed_data = json.load(f)
    conn = sqlite3.connect(db_path)
    create_tables(conn)
    # Only clear summary stats, not cars or videos
    conn.execute('DELETE FROM SUMMARY_STATS')
    conn.commit()
    # Insert video row
    if video_filename is None:
        video_filename = 'unknown_video.mp4'
    video_id = get_or_create_video(conn, video_filename)
    # Process each unique car for this video
    for car_data in transformed_data['cars']:
        car_id = get_or_create_car(conn, car_data, video_id)
        # Extract and store state information
        state = extract_state_from_plate(car_data['license_plate'])
        if state:
            get_or_create_state(conn, state, car_id)
    # Store summary statistics, pass demographics for color_distribution
    store_summary_stats(conn, transformed_data['summary'], demographics=transformed_data.get('demographics'))
    conn.close()
    print(f"Processed {len(transformed_data['cars'])} unique cars for video '{video_filename}' into normalized tables in {db_path}.")
    print(f"Summary: {transformed_data['summary']['total_cars']} total cars, "
          f"{transformed_data['summary']['unique_license_plates']} unique license plates")
if __name__ == '__main__':
    import sys
    video_filename = sys.argv[1] if len(sys.argv) > 1 else None
    main(video_filename)
