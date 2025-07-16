import json
import sqlite3
import os
import re

def create_tables(conn):
    conn.execute('''
        CREATE TABLE IF NOT EXISTS CARS (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT,
            type TEXT,
            color TEXT
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS NUMBER_PLATES (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT UNIQUE,
            car_id INTEGER,
            FOREIGN KEY(car_id) REFERENCES CARS(id)
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS STATES (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            state TEXT,
            number_plate_id INTEGER,
            FOREIGN KEY(number_plate_id) REFERENCES NUMBER_PLATES(id)
        )
    ''')
    conn.commit()

def get_or_create_car(conn, model, type_, color):
    color_str = f"rgb({','.join(str(c) for c in color)})" if isinstance(color, list) else str(color)
    cur = conn.execute('SELECT id FROM CARS WHERE model=? AND type=? AND color=?', (model, type_, color_str))
    row = cur.fetchone()
    if row:
        return row[0]
    cur = conn.execute('INSERT INTO CARS (model, type, color) VALUES (?, ?, ?)', (model, type_, color_str))
    conn.commit()
    return cur.lastrowid

def get_or_create_number_plate(conn, plate, car_id):
    cur = conn.execute('SELECT id FROM NUMBER_PLATES WHERE plate=?', (plate,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur = conn.execute('INSERT INTO NUMBER_PLATES (plate, car_id) VALUES (?, ?)', (plate, car_id))
    conn.commit()
    return cur.lastrowid

def get_or_create_state(conn, state, number_plate_id):
    cur = conn.execute('SELECT id FROM STATES WHERE state=? AND number_plate_id=?', (state, number_plate_id))
    row = cur.fetchone()
    if row:
        return row[0]
    cur = conn.execute('INSERT INTO STATES (state, number_plate_id) VALUES (?, ?)', (state, number_plate_id))
    conn.commit()
    return cur.lastrowid

def extract_state_from_plate(plate):
    if not plate:
        return None
    match = re.match(r'([A-Z]{2})', plate.upper())
    return match.group(1) if match else None

def main():
    json_path = 'detection.json'
    db_path = 'detections.db'
    if not os.path.exists(json_path):
        print(f"File {json_path} not found.")
        return
    with open(json_path, 'r') as f:
        detections = json.load(f)
    conn = sqlite3.connect(db_path)
    create_tables(conn)
    # Optional: clear old data
    conn.execute('DELETE FROM STATES')
    conn.execute('DELETE FROM NUMBER_PLATES')
    conn.execute('DELETE FROM CARS')
    conn.commit()
    car_cache = {}
    plate_cache = {}
    for det in detections:
        model = det.get('model')
        type_ = det.get('type')
        color = det.get('color')
        plate = det.get('number_plate')
        if not (model and type_ and color and plate):
            continue
        car_key = (model, type_, tuple(color) if isinstance(color, list) else color)
        if car_key in car_cache:
            car_id = car_cache[car_key]
        else:
            car_id = get_or_create_car(conn, model, type_, color)
            car_cache[car_key] = car_id
        if plate in plate_cache:
            plate_id = plate_cache[plate]
        else:
            plate_id = get_or_create_number_plate(conn, plate, car_id)
            plate_cache[plate] = plate_id
        state = extract_state_from_plate(plate)
        if state:
            get_or_create_state(conn, state, plate_id)
    conn.close()
    print(f"Processed {len(detections)} detections into normalized tables in {db_path}.")

if __name__ == '__main__':
    main() 