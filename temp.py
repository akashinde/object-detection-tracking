import os
import json

json_path = 'detection.json'

with open(json_path, 'r') as f:
    detections = json.load(f)

data = {}

for item in detections:
    track_id = item['track_id']
    number_plate = item['number_plate']
    car_type = item['type']
    print(track_id, number_plate, car_type)
    # skip if track_id is already in data
    if track_id in data:
        continue
    if track_id not in data:
        data[track_id] = {
            'number_plate': number_plate,
            'type': car_type
        }

print(data)