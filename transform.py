import json
import numpy as np
from collections import defaultdict, Counter
import colorsys
import datetime

def rgb_to_color_name(rgb):
    """Convert RGB values to a color name"""
    if not rgb or len(rgb) != 3:
        return "unknown"
    
    r, g, b = rgb[0]/255, rgb[1]/255, rgb[2]/255
    
    # Convert to HSV for better color classification
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    
    # Color classification based on HSV values
    if v < 0.2:
        return "black"
    elif v > 0.8 and s < 0.1:
        return "white"
    elif s < 0.1:
        return "gray"
    elif h < 0.05 or h > 0.95:
        return "red"
    elif 0.05 <= h < 0.15:
        return "orange"
    elif 0.15 <= h < 0.25:
        return "yellow"
    elif 0.25 <= h < 0.45:
        return "green"
    elif 0.45 <= h < 0.65:
        return "cyan"
    elif 0.65 <= h < 0.85:
        return "blue"
    elif 0.85 <= h < 0.95:
        return "purple"
    else:
        return "pink"

def get_dominant_color(rgb_list):
    """Get the most common color from a list of RGB values"""
    if not rgb_list:
        return "unknown"
    
    color_names = [rgb_to_color_name(rgb) for rgb in rgb_list if rgb]
    if not color_names:
        return "unknown"
    
    return Counter(color_names).most_common(1)[0][0]

def transform_detections(input_file, output_file, fps=25.0):
    """Transform detection.json to frontend format"""
    
    # Read the detection results
    with open(input_file, 'r') as f:
        detections = json.load(f)
    
    # Group by track_id to get unique cars
    cars_by_track = defaultdict(list)
    for detection in detections:
        track_id = detection['track_id']
        cars_by_track[track_id].append(detection)
    
    # Create unique car objects
    unique_cars = []
    dwell_times = []  # Collect dwell times for summary
    for track_id, car_detections in cars_by_track.items():
        # Calculate statistics for this car
        frame_count = len(car_detections)
        dwell_time = frame_count / fps  # Assuming 25 fps, adjust as needed
        dwell_times.append(dwell_time)
        
        # Find the most common color for this car
        color_counts = Counter([det.get('color', 'unknown') for det in car_detections if det.get('color')])
        color_name = color_counts.most_common(1)[0][0] if color_counts else 'unknown'
        
        # Find the most common license plate and its confidence
        plate_counts = Counter([det.get('number_plate', '') for det in car_detections if det.get('number_plate')])
        best_plate = plate_counts.most_common(1)[0][0] if plate_counts else ''
        # Find the highest confidence for this plate
        best_conf = None
        for det in car_detections:
            if det.get('number_plate') == best_plate:
                conf = det.get('number_plate_confidence')
                if conf is not None and (best_conf is None or conf > best_conf):
                    best_conf = conf
        # Get video name for image path
        video_name = car_detections[0].get('video_name') if 'video_name' in car_detections[0] else None
        if not video_name:
            # Try to infer from label or fallback
            video_name = car_detections[0].get('label', '').split('_')[0] or 'unknown_video'
        image_path = f"videos/processed/{video_name}/car_{track_id}.jpg"
        # Create unique car object
        car_obj = {
            "track_id": track_id,
            "label": f"car{track_id}",
            "model": "unknown",  # Placeholder - you can add model detection later
            "color": color_name,
            "license_plate": best_plate,
            "license_plate_confidence": best_conf,
            "track_frame_counts": frame_count,
            "scene_count": 1,  # Placeholder - you can calculate actual scene count
            "dwell_time_seconds": round(dwell_time, 1),
            "type": "sedan",  # Placeholder - you can add type detection later
            "image_path": image_path
        }
        unique_cars.append(car_obj)
    
    # Calculate demographics
    type_distribution = Counter([car['type'] for car in unique_cars])
    # color_distribution = Counter([car['color'] for car in unique_cars])
    model_distribution = Counter([car['model'] for car in unique_cars])

    # Calculate all detected colors across all frames
    all_detected_colors_counter = Counter()
    for car_detections in cars_by_track.values():
        for det in car_detections:
            color = det.get('color')
            if color:
                all_detected_colors_counter[color] += 1

    # Calculate summary statistics
    total_cars = len(unique_cars)
    unique_models = len(set(car['model'] for car in unique_cars))
    unique_license_plates = len(set(car['license_plate'] for car in unique_cars))
    avg_dwell_time = np.mean(dwell_times) if dwell_times else 0

    # --- New: Aggregate by hour of day ---
    hour_to_track_ids = defaultdict(set)
    for detection in detections:
        ts = detection.get('timestamp')
        if ts:
            try:
                dt = datetime.datetime.fromisoformat(ts)
                hour = dt.hour
                hour_to_track_ids[hour].add(detection['track_id'])
            except Exception:
                continue
    hour_counts = {h: len(s) for h, s in hour_to_track_ids.items()}
    if hour_counts:
        max_count = max(hour_counts.values())
        peak_hours = [h for h, c in hour_counts.items() if c == max_count]
        # Format as a range if consecutive, else as a list
        if len(peak_hours) == 1:
            peak_hour_range = f"{peak_hours[0]:02d}:00-{(peak_hours[0]+1)%24:02d}:00"
        else:
            sorted_hours = sorted(peak_hours)
            # Check if consecutive
            if all(sorted_hours[i]+1 == sorted_hours[i+1] for i in range(len(sorted_hours)-1)):
                peak_hour_range = f"{sorted_hours[0]:02d}:00-{(sorted_hours[-1]+1)%24:02d}:00"
            else:
                peak_hour_range = ','.join(f"{h:02d}:00" for h in sorted_hours)
        peak_hour_count = max_count
    else:
        peak_hour_range = 'NA'
        peak_hour_count = 0

    # Create the final structure
    transformed_data = {
        "cars": unique_cars,
        "summary": {
            "total_cars": total_cars,
            "unique_models": unique_models,
            "unique_license_plates": unique_license_plates,
            "average_dwell_time": round(avg_dwell_time, 1),
            # --- New: Add peak hour info ---
            "peak_hour_range": peak_hour_range,
            "peak_hour_count": peak_hour_count
        },
        "demographics": {
            "type_distribution": dict(type_distribution),
            "color_distribution": dict(all_detected_colors_counter),
            "model_distribution": dict(model_distribution)
        }
    }
    
    # Save the transformed data
    with open(output_file, 'w') as f:
        json.dump(transformed_data, f, indent=2)
    
    print(f"Transformation complete!")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Total unique cars: {total_cars}")
    print(f"Average dwell time: {avg_dwell_time:.1f}s")
    
    # Print license plate detection statistics
    detected_plates = [car['license_plate'] for car in unique_cars if car['license_plate'] and not car['license_plate'].startswith('UNKNOWN')]
    print(f"License plates detected: {len(detected_plates)}/{total_cars}")
    if detected_plates:
        print(f"Detected plates: {', '.join(detected_plates)}")

def main():
    input_file = "detection.json"
    output_file = "frontend_data.json"
    fps = 25.0

    try:
        transform_detections(input_file, output_file, fps)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Please run main.py first to generate detection results.")
    except Exception as e:
        print(f"Error during transformation: {e}")

if __name__ == "__main__":
    main()
