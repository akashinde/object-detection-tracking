import json
import numpy as np
from collections import defaultdict, Counter
import colorsys

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
        
        # Get all colors for this car
        colors = [det['color'] for det in car_detections if det['color']]
        dominant_color = get_dominant_color(colors)
        
        # Get license plate - keep the first valid detection, don't replace with subsequent ones
        license_plate = f"UNKNOWN_{track_id}"  # Default fallback
        
        for det in car_detections:
            if det['number_plate'] and det['number_plate'] != 'None':
                # Clean the license plate text
                plate = det['number_plate'].strip().upper()
                if len(plate) >= 4:  # Minimum length for a valid plate
                    license_plate = plate
                    break  # Use the first valid detection, don't look for more
        
        # Create unique car object
        car_obj = {
            "track_id": track_id,
            "label": f"car{track_id}",
            "model": "unknown",  # Placeholder - you can add model detection later
            "color": dominant_color,
            "license_plate": license_plate,
            "track_frame_counts": frame_count,
            "scene_count": 1,  # Placeholder - you can calculate actual scene count
            "dwell_time_seconds": round(dwell_time, 1),
            "type": "sedan"  # Placeholder - you can add type detection later
        }
        unique_cars.append(car_obj)
    
    # Calculate demographics
    type_distribution = Counter([car['type'] for car in unique_cars])
    color_distribution = Counter([car['color'] for car in unique_cars])
    model_distribution = Counter([car['model'] for car in unique_cars])
    
    # Calculate summary statistics
    total_cars = len(unique_cars)
    unique_models = len(set(car['model'] for car in unique_cars))
    unique_license_plates = len(set(car['license_plate'] for car in unique_cars))
    avg_dwell_time = np.mean(dwell_times) if dwell_times else 0
    
    # Create the final structure
    transformed_data = {
        "cars": unique_cars,
        "summary": {
            "total_cars": total_cars,
            "unique_models": unique_models,
            "unique_license_plates": unique_license_plates,
            "average_dwell_time": round(avg_dwell_time, 1)
        },
        "demographics": {
            "type_distribution": dict(type_distribution),
            "color_distribution": dict(color_distribution),
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
    detected_plates = [car['license_plate'] for car in unique_cars if not car['license_plate'].startswith('UNKNOWN')]
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
