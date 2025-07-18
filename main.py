import os
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import json
import torch
import torchvision.transforms as transforms
from torchvision import models
import subprocess
import argparse
from collections import defaultdict, Counter
import re
from sklearn.cluster import KMeans
import requests
import base64
import io
import time

CAR_TYPE_CLASSES = ['sedan', 'suv', 'hatchback', 'truck', 'van', 'coupe', 'convertible', 'wagon', 'other']

# Load ResNet18 pre-trained model
resnet_model = models.resnet18(pretrained=True)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, len(CAR_TYPE_CLASSES))  # Adjust for car types
resnet_model.eval()  # Set to eval mode

# If you have a trained checkpoint, load it here:
# resnet_model.load_state_dict(torch.load('car_type_resnet.pth', map_location='cpu'))

# Placeholder for car type/model classifier
# from car_classifier import classify_car

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define car color labels with RGB hex mapping
CAR_COLOR_HEX = {
    "white": (245, 245, 245),
    "black": (15, 15, 15),
    "silver": (192, 192, 192),
    "gray": (128, 128, 128),
    "blue": (25, 42, 96),
    "red": (138, 3, 3),
    "green": (0, 86, 63),
    "brown": (101, 67, 33),
    "gold": (212, 175, 55),
    "beige": (245, 245, 220),
    "yellow": (255, 211, 0),
    "orange": (255, 140, 0),
    "maroon": (128, 0, 0),
    "purple": (75, 0, 130),
    "teal": (0, 128, 128),
    "pink": (255, 182, 193),
    "navy": (0, 0, 128),
    "cyan": (0, 255, 255)
}

def closest_named_color(rgb):
    min_dist = float('inf')
    closest_color = None
    for name, ref_rgb in CAR_COLOR_HEX.items():
        dist = np.linalg.norm(np.array(rgb) - np.array(ref_rgb))
        if dist < min_dist:
            min_dist = dist
            closest_color = name
    return closest_color

def extract_color(image, k=3):
    if image is None or image.size == 0:
        return None
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, n_init='auto')
    labels = kmeans.fit_predict(img)
    counts = Counter(labels)
    dominant = kmeans.cluster_centers_[counts.most_common(1)[0][0]]
    dominant = [int(v) for v in dominant]
    return closest_named_color(dominant)

def convert_to_h264(input_path, output_path="output_tracked.mp4"):
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vcodec", "libx264", "-crf", "23", "-preset", "veryfast",
        output_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("FFMPEG Error:", result.stderr.decode())
        raise RuntimeError("FFMPEG conversion to H.264 failed.")
    print(f"H.264 MP4 video written to {output_path}")

def get_video_creation_time(video_path):
    """Try to extract the video creation time using ffprobe. Returns a datetime string or None."""
    import subprocess, json, datetime
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            return None
        info = json.loads(result.stdout.decode())
        # Try to get creation_time from format tags or stream tags
        creation_time = None
        if 'format' in info and 'tags' in info['format']:
            creation_time = info['format']['tags'].get('creation_time')
        if not creation_time and 'streams' in info:
            for stream in info['streams']:
                if 'tags' in stream and 'creation_time' in stream['tags']:
                    creation_time = stream['tags']['creation_time']
                    break
        if creation_time:
            # Normalize to ISO format if possible
            try:
                dt = datetime.datetime.fromisoformat(creation_time.replace('Z', '+00:00'))
                return dt.isoformat()
            except Exception:
                return creation_time
        return None
    except Exception:
        return None

def detect_and_track(video_path):
    # Load models
    detector = YOLO('yolov8n.pt')  # Use YOLOv8 nano for speed; replace with better weights if needed
    tracker = DeepSort(max_age=30)
    ocr_reader = easyocr.Reader(['en'])
    filename = os.path.basename(video_path)
    video_name = os.path.splitext(filename)[0]
    base_output_dir = os.path.join('videos', 'processed')
    output_dir = os.path.join(base_output_dir, video_name)
    suffix = 1
    while os.path.exists(output_dir):
        output_dir = os.path.join(base_output_dir, f"{video_name}_{suffix}")
        suffix += 1
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Use OpenCV's VideoWriter_fourcc (modern OpenCV >= 3.x)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    temp_avi_path = os.path.join(output_dir, 'temp_output.avi')
    out = cv2.VideoWriter(temp_avi_path, fourcc, fps, (width, height))

    # --- New: Extract video creation time ---
    video_capture_time = get_video_creation_time(video_path)
    if video_capture_time is None:
        # Fallback: use file modification time
        import datetime
        ts = os.path.getmtime(video_path)
        video_capture_time = datetime.datetime.fromtimestamp(ts).isoformat()

    results = []
    plate_history = defaultdict(list)  # track_id -> list of (plate, confidence)
    first_car_image_saved = set()  # track_ids for which image is saved
    car_plate_found = {}  # track_id -> bool (whether plate image has been saved)
    frame_idx = 0
    # Store car image paths for each track_id
    car_image_paths = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        original_frame = frame.copy()  # Save a clean copy for cropping
        detections = detector(frame)[0]
        car_detections = []
        for det in detections.boxes:
            cls = int(det.cls[0])
            if detector.names[cls] == 'car':
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                conf = float(det.conf[0])
                if conf < 0.5:
                    continue
                car_detections.append(([x1, y1, x2-x1, y2-y1], conf, 'car'))
        tracks = tracker.update_tracks(car_detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            car_img = frame[y1:y2, x1:x2]
            # Use original_frame for saving cropped car image (no overlays)
            car_img_clean = original_frame[y1:y2, x1:x2]
            color = extract_color(car_img)
            car_label = f'car{track_id}'
            car_img_path = os.path.join(output_dir, f'car_{track_id}.jpg')
            # Save car image only the first time for this track_id
            if track_id not in first_car_image_saved and car_img_clean.size > 0:
                cv2.imwrite(car_img_path, car_img_clean)
                first_car_image_saved.add(track_id)
                car_image_paths[track_id] = car_img_path
            bbox_list = [int(x) for x in [x1, y1, x2, y2]]
            color_str = color if color is not None else None
            # --- New: Calculate timestamp for this frame ---
            import datetime
            try:
                base_dt = datetime.datetime.fromisoformat(video_capture_time)
                timestamp = (base_dt + datetime.timedelta(seconds=frame_idx / fps)).isoformat()
            except Exception:
                timestamp = video_capture_time  # fallback
            results.append({
                'frame': frame_idx,
                'track_id': int(track_id),
                'bbox': bbox_list,
                'color': color_str,
                'label': car_label,
                'timestamp': timestamp
            })
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{car_label}'
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()

    final_mp4_path = os.path.join(output_dir, f'{video_name}_tracked.mp4')
    convert_to_h264(temp_avi_path, final_mp4_path)
    os.remove(temp_avi_path)

    # --- Plate Recognizer API: Call once per car using best image ---
    track_id_to_api_result = {}
    for track_id, img_path in car_image_paths.items():
        car_img = cv2.imread(img_path)
        api_result = call_plate_recognizer_api(car_img)
        number_plate = None
        number_plate_confidence = None
        if api_result and api_result.get('results'):
            res = api_result['results'][0]
            number_plate = res.get('plate')
            number_plate_confidence = res.get('score')
        # If no plate found, set to empty string
        if not number_plate:
            number_plate = ''
        track_id_to_api_result[str(track_id)] = {
            'number_plate': number_plate,
            'number_plate_confidence': number_plate_confidence
        }
        time.sleep(1)

    # Assign API results to all frames for each track_id
    for r in results:
        tid = str(r['track_id'])
        api_res = track_id_to_api_result.get(tid, {})
        r['number_plate'] = api_res.get('number_plate')
        r['number_plate_confidence'] = api_res.get('number_plate_confidence')

    return results

def detect_number_plate_from_frame(car_img, ocr_reader):
    # Convert the car image to grayscale (optional but may help OCR)
    gray = cv2.cvtColor(car_img, cv2.COLOR_BGR2GRAY)
    result = ocr_reader.readtext(gray)
    plates = []
    best_plate = None
    best_conf = 0.0
    for (bbox, text, prob) in result:
        if prob > 0.5 and len(text) >= 5:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            cv2.rectangle(car_img, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(car_img, text, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            plates.append((text, prob))
            if prob > best_conf:
                best_plate = text
                best_conf = prob
    return car_img, best_plate, best_conf if best_plate else (car_img, None, None)

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

PLATE_RECOGNIZER_TOKEN = os.getenv("PLATE_RECOGNIZER_TOKEN")  # Set your token as an env variable

def call_plate_recognizer_api(car_img):
    # Encode image to bytes
    _, img_encoded = cv2.imencode('.jpg', car_img)
    img_bytes = img_encoded.tobytes()
    files = {'upload': ('car.jpg', img_bytes, 'image/jpeg')}
    data = {
        'mmc': 'true',  # get make/model/color
        'detection_mode': 'vehicle'
        # 'regions': 'by'  # Belarus, optional
    }
    headers = {'Authorization': f'Token 1ceed4f85893d80f7afbba83b5bdb5764dad74e1'}
    response = requests.post(
        'https://api.platerecognizer.com/v1/plate-reader/',
        data=data,
        files=files,
        headers=headers
    )
    if response.status_code == 201:
        return response.json()
    else:
        print("Plate Recognizer API error:", response.text)
        return None

def main():
    parser = argparse.ArgumentParser(description="Detect and track cars in a video.")
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file')
    args = parser.parse_args()
    video = args.video
    output = "detection.json"
    results = detect_and_track(video)
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {output}')
    # Save the processed video filename for downstream use
    processed_video_filename = os.path.basename(video)
    with open('processed_video.txt', 'w') as f:
        f.write(processed_video_filename)

if __name__ == '__main__':
    main()