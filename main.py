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
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    temp_avi_path = os.path.join(output_dir, 'temp_output.avi')
    out = cv2.VideoWriter(temp_avi_path, fourcc, fps, (width, height))

    results = []
    plate_history = defaultdict(list)  # track_id -> list of (plate, confidence)
    first_car_image_saved = set()  # track_ids for which image is saved
    car_plate_found = {}  # track_id -> bool (whether plate image has been saved)
    frame_idx = 0
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
            _, number_plate, number_plate_confidence = detect_number_plate_from_frame(car_img, ocr_reader)
            car_label = f'car{track_id}'
            car_img_path = os.path.join(output_dir, f'car_{track_id}.jpg')
            # Save/replace car image logic
            plate_conf_val = None
            if isinstance(number_plate_confidence, (list, tuple, np.ndarray)):
                try:
                    plate_conf_val = float(number_plate_confidence[0])
                except Exception:
                    plate_conf_val = None
            elif number_plate_confidence is not None:
                try:
                    plate_conf_val = float(number_plate_confidence)
                except Exception:
                    plate_conf_val = None
            plate_visible = number_plate and plate_conf_val is not None and plate_conf_val > 0.5
            if track_id not in first_car_image_saved and car_img_clean.size > 0:
                cv2.imwrite(car_img_path, car_img_clean)
                first_car_image_saved.add(track_id)
                car_plate_found[track_id] = plate_visible
            elif plate_visible and car_img_clean.size > 0 and not car_plate_found.get(track_id, False):
                # Replace with image where plate is visible
                cv2.imwrite(car_img_path, car_img_clean)
                car_plate_found[track_id] = True
            # Ensure bbox and color are lists, not numpy arrays
            bbox_list = [int(x) for x in [x1, y1, x2, y2]]
            color_str = color if color is not None else None
            # Ensure number_plate_confidence is a float or None
            if isinstance(number_plate_confidence, (list, tuple, np.ndarray)):
                try:
                    number_plate_confidence = float(number_plate_confidence[0])
                except Exception:
                    number_plate_confidence = None
            elif number_plate_confidence is not None:
                number_plate_confidence = float(number_plate_confidence)
            else:
                number_plate_confidence = None
            # Save plate and confidence for aggregation
            if number_plate:
                plate_history[track_id].append((number_plate, number_plate_confidence))
            results.append({
                'frame': frame_idx,
                'track_id': int(track_id),
                'bbox': bbox_list,
                'color': color_str,
                # 'number_plate': number_plate,  # Remove per-frame plate
                # 'number_plate_confidence': number_plate_confidence,  # Remove per-frame confidence
                'label': car_label
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
    # Optionally delete the temp file
    os.remove(temp_avi_path)

    # Aggregate number plates for each track_id
    plate_regex = re.compile(r'^[A-Z0-9\-]{5,12}$')  # Allow hyphens, 5-12 chars
    track_id_to_plate = {}
    for tid, plates in plate_history.items():
        # Clean and filter plates
        cleaned = []
        for p, c in plates:
            if p and c is not None:
                p_clean = p.strip().replace(' ', '').upper()  # Keep hyphens
                if plate_regex.match(p_clean):
                    cleaned.append((p_clean, c))
        if not cleaned:
            track_id_to_plate[tid] = (None, None)
            continue
        # Levenshtein clustering: merge similar plates (distance <= 2)
        clusters = []
        for plate, conf in cleaned:
            found_cluster = False
            for cluster in clusters:
                if levenshtein(plate, cluster['plate']) <= 2:
                    cluster['votes'].append(conf)
                    cluster['conf_sum'] += conf
                    cluster['count'] += 1
                    found_cluster = True
                    break
            if not found_cluster:
                clusters.append({'plate': plate, 'votes': [conf], 'conf_sum': conf, 'count': 1})
        # Confidence-weighted voting
        best_cluster = max(clusters, key=lambda cl: (cl['conf_sum'], cl['count']))
        best_plate = best_cluster['plate']
        best_conf = np.mean(best_cluster['votes'])
        track_id_to_plate[tid] = (best_plate, float(best_conf))

    # Assign aggregated plate/confidence to each result using track_id_to_plate
    for r in results:
        tid = str(r['track_id'])
        plate, conf = track_id_to_plate.get(tid, (None, None))
        r['number_plate'] = plate
        r['number_plate_confidence'] = conf

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