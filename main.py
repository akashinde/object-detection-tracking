import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import json
import torch
import torchvision.transforms as transforms
from torchvision import models
from collections import defaultdict

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

CONFIDENCE_THRESHOLD = 0.5  # You can adjust this

def detect_and_track(video_path):
    # Load models
    detector = YOLO('yolov8n.pt')  # Use YOLOv8 nano for speed; replace with better weights if needed
    tracker = DeepSort(max_age=30)
    ocr_reader = easyocr.Reader(['en'])

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_tracked.mp4', fourcc, fps, (width, height))

    results = []
    frame_idx = 0
    track_frame_counts = defaultdict(int)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector(frame)[0]
        car_detections = []
        for det in detections.boxes:
            cls = int(det.cls[0])
            if detector.names[cls] == 'car':
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                conf = float(det.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                car_detections.append(([x1, y1, x2-x1, y2-y1], conf, 'car'))
        tracks = tracker.update_tracks(car_detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            track_frame_counts[track_id] += 1
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            car_img = frame[y1:y2, x1:x2]
            color = extract_color(car_img)
            number_plate = extract_number_plate(car_img, ocr_reader)
            car_type, car_model = classify_car_type_model(car_img)
            results.append({
                'frame': frame_idx,
                'track_id': track_id,
                'bbox': [x1, y1, x2, y2],
                'color': color,
                'number_plate': number_plate,
                'type': car_type,
                'model': car_model
            })
            # Draw bounding box
            if track_frame_counts[track_id] >= int(fps): # Only draw if track has been confirmed for at least 1 second
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'ID:{track_id} {car_type}'
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()

    min_track_frames = int(fps)  # 1 second, or use 2*fps for 2 seconds

    valid_track_ids = {tid for tid, count in track_frame_counts.items() if count >= min_track_frames}
    filtered_results = [r for r in results if r['track_id'] in valid_track_ids]

    return filtered_results

def extract_color(car_img):
    # Simple dominant color extraction
    if car_img.size == 0:
        return None
    avg_color = car_img.mean(axis=(0, 1))
    return [int(c) for c in avg_color]

def extract_number_plate(car_img, ocr_reader):
    # Check if car_img is valid
    if car_img is None or car_img.size == 0:
        return None
    # Use EasyOCR to read number plate
    result = ocr_reader.readtext(car_img)
    if result:
        return result[0][1]
    return None

def classify_car_type_model(car_img):
    # Check for empty image
    if car_img is None or car_img.size == 0:
        return 'unknown', 'unknown'
    try:
        input_tensor = preprocess(car_img)
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = resnet_model(input_batch)
            _, pred = torch.max(output, 1)
            car_type = CAR_TYPE_CLASSES[int(pred.item())]
        return car_type, 'unknown'  # Model prediction, model is unknown
    except Exception as e:
        print(f"Car type classification error: {e}")
        return 'unknown', 'unknown'

def main():
    video = "video_road.mp4"
    output = "detection.json"
    results = detect_and_track(video)
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {output}')

if __name__ == '__main__':
    main() 