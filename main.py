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

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('temp_output.avi', fourcc, fps, (width, height))

    results = []
    frame_idx = 0
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
            color = extract_color(car_img)
            number_plate = extract_number_plate(car_img, ocr_reader, bbox=[x1, y1, x2, y2], frame_height=height)
            car_label = f'car{track_id}'
            results.append({
                'frame': frame_idx,
                'track_id': track_id,
                'bbox': [x1, y1, x2, y2],
                'color': color,
                'number_plate': number_plate,
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

    temp_avi_path = "temp_output.avi"
    final_mp4_path = "videos/processed/"+ filename.split('.')[0] + "_tracked.mp4"

    convert_to_h264(temp_avi_path, final_mp4_path)

    # Optionally delete the temp file
    os.remove(temp_avi_path)

    return results

def extract_color(car_img):
    # Simple dominant color extraction
    if car_img.size == 0:
        return None
    avg_color = car_img.mean(axis=(0, 1))
    return [int(c) for c in avg_color]

def extract_number_plate(car_img, ocr_reader, bbox=None, frame_height=None):
    # Check if car_img is valid
    if car_img is None or car_img.size == 0:
        return None
    
    # Calculate car size relative to frame height for distance estimation
    car_height = car_img.shape[0]
    car_width = car_img.shape[1]
    
    # Distance estimation based on car size
    # Larger cars in frame = closer to camera
    if frame_height and bbox:
        car_area = car_width * car_height
        frame_area = frame_height * frame_height  # Approximate frame area
        car_ratio = car_area / frame_area
        
        # If car is too small (far away), skip OCR
        if car_ratio < 0.01:  # Car takes less than 1% of frame
            return None
    
    # Use EasyOCR to read number plate
    result = ocr_reader.readtext(car_img)
    
    if not result:
        return None
    
    # Filter results based on confidence and text characteristics
    valid_plates = []
    for (bbox_coords, text, confidence) in result:
        # Only consider high confidence results
        if confidence < 0.7:  # High confidence threshold
            continue
        
        # Clean and validate the text
        cleaned_text = text.strip().upper()
        
        # Basic license plate validation patterns
        # Common patterns: ABC1234, ABC-1234, ABC 1234, etc.
        import re
        
        # Pattern for alphanumeric license plates
        plate_pattern = re.compile(r'^[A-Z0-9\s\-\.]{3,12}$')
        
        # Additional validation: should contain both letters and numbers
        has_letters = any(c.isalpha() for c in cleaned_text)
        has_numbers = any(c.isdigit() for c in cleaned_text)
        
        if (plate_pattern.match(cleaned_text) and 
            has_letters and has_numbers and
            len(cleaned_text) >= 4):  # Minimum length
            
            valid_plates.append((cleaned_text, confidence))
    
    # Return the highest confidence valid plate
    if valid_plates:
        # Sort by confidence and return the best match
        valid_plates.sort(key=lambda x: x[1], reverse=True)
        return valid_plates[0][0]
    
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