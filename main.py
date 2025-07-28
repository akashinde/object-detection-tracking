import os
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import json
import torch
import subprocess
import argparse
import base64
from openai import OpenAI
import time
import redis

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

def update_progress(job_id, progress, status="processing", message=""):
    """Update progress in Redis"""
    if not job_id:
        return
    
    try:
        redis_conn = redis.Redis()
        progress_data = {
            "progress": progress,
            "status": status,
            "message": message,
            "timestamp": time.time()
        }
        redis_conn.setex(f"progress:{job_id}", 3600, json.dumps(progress_data))  # Expire in 1 hour
    except Exception as e:
        print(f"Error updating progress: {e}")

def call_openai_car_analysis(car_img_path):
    """
    Sends the cropped car image to OpenAI API and asks for logo, number plate, make, model, color, car type, and their confidences.
    Returns a dict with the results and always includes the raw OpenAI response.
    Handles OpenAI's markdown-wrapped JSON output.
    """
    import time
    import re
    api_key = os.environ.get("OPENAI_API")
    client = OpenAI(api_key=api_key)
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    base64_image = encode_image(car_img_path)
    prompt = (
        "You are an expert vehicle analyst. "
        "Given the following car image, detect and return standardized information. "
        "IMPORTANT: Use ONLY the exact standardized names listed below. Do not add extra words, descriptions, or variations. "
        "If any information is not visible or cannot be determined, use Unknown as value. "
        ""
        "BRAND/MAKE STANDARDIZED NAMES (use exactly these): "
        "Volkswagen, Toyota, BMW, Mercedes-Benz, Audi, Ford, Honda, Hyundai, Nissan, Mazda, Kia, "
        "Chevrolet, Jeep, Subaru, Lexus, Volvo, Porsche, Ferrari, Lamborghini, Tesla, "
        "Dacia, Renault, Peugeot, Citroën, Fiat, Alfa Romeo, Skoda, Seat, Opel, "
        "Jaguar, Land Rover, Mini, Smart, Mitsubishi, Suzuki, Daihatsu, "
        "Chrysler, Dodge, Buick, Cadillac, Lincoln, Pontiac, Saturn, "
        "Infiniti, Acura, Genesis, Scion, Saab, Lancia, Maserati, "
        "Bentley, Rolls-Royce, Aston Martin, McLaren, Bugatti, Koenigsegg, "
        "Lotus, Caterham, Morgan, TVR, Noble, Pagani, Rimac, "
        "McLaren, Alpina, Brabus, AMG, M, RS, S, GT, ST, Type-R, "
        "GTR, Nismo, TRD, Mugen, Spoon, HKS, Blitz, "
        "RUF, Singer, Gunther Werks, Liberty Walk, Rocket Bunny, "
        "Pandem, Veilside, RE Amemiya, JUN, Top Secret, "
        "HPA, APR, Unitronic, GIAC, Revo, Cobb, "
        "Unknown (only if completely uncertain)"
        ""
        "COLOR STANDARDIZED NAMES (use exactly these, lowercase): "
        "white, black, silver, gray, red, blue, green, yellow, orange, purple, pink, brown, "
        "gold, navy, maroon, olive, lime, aqua, teal, fuchsia, beige, cream, tan, "
        "bronze, copper, chrome, pearl, metallic, matte, gloss, satin, "
        "crimson, burgundy, emerald, turquoise, indigo, violet, magenta, "
        "coral, salmon, peach, lavender, mint, sage, rust, "
        "charcoal, slate, stone, sand, khaki, camel, taupe, "
        "Unknown (only if completely uncertain)"
        ""
        "CAR TYPE STANDARDIZED NAMES (use exactly these): "
        "sedan, suv, hatchback, truck, van, coupe, convertible, wagon, "
        "pickup, minivan, crossover, sports car, luxury car, compact, "
        "subcompact, midsize, fullsize, executive, limousine, "
        "roadster, targa, shooting brake, fastback, notchback, "
        "liftback, estate, touring, grand tourer, supercar, hypercar, "
        "muscle car, pony car, hot hatch, sleeper, tuner car, "
        "rally car, drift car, drag car, track car, race car, "
        "Unknown (only if completely uncertain)"
        ""
        "MODEL NAMES: Use the most common/recognizable model name without extra words. "
        "Examples: 'Civic' not 'Honda Civic', 'Golf' not 'Volkswagen Golf', 'Camry' not 'Toyota Camry' "
        "If model cannot be determined, use null."
        ""
        "Return the answer as a JSON object with keys: "
        "logo, logo_confidence, number_plate, number_plate_confidence, make, make_confidence, "
        "model, model_confidence, color, color_confidence, car_type, car_type_confidence. "
        ""
        "CONFIDENCE SCORES: Use values between 0.0 and 1.0 (as floats). "
        "Use 0.0 for null values or when completely uncertain. "
        "Use higher values (0.7-1.0) when very confident. "
        ""
        "EXAMPLE RESPONSE: "
        '{"logo": "Toyota", "logo_confidence": 0.9, "number_plate": Unknown, "number_plate_confidence": 0.0, '
        '"make": "Toyota", "make_confidence": 0.95, "model": "Camry", "model_confidence": 0.8, '
        '"color": "silver", "color_confidence": 0.85, "car_type": "sedan", "car_type_confidence": 0.9}'
    )
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model="gpt-4.1",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"},
                        ],
                    }
                ],
                timeout=60  # seconds
            )
            import json as pyjson
            raw = response.output_text
            cleaned = re.sub(r'^```json|^```|```$', '', raw.strip(), flags=re.MULTILINE).strip()
            cleaned = cleaned.strip('`').strip()
            try:
                result = pyjson.loads(cleaned)
                result['raw_response'] = raw
                return result
            except Exception:
                if attempt == max_retries - 1:
                    print(f"[OpenAI API] Failed to parse JSON after {max_retries} attempts. Returning raw response.")
                    return {"raw_response": raw}
                else:
                    print(f"[OpenAI API] JSON parse error, retrying ({attempt+1}/{max_retries})...")
                    time.sleep(2)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"OpenAI API error after {max_retries} attempts: {e}")
                return {"raw_response": str(e)}
            else:
                print(f"OpenAI API error, retrying ({attempt+1}/{max_retries}): {e}")
                time.sleep(2)

def detect_and_track(video_path, job_id=None):
    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    update_progress(job_id, 15, "processing", "Loading YOLO detection model...")
    detector = YOLO('yolov8n.pt').to(device)  # Use YOLOv8 nano for speed; replace with better weights if needed
    
    update_progress(job_id, 20, "processing", "Loading YOLO segmentation model...")
    seg_model = YOLO('yolov8n-seg.pt').to(device)  # Load YOLOv8 segmentation model
    
    update_progress(job_id, 25, "processing", "Initializing DeepSORT tracker...")
    tracker = DeepSort(max_age=30)
    
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
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
    frame_idx = 0
    # Store car image paths for each track_id
    car_image_paths = {}
    best_visible_per_track = {}  # track_id -> (percent_visible, img_path)
    
    update_progress(job_id, 30, "processing", "Starting video frame processing...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Update progress based on frame processing
        if total_frames > 0:
            frame_progress = 30 + (frame_idx / total_frames) * 45  # 30% to 75%
            if frame_idx % 30 == 0:  # Update every 30 frames to avoid too many Redis calls
                update_progress(job_id, int(frame_progress), "processing", f"Processing frame {frame_idx}/{total_frames}...")
        
        original_frame = frame.copy()  # Save a clean copy for cropping
        detections = detector(frame)[0]
        # --- Run segmentation model on this frame ---
        seg_results = seg_model(frame)[0]
        seg_masks = []
        for i, det in enumerate(seg_results.boxes):
            cls = int(det.cls[0])
            if seg_model.names[cls] == 'car':
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                conf = float(det.conf[0])
                if conf < 0.5:
                    continue
                # Get mask for this detection
                if hasattr(seg_results, 'masks') and seg_results.masks is not None:
                    mask = seg_results.masks.data[i].cpu().numpy()  # shape: (H, W)
                    seg_masks.append({'bbox': [x1, y1, x2, y2], 'mask': mask})
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
            car_label = f'car{track_id}'
            car_img_path = os.path.join(output_dir, f'car_{track_id}.jpg')
            bbox_list = [int(x) for x in [x1, y1, x2, y2]]
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
                'label': car_label,
                'timestamp': timestamp,
            })
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{car_label}'
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            # Find best mask for this car bbox by IoU
            best_iou = 0
            best_mask_poly = None
            for i, seg in enumerate(seg_masks):
                sx1, sy1, sx2, sy2 = seg['bbox']
                # Compute IoU between car bbox and seg bbox
                xx1 = max(x1, sx1)
                yy1 = max(y1, sy1)
                xx2 = min(x2, sx2)
                yy2 = min(y2, sy2)
                iw = max(0, xx2 - xx1)
                ih = max(0, yy2 - yy1)
                inter = iw * ih
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (sx2 - sx1) * (sy2 - sy1)
                union = area1 + area2 - inter
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_mask_poly = seg_results.masks.xy[i] if hasattr(seg_results, 'masks') and seg_results.masks is not None else None
            # Calculate percent visible
            percent_visible = 0.0
            if best_mask_poly is not None and len(best_mask_poly) > 0:
                poly = np.array(best_mask_poly, dtype=np.int32)
                # Shift polygon to crop-relative coordinates
                poly_crop = poly.copy()
                poly_crop[:, 0] -= x1
                poly_crop[:, 1] -= y1
                mask_area = cv2.contourArea(poly_crop)
                crop_area = car_img_clean.shape[0] * car_img_clean.shape[1]
                percent_visible = (mask_area / crop_area) if crop_area > 0 else 0.0
            else:
                ix1 = max(0, x1)
                iy1 = max(0, y1)
                ix2 = min(width, x2)
                iy2 = min(height, y2)
                inter_w = max(0, ix2 - ix1)
                inter_h = max(0, iy2 - iy1)
                inter_area = inter_w * inter_h
                bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
                percent_visible = (inter_area / bbox_area) if bbox_area > 0 else 0.0
            # Save/overwrite only if this is the best visible for this track
            if car_img_clean.size > 0 and (track_id not in best_visible_per_track or percent_visible > best_visible_per_track[track_id][0]):
                cv2.imwrite(car_img_path, car_img_clean)
                best_visible_per_track[track_id] = (percent_visible, car_img_path)
                car_image_paths[track_id] = car_img_path  # Ensure this always points to the best crop
                # --- Segmentation mask overlay using YOLOv8-seg polygons ---
                if best_mask_poly is not None and len(best_mask_poly) > 0:
                    poly = np.array(best_mask_poly, dtype=np.int32)
                    poly[:, 0] -= x1
                    poly[:, 1] -= y1
                    seg_img = car_img_clean.copy()
                    cv2.fillPoly(seg_img, [poly], (0, 0, 255))  # Red overlay
                    alpha = 0.4
                    overlay = cv2.addWeighted(car_img_clean, 1 - alpha, seg_img, alpha, 0)
                else:
                    mask = np.zeros(car_img_clean.shape[:2], dtype=np.uint8)
                    mask[:, :] = 255
                    seg_img = car_img_clean.copy()
                    seg_img[mask == 255] = [0, 0, 255]
                    alpha = 0.4
                    overlay = cv2.addWeighted(car_img_clean, 1 - alpha, seg_img, alpha, 0)
                seg_filename = f'car_{track_id}_seg.jpg'
                seg_path = os.path.join(output_dir, seg_filename)
                cv2.imwrite(seg_path, overlay)
        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()

    update_progress(job_id, 75, "processing", "Converting video to H.264 format...")
    final_mp4_path = os.path.join(output_dir, f'{video_name}_tracked.mp4')
    convert_to_h264(temp_avi_path, final_mp4_path)
    os.remove(temp_avi_path)

    # --- OpenAI API: Call once per car using best image ---
    track_id_to_openai_result = {}
    print("[OpenAI API] Starting car analysis...")
    update_progress(job_id, 78, "processing", "Analyzing car images with AI...")
    
    total_cars = len(car_image_paths)
    for idx, (track_id, img_path) in enumerate(car_image_paths.items()):
        openai_result = call_openai_car_analysis(img_path)
        print(f"[OpenAI API] Track ID {track_id} analysis result: {openai_result}")
        track_id_to_openai_result[str(track_id)] = openai_result
        
        # Update progress for AI analysis
        ai_progress = 78 + (idx / total_cars) * 2  # 78% to 80%
        update_progress(job_id, int(ai_progress), "processing", f"AI analysis: {idx+1}/{total_cars} cars...")
        
        time.sleep(1)  # avoid rate limits

    # Assign OpenAI results to all frames for each track_id
    for r in results:
        tid = str(r['track_id'])
        openai_res = track_id_to_openai_result.get(tid, {})
        r['logo'] = openai_res.get('logo') if openai_res else None
        r['logo_confidence'] = openai_res.get('logo_confidence') if openai_res else None
        r['number_plate'] = openai_res.get('number_plate') if openai_res else None
        r['number_plate_confidence'] = openai_res.get('number_plate_confidence') if openai_res else None
        r['make'] = openai_res.get('make') if openai_res else None
        r['make_confidence'] = openai_res.get('make_confidence') if openai_res else None
        r['model'] = openai_res.get('model') if openai_res else None
        r['model_confidence'] = openai_res.get('model_confidence') if openai_res else None
        r['color'] = openai_res.get('color') if openai_res else None
        r['color_confidence'] = openai_res.get('color_confidence') if openai_res else None
        r['car_type'] = openai_res.get('car_type') if openai_res else None
        r['car_type_confidence'] = openai_res.get('car_type_confidence') if openai_res else None
        r['openai_raw_response'] = openai_res.get('raw_response') if openai_res else None

    return results, car_image_paths, final_mp4_path, fps

def run_detection(video_path, job_id=None):
    """
    Run detection + tracking on a video and return:
      • detections:  list[dict]   raw per-frame objects
      • meta:        dict         {car_image_paths, video_filepath, fps, ...}
    No writes to disk except the tracked MP4 + cropped JPEGs produced
    internally by detect_and_track().
    """
    start_time = time.time()
    detections, car_image_paths, final_mp4_path, fps = detect_and_track(video_path, job_id)
    end_time = time.time()
    avg_processing_time = end_time - start_time

    # Prepare metadata object
    meta = {
        "car_image_paths": car_image_paths,  # dict: track_id -> image path
        "video_filepath": final_mp4_path,
        "avg_processing_time": avg_processing_time,
        "fps": fps
    }

    return detections, meta


# --- CLI helper for ad‑hoc testing (optional) ------------------------------
if __name__ == "__main__":                 # pragma: no cover
    import argparse, json, pprint, textwrap
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    args = ap.parse_args()
    det, meta = run_detection(args.video)
    print("▶︎ detections =", len(det))
    print("▶︎ meta =")
    pprint.pprint(meta, indent=2, width=120)