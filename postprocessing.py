from collections import defaultdict, Counter
import re
import os

def normalize_file_path(file_path: str) -> str:
    """
    Convert file path to generalized format regardless of OS.
    Converts Windows backslashes to forward slashes and extracts relative path.
    """
    if not file_path:
        return "unknown.mp4"
    
    # Normalize path separators
    normalized = file_path.replace('\\', '/')
    
    # Extract just the filename if it's a full path
    filename = os.path.basename(normalized)
    
    # If it's a processed video path, extract the folder structure
    if 'videos/processed/' in normalized:
        # Extract the part after 'videos/processed/'
        parts = normalized.split('videos/processed/')
        if len(parts) > 1:
            return parts[1]  # Returns 'folder_name/filename.mp4'
    
    # For other cases, just return the filename
    return filename

def _infer_region(plate: str) -> str | None:
    if not plate or plate.lower().startswith("not"):
        return None
    plate_up = plate.upper()
    if re.search(r'\bCA\b', plate_up):
        return "California"
    if re.search(r'\bTX\b', plate_up):
        return "Texas"
    if re.search(r'[A-Z]{2}\d{2}\s?[A-Z]{2}', plate_up):
        return "UK-GB"
    return "Unknown"

def transform_detections_from_obj(detections: list[dict], meta: dict) -> dict:
    cars_by_track = defaultdict(list)
    for d in detections:
        cars_by_track[d["track_id"]].append(d)

    fps = meta.get("fps", 25.0)
    video_file = normalize_file_path(meta.get("video_filepath", "unknown.mp4"))
    source_video = meta.get("source_video", "local")

    color_ctr = Counter()
    brand_ctr = Counter()
    model_ctr = Counter()
    region_ctr = Counter()
    plates_seen = set()
    vehicles = []
    max_frame = 0
    activity_by_second = Counter()

    for track_id, det_list in cars_by_track.items():
        det_list.sort(key=lambda d: d["frame"])
        first_det, last_det = det_list[0], det_list[-1]
        first_f, last_f = first_det["frame"], last_det["frame"]
        max_frame = max(max_frame, last_f)
        dwell_time_sec = (last_f - first_f + 1) / fps

        plate = first_det.get("number_plate", "unknown")
        color = first_det.get("color", "unknown")
        brand = first_det.get("make", "unknown")
        model = first_det.get("model", "unknown")
        logo = first_det.get("logo", brand)
        region = _infer_region(plate)

        if color != "unknown":
            color_ctr[color] += 1
        if brand != "unknown":
            brand_ctr[brand] += 1
        if model != "unknown":
            model_ctr[(brand, model)] += 1
        if region and region != "Unknown":
            region_ctr[region] += 1
        if plate and not plate.lower().startswith("not"):
            plates_seen.add(plate)

        for t in range(int(first_f / fps), int(last_f / fps) + 1):
            activity_by_second[t] += 1

        vehicles.append({
            "track_id": track_id,
            "first_seen_sec": first_f / fps,
            "last_seen_sec": last_f / fps,
            "dwell_time_seconds": dwell_time_sec,
            "type": first_det.get("car_type", "unknown"),
            "type_confidence": first_det.get("car_type_confidence"),
            "color": color,
            "color_confidence": first_det.get("color_confidence"),
            "brand": brand,
            "make_confidence": first_det.get("make_confidence"),
            "model": model,
            "model_confidence": first_det.get("model_confidence"),
            "logo": logo,
            "logo_confidence": first_det.get("logo_confidence"),
            "license_plate": plate,
            "license_plate_confidence": first_det.get("number_plate_confidence"),
            "license_region": region,
            "image_path": meta.get("car_image_paths", {}).get(str(track_id)),
            "is_moving": len(det_list) > 1 and dwell_time_sec > 0.5,
            "source_video": source_video
        })

    duration_sec = max_frame / fps if max_frame else None
    total_car_seconds = sum(v["dwell_time_seconds"] for v in vehicles)
    avg_cars_per_second = round(total_car_seconds / duration_sec, 3) if duration_sec else None
    car_visibility_percent = round((total_car_seconds / duration_sec) * 100, 2) if duration_sec else None
    average_confidence = round(sum(v.get("make_confidence", 0.0) or 0 for v in vehicles) / len(vehicles), 3) if vehicles else None

    if activity_by_second:
        best_sec = max(activity_by_second, key=activity_by_second.get)
        most_active_segment = {
            "startTimeSec": best_sec,
            "endTimeSec": best_sec + 10,
            "avgCarsPerFrame": round(sum(activity_by_second[s] for s in range(best_sec, best_sec + 10)) / 10, 2)
        }
    else:
        most_active_segment = {
            "startTimeSec": None,
            "endTimeSec": None,
            "avgCarsPerFrame": None
        }

    video_summary = {
        "total_cars": len(vehicles),
        "distinct_plates": len(plates_seen),
        "duration_sec": duration_sec,
        "avg_cars_per_second": avg_cars_per_second,
        "avg_car_visibility_percent": car_visibility_percent,
        "average_confidence": average_confidence,
        "most_active_start_sec": most_active_segment["startTimeSec"],
        "most_active_end_sec": most_active_segment["endTimeSec"],
        "avg_cars_per_frame": most_active_segment["avgCarsPerFrame"]
    }

    color_stats = [{"color": c, "count": n} for c, n in color_ctr.items()]
    brand_stats = [{"brand": b, "logo_count": n} for b, n in brand_ctr.items()]
    model_stats = [{"make": m, "model": mo, "count": n} for (m, mo), n in model_ctr.items()]
    region_stats = [{"region": r, "plate_count": n} for r, n in region_ctr.items()]

    return {
        "video": {
            "filename": video_file,
            "source": source_video,
            "fps": fps,
            "total_frames": max_frame + 1,
            "duration_sec": duration_sec
        },
        "video_summary": video_summary,
        "vehicles": vehicles,
        "color_stats": color_stats,
        "brand_stats": brand_stats,
        "model_stats": model_stats,
        "region_stats": region_stats
    }
