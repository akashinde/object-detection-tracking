import json
import logging
import os
import threading
import time
import uuid

from flask import Blueprint, jsonify, request, send_from_directory, abort
from rq import Queue

from main import run_detection
from postprocessing import transform_detections_from_obj
from save_to_db import save_analytics_to_db
from helper import build_dashboard_from_db
from .db import get_db_connection, ensure_normalized_db_tables, UPLOAD_FOLDER, PROCESSED_FOLDER
from .progress import update_progress, REDIS_AVAILABLE, redis_conn

logger = logging.getLogger(__name__)

bp = Blueprint('api', __name__)

q = Queue('video-processing', connection=redis_conn) if REDIS_AVAILABLE else None

ensure_normalized_db_tables()


@bp.route('/api', methods=['GET'])
def api_root():
    return jsonify({"message": "Hello, World!"})


@bp.route('/api/vehicles', methods=['GET'])
def get_vehicles():
    conn = get_db_connection()

    search = request.args.get('search', '')
    brand_filter = request.args.get('brand', '')
    color_filter = request.args.get('color', '')
    vehicle_type_filter = request.args.get('vehicleType', '')
    region_filter = request.args.get('region', '')
    sponsor_brand = request.args.get('sponsorBrand', '')
    high_exposure_only = request.args.get('highExposureOnly', 'false').lower() == 'true'

    query = '''
        SELECT
            v.id,
            v.video_id,
            v.track_id,
            v.first_seen_sec,
            v.last_seen_sec,
            v.dwell_time_seconds,
            v.type,
            v.color,
            v.brand,
            v.model,
            v.license_plate,
            v.license_region,
            v.is_moving,
            v.image_path,
            vid.filename as video_filename,
            vid.duration_sec as video_duration
        FROM vehicles v
        LEFT JOIN videos vid ON v.video_id = vid.video_id
        WHERE 1=1
    '''
    params = []

    if search:
        query += " AND (v.brand LIKE ? OR v.model LIKE ? OR v.license_plate LIKE ? OR v.color LIKE ?)"
        search_param = f"%{search}%"
        params.extend([search_param, search_param, search_param, search_param])

    if brand_filter:
        query += " AND v.brand LIKE ?"
        params.append(f"%{brand_filter}%")

    if color_filter:
        query += " AND v.color LIKE ?"
        params.append(f"%{color_filter}%")

    if vehicle_type_filter:
        query += " AND v.type LIKE ?"
        params.append(f"%{vehicle_type_filter}%")

    if region_filter:
        query += " AND v.license_region LIKE ?"
        params.append(f"%{region_filter}%")

    if sponsor_brand:
        query += " AND v.brand LIKE ?"
        params.append(f"%{sponsor_brand}%")

    if high_exposure_only:
        query += " AND v.dwell_time_seconds >= (SELECT AVG(dwell_time_seconds) * 1.5 FROM vehicles)"

    query += " ORDER BY v.track_id"

    try:
        vehicles = [dict(row) for row in conn.execute(query, params).fetchall()]
        conn.close()
        return jsonify({"vehicles": vehicles})
    except Exception as e:
        logger.error(f"Error fetching vehicles: {e}")
        conn.close()
        return jsonify({"error": str(e)}), 500


def process_video_with_progress(video_path, job_id):
    try:
        update_progress(job_id, 0, "processing", "Starting video processing...")
        update_progress(job_id, 10, "processing", "Loading YOLO models...")

        logger.info(f'Running detection for {video_path}')
        detections, meta = run_detection(video_path, job_id)

        update_progress(job_id, 80, "processing", "Processing detection results...")

        meta['source_video'] = 'local'

        update_progress(job_id, 85, "processing", "Transforming analytics data...")

        analytics = transform_detections_from_obj(detections, meta)

        update_progress(job_id, 90, "processing", "Saving to database...")
        db_result = save_analytics_to_db(analytics)

        update_progress(job_id, 95, "processing", "Building dashboard...")
        build_dashboard_from_db()

        update_progress(job_id, 100, "completed", "Video processing completed successfully!")

        if os.path.exists(video_path):
            os.remove(video_path)

        return {
            'status': 'success',
            'job_id': job_id,
            'analytics': analytics,
            'db_result': db_result,
            'meta': meta
        }
    except Exception as e:
        logger.exception(f'Error during processing video {video_path}: {e}')
        update_progress(job_id, 0, "error", f"Error: {str(e)}")
        if os.path.exists(video_path):
            os.remove(video_path)
        return {'status': 'error', 'error': str(e), 'job_id': job_id}


@bp.route('/api/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        logger.warning('No video file part in request')
        return jsonify({'error': 'No video file part'}), 400

    file = request.files['video']
    if file.filename == '':
        logger.warning('No selected file for upload')
        return jsonify({'error': 'No selected file'}), 400

    job_id = str(uuid.uuid4())[:8]
    filename = f"{job_id}_{file.filename}"
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    logger.info(f'Saving uploaded video as {video_path}')
    file.save(video_path)

    def process_thread():
        process_video_with_progress(video_path, job_id)

    thread = threading.Thread(target=process_thread)
    thread.daemon = True
    thread.start()

    return jsonify({'status': 'processing', 'job_id': job_id, 'message': 'Video processing started'})


@bp.route('/api/progress/<job_id>', methods=['GET'])
def get_progress(job_id):
    if not REDIS_AVAILABLE:
        return jsonify({
            "progress": 0,
            "status": "redis_unavailable",
            "message": "Progress tracking not available",
            "timestamp": time.time(),
        })

    try:
        progress_data = redis_conn.get(f"progress:{job_id}")
        if progress_data:
            return jsonify(json.loads(progress_data))
        return jsonify({"progress": 0, "status": "not_found", "message": "Job not found", "timestamp": time.time()})
    except Exception as e:
        logger.error(f"Error getting progress for job {job_id}: {e}")
        return jsonify({"error": str(e)}), 500


@bp.route('/api/videos', methods=['GET'])
def list_videos():
    conn = get_db_connection()
    db_videos = conn.execute('SELECT filename FROM VIDEOS').fetchall()
    db_video_set = set(os.path.splitext(row['filename'])[0] for row in db_videos)
    conn.close()

    folders = [f for f in os.listdir(PROCESSED_FOLDER) if os.path.isdir(os.path.join(PROCESSED_FOLDER, f))]
    video_files = []
    for folder in folders:
        if folder not in db_video_set:
            continue
        folder_path = os.path.join(PROCESSED_FOLDER, folder)
        for f in os.listdir(folder_path):
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append({'folder': folder, 'filename': f})
    return jsonify({'videos': video_files})


@bp.route('/api/videos/<video_folder>/<filename>', methods=['GET'])
def get_video(video_folder, filename):
    folder_path = os.path.join(PROCESSED_FOLDER, video_folder)
    return send_from_directory(folder_path, filename)


@bp.route('/api/car_image')
def get_car_image():
    rel_path = request.args.get('path')
    logger.debug(f'GET /api/car_image called with path: {rel_path}')
    if not rel_path:
        return jsonify({'error': 'No image path provided'}), 400
    if '..' in rel_path or rel_path.startswith('/'):
        return abort(403)

    abs_path = os.path.abspath(rel_path)
    allowed_root = os.path.abspath('videos/processed')
    if not abs_path.startswith(allowed_root):
        return abort(403)
    dir_name = os.path.dirname(abs_path)
    file_name = os.path.basename(abs_path)
    if not os.path.exists(abs_path):
        return abort(404)
    return send_from_directory(dir_name, file_name)


@bp.route('/api/dashboard', methods=['GET'])
def get_summary():
    try:
        resp = build_dashboard_from_db()
        if not resp:
            logger.warning('No data found in dashboard summary')
            return jsonify({'error': 'No data found'}), 404
        return jsonify(resp)
    except Exception as e:
        logger.error(f'Error building dashboard summary: {e}')
        return jsonify({'error': str(e)}), 500
