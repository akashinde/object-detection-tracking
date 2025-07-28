from flask import Flask, jsonify, request, send_from_directory, abort
import sqlite3
import os
import subprocess
from flask_cors import CORS
import redis
from rq import Queue
import uuid
import logging
from dotenv import load_dotenv

from main import run_detection
from postprocessing import transform_detections_from_obj
from save_to_db import _create_tables, save_analytics_to_db
from helper import build_dashboard_from_db

app = Flask(__name__)
CORS(app)
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = 'analytics.db'
UPLOAD_FOLDER = 'videos/uploads'
PROCESSED_FOLDER = 'videos/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

redis_conn = redis.Redis()
q = Queue('video-processing', connection=redis_conn)

def process_video_job(video_path):
    job_id = os.path.splitext(os.path.basename(video_path))[0]
    try:
        result = subprocess.run([
            'python3', 'main.py', '--video', video_path
        ], capture_output=True, text=True, check=True)
        output_filename = f"{job_id}_output.txt"
        with open(output_filename, "w") as f:
            f.write("STDOUT:\n" + result.stdout + "\n")
            f.write("STDERR:\n" + (result.stderr or "") + "\n")
        return {'status': 'success', 'stdout': result.stdout}
    except subprocess.CalledProcessError as e:
        output_filename = f"{job_id}_output.txt"
        with open(output_filename, "w") as f:
            f.write("STDOUT:\n" + (e.stdout or "") + "\n")
            f.write("STDERR:\n" + (e.stderr or "") + "\n")
        return {'status': 'error', 'stderr': e.stderr}

def ensure_normalized_db_tables():
    conn = sqlite3.connect(DB_PATH)
    _create_tables(conn)
    conn.close()

# Call this on startup
ensure_normalized_db_tables()

@app.route('/api', methods=['GET'])
def api():
    return jsonify({"message": "Hello, World!"})

@app.route('/api/vehicles', methods=['GET'])
def get_vehicles():
    conn = get_db_connection()
    
    # Get query parameters for filtering
    search = request.args.get('search', '')
    brand_filter = request.args.get('brand', '')
    color_filter = request.args.get('color', '')
    vehicle_type_filter = request.args.get('vehicleType', '')
    region_filter = request.args.get('region', '')
    sponsor_brand = request.args.get('sponsorBrand', '')
    high_exposure_only = request.args.get('highExposureOnly', 'false').lower() == 'true'
    
    # Build the base query
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
    
    # Add search filter
    if search:
        query += " AND (v.brand LIKE ? OR v.model LIKE ? OR v.license_plate LIKE ? OR v.color LIKE ?)"
        search_param = f"%{search}%"
        params.extend([search_param, search_param, search_param, search_param])
    
    # Add brand filter
    if brand_filter:
        query += " AND v.brand LIKE ?"
        params.append(f"%{brand_filter}%")
    
    # Add color filter
    if color_filter:
        query += " AND v.color LIKE ?"
        params.append(f"%{color_filter}%")
    
    # Add vehicle type filter
    if vehicle_type_filter:
        query += " AND v.type LIKE ?"
        params.append(f"%{vehicle_type_filter}%")
    
    # Add region filter
    if region_filter:
        query += " AND v.license_region LIKE ?"
        params.append(f"%{region_filter}%")
    
    # Add sponsor brand filter
    if sponsor_brand:
        query += " AND v.brand LIKE ?"
        params.append(f"%{sponsor_brand}%")
    
    # Add high exposure filter (top 10% by dwell time)
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

@app.route('/api/process_video', methods=['POST'])
def process_video():
    video_source = request.form.get('videoSource') or \
                   (request.json.get('videoSource') if request.is_json and 'videoSource' in request.json else None) \
                   or 'local'
    if 'video' not in request.files:
        logger.warning('No video file part in request')
        return jsonify({'error': 'No video file part'}), 400
    file = request.files['video']
    if file.filename == '':
        logger.warning('No selected file for upload')
        return jsonify({'error': 'No selected file'}), 400

    job_id = str(uuid.uuid4())[:8]
    filename = f"{job_id}_{file.filename}"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    logger.info(f'Saving uploaded video as {video_path}')
    file.save(video_path)
    logger.debug(f'Video saved: {video_path}')

    try:
        logger.info(f'Running detection for {video_path}')
        detections, meta = run_detection(video_path)
        meta['source_video'] = video_source
        # logger.debug(f'Detection result meta: {meta}')
        
        analytics = transform_detections_from_obj(detections, meta)
        # logger.debug(f'Analytics: {analytics}')
        
        db_result = save_analytics_to_db(analytics)
        # logger.debug(f'DB save result: {db_result}')
        
        dashboard = build_dashboard_from_db()
        # logger.debug(f'Dashboard data: {dashboard}')
        
        if os.path.exists(video_path):
            os.remove(video_path)
            # logger.debug(f'Uploaded video deleted: {video_path}')
        return jsonify({
            'status': 'success',
            'job_id': job_id,
            'analytics': analytics,
            'db_result': db_result,
            'meta': meta
        })
    except Exception as e:
        logger.exception(f'Error during processing video {video_path}: {e}')
        if os.path.exists(video_path):
            os.remove(video_path)
            logger.debug(f'Uploaded video deleted after error: {video_path}')
        return jsonify({'status': 'error', 'error': str(e), 'job_id': job_id}), 500

@app.route('/api/videos', methods=['GET'])
def list_videos():
    conn = get_db_connection()
    db_videos = conn.execute('SELECT filename FROM VIDEOS').fetchall()
    db_video_set = set(os.path.splitext(row['filename'])[0] for row in db_videos)
    conn.close()
    folders = [f for f in os.listdir(app.config['PROCESSED_FOLDER']) if os.path.isdir(os.path.join(app.config['PROCESSED_FOLDER'], f))]
    video_files = []
    for folder in folders:
        if folder not in db_video_set:
            continue
        folder_path = os.path.join(app.config['PROCESSED_FOLDER'], folder)
        for f in os.listdir(folder_path):
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append({'folder': folder, 'filename': f})
    return jsonify({'videos': video_files})

@app.route('/api/videos/<video_folder>/<filename>', methods=['GET'])
def get_video(video_folder, filename):
    folder_path = os.path.join(app.config['PROCESSED_FOLDER'], video_folder)
    return send_from_directory(folder_path, filename)

@app.route('/api/car_image')
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

@app.route('/api/dashboard', methods=['GET'])
def get_summary():
    try:
        resp = build_dashboard_from_db()
        # logger.debug(f'Dashboard summary: {resp}')
        if not resp:
            logger.warning('No data found in dashboard summary')
            return jsonify({'error': 'No data found'}), 404
        return jsonify(resp)
    except Exception as e:
        logger.error(f'Error building dashboard summary: {e}')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)