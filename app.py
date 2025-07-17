from flask import Flask, jsonify, request, send_from_directory
import sqlite3
import os
import subprocess
from flask_cors import CORS
import redis
from rq import Queue
import uuid

app = Flask(__name__)
CORS(app)

DB_PATH = 'detections.db'
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


# Redis setup
redis_conn = redis.Redis()
q = Queue('video-processing', connection=redis_conn)

def process_video_job(video_path):
    job_id = os.path.splitext(os.path.basename(video_path))[0]
    try:
        result = subprocess.run([
            'python3', 'main.py', '--video', video_path
        ], capture_output=True, text=True, check=True)
        # Save output to a file
        output_filename = f"{job_id}_output.txt"
        with open(output_filename, "w") as f:
            f.write("STDOUT:\n" + result.stdout + "\n")
            f.write("STDERR:\n" + (result.stderr or "") + "\n")
        # Move or copy the processed video to processed_folder if needed
        # redis_conn.set(f"job_status:{job_id}", "completed")
        return {'status': 'success', 'stdout': result.stdout}
    except subprocess.CalledProcessError as e:
        # Save error output to a file
        output_filename = f"{job_id}_output.txt"
        with open(output_filename, "w") as f:
            f.write("STDOUT:\n" + (e.stdout or "") + "\n")
            f.write("STDERR:\n" + (e.stderr or "") + "\n")
        # redis_conn.set(f"job_status:{job_id}", "failed")
        return {'status': 'error', 'stderr': e.stderr}


@app.route('/api', methods=['GET'])
def api():
    return jsonify({"message": "Hello, World!"})

@app.route('/api/cars', methods=['GET'])
def get_cars():
    conn = get_db_connection()
    query = '''
        SELECT 
            c.track_id, c.label, 
            t.name as type, 
            m.name as model, 
            col.name as color, 
            c.license_plate, 
            c.track_frame_counts, 
            c.scene_count, 
            c.dwell_time_seconds,
            c.video_id,
            v.filename as video_filename
        FROM CARS c
        LEFT JOIN CAR_TYPES t ON c.type_id = t.id
        LEFT JOIN CAR_MODELS m ON c.model_id = m.id
        LEFT JOIN CAR_COLORS col ON c.color_id = col.id
        LEFT JOIN VIDEOS v ON c.video_id = v.id
        ORDER BY c.track_id
    '''
    cars = [dict(row) for row in conn.execute(query).fetchall()]
    conn.close()
    return jsonify({"cars": cars})

@app.route('/api/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file part'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    # Save the uploaded file
    job_id = str(uuid.uuid4())
    filename = f"{job_id}_{file.filename}"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)
    # Instead, call directly:
    result = process_video_job(video_path)
    transform_result = None
    store_result = None
    # After processing, delete the uploaded file
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
    except Exception as e:
        # Log error but don't fail the request
        print(f"Error deleting uploaded file: {e}")
    if result.get('status') == 'success':
        # Run transform.py and store_detections.py as function calls
        try:
            import transform
            transform.main()
            transform_result = {'status': 'success'}
        except Exception as e:
            transform_result = {'status': 'error', 'error': str(e)}
        try:
            import store_detections
            store_detections.main(filename)
            store_result = {'status': 'success'}
        except Exception as e:
            store_result = {'status': 'error', 'error': str(e)}
    return jsonify({'status': result.get('status'), 'job_id': job_id, 'stdout': result.get('stdout'), 'stderr': result.get('stderr'), 'transform_result': transform_result, 'store_result': store_result})

@app.route('/api/videos', methods=['GET'])
def list_videos():
    files = [f for f in os.listdir(app.config['PROCESSED_FOLDER']) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    return jsonify({'videos': files})

@app.route('/api/videos/<filename>', methods=['GET'])
def get_video(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

# Comment out job status endpoint
# @app.route('/api/job_status/<job_id>', methods=['GET'])
# def job_status(job_id):
#     status = redis_conn.get(f"job_status:{job_id}")
#     if status is None:
#         return jsonify({'status': 'unknown'}), 404
#     return jsonify({'status': status.decode()})

if __name__ == '__main__':
    app.run(debug=True) 