#!/bin/bash
# Start Redis in the background
# redis-server &

# Start RQ worker in the background
# rq worker video-processing --url redis://localhost:6379 &

# Start Flask backend
FLASK_APP=app.py flask run --host=0.0.0.0 --port=5000 &

# Start React frontend (serve static build)
npx serve -s frontend_build -l 3000