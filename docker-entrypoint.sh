#!/bin/bash

# Start Redis server in the background with custom config
echo "Starting Redis server..."
redis-server redis.conf --daemonize yes

# Wait a moment for Redis to start
sleep 3

# Test Redis connection
echo "Testing Redis connection..."
if redis-cli ping; then
    echo "Redis is running successfully!"
else
    echo "Warning: Redis connection failed, but continuing..."
fi

# Start RQ worker in the background (optional, for future use)
# echo "Starting RQ worker..."
# rq worker video-processing --url redis://localhost:6379 &

# Start Flask backend
echo "Starting Flask backend..."
FLASK_APP=app.py flask run --host=0.0.0.0 --port=5000 &

# Start React frontend (serve static build)
echo "Starting React frontend..."
npx serve -s frontend_build -l 3000