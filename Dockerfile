# --- Build React Frontend ---
FROM node:20 AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# --- Python Backend ---
FROM python:3.10-slim AS backend
WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg gcc curl && \
    # Install Node.js (LTS) and npm
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean

# Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files
COPY main.py app.py ./
COPY postprocessing.py save_to_db.py helper.py ./
COPY docker-entrypoint.sh ./

# Copy YOLO model files
COPY yolov8n.pt ./
COPY yolov8n-seg.pt ./

# Create necessary directories
RUN mkdir -p videos/uploads videos/processed

# Copy built frontend
COPY --from=frontend-build /app/frontend/build ./frontend_build

# Expose ports
EXPOSE 5000 3000

# Entrypoint script
RUN chmod +x /docker-entrypoint.sh

CMD ["/docker-entrypoint.sh"]