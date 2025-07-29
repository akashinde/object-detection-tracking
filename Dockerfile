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

# Accept OpenAI API key as build argument
ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY

# System dependencies including Redis
RUN apt-get update && \
    apt-get install -y ffmpeg gcc curl redis-server && \
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
COPY redis.conf ./

# Copy YOLO model files
COPY yolo*.pt ./

# Create necessary directories
RUN mkdir -p videos/uploads videos/processed

# Copy built frontend
COPY --from=frontend-build /app/frontend/build ./frontend_build

# Make entrypoint script executable
RUN chmod +x ./docker-entrypoint.sh

# Expose ports
EXPOSE 5000 3000 6379

CMD ["./docker-entrypoint.sh"]