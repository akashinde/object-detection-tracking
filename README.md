# Outline

- Object Detection: YOLOv8 model
- Object Tracking: DeepSORT model
- Number Plate: ANPR model
- Classifier: ResNet, EfficientNet
- Database: SQL (Why? - We have fixed number of columns and fetching of rows is faster compare to NoSQL.)

## Features

### Real-Time Progress Tracking
The application now includes real-time progress tracking for video processing:

- **Progress Bar**: Shows real-time progress percentage during video processing
- **Status Updates**: Displays current processing stage (loading models, frame processing, AI analysis, etc.)
- **Redis Integration**: Uses Redis for progress storage and retrieval
- **Fallback Support**: Works even if Redis is unavailable (with limited progress tracking)

#### Progress Stages:
1. **0-10%**: Starting video processing
2. **10-30%**: Loading YOLO models and initializing
3. **30-75%**: Processing video frames (progress updates every 30 frames)
4. **75-78%**: Converting video to H.264 format
5. **78-80%**: AI analysis of car images
6. **80-100%**: Database operations and dashboard updates

---

## Running Locally with Docker (All-in-One, No Compose)

This project can be run in a single Docker container using a `docker-entrypoint.sh` script. This will start the Flask backend, React frontend (static build), Redis server, and RQ worker all together.

### 1. Build the Docker Image

**Option A: Using the build script (Recommended)**

**Linux/Mac:**
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-openai-api-key-here"

# Run the build script
chmod +x build.sh
./build.sh
```

**Windows:**
```cmd
# Set your OpenAI API key
set OPENAI_API_KEY=your-openai-api-key-here

# Run the build script
build.bat
```

**Option B: Direct Docker build**
```bash
docker build --build-arg OPENAI_API_KEY="your-openai-api-key-here" -t detection-app .
```

### 2. Run the Container

```
docker run --rm -p 3000:3000 -p 5000:5000 detection-app
```

- The **frontend** will be available at [http://localhost:3000](http://localhost:3000)
- The **backend API** will be available at [http://localhost:5000](http://localhost:5000)

### 3. What Happens Inside the Container
- Redis server starts in the background with custom configuration
- RQ worker starts and listens for video processing jobs
- Flask backend starts on port 5000
- React frontend (static build) is served on port 3000

### 4. Uploading and Processing Videos
- Use the web UI to upload a video
- The backend enqueues the job and processes it with real-time progress updates
- Progress is displayed in the frontend with a progress bar and status messages
- You can monitor job status and progress in real-time

### 5. Requirements
- Docker (no need for docker-compose)
- All dependencies are installed inside the container
- Redis is automatically installed and configured

---

**For advanced/multi-container setups, use docker-compose instead.**
