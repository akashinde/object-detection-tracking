# Outline

- Object Detection: YOLOv8 model
- Object Tracking: DeepSORT model
- Number Plate: ANPR model
- Classifier: ResNet, EfficientNet
- Database: SQL (Why? - We have fixed number of columns and fetching of rows is faster compare to NoSQL.)

---

## Running Locally with Docker (All-in-One, No Compose)

This project can be run in a single Docker container using a `docker-entrypoint.sh` script. This will start the Flask backend, React frontend (static build), Redis server, and RQ worker all together.

### 1. Build the Docker Image

```
docker build -t detection-app .
```

### 2. Run the Container

```
docker run -p 3000:3000 -p 5000:5000 detection-app
```

- The **frontend** will be available at [http://localhost:3000](http://localhost:3000)
- The **backend API** will be available at [http://localhost:5000](http://localhost:5000)

### 3. What Happens Inside the Container
- Redis server starts in the background
- RQ worker starts and listens for video processing jobs
- Flask backend starts on port 5000
- React frontend (static build) is served on port 3000

### 4. Uploading and Processing Videos
- Use the web UI to upload a video
- The backend enqueues the job and processes it
- You can monitor job status in the UI

### 5. Requirements
- Docker (no need for docker-compose)
- All dependencies are installed inside the container

---

**For advanced/multi-container setups, use docker-compose instead.**
