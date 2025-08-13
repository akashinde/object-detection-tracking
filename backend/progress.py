import json
import logging
import time

import redis

logger = logging.getLogger(__name__)

try:
    redis_conn = redis.Redis()
    redis_conn.ping()
    REDIS_AVAILABLE = True
    logger.info("Redis connection established successfully")
except Exception as e:
    logger.warning(f"Redis not available: {e}. Progress tracking will be disabled.")
    REDIS_AVAILABLE = False
    redis_conn = None


def update_progress(job_id, progress, status="processing", message=""):
    """Update progress information in Redis if available."""
    if not REDIS_AVAILABLE or not job_id:
        return

    try:
        progress_data = {
            "progress": progress,
            "status": status,
            "message": message,
            "timestamp": time.time(),
        }
        redis_conn.setex(f"progress:{job_id}", 3600, json.dumps(progress_data))
    except Exception as e:
        logger.error(f"Error updating progress: {e}")
