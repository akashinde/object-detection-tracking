# db_dashboard.py  – put this next to save_to_db.py
# -------------------------------------------------
import sqlite3
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple

DB_PATH = "analytics.db"  # adjust/import from settings


# ---------------------------------------------------------------------------
def _fetchall_as_dict(cur) -> List[Dict[str, Any]]:
    "helper: cursor rows → list[dict]"
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
def build_dashboard_from_db(db_path: str = DB_PATH, top_n: int = 4) -> Dict[str, Any]:
    """
    Assemble sponsor-ready dashboard JSON from DB.

    Returns:
        {
          "dashboardSummary": {...},
          "videos": [...]
        }
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # -------------------- 1. DASHBOARD‑LEVEL KPIs --------------------------
    cur.execute(
        "SELECT COUNT(*)             AS total_videos, "
        "       COALESCE(SUM(duration_sec),0) AS total_duration "
        "FROM videos;"
    )
    vids_row = cur.fetchone()
    total_videos, total_duration = vids_row["total_videos"], vids_row["total_duration"]

    cur.execute(
        "SELECT COALESCE(SUM(total_cars),0) AS total_cars, "
        "       COALESCE(SUM(distinct_plates),0) AS total_plates "
        "FROM video_summary;"
    )
    cars_row = cur.fetchone()
    total_cars, distinct_plates = cars_row["total_cars"], cars_row["total_plates"]

    # ---- overall top brands/colors/models --------------------------------
    def _top_list_with_counts(table: str, col: str, label: str) -> List[Dict[str, Any]]:
        cur.execute(
            f"SELECT {col} AS key, SUM(count) AS cnt "
            f"FROM {table} GROUP BY {col} "
            f"ORDER BY cnt DESC LIMIT {top_n};"
        )
        return [{"name": r["key"], "count": r["cnt"]} for r in cur.fetchall()]

    # brand_stats -> SUM(logo_count) rather than count
    cur.execute(
        "SELECT brand AS key, SUM(logo_count) AS cnt "
        "FROM brand_stats GROUP BY brand "
        "ORDER BY cnt DESC LIMIT ?;",
        (top_n,),
    )
    top_brands = [{"name": r["key"], "count": r["cnt"]} for r in cur.fetchall()]

    top_colors = _top_list_with_counts("color_stats", "color", "color")
    # model_stats: concatenate make + model
    cur.execute(
        "SELECT make||' '||model AS key, SUM(count) AS cnt "
        "FROM model_stats GROUP BY make, model "
        "ORDER BY cnt DESC LIMIT ?;",
        (top_n,),
    )
    top_models = [{"name": r["key"], "count": r["cnt"]} for r in cur.fetchall()]

    dashboard_summary = {
        "totalVideosProcessed": total_videos,
        "totalDurationSec": total_duration,
        "totalCarsDetected": total_cars,
        "distinctPlatesDetected": distinct_plates,
        "topBrandsOverall": top_brands,
        "topColorsOverall": top_colors,
        "topModelsOverall": top_models,
    }

    # -------------------- 2. PER‑VIDEO DETAILS ----------------------------
    cur.execute("SELECT * FROM videos;")
    video_rows = _fetchall_as_dict(cur)

    videos_block: List[Dict[str, Any]] = []

    for v in video_rows:
        vid = v["video_id"]

        # --- summary & perf
        cur.execute("SELECT * FROM video_summary WHERE video_id = ?", (vid,))
        vs = cur.fetchone() or {}

        # fallback helpers
        def _gf(k, default=None):
            return vs[k] if vs and k in vs.keys() else default

        # --- vehicle type counts
        cur.execute(
            "SELECT type, COUNT(*) AS cnt "
            "FROM vehicles WHERE video_id = ? GROUP BY type;",
            (vid,),
        )
        vehicle_types = {r["type"]: r["cnt"] for r in cur.fetchall()}

        # --- color distribution
        cur.execute(
            "SELECT color, count FROM color_stats " "WHERE video_id = ?;", (vid,)
        )
        color_dist = {r["color"]: r["count"] for r in cur.fetchall()}

        # --- make‑model stats
        cur.execute(
            "SELECT make, model, count FROM model_stats " "WHERE video_id = ?;", (vid,)
        )
        make_model_stats = {
            f"{r['make']} {r['model']}": r["count"] for r in cur.fetchall()
        }

        # --- brand logo stats
        cur.execute(
            "SELECT brand, logo_count FROM brand_stats " "WHERE video_id = ?;", (vid,)
        )
        brand_logo_stats = {r["brand"]: r["logo_count"] for r in cur.fetchall()}

        # --- brand‑color matrix
        cur.execute(
            "SELECT brand, color, COUNT(*) AS cnt "
            "FROM vehicles WHERE video_id = ? "
            "GROUP BY brand, color;",
            (vid,),
        )
        brand_color_matrix: Dict[str, Dict[str, int]] = defaultdict(dict)
        for r in cur.fetchall():
            brand_color_matrix[r["brand"]][r["color"]] = r["cnt"]

        # --- plate summary & regions
        cur.execute(
            "SELECT COUNT(*) FROM vehicles "
            "WHERE video_id = ? AND license_plate IS NOT NULL "
            "AND license_plate != '';",
            (vid,),
        )
        plates_detected = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(DISTINCT license_plate) FROM vehicles "
            "WHERE video_id = ? AND license_plate IS NOT NULL "
            "AND license_plate != '';",
            (vid,),
        )
        distinct_plates = cur.fetchone()[0]

        cur.execute(
            "SELECT license_region, COUNT(*) AS cnt "
            "FROM vehicles WHERE video_id = ? "
            "AND license_region IS NOT NULL GROUP BY license_region;",
            (vid,),
        )
        estimated_regions = {r["license_region"]: r["cnt"] for r in cur.fetchall()}

        # top plates by frequency
        cur.execute(
            "SELECT license_plate, COUNT(*) AS freq "
            "FROM vehicles WHERE video_id = ? "
            "AND license_plate IS NOT NULL "
            "AND license_plate != '' "
            "GROUP BY license_plate ORDER BY freq DESC LIMIT 5;",
            (vid,),
        )
        top_plates = [
            {"plate": r["license_plate"], "frameCount": r["freq"]}
            for r in cur.fetchall()
        ]

        # --- motion stats
        cur.execute(
            "SELECT is_moving, COUNT(*) AS cnt "
            "FROM vehicles WHERE video_id = ? GROUP BY is_moving;",
            (vid,),
        )
        motion = {
            ("movingVehicles" if r["is_moving"] else "staticVehicles"): r["cnt"]
            for r in cur.fetchall()
        }
        if "movingVehicles" not in motion:
            motion["movingVehicles"] = 0
        if "staticVehicles" not in motion:
            motion["staticVehicles"] = 0

        # --- detection performance (optional cols may be NULL)
        detection_perf = {
            "framesProcessed": _gf("frames_processed"),
            "avgProcessingTimeMs": _gf("avg_processing_time_ms"),
            "falsePositiveRate": _gf("false_positive_rate"),
            "falseNegativeRate": _gf("false_negative_rate"),
        }

        # Build per‑video dict
        videos_block.append(
            {
                "videoId": vid,
                "filename": v["filename"],
                "durationSec": v["duration_sec"],
                "totalCarsDetected": _gf("total_cars"),
                "carVisibilityPercent": _gf("avg_car_visibility_percent"),
                "averageDetectionConfidence": _gf("average_confidence"),
                "vehicleTypes": vehicle_types,
                "colorDistribution": color_dist,
                "makeModelStats": make_model_stats,
                "brandLogoStats": brand_logo_stats,
                "brandColorMatrix": brand_color_matrix,
                "numberPlateSummary": {
                    "platesDetected": plates_detected,
                    "distinctPlates": distinct_plates,
                    "estimatedRegions": estimated_regions,
                    "topPlates": top_plates,
                },
                # placeholders for advanced fields if you later persist them
                "mostActiveSegment": {
                    "startTimeSec": _gf("most_active_start_sec"),
                    "endTimeSec": _gf("most_active_end_sec"),
                    "avgCarsPerFrame": _gf("avg_cars_per_frame"),
                },
                "spatialPresenceHeatmap": {},  # populate if stored
                "motionStats": motion,
                "detectionPerformance": detection_perf,
            }
        )

    conn.close()

    return {"dashboardSummary": dashboard_summary, "videos": videos_block}
