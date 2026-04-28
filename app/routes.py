"""
routes.py — Flask routes: video upload, MJPEG streaming, stats, dashboard, webcam
Computer Vision Project — Traffic Monitoring
"""

import os
import json
import threading
import datetime
from flask import (Blueprint, render_template, request, jsonify,
                   Response, redirect, url_for, flash, send_file)
from werkzeug.utils import secure_filename

from .detector import TrafficDetector, TRAFFIC_CLASSES
from .logger   import build_shared_log, save_log_json, save_summary_csv, load_and_merge_logs

import cv2

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
LOGS_FOLDER   = os.path.join(BASE_DIR, "outputs", "logs")
ALLOWED_EXTS  = {"mp4", "avi", "mov", "mkv", "webm"}
MODEL_PATH    = os.path.join(BASE_DIR, "models", "yolo11s.pt")

bp = Blueprint("main", __name__)

_detector: TrafficDetector = None
_processing_lock           = threading.Lock()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


def get_detector(selected_classes=None):
    global _detector
    _detector = TrafficDetector(
        model_path=MODEL_PATH,
        selected_classes=selected_classes,
        conf_threshold=0.3
    )
    return _detector


# ─── Home ─────────────────────────────────────────────────────────────────────

@bp.route("/")
def index():
    class_names = list(TRAFFIC_CLASSES.values())
    videos = []
    if os.path.isdir(UPLOAD_FOLDER):
        videos = [f for f in os.listdir(UPLOAD_FOLDER) if allowed_file(f)]
    return render_template("index.html", class_names=class_names, videos=videos)


# ─── Upload ───────────────────────────────────────────────────────────────────

@bp.route("/upload", methods=["POST"])
def upload_video():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    selected_classes = request.form.getlist("classes")
    scene_name       = request.form.get("scene_name", "").strip()
    classes_param    = ",".join(selected_classes) if selected_classes else "all"

    # Case 1: existing video selected
    existing = request.form.get("existing_video", "").strip()
    if existing:
        if not scene_name:
            scene_name = os.path.splitext(existing)[0].replace("_", " ").title()
        existing_path = os.path.join(UPLOAD_FOLDER, secure_filename(existing))
        if os.path.isfile(existing_path):
            return redirect(url_for("main.visualize",
                                    video=existing,
                                    classes=classes_param,
                                    scene=scene_name))

    # Case 2: new file uploaded
    if "video" not in request.files:
        flash("No file received.", "error")
        return redirect(url_for("main.index"))

    file = request.files["video"]
    if file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("main.index"))

    if not allowed_file(file.filename):
        flash("Unsupported format. Please use mp4, avi, mov or mkv.", "error")
        return redirect(url_for("main.index"))

    filename = secure_filename(file.filename)
    if not scene_name:
        scene_name = os.path.splitext(filename)[0].replace("_", " ").title()

    video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(video_path)

    return redirect(url_for("main.visualize",
                            video=filename,
                            classes=classes_param,
                            scene=scene_name))


# ─── Visualize ────────────────────────────────────────────────────────────────

@bp.route("/visualize")
def visualize():
    video      = request.args.get("video", "")
    classes    = request.args.get("classes", "all")
    scene_name = request.args.get("scene", "Scene")

    if not video:
        return redirect(url_for("main.index"))

    class_names = list(TRAFFIC_CLASSES.values())
    selected    = class_names if classes == "all" else classes.split(",")

    return render_template("visualize.html",
                           video=video,
                           selected_classes=selected,
                           scene_name=scene_name,
                           classes_param=classes)


# ─── MJPEG Video stream ───────────────────────────────────────────────────────

@bp.route("/video_feed")
def video_feed():
    video      = request.args.get("video", "")
    classes    = request.args.get("classes", "all")
    video_path = os.path.join(UPLOAD_FOLDER, secure_filename(video))

    if not os.path.isfile(video_path):
        return "Video not found", 404

    class_names = list(TRAFFIC_CLASSES.values())
    selected    = class_names if classes == "all" else classes.split(",")
    detector    = get_detector(selected)

    return Response(
        detector.generate_frame_generator(video_path),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ─── Webcam page ──────────────────────────────────────────────────────────────

@bp.route("/webcam")
def webcam():
    """Live webcam detection page."""
    class_names = list(TRAFFIC_CLASSES.values())
    return render_template("webcam.html", class_names=class_names)


# ─── Webcam MJPEG stream ──────────────────────────────────────────────────────

@bp.route("/webcam_feed")
def webcam_feed():
    """MJPEG stream from webcam."""
    classes     = request.args.get("classes", "all")
    class_names = list(TRAFFIC_CLASSES.values())
    selected    = class_names if classes == "all" else classes.split(",")
    detector    = get_detector(selected)

    def generate():
        cap = cv2.VideoCapture(0)  # 0 = default webcam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        detector.counted_ids.clear()
        detector.counts_per_class = {n: 0 for n in detector.selected_class_names}
        detector.frame_index = 0
        detector.start_time  = datetime.datetime.now()
        detector.counting_line_y = None
        detector.finished = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated, stats = detector.process_frame(frame)
            ret2, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret2:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       buffer.tobytes() + b"\r\n")

        cap.release()

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ─── API stats ────────────────────────────────────────────────────────────────

@bp.route("/api/stats")
def api_stats():
    global _detector
    if _detector is None:
        return jsonify({"counts": {}, "total": 0, "frame": 0, "finished": False})

    counts = dict(_detector.counts_per_class)
    total  = sum(counts.values())

    return jsonify({
        "counts":    counts,
        "total":     total,
        "frame":     _detector.frame_index,
        "finished":  getattr(_detector, "finished", False),
        "timestamp": round((
            datetime.datetime.now() - _detector.start_time
        ).total_seconds(), 1) if _detector.start_time else 0
    })


# ─── Save logs ────────────────────────────────────────────────────────────────

@bp.route("/save_logs", methods=["POST"])
def save_logs():
    """Save logs from current detector state after streaming."""
    global _detector
    os.makedirs(LOGS_FOLDER, exist_ok=True)

    data       = request.get_json() or {}
    video      = data.get("video", "")
    scene_name = data.get("scene_name", "Unknown Scene")
    group_id   = data.get("group_id", "group_01")

    if _detector is None:
        return jsonify({"error": "No detector active"}), 400

    video_path = os.path.join(UPLOAD_FOLDER, secure_filename(video))

    try:
        global_stats = {
            "video_path":       video_path,
            "total_frames":     _detector.frame_index,
            "fps":              30,
            "duration_seconds": round(
                (datetime.datetime.now() - _detector.start_time).total_seconds(), 2
            ) if _detector.start_time else 0,
            "selected_classes": _detector.selected_class_names,
            "counts_per_class": dict(_detector.counts_per_class),
            "total_objects":    sum(_detector.counts_per_class.values()),
            "detection_logs":   _detector.detection_logs,
        }

        log_data  = build_shared_log(global_stats, scene_name, video_path, group_id)
        json_path = save_log_json(log_data, LOGS_FOLDER, scene_name)
        csv_path  = save_summary_csv(log_data, LOGS_FOLDER, scene_name)

        return jsonify({
            "success":  True,
            "log_json": json_path,
            "log_csv":  csv_path,
            "summary":  log_data["summary"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Download logs ────────────────────────────────────────────────────────────

@bp.route("/download_logs")
def download_logs():
    """Download the latest log file (JSON or CSV)."""
    fmt = request.args.get("format", "json")

    # Chemin absolu basé sur la racine du projet
    base_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "logs"))
    files = []
    if os.path.isdir(base_dir):
        files = sorted([f for f in os.listdir(base_dir) if f.endswith(f".{fmt}")])

    if not files:
        return f"No {fmt.upper()} log found. Please process a video first.", 404

    latest = os.path.join(base_dir, files[-1])
    return send_file(os.path.abspath(latest), as_attachment=True)


# ─── Dashboard ────────────────────────────────────────────────────────────────

@bp.route("/dashboard")
def dashboard():
    """Dashboard comparing all analyzed scenes."""
    all_logs    = load_and_merge_logs(LOGS_FOLDER)
    scenes_data = []

    for log in all_logs:
        scenes_data.append({
            "scene_name": log["scene"]["name"],
            "duration_s": log["scene"]["duration_s"],
            "counts":     log["summary"]["counts_per_class"],
            "total":      log["summary"]["total_unique_objects"],
            "temporal":   log.get("temporal_distribution", []),
            "generated":  log["generated_at"]
        })

    return render_template("dashboard.html", scenes=scenes_data,
                           scenes_json=json.dumps(scenes_data))
