"""
detector.py — YOLO Detection + ByteTrack Tracking + Unique Counting
Computer Vision Project — Traffic Monitoring
"""

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from datetime import datetime
import json
import os

# ─── Traffic classes to detect ───────────────────────────────────────────────
# Matching COCO class IDs used by YOLOv11
TRAFFIC_CLASSES = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
}

# Colors per class (BGR for OpenCV)
CLASS_COLORS = {
    "person":     (255, 100,  50),
    "bicycle":    ( 50, 220, 100),
    "car":        ( 50, 150, 255),
    "motorcycle": (200,  50, 255),
    "bus":        (255, 200,   0),
    "truck":      (  0, 180, 255),
}


class TrafficDetector:
    """
    Detects, tracks and counts traffic objects in a video.
    Uses YOLOv11s + ByteTrack from supervision.
    """

    def __init__(self, model_path="yolo11s.pt", selected_classes=None, conf_threshold=0.3):
        """
        Args:
            model_path:        path to YOLO weights (auto-downloaded if missing)
            selected_classes:  list of class names to track. None = all.
            conf_threshold:    minimum confidence threshold (0.0 to 1.0)
        """
        print(f"[INFO] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

        # Selected classes (user filter)
        if selected_classes:
            self.selected_class_ids = [
                cid for cid, name in TRAFFIC_CLASSES.items()
                if name in selected_classes
            ]
            self.selected_class_names = selected_classes
        else:
            self.selected_class_ids = list(TRAFFIC_CLASSES.keys())
            self.selected_class_names = list(TRAFFIC_CLASSES.values())

        # ByteTrack tracker via supervision
        self.tracker = sv.ByteTrack()

        # Supervision annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_padding=4
        )

        # Unique counting by tracker ID
        self.counted_ids = set()
        self.counts_per_class = {name: 0 for name in self.selected_class_names}

        # Detection logs
        self.detection_logs = []
        self.frame_index = 0
        self.start_time = datetime.now()
        self.finished = False

        # Counting line (middle of frame, horizontal)
        self.counting_line_y = None

        print(f"[INFO] Selected classes: {self.selected_class_names}")
        print(f"[INFO] Tracker: ByteTrack | Confidence threshold: {conf_threshold}")

    def process_frame(self, frame):
        """
        Process a video frame: detection → tracking → counting → annotation.

        Args:
            frame: OpenCV image (numpy array BGR)

        Returns:
            annotated_frame: frame with boxes and IDs drawn
            frame_stats:     dict with current frame stats
        """
        h, w = frame.shape[:2]

        if self.counting_line_y is None:
            self.counting_line_y = h // 2

        self.frame_index += 1
        timestamp = (datetime.now() - self.start_time).total_seconds()

        # ── 1. YOLO Detection ─────────────────────────────────────────────
        results = self.model(
            frame,
            classes=self.selected_class_ids,
            conf=self.conf_threshold,
            verbose=False
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        # ── 2. ByteTrack Tracking ─────────────────────────────────────────
        detections = self.tracker.update_with_detections(detections)

        # ── 3. Unique counting (crossing the center line) ─────────────────
        frame_new_counts = {}
        frame_detections_log = []

        for i in range(len(detections)):
            tracker_id = detections.tracker_id[i] if detections.tracker_id is not None else None
            class_id   = int(detections.class_id[i])
            confidence = float(detections.confidence[i])
            bbox       = detections.xyxy[i].tolist()

            class_name = TRAFFIC_CLASSES.get(class_id, "unknown")

            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int(bbox[3])

            if tracker_id is not None and tracker_id not in self.counted_ids:
                if cy >= self.counting_line_y:
                    self.counted_ids.add(tracker_id)
                    if class_name in self.counts_per_class:
                        self.counts_per_class[class_name] += 1
                        frame_new_counts[class_name] = frame_new_counts.get(class_name, 0) + 1

            frame_detections_log.append({
                "tracker_id": int(tracker_id) if tracker_id is not None else None,
                "class_id":   class_id,
                "class_name": class_name,
                "confidence": round(confidence, 3),
                "bbox":       [round(v, 1) for v in bbox],
                "center":     [cx, cy]
            })

        if frame_detections_log:
            self.detection_logs.append({
                "frame":      self.frame_index,
                "timestamp":  round(timestamp, 3),
                "detections": frame_detections_log,
                "new_counts": frame_new_counts
            })

        # ── 4. Frame annotation ───────────────────────────────────────────
        annotated = frame.copy()

        if detections.tracker_id is not None and len(detections) > 0:
            labels = []
            for i in range(len(detections)):
                cid   = int(detections.class_id[i])
                cname = TRAFFIC_CLASSES.get(cid, "?")
                tid   = detections.tracker_id[i] if detections.tracker_id is not None else "?"
                conf  = float(detections.confidence[i])
                labels.append(f"{cname} #{tid} {conf:.0%}")

            annotated = self.box_annotator.annotate(annotated, detections)
            annotated = self.label_annotator.annotate(annotated, detections, labels=labels)

        # Counting line
        cv2.line(annotated, (0, self.counting_line_y), (w, self.counting_line_y),
                 (0, 255, 255), 2)
        cv2.putText(annotated, "Counting line", (10, self.counting_line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Counters panel
        annotated = self._draw_counters(annotated)

        # No object message
        if len(detections) == 0:
            cv2.putText(annotated, "No object detected", (w//2 - 120, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)

        frame_stats = {
            "frame":        self.frame_index,
            "timestamp":    round(timestamp, 3),
            "n_detections": len(detections),
            "counts":       dict(self.counts_per_class)
        }

        return annotated, frame_stats

    def _draw_counters(self, frame):
        """Draw a counters panel on the frame."""
        panel_x, panel_y = 10, 10
        line_h  = 22
        padding = 8
        n_lines = len(self.counts_per_class) + 1

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (panel_x, panel_y),
                      (panel_x + 180, panel_y + n_lines * line_h + padding),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Title
        cv2.putText(frame, "COUNTERS",
                    (panel_x + padding, panel_y + line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Count per class
        for i, (cname, count) in enumerate(self.counts_per_class.items()):
            color = CLASS_COLORS.get(cname, (200, 200, 200))
            y = panel_y + (i + 2) * line_h
            cv2.putText(frame, f"{cname}: {count}",
                        (panel_x + padding, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        return frame

    def process_video(self, video_path, output_path=None, show_preview=False):
        """
        Process a full video frame by frame.

        Args:
            video_path:   path to source video
            output_path:  path to save annotated video (None = no save)
            show_preview: show preview window (useful locally)

        Returns:
            dict with global stats and full logs
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[INFO] Video: {width}x{height} @ {fps:.1f}fps | {total} frames")

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Reset counters
        self.counted_ids.clear()
        self.counts_per_class = {name: 0 for name in self.selected_class_names}
        self.detection_logs.clear()
        self.frame_index = 0
        self.start_time = datetime.now()
        self.counting_line_y = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, stats = self.process_frame(frame)

            if writer:
                writer.write(annotated)

            if show_preview:
                cv2.imshow("Traffic Monitor", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if self.frame_index % 50 == 0:
                pct = (self.frame_index / total * 100) if total > 0 else 0
                print(f"  [Frame {self.frame_index}/{total}] {pct:.1f}% | Counts: {self.counts_per_class}")

        cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()

        total_objects = sum(self.counts_per_class.values())
        global_stats = {
            "video_path":       video_path,
            "total_frames":     self.frame_index,
            "fps":              fps,
            "duration_seconds": round(self.frame_index / fps, 2) if fps > 0 else 0,
            "selected_classes": self.selected_class_names,
            "counts_per_class": self.counts_per_class,
            "total_objects":    total_objects,
            "detection_logs":   self.detection_logs,
        }

        print(f"\n[RESULTS] Total unique objects: {total_objects}")
        print(f"  Details: {self.counts_per_class}")

        return global_stats

    def generate_frame_generator(self, video_path):
        """
        Frame generator for Flask MJPEG streaming.
        Reads the video once then stops.
        Yields: JPEG-encoded bytes of each annotated frame.
        """
        self.counted_ids.clear()
        self.counts_per_class = {name: 0 for name in self.selected_class_names}
        self.detection_logs.clear()
        self.frame_index = 0
        self.start_time = datetime.now()
        self.counting_line_y = None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # end of video → stop

            annotated, stats = self.process_frame(frame)

            ret2, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret2:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       buffer.tobytes() + b"\r\n")

        cap.release()
        self.finished = True  # signal that video is done