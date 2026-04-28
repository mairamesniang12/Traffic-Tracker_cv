<<<<<<< HEAD
# 🚦 Traffic Monitor — Road Traffic Object Detection & Tracking

> Computer Vision Project 2 — AIMS Senegal, April 2026  
> Real-time detection, tracking, and counting of road traffic objects across multiple video scenes.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Data Schema](#data-schema)
- [Dashboard](#dashboard)
- [Results](#results)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)

---

## Overview

This system automatically detects, tracks, and counts road traffic objects (cars, buses, trucks, motorcycles, bicycles, and pedestrians) in video footage using:

- **YOLOv11** for real-time object detection
- **ByteTrack** (via Supervision) for persistent multi-object tracking
- **Flask** for the web interface with live MJPEG streaming
- **Chart.js** for the global comparison dashboard

The project contributes to a shared dataset that helps public institutions in Senegal make data-driven decisions about road infrastructure.

---

## Features

- ✅ Real-time detection with YOLOv11 (pretrained on COCO)
- ✅ Persistent object tracking with unique IDs (ByteTrack)
- ✅ Unique object counting via virtual counting line
- ✅ Supports 6 traffic classes: car, bus, truck, motorcycle, bicycle, person
- ✅ Live MJPEG video streaming in browser
- ✅ JSON + CSV log export (shared schema)
- ✅ Multi-scene comparison dashboard
- ✅ "No object detected" indicator
- ✅ Temporal traffic distribution analysis

---

## Project Structure

```
traffic_project/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── detector.py          # YOLO + ByteTrack detection & tracking
│   ├── logger.py            # Log generation (JSON/CSV shared schema)
│   ├── routes.py            # Flask routes & API endpoints
│   └── templates/
│       ├── index.html       # Upload page & class selection
│       ├── visualize.html   # Live visualization with counters
│       └── dashboard.html   # Multi-scene comparison dashboard
├── uploads/                 # Input videos (created automatically)
├── outputs/                 # Annotated videos + logs (created automatically)
│   └── logs/                # JSON & CSV logs per scene
├── models/                  # YOLO weights (downloaded automatically)
├── run.py                   # Application entry point
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.9+
- pip
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/traffic-monitor.git
cd traffic-monitor

# 2. Create a virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python run.py
```

The app starts at **http://localhost:5000**  
The YOLOv11 model (~6MB) downloads automatically on first run.

---

## Usage

### 1. Upload a video
Go to `http://localhost:5000`, upload a traffic video (mp4, avi, mov, mkv).

### 2. Select classes
Choose which object types to detect: car, bus, truck, motorcycle, bicycle, person.

### 3. Name the scene
Give a descriptive name (e.g. "Dakar - Rond-point Étoile").

### 4. Watch live
The `/visualize` page streams the annotated video with:
- Bounding boxes + class labels + tracker IDs
- Live counters per class
- Counting line in the middle of the frame
- "No object detected" overlay when the scene is empty

### 5. View the dashboard
Go to `/dashboard` to compare all analyzed scenes with charts.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Home page |
| POST | `/upload` | Upload a video |
| GET | `/visualize` | Live visualization page |
| GET | `/video_feed` | MJPEG stream |
| GET | `/api/stats` | Current counters (JSON) |
| POST | `/process` | Full processing + log export |
| GET | `/dashboard` | Global dashboard |

---

## Models

| Model | Size | Speed | mAP |
|-------|------|-------|-----|
| YOLOv11n (default) | ~6MB | ~45 FPS | 39.5 |
| YOLOv11s | ~19MB | ~35 FPS | 47.0 |
| YOLOv11m | ~40MB | ~25 FPS | 51.5 |

To use a larger model, change `MODEL_PATH` in `app/routes.py`:
```python
MODEL_PATH = "models/yolo11s.pt"   # or yolo11m.pt
```

---

## Data Schema

All logs follow a shared JSON schema for merging across groups:

```json
{
  "schema_version": "1.0",
  "group_id": "group_01",
  "generated_at": "2026-04-27T14:30:00",
  "scene": {
    "name": "Dakar - Rond-point Étoile",
    "video_file": "traffic_dakar.mp4",
    "duration_s": 62.5,
    "fps": 30.0,
    "scene_type": "roundabout",
    "location": "Dakar, Sénégal"
  },
  "summary": {
    "total_unique_objects": 142,
    "counts_per_class": {
      "car": 87, "bus": 12, "truck": 8,
      "motorcycle": 31, "bicycle": 4, "person": 0
    }
  },
  "temporal_distribution": [
    {
      "interval_start": 0.0,
      "interval_end": 10.0,
      "counts_per_class": {"car": 15, "motorcycle": 6},
      "total": 21
    }
  ],
  "frame_logs": [...]
}
```

---

## Dashboard

The global dashboard at `/dashboard` shows:
- Total unique objects across all scenes
- Per-class bar chart (aggregated)
- Scene comparison donut chart
- Temporal traffic distribution line chart

To merge logs from multiple groups, place all JSON files in `outputs/logs/`.

---

## Results

### Scene 1 — [Scene name]
- Duration: X min
- Total unique objects: X
- Peak traffic: X objects/10s at Xs

### Scene 2 — [Scene name]
- Duration: X min
- Total unique objects: X
- Peak traffic: X objects/10s at Xs

*(Screenshots and plots to be added in the final report)*

---

## Limitations & Future Work

**Current limitations:**
- Counting line is fixed at the middle of the frame (not configurable via UI)
- No speed estimation
- No direction tracking (in/out)
- Performance drops on low-end hardware with high-res videos

**Potential improvements:**
- 🔥 Speed estimation using camera calibration
- 🗺️ Heatmap of object positions over time
- ↕️ Direction detection (entering vs leaving)
- 🎯 Fine-tuning YOLO on Dakar-specific traffic data
- 📱 Mobile-responsive interface improvements

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ for AIMS Senegal — Computer Vision 2026*
=======
# traffic-monitor-cv
This project focuses on developing a real-time computer vision system dedicated to the detection, tracking, and counting of road traffic objects in multiple video scenes.
>>>>>>> 0a54b687e02d7d6808d00b1dad63c9350a831bdd
