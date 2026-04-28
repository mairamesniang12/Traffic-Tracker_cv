"""
train.py — Fine-tuning YOLOv11s on a traffic dataset
Computer Vision Project — Traffic Monitoring

STEPS:
1. Create a free account on https://roboflow.com
2. Download a "Vehicle Detection" dataset (link in instructions)
3. Run: python train.py

The fine-tuned model will be saved in models/traffic_yolo11/weights/best.pt
"""

from ultralytics import YOLO
import os
import yaml


# ─── Training configuration ───────────────────────────────────────────────────

DATASET_YAML   = "dataset/data.yaml"
BASE_MODEL     = "yolo11s.pt"
OUTPUT_DIR     = "models"
PROJECT_NAME   = "traffic_yolo11"

TRAIN_CONFIG = {
    "epochs":      30,
    "imgsz":       640,
    "batch":       8,
    "lr0":         0.001,
    "patience":    10,
    "device":      "0",
    "workers":     2,
    "augment":     True,
    "cache":       False,
    "pretrained":  True,
    "verbose":     True,
}


def check_dataset():
    """Check that the dataset is present."""
    if not os.path.isfile(DATASET_YAML):
        print("""
[ERROR] Dataset not found!

How to get a free dataset:

OPTION 1 — Roboflow (easiest):
────────────────────────────────
1. Go to https://universe.roboflow.com
2. Search "vehicle detection" or "traffic detection"
3. Choose a dataset (e.g. "Vehicle Detection" by Roboflow)
4. Click "Download Dataset" → Format: YOLOv11 → "show download code"
5. Copy the Python code and run it in this folder
6. Rename the downloaded folder to "dataset"

OPTION 2 — Recommended dataset:
─────────────────────────────────
pip install roboflow

Then in Python:
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace("roboflow-100").project("vehicles-q0x2v")
    dataset = project.version(2).download("yolov11")

OPTION 3 — Annotate your own images:
──────────────────────────────────────
1. Extract frames from your videos with: python train.py --mode extract --video uploads/your_video.mp4
2. Annotate on https://www.makesense.ai (free, no account needed)
3. Export in YOLO format
        """)
        return False
    return True


def create_custom_yaml(classes=None):
    """Create a custom data.yaml for traffic classes only."""
    if classes is None:
        classes = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

    yaml_content = {
        "path":  os.path.abspath("dataset"),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    len(classes),
        "names": classes
    }

    os.makedirs("dataset", exist_ok=True)
    yaml_path = "dataset/data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"[INFO] data.yaml created: {yaml_path}")
    return yaml_path


def train():
    """Launch YOLOv11s fine-tuning."""
    if not check_dataset():
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 55)
    print("  Fine-tuning YOLOv11s — Traffic Detection")
    print("=" * 55)
    print(f"  Base model  : {BASE_MODEL}")
    print(f"  Dataset     : {DATASET_YAML}")
    print(f"  Epochs      : {TRAIN_CONFIG['epochs']}")
    print(f"  Batch size  : {TRAIN_CONFIG['batch']}")
    print(f"  Image size  : {TRAIN_CONFIG['imgsz']}")
    print("=" * 55 + "\n")

    model = YOLO(BASE_MODEL)

    results = model.train(
        data      = DATASET_YAML,
        epochs    = TRAIN_CONFIG["epochs"],
        imgsz     = TRAIN_CONFIG["imgsz"],
        batch     = TRAIN_CONFIG["batch"],
        lr0       = TRAIN_CONFIG["lr0"],
        patience  = TRAIN_CONFIG["patience"],
        device    = TRAIN_CONFIG["device"],
        workers   = TRAIN_CONFIG["workers"],
        augment   = TRAIN_CONFIG["augment"],
        cache     = TRAIN_CONFIG["cache"],
        pretrained= TRAIN_CONFIG["pretrained"],
        project   = OUTPUT_DIR,
        name      = PROJECT_NAME,
        verbose   = TRAIN_CONFIG["verbose"],
    )

    best_model = os.path.join(OUTPUT_DIR, PROJECT_NAME, "weights", "best.pt")

    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE!")
    print("=" * 55)
    print(f"  Best model : {best_model}")
    print(f"  mAP50      : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.3f}")
    print(f"  mAP50-95   : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
    print("=" * 55)
    print()
    print("  To use this model in the app,")
    print("  update MODEL_PATH in app/routes.py:")
    print(f'  MODEL_PATH = "{best_model}"')
    print()

    return best_model


def evaluate(model_path=None):
    """Evaluate the model on the test set."""
    if model_path is None:
        model_path = os.path.join(OUTPUT_DIR, PROJECT_NAME, "weights", "best.pt")

    if not os.path.isfile(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return

    print(f"\n[INFO] Evaluating model: {model_path}")
    model = YOLO(model_path)

    metrics = model.val(data=DATASET_YAML, imgsz=640, device=TRAIN_CONFIG["device"])

    print("\n── Evaluation Metrics ──")
    print(f"  mAP50     : {metrics.box.map50:.3f}")
    print(f"  mAP50-95  : {metrics.box.map:.3f}")
    print(f"  Precision : {metrics.box.mp:.3f}")
    print(f"  Recall    : {metrics.box.mr:.3f}")

    return metrics


def extract_frames(video_path, output_dir="dataset_frames", every_n=30):
    """
    Extract frames from a video to build your own dataset.
    Useful for annotating your own Dakar traffic videos.

    Args:
        video_path: path to the video
        output_dir: output folder for frames
        every_n:    extract 1 frame every N frames (default: 1 image/second at 30fps)
    """
    import cv2

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    saved = 0

    print(f"[INFO] Extracting frames from {video_path}")
    print(f"       1 frame every {every_n} frames → ~{total // every_n} images")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n == 0:
            fname = os.path.join(output_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(fname, frame)
            saved += 1
        count += 1

    cap.release()
    print(f"[INFO] {saved} frames extracted to {output_dir}/")
    print(f"       → Annotate them at https://www.makesense.ai")
    return saved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv11s Fine-tuning for Traffic Detection")
    parser.add_argument("--mode", choices=["train", "eval", "extract"],
                        default="train", help="Mode: train / eval / extract")
    parser.add_argument("--video", help="Source video (extract mode)")
    parser.add_argument("--model", help="Model path (eval mode)")
    parser.add_argument("--every", type=int, default=30,
                        help="Extract 1 frame every N frames (extract mode)")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "eval":
        evaluate(args.model)
    elif args.mode == "extract":
        if not args.video:
            print("[ERROR] Please specify --video for extract mode")
        else:
            extract_frames(args.video, every_n=args.every)
