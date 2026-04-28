"""
logger.py — Génération des logs JSON selon le schéma partagé du groupe
Projet Computer Vision — Traffic Monitoring
"""

import json
import os
import csv
from datetime import datetime


# ─── Schéma partagé entre tous les groupes ────────────────────────────────────
# Chaque groupe doit respecter ce format pour fusionner dans le dashboard global

def build_shared_log(global_stats, scene_name, video_path, group_id="group_01"):
    """
    Construit le log au format partagé entre tous les groupes.

    Args:
        global_stats: dict retourné par TrafficDetector.process_video()
        scene_name:   nom descriptif de la scène (ex: "Dakar - Rond-point Étoile")
        video_path:   chemin de la vidéo source
        group_id:     identifiant du groupe

    Returns:
        dict formaté selon le schéma partagé
    """
    now = datetime.now().isoformat()

    # ── Statistiques temporelles : distribution par tranche de 10s ───────────
    temporal_distribution = compute_temporal_distribution(
        global_stats["detection_logs"],
        interval_seconds=10
    )

    log = {
        # ── Métadonnées du groupe et de la scène ──────────────────────────
        "schema_version": "1.0",
        "group_id":       group_id,
        "generated_at":   now,
        "scene": {
            "name":        scene_name,
            "video_file":  os.path.basename(video_path),
            "duration_s":  global_stats["duration_seconds"],
            "fps":         global_stats["fps"],
            "total_frames": global_stats["total_frames"],
            "location":    "Dakar, Sénégal",         # à adapter
            "scene_type":  "intersection",            # intersection / roundabout / highway / urban
            "recorded_at": now                        # idéalement l'heure réelle
        },

        # ── Comptages globaux ─────────────────────────────────────────────
        "summary": {
            "total_unique_objects": global_stats["total_objects"],
            "counts_per_class":     global_stats["counts_per_class"],
            "selected_classes":     global_stats["selected_classes"],
        },

        # ── Distribution temporelle (pour les graphes du dashboard) ───────
        "temporal_distribution": temporal_distribution,

        # ── Logs détaillés frame par frame ────────────────────────────────
        "frame_logs": [
            {
                "frame":      entry["frame"],
                "timestamp":  entry["timestamp"],
                "detections": [
                    {
                        "tracker_id": d["tracker_id"],
                        "class_name": d["class_name"],
                        "confidence": d["confidence"],
                        "bbox":       d["bbox"],      # [x1, y1, x2, y2]
                        "center":     d["center"]     # [cx, cy]
                    }
                    for d in entry["detections"]
                ]
            }
            for entry in global_stats["detection_logs"]
        ]
    }

    return log


def compute_temporal_distribution(detection_logs, interval_seconds=10):
    """
    Calcule la distribution du nombre de véhicules par tranche de temps.

    Args:
        detection_logs: liste des logs de frames
        interval_seconds: durée de chaque tranche (en secondes)

    Returns:
        liste de dicts { "interval_start", "interval_end", "counts_per_class", "total" }
    """
    if not detection_logs:
        return []

    max_t = max(entry["timestamp"] for entry in detection_logs)
    intervals = []
    t = 0.0

    while t < max_t:
        t_end = t + interval_seconds

        # Frames dans cet intervalle
        frames_in_interval = [
            entry for entry in detection_logs
            if t <= entry["timestamp"] < t_end
        ]

        # Compter les objets uniques dans cet intervalle
        ids_seen = {}
        for entry in frames_in_interval:
            for det in entry["detections"]:
                tid = det.get("tracker_id")
                cname = det["class_name"]
                if tid is not None and tid not in ids_seen:
                    ids_seen[tid] = cname

        counts = {}
        for cname in ids_seen.values():
            counts[cname] = counts.get(cname, 0) + 1

        intervals.append({
            "interval_start": round(t, 1),
            "interval_end":   round(t_end, 1),
            "counts_per_class": counts,
            "total": sum(counts.values())
        })

        t = t_end

    return intervals


def save_log_json(log_data, output_dir, scene_name):
    """
    Sauvegarde le log en JSON dans le dossier de sortie.

    Returns:
        chemin du fichier créé
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_name = scene_name.replace(" ", "_").replace("/", "-").lower()
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename   = f"log_{safe_name}_{timestamp}.json"
    filepath   = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    print(f"[LOG] Sauvegardé : {filepath}")
    return filepath


def save_summary_csv(log_data, output_dir, scene_name):
    """
    Sauvegarde un résumé CSV des comptages par classe.
    Utile pour le dashboard global.

    Returns:
        chemin du fichier créé
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_name = scene_name.replace(" ", "_").replace("/", "-").lower()
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename   = f"summary_{safe_name}_{timestamp}.csv"
    filepath   = os.path.join(output_dir, filename)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group_id", "scene_name", "class_name", "count",
                         "duration_s", "generated_at"])

        gid       = log_data["group_id"]
        dur       = log_data["scene"]["duration_s"]
        gen       = log_data["generated_at"]
        for cname, count in log_data["summary"]["counts_per_class"].items():
            writer.writerow([gid, scene_name, cname, count, dur, gen])

    print(f"[CSV]  Sauvegardé : {filepath}")
    return filepath


def load_and_merge_logs(log_dir):
    """
    Charge tous les fichiers JSON du dossier et les fusionne.
    Utilisé par le dashboard global.

    Returns:
        liste de tous les logs chargés
    """
    all_logs = []
    if not os.path.isdir(log_dir):
        return all_logs

    for fname in os.listdir(log_dir):
        if fname.endswith(".json"):
            fpath = os.path.join(log_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    all_logs.append(data)
            except Exception as e:
                print(f"[WARN] Impossible de lire {fname} : {e}")

    print(f"[MERGE] {len(all_logs)} logs chargés depuis {log_dir}")
    return all_logs