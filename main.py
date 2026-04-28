"""
main.py — Script standalone pour traiter une vidéo en ligne de commande
Projet Computer Vision — Traffic Monitoring

Usage :
    python main.py --video uploads/traffic.mp4
    python main.py --video uploads/traffic.mp4 --classes car bus truck
    python main.py --video uploads/traffic.mp4 --preview
    python main.py --video uploads/traffic.mp4 --model models/yolo11s.pt
"""

import argparse
import os
import sys

from app.detector import TrafficDetector, TRAFFIC_CLASSES
from app.logger   import build_shared_log, save_log_json, save_summary_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Traffic Monitor — Détection & comptage de trafic routier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python main.py --video uploads/traffic.mp4
  python main.py --video uploads/traffic.mp4 --classes car motorcycle bus
  python main.py --video uploads/traffic.mp4 --preview --conf 0.4
  python main.py --video uploads/traffic.mp4 --scene "Dakar - Rond-point"
        """
    )

    parser.add_argument(
        "--video", "-v",
        required=True,
        help="Chemin vers la vidéo à traiter"
    )
    parser.add_argument(
        "--classes", "-c",
        nargs="+",
        choices=list(TRAFFIC_CLASSES.values()),
        default=None,
        help=f"Classes à détecter. Défaut : toutes. Choix : {list(TRAFFIC_CLASSES.values())}"
    )
    parser.add_argument(
        "--model", "-m",
        default="models/yolo11n.pt",
        help="Chemin vers les poids YOLO (défaut : models/yolo11n.pt)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Seuil de confiance YOLO (0.0 à 1.0, défaut : 0.3)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Chemin de la vidéo annotée en sortie (défaut : outputs/annotated_<nom>.mp4)"
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Nom de la scène pour les logs (défaut : nom du fichier vidéo)"
    )
    parser.add_argument(
        "--group",
        default="group_01",
        help="Identifiant du groupe pour les logs partagés (défaut : group_01)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Afficher la fenêtre de prévisualisation pendant le traitement"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Ne pas sauvegarder la vidéo annotée (logs uniquement)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Vérifications ─────────────────────────────────────────────────────
    if not os.path.isfile(args.video):
        print(f"[ERREUR] Vidéo introuvable : {args.video}")
        sys.exit(1)

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # ── Paramètres ────────────────────────────────────────────────────────
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    scene_name = args.scene or f"Scene - {video_name}"

    output_path = None
    if not args.no_save:
        output_path = args.output or f"outputs/annotated_{video_name}.mp4"

    print("\n" + "=" * 55)
    print("  Traffic Monitor — Computer Vision Project 2026")
    print("=" * 55)
    print(f"  Vidéo      : {args.video}")
    print(f"  Scène      : {scene_name}")
    print(f"  Classes    : {args.classes or 'toutes'}")
    print(f"  Modèle     : {args.model}")
    print(f"  Confiance  : {args.conf}")
    print(f"  Sortie     : {output_path or 'aucune sauvegarde'}")
    print(f"  Prévisual. : {'oui' if args.preview else 'non'}")
    print("=" * 55 + "\n")

    # ── Détection + Tracking ──────────────────────────────────────────────
    detector = TrafficDetector(
        model_path=args.model,
        selected_classes=args.classes,
        conf_threshold=args.conf
    )

    global_stats = detector.process_video(
        video_path=args.video,
        output_path=output_path,
        show_preview=args.preview
    )

    # ── Export des logs ───────────────────────────────────────────────────
    log_data  = build_shared_log(global_stats, scene_name, args.video, args.group)
    json_path = save_log_json(log_data, "outputs/logs", scene_name)
    csv_path  = save_summary_csv(log_data, "outputs/logs", scene_name)

    # ── Résumé final ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RÉSULTATS FINAUX")
    print("=" * 55)
    print(f"  Scène           : {scene_name}")
    print(f"  Durée traitée   : {global_stats['duration_seconds']} secondes")
    print(f"  Frames total    : {global_stats['total_frames']}")
    print(f"  Objets uniques  : {global_stats['total_objects']}")
    print()
    print("  Détail par classe :")
    for cname, count in global_stats["counts_per_class"].items():
        bar = "█" * min(count, 30)
        print(f"    {cname:<12} {count:>4}  {bar}")
    print()
    print(f"  Logs JSON : {json_path}")
    print(f"  Logs CSV  : {csv_path}")
    if output_path:
        print(f"  Vidéo     : {output_path}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
