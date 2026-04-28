"""
run.py — Point d'entrée principal de l'application Flask
Usage : python run.py
"""

from app import create_app

app = create_app()

if __name__ == "__main__":
    print("=" * 55)
    print("  Traffic Monitor — Computer Vision Project 2026")
    print("  Démarrage sur http://localhost:5000")
    print("=" * 55)
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
