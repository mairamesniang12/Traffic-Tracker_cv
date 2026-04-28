"""
__init__.py — Création de l'application Flask
"""

import os
from flask import Flask


def create_app():
    app = Flask(__name__, template_folder="templates")
    app.secret_key = "traffic-cv-project-2026"

    # Créer les dossiers nécessaires
    for folder in ["uploads", "outputs", "outputs/logs", "models"]:
        os.makedirs(folder, exist_ok=True)

    # Enregistrer les routes
    from .routes import bp
    app.register_blueprint(bp)

    return app