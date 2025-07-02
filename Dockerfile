FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source FastAPI
COPY src/ ./src/

# Copier le dossier contenant le modèle
COPY model ./model

# Créer le dossier models si besoin
RUN mkdir -p ./models

# Définir le PYTHONPATH pour que les imports fonctionnent correctement
ENV PYTHONPATH=/app

# Variable d'environnement pour désactiver le buffering
ENV PYTHONUNBUFFERED=1

# Utilisateur non-root pour la sécurité
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Exposer le port de l'API
EXPOSE 8000

# Commande pour lancer FastAPI
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
