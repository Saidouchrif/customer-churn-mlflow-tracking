# Dockerfile pour l'API de Prédiction du Churn Bancaire
# Base image Python 3.11 (plus stable que 3.13 pour les packages ML)
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="ML Team"
LABEL description="API FastAPI pour la prédiction du churn bancaire"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV PORT=8000

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers de requirements
COPY requirements_py313.txt requirements.txt

# Installation des dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn[standard] requests

# Création des dossiers nécessaires
RUN mkdir -p /app/models /app/reports /app/mlruns /app/Apis

# Copie des fichiers de l'application
COPY models/ /app/models/
COPY Apis/ /app/Apis/
COPY notebooks/ /app/notebooks/
COPY data/ /app/data/

# Copie des fichiers de configuration
COPY requirements*.txt /app/
COPY README.md /app/

# Permissions
RUN chmod +x /app/Apis/churn_api.py

# Exposition du port
EXPOSE $PORT

# Vérification de santé
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Point d'entrée
CMD ["python", "/app/Apis/churn_api.py"]