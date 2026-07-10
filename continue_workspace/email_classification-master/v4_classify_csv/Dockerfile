# ============================================================
# Dockerfile — Email Classifier (CPU)
# ============================================================
FROM python:3.11-slim

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY train.py predict.py ./

# Créer les dossiers nécessaires
RUN mkdir -p /app/models /app/data

# Pré-télécharger les données NLTK au build
RUN python -c "import nltk; [nltk.download(p, quiet=True) for p in ['stopwords','punkt','punkt_tab']]"

# Port de l'API
EXPOSE 8000

# Par défaut : lancer l'API
# Pour entraîner : docker run ... python train.py train --csv /app/data/emails.csv
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
