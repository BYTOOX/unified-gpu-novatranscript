# Image de base PyTorch officielle (optimisée pour CUDA si dispo, mais fonctionne CPU)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Installation des dépendances système (libsndfile pour l'audio)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copie des requirements
COPY requirements.txt .

# Installation des dépendances Python
# pyannote.audio nécessite une version spécifique de torch parfois, mais l'image de base aide.
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code serveur
COPY server.py .

# Création du dossier temporaire pour les uploads
RUN mkdir -p /tmp/uploads

# Exposition du port
EXPOSE 8000

# Lancement du serveur Uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

