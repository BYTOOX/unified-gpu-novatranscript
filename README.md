# Remote Diarization Service

Ce service expose une API pour effectuer la diarisation (reconnaissance des locuteurs) via **Pyannote Audio 3.1**.
Il est conçu pour être hébergé sur une machine puissante (idéalement avec GPU Nvidia), séparée de l'application principale PHP.

## Prérequis
- Docker & Docker Compose
- Un token Hugging Face (HF) valide avec accès accepté aux modèles :
  - `pyannote/speaker-diarization-3.1`
  - `pyannote/segmentation-3.0`

## Installation & Démarrage

1. **Construire l'image Docker**
   ```bash
   docker build -t remote-diarization .
   ```

2. **Lancer le conteneur**
   
   *Sur CPU (votre Ryzen 9) :*
   ```bash
   docker run -d -p 8000:8000 --name diarization-service remote-diarization
   ```

   *Sur GPU (si disponible plus tard) :*
   ```bash
   docker run -d --gpus all -p 8000:8000 --name diarization-service remote-diarization
   ```

## Test
L'API sera accessible sur `http://VOTRE_IP:8000`.

Endpoint de santé :
`GET /health`

## Utilisation dans l'application
Dans le panneau admin de l'application PHP, configurez :
- URL du service : `http://VOTRE_IP:8000` (ex: `http://192.168.1.50:8000`)
- Token Hugging Face : Votre token `hf_...`

