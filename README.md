# Remote Processing Service

Service unifié de **transcription** (Whisper) + **diarization** (Pyannote) optimisé pour **AMD ROCm**.

**Nouvelles fonctionnalités v2.1 :**
- ⚡ Multi-threading : les endpoints `/health` et `/status` répondent même pendant un traitement
- 📊 Status en temps réel : progression, étape en cours, segments générés
- 🔒 Protection contre les requêtes simultanées (503 si déjà occupé)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Distrobox (llama-rocm-7.1.1)                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  FastAPI Server (uvicorn + asyncio.to_thread)       │    │
│  │  ├── /process   → Whisper + Pyannote combinés       │    │
│  │  ├── /transcribe → Whisper seul                     │    │
│  │  ├── /diarize   → Pyannote seul (rétrocompat)       │    │
│  │  ├── /health    → Status détaillé + job en cours    │    │
│  │  └── /status    → Status job uniquement (léger)     │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  GPU AMD (ROCm)                                     │    │
│  │  ├── Whisper large-v3-turbo (809M params)           │    │
│  │  └── Pyannote speaker-diarization-3.1               │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Prérequis

### Matériel
- GPU AMD compatible ROCm (RDNA 3 ou supérieur recommandé)
- 16 Go+ de RAM (128 Go de mémoire unifiée sur Strix Halo = optimal)

### Logiciels
- Linux avec Distrobox installé
- Conteneur `llama-rocm-7.1.1` créé et fonctionnel
- Token Hugging Face avec accès aux modèles :
  - `pyannote/speaker-diarization-3.1`
  - `pyannote/segmentation-3.0`

### Création du conteneur Distrobox (si pas déjà fait)

```bash
# Image ROCm optimisée pour Strix Halo avec ROCWMMA
distrobox create -n llama-rocm-7.1.1 \
  --image docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.1.1-rocwmma \
  --additional-flags "--device /dev/kfd --device /dev/dri --group-add video --group-add render --security-opt seccomp=unconfined -v /srv/transcript:/srv/transcript"
```

## Installation

```bash
# 1. Installer les dépendances
./start-server.sh --install

# 2. Vérifier l'accès GPU
distrobox enter llama-rocm-7.1.1 -- python3 -c "import torch; print(torch.cuda.is_available())"
# Doit afficher: True
```

## Démarrage

```bash
# Mode interactif (logs dans le terminal)
./start-server.sh

# Mode arrière-plan
./start-server.sh --background

# Arrêter le serveur
./start-server.sh --stop
```

Le serveur écoute sur `http://0.0.0.0:8000` par défaut.

## Endpoints

### `GET /health`
Retourne l'état du service. **Répond toujours**, même pendant un traitement en cours.

```json
{
  "status": "ok",
  "gpu_available": true,
  "device": "cuda",
  "gpu_name": "AMD Radeon RX 8060S",
  "memory_total_gb": 128.0,
  "memory_available_gb": 120.0,
  "models_loaded": {
    "whisper": true,
    "pyannote": true
  },
  "busy": true,
  "current_job": {
    "busy": true,
    "job_id": "a1b2c3d4",
    "filename": "interview.mp3",
    "stage": "transcribing",
    "progress": 35.0,
    "elapsed_seconds": 45.2,
    "stage_elapsed_seconds": 20.1,
    "transcription_segments": 0,
    "diarization_segments": 0,
    "error_message": null,
    "details": {"audio_duration_seconds": 180.5}
  }
}
```

### `GET /status`
Retourne uniquement le status du job en cours (plus léger que `/health`). Idéal pour le polling fréquent.

```json
{
  "busy": true,
  "job_id": "a1b2c3d4",
  "filename": "interview.mp3",
  "stage": "diarizing",
  "progress": 75.0,
  "elapsed_seconds": 120.5,
  "stage_elapsed_seconds": 45.3,
  "transcription_segments": 42,
  "diarization_segments": 15,
  "error_message": null,
  "details": {
    "audio_duration_seconds": 180.5,
    "speakers_detected": 3
  }
}
```

**Stages possibles :**
| Stage | Description |
|-------|-------------|
| `idle` | En attente |
| `uploading` | Réception du fichier |
| `loading` | Chargement des modèles |
| `transcribing` | Transcription Whisper en cours |
| `diarizing` | Diarization Pyannote en cours |
| `merging` | Fusion des segments |
| `completed` | Terminé avec succès |
| `error` | Erreur (voir `error_message`) |

### `POST /process` (recommandé)
Transcription + Diarization combinées.

> ⚠️ **Protection anti-concurrence** : Si un traitement est déjà en cours, retourne `503 Service Unavailable` avec le status du job actuel.

**Paramètres (form-data) :**
| Param | Type | Requis | Description |
|-------|------|--------|-------------|
| `file` | File | ✓ | Fichier audio (WAV, MP3, OGG...) |
| `hf_token` | string | ✓ | Token Hugging Face |
| `language` | string | | Code langue (ex: "fr", "en") |
| `min_speakers` | int | | Nombre minimum de locuteurs |
| `max_speakers` | int | | Nombre maximum de locuteurs |
| `skip_diarization` | bool | | Si true, transcription seule |

**Réponse (succès) :**
```json
{
  "text": "Transcription complète...",
  "segments": [
    {"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00", "text": "Bonjour..."}
  ],
  "transcription_segments": [...],
  "diarization_segments": [...]
}
```

**Réponse (503 - déjà occupé) :**
```json
{
  "error": "Service occupé",
  "message": "Un traitement est déjà en cours: interview.mp3",
  "current_job": {...}
}
```

### `POST /transcribe`
Transcription seule (sans token HF requis).

### `POST /diarize`
Diarization seule (compatibilité avec l'ancienne API).

## Configuration dans l'application PHP

Dans le panneau admin :
- **URL du service** : `http://IP_DU_SERVEUR:8000`
- **Token Hugging Face** : Votre token `hf_...`

## Performance

| Métrique | Valeur attendue (Strix Halo) |
|----------|------------------------------|
| Whisper v3-turbo | ~10x temps réel |
| Pyannote 3.1 | ~5x temps réel |
| Total (10 min audio) | ~2-3 min |

## Configuration avancée

### Variable `PYANNOTE_DEVICE`

Contrôle le device utilisé pour Pyannote (indépendamment de Whisper qui utilise toujours le GPU si disponible).

| Valeur | Description |
|--------|-------------|
| `auto` | (défaut) GPU si disponible, fallback CPU automatique pour GPUs AMD RDNA 3.5 |
| `cpu` | Force l'utilisation du CPU |
| `cuda` | Force l'utilisation du GPU |

Configurable via le fichier `.env` :
```bash
PYANNOTE_DEVICE=cpu
```

Ou en ligne de commande :
```bash
PYANNOTE_DEVICE=cpu ./start-server.sh
```

## Dépannage

### "No GPU available"
```bash
# Vérifier que les devices sont accessibles
ls -la /dev/dri /dev/kfd

# Vérifier que l'utilisateur est dans le groupe video
groups | grep video
```

### "miopenStatusUnknownError" ou erreurs MIOpen
Ce problème survient sur certains GPUs AMD récents (RDNA 3.5 : Radeon 8060S, 8050S, etc.) où MIOpen ne supporte pas encore toutes les opérations.

**Solution :** Forcer Pyannote sur CPU :
```bash
# Dans .env
PYANNOTE_DEVICE=cpu

# Ou en ligne de commande
PYANNOTE_DEVICE=cpu ./start-server.sh
```

Whisper continuera à utiliser le GPU, seul Pyannote utilisera le CPU.

### "Failed to load Pyannote"
- Vérifiez que le token HF est valide
- Acceptez les conditions d'utilisation sur https://huggingface.co/pyannote/speaker-diarization-3.1

### Logs
```bash
# Mode arrière-plan
tail -f server.log

# Mode interactif
# Les logs s'affichent dans le terminal
```
