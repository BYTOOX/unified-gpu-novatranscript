"""
Remote Processing Service - Transcription + Diarization
Optimisé pour AMD ROCm avec mémoire unifiée

Endpoints:
    POST /process   - Transcription + Diarization combinées
    POST /diarize   - Diarization seule (rétrocompatibilité)
    POST /transcribe - Transcription seule
    GET /health     - Status détaillé du service
    GET /status     - Status du job en cours (toujours disponible)
"""

import os
import shutil
import logging
import tempfile
import warnings
import asyncio
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

# Filtrer les warnings répétitifs de ROCm/HIP
warnings.filterwarnings("ignore", message=".*bgemm_internal_cublaslt.*")
warnings.filterwarnings("ignore", message=".*Flash Efficient attention.*experimental.*")
warnings.filterwarnings("ignore", message=".*Mem Efficient attention.*experimental.*")

import torch

# ============================================================================
# Monkey-patches pour compatibilité PyTorch 2.6+ et torchaudio >= 2.0 avec ROCm
# ============================================================================

# Patch pour PyTorch 2.6+: autoriser le chargement des modèles Pyannote
# PyTorch 2.6 a changé weights_only=True par défaut, ce qui casse Pyannote
# lightning_fabric passe explicitement weights_only=True, donc on doit forcer False
import torch.serialization

_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    # Forcer weights_only=False pour tous les appels
    # Nécessaire car Pyannote/lightning passent explicitement weights_only=True
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

import torchaudio

# Patch 1: set_audio_backend a été supprimé dans torchaudio 2.0+
if not hasattr(torchaudio, 'set_audio_backend'):
    def _dummy_set_audio_backend(backend):
        pass
    torchaudio.set_audio_backend = _dummy_set_audio_backend

# Patch 2: list_audio_backends a été supprimé dans torchaudio 2.0+
if not hasattr(torchaudio, 'list_audio_backends'):
    def _dummy_list_audio_backends():
        return ['soundfile', 'sox']  # Retourner les backends courants
    torchaudio.list_audio_backends = _dummy_list_audio_backends

# Patch 3: get_audio_backend a été supprimé dans torchaudio 2.0+
if not hasattr(torchaudio, 'get_audio_backend'):
    def _dummy_get_audio_backend():
        return 'soundfile'
    torchaudio.get_audio_backend = _dummy_get_audio_backend

# Patch 4: AudioMetaData est maintenant dans torchaudio._backend.utils
if not hasattr(torchaudio, 'AudioMetaData'):
    try:
        from torchaudio._backend.utils import AudioMetaData
        torchaudio.AudioMetaData = AudioMetaData
    except ImportError:
        class AudioMetaData:
            def __init__(self, sample_rate=0, num_frames=0, num_channels=0, 
                         bits_per_sample=0, encoding=None):
                self.sample_rate = sample_rate
                self.num_frames = num_frames
                self.num_channels = num_channels
                self.bits_per_sample = bits_per_sample
                self.encoding = encoding
        torchaudio.AudioMetaData = AudioMetaData

# Patch 5: torchaudio.info a été supprimé dans torchaudio 2.0+
# Pyannote l'utilise pour obtenir les métadonnées audio
# On doit charger l'audio pour avoir les vraies dimensions (cohérent avec load)
if not hasattr(torchaudio, 'info'):
    def _patched_torchaudio_info(filepath, backend=None):
        """
        Remplacement de torchaudio.info pour torchaudio 2.0+
        Charge l'audio pour obtenir les métadonnées exactes (cohérent avec load).
        """
        import soundfile as sf
        
        filepath_str = str(filepath)
        
        class AudioInfo:
            def __init__(self, sample_rate, num_frames, num_channels):
                self.sample_rate = sample_rate
                self.num_frames = num_frames
                self.num_channels = num_channels
        
        # Essayer soundfile d'abord
        try:
            audio_data, sample_rate = sf.read(filepath_str, dtype='float32')
            if len(audio_data.shape) == 1:
                num_channels = 1
                num_frames = len(audio_data)
            else:
                num_frames = audio_data.shape[0]
                num_channels = audio_data.shape[1]
            return AudioInfo(sample_rate, num_frames, num_channels)
        except Exception:
            pass
        
        # Fallback avec librosa
        try:
            import librosa
            y, sr = librosa.load(filepath_str, sr=None, mono=False)
            if len(y.shape) == 1:
                num_channels = 1
                num_frames = len(y)
            else:
                num_channels = y.shape[0]
                num_frames = y.shape[1]
            return AudioInfo(sr, num_frames, num_channels)
        except Exception as e:
            raise RuntimeError(f"Impossible de lire les métadonnées audio: {e}")
    
    torchaudio.info = _patched_torchaudio_info

# Patch 6: Forcer le backend soundfile au lieu de torchcodec (non installé)
# torchaudio >= 2.5 utilise torchcodec par défaut qui n'est pas disponible sur ROCm
_original_torchaudio_load = torchaudio.load

def _patched_torchaudio_load(filepath, frame_offset=0, num_frames=-1, *args, **kwargs):
    """
    Patch pour contourner torchaudio 2.9+ qui utilise torchcodec par défaut.
    Supporte frame_offset et num_frames pour le cropping (utilisé par Pyannote).
    """
    import soundfile as sf
    
    filepath_str = str(filepath)
    
    # Essayer soundfile d'abord (rapide, supporte WAV/FLAC/OGG)
    try:
        # Lire avec offset et limite si spécifiés
        if frame_offset > 0 or num_frames > 0:
            stop = frame_offset + num_frames if num_frames > 0 else None
            audio_data, sample_rate = sf.read(
                filepath_str, 
                start=frame_offset, 
                stop=stop,
                dtype='float32'
            )
        else:
            audio_data, sample_rate = sf.read(filepath_str, dtype='float32')
        
        # Convertir en tensor torch au format torchaudio: (channels, samples)
        if len(audio_data.shape) == 1:
            waveform = torch.from_numpy(audio_data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio_data.T)
        return waveform, sample_rate
    except Exception:
        pass
    
    # Fallback sur librosa (supporte MP3 et autres formats via ffmpeg)
    try:
        import librosa
        # librosa.load ne supporte pas offset/duration en frames, on charge tout et on slice
        audio_array, sample_rate = librosa.load(filepath_str, sr=None, mono=False)
        
        if len(audio_array.shape) == 1:
            waveform = torch.from_numpy(audio_array).unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio_array)
        
        # Appliquer offset et num_frames
        if frame_offset > 0 or num_frames > 0:
            if num_frames > 0:
                waveform = waveform[:, frame_offset:frame_offset + num_frames]
            else:
                waveform = waveform[:, frame_offset:]
        
        return waveform.float(), sample_rate
    except Exception:
        pass
    
    # Dernier recours: torchaudio original (peut échouer si torchcodec manque)
    if 'backend' not in kwargs:
        kwargs['backend'] = 'soundfile'
    return _original_torchaudio_load(filepath, frame_offset, num_frames, *args, **kwargs)

torchaudio.load = _patched_torchaudio_load
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import zlib
import re

# Silero VAD pour filtrer les silences (prévention hallucinations)
try:
    from silero_vad import load_silero_vad, get_speech_timestamps
    SILERO_VAD_AVAILABLE = True
except ImportError:
    SILERO_VAD_AVAILABLE = False
    logger = logging.getLogger(__name__)  # Temporary logger before full config

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Filtre pour supprimer les logs des endpoints de polling (/status, /health)
class PollingEndpointFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        # Supprimer les logs uvicorn pour /status et /health
        if '/status' in message or '/health' in message:
            return False
        return True

# Appliquer le filtre au logger uvicorn.access
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.addFilter(PollingEndpointFilter())

# ============================================================================
# Configuration globale
# ============================================================================

UPLOAD_DIR = Path("/tmp/uploads")
WHISPER_MODEL_ID = "openai/whisper-large-v3-turbo"
PYANNOTE_MODEL_ID = "pyannote/speaker-diarization-3.1"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
# Device pour Pyannote: "auto" (GPU avec fallback CPU), "cpu", ou "cuda"
# Utile pour les GPUs AMD récents (RDNA 3.5) où MIOpen peut avoir des problèmes
PYANNOTE_DEVICE = os.environ.get("PYANNOTE_DEVICE", "auto")

# ============================================================================
# Configuration PyAnnote Diarization (optimisation qualité)
# ============================================================================

# Hyperparamètres de clustering (ajustables pour améliorer la séparation des speakers)
PYANNOTE_HYPERPARAMS = {
    # Seuil de clustering: plus bas = speakers plus facilement séparés
    # Valeur par défaut PyAnnote 3.1: ~0.7045
    # Réduire si les speakers sont confondus, augmenter si trop fragmenté
    "clustering_threshold": float(os.environ.get("PYANNOTE_CLUSTERING_THRESHOLD", "0.65")),
    
    # Durée minimale d'un segment de silence pour couper (en secondes)
    # Plus bas = plus sensible aux changements de speaker
    "min_duration_off": float(os.environ.get("PYANNOTE_MIN_DURATION_OFF", "0.3")),
}

# Post-traitement des segments de diarization
DIARIZATION_POST_PROCESSING = {
    # Durée minimale d'un segment (en secondes) - segments plus courts sont ignorés
    "min_segment_duration": 0.3,
    
    # Écart maximal (en secondes) pour fusionner segments consécutifs du même speaker
    "merge_gap_threshold": 0.5,
    
    # Durée minimale entre deux speakers différents pour valider un changement
    "min_speaker_change_duration": 0.2,
}

# Modèles globaux (chargés une seule fois au démarrage)
whisper_model = None
whisper_processor = None
pyannote_pipeline = None
device = None
hf_token_loaded = None
silero_vad_model = None  # Chargé à la demande

# ============================================================================
# Configuration qualité transcription
# ============================================================================

# Patterns d'hallucination connus (sous-titrage TV, YouTube, etc.)
HALLUCINATION_PATTERNS = [
    # Sous-titrage TV/Radio français
    r"sous[- ]?titrag?e?\s*(st['\u2019]?\s*\d*)?",
    r"sous[- ]?titrag?e?\s*soci[ée]t[ée]\s*radio[- ]?canada",
    r"sous[- ]?titr[ée]\s*(par|par\s+la)?",
    
    # Sous-titrage anglais
    r"subtitl(es?|ing|ed)\s*(by)?",
    r"translated?\s*by\s*amara",
    r"amara\.?org",
    r"transcri(bed?|ption)\s*by",
    
    # Phrases YouTube typiques (hallucinations de fin)
    r"n['\u2019]oubliez\s*pas\s*(de\s*)?(vous\s*)?abonn",
    r"merci\s+(d['\u2019]avoir\s+)?regard[ée]",
    r"(thanks?|thank\s*you)\s*(for\s*)?(watching|listening)",
    r"(don['\u2019]t\s*)?forget\s*to\s*(like|subscribe)",
    r"like\s*(and|&)\s*subscribe",
    r"abonnez[- ]?vous",
    r"laissez\s*(un\s*)?commentaire",
    
    # Artefacts génériques
    r"^\s*[.!?,;:\-\u2013\u2014]+\s*$",  # Que de la ponctuation
    r"^\.{3,}$",  # Ellipses multiples
    r"^\s*\.\.\.\s*$",
]

# Seuils de qualité
QUALITY_THRESHOLDS = {
    "compression_ratio_max": 2.8,      # Ratio zlib > 2.8 = trop répétitif
    "repetition_ratio_max": 0.65,      # > 65% de mots identiques = boucle
    "min_segment_chars": 2,            # Segments trop courts = probablement erreur
    "max_word_repeat_count": 4,        # Même mot 4+ fois de suite = loop
    "min_unique_words_ratio": 0.25,    # Au moins 25% de mots uniques
}

# Options de génération Whisper optimisées pour la qualité
WHISPER_GENERATE_KWARGS = {
    "no_repeat_ngram_size": 3,         # Empêche répétition de n-grams
    "num_beams": 5,                    # Beam search pour meilleure qualité
}

# Temperature fallback (commence à 0, monte si échec)
WHISPER_TEMPERATURE_FALLBACK = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# ============================================================================
# État global du job en cours (pour multi-threading et status)
# ============================================================================

@dataclass
class JobStatus:
    """État du job en cours de traitement."""
    busy: bool = False
    job_id: Optional[str] = None
    filename: Optional[str] = None
    stage: str = "idle"  # idle, uploading, transcribing, diarizing, merging, completed, error
    progress: float = 0.0  # 0-100
    started_at: Optional[datetime] = None
    stage_started_at: Optional[datetime] = None
    transcription_segments: int = 0
    diarization_segments: int = 0
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

# Instance globale du status
current_job = JobStatus()
job_lock = threading.Lock()

def update_job_status(
    busy: Optional[bool] = None,
    job_id: Optional[str] = None,
    filename: Optional[str] = None,
    stage: Optional[str] = None,
    progress: Optional[float] = None,
    transcription_segments: Optional[int] = None,
    diarization_segments: Optional[int] = None,
    error_message: Optional[str] = None,
    **details
):
    """Met à jour l'état du job en cours (thread-safe)."""
    global current_job
    with job_lock:
        if busy is not None:
            current_job.busy = busy
            if busy:
                current_job.started_at = datetime.now()
            else:
                current_job.started_at = None
        if job_id is not None:
            current_job.job_id = job_id
        if filename is not None:
            current_job.filename = filename
        if stage is not None:
            current_job.stage = stage
            current_job.stage_started_at = datetime.now()
        if progress is not None:
            current_job.progress = progress
        if transcription_segments is not None:
            current_job.transcription_segments = transcription_segments
        if diarization_segments is not None:
            current_job.diarization_segments = diarization_segments
        if error_message is not None:
            current_job.error_message = error_message
        if details:
            current_job.details.update(details)

def reset_job_status():
    """Réinitialise l'état du job."""
    global current_job
    with job_lock:
        current_job = JobStatus()

def get_job_status_dict() -> dict:
    """Retourne l'état du job sous forme de dict (thread-safe)."""
    with job_lock:
        elapsed = None
        stage_elapsed = None
        if current_job.started_at:
            elapsed = (datetime.now() - current_job.started_at).total_seconds()
        if current_job.stage_started_at:
            stage_elapsed = (datetime.now() - current_job.stage_started_at).total_seconds()
        
        return {
            "busy": current_job.busy,
            "job_id": current_job.job_id,
            "filename": current_job.filename,
            "stage": current_job.stage,
            "progress": current_job.progress,
            "elapsed_seconds": elapsed,
            "stage_elapsed_seconds": stage_elapsed,
            "transcription_segments": current_job.transcription_segments,
            "diarization_segments": current_job.diarization_segments,
            "error_message": current_job.error_message,
            "details": current_job.details.copy()
        }


# ============================================================================
# Modèles Pydantic
# ============================================================================

class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str


class DiarizationSegment(BaseModel):
    start: float
    end: float
    speaker: str


class MergedSegment(BaseModel):
    start: float
    end: float
    speaker: str
    text: str


class ProcessResponse(BaseModel):
    text: str
    segments: List[MergedSegment]
    transcription_segments: List[TranscriptionSegment]
    diarization_segments: List[DiarizationSegment]


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    device: str
    gpu_name: Optional[str]
    memory_total_gb: Optional[float]
    memory_available_gb: Optional[float]
    models_loaded: dict
    busy: bool = False
    current_job: Optional[dict] = None


class StatusResponse(BaseModel):
    busy: bool
    job_id: Optional[str]
    filename: Optional[str]
    stage: str
    progress: float
    elapsed_seconds: Optional[float]
    stage_elapsed_seconds: Optional[float]
    transcription_segments: int
    diarization_segments: int
    error_message: Optional[str]
    details: dict


# ============================================================================
# Fonctions utilitaires
# ============================================================================

def get_device_info() -> dict:
    """Récupère les informations sur le device GPU/CPU."""
    info = {
        "gpu_available": torch.cuda.is_available(),
        "device": "cpu",
        "gpu_name": None,
        "memory_total_gb": None,
        "memory_available_gb": None
    }
    
    if torch.cuda.is_available():
        info["device"] = "cuda"
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            # Mémoire en GB
            props = torch.cuda.get_device_properties(0)
            info["memory_total_gb"] = round(props.total_memory / (1024**3), 2)
            
            # Mémoire disponible
            free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            info["memory_available_gb"] = round(free_memory / (1024**3), 2)
        except Exception as e:
            logger.warning(f"Impossible de récupérer les infos GPU: {e}")
    
    return info


def detect_rocm() -> bool:
    """Détecte si on utilise ROCm au lieu de CUDA natif."""
    if torch.cuda.is_available():
        # ROCm expose torch.version.hip
        return hasattr(torch.version, 'hip') and torch.version.hip is not None
    return False


# ============================================================================
# Fonctions de filtrage qualité
# ============================================================================

def calculate_compression_ratio(text: str) -> float:
    """
    Calcule le ratio de compression zlib.
    Un texte très répétitif aura un ratio élevé car il se compresse bien.
    """
    if not text or len(text) < 10:
        return 0.0
    text_bytes = text.encode('utf-8')
    compressed = zlib.compress(text_bytes, level=9)
    return len(text_bytes) / len(compressed)


def detect_word_loops(text: str) -> tuple[bool, Optional[str]]:
    """
    Détecte les boucles de mots répétés.
    Ex: "d'un d'un d'un d'un" ou "3, 2, 3, 2, 3, 2"
    
    Returns:
        (is_loop, detected_pattern)
    """
    if not text:
        return False, None
    
    # Normaliser le texte
    text_lower = text.lower().strip()
    
    # Pattern 1: Mot simple répété 4+ fois consécutives
    # Matches: "oui oui oui oui", "non non non non"
    pattern1 = r'\b([\w\'\u2019àâäéèêëïîôùûüç-]+)\s+(?:\1[\s,]*){3,}'
    match1 = re.search(pattern1, text_lower, re.IGNORECASE | re.UNICODE)
    if match1:
        return True, f"mot répété: '{match1.group(1)}'"
    
    # Pattern 2: Séquence courte répétée (2-4 tokens)
    # Matches: "c'est parti, c'est parti, c'est parti"
    words = re.findall(r'[\w\'\u2019àâäéèêëïîôùûüç-]+', text_lower)
    if len(words) >= 8:
        # Chercher des patterns de 2-4 mots répétés
        for pattern_len in range(2, 5):
            for start in range(len(words) - pattern_len * 3):
                pattern_words = tuple(words[start:start + pattern_len])
                repeat_count = 1
                pos = start + pattern_len
                while pos + pattern_len <= len(words):
                    if tuple(words[pos:pos + pattern_len]) == pattern_words:
                        repeat_count += 1
                        pos += pattern_len
                    else:
                        break
                if repeat_count >= 3:
                    return True, f"phrase répétée {repeat_count}x: '{' '.join(pattern_words)}'"
    
    # Pattern 3: Séquence de chiffres/nombres alternés
    # Matches: "3, 2, 3, 2, 3, 2" ou "1 2 1 2 1 2"
    numbers = re.findall(r'\d+', text_lower)
    if len(numbers) >= 6:
        for pattern_len in range(2, 4):
            pattern = tuple(numbers[:pattern_len])
            matches = 0
            for i in range(0, len(numbers) - pattern_len + 1, pattern_len):
                if tuple(numbers[i:i + pattern_len]) == pattern:
                    matches += 1
            if matches >= 3:
                return True, f"nombres répétés: {pattern}"
    
    return False, None


def calculate_repetition_ratio(text: str) -> float:
    """
    Calcule le ratio de répétition dans un texte.
    Retourne la proportion de mots non-uniques.
    """
    words = re.findall(r'[\w\'\u2019àâäéèêëïîôùûüç-]+', text.lower())
    if len(words) < 3:
        return 0.0
    unique = len(set(words))
    return 1.0 - (unique / len(words))


def is_hallucination(text: str) -> bool:
    """Vérifie si le texte correspond à un pattern d'hallucination connu."""
    if not text:
        return False
    text_lower = text.lower().strip()
    
    for pattern in HALLUCINATION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE | re.UNICODE):
            return True
    return False


def filter_hallucinations(text: str) -> str:
    """Supprime les patterns d'hallucination du texte."""
    if not text:
        return text
    
    result = text
    for pattern in HALLUCINATION_PATTERNS:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE | re.UNICODE)
    
    # Nettoyer les espaces multiples
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def validate_segment(segment: dict) -> tuple[bool, Optional[str]]:
    """
    Valide un segment de transcription.
    
    Returns:
        (is_valid, reason_if_invalid)
    """
    text = segment.get("text", "").strip()
    
    # Segment vide ou trop court
    if len(text) < QUALITY_THRESHOLDS["min_segment_chars"]:
        return False, "segment trop court"
    
    # Vérifier si c'est une hallucination connue
    if is_hallucination(text):
        return False, "hallucination détectée"
    
    # Vérifier le ratio de compression (texte répétitif)
    compression = calculate_compression_ratio(text)
    if compression > QUALITY_THRESHOLDS["compression_ratio_max"]:
        return False, f"compression ratio trop élevé ({compression:.2f})"
    
    # Vérifier les boucles de mots
    is_loop, loop_pattern = detect_word_loops(text)
    if is_loop:
        return False, f"boucle détectée: {loop_pattern}"
    
    # Vérifier le ratio de répétition
    rep_ratio = calculate_repetition_ratio(text)
    if rep_ratio > QUALITY_THRESHOLDS["repetition_ratio_max"]:
        return False, f"ratio répétition trop élevé ({rep_ratio:.2%})"
    
    return True, None


def clean_segment_text(text: str) -> str:
    """Nettoie le texte d'un segment (supprime artefacts, loops partiels)."""
    if not text:
        return text
    
    # Supprimer les hallucinations
    cleaned = filter_hallucinations(text)
    
    # Supprimer les répétitions de mots simples (garde 1 occurence)
    # Ex: "oui oui oui" -> "oui"
    cleaned = re.sub(
        r'\b([\w\'\u2019àâäéèêëïîôùûüç-]+)(?:\s+\1){2,}\b',
        r'\1',
        cleaned,
        flags=re.IGNORECASE | re.UNICODE
    )
    
    # Supprimer les répétitions avec virgules
    # Ex: "oui, oui, oui, oui" -> "oui"
    cleaned = re.sub(
        r'\b([\w\'\u2019àâäéèêëïîôùûüç-]+)(?:\s*,\s*\1){2,}\b',
        r'\1',
        cleaned,
        flags=re.IGNORECASE | re.UNICODE
    )
    
    # Nettoyer les espaces et ponctuation multiples
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'([,.])\1+', r'\1', cleaned)
    
    return cleaned.strip()


def load_silero_vad_model():
    """Charge le modèle Silero VAD si disponible."""
    global silero_vad_model
    
    if not SILERO_VAD_AVAILABLE:
        logger.warning("Silero VAD non disponible - pip install silero-vad")
        return None
    
    if silero_vad_model is None:
        logger.info("Chargement du modèle Silero VAD...")
        silero_vad_model = load_silero_vad()
        logger.info("Silero VAD chargé avec succès")
    
    return silero_vad_model


def apply_vad_to_audio(audio_array, sample_rate: int = 16000) -> tuple:
    """
    Applique le VAD pour extraire uniquement les segments de parole.
    Retourne l'audio filtré et les timestamps des segments.
    """
    vad_model = load_silero_vad_model()
    if vad_model is None:
        return audio_array, []  # Retourner l'audio original si pas de VAD
    
    # Convertir en tensor si nécessaire
    if not isinstance(audio_array, torch.Tensor):
        audio_tensor = torch.from_numpy(audio_array).float()
    else:
        audio_tensor = audio_array.float()
    
    # Obtenir les timestamps de parole
    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        vad_model,
        sampling_rate=sample_rate,
        threshold=0.5,  # Seuil de détection
        min_speech_duration_ms=250,  # Minimum 250ms de parole
        min_silence_duration_ms=100,  # Minimum 100ms de silence pour couper
        speech_pad_ms=30,  # Padding autour de la parole
    )
    
    if not speech_timestamps:
        logger.warning("VAD: Aucun segment de parole détecté!")
        return audio_array, []
    
    # Assembler les segments de parole
    speech_segments = []
    for ts in speech_timestamps:
        speech_segments.append(audio_tensor[ts['start']:ts['end']])
    
    # Concaténer tous les segments
    if speech_segments:
        filtered_audio = torch.cat(speech_segments).numpy()
        logger.info(f"VAD: {len(speech_timestamps)} segments de parole, "
                   f"{len(filtered_audio)/sample_rate:.1f}s gardé sur {len(audio_array)/sample_rate:.1f}s original")
        return filtered_audio, speech_timestamps
    
    return audio_array, []


def load_whisper_model():
    """Charge le modèle Whisper une seule fois."""
    global whisper_model, whisper_processor, device
    
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    
    logger.info(f"Chargement de Whisper: {WHISPER_MODEL_ID}")
    
    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL_ID,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    whisper_model.to(device)
    
    whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
    
    logger.info("Whisper chargé avec succès")


def load_pyannote_model(hf_token: str):
    """Charge le pipeline Pyannote avec hyperparamètres optimisés."""
    global pyannote_pipeline, hf_token_loaded, device
    
    from pyannote.audio import Pipeline
    
    # Si déjà chargé avec le même token, on skip
    if pyannote_pipeline is not None and hf_token_loaded == hf_token:
        return
    
    logger.info(f"Chargement de Pyannote: {PYANNOTE_MODEL_ID}")
    
    pyannote_pipeline = Pipeline.from_pretrained(
        PYANNOTE_MODEL_ID,
        use_auth_token=hf_token
    )
    
    # Appliquer les hyperparamètres personnalisés pour améliorer la qualité
    try:
        # Ajuster le seuil de clustering (plus bas = meilleure séparation des speakers)
        if hasattr(pyannote_pipeline, 'parameters') and callable(pyannote_pipeline.parameters):
            params = pyannote_pipeline.parameters(instantiated=True)
            logger.info(f"Paramètres Pyannote avant ajustement: {params}")
        
        # Modifier le threshold de clustering si disponible
        clustering_threshold = PYANNOTE_HYPERPARAMS.get("clustering_threshold", 0.65)
        if hasattr(pyannote_pipeline, '_segmentation'):
            # La segmentation a ses propres paramètres
            pass
        
        # Certaines versions permettent d'ajuster via pyannote_pipeline.clustering
        if hasattr(pyannote_pipeline, 'clustering') and hasattr(pyannote_pipeline.clustering, 'threshold'):
            old_threshold = pyannote_pipeline.clustering.threshold
            pyannote_pipeline.clustering.threshold = clustering_threshold
            logger.info(f"Clustering threshold ajusté: {old_threshold} -> {clustering_threshold}")
        
        # Ajuster min_duration_off pour la segmentation
        min_duration_off = PYANNOTE_HYPERPARAMS.get("min_duration_off", 0.3)
        if hasattr(pyannote_pipeline, 'segmentation') and hasattr(pyannote_pipeline.segmentation, 'min_duration_off'):
            pyannote_pipeline.segmentation.min_duration_off = min_duration_off
            logger.info(f"min_duration_off ajusté: {min_duration_off}")
            
    except Exception as e:
        logger.warning(f"Impossible d'ajuster les hyperparamètres Pyannote: {e}")
        logger.warning("Utilisation des paramètres par défaut")
    
    # Déterminer le device pour Pyannote
    pyannote_device = _get_pyannote_device()
    if pyannote_device == "cuda":
        pyannote_pipeline.to(torch.device("cuda"))
        logger.info("Pyannote chargé sur GPU (CUDA)")
    else:
        pyannote_pipeline.to(torch.device("cpu"))
        logger.info("Pyannote chargé sur CPU")
    
    logger.info(f"Pyannote configuré - clustering_threshold: {PYANNOTE_HYPERPARAMS.get('clustering_threshold')}")
    hf_token_loaded = hf_token


def _get_pyannote_device() -> str:
    """
    Détermine le device à utiliser pour Pyannote.
    
    Returns:
        "cuda" ou "cpu"
    """
    if PYANNOTE_DEVICE.lower() == "cpu":
        return "cpu"
    
    if PYANNOTE_DEVICE.lower() == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        else:
            logger.warning("PYANNOTE_DEVICE=cuda mais pas de GPU disponible, utilisation du CPU")
            return "cpu"
    
    # Mode "auto": essayer GPU, fallback sur CPU si erreur connue
    if torch.cuda.is_available():
        # Sur les GPUs AMD récents (RDNA 3.5), MIOpen peut avoir des problèmes
        # avec certaines opérations (InstanceNorm dans SincNet)
        # On tente quand même le GPU, l'erreur sera gérée à l'exécution
        if detect_rocm():
            # Pour ROCm, vérifier si c'est un GPU potentiellement problématique
            try:
                gpu_name = torch.cuda.get_device_name(0).lower()
                # GPUs RDNA 3.5 (Strix Point) connus pour avoir des problèmes
                problematic_gpus = ["8060", "8050", "8040", "strix"]
                if any(prob in gpu_name for prob in problematic_gpus):
                    logger.warning(f"GPU {torch.cuda.get_device_name(0)} détecté - "
                                   "MIOpen peut avoir des problèmes avec Pyannote")
                    logger.warning("Utilisation du CPU pour Pyannote (définir PYANNOTE_DEVICE=cuda pour forcer le GPU)")
                    return "cpu"
            except Exception:
                pass
        return "cuda"
    
    return "cpu"


def transcribe_audio(audio_path: str, language: Optional[str] = None) -> dict:
    """
    Transcrit l'audio avec Whisper.
    
    Améliorations qualité:
    - VAD (Voice Activity Detection) pour filtrer les silences
    - Paramètres de génération optimisés anti-répétition
    - Filtrage des hallucinations et boucles post-transcription
    """
    global whisper_model, whisper_processor, device
    
    from transformers import pipeline
    import librosa
    
    update_job_status(stage="transcribing", progress=5)
    
    # Charger l'audio avec librosa (évite torchcodec)
    logger.info("Chargement de l'audio...")
    update_job_status(progress=8, audio_loading=True)
    audio_array, sample_rate = librosa.load(audio_path, sr=16000)
    
    # Calculer la durée originale
    original_duration = len(audio_array) / sample_rate
    logger.info(f"Audio chargé: {original_duration:.1f} secondes")
    
    # Appliquer VAD pour filtrer les silences (prévention hallucinations)
    update_job_status(progress=10, stage_detail="VAD filtering")
    if SILERO_VAD_AVAILABLE:
        logger.info("Application du VAD pour filtrer les silences...")
        audio_array, speech_timestamps = apply_vad_to_audio(audio_array, sample_rate)
        vad_duration = len(audio_array) / sample_rate
        update_job_status(
            progress=15, 
            vad_applied=True,
            original_duration_s=round(original_duration, 1),
            vad_duration_s=round(vad_duration, 1),
            speech_segments=len(speech_timestamps)
        )
    else:
        logger.warning("VAD non disponible, traitement de l'audio complet")
        update_job_status(progress=15, vad_applied=False)
    
    duration_seconds = len(audio_array) / sample_rate
    update_job_status(audio_duration_seconds=round(duration_seconds, 1))
    
    # Créer le pipeline de transcription
    pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=whisper_processor.tokenizer,
        feature_extractor=whisper_processor.feature_extractor,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device=device
    )
    
    # Options de génération optimisées pour la qualité
    generate_kwargs = {
        **WHISPER_GENERATE_KWARGS,  # no_repeat_ngram_size, num_beams, patience
    }
    if language:
        generate_kwargs["language"] = language
    
    # Passer l'audio comme dict
    audio_input = {"array": audio_array, "sampling_rate": sample_rate}
    
    update_job_status(progress=20)
    logger.info(f"Transcription en cours (generate_kwargs: {generate_kwargs})...")
    
    # Transcription avec timestamps
    result = pipe(
        audio_input,
        return_timestamps=True,
        generate_kwargs=generate_kwargs,
        chunk_length_s=30,
        batch_size=8
    )
    
    update_job_status(progress=40)
    
    # Debug: afficher la structure du résultat
    logger.info(f"Whisper result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
    
    # Formater et filtrer les segments
    raw_segments = []
    if "chunks" in result:
        logger.info(f"Found {len(result['chunks'])} chunks bruts dans le résultat")
        for chunk in result["chunks"]:
            if chunk.get("timestamp"):
                start, end = chunk["timestamp"]
                if start is not None and end is not None:
                    raw_segments.append({
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "text": chunk["text"].strip()
                    })
    else:
        logger.warning("No 'chunks' found in Whisper result - segments will be empty")
    
    update_job_status(progress=45, raw_segments=len(raw_segments))
    
    # === FILTRAGE QUALITÉ ===
    logger.info("Application du filtrage qualité des segments...")
    
    filtered_segments = []
    rejected_count = 0
    cleaned_count = 0
    
    for seg in raw_segments:
        original_text = seg["text"]
        
        # 1. Valider le segment
        is_valid, rejection_reason = validate_segment(seg)
        if not is_valid:
            logger.debug(f"Segment rejeté ({rejection_reason}): '{original_text[:50]}...'")
            rejected_count += 1
            continue
        
        # 2. Nettoyer le texte (supprime loops partiels, hallucinations)
        cleaned_text = clean_segment_text(original_text)
        
        # Si le nettoyage a significativement modifié le texte, log it
        if cleaned_text != original_text:
            logger.debug(f"Segment nettoyé: '{original_text[:40]}' -> '{cleaned_text[:40]}'")
            cleaned_count += 1
        
        # Rejeter si le nettoyage a vidé le segment
        if len(cleaned_text.strip()) < QUALITY_THRESHOLDS["min_segment_chars"]:
            rejected_count += 1
            continue
        
        filtered_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": cleaned_text
        })
    
    logger.info(f"Filtrage qualité: {len(raw_segments)} bruts -> {len(filtered_segments)} gardés "
               f"({rejected_count} rejetés, {cleaned_count} nettoyés)")
    
    # Reconstruire le texte complet à partir des segments filtrés
    full_text = " ".join(seg["text"] for seg in filtered_segments)
    
    # Appliquer un nettoyage final au texte complet
    full_text = clean_segment_text(full_text)
    
    update_job_status(
        progress=50, 
        transcription_segments=len(filtered_segments),
        rejected_segments=rejected_count,
        cleaned_segments=cleaned_count
    )
    logger.info(f"Transcription terminée: {len(full_text)} caractères, {len(filtered_segments)} segments")
    
    return {
        "text": full_text,
        "segments": filtered_segments
    }


def diarize_audio(audio_path: str, min_speakers: Optional[int] = None, 
                  max_speakers: Optional[int] = None) -> List[dict]:
    """
    Effectue la diarization avec Pyannote.
    
    Optimisations incluses:
    - Filtrage des segments trop courts (bruit)
    - Fusion des segments consécutifs du même speaker
    - Validation des changements de speaker
    """
    global pyannote_pipeline
    
    update_job_status(stage="diarizing", progress=55)
    logger.info("Diarization en cours (peut prendre plusieurs minutes)...")
    
    params = {}
    if min_speakers:
        params["min_speakers"] = min_speakers
    if max_speakers:
        params["max_speakers"] = max_speakers
    
    try:
        diarization = pyannote_pipeline(audio_path, **params)
    except RuntimeError as e:
        # Fallback automatique sur CPU si erreur MIOpen (GPU AMD incompatible)
        if "miopenStatus" in str(e) or "MIOpen" in str(e):
            logger.warning(f"Erreur MIOpen détectée: {e}")
            logger.warning("Basculement automatique de Pyannote sur CPU...")
            update_job_status(pyannote_fallback_cpu=True)
            _fallback_pyannote_to_cpu()
            # Réessayer sur CPU
            diarization = pyannote_pipeline(audio_path, **params)
        else:
            raise
    
    update_job_status(progress=80)
    
    # Extraire les segments bruts
    raw_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        raw_segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker
        })
    
    logger.info(f"Diarization brute: {len(raw_segments)} segments")
    
    # === POST-TRAITEMENT DES SEGMENTS ===
    segments = post_process_diarization_segments(raw_segments)
    
    update_job_status(progress=85)
    
    # Compter les speakers uniques
    unique_speakers = len(set(seg["speaker"] for seg in segments))
    update_job_status(progress=90, diarization_segments=len(segments), speakers_detected=unique_speakers)
    
    logger.info(f"Diarization terminée: {len(raw_segments)} bruts -> {len(segments)} après post-traitement, "
               f"{unique_speakers} locuteurs détectés")
    
    if not segments:
        logger.warning("Aucun locuteur détecté - le fichier audio est peut-être trop court ou silencieux")
    
    return segments


def post_process_diarization_segments(segments: List[dict]) -> List[dict]:
    """
    Post-traitement des segments de diarization pour améliorer la qualité.
    
    1. Filtre les segments trop courts (< min_segment_duration)
    2. Fusionne les segments consécutifs du même speaker (si gap < merge_gap_threshold)
    3. Valide les changements de speaker (ignore les micro-changements)
    """
    if not segments:
        return segments
    
    config = DIARIZATION_POST_PROCESSING
    min_duration = config.get("min_segment_duration", 0.3)
    merge_gap = config.get("merge_gap_threshold", 0.5)
    min_change_duration = config.get("min_speaker_change_duration", 0.2)
    
    # Étape 1: Filtrer les segments trop courts
    filtered = []
    for seg in segments:
        duration = seg["end"] - seg["start"]
        if duration >= min_duration:
            filtered.append(seg.copy())
        else:
            logger.debug(f"Segment ignoré (trop court: {duration:.2f}s): {seg['speaker']}")
    
    if not filtered:
        return segments  # Retourner les originaux si tout est filtré
    
    logger.debug(f"Après filtrage durée min: {len(segments)} -> {len(filtered)} segments")
    
    # Étape 2: Fusionner les segments consécutifs du même speaker
    merged = [filtered[0]]
    for seg in filtered[1:]:
        last = merged[-1]
        gap = seg["start"] - last["end"]
        
        # Même speaker et gap acceptable -> fusionner
        if seg["speaker"] == last["speaker"] and gap <= merge_gap:
            last["end"] = seg["end"]  # Étendre le segment précédent
            logger.debug(f"Fusion: {last['speaker']} [{last['start']:.1f} - {seg['end']:.1f}]")
        else:
            merged.append(seg)
    
    logger.debug(f"Après fusion: {len(filtered)} -> {len(merged)} segments")
    
    # Étape 3: Valider les changements de speaker (éviter les micro-changements)
    validated = [merged[0]]
    for seg in merged[1:]:
        last = validated[-1]
        
        # Si le segment est trop court pour un changement de speaker valide
        seg_duration = seg["end"] - seg["start"]
        if seg_duration < min_change_duration and seg["speaker"] != last["speaker"]:
            # Vérifier le segment suivant (s'il existe) pour confirmer le changement
            # Pour l'instant, on garde le segment mais on log
            logger.debug(f"Changement de speaker court ({seg_duration:.2f}s): {last['speaker']} -> {seg['speaker']}")
        
        validated.append(seg)
    
    logger.info(f"Post-traitement diarization: {len(segments)} bruts -> {len(validated)} optimisés")
    return validated


def _fallback_pyannote_to_cpu():
    """Bascule le pipeline Pyannote sur CPU (fallback après erreur GPU)."""
    global pyannote_pipeline
    
    if pyannote_pipeline is not None:
        pyannote_pipeline.to(torch.device("cpu"))
        logger.info("Pyannote basculé sur CPU avec succès")


def merge_transcription_diarization(
    transcription_segments: List[dict],
    diarization_segments: List[dict]
) -> List[dict]:
    """
    Fusionne les segments de transcription avec la diarization.
    Assigne chaque segment de transcription au speaker qui parle le plus pendant ce segment.
    """
    merged = []
    
    for trans_seg in transcription_segments:
        t_start = trans_seg["start"]
        t_end = trans_seg["end"]
        
        # Trouver le speaker dominant pour ce segment
        speaker_times = {}
        
        for diar_seg in diarization_segments:
            d_start = diar_seg["start"]
            d_end = diar_seg["end"]
            speaker = diar_seg["speaker"]
            
            # Calculer l'overlap
            overlap_start = max(t_start, d_start)
            overlap_end = min(t_end, d_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > 0:
                speaker_times[speaker] = speaker_times.get(speaker, 0) + overlap
        
        # Assigner au speaker avec le plus d'overlap
        if speaker_times:
            dominant_speaker = max(speaker_times, key=speaker_times.get)
        else:
            dominant_speaker = "SPEAKER_UNKNOWN"
        
        merged.append({
            "start": t_start,
            "end": t_end,
            "speaker": dominant_speaker,
            "text": trans_seg["text"]
        })
    
    return merged


def save_upload_file(upload_file: UploadFile) -> str:
    """Sauvegarde le fichier uploadé et retourne le chemin."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Utiliser un nom unique
    suffix = Path(upload_file.filename).suffix if upload_file.filename else ".wav"
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        dir=UPLOAD_DIR
    )
    
    try:
        shutil.copyfileobj(upload_file.file, temp_file)
        temp_file.close()
        return temp_file.name
    except Exception:
        temp_file.close()
        os.unlink(temp_file.name)
        raise


# ============================================================================
# Lifespan - Chargement des modèles au démarrage
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère le cycle de vie de l'application."""
    global device
    
    # Startup
    logger.info("=" * 60)
    logger.info("Démarrage du service Remote Processing")
    logger.info("=" * 60)
    
    # Détecter le device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if detect_rocm():
            logger.info("🔥 ROCm GPU détecté")
        else:
            logger.info("🔥 CUDA GPU détecté")
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("⚠️  Pas de GPU disponible, utilisation du CPU")
    
    # Créer le dossier uploads
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Dossier uploads: {UPLOAD_DIR}")
    
    # Charger Whisper (ne nécessite pas de token HF)
    try:
        load_whisper_model()
    except Exception as e:
        logger.error(f"Erreur au chargement de Whisper: {e}")
        raise
    
    # Afficher la configuration Pyannote
    logger.info(f"🎯 PYANNOTE_DEVICE={PYANNOTE_DEVICE} (device cible pour Pyannote)")
    
    # Charger Pyannote au démarrage si HF_TOKEN est défini
    if HF_TOKEN:
        logger.info("🔑 HF_TOKEN trouvé, chargement de Pyannote au démarrage...")
        try:
            load_pyannote_model(HF_TOKEN)
        except Exception as e:
            logger.error(f"Erreur au chargement de Pyannote: {e}", exc_info=True)
            raise
    else:
        logger.warning("⚠️  HF_TOKEN non défini, Pyannote sera chargé à la première requête")
    
    logger.info("✅ Service prêt")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("Arrêt du service...")


# ============================================================================
# Application FastAPI
# ============================================================================

app = FastAPI(
    title="Remote Processing Service",
    description="Transcription + Diarization avec Whisper et Pyannote",
    version="2.0.0",
    lifespan=lifespan
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Retourne l'état détaillé du service.
    Répond toujours, même pendant un traitement en cours.
    """
    device_info = get_device_info()
    job_status = get_job_status_dict()
    
    return HealthResponse(
        status="ok",
        gpu_available=device_info["gpu_available"],
        device=device_info["device"],
        gpu_name=device_info["gpu_name"],
        memory_total_gb=device_info["memory_total_gb"],
        memory_available_gb=device_info["memory_available_gb"],
        models_loaded={
            "whisper": whisper_model is not None,
            "pyannote": pyannote_pipeline is not None
        },
        busy=job_status["busy"],
        current_job=job_status if job_status["busy"] else None
    )


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Retourne le status détaillé du job en cours.
    Répond toujours instantanément.
    """
    status = get_job_status_dict()
    return StatusResponse(**status)


def _do_process_sync(
    temp_file: str,
    hf_token: str,
    language: Optional[str],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    skip_diarization: bool
) -> dict:
    """
    Traitement synchrone (exécuté dans un thread séparé).
    Permet au health check de répondre pendant le traitement.
    """
    # Charger Pyannote si nécessaire (et si diarization demandée)
    if not skip_diarization:
        load_pyannote_model(hf_token)
    
    # 1. Transcription
    logger.info("Début transcription...")
    transcription_result = transcribe_audio(temp_file, language)
    logger.info(f"Transcription terminée: {len(transcription_result['segments'])} segments")
    
    # 2. Diarization (optionnelle)
    diarization_segments = []
    if not skip_diarization:
        logger.info("Début diarization...")
        diarization_segments = diarize_audio(temp_file, min_speakers, max_speakers)
        logger.info(f"Diarization terminée: {len(diarization_segments)} segments")
    
    # 3. Fusion
    update_job_status(stage="merging", progress=92)
    if diarization_segments:
        merged_segments = merge_transcription_diarization(
            transcription_result["segments"],
            diarization_segments
        )
    else:
        # Sans diarization, on met un speaker par défaut
        merged_segments = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "speaker": "SPEAKER_00",
                "text": seg["text"]
            }
            for seg in transcription_result["segments"]
        ]
    
    update_job_status(stage="completed", progress=100)
    logger.info("Traitement terminé avec succès")
    
    return {
        "text": transcription_result["text"],
        "segments": merged_segments,
        "transcription_segments": transcription_result["segments"],
        "diarization_segments": diarization_segments
    }


@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    hf_token: str = Form(...),
    language: Optional[str] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    skip_diarization: bool = Form(False)
):
    """
    Endpoint principal: Transcription + Diarization combinées.
    
    Le traitement s'exécute dans un thread séparé pour permettre
    au endpoint /health et /status de répondre pendant le traitement.
    
    Args:
        file: Fichier audio (WAV, MP3, OGG, etc.)
        hf_token: Token Hugging Face (requis pour Pyannote)
        language: Code langue optionnel (ex: "fr", "en")
        min_speakers: Nombre minimum de speakers
        max_speakers: Nombre maximum de speakers
        skip_diarization: Si True, fait seulement la transcription
    
    Returns:
        JSON avec text, segments fusionnés, et segments bruts
    """
    # Vérifier si déjà occupé
    if current_job.busy:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service occupé",
                "message": f"Un traitement est déjà en cours: {current_job.filename}",
                "current_job": get_job_status_dict()
            }
        )
    
    temp_file = None
    job_id = str(uuid.uuid4())[:8]
    
    try:
        # Marquer comme occupé
        update_job_status(
            busy=True,
            job_id=job_id,
            filename=file.filename,
            stage="uploading",
            progress=0,
            transcription_segments=0,
            diarization_segments=0,
            error_message=None
        )
        
        # Sauvegarder le fichier
        logger.info(f"[{job_id}] Réception fichier: {file.filename}")
        temp_file = save_upload_file(file)
        logger.info(f"[{job_id}] Fichier sauvegardé: {temp_file}")
        
        update_job_status(stage="loading", progress=2)
        
        # Exécuter le traitement dans un thread séparé
        # Cela permet au health check de répondre pendant le traitement
        result = await asyncio.to_thread(
            _do_process_sync,
            temp_file,
            hf_token,
            language,
            min_speakers,
            max_speakers,
            skip_diarization
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{job_id}] Erreur traitement: {e}", exc_info=True)
        update_job_status(stage="error", error_message=error_msg)
        return JSONResponse(
            status_code=500,
            content={"error": f"Erreur traitement: {error_msg}"}
        )
    
    finally:
        # Nettoyage du fichier temporaire
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
                logger.debug(f"[{job_id}] Fichier temporaire supprimé: {temp_file}")
            except Exception as e:
                logger.warning(f"[{job_id}] Impossible de supprimer {temp_file}: {e}")
        
        # Réinitialiser le status après un court délai (pour que le frontend puisse voir "completed")
        await asyncio.sleep(2)
        reset_job_status()


def _do_diarize_sync(temp_file: str, hf_token: str, min_speakers: Optional[int], max_speakers: Optional[int]) -> List[dict]:
    """Diarization synchrone dans un thread séparé."""
    load_pyannote_model(hf_token)
    segments = diarize_audio(temp_file, min_speakers, max_speakers)
    update_job_status(stage="completed", progress=100)
    return segments


@app.post("/diarize")
async def diarize_only(
    file: UploadFile = File(...),
    hf_token: str = Form(...),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None)
):
    """
    Endpoint de rétrocompatibilité: Diarization seule.
    Exécuté dans un thread séparé pour ne pas bloquer le health check.
    """
    if current_job.busy:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service occupé",
                "message": f"Un traitement est déjà en cours: {current_job.filename}",
                "current_job": get_job_status_dict()
            }
        )
    
    temp_file = None
    job_id = str(uuid.uuid4())[:8]
    
    try:
        update_job_status(
            busy=True,
            job_id=job_id,
            filename=file.filename,
            stage="uploading",
            progress=0
        )
        
        temp_file = save_upload_file(file)
        update_job_status(stage="loading", progress=5)
        
        segments = await asyncio.to_thread(
            _do_diarize_sync,
            temp_file,
            hf_token,
            min_speakers,
            max_speakers
        )
        
        return JSONResponse(content={"segments": segments})
        
    except Exception as e:
        logger.error(f"[{job_id}] Erreur diarization: {e}", exc_info=True)
        update_job_status(stage="error", error_message=str(e))
        return JSONResponse(
            status_code=500,
            content={"error": f"Diarization échouée: {str(e)}"}
        )
    
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass
        await asyncio.sleep(2)
        reset_job_status()


def _do_transcribe_sync(temp_file: str, language: Optional[str]) -> dict:
    """Transcription synchrone dans un thread séparé."""
    result = transcribe_audio(temp_file, language)
    update_job_status(stage="completed", progress=100)
    return result


@app.post("/transcribe")
async def transcribe_only(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None)
):
    """
    Endpoint: Transcription seule (sans diarization).
    Ne nécessite pas de token HF.
    Exécuté dans un thread séparé pour ne pas bloquer le health check.
    """
    if current_job.busy:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service occupé",
                "message": f"Un traitement est déjà en cours: {current_job.filename}",
                "current_job": get_job_status_dict()
            }
        )
    
    temp_file = None
    job_id = str(uuid.uuid4())[:8]
    
    try:
        update_job_status(
            busy=True,
            job_id=job_id,
            filename=file.filename,
            stage="uploading",
            progress=0
        )
        
        temp_file = save_upload_file(file)
        update_job_status(stage="loading", progress=2)
        
        result = await asyncio.to_thread(
            _do_transcribe_sync,
            temp_file,
            language
        )
        
        return JSONResponse(content={
            "text": result["text"],
            "segments": result["segments"]
        })
        
    except Exception as e:
        logger.error(f"[{job_id}] Erreur transcription: {e}", exc_info=True)
        update_job_status(stage="error", error_message=str(e))
        return JSONResponse(
            status_code=500,
            content={"error": f"Transcription échouée: {str(e)}"}
        )
    
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass
        await asyncio.sleep(2)
        reset_job_status()
