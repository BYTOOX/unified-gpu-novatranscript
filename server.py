"""
Remote Processing Service - Transcription + Diarization
Optimisé pour AMD ROCm avec mémoire unifiée

Endpoints:
    POST /process - Transcription + Diarization combinées
    POST /diarize - Diarization seule (rétrocompatibilité)
    GET /health   - Status détaillé du service
"""

import os
import shutil
import logging
import tempfile
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List

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
if not hasattr(torchaudio, 'info'):
    def _patched_torchaudio_info(filepath, backend=None):
        """
        Remplacement de torchaudio.info pour torchaudio 2.0+
        Retourne les métadonnées audio en utilisant soundfile ou librosa.
        """
        import soundfile as sf
        
        filepath_str = str(filepath)
        
        try:
            # Utiliser soundfile pour obtenir les infos
            info = sf.info(filepath_str)
            
            # Créer un objet compatible avec ce que Pyannote attend
            class AudioInfo:
                def __init__(self, sample_rate, num_frames, num_channels):
                    self.sample_rate = sample_rate
                    self.num_frames = num_frames
                    self.num_channels = num_channels
            
            return AudioInfo(
                sample_rate=info.samplerate,
                num_frames=info.frames,
                num_channels=info.channels
            )
        except Exception:
            # Fallback avec librosa
            import librosa
            y, sr = librosa.load(filepath_str, sr=None, mono=False)
            
            class AudioInfo:
                def __init__(self, sample_rate, num_frames, num_channels):
                    self.sample_rate = sample_rate
                    self.num_frames = num_frames
                    self.num_channels = num_channels
            
            if len(y.shape) == 1:
                num_channels = 1
                num_frames = len(y)
            else:
                num_channels = y.shape[0]
                num_frames = y.shape[1]
            
            return AudioInfo(
                sample_rate=sr,
                num_frames=num_frames,
                num_channels=num_channels
            )
    
    torchaudio.info = _patched_torchaudio_info

# Patch 6: Forcer le backend soundfile au lieu de torchcodec (non installé)
# torchaudio >= 2.5 utilise torchcodec par défaut qui n'est pas disponible sur ROCm
_original_torchaudio_load = torchaudio.load

def _patched_torchaudio_load(filepath, *args, **kwargs):
    """
    Patch pour contourner torchaudio 2.9+ qui utilise torchcodec par défaut.
    torchcodec n'est pas disponible sur ROCm, donc on utilise soundfile/librosa directement.
    """
    # Essayer soundfile d'abord (rapide, supporte WAV/FLAC/OGG)
    try:
        import soundfile as sf
        audio_data, sample_rate = sf.read(str(filepath), dtype='float32')
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
        audio_array, sample_rate = librosa.load(str(filepath), sr=None, mono=False)
        if len(audio_array.shape) == 1:
            waveform = torch.from_numpy(audio_array).unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio_array)
        return waveform.float(), sample_rate
    except Exception:
        pass
    
    # Dernier recours: torchaudio original (peut échouer si torchcodec manque)
    if 'backend' not in kwargs:
        kwargs['backend'] = 'soundfile'
    return _original_torchaudio_load(filepath, *args, **kwargs)

torchaudio.load = _patched_torchaudio_load
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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

# Modèles globaux (chargés une seule fois au démarrage)
whisper_model = None
whisper_processor = None
pyannote_pipeline = None
device = None
hf_token_loaded = None


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
    """Charge le pipeline Pyannote."""
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
    
    # Déterminer le device pour Pyannote
    pyannote_device = _get_pyannote_device()
    if pyannote_device == "cuda":
        pyannote_pipeline.to(torch.device("cuda"))
        logger.info("Pyannote chargé sur GPU (CUDA)")
    else:
        pyannote_pipeline.to(torch.device("cpu"))
        logger.info("Pyannote chargé sur CPU")
    
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
    """Transcrit l'audio avec Whisper."""
    global whisper_model, whisper_processor, device
    
    from transformers import pipeline
    import librosa
    
    # Charger l'audio avec librosa (évite torchcodec)
    audio_array, sample_rate = librosa.load(audio_path, sr=16000)
    
    # Créer le pipeline de transcription
    pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=whisper_processor.tokenizer,
        feature_extractor=whisper_processor.feature_extractor,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device=device
    )
    
    # Options de génération
    generate_kwargs = {
        "return_timestamps": True,
    }
    if language:
        generate_kwargs["language"] = language
    
    # Passer l'audio comme dict pour éviter que le pipeline essaie de le charger
    audio_input = {"array": audio_array, "sampling_rate": sample_rate}
    
    # Transcription (ignore_warning pour chunk_length_s expérimental)
    result = pipe(
        audio_input,
        generate_kwargs=generate_kwargs,
        chunk_length_s=30,
        batch_size=8,
        ignore_warning=True
    )
    
    # Formater les segments
    segments = []
    if "chunks" in result:
        for chunk in result["chunks"]:
            if chunk.get("timestamp"):
                start, end = chunk["timestamp"]
                if start is not None and end is not None:
                    segments.append({
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "text": chunk["text"].strip()
                    })
    
    return {
        "text": result.get("text", "").strip(),
        "segments": segments
    }


def diarize_audio(audio_path: str, min_speakers: Optional[int] = None, 
                  max_speakers: Optional[int] = None) -> List[dict]:
    """Effectue la diarization avec Pyannote."""
    global pyannote_pipeline
    
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
            _fallback_pyannote_to_cpu()
            # Réessayer sur CPU
            diarization = pyannote_pipeline(audio_path, **params)
        else:
            raise
    
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker  # str comme "SPEAKER_00"
        })
    
    return segments


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
    """Retourne l'état détaillé du service."""
    device_info = get_device_info()
    
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
        }
    )


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
    temp_file = None
    
    try:
        # Sauvegarder le fichier
        logger.info(f"Réception fichier: {file.filename}")
        temp_file = save_upload_file(file)
        logger.info(f"Fichier sauvegardé: {temp_file}")
        
        # Charger Pyannote si nécessaire (et si diarization demandée)
        if not skip_diarization:
            try:
                load_pyannote_model(hf_token)
            except Exception as e:
                logger.error(f"Erreur chargement Pyannote: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Échec chargement Pyannote. Vérifiez le token HF. Erreur: {str(e)}"}
                )
        
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
        
        logger.info("Traitement terminé avec succès")
        
        return JSONResponse(content={
            "text": transcription_result["text"],
            "segments": merged_segments,
            "transcription_segments": transcription_result["segments"],
            "diarization_segments": diarization_segments
        })
        
    except Exception as e:
        logger.error(f"Erreur traitement: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Erreur traitement: {str(e)}"}
        )
    
    finally:
        # Nettoyage du fichier temporaire
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
                logger.debug(f"Fichier temporaire supprimé: {temp_file}")
            except Exception as e:
                logger.warning(f"Impossible de supprimer {temp_file}: {e}")


@app.post("/diarize")
async def diarize_only(
    file: UploadFile = File(...),
    hf_token: str = Form(...),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None)
):
    """
    Endpoint de rétrocompatibilité: Diarization seule.
    
    Retourne le même format que l'ancienne API.
    """
    temp_file = None
    
    try:
        # Sauvegarder le fichier
        temp_file = save_upload_file(file)
        
        # Charger Pyannote
        try:
            load_pyannote_model(hf_token)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Échec chargement pipeline. Vérifiez le token HF. Erreur: {str(e)}"}
            )
        
        # Diarization
        segments = diarize_audio(temp_file, min_speakers, max_speakers)
        
        return JSONResponse(content={"segments": segments})
        
    except Exception as e:
        logger.error(f"Erreur diarization: {e}", exc_info=True)
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


@app.post("/transcribe")
async def transcribe_only(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None)
):
    """
    Endpoint: Transcription seule (sans diarization).
    
    Ne nécessite pas de token HF.
    """
    temp_file = None
    
    try:
        temp_file = save_upload_file(file)
        
        result = transcribe_audio(temp_file, language)
        
        return JSONResponse(content={
            "text": result["text"],
            "segments": result["segments"]
        })
        
    except Exception as e:
        logger.error(f"Erreur transcription: {e}", exc_info=True)
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
