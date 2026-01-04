#!/bin/bash
#
# Script de démarrage du service Remote Processing
# Optimisé pour AMD Strix Halo (gfx1151) avec ROCm 7.1.1
#
# Usage:
#   ./start-server.sh              # Mode normal
#   ./start-server.sh --background # Mode background (nohup)
#   ./start-server.sh --setup      # Configuration complète (venv + deps)
#   ./start-server.sh --install    # Installer les dépendances seulement
#

set -e

# Configuration
DISTROBOX_NAME="llama-rocm-7.1.1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="0.0.0.0"
PORT="${PORT:-8000}"
LOG_FILE="${SCRIPT_DIR}/server.log"

# Configuration ROCm 7.1.1
VENV_PATH="/opt/whisper-venv"
ROCM_LIB_PATH="/opt/rocm-7.1.1/lib"
ROCM_WHEELS_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1"

# Wheels PyTorch ROCm 7.1.1 pour Python 3.12
TORCH_WHEEL="torch-2.9.1%2Brocm7.1.1.lw.git351ff442-cp312-cp312-linux_x86_64.whl"
TORCHVISION_WHEEL="torchvision-0.24.0%2Brocm7.1.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl"
TORCHAUDIO_WHEEL="torchaudio-2.9.0%2Brocm7.1.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl"
TRITON_WHEEL="triton-3.5.1%2Brocm7.1.1.gita272dfa8-cp312-cp312-linux_x86_64.whl"

# Charger les variables depuis .env si le fichier existe
ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

HF_TOKEN="${HF_TOKEN:-}"

# Commande pour activer l'environnement dans distrobox
run_in_venv() {
    distrobox enter "$DISTROBOX_NAME" -- bash -c "source $VENV_PATH/bin/activate && export LD_LIBRARY_PATH=$ROCM_LIB_PATH:\${LD_LIBRARY_PATH:-} && export HF_TOKEN='$HF_TOKEN' && $*"
}

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Vérifier que distrobox existe
check_distrobox() {
    if ! command -v distrobox &> /dev/null; then
        log_error "distrobox n'est pas installé"
        exit 1
    fi
    
    if ! distrobox list | grep -q "$DISTROBOX_NAME"; then
        log_error "Le conteneur distrobox '$DISTROBOX_NAME' n'existe pas"
        log_info "Créez-le avec:"
        log_info "  distrobox create -n $DISTROBOX_NAME \\"
        log_info "    --image docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.1.1-rocwmma \\"
        log_info "    --additional-flags \"--device /dev/kfd --device /dev/dri --group-add video --group-add render --security-opt seccomp=unconfined\""
        exit 1
    fi
}

# Configurer l'environnement (créer venv avec Python 3.12)
setup_environment() {
    log_info "Configuration de l'environnement..."
    log_info "Venv cible: $VENV_PATH"
    log_info "Python requis: 3.12 (pour compatibilité wheels ROCm)"
    
    # Créer un script temporaire pour éviter les problèmes de quotes
    local SETUP_SCRIPT=$(mktemp)
    cat > "$SETUP_SCRIPT" << 'SETUP_EOF'
#!/bin/bash
set -e
VENV_PATH="$1"

echo "[INFO] Vérification de Python 3.12..."

# Sur Fedora, installer python3.12 spécifiquement
if ! command -v python3.12 &> /dev/null; then
    echo "[INFO] Installation de Python 3.12..."
    # Sur Fedora: python3.12 inclut pip, -devel pour compiler les extensions
    sudo dnf install -y python3.12 python3.12-devel || {
        # Fallback si le package exact n'existe pas
        echo "[WARN] python3.12 non trouvé, tentative avec python312..."
        sudo dnf install -y python312 python312-devel || {
            echo "[ERROR] Impossible d'installer Python 3.12"
            echo "[INFO] Essayez: sudo dnf install python3.12"
            exit 1
        }
    }
fi

# Vérifier que python3.12 est disponible
if ! command -v python3.12 &> /dev/null; then
    echo "[ERROR] Python 3.12 non disponible après installation"
    echo "[ERROR] Les wheels PyTorch ROCm nécessitent Python 3.12"
    exit 1
fi

echo "[INFO] Python 3.12 trouvé: $(python3.12 --version)"

# Supprimer le venv existant s'il utilise une mauvaise version de Python
if [ -d "$VENV_PATH" ]; then
    VENV_PYTHON_VERSION=$("$VENV_PATH/bin/python3" --version 2>/dev/null | grep -oP '\d+\.\d+' | head -1 || echo "unknown")
    if [ "$VENV_PYTHON_VERSION" != "3.12" ]; then
        echo "[WARN] Venv existant utilise Python $VENV_PYTHON_VERSION, suppression..."
        sudo rm -rf "$VENV_PATH"
    fi
fi

# Créer le venv avec Python 3.12
if [ ! -d "$VENV_PATH" ]; then
    echo "[INFO] Création du venv avec Python 3.12: $VENV_PATH"
    sudo python3.12 -m venv "$VENV_PATH"
    sudo chown -R $(whoami):$(whoami) "$VENV_PATH"
else
    echo "[INFO] Venv Python 3.12 déjà existant: $VENV_PATH"
fi

# Vérifier la version dans le venv
VENV_VERSION=$("$VENV_PATH/bin/python3" --version)
echo "[INFO] Version Python dans le venv: $VENV_VERSION"
SETUP_EOF
    
    chmod +x "$SETUP_SCRIPT"
    distrobox enter "$DISTROBOX_NAME" -- bash "$SETUP_SCRIPT" "$VENV_PATH"
    rm -f "$SETUP_SCRIPT"
    
    log_info "Environnement configuré avec Python 3.12"
}

# Installer les wheels PyTorch ROCm 7.1.1
install_rocm_pytorch() {
    log_step "Installation de PyTorch ROCm 7.1.1 pour Strix Halo..."
    
    # Créer un script d'installation
    local INSTALL_SCRIPT=$(mktemp)
    cat > "$INSTALL_SCRIPT" << ROCM_EOF
#!/bin/bash
set -e

VENV_PATH="$VENV_PATH"
ROCM_WHEELS_URL="$ROCM_WHEELS_URL"
TORCH_WHEEL="$TORCH_WHEEL"
TORCHVISION_WHEEL="$TORCHVISION_WHEEL"
TORCHAUDIO_WHEEL="$TORCHAUDIO_WHEEL"
TRITON_WHEEL="$TRITON_WHEEL"

source "\$VENV_PATH/bin/activate"

# Vérifier si PyTorch ROCm est déjà installé
TORCH_VERSION=\$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [[ "\$TORCH_VERSION" == *"rocm"* ]]; then
    echo "[INFO] PyTorch ROCm déjà installé: \$TORCH_VERSION"
    # Vérifier que le GPU est détecté
    GPU_OK=\$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    if [ "\$GPU_OK" = "True" ]; then
        echo "[INFO] GPU ROCm détecté correctement"
        exit 0
    else
        echo "[WARN] GPU non détecté, réinstallation des wheels..."
    fi
fi

# Supprimer les anciennes versions (CUDA ou ROCm corrompue)
echo "[INFO] Suppression des anciennes versions de PyTorch..."
pip uninstall -y torch torchvision torchaudio triton 2>/dev/null || true

# Supprimer les packages nvidia si présents
pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
    nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-cufile-cu12 \
    nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 \
    nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 nvidia-nvtx-cu12 2>/dev/null || true

# Télécharger les wheels ROCm
WHEEL_DIR="/tmp/rocm-wheels"
mkdir -p "\$WHEEL_DIR"
cd "\$WHEEL_DIR"

echo "[INFO] Téléchargement des wheels PyTorch ROCm 7.1.1..."
wget -nc -q --show-progress "\${ROCM_WHEELS_URL}/\${TRITON_WHEEL}" || true
wget -nc -q --show-progress "\${ROCM_WHEELS_URL}/\${TORCH_WHEEL}" || true
wget -nc -q --show-progress "\${ROCM_WHEELS_URL}/\${TORCHVISION_WHEEL}" || true
wget -nc -q --show-progress "\${ROCM_WHEELS_URL}/\${TORCHAUDIO_WHEEL}" || true

# Installer les wheels (ordre important: triton d'abord)
echo "[INFO] Installation des wheels..."
pip install "\${TRITON_WHEEL//%2B/+}"
pip install "\${TORCH_WHEEL//%2B/+}"
pip install "\${TORCHVISION_WHEEL//%2B/+}"
pip install "\${TORCHAUDIO_WHEEL//%2B/+}"

# Vérification
echo "[INFO] Vérification de l'installation..."
python3 -c "import torch; print('PyTorch:', torch.__version__); print('HIP:', torch.version.hip); print('GPU:', torch.cuda.is_available())"

echo "[OK] PyTorch ROCm 7.1.1 installé avec succès"
ROCM_EOF
    
    chmod +x "$INSTALL_SCRIPT"
    distrobox enter "$DISTROBOX_NAME" -- bash "$INSTALL_SCRIPT"
    rm -f "$INSTALL_SCRIPT"
}

# Installer les dépendances Python
install_deps() {
    log_info "Installation des dépendances Python..."
    log_info "Dossier: $SCRIPT_DIR"
    log_info "Venv: $VENV_PATH"
    
    # D'abord installer PyTorch ROCm
    install_rocm_pytorch
    
    # Ensuite installer les autres dépendances
    log_step "Installation des dépendances requirements.txt..."
    run_in_venv "pip install --upgrade pip"
    run_in_venv "pip install -r $SCRIPT_DIR/requirements.txt"
    
    log_info "Dépendances installées avec succès"
    log_warn "Note: silero-vad désactivé (onnxruntime non disponible sur ROCm)"
    log_info "Le service fonctionne sans VAD (fonctionnalité optionnelle)"
}

# Démarrer le serveur
start_server() {
    log_info "Démarrage du serveur sur $HOST:$PORT..."
    log_info "Dossier: $SCRIPT_DIR"
    log_info "Venv: $VENV_PATH"
    log_info "ROCm: $ROCM_LIB_PATH"
    if [ -n "$HF_TOKEN" ]; then
        log_info "HF_TOKEN: défini (Pyannote sera chargé au démarrage)"
    else
        log_warn "HF_TOKEN: non défini (Pyannote sera chargé à la première requête)"
    fi
    
    # Vérifier l'accès GPU
    log_info "Vérification GPU..."
    run_in_venv 'python3 -c "import torch; print(\"GPU:\", torch.cuda.is_available(), \"| HIP:\", torch.version.hip)"' || true
    
    # Lancer uvicorn
    cd "$SCRIPT_DIR"
    run_in_venv "cd $SCRIPT_DIR && uvicorn server:app --host $HOST --port $PORT"
}

# Démarrer en background
start_background() {
    log_info "Démarrage du serveur en arrière-plan..."
    log_info "Venv: $VENV_PATH"
    log_info "ROCm: $ROCM_LIB_PATH"
    
    cd "$SCRIPT_DIR"
    nohup distrobox enter "$DISTROBOX_NAME" -- bash -c "source $VENV_PATH/bin/activate && export LD_LIBRARY_PATH=$ROCM_LIB_PATH:\${LD_LIBRARY_PATH:-} && export HF_TOKEN='$HF_TOKEN' && cd $SCRIPT_DIR && uvicorn server:app --host $HOST --port $PORT" > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo $PID > "${SCRIPT_DIR}/server.pid"
    
    log_info "Serveur démarré avec PID: $PID"
    log_info "Logs: tail -f $LOG_FILE"
    log_info "Arrêter: kill \$(cat ${SCRIPT_DIR}/server.pid)"
}

# Arrêter le serveur
stop_server() {
    if [ -f "${SCRIPT_DIR}/server.pid" ]; then
        PID=$(cat "${SCRIPT_DIR}/server.pid")
        if kill -0 "$PID" 2>/dev/null; then
            log_info "Arrêt du serveur (PID: $PID)..."
            kill "$PID"
            rm "${SCRIPT_DIR}/server.pid"
            log_info "Serveur arrêté"
        else
            log_warn "Le processus $PID n'est plus actif"
            rm "${SCRIPT_DIR}/server.pid"
        fi
    else
        log_warn "Aucun fichier PID trouvé"
    fi
}

# Main
main() {
    case "${1:-}" in
        --setup)
            check_distrobox
            setup_environment
            install_deps
            ;;
        --install)
            check_distrobox
            install_deps
            ;;
        --background)
            check_distrobox
            start_background
            ;;
        --stop)
            stop_server
            ;;
        --help|-h)
            echo "Usage: $0 [--setup|--install|--background|--stop|--help]"
            echo ""
            echo "Options:"
            echo "  --setup       Configuration complète (créer venv + installer PyTorch ROCm + deps)"
            echo "  --install     Installer les dépendances (PyTorch ROCm + requirements.txt)"
            echo "  --background  Démarrer le serveur en arrière-plan"
            echo "  --stop        Arrêter le serveur en arrière-plan"
            echo "  --help        Afficher cette aide"
            echo ""
            echo "Variables d'environnement:"
            echo "  PORT          Port du serveur (défaut: 8000)"
            echo "  HF_TOKEN      Token Hugging Face pour charger Pyannote au démarrage"
            echo ""
            echo "Configuration GPU:"
            echo "  Ce script est optimisé pour AMD Strix Halo (gfx1151) avec ROCm 7.1.1"
            echo "  Les wheels PyTorch sont téléchargés automatiquement depuis:"
            echo "  $ROCM_WHEELS_URL"
            ;;
        *)
            check_distrobox
            start_server
            ;;
    esac
}

main "$@"
