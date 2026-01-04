#!/bin/bash
#
# Script de démarrage du service Remote Processing
# Utilise distrobox avec le conteneur ROCm pour accéder au GPU AMD
#
# Usage:
#   ./start-server.sh              # Mode normal
#   ./start-server.sh --background # Mode background (nohup)
#   ./start-server.sh --install    # Installer les dépendances seulement
#

set -e

# Configuration
DISTROBOX_NAME="llama-rocm-7.1.1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="0.0.0.0"
PORT="${PORT:-8000}"
LOG_FILE="${SCRIPT_DIR}/server.log"

# Charger les variables depuis .env si le fichier existe
ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

HF_TOKEN="${HF_TOKEN:-}"

# Configuration ROCm 7.1.1 et venv Python 3.12
VENV_PATH="/opt/whisper-venv"
ROCM_LIB_PATH="/opt/rocm-7.1.1/lib"

# Commande pour activer l'environnement dans distrobox
run_in_venv() {
    distrobox enter "$DISTROBOX_NAME" -- bash -c "source $VENV_PATH/bin/activate && export LD_LIBRARY_PATH=$ROCM_LIB_PATH:\${LD_LIBRARY_PATH:-} && export HF_TOKEN='$HF_TOKEN' && $*"
}

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Configurer l'environnement (créer venv, installer Python si nécessaire)
setup_environment() {
    log_info "Configuration de l'environnement..."
    log_info "Venv cible: $VENV_PATH"
    
    # Créer un script temporaire pour éviter les problèmes de quotes
    local SETUP_SCRIPT=$(mktemp)
    cat > "$SETUP_SCRIPT" << 'SETUP_EOF'
#!/bin/bash
VENV_PATH="$1"

# Vérifier/installer Python
if ! command -v python3 &> /dev/null; then
    echo '[INFO] Installation de Python 3...'
    sudo dnf install -y python3 python3-pip python3-devel
else
    echo '[INFO] Python 3 déjà installé:' $(python3 --version)
fi

# Créer le venv s'il n'existe pas
if [ ! -d "$VENV_PATH" ]; then
    echo "[INFO] Création du venv: $VENV_PATH"
    sudo python3 -m venv "$VENV_PATH"
    sudo chown -R $(whoami):$(whoami) "$VENV_PATH"
else
    echo "[INFO] Venv déjà existant: $VENV_PATH"
fi
SETUP_EOF
    
    chmod +x "$SETUP_SCRIPT"
    distrobox enter "$DISTROBOX_NAME" -- bash "$SETUP_SCRIPT" "$VENV_PATH"
    rm -f "$SETUP_SCRIPT"
    
    log_info "Environnement configuré"
}

# Installer les dépendances
install_deps() {
    log_info "Installation des dépendances Python dans le venv..."
    log_info "Dossier: $SCRIPT_DIR"
    log_info "Venv: $VENV_PATH"
    
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
    run_in_venv 'python -c "import torch; print(\"GPU:\", torch.cuda.is_available()); print(\"HIP:\", torch.version.hip)"' || true
    
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
            echo "  --setup       Configurer l'environnement (créer venv, installer Python) + dépendances"
            echo "  --install     Installer les dépendances Python (venv doit exister)"
            echo "  --background  Démarrer le serveur en arrière-plan"
            echo "  --stop        Arrêter le serveur en arrière-plan"
            echo "  --help        Afficher cette aide"
            echo ""
            echo "Variables d'environnement:"
            echo "  PORT          Port du serveur (défaut: 8000)"
            echo "  HF_TOKEN      Token Hugging Face pour charger Pyannote au démarrage"
            ;;
        *)
            check_distrobox
            start_server
            ;;
    esac
}

main "$@"
