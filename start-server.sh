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

# Configuration ROCm 7.1.1 et venv Python 3.12
VENV_PATH="/opt/whisper-venv"
ROCM_LIB_PATH="/opt/rocm-7.1.1/lib"

# Commande pour activer l'environnement dans distrobox
run_in_venv() {
    distrobox enter "$DISTROBOX_NAME" -- bash -c "source $VENV_PATH/bin/activate && export LD_LIBRARY_PATH=$ROCM_LIB_PATH:\$LD_LIBRARY_PATH && $*"
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

# Installer les dépendances
install_deps() {
    log_info "Installation des dépendances Python dans le venv..."
    log_info "Dossier: $SCRIPT_DIR"
    log_info "Venv: $VENV_PATH"
    
    run_in_venv "pip install --upgrade pip"
    run_in_venv "pip install -r $SCRIPT_DIR/requirements.txt"
    
    log_info "Dépendances installées avec succès"
}

# Démarrer le serveur
start_server() {
    log_info "Démarrage du serveur sur $HOST:$PORT..."
    log_info "Dossier: $SCRIPT_DIR"
    log_info "Venv: $VENV_PATH"
    log_info "ROCm: $ROCM_LIB_PATH"
    
    # Vérifier l'accès GPU
    log_info "Vérification GPU..."
    run_in_venv "python -c \"import torch; print('GPU:', torch.cuda.is_available()); print('HIP:', torch.version.hip)\"" || true
    
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
    nohup distrobox enter "$DISTROBOX_NAME" -- bash -c "source $VENV_PATH/bin/activate && export LD_LIBRARY_PATH=$ROCM_LIB_PATH:\$LD_LIBRARY_PATH && cd $SCRIPT_DIR && uvicorn server:app --host $HOST --port $PORT" > "$LOG_FILE" 2>&1 &
    
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
            echo "Usage: $0 [--install|--background|--stop|--help]"
            echo ""
            echo "Options:"
            echo "  --install     Installer les dépendances Python"
            echo "  --background  Démarrer le serveur en arrière-plan"
            echo "  --stop        Arrêter le serveur en arrière-plan"
            echo "  --help        Afficher cette aide"
            echo ""
            echo "Variables d'environnement:"
            echo "  PORT          Port du serveur (défaut: 8000)"
            ;;
        *)
            check_distrobox
            start_server
            ;;
    esac
}

main "$@"
