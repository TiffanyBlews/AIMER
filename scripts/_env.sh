# scripts/_env.sh
# Shared helpers for one-command train / inference scripts.
# Sourced by train_*.sh and infer_*.sh — do NOT run directly.

set -euo pipefail

# ---------- user-tunable defaults ----------
: "${DOCKER_NAME:=cuda_1111}"
: "${CONDA_ENV:=neo_meme}"
: "${REPO_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${HF_ENDPOINT:=https://hf-mirror.com}"
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${PYTORCH_ALLOC_CONF:=expandable_segments:True}"
: "${JINA_API_KEY:=}"            # only needed if you also enable Jina reranker

# ---------- helpers ----------
log()  { printf '\033[1;36m[%s]\033[0m %s\n'  "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# require_file <path>
require_file() {
    [[ -e "$1" ]] || die "Required file/dir missing: $1"
}

# ---------- main entry point ----------
# run_in_docker <bash-script-fragment>
# Runs the fragment inside the docker container with all env vars exported.
# If $DRY_RUN is set, it just prints the command without executing it.
run_in_docker() {
    local cmd="$1"
    local docker_cmd
    docker_cmd="source /root/anaconda3/etc/profile.d/conda.sh && \
                conda activate '$CONDA_ENV' && \
                cd '$REPO_ROOT' && \
                export HF_ENDPOINT='$HF_ENDPOINT' CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' PYTORCH_ALLOC_CONF='$PYTORCH_ALLOC_CONF'"
    if [[ -n "$JINA_API_KEY" ]]; then
        docker_cmd+=" JINA_API_KEY='$JINA_API_KEY'"
    fi
    docker_cmd+=" && $cmd"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '\n--- DRY RUN (in $%s) ---\n' "$DOCKER_NAME"
        printf 'docker exec -it %s bash -lc %s\n' "$DOCKER_NAME" "$(printf '%q' "$docker_cmd")"
        return 0
    fi
    log "docker exec ${DOCKER_NAME} ..."
    docker exec -e "HF_ENDPOINT=$HF_ENDPOINT" \
                -e "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
                -e "PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF" \
                ${JINA_API_KEY:+-e "JINA_API_KEY=$JINA_API_KEY"} \
                "$DOCKER_NAME" bash -lc "$docker_cmd"
}
