#!/usr/bin/env bash
# scripts/train_imgflip.sh
#
# Fine-tune CLIP ViT-B/32 on the imgflip dataset end-to-end.
#
# Usage:
#   bash scripts/train_imgflip.sh                 # full run with defaults
#   BATCH_SIZE=256 EPOCHS=3 bash scripts/train_imgflip.sh
#   DRY_RUN=1 bash scripts/train_imgflip.sh       # only print the command
#
# All knobs come from environment variables; defaults match the canonical
# training settings documented in CLAUDE.md.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"

# ---------- dataset-specific paths ----------
DATA_PATH="${DATA_PATH:-./imgflip_data/msrvtt}"
IMAGE_PATH="${IMAGE_PATH:-./imgflip_data/images}"
SAVE_DIR="${SAVE_DIR:-./clip_imgflip}"
MODEL_NAME="${MODEL_NAME:-clip_imgflip.pt}"

# ---------- training hyperparameters (override via env) ----------
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-5e-6}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
SEED="${SEED:-42}"

# ---------- sanity checks ----------
log "imgflip fine-tune (CLIP ViT-B/32)"
log "data_path=${DATA_PATH} image_path=${IMAGE_PATH} save_dir=${SAVE_DIR}"
require_file "${DATA_PATH}/train_data.json"
require_file "${DATA_PATH}/train_emotion.json"
require_file "${DATA_PATH}/test_data.json"
require_file "${DATA_PATH}/train_ids.csv"
require_file "${IMAGE_PATH}"
if [[ -f "${SAVE_DIR}/${MODEL_NAME}" && "${FORCE_TRAIN:-0}" != "1" ]]; then
    warn "Checkpoint already exists at ${SAVE_DIR}/${MODEL_NAME}; skipping (set FORCE_TRAIN=1 to re-train)."
    exit 0
fi

# ---------- build the python command ----------
PY_CMD="python -u models/clip/clip_train.py \
    --data_path '${DATA_PATH}' \
    --image_path '${IMAGE_PATH}' \
    --save_dir '${SAVE_DIR}' \
    --model_save_name '${MODEL_NAME}' \
    --pretrained_clip_name ViT-B/32 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --use_image_emotion_fusion False \
    --use_text_emotion_fusion False"

run_in_docker "$PY_CMD"
log "Training finished; checkpoint at ${SAVE_DIR}/${MODEL_NAME}"
