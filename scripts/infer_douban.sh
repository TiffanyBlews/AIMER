#!/usr/bin/env bash
# scripts/infer_douban.sh
#
# Run the best-tuned douban retrieval/evaluation pipeline.
# Expects a trained checkpoint at ${CHECKPOINT} (default
# ./clip_douban/clip_douban.pt) AND the learned ridge projection matrix at
# ${IMAGE_PROJECTION_PATH} (default ./autoresearch_cache_douban/image_proj_ridge_l1.pt).
#
# Expected result on the full test set:
#   R@1≈13.85%   R@5≈19.55%   R@10≈24.32%   comp = 3*R@1 + 2*R@5 + R@10 = 104.97
#
# Usage:
#   bash scripts/infer_douban.sh
#   NUM_SAMPLES=100 SAMPLE_MODE=random bash scripts/infer_douban.sh
#   DRY_RUN=1 bash scripts/infer_douban.sh
#
# Knobs:
#   CHECKPOINT              Path to a CLIP checkpoint
#   IMAGE_PROJECTION_PATH   Path to a learned ridge projection matrix (.pt with key 'W')
#   SAVE_DIR                Cache directory
#   NUM_SAMPLES, SAMPLE_MODE, SAMPLE_SEED, TOPK_BASE, BATCH_SIZE

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"

# ---------- dataset paths ----------
DATA_PATH="${DATA_PATH:-./douban_data/input_file}"
IMAGE_PATH="${IMAGE_PATH:-./douban_data/image}"
CHECKPOINT="${CHECKPOINT:-./clip_douban/clip_douban.pt}"
SAVE_DIR="${SAVE_DIR:-./autoresearch_cache_douban}"
IMAGE_PROJECTION_PATH="${IMAGE_PROJECTION_PATH:-${SAVE_DIR}/image_proj_ridge_l1.pt}"
TEST_JSON="${TEST_JSON:-test_data.json}"
TEST_EMOTION_JSON="${TEST_EMOTION_JSON:-test_emotion.json}"
TOPK_BASE="${TOPK_BASE:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
SAMPLE_MODE="${SAMPLE_MODE:-first}"
SAMPLE_SEED="${SAMPLE_SEED:-114}"

# ---------- sanity ----------
log "douban inference (best config: comp=104.97, R@1=13.85 / R@5=19.55 / R@10=24.32)"
log "checkpoint=${CHECKPOINT} projection=${IMAGE_PROJECTION_PATH} save_dir=${SAVE_DIR}"
require_file "${DATA_PATH}/${TEST_JSON}"
require_file "${DATA_PATH}/${TEST_EMOTION_JSON}"
require_file "${IMAGE_PATH}"
require_file "${CHECKPOINT}"
require_file "${IMAGE_PROJECTION_PATH}"

mkdir -p "${SAVE_DIR}"

# ---------- best-tuned command (matches best_configs/douban_best_config.md) ----------
PY_CMD="python -u models/clip/clip_infer.py \
    --data_path '${DATA_PATH}' \
    --image_path '${IMAGE_PATH}' \
    --checkpoint '${CHECKPOINT}' \
    --test_json '${TEST_JSON}' \
    --test_emotion_json '${TEST_EMOTION_JSON}' \
    --save_dir '${SAVE_DIR}' \
    --topk_base ${TOPK_BASE} \
    --batch_size ${BATCH_SIZE} \
    --num_samples ${NUM_SAMPLES} \
    --sample_mode ${SAMPLE_MODE} \
    --sample_seed ${SAMPLE_SEED} \
    --query_expansion_alpha 0.03 \
    --query_expansion_alpha_image 0.027 \
    --query_expansion_k_schedule '3,2,2' \
    --query_expansion_k_image_schedule '2,3,1' \
    --query_expansion_iters 3 \
    --image_projection_path '${IMAGE_PROJECTION_PATH}' \
    --projection_mix 0.9 \
    --image_projection_sim_fusion 0.5"

run_in_docker "$PY_CMD"
log "Done. Check ${SAVE_DIR}/ for cached features and metrics in the log."
