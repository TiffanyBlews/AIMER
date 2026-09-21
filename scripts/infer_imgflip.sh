#!/usr/bin/env bash
# scripts/infer_imgflip.sh
#
# Run the best-tuned imgflip retrieval/evaluation pipeline.
# Expects a trained checkpoint at ${CHECKPOINT} (defaults to
# ./clip_imgflip/clip_imgflip.pt, the file produced by train_imgflip.sh).
#
# Expected result on the full test set:
#   R@1≈26.79%   R@5≈39.53%   R@10≈44.42%   comp = 3*R@1 + 2*R@5 + R@10 = 203.87
#
# Usage:
#   bash scripts/infer_imgflip.sh                                # full inference
#   NUM_SAMPLES=100 SAMPLE_MODE=random bash scripts/infer_imgflip.sh   # quick smoke test
#   DRY_RUN=1 bash scripts/infer_imgflip.sh                      # only print
#
# Knobs:
#   CHECKPOINT         Path to a CLIP checkpoint (default: ./clip_imgflip/clip_imgflip.pt)
#   SAVE_DIR           Cache directory (default: ./autoresearch_cache_imgflip)
#   NUM_SAMPLES        0 = full set; >0 = sample this many queries
#   SAMPLE_MODE        "first" | "random"
#   TOPK_BASE          base retrieval depth (default: 50)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"

# ---------- dataset paths ----------
DATA_PATH="${DATA_PATH:-./imgflip_data/msrvtt}"
IMAGE_PATH="${IMAGE_PATH:-./imgflip_data/images}"
CHECKPOINT="${CHECKPOINT:-./clip_imgflip/clip_imgflip.pt}"
SAVE_DIR="${SAVE_DIR:-./autoresearch_cache_imgflip}"
TEST_JSON="${TEST_JSON:-test_data.json}"
TEST_EMOTION_JSON="${TEST_EMOTION_JSON:-test_emotion.json}"
TOPK_BASE="${TOPK_BASE:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
SAMPLE_MODE="${SAMPLE_MODE:-first}"
SAMPLE_SEED="${SAMPLE_SEED:-114}"

# ---------- sanity ----------
log "imgflip inference (best config: comp=203.87, R@1=26.79 / R@5=39.53 / R@10=44.42)"
log "checkpoint=${CHECKPOINT} save_dir=${SAVE_DIR} num_samples=${NUM_SAMPLES}"
require_file "${DATA_PATH}/${TEST_JSON}"
require_file "${DATA_PATH}/${TEST_EMOTION_JSON}"
require_file "${IMAGE_PATH}"
require_file "${CHECKPOINT}"

mkdir -p "${SAVE_DIR}"

# ---------- best-tuned command (matches best_configs/imgflip_best_config.md) ----------
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
    --query_expansion_alpha 0.01 \
    --query_expansion_alpha_image 0.03 \
    --query_expansion_k_schedule '1,1,1' \
    --query_expansion_k_image_schedule '1,1,1' \
    --query_expansion_iters 3"

run_in_docker "$PY_CMD"
log "Done. Check ${SAVE_DIR}/ for cached features and metrics in the log."
