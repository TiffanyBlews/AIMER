#!/usr/bin/env bash
# scripts/eval_jina_douban.sh
#
# Evaluate the trained Jina-reranker-M0 score head on the Douban test set.
#
# Usage:
#   bash scripts/eval_jina_douban.sh
#   bash scripts/eval_jina_douban.sh /path/to/checkpoint.pt
#   CHECKPOINT=/path/to/checkpoint.pt bash scripts/eval_jina_douban.sh
#   DRY_RUN=1 bash scripts/eval_jina_douban.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"

# ---------- paths ----------
DATA_PATH="${DATA_PATH:-./douban_data/input_file}"
IMAGE_PATH="${IMAGE_PATH:-./douban_data/image}"
SAVE_DIR="${SAVE_DIR:-./checkpoints_jina_douban_topk25_768}"
CLIP_CKPT="${CLIP_CKPT:-./clip_douban/clip_douban.pt}"
DEFAULT_CHECKPOINT="$SAVE_DIR/jina_m0_lora_best_douban.pt"
CHECKPOINT="${CHECKPOINT:-${1:-$DEFAULT_CHECKPOINT}}"

EVAL_CACHE="$SAVE_DIR/jina_features_eval_cache.pt"
PRECOMPUTED_TOPK="${PRECOMPUTED_TOPK:-./autoresearch_cache_douban/clip_best_topk/clip_best_topk_for_jina.pt}"
PAIRWISE_CACHE_EVAL="${PAIRWISE_CACHE_EVAL:-./autoresearch_cache_douban/jina_pairwise_cache_768_eval.pt}"

# ---------- fixed configuration ----------
DEVICE="${DEVICE:-cuda:0}"
TOPK_BASE="${TOPK_BASE:-25}"
EXTRA_NEGATIVES="${EXTRA_NEGATIVES:-25}"
IMAGE_MAX_SIDE="${IMAGE_MAX_SIDE:-768}"
JINA_MICRO_BATCH="${JINA_MICRO_BATCH:-8}"

mkdir -p "$SAVE_DIR"
require_file "$CHECKPOINT"
require_file "$DATA_PATH/test_data.json"
require_file "$IMAGE_PATH"

if [[ ! -f "$EVAL_CACHE" ]]; then
  log 'Precomputing Jina evaluation features (768px).'
  require_file "$CLIP_CKPT"
  PRE_CMD="python -u models/rerank/precompute_jina_features.py \
    --data_path '$DATA_PATH' \
    --image_path '$IMAGE_PATH' \
    --save_dir '$SAVE_DIR' \
    --device '$DEVICE' \
    --topk_base '$TOPK_BASE' \
    --candidate_mode topk_plus_pos \
    --use_clip_sim --use_emotion \
    --clip_checkpoint '$CLIP_CKPT' \
    --eval_cache \
    --image_max_side '$IMAGE_MAX_SIDE' \
    --jina_micro_batch '$JINA_MICRO_BATCH'"
  [[ -n "$PRECOMPUTED_TOPK" ]] && PRE_CMD="$PRE_CMD --precomputed_topk '$PRECOMPUTED_TOPK'"
  [[ -n "$PAIRWISE_CACHE_EVAL" ]] && PRE_CMD="$PRE_CMD --pairwise_cache '$PAIRWISE_CACHE_EVAL'"
  run_in_docker "$PRE_CMD 2>&1 | tee '$SAVE_DIR/precompute_eval.log'"
else
  log "Reusing evaluation cache: $EVAL_CACHE"
fi

EVAL_CMD="python -u models/rerank/jina_m0_lora_train.py \
  --data_path '$DATA_PATH' \
  --image_path '$IMAGE_PATH' \
  --save_dir '$SAVE_DIR' \
  --device '$DEVICE' \
  --topk_base '$TOPK_BASE' \
  --extra_negatives '$EXTRA_NEGATIVES' \
  --candidate_mode topk \
  --label_mode inv_rank \
  --use_clip_sim \
  --use_emotion \
  --use_jina_cache \
  --train_lora False \
  --eval_sample_limit 0 \
  --clip_checkpoint '$CLIP_CKPT' \
  --wandb_mode disabled \
  --resume_path '$CHECKPOINT' \
  --eval_only \
  2>&1 | tee '$SAVE_DIR/eval_best.log'"

log "Evaluating checkpoint: $CHECKPOINT"
run_in_docker "$EVAL_CMD"
log 'Done. Metrics are printed above.'
