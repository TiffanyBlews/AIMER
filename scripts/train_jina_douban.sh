#!/usr/bin/env bash
# scripts/train_jina_douban.sh
#
# Train the Jina-reranker-M0 score head on the Douban dataset.
# Best recorded setting: composite = 109.42.
#
# Usage:
#   bash scripts/train_jina_douban.sh
#   SAVE_DIR=/path/to/output bash scripts/train_jina_douban.sh
#   DRY_RUN=1 bash scripts/train_jina_douban.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"

# ---------- paths (all can be overridden by environment variables) ----------
DATA_PATH="${DATA_PATH:-./douban_data/input_file}"
IMAGE_PATH="${IMAGE_PATH:-./douban_data/image}"
SAVE_DIR="${SAVE_DIR:-./checkpoints_jina_douban_topk25_768}"
CLIP_CKPT="${CLIP_CKPT:-./clip_douban/clip_douban.pt}"

TRAIN_CACHE="$SAVE_DIR/jina_features_cache.pt"
EVAL_CACHE="$SAVE_DIR/jina_features_eval_cache.pt"
PRECOMPUTED_TOPK="${PRECOMPUTED_TOPK:-./autoresearch_cache_douban/clip_best_topk/clip_best_topk_for_jina.pt}"
PAIRWISE_CACHE_TRAIN="${PAIRWISE_CACHE_TRAIN:-./autoresearch_cache_douban/jina_pairwise_cache_768.pt}"
PAIRWISE_CACHE_EVAL="${PAIRWISE_CACHE_EVAL:-./autoresearch_cache_douban/jina_pairwise_cache_768_eval.pt}"

# ---------- fixed best configuration ----------
DEVICE="${DEVICE:-cuda:0}"
TOPK_BASE="${TOPK_BASE:-25}"
EXTRA_NEGATIVES="${EXTRA_NEGATIVES:-25}"
IMAGE_MAX_SIDE="${IMAGE_MAX_SIDE:-768}"
JINA_MICRO_BATCH="${JINA_MICRO_BATCH:-8}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-jina_m0_lora_epoch_1_pre_eval_jina_m0_lora_focal_ranknet_inv_rank.pt}"
BEST_ALIAS="$SAVE_DIR/jina_m0_lora_best_douban.pt"

mkdir -p "$SAVE_DIR"

common_precompute_args() {
  local mode="$1"
  local pairwise="$2"
  local cmd="python -u models/rerank/precompute_jina_features.py \
    --data_path '$DATA_PATH' \
    --image_path '$IMAGE_PATH' \
    --save_dir '$SAVE_DIR' \
    --device '$DEVICE' \
    --topk_base '$TOPK_BASE' \
    --image_max_side '$IMAGE_MAX_SIDE' \
    --jina_micro_batch '$JINA_MICRO_BATCH' \
    --use_clip_sim --use_emotion \
    --clip_checkpoint '$CLIP_CKPT'"

  if [[ "$mode" == "train" ]]; then
    cmd="$cmd --extra_negatives '$EXTRA_NEGATIVES' \
      --candidate_mode topk --label_mode inv_rank --resume"
  else
    cmd="$cmd --candidate_mode topk_plus_pos --eval_cache"
  fi

  [[ -n "$PRECOMPUTED_TOPK" ]] && cmd="$cmd --precomputed_topk '$PRECOMPUTED_TOPK'"
  [[ -n "$pairwise" ]] && cmd="$cmd --pairwise_cache '$pairwise'"
  printf '%s' "$cmd"
}

if [[ ! -f "$TRAIN_CACHE" ]]; then
  log 'Precomputing Jina training features (768px, topk25).'
  require_file "$DATA_PATH/train_data.json"
  require_file "$IMAGE_PATH"
  require_file "$CLIP_CKPT"
  run_in_docker "$(common_precompute_args train "$PAIRWISE_CACHE_TRAIN") \
    2>&1 | tee '$SAVE_DIR/precompute_train.log'"
else
  log "Reusing training cache: $TRAIN_CACHE"
fi

if [[ ! -f "$EVAL_CACHE" ]]; then
  log 'Precomputing Jina evaluation features (768px).'
  require_file "$DATA_PATH/test_data.json"
  require_file "$IMAGE_PATH"
  require_file "$CLIP_CKPT"
  run_in_docker "$(common_precompute_args eval "$PAIRWISE_CACHE_EVAL") \
    2>&1 | tee '$SAVE_DIR/precompute_eval.log'"
else
  log "Reusing evaluation cache: $EVAL_CACHE"
fi

TRAIN_CMD="python -u models/rerank/jina_m0_lora_train.py \
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
  --eval_before_train \
  --eval_sample_limit 0 \
  --clip_checkpoint '$CLIP_CKPT' \
  --wandb_mode disabled \
  2>&1 | tee '$SAVE_DIR/train_best.log'"

log 'Training Jina score head with the best Douban configuration.'
run_in_docker "$TRAIN_CMD"

COPY_CMD="if [[ -f '$SAVE_DIR/$CHECKPOINT_NAME' ]]; then \
  cp -f '$SAVE_DIR/$CHECKPOINT_NAME' '$BEST_ALIAS'; \
  echo '[train] copied stable checkpoint alias: $BEST_ALIAS'; \
else \
  echo '[train] expected checkpoint not found: $SAVE_DIR/$CHECKPOINT_NAME' >&2; \
  exit 1; \
fi"
run_in_docker "$COPY_CMD"

log "Done. Artifacts are under $SAVE_DIR."
