#!/bin/bash
# One pretraining run (AR or MDM) followed by the full base-model eval.
# Usage (env vars):
#   MODEL_TYPE=ar|mdm  DEPTH=12  RATIO=12  TAG=d12_ar  NPROC=8  [DBS=32] [BATCH=524288] [EXTRA="..."] bash runs/exp_pretrain.sh
# RATIO is the tokens : scaling-params ratio (nanochat default 12; MDM 2x/4x budgets use 24/48).
# Logs: $NANOCHAT_BASE_DIR/logs/nanochat/$TAG (TensorBoard events + metrics.jsonl); checkpoints under base_checkpoints/$TAG.
set -euo pipefail
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
MODEL_TYPE="${MODEL_TYPE:-ar}"
DEPTH="${DEPTH:-12}"
RATIO="${RATIO:-12}"
TAG="${TAG:-d${DEPTH}_${MODEL_TYPE}}"
NPROC="${NPROC:-8}"
DBS="${DBS:-32}"
BATCH="${BATCH:--1}"            # -1 = nanochat auto batch size; set explicitly to hold it fixed across token budgets
EXTRA="${EXTRA:-}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"       # 1 = checkpoint already exists, only run base_eval
source .venv/bin/activate
mkdir -p "$NANOCHAT_BASE_DIR/logs"

if [ "$SKIP_TRAIN" != "1" ]; then
echo "=== pretrain $TAG: model_type=$MODEL_TYPE depth=$DEPTH ratio=$RATIO nproc=$NPROC batch=$BATCH extra='$EXTRA'"
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    --model-type=$MODEL_TYPE --depth=$DEPTH --target-param-data-ratio=$RATIO \
    --total-batch-size=$BATCH --device-batch-size=$DBS \
    --run=$TAG --model-tag=$TAG \
    --core-metric-every=-1 --sample-every=-1 --save-every=$SAVE_EVERY \
    $EXTRA 2>&1 | tee -a "$NANOCHAT_BASE_DIR/logs/${TAG}_pretrain.log"
fi

echo "=== base_eval $TAG"
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_eval -- \
    --model-tag=$TAG --device-batch-size=$DBS --mdm-score=l2r --mdm-mc=4 \
    2>&1 | tee -a "$NANOCHAT_BASE_DIR/logs/${TAG}_base_eval.log"
