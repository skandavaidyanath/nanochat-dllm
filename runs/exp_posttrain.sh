#!/bin/bash
# SFT + chat eval (+ optional RL) for a pretrained checkpoint tag. Works for AR and MDM checkpoints.
# Usage: TAG=d12_ar NPROC=8 [SEED_TAG=s0] [RL=1] [SFT_EXTRA="..."] bash runs/exp_posttrain.sh
set -euo pipefail
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
TAG="${TAG:?set TAG}"
NPROC="${NPROC:-8}"
DBS="${DBS:-16}"
RL="${RL:-0}"
SFT_EXTRA="${SFT_EXTRA:-}"
RL_EXTRA="${RL_EXTRA:-}"
source .venv/bin/activate

echo "=== sft $TAG"
torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_sft -- \
    --model-tag=$TAG --run=${TAG}_sft --device-batch-size=$DBS $SFT_EXTRA \
    2>&1 | tee -a "$NANOCHAT_BASE_DIR/logs/${TAG}_sft.log"

echo "=== chat_eval sft $TAG"
torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_eval -- -i sft -g $TAG -b $DBS \
    2>&1 | tee -a "$NANOCHAT_BASE_DIR/logs/${TAG}_chat_eval_sft.log"

if [ "$RL" = "1" ]; then
    echo "=== rl $TAG"
    torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_rl -- \
        --model-tag=$TAG --run=${TAG}_rl $RL_EXTRA \
        2>&1 | tee -a "$NANOCHAT_BASE_DIR/logs/${TAG}_rl.log"
    echo "=== chat_eval rl $TAG"
    torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_eval -- -i rl -g $TAG -b $DBS \
        2>&1 | tee -a "$NANOCHAT_BASE_DIR/logs/${TAG}_chat_eval_rl.log"
fi
