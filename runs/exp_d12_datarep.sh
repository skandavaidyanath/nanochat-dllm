#!/bin/bash
# Phase 3, data-constrained regime at d12: same 1.32B-token budget, but only the first 5 shards
# (4 train + 1 val, ~136M unique training tokens => ~10 epochs). Literature claim: diffusion wins under heavy repetition.
# Usage: NPROC=1 bash runs/exp_d12_datarep.sh   (waits for a running exp_d12_sweep.sh to finish first)
set -euo pipefail
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
NPROC="${NPROC:-1}"
while pgrep -f exp_d12_sweep.sh >/dev/null; do sleep 60; done
source .venv/bin/activate
COMMON="DEPTH=12 NPROC=$NPROC BATCH=524288 SAVE_EVERY=500"
for MT in ar mdm; do
    TAG=d12_${MT}_rep10
    if ls "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG"/model_*.pt >/dev/null 2>&1; then echo "=== skip $TAG"; continue; fi
    env $COMMON TAG=$TAG MODEL_TYPE=$MT RATIO=12 EXTRA="--max-shards=5" bash runs/exp_pretrain.sh
done
echo "===== D12 DATAREP DONE ====="
