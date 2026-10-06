#!/bin/bash
# Phase 1 at d12: AR vs MDM at matched tokens, MDM at 2x and 4x tokens, LR sweep for both, SFT on the best of each.
# Everything uses a fixed 524,288-token batch so loss-vs-tokens curves are directly comparable.
# Usage: NPROC=8 bash runs/exp_d12_sweep.sh   (set SKIP_SETUP=1 if data/tokenizer already exist)
set -euo pipefail
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
NPROC="${NPROC:-8}"
if [ -z "${SKIP_SETUP:-}" ]; then
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
    source .venv/bin/activate
    python -m nanochat.dataset -n 60   # d12 x4 tokens ~ 5.2B tokens ~ 55 shards
    python -m scripts.tok_train --max-chars=2000000000
    python -m scripts.tok_eval
fi
COMMON="DEPTH=12 NPROC=$NPROC BATCH=524288 SAVE_EVERY=500"

# 1) matched-token pair (nanochat default ratio 12)
env $COMMON MODEL_TYPE=ar  RATIO=12 TAG=d12_ar  bash runs/exp_pretrain.sh
env $COMMON MODEL_TYPE=mdm RATIO=12 TAG=d12_mdm bash runs/exp_pretrain.sh

# 2) matrix-LR sweep (Muon LR, default 0.02) for both architectures
for LR in 0.01 0.04; do
    env $COMMON MODEL_TYPE=ar  RATIO=12 TAG=d12_ar_lr$LR  EXTRA="--matrix-lr=$LR" bash runs/exp_pretrain.sh
    env $COMMON MODEL_TYPE=mdm RATIO=12 TAG=d12_mdm_lr$LR EXTRA="--matrix-lr=$LR" bash runs/exp_pretrain.sh
done

# 3) MDM at 2x and 4x the AR token budget: where does it cross the AR loss?
env $COMMON MODEL_TYPE=mdm RATIO=24 TAG=d12_mdm_2x bash runs/exp_pretrain.sh
env $COMMON MODEL_TYPE=mdm RATIO=48 TAG=d12_mdm_4x bash runs/exp_pretrain.sh

# 4) post-training on the matched pair (3 SFT seeds each are run by exp_d12_posttrain_seeds.sh)
env TAG=d12_ar  NPROC=$NPROC bash runs/exp_posttrain.sh
env TAG=d12_mdm NPROC=$NPROC bash runs/exp_posttrain.sh
