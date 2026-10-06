#!/bin/bash
# Phase 2 at d24 (the nanochat speedrun config): AR vs MDM pretraining at matched tokens, then SFT + evals.
# Usage: NPROC=8 bash runs/exp_d24_pair.sh
set -euo pipefail
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
NPROC="${NPROC:-8}"
if [ -z "${SKIP_SETUP:-}" ]; then
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
    source .venv/bin/activate
    python -m nanochat.dataset -n 170
    python -m scripts.tok_train --max-chars=2000000000
    python -m scripts.tok_eval
fi
# speedrun settings: depth 24, ratio 8, fp8, device batch 16 (see runs/speedrun.sh)
COMMON="DEPTH=24 RATIO=8 NPROC=$NPROC DBS=16 SAVE_EVERY=1000"
env $COMMON MODEL_TYPE=ar  TAG=d24_ar  EXTRA="--fp8" bash runs/exp_pretrain.sh
env $COMMON MODEL_TYPE=mdm TAG=d24_mdm EXTRA="--fp8" bash runs/exp_pretrain.sh
env TAG=d24_ar  NPROC=$NPROC DBS=16 bash runs/exp_posttrain.sh
env TAG=d24_mdm NPROC=$NPROC DBS=16 bash runs/exp_posttrain.sh
