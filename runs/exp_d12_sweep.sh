#!/bin/bash
# Phase 1 at d12: AR vs MDM at matched tokens, SFT on both, MDM at 2x and 4x tokens, Muon-LR sweep for both.
# Ordered so the most important results land first; every stage is skipped if its checkpoint already exists,
# so the script can be re-run after an interruption. Fixed 524,288-token batch so loss-vs-tokens curves are
# directly comparable. Needs ~160 shards for the 4x run (nanochat's loader would otherwise cycle the data).
# Usage: NPROC=1 bash runs/exp_d12_sweep.sh   (set SKIP_SETUP=1 if venv/data/tokenizer already exist)
set -euo pipefail
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
NPROC="${NPROC:-8}"
if [ -z "${SKIP_SETUP:-}" ]; then
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
    source .venv/bin/activate
    python -m scripts.tok_train --max-chars=2000000000
    python -m scripts.tok_eval
fi
source .venv/bin/activate
python -m nanochat.dataset -n 170   # 4x d12 budget = 5.3B tokens ~ 160 shards at ~34M usable tokens/shard
COMMON="DEPTH=12 NPROC=$NPROC BATCH=524288 SAVE_EVERY=500"
have_base() { ls "$NANOCHAT_BASE_DIR/base_checkpoints/$1"/model_*.pt >/dev/null 2>&1; }
have_sft()  { ls "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$1"/model_*.pt >/dev/null 2>&1; }
pretrain() { # TAG MODEL_TYPE RATIO [EXTRA]
    if have_base "$1"; then echo "=== skip pretrain $1 (checkpoint exists)"; return; fi
    env $COMMON TAG="$1" MODEL_TYPE="$2" RATIO="$3" EXTRA="${4:-}" bash runs/exp_pretrain.sh
}
posttrain() { # TAG
    if have_sft "$1"; then echo "=== skip posttrain $1 (checkpoint exists)"; return; fi
    env TAG="$1" NPROC=$NPROC DBS=16 bash runs/exp_posttrain.sh
}

# 1) matched-token pair (nanochat default ratio 12) and post-training on both
pretrain d12_ar  ar  12
pretrain d12_mdm mdm 12
posttrain d12_ar
posttrain d12_mdm

# 2) MDM at 2x the AR token budget: does it cross the AR loss?
pretrain d12_mdm_2x mdm 24

# 3) Muon matrix-LR sweep (default 0.02) for both architectures
for LR in 0.01 0.04; do
    pretrain d12_ar_lr$LR  ar  12 "--matrix-lr=$LR"
    pretrain d12_mdm_lr$LR mdm 12 "--matrix-lr=$LR"
done

# 4) MDM at 4x tokens
pretrain d12_mdm_4x mdm 48
echo "===== D12 SWEEP DONE ====="
