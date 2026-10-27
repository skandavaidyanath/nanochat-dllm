#!/bin/bash
# Phase 2b at d12, one GPU:
#   A) diffusion SFT with deterministic masking of short answers (--mdm-force-mask-short=2) on the 3x-MMLU / 0.3x-LR
#      recipe, i.e. phase-2a run sft2_mdm_m9_lr0.3_x2 with the answer letter always masked and weighted 1;
#   B) AR-initialised diffusion: the d12_ar checkpoint continued with the diffusion objective for 1260 steps
#      (half the d12 budget, same data order, fresh optimizer, 100-step warmup), then the base eval (CORE).
# Both skip when their checkpoint already exists. Usage: bash runs/exp_d12_phase2b.sh
set -u
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}" OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
DBS="${DBS:-16}"
source .venv/bin/activate
mkdir -p "$NANOCHAT_BASE_DIR/logs"
have_sft()  { ls "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$1"/model_*.pt >/dev/null 2>&1; }
have_base() { ls "$NANOCHAT_BASE_DIR/base_checkpoints/$1"/model_*.pt >/dev/null 2>&1; }
have_eval() { ls "$NANOCHAT_BASE_DIR/base_eval/$1"_*.csv >/dev/null 2>&1; }

# A) forced answer masking
NAME=sft3_mdm_m9_lr0.3_force2
EVAL_LOG="$NANOCHAT_BASE_DIR/logs/${NAME}_chat_eval.log"
if have_sft "$NAME" && grep -q "GSM8K" "$EVAL_LOG" 2>/dev/null; then echo "=== skip $NAME (done)"; else
  if ! have_sft "$NAME"; then
    echo "===== SFT $NAME START $(date -u +%H:%M:%S) ====="
    python -m scripts.chat_sft --model-tag=d12_mdm --output-tag=$NAME --run=$NAME --device-batch-size=$DBS \
        --mmlu-epochs=9 --init-lr-frac=0.3 --num-iterations=29888 --mdm-force-mask-short=2 \
        --eval-every=400 --chatcore-every=400 --chatcore-max-cat=400 --chatcore-max-sample=8 \
        2>&1 | tee "$NANOCHAT_BASE_DIR/logs/${NAME}_sft.log" \
        | grep --line-buffered -E "Training mixture|Validation bpb|ChatCORE|ARC|MMLU|GSM8K|HumanEval|Traceback|Error|Total training time"
  fi
  if have_sft "$NAME"; then
    echo "===== CHAT_EVAL $NAME START $(date -u +%H:%M:%S) ====="
    python -m scripts.chat_eval -i sft -g $NAME -b $DBS -a "ARC-Easy|ARC-Challenge|MMLU" 2>&1 | tee "$EVAL_LOG" | grep --line-buffered -E "ARC|MMLU|ChatCORE|Traceback|Error"
    python -m scripts.chat_eval -i sft -g $NAME -b $DBS -a "GSM8K" -x 100 2>&1 | tee -a "$EVAL_LOG" | grep --line-buffered -E "GSM8K|Traceback|Error"
    echo "===== SFT $NAME DONE $(date -u +%H:%M:%S) ====="
  else
    echo "===== SFT $NAME FAILED (no checkpoint) $(date -u +%H:%M:%S) ====="
  fi
fi

# B) AR-initialised diffusion pretraining
TAG=d12_mdm_from_ar
if have_base "$TAG" && have_eval "$TAG"; then echo "=== skip $TAG (done)"; else
  if ! have_base "$TAG"; then
    echo "===== PRETRAIN $TAG START $(date -u +%H:%M:%S) ====="
    python -m scripts.base_train --model-type=mdm --depth=12 --num-iterations=1260 --total-batch-size=524288 --device-batch-size=32 \
        --init-from-tag=d12_ar --init-from-step=2520 --warmup-steps=100 \
        --run=$TAG --model-tag=$TAG --core-metric-every=-1 --sample-every=-1 --save-every=500 \
        2>&1 | tee "$NANOCHAT_BASE_DIR/logs/${TAG}_pretrain.log" \
        | grep --line-buffered -E "Initialised|Validation|val/bpb|step 0*(100|500|1000|1260) |Traceback|Error|Total training time|Minimum validation"
  fi
  if have_base "$TAG"; then
    echo "===== BASE_EVAL $TAG START $(date -u +%H:%M:%S) ====="
    python -m scripts.base_eval --model-tag=$TAG --device-batch-size=32 --mdm-score=l2r --mdm-mc=4 2>&1 | tee "$NANOCHAT_BASE_DIR/logs/${TAG}_base_eval.log" \
        | grep --line-buffered -E "CORE|bpb|Traceback|Error"
    echo "===== PRETRAIN $TAG DONE $(date -u +%H:%M:%S) ====="
  else
    echo "===== PRETRAIN $TAG FAILED (no checkpoint) $(date -u +%H:%M:%S) ====="
  fi
fi
echo "===== PHASE2B DONE $(date -u +%H:%M:%S) ====="
