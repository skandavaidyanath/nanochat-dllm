#!/bin/bash
# Phase 2a: SFT recipe sweep at d12 (post-training open question 1) on the existing d12 base checkpoints.
# Recipe axes from the phase-1 diagnostics: MMLU share (--mmlu-epochs 3 -> 9), initial LR fraction (default 0.8 ->
# 0.3 / 0.1) and steps (one epoch = 934 optimizer steps = 14,944 micro-batches at device batch 16; x2 = 29,888,
# x3 = 44,832; --num-iterations counts micro-batches). Every run: chat_sft with val bpb and ChatCORE every 400
# steps (categorical tasks on 400 problems each, 8 generative samples each), then the full categorical chat_eval
# and GSM8K on 100 problems. Runs whose SFT checkpoint exists are skipped, so the script can be re-run after an
# interruption. One GPU per lane; LANE picks the subset (the lanes are balanced by expected wall time).
# Note: the base checkpoints carry no optimizer state here, so every run (including the AR control) starts from a
# fresh optimizer; the phase-1 SFT runs warm-started momentum from pretraining.
# Usage: CUDA_VISIBLE_DEVICES=0 LANE=0 bash runs/exp_d12_sft_sweep.sh     (LANE=all runs everything sequentially)
set -u
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}" OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
LANE="${LANE:-all}"
DBS="${DBS:-16}"
source .venv/bin/activate
mkdir -p "$NANOCHAT_BASE_DIR/logs"
X2=29888; X3=44832
# name | base tag | extra chat_sft args
RUNS=(
  "sft2_mdm_m9_lr0.3_x3|d12_mdm|--mmlu-epochs=9 --init-lr-frac=0.3 --num-iterations=$X3"
  "sft2_mdm_m3_lr0.3_x2|d12_mdm|--mmlu-epochs=3 --init-lr-frac=0.3 --num-iterations=$X2"
  "sft2_mdm_m9_lr0.3_x2|d12_mdm|--mmlu-epochs=9 --init-lr-frac=0.3 --num-iterations=$X2"
  "sft2_ar_m9_lr0.3_x2|d12_ar|--mmlu-epochs=9 --init-lr-frac=0.3 --num-iterations=$X2"
  "sft2_mdm_m9_lr0.1_x2|d12_mdm|--mmlu-epochs=9 --init-lr-frac=0.1 --num-iterations=$X2"
)
case "$LANE" in
  0) IDX=(0 1);;
  1) IDX=(2 3 4);;
  all) IDX=(0 1 2 3 4);;
  *) echo "LANE must be 0, 1 or all"; exit 1;;
esac
have_sft() { ls "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$1"/model_*.pt >/dev/null 2>&1; }
for i in "${IDX[@]}"; do
  IFS='|' read -r NAME BASE EXTRA <<< "${RUNS[$i]}"
  if have_sft "$NAME"; then echo "=== skip $NAME (checkpoint exists)"; continue; fi
  echo "===== SFT $NAME ($BASE $EXTRA) START $(date -u +%H:%M:%S) ====="
  python -m scripts.chat_sft --model-tag=$BASE --output-tag=$NAME --run=$NAME --device-batch-size=$DBS \
      --eval-every=400 --chatcore-every=400 --chatcore-max-cat=400 --chatcore-max-sample=8 $EXTRA \
      2>&1 | tee "$NANOCHAT_BASE_DIR/logs/${NAME}_sft.log" \
      | grep --line-buffered -E "Training mixture|Validation bpb|ChatCORE|ARC|MMLU|GSM8K|HumanEval|Traceback|Error|Total training time"
  if ! have_sft "$NAME"; then echo "===== SFT $NAME FAILED (no checkpoint) $(date -u +%H:%M:%S) ====="; continue; fi
  echo "===== CHAT_EVAL $NAME START $(date -u +%H:%M:%S) ====="
  python -m scripts.chat_eval -i sft -g $NAME -b $DBS -a "ARC-Easy|ARC-Challenge|MMLU" \
      2>&1 | tee "$NANOCHAT_BASE_DIR/logs/${NAME}_chat_eval.log" | grep --line-buffered -E "ARC|MMLU|ChatCORE|Traceback|Error"
  python -m scripts.chat_eval -i sft -g $NAME -b $DBS -a "GSM8K" -x 100 \
      2>&1 | tee -a "$NANOCHAT_BASE_DIR/logs/${NAME}_chat_eval.log" | grep --line-buffered -E "GSM8K|Traceback|Error"
  echo "===== SFT $NAME DONE $(date -u +%H:%M:%S) ====="
done
echo "===== LANE $LANE DONE $(date -u +%H:%M:%S) ====="
